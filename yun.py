#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import asyncio
import logging
import base64
import sqlite3
import math

import uvloop                                       # High‑performance event loop
import pandas as pd
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import aiohttp
import websockets
from aiohttp_retry import RetryClient, ExponentialRetry
from ta.volatility import BollingerBands
from ta.trend import MACD

# ————————— uvloop as default asyncio loop —————————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ————————— Configuration —————————
ENV_PATH    = '/root/zhibai/.env'
DB_PATH     = '/root/zhibai/trade_state.db'
FUT_WS_BASE = 'wss://fstream.binance.com/stream?streams'
FUT_API     = 'https://fapi.binance.com'
SYMBOL      = 'ETHUSDC'
PAIR_LOWER  = SYMBOL.lower()

# Load environment
load_dotenv(ENV_PATH)
API_KEY     = os.getenv('BINANCE_API_KEY')
PRIV_KEY    = os.getenv('ED25519_KEY_PATH')  # e.g. /root/zhibai/ed25519-prv.pem
with open(PRIV_KEY, 'rb') as f:
    priv_key = load_pem_private_key(f.read(), password=None)

HEADERS     = {'X-MBX-APIKEY': API_KEY}
rest_opts   = ExponentialRetry(attempts=3)
rest_client = RetryClient(retry_options=rest_opts, raise_for_status=False)

# ————————— Global queues & state —————————
price_q     = asyncio.Queue(maxsize=1)
klines      = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
last_side   = None
lock        = asyncio.Lock()

# ExchangeInfo filters & limits
exchange_filters = {}
max_orders       = max_algo_orders = None
open_orders      = open_algo_orders = 0

# ————————— Persistence to SQLite —————————
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('CREATE TABLE IF NOT EXISTS events(ts INTEGER, type TEXT, data TEXT)')
    conn.commit()
    conn.close()

def _persist(ts, typ, data):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO events VALUES (?,?,?)", (ts, typ, json.dumps(data)))
    conn.commit()
    conn.close()

async def persist_worker():
    # simple loop to persist market events (reusing price_q)
    while True:
        ts, typ, data = await price_q.get()
        await asyncio.get_event_loop().run_in_executor(None, _persist, ts, typ, data)
        price_q.task_done()

# ————————— Signing helper —————————
def sign(params: dict) -> str:
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = priv_key.sign(payload.encode('ascii'))
    return base64.b64encode(sig).decode()

# ————————— Rounding helpers —————————
def round_down(val, step):
    return math.floor(val/step) * step

def round_up(val, step):
    return math.ceil(val/step) * step

# ————————— Load ExchangeInfo —————————
async def load_contract_info():
    global exchange_filters, max_orders, max_algo_orders
    url = f"{FUT_API}/fapi/v1/exchangeInfo"
    async with rest_client.get(url) as r:
        info = await r.json()
    for s in info['symbols']:
        if s['symbol'] == SYMBOL:
            assert s['marginAsset'] == 'USDC', "Contract not USDC‑margined"
            for f in s['filters']:
                exchange_filters[f['filterType']] = f
            max_orders       = exchange_filters['MAX_NUM_ORDERS'     ]['limit']
            max_algo_orders  = exchange_filters['MAX_NUM_ALGO_ORDERS']['limit']
            break

def apply_filters(price: float, qty: float):
    pf = exchange_filters['PRICE_FILTER']
    lf = exchange_filters['LOT_SIZE']
    tsz = float(pf['tickSize'])
    qsz = float(lf['stepSize'])
    price = round_down(price, tsz)
    qty   = round_down(qty,   qsz)
    min_not = float(exchange_filters['MIN_NOTIONAL']['notional'])
    if price * qty < min_not:
        raise ValueError(f"Notional {price*qty:.2f} < min {min_not:.2f}")
    return price, qty

# ————————— Indicators update —————————
def update_indicators():
    for tf, df in klines.items():
        if df.empty: continue
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_up']   = bb.bollinger_hband()
        df['bb_dn']   = bb.bollinger_lband()
        df['bb_pct']  = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        df['bb_w']    = (df['bb_up'] - df['bb_dn']) / df['bb_dn']
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi']  = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ————————— Market WebSocket listener —————————
async def market_ws():
    streams = ','.join(f"{PAIR_LOWER}@kline_{tf}" for tf in klines)
    url     = f"{FUT_WS_BASE}={PAIR_LOWER}@markPrice/{streams}"
    async with websockets.connect(url) as ws:
        async for msg in ws:
            o = json.loads(msg)
            # markPrice update
            if o.get('stream','').endswith('@markPrice'):
                price = float(o['data']['p'])
                ts    = int(time.time()*1000)
                if price_q.full(): _ = price_q.get_nowait()
                await price_q.put((price, ts))
                await _persist(ts, 'price', {'price': price})
            # kline update
            if 'kline' in o.get('stream',''):
                tf = o['stream'].split('_')[1]
                k  = o['data']['k']
                rec = {
                    'open':  float(k['o']),
                    'high':  float(k['h']),
                    'low':   float(k['l']),
                    'close': float(k['c'])
                }
                df = klines[tf]
                if df.empty or int(k['t']) > df.index[-1]:
                    df = df.append(rec, ignore_index=True)
                else:
                    df.iloc[-1] = list(rec.values())
                klines[tf] = df
                update_indicators()

# ————————— Order submission —————————
async def send_order(params: dict):
    url = f"{FUT_API}/fapi/v1/order"
    params.update({'timestamp': int(time.time()*1000), 'recvWindow': 5000})
    params['signature'] = sign(params)
    async with rest_client.post(url, headers=HEADERS, data=params) as r:
        res = await r.json()
        logging.info("Order response: %s", res)
        return res

# ————————— Bracket trade (entry + TP + SL) —————————
async def bracket_trade(side: str, qty: float, entry_price: float):
    global open_orders, open_algo_orders

    # Check rate‑limit compliance
    if open_orders >= max_orders or open_algo_orders + 2 > max_algo_orders:
        logging.warning("Order limit reached; skipping trade")
        return

    # Adjust to filter
    entry_price, qty = apply_filters(entry_price, qty)

    # 1) Market entry
    res1 = await send_order({
        'symbol': SYMBOL, 'side': side, 'type': 'MARKET', 'quantity': f"{qty:.8f}"
    })
    open_orders += 1

    # 2) Take profit market
    tp_price = entry_price * (1.02 if side == 'BUY' else 0.98)
    res2 = await send_order({
        'symbol': SYMBOL,
        'side': 'SELL' if side == 'BUY' else 'BUY',
        'type': 'TAKE_PROFIT_MARKET',
        'stopPrice': f"{tp_price:.8f}",
        'closePosition': 'true'
    })
    open_algo_orders += 1

    # 3) Stop market
    sl_price = entry_price * (0.98 if side == 'BUY' else 1.02)
    res3 = await send_order({
        'symbol': SYMBOL,
        'side': 'SELL' if side == 'BUY' else 'BUY',
        'type': 'STOP_MARKET',
        'stopPrice': f"{sl_price:.8f}",
        'closePosition': 'true'
    })
    open_algo_orders += 1

# ————————— Main strategy —————————
async def main_strategy():
    global last_side
    # Preload exchange info
    await load_contract_info()

    while True:
        price, ts = await price_q.get()
        async with lock:
            st   = klines['15m']['st'].iloc[-1]
            bb1h = klines['1h']['bb_pct'].iloc[-1]
            bb3m = klines['3m']['bb_pct'].iloc[-1]
            up, down   = price > st, price < st
            long_s     = bb1h < 0.2
            short_s    = bb1h > 0.8

            # Strong trend‑aligned signal
            if up and long_s and bb3m <= 0 and last_side != 'LONG':
                await bracket_trade('BUY', 0.12, price)
                last_side = 'LONG'
            elif down and short_s and bb3m >= 1 and last_side != 'SHORT':
                await bracket_trade('SELL', 0.12, price)
                last_side = 'SHORT'

            # You can add weak/contrarian signals here...

        price_q.task_done()

# ————————— Entry point —————————
async def main():
    init_db()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    await asyncio.gather(
        persist_worker(),
        market_ws(),
        main_strategy(),
    )

if __name__ == '__main__':
    asyncio.run(main())