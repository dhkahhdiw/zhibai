#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import uuid
import asyncio
import logging
import base64
import math
import pandas as pd
import uvloop
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import websockets
from ta.volatility import BollingerBands
from ta.trend import MACD

# ————————— uvloop as default —————————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ————————— Configuration —————————
ENV_PATH      = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY       = os.getenv('BINANCE_API_KEY')
PRIV_KEY_PATH = os.getenv('ED25519_KEY_PATH')
SYMBOL        = 'ETHUSDC'
PAIR_LOWER    = SYMBOL.lower()
WS_DATA_URL   = (
    f"wss://fstream.binance.com/stream?streams="
    f"{PAIR_LOWER}@kline_3m/{PAIR_LOWER}@kline_15m/{PAIR_LOWER}@kline_1h/{PAIR_LOWER}@markPrice"
)
WS_ORDER_URL  = 'wss://ws-fapi.binance.com/ws-fapi/v1'
RECV_WINDOW   = 5000

# ————————— Global state —————————
klines      = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
price       = None
price_ts    = None
order_ws    = None
last_side   = None
lock        = asyncio.Lock()

# ————————— Load private key & sign —————————
with open(PRIV_KEY_PATH, 'rb') as f:
    priv_key = load_pem_private_key(f.read(), password=None)

def sign_payload(params: dict) -> str:
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = priv_key.sign(payload.encode('ascii'))
    return base64.b64encode(sig).decode('ascii')

# ————————— Indicators —————————
def update_indicators():
    for tf, df in klines.items():
        if df.empty: continue
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_up'], df['bb_dn'] = bb.bollinger_hband(), bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        df['bb_w']   = (df['bb_up'] - df['bb_dn']) / df['bb_dn']
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st']   = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi']  = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ————————— Market Data WS —————————
async def market_data_ws():
    global price, price_ts
    async with websockets.connect(WS_DATA_URL) as ws:
        async for msg in ws:
            o = json.loads(msg)
            stream = o.get('stream', '')
            data   = o.get('data', {})
            # markPrice
            if stream.endswith('@markPrice'):
                price     = float(data['p'])
                price_ts  = int(time.time() * 1000)
            # kline
            if 'kline' in stream:
                tf = stream.split('@')[1].split('_')[1]
                k  = data['k']
                rec = {
                    'open':  float(k['o']),
                    'high':  float(k['h']),
                    'low':   float(k['l']),
                    'close': float(k['c'])
                }
                df = klines[tf]
                # Replace deprecated append with loc
                if df.empty:
                    klines[tf] = pd.DataFrame([rec])
                else:
                    # new candle?
                    if int(k['t']) > df.index[-1]:
                        df.loc[len(df)] = rec
                    else:
                        df.iloc[-1] = [rec['open'], rec['high'], rec['low'], rec['close']]
                    klines[tf] = df
                update_indicators()

# ————————— Order WS —————————
async def init_order_ws():
    global order_ws
    order_ws = await websockets.connect(WS_ORDER_URL)
    # ping loop to keep alive
    async def ping():
        while True:
            await order_ws.ping()
            await asyncio.sleep(60)
    asyncio.create_task(ping())

async def place_order_rpc(params: dict):
    req = {
        'id': str(uuid.uuid4()),
        'method': 'order.place',
        'params': params
    }
    await order_ws.send(json.dumps(req))
    resp = await order_ws.recv()
    return json.loads(resp)

# ————————— Bracket helper —————————
async def bracket(side, qty, price_val):
    ts = int(time.time() * 1000)
    base = {
        'apiKey': API_KEY,
        'symbol': SYMBOL,
        'recvWindow': RECV_WINDOW,
        'timestamp': ts
    }
    # entry LIMIT
    p1 = {**base,
        'side': side, 'type': 'LIMIT', 'timeInForce': 'GTC',
        'quantity': f"{qty:.6f}", 'price': f"{price_val:.2f}"
    }
    p1['signature'] = sign_payload(p1)
    r1 = await place_order_rpc(p1)
    # TP
    tp_val = price_val * (1.02 if side=='BUY' else 0.98)
    p2 = {**base,
        'side': 'SELL' if side=='BUY' else 'BUY',
        'type': 'TAKE_PROFIT_MARKET',
        'closePosition': 'true',
        'stopPrice': f"{tp_val:.2f}"
    }
    p2['signature'] = sign_payload(p2)
    r2 = await place_order_rpc(p2)
    # SL
    sl_val = price_val * (0.98 if side=='BUY' else 1.02)
    p3 = {**base,
        'side': 'SELL' if side=='BUY' else 'BUY',
        'type': 'STOP_MARKET',
        'closePosition': 'true',
        'stopPrice': f"{sl_val:.2f}"
    }
    p3['signature'] = sign_payload(p3)
    r3 = await place_order_rpc(p3)
    logging.info("Bracket orders: %s / %s / %s", r1, r2, r3)

# ————————— Main strategy —————————
async def main_strategy():
    global last_side
    while True:
        await asyncio.sleep(0)  # yield to event loop
        if price_ts is None:
            await asyncio.sleep(1)
            continue
        async with lock:
            p    = price
            st   = klines['15m']['st'].iloc[-1]
            bb1h = klines['1h']['bb_pct'].iloc[-1]
            bb3m = klines['3m']['bb_pct'].iloc[-1]
            up   = p > st
            down = p < st
            long_s  = bb1h < 0.2
            short_s = bb1h > 0.8

            # Strong trend‑aligned
            if up and long_s and bb3m <= 0 and last_side!='LONG':
                await bracket('BUY', 0.12, p); last_side='LONG'
            elif down and short_s and bb3m >= 1 and last_side!='SHORT':
                await bracket('SELL', 0.12, p); last_side='SHORT'
            # Weak trend‑aligned
            elif up and not long_s and 0 < bb3m <= 0.5 and last_side!='LONG':
                await bracket('BUY', 0.03, p); last_side='LONG'
            elif down and not short_s and 0.5 <= bb3m < 1 and last_side!='SHORT':
                await bracket('SELL', 0.03, p); last_side='SHORT'
            # Contrarian (strong)
            elif up and short_s and bb3m >= 1 and last_side!='LONG':
                await bracket('BUY', 0.07, p); last_side='LONG'
            elif down and long_s and bb3m <= 0 and last_side!='SHORT':
                await bracket('SELL', 0.07, p); last_side='SHORT'
        await asyncio.sleep(0.5)

# ————————— Child: 15m MACD —————————
async def macd_strategy():
    triggered = False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.empty: continue
            macd, prev = df['macd'].iloc[-1], df['macd'].iloc[-2]
            if not triggered and prev>0 and macd<prev and macd>=11:
                await bracket('SELL', 0.15, price)
                triggered = True
            if triggered and prev<0 and macd>prev and macd<=-11:
                await bracket('BUY', 0.15, price)
                triggered = False

# ————————— Child: 15m RVGI —————————
async def rvgi_strategy():
    long_cnt = short_cnt = 0
    while True:
        await asyncio.sleep(20)
        async with lock:
            df = klines['15m']
            if df.empty: continue
            rv, sig = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv>sig and long_cnt*0.05<0.2:
                await bracket('BUY', 0.05, price); long_cnt += 1
            if rv<sig and short_cnt*0.05<0.2:
                await bracket('SELL', 0.05, price); short_cnt += 1

# ————————— Child: Triple SuperTrend —————————
async def triple_st_strategy():
    firing = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.empty: continue
            p    = price
            st   = df['st']
            # rising if last three points ascending
            rising  = st.iloc[-1] > st.iloc[-2] > st.iloc[-3] and p > st.iloc[-1]
            falling = st.iloc[-1] < st.iloc[-2] < st.iloc[-3] and p < st.iloc[-1]
            if rising and not firing:
                await bracket('BUY', 0.15, p); firing = True
            if falling and not firing:
                await bracket('SELL',0.15, p); firing = True
            # exit on reversal
            if firing and ((rising and p < st.iloc[-2]) or (falling and p > st.iloc[-2])):
                side = 'SELL' if rising else 'BUY'
                await bracket(side, 0.15, p)
                firing = False

# ————————— Entry Point —————————
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    await init_order_ws()
    await asyncio.gather(
        market_data_ws(),
        main_strategy(),
        macd_strategy(),
        rvgi_strategy(),
        triple_st_strategy(),
    )

if __name__ == '__main__':
    asyncio.run(main())