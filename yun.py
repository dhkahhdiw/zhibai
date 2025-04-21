#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import hmac
import hashlib
import asyncio
import logging
import urllib.parse

import uvloop
import aiohttp
import pandas as pd
import websockets

from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# ———— Use uvloop for high-performance asyncio ————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ———— Load environment variables ————
load_dotenv('/root/zhibai/.env')
API_KEY          = os.getenv('BINANCE_API_KEY')
SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode('ascii')
ED25519_API_KEY  = os.getenv('ED25519_API_KEY')
ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
SYMBOL           = 'ETHUSDC'
PAIR_LOWER       = SYMBOL.lower()

# WebSocket market data URL
WS_DATA_URL = (
    'wss://fstream.binance.com/stream?streams='
    f'{PAIR_LOWER}@kline_3m/'
    f'{PAIR_LOWER}@kline_15m/'
    f'{PAIR_LOWER}@kline_1h/'
    f'{PAIR_LOWER}@markPrice'
)

# REST API base
REST_BASE   = 'https://fapi.binance.com'
RECV_WINDOW = 5000

# ———— Global state ————
klines       = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
latest_price = None
price_ts     = None
lock         = asyncio.Lock()
last_side    = None
session      = None
TIME_OFFSET  = 0   # ms

# ———— Load Ed25519 private key (for future use) ————
with open(ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# ———— Sync local time with server ————
async def sync_time():
    global TIME_OFFSET
    async with session.get(f"{REST_BASE}/fapi/v1/time") as resp:
        data = await resp.json()
        srv = int(data['serverTime'])
    local = int(time.time() * 1000)
    TIME_OFFSET = srv - local
    logging.info("Time offset: %d ms", TIME_OFFSET)

# ———— Build & sign REST query ————
def build_signed_query(params: dict) -> str:
    # sort params, urlencode
    qs = urllib.parse.urlencode(sorted(params.items()), safe='')
    sig = hmac.new(SECRET_KEY, qs.encode('ascii'), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

# ———— REST order helper with manual query string ————
async def rest_order(side: str, otype: str, qty: float, price: float = None, stopPrice: float = None):
    ts = int(time.time() * 1000 + TIME_OFFSET)
    params = {
        'symbol': SYMBOL,
        'side': side,
        'type': otype,
        'timestamp': ts,
        'recvWindow': RECV_WINDOW
    }
    if otype == 'LIMIT':
        params.update({
            'timeInForce': 'GTC',
            'quantity': f"{qty:.6f}",
            'price':    f"{price:.2f}"
        })
    else:
        params.update({
            'closePosition': 'true',
            'stopPrice':      f"{stopPrice:.2f}"
        })
    query = build_signed_query(params)
    url = f"{REST_BASE}/fapi/v1/order?{query}"
    async with session.post(url, headers={'X-MBX-APIKEY': API_KEY}) as r:
        res = await r.json()
        logging.info("REST %s %s => %s", otype, side, res)
        return res

# ———— Bracket orders (entry + TP + SL) ————
async def bracket(qty: float, entry: float, side: str):
    # entry
    await rest_order(side, 'LIMIT', qty, price=entry)
    # take profit
    tp = entry * (1.02 if side == 'BUY' else 0.98)
    await rest_order('SELL' if side == 'BUY' else 'BUY', 'TAKE_PROFIT_MARKET', qty, stopPrice=tp)
    # stop loss
    sl = entry * (0.98 if side == 'BUY' else 1.02)
    await rest_order('SELL' if side == 'BUY' else 'BUY', 'STOP_MARKET', qty, stopPrice=sl)

# ———— Indicator updater ————
def update_indicators():
    for tf, df in klines.items():
        if df.empty:
            continue
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        if tf == '15m':
            hl2      = (df['high'] + df['low']) / 2
            atr      = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num        = (df['close'] - df['open']).ewm(span=10).mean()
            den        = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi']  = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ———— Market data WebSocket ————
async def market_ws():
    global latest_price, price_ts
    async with websockets.connect(WS_DATA_URL) as ws:
        async for msg in ws:
            o = json.loads(msg)
            s = o['stream']; d = o['data']
            if s.endswith('@markPrice'):
                latest_price = float(d['p'])
                price_ts = int(time.time() * 1000)
            if 'kline' in s:
                tf = s.split('@')[1].split('_')[1]
                k  = d['k']
                rec = {
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low':  float(k['l']),
                    'close':float(k['c'])
                }
                df = klines[tf]
                if df.empty:
                    klines[tf] = pd.DataFrame([rec])
                else:
                    if int(k['t']) > df.index[-1]:
                        df.loc[len(df)] = rec
                    else:
                        df.iloc[-1] = [rec['open'], rec['high'], rec['low'], rec['close']]
                    klines[tf] = df
                update_indicators()

# ———— Main strategy ————
async def main_strategy():
    global last_side
    while price_ts is None:
        await asyncio.sleep(0.5)
    while True:
        async with lock:
            p = latest_price
            if ('st' not in klines['15m'] or
                'bb_pct' not in klines['1h'] or
                'bb_pct' not in klines['3m']):
                await asyncio.sleep(0.5)
                continue
            st   = klines['15m']['st'].iloc[-1]
            bb1h = klines['1h']['bb_pct'].iloc[-1]
            bb3m = klines['3m']['bb_pct'].iloc[-1]
            up, down = p > st, p < st
            strong_long  = up and bb1h < 0.2
            strong_short = down and bb1h > 0.8

            # trend-aligned strong
            if strong_long  and bb3m <= 0 and last_side != 'LONG':
                await bracket(0.12, p, 'BUY');  last_side = 'LONG'
            elif strong_short and bb3m >= 1 and last_side != 'SHORT':
                await bracket(0.12, p, 'SELL'); last_side = 'SHORT'
            # trend-aligned weak
            elif up and not strong_long and 0 < bb3m <= 0.5 and last_side != 'LONG':
                await bracket(0.03, p, 'BUY');  last_side = 'LONG'
            elif down and not strong_short and 0.5 <= bb3m < 1 and last_side != 'SHORT':
                await bracket(0.03, p, 'SELL'); last_side = 'SHORT'
            # counter-trend strong
            elif strong_long and bb3m >= 1 and last_side != 'SHORT':
                await bracket(0.07, p, 'SELL'); last_side = 'SHORT'
            elif strong_short and bb3m <= 0 and last_side != 'LONG':
                await bracket(0.07, p, 'BUY');  last_side = 'LONG'
        await asyncio.sleep(0.5)

# ———— 15m MACD sub-strategy ————
async def macd_strategy():
    fired = False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.empty or 'macd' not in df: continue
            cur, prev = df['macd'].iloc[-1], df['macd'].iloc[-2]
            if not fired and prev > 0 and cur < prev and cur >= 11:
                await bracket(0.15, latest_price, 'SELL'); fired = True
            if fired and prev < 0 and cur > prev and cur <= -11:
                await bracket(0.15, latest_price, 'BUY');  fired = False

# ———— 15m RVGI sub-strategy ————
async def rvgi_strategy():
    cnt_long = cnt_short = 0
    while True:
        await asyncio.sleep(20)
        async with lock:
            df = klines['15m']
            if df.empty or 'rvgi' not in df or 'rvsig' not in df: continue
            rv, sig = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv > sig and cnt_long*0.05 < 0.2:
                await bracket(0.05, latest_price, 'BUY');  cnt_long += 1
            if rv < sig and cnt_short*0.05 < 0.2:
                await bracket(0.05, latest_price, 'SELL'); cnt_short += 1

# ———— Triple SuperTrend sub-strategy ————
async def triple_st_strategy():
    active = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.empty or 'st' not in df or len(df['st']) < 3: continue
            p = latest_price; stv = df['st']
            rising  = stv.iloc[-3] < stv.iloc[-2] < stv.iloc[-1] < p
            falling = stv.iloc[-3] > stv.iloc[-2] > stv.iloc[-1] > p
            if rising  and not active:
                await bracket(0.15, p, 'BUY');  active = True
            if falling and not active:
                await bracket(0.15, p, 'SELL'); active = True
            prev_st = stv.iloc[-2]
            if active and ((rising and p < prev_st) or (falling and p > prev_st)):
                side = 'SELL' if rising else 'BUY'
                await bracket(0.15, p, side); active = False

# ———— Entry point ————
async def main():
    global session
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    session = aiohttp.ClientSession()
    await sync_time()
    try:
        await asyncio.gather(
            market_ws(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy(),
        )
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())