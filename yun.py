#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import asyncio
import logging
import base64

import uvloop
import aiohttp
import pandas as pd
import websockets

from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# ———— uvloop for performance ————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ———— Configuration ————
load_dotenv('/root/zhibai/.env')
API_KEY          = os.getenv('ED25519_API_KEY')
PRIVATE_KEY_PATH = os.getenv('ED25519_KEY_PATH')
SYMBOL           = 'ETHUSDC'
PAIR_LOWER       = SYMBOL.lower()
WS_DATA_URL      = (
    'wss://fstream.binance.com/stream?streams='
    f'{PAIR_LOWER}@kline_3m/'
    f'{PAIR_LOWER}@kline_15m/'
    f'{PAIR_LOWER}@kline_1h/'
    f'{PAIR_LOWER}@markPrice'
)
REST_BASE   = 'https://fapi.binance.com'
RECV_WINDOW = 5000

# ———— Global state ————
klines       = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
latest_price = None
price_ts     = None
lock         = asyncio.Lock()
last_side    = None
session      = None  # initialized in main()

# ———— Load Ed25519 key & sign function ————
with open(PRIVATE_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

def sign_ed25519(params: dict) -> str:
    """
    拼接 payload，Ed25519 签名，返回 raw Base64 字符串（不再 urlencode）。
    """
    payload = '&'.join(f"{k}={v}" for k,v in sorted(params.items()))
    raw = ed_priv.sign(payload.encode('ascii'))
    return base64.b64encode(raw).decode('ascii')

# ———— Update indicators ————
def update_indicators():
    for tf, df in klines.items():
        if df.empty: continue
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn'])/(df['bb_up'] - df['bb_dn'])
        if tf=='15m':
            hl2 = (df['high']+df['low'])/2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st']   = hl2 - 3*atr
            df['macd']= MACD(df['close'],12,26,9).macd_diff()
        if tf=='3m':
            num=(df['close']-df['open']).ewm(span=10).mean()
            den=(df['high']-df['low']).ewm(span=10).mean()
            df['rvgi']  = num/den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ———— Market data via WebSocket ————
async def market_ws():
    global latest_price, price_ts
    async with websockets.connect(WS_DATA_URL) as ws:
        async for msg in ws:
            o = json.loads(msg)
            s, d = o.get('stream',''), o.get('data',{})
            if s.endswith('@markPrice'):
                latest_price = float(d['p'])
                price_ts     = int(time.time()*1000)
            if 'kline' in s:
                tf = s.split('@')[1].split('_')[1]; k=d['k']
                rec = {'open':float(k['o']),'high':float(k['h']),
                       'low':float(k['l']),'close':float(k['c'])}
                df = klines[tf]
                if df.empty:
                    klines[tf] = pd.DataFrame([rec])
                else:
                    if int(k['t'])>df.index[-1]:
                        df.loc[len(df)] = rec
                    else:
                        df.iloc[-1] = [rec['open'],rec['high'],rec['low'],rec['close']]
                    klines[tf] = df
                update_indicators()

# ———— REST order helper (body form-encoded) ————
async def rest_order(side: str, order_type: str, qty: float, price: float=None, stopPrice: float=None):
    ts = int(time.time()*1000)
    params = {
        'symbol': SYMBOL,
        'side':   side,
        'type':   order_type,
        'timestamp': ts,
        'recvWindow': RECV_WINDOW
    }
    if order_type == 'LIMIT':
        params.update({
            'timeInForce':'GTC',
            'quantity': f"{qty:.6f}",
            'price':    f"{price:.2f}"
        })
    else:
        params.update({
            'closePosition':'true',
            'stopPrice': f"{stopPrice:.2f}"
        })
    params['signature'] = sign_ed25519(params)
    async with session.post(f"{REST_BASE}/fapi/v1/order", data=params) as resp:
        result = await resp.json()
        logging.info("REST %s %s => %s", order_type, side, result)
        return result

# ———— Bracket orders (entry + TP + SL) ————
async def bracket(qty: float, entry: float, side: str):
    await rest_order(side, 'LIMIT', qty, price=entry)
    tp = entry * (1.02 if side=='BUY' else 0.98)
    await rest_order('SELL' if side=='BUY' else 'BUY', 'TAKE_PROFIT_MARKET', qty, stopPrice=tp)
    sl = entry * (0.98 if side=='BUY' else 1.02)
    await rest_order('SELL' if side=='BUY' else 'BUY', 'STOP_MARKET', qty, stopPrice=sl)

# ———— Main strategy & child strategies (as before) ————
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
                await asyncio.sleep(0.5); continue
            st   = klines['15m']['st'].iloc[-1]
            bb1h = klines['1h']['bb_pct'].iloc[-1]
            bb3m = klines['3m']['bb_pct'].iloc[-1]
            up, down = p>st, p<st
            s_long  = up  and bb1h<0.2
            s_short = down and bb1h>0.8

            if s_long and bb3m<=0 and last_side!='LONG':
                await bracket(0.12, p, 'BUY');  last_side='LONG'
            elif s_short and bb3m>=1 and last_side!='SHORT':
                await bracket(0.12, p, 'SELL'); last_side='SHORT'
            elif up and not s_long and 0<bb3m<=0.5 and last_side!='LONG':
                await bracket(0.03, p, 'BUY');  last_side='LONG'
            elif down and not s_short and 0.5<=bb3m<1 and last_side!='SHORT':
                await bracket(0.03, p, 'SELL'); last_side='SHORT'
            elif s_long and bb3m>=1 and last_side!='SHORT':
                await bracket(0.07, p, 'SELL'); last_side='SHORT'
            elif s_short and bb3m<=0 and last_side!='LONG':
                await bracket(0.07, p, 'BUY');  last_side='LONG'
        await asyncio.sleep(0.5)

async def macd_strategy():
    trig = False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.empty or 'macd' not in df: continue
            macd, prev = df['macd'].iloc[-1], df['macd'].iloc[-2]
            if not trig and prev>0 and macd<prev and macd>=11:
                await bracket(0.15, latest_price, 'SELL'); trig=True
            if trig and prev<0 and macd>prev and macd<=-11:
                await bracket(0.15, latest_price, 'BUY'); trig=False

async def rvgi_strategy():
    cnt_l = cnt_s = 0
    while True:
        await asyncio.sleep(20)
        async with lock:
            df = klines['15m']
            if df.empty or 'rvgi' not in df or 'rvsig' not in df: continue
            rv, sig = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv>sig and cnt_l*0.05<0.2:
                await bracket(0.05, latest_price, 'BUY'); cnt_l+=1
            if rv<sig and cnt_s*0.05<0.2:
                await bracket(0.05, latest_price, 'SELL'); cnt_s+=1

async def triple_st_strategy():
    firing = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.empty or 'st' not in df or len(df['st'])<3: continue
            p = latest_price; stv = df['st']
            rising  = stv.iloc[-3]<stv.iloc[-2]<stv.iloc[-1]<p
            falling = stv.iloc[-3]>stv.iloc[-2]>stv.iloc[-1]>p
            if rising and not firing:
                await bracket(0.15,p,'BUY');  firing=True
            if falling and not firing:
                await bracket(0.15,p,'SELL'); firing=True
            prev_st = stv.iloc[-2]
            if firing and ((rising and p<prev_st) or (falling and p>prev_st)):
                side = 'SELL' if rising else 'BUY'
                await bracket(0.15,p,side); firing=False

# ———— Entry point ————
async def main():
    global session
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    session = aiohttp.ClientSession(headers={'X-MBX-APIKEY': API_KEY})
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