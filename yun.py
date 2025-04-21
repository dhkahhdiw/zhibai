#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import asyncio
import logging
import base64
import urllib.parse
import math

import uvloop                                      # 高性能事件循环
import aiohttp
import pandas as pd
import websockets

from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# ————————— uvloop 替代默认 asyncio —————————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ————————— 配置 —————————
ENV_PATH        = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY         = os.getenv('BINANCE_API_KEY')
PRIVATE_KEY_PATH= os.getenv('ED25519_KEY_PATH')  # Ed25519 私钥路径
SYMBOL          = 'ETHUSDC'
PAIR_LOWER      = SYMBOL.lower()
WS_DATA_URL     = (
    'wss://fstream.binance.com/stream?streams='
    f'{PAIR_LOWER}@kline_3m/'
    f'{PAIR_LOWER}@kline_15m/'
    f'{PAIR_LOWER}@kline_1h/'
    f'{PAIR_LOWER}@markPrice'
)
REST_BASE       = 'https://fapi.binance.com'
RECV_WINDOW     = 5000

# ————————— 全局状态 —————————
klines       = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
latest_price = None
price_ts     = None
lock         = asyncio.Lock()
last_side    = None

# aiohttp session for REST
session = aiohttp.ClientSession(
    headers={'X-MBX-APIKEY': API_KEY}
)

# ————————— 加载 Ed25519 私钥 & 签名函数 —————————
with open(PRIVATE_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

def sign_ed25519(params: dict) -> str:
    """
    按字母序拼接 payload，用 Ed25519 私钥签名，Base64 后 URL encode。
    """
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = ed_priv.sign(payload.encode('ascii'))
    b64 = base64.b64encode(sig).decode('ascii')
    return urllib.parse.quote(b64, safe='')

# ————————— 指标更新 —————————
def update_indicators():
    for tf, df in klines.items():
        if df.empty: continue
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st']   = hl2 - 3 * atr
            df['macd']= MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi']  = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ————————— WebSocket 市场数据 —————————
async def market_ws():
    global latest_price, price_ts
    async with websockets.connect(WS_DATA_URL) as ws:
        async for msg in ws:
            o = json.loads(msg)
            s = o.get('stream','')
            d = o.get('data',{})
            # markPrice 更新
            if s.endswith('@markPrice'):
                latest_price = float(d['p'])
                price_ts     = int(time.time()*1000)
            # K 线更新
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
                        df.iloc[-1] = [rec['open'],rec['high'],rec['low'],rec['close']]
                    klines[tf] = df
                update_indicators()

# ————————— REST 限价下单 —————————
async def place_limit(side: str, qty: float, price: float) -> dict:
    ts = int(time.time()*1000)
    params = {
        'symbol': SYMBOL,
        'side': side,
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': f"{qty:.6f}",
        'price':    f"{price:.2f}",
        'timestamp': ts,
        'recvWindow': RECV_WINDOW
    }
    params['signature'] = sign_ed25519(params)
    async with session.post(f"{REST_BASE}/fapi/v1/order", data=params) as r:
        return await r.json()

# ————————— REST OCO 止盈止损 —————————
async def place_oco(side: str, qty: float, entry: float):
    # 主单
    r_main = await place_limit(side, qty, entry)
    # 组合单
    tp = entry * (1.02 if side=='BUY' else 0.98)
    sl = entry * (0.98 if side=='BUY' else 1.02)
    ts = int(time.time()*1000)
    oco = {
        'symbol': SYMBOL,
        'side':   'SELL' if side=='BUY' else 'BUY',
        'quantity': f"{qty:.6f}",
        'price':     f"{tp:.2f}",
        'stopPrice': f"{sl:.2f}",
        'stopLimitPrice': f"{sl*(1.001 if side=='BUY' else 0.999):.2f}",
        'stopLimitTimeInForce': 'GTC',
        'timestamp': ts,
        'recvWindow': RECV_WINDOW
    }
    oco['signature'] = sign_ed25519(oco)
    async with session.post(f"{REST_BASE}/api/v3/orderList/oco", data=oco) as r:
        r_oco = await r.json()
    logging.info("OCO result: %s / %s", r_main, r_oco)

# ————————— 主策略 + 子策略 —————————
async def main_strategy():
    global last_side
    # 等待首条 markPrice
    while price_ts is None:
        await asyncio.sleep(0.5)
    while True:
        async with lock:
            p = latest_price
            # 主策略指标齐全检查
            if 'st' not in klines['15m'] or 'bb_pct' not in klines['1h'] or 'bb_pct' not in klines['3m']:
                await asyncio.sleep(0.5)
                continue
            st   = klines['15m']['st'].iloc[-1]
            bb1h = klines['1h']['bb_pct'].iloc[-1]
            bb3m = klines['3m']['bb_pct'].iloc[-1]
            up, down = p>st, p<st
            strong_long  = up  and bb1h<0.2
            strong_short = down and bb1h>0.8

            # 顺势强信号
            if strong_long and bb3m<=0 and last_side!='LONG':
                await place_oco('BUY',0.12,p); last_side='LONG'
            elif strong_short and bb3m>=1 and last_side!='SHORT':
                await place_oco('SELL',0.12,p); last_side='SHORT'
            # 顺势弱信号
            elif up and 0<bb1h<0.2 and 0<bb3m<=1 and last_side!='LONG':
                await place_oco('BUY',0.03,p); last_side='LONG'
            elif down and 0.8<bb1h<1 and 0<=bb3m<1 and last_side!='SHORT':
                await place_oco('SELL',0.03,p); last_side='SHORT'
            # 逆势强信号
            elif strong_long and bb3m>=1 and last_side!='SHORT':
                await place_oco('SELL',0.07,p); last_side='SHORT'
            elif strong_short and bb3m<=0 and last_side!='LONG':
                await place_oco('BUY',0.07,p); last_side='LONG'
        await asyncio.sleep(0.5)

# ————————— 子策略：15m MACD —————————
async def macd_strategy():
    trig = False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.empty or 'macd' not in df: continue
            macd, prev = df['macd'].iloc[-1], df['macd'].iloc[-2]
            if not trig and prev>0 and macd<prev and macd>=11:
                await place_oco('SELL',0.15,latest_price); trig=True
            if trig and prev<0 and macd>prev and macd<=-11:
                await place_oco('BUY',0.15,latest_price); trig=False

# ————————— 子策略：15m RVGI —————————
async def rvgi_strategy():
    cnt_long = cnt_short = 0
    while True:
        await asyncio.sleep(20)
        async with lock:
            df = klines['15m']
            if df.empty or 'rvgi' not in df or 'rvsig' not in df: continue
            rv, sig = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv>sig and cnt_long*0.05<0.2:
                await place_oco('BUY',0.05,latest_price); cnt_long+=1
            if rv<sig and cnt_short*0.05<0.2:
                await place_oco('SELL',0.05,latest_price); cnt_short+=1

# ————————— 子策略：Triple SuperTrend —————————
async def triple_st_strategy():
    firing = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.empty or 'st' not in df or len(df['st'])<3: continue
            p    = latest_price
            stv  = df['st']
            rising  = stv.iloc[-3] < stv.iloc[-2] < stv.iloc[-1] < p
            falling = stv.iloc[-3] > stv.iloc[-2] > stv.iloc[-1] > p
            if rising and not firing:
                await place_oco('BUY',0.15,p); firing = True
            if falling and not firing:
                await place_oco('SELL',0.15,p); firing = True
            prev = stv.iloc[-2]
            if firing and ((rising and p<prev) or (falling and p>prev)):
                side = 'SELL' if rising else 'BUY'
                await place_oco(side,0.15,p)
                firing = False

# ————————— 启动 —————————
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    await asyncio.gather(
        market_ws(),
        main_strategy(),
        macd_strategy(),
        rvgi_strategy(),
        triple_st_strategy(),
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        asyncio.run(session.close())