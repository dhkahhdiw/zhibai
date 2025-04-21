#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import uuid
import asyncio
import logging
import base64
import urllib.parse
import math

import uvloop                                      # 高性能事件循环替换
import pandas as pd
import websockets
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD
from websockets.exceptions import ConnectionClosedError

# ————————— uvloop 替代默认 asyncio —————————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ————————— 配置 —————————
ENV_PATH         = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY          = os.getenv('BINANCE_API_KEY')
PRIVATE_KEY_PATH = os.getenv('ED25519_KEY_PATH')  # Ed25519 私钥路径
SYMBOL           = 'ETHUSDC'
PAIR_LOWER       = SYMBOL.lower()
WS_DATA_URL      = (
    'wss://fstream.binance.com/stream?streams='
    f'{PAIR_LOWER}@kline_3m/'
    f'{PAIR_LOWER}@kline_15m/'
    f'{PAIR_LOWER}@kline_1h/'
    f'{PAIR_LOWER}@markPrice'
)
WS_ORDER_URL     = 'wss://ws-fapi.binance.com/ws-fapi/v1'  # JSON‑RPC 下单
RECV_WINDOW      = 5000

# ————————— 全局状态 —————————
klines      = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
latest_price= None
price_ts    = None
order_ws    = None
last_side   = None
lock        = asyncio.Lock()
# 防止超过 5 消息/秒
order_semaphore = asyncio.Semaphore(5)

# ————————— 加载 Ed25519 私钥 & 签名 —————————
with open(PRIVATE_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

def sign_ed25519(params: dict) -> str:
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = ed_priv.sign(payload.encode('ASCII'))
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
            hl2 = (df['high'] + df['low'])/2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st']   = hl2 - 3*atr
            df['macd']= MACD(df['close'],12,26,9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi']  = num/den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ————————— WebSocket 市场数据 —————————
async def market_data_ws():
    global latest_price, price_ts
    async with websockets.connect(WS_DATA_URL) as ws:
        async for msg in ws:
            o = json.loads(msg)
            s = o.get('stream',''); d = o.get('data',{})
            if s.endswith('@markPrice'):
                latest_price = float(d['p'])
                price_ts     = int(time.time()*1000)
            if 'kline' in s:
                tf = s.split('@')[1].split('_')[1]
                k  = d['k']
                rec = {'open':float(k['o']),'high':float(k['h']),
                       'low': float(k['l']),'close':float(k['c'])}
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

# ————————— 初始化 JSON‑RPC WS 下单连接 —————————
async def init_order_ws():
    global order_ws
    order_ws = await websockets.connect(WS_ORDER_URL)
    # 定期 ping
    async def ping_loop():
        while True:
            try:
                await order_ws.ping()
            except ConnectionClosedError:
                return
            await asyncio.sleep(60)
    asyncio.create_task(ping_loop())

async def place_order_rpc(params: dict) -> dict:
    async with order_semaphore:
        await asyncio.sleep(0.21)  # 确保不超 5 msg/s
        req = {'id':str(uuid.uuid4()),'method':'order.place','params':params}
        try:
            await order_ws.send(json.dumps(req))
            resp = await order_ws.recv()
            return json.loads(resp)
        except ConnectionClosedError:
            logging.warning("Order WS closed, reconnecting...")
            await init_order_ws()
            await order_ws.send(json.dumps(req))
            resp = await order_ws.recv()
            return json.loads(resp)

# ————————— Bracket：入口单 + 止盈 + 止损 —————————
async def bracket(side: str, qty: float, price_val: float):
    ts = int(time.time()*1000)
    base = {
        'apiKey': API_KEY,
        'symbol': SYMBOL,
        'recvWindow': RECV_WINDOW,
        'timestamp': ts
    }
    # 入口限价单
    p1 = {**base, 'side':side,'type':'LIMIT','timeInForce':'GTC',
          'quantity':f"{qty:.6f}",'price':f"{price_val:.2f}"}
    p1['signature'] = sign_ed25519(p1)
    r1 = await place_order_rpc(p1)
    # 止盈
    tp = price_val * (1.02 if side=='BUY' else 0.98)
    p2 = {**base,
        'side':'SELL' if side=='BUY' else 'BUY',
        'type':'TAKE_PROFIT_MARKET','closePosition':'true',
        'stopPrice':f"{tp:.2f}"}
    p2['signature'] = sign_ed25519(p2)
    r2 = await place_order_rpc(p2)
    # 止损
    sl = price_val * (0.98 if side=='BUY' else 1.02)
    p3 = {**base,
        'side':'SELL' if side=='BUY' else 'BUY',
        'type':'STOP_MARKET','closePosition':'true',
        'stopPrice':f"{sl:.2f}"}
    p3['signature'] = sign_ed25519(p3)
    r3 = await place_order_rpc(p3)
    logging.info("Bracket: %s / %s / %s", r1, r2, r3)

# ————————— 主策略 + 子策略 —————————
async def main_strategy():
    global last_side
    while price_ts is None:
        await asyncio.sleep(0.5)
    while True:
        async with lock:
            p = latest_price
            if 'st' not in klines['15m'] or 'bb_pct' not in klines['1h'] or 'bb_pct' not in klines['3m']:
                await asyncio.sleep(0.5); continue
            st   = klines['15m']['st'].iloc[-1]
            bb1h = klines['1h']['bb_pct'].iloc[-1]
            bb3m = klines['3m']['bb_pct'].iloc[-1]
            up, down = p>st, p<st
            long_s  = up  and bb1h<0.2
            short_s = down and bb1h>0.8

            # 顺势强信号
            if long_s and bb3m<=0 and last_side!='LONG':
                await bracket('BUY',0.12,p); last_side='LONG'
            elif short_s and bb3m>=1 and last_side!='SHORT':
                await bracket('SELL',0.12,p); last_side='SHORT'
            # 顺势弱信号
            elif up and not long_s and 0<bb3m<=0.5 and last_side!='LONG':
                await bracket('BUY',0.03,p); last_side='LONG'
            elif down and not short_s and 0.5<=bb3m<1 and last_side!='SHORT':
                await bracket('SELL',0.03,p); last_side='SHORT'
            # 逆势强信号
            elif long_s and bb3m>=1 and last_side!='SHORT':
                await bracket('SELL',0.07,p); last_side='SHORT'
            elif short_s and bb3m<=0 and last_side!='LONG':
                await bracket('BUY',0.07,p); last_side='LONG'
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
                await bracket('SELL',0.15,latest_price); trig=True
            if trig and prev<0 and macd>prev and macd<=-11:
                await bracket('BUY',0.15,latest_price); trig=False

async def rvgi_strategy():
    cnt_l = cnt_s = 0
    while True:
        await asyncio.sleep(20)
        async with lock:
            df = klines['15m']
            if df.empty or 'rvgi' not in df or 'rvsig' not in df: continue
            rv, sig = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv>sig and cnt_l*0.05<0.2:
                await bracket('BUY',0.05,latest_price); cnt_l+=1
            if rv<sig and cnt_s*0.05<0.2:
                await bracket('SELL',0.05,latest_price); cnt_s+=1

async def triple_st_strategy():
    firing = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.empty or 'st' not in df or len(df['st'])<3: continue
            p   = latest_price
            stv = df['st']
            rising  = stv.iloc[-3] < stv.iloc[-2] < stv.iloc[-1] < p
            falling = stv.iloc[-3] > stv.iloc[-2] > stv.iloc[-1] > p
            if rising and not firing:
                await bracket('BUY',0.15,p); firing=True
            if falling and not firing:
                await bracket('SELL',0.15,p); firing=True
            prev_st = stv.iloc[-2]
            if firing and ((rising and p<prev_st) or (falling and p>prev_st)):
                side = 'SELL' if rising else 'BUY'
                await bracket(side,0.15,p); firing=False

# ————————— 程序入口 —————————
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    await init_order_ws()  # 建立下单 WS 连接 binanec合约.txt](file-service://file-LMnfGcqyxu7gkTs7aSjyZ3)币安更新文档.txt](file-service://file-UVewvyMfPwAbfn3RsNAkj1)
    await asyncio.gather(
        market_data_ws(),
        main_strategy(),
        macd_strategy(),
        rvgi_strategy(),
        triple_st_strategy(),
    )

if __name__ == '__main__':
    asyncio.run(main())