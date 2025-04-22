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
import base64
import uuid

import uvloop
import aiohttp
import pandas as pd
import websockets

from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# ———— 高性能事件循环 ————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ———— 加载环境变量 ————
load_dotenv('/root/zhibai/.env')

# ———— 全局配置 ————
class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    API_KEY          = os.getenv('BINANCE_API_KEY')
    SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode('ascii')
    ED25519_API      = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET_URL    = (
        'wss://fstream.binance.com/stream?streams='
        f'{PAIR}@kline_3m/'
        f'{PAIR}@kline_15m/'
        f'{PAIR}@kline_1h/'
        f'{PAIR}@markPrice'
    )
    WS_USER_URL      = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE        = 'https://fapi.binance.com'
    RECV_WINDOW      = 5000
    MAX_POS_ETH      = 0.49

# ———— 日志配置 ————
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ———— 全局状态 ————
session: aiohttp.ClientSession
time_offset = 0
latest_price = None
price_ts = None
klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
lock = asyncio.Lock()
last_side = None        # 'BUY' or 'SELL'
last_trend = None       # 'UP' or 'DOWN'
position = {'long': 0.0, 'short': 0.0}

# ———— 加载 Ed25519 私钥 ————
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# ———— 时间同步 ————
async def sync_time():
    global time_offset
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    loc = int(time.time() * 1000)
    time_offset = srv - loc
    logging.info("Time offset: %d ms", time_offset)

# ———— HMAC 签名 ————
def build_signed_query(params: dict) -> str:
    qs = urllib.parse.urlencode(sorted(params.items()), safe='')
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

# ———— WebSocket 用户签名 ————
def sign_ws(params: dict) -> str:
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = ed_priv.sign(payload.encode('ascii'))
    return base64.b64encode(sig).decode('ascii')

# ———— 下单函数（修正版） ————
async def rest_order(side: str, otype: str, qty: float=None, price: float=None, stopPrice: float=None):
    ts = int(time.time() * 1000 + time_offset)
    params = {
        'symbol':       Config.SYMBOL,
        'side':         side,
        'type':         otype,
        'timestamp':    ts,
        'recvWindow':   Config.RECV_WINDOW,
        'positionSide': 'BOTH',  # 单向模式下可用 BOTH；双向模式请调整为 LONG/SHORT
    }

    if otype == 'LIMIT':
        params.update({
            'timeInForce': 'GTC',
            'quantity':    f"{qty:.6f}",
            'price':       f"{price:.2f}"
        })
    elif otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET'):
        # 条件单：只带 closePosition 与 stopPrice，移除 qty
        params.update({
            'closePosition': 'true',
            'stopPrice':     f"{stopPrice:.2f}"
        })
    else:  # MARKET
        params.update({'quantity': f"{qty:.6f}"})

    query = build_signed_query(params)
    url   = f"{Config.REST_BASE}/fapi/v1/order?{query}"
    async with session.post(url, headers={'X-MBX-APIKEY': Config.API_KEY}) as r:
        res = await r.json()
    if res.get('code'):
        logging.error("Order ERR %s %s: %s", otype, side, res)
        return False
    logging.info("Order OK %s %s qty=%.4f", otype, side, qty or 0)
    # 仅 LIMIT/MARKET 调整持仓量
    if otype in ('LIMIT', 'MARKET'):
        if side == 'BUY':  position['long']  += qty or 0
        else:              position['short'] += qty or 0
    return True

# ———— OCO 挂单（修正版） ————
async def bracket(qty: float, entry: float, side: str):
    global last_side
    # 轮流触发：新方向必须与 last_side 不同
    if last_side == side:
        return
    # 仓位限额
    total = position['long'] + position['short'] + (qty or 0)
    if total > Config.MAX_POS_ETH:
        logging.warning("Max position exceeded: %.4f", total)
        return
    # 主单
    ok = await rest_order(side, 'LIMIT', qty=qty, price=entry)
    if not ok:
        return
    last_side = side
    # 止盈止损
    tp_price = entry * (1.02 if side == 'BUY' else 0.98)
    sl_price = entry * (0.98 if side == 'BUY' else 1.02)
    ps = 'SELL' if side == 'BUY' else 'BUY'
    await rest_order(ps, 'TAKE_PROFIT_MARKET', stopPrice=tp_price)
    await rest_order(ps, 'STOP_MARKET',          stopPrice=sl_price)

# ———— 指标计算 ————
def update_indicators():
    for tf, df in klines.items():
        if df.shape[0] < 20:
            continue
        bb = BollingerBands(df['close'], 20, 2)
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
            den = (df['high']  - df['low']).ewm(span=10).mean()
            df['rvgi']  = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ———— 市场数据 WS ————
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o = json.loads(msg); s=o['stream']; d=o['data']
                    if s.endswith('@markPrice'):
                        latest_price = float(d['p'])
                        price_ts     = time.time()
                    if 'kline' in s:
                        tf = s.split('@')[1].split('_')[1]; k=d['k']
                        rec = {
                            'open':  float(k['o']),
                            'high':  float(k['h']),
                            'low':   float(k['l']),
                            'close': float(k['c'])
                        }
                        async with lock:
                            df = klines[tf]
                            if df.empty or int(k['t']) > df.index[-1]:
                                klines[tf] = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
                            else:
                                df.iloc[-1] = list(rec.values()); klines[tf] = df
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(5)

# ———— Ed25519 用户流 WS ————
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params = {'apiKey': Config.ED25519_API, 'timestamp': int(time.time()*1000)}
                params['signature'] = sign_ws(params)
                await ws.send(json.dumps({'id': str(uuid.uuid4()), 'method':'session.logon','params':params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({'id': str(uuid.uuid4()), 'method':'session.status'}))
                asyncio.create_task(hb())
                async for msg in ws:
                    logging.debug("User WS <- %s", msg)
        except Exception as e:
            logging.error("User WS error: %s", e)
            await asyncio.sleep(5)

# ———— 趋势监控，重置周期 ————
async def trend_watcher():
    global last_trend, last_side
    while True:
        await asyncio.sleep(0.1)
        async with lock:
            if 'st' not in klines['15m'] or latest_price is None:
                continue
            st = klines['15m']['st'].iloc[-1]
            trend = 'UP' if latest_price > st else 'DOWN'
            if trend != last_trend:
                last_trend = trend
                last_side = None

# ———— 主策略 ————
async def main_strategy():
    while price_ts is None:
        await asyncio.sleep(0.2)
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if any(klines[tf].shape[0]<20 for tf in ('3m','15m','1h')):
                continue
            p   = latest_price
            bb1 = klines['1h']['bb_pct'].iloc[-1]
            bb3 = klines['3m']['bb_pct'].iloc[-1]
            st  = klines['15m']['st'].iloc[-1]
            up, down = p > st, p < st
            if up   and bb1 < 0.2 and bb3 <= 0:
                await bracket(0.12, p, 'BUY')
            elif down and bb1 > 0.8 and bb3 >= 1:
                await bracket(0.12, p, 'SELL')

# ———— 15m MACD 子策略 ————
async def macd_strategy():
    fired = False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.shape[0] < 26 or 'macd' not in df:
                continue
            prev, cur = df['macd'].iloc[-2], df['macd'].iloc[-1]
            if not fired and prev > 0 and cur < prev:
                await bracket(0.15, latest_price, 'SELL'); fired = True
            elif fired and prev < 0 and cur > prev:
                await bracket(0.15, latest_price, 'BUY');  fired = False

# ———— 3m RVGI 子策略 ————
async def rvgi_strategy():
    cnt_l = cnt_s = 0
    while True:
        await asyncio.sleep(10)
        async with lock:
            df = klines['3m']
            if df.shape[0] < 10 or 'rvgi' not in df:
                continue
            rv, sg = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv > sg and cnt_l*0.05 < 0.2:
                await bracket(0.05, latest_price, 'BUY');  cnt_l += 1
            if rv < sg and cnt_s*0.05 < 0.2:
                await bracket(0.05, latest_price, 'SELL'); cnt_s += 1

# ———— 15m Triple SuperTrend 子策略 ————
async def triple_st_strategy():
    active = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.shape[0] < 3 or 'st' not in df:
                continue
            stv = df['st']; p = latest_price
            rise = stv.iloc[-3] < stv.iloc[-2] < stv.iloc[-1] < p
            fall = stv.iloc[-3] > stv.iloc[-2] > stv.iloc[-1] > p
            if rise and not active:
                await bracket(0.15, p, 'BUY');  active = True
            elif fall and not active:
                await bracket(0.15, p, 'SELL'); active = True
            prev = stv.iloc[-2]
            if active and ((rise and p < prev) or (fall and p > prev)):
                side = 'SELL' if rise else 'BUY'
                await bracket(0.15, p, side)
                active = False

# ———— 启动 ————
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            trend_watcher(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy(),
        )
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())