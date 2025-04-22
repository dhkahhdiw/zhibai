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

# —— 高性能事件循环 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# —— 环境变量 ——
load_dotenv('/root/zhibai/.env')

class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    API_KEY          = os.getenv('BINANCE_API_KEY')
    SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode('ascii')
    ED25519_API      = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET_URL    = (
        'wss://fstream.binance.com/stream?streams='
        f'{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice'
    )
    WS_USER_URL      = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE        = 'https://fapi.binance.com'
    RECV_WINDOW      = 5000  # ms

# —— 日志 ——
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# —— 全局变量 ——
session: aiohttp.ClientSession
time_offset = 0
latest_price = None
klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
lock = asyncio.Lock()
last_side = None     # 用于防止重复下单
last_trend = None    # 趋势切换时重置 last_side

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— 时间同步 ——
async def sync_time():
    global time_offset
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    loc = int(time.time() * 1000)
    time_offset = srv - loc
    logging.info("Time offset: %d ms", time_offset)

# —— HMAC SHA256 签名 ——
def sign_hmac(params: dict) -> str:
    """
    params: 不含 signature 的参数字典
    返回：hex 格式签名
    """
    # 按 key 字母排序，再 urlencode
    ordered = sorted(params.items())
    qs = urllib.parse.urlencode(ordered, safe='')
    return hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

# —— Ed25519 WebSocket 下单签名 ——
def sign_ws(params: dict) -> str:
    ordered = sorted(params.items())
    payload = '&'.join(f"{k}={v}" for k, v in ordered)
    sig = ed_priv.sign(payload.encode('ascii'))
    return base64.b64encode(sig).decode('ascii')

# —— REST 下单 ——
async def rest_order(side, otype, qty=None, price=None, stopPrice=None):
    """
    统一使用 GET query string 方式签名下单，确保签名串顺序与 HMAC 输入一致 binanec合约.txt](file-service://file-P3hAJHTSSJfaiw6k88eQkg)
    """
    ts = int(time.time() * 1000 + time_offset)
    # 1. 准备参数 dict（不含 signature）
    params = {
        'symbol': Config.SYMBOL,
        'side': side,
        'type': otype,
        'timestamp': ts,
        'recvWindow': Config.RECV_WINDOW,
    }
    if otype == 'LIMIT':
        params.update({
            'timeInForce': 'GTC',
            'quantity': f"{qty:.6f}",
            'price': f"{price:.2f}"
        })
    elif otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET'):
        params.update({
            'closePosition': 'true',
            'stopPrice': f"{stopPrice:.2f}"
        })
    else:
        params.update({'quantity': f"{qty:.6f}"})

    # 2. 计算签名
    signature = sign_hmac(params)

    # 3. 拼接最终 query string（与签名输入完全一致的排序）
    ordered = sorted(params.items())
    qs = urllib.parse.urlencode(ordered, safe='')
    url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={signature}"

    # 4. 发起请求
    async with session.post(url, headers={'X-MBX-APIKEY': Config.API_KEY}) as r:
        ret = await r.json()
    if ret.get('code'):
        logging.error("Order ERR %s %s: %s", otype, side, ret)
        return False
    logging.info("Order OK %s %s qty=%s", otype, side, qty or '')
    return True

# —— 多级挂单＋止盈＋止损 ——
async def bracket(entry, direction, strength):
    global last_side, last_trend
    if last_side == direction:
        return
    last_side = direction

    # 1. 计算单笔仓位
    qty = 0.12 if (strength == 'strong' and direction == last_trend) else \
          0.07 if strength == 'strong' else \
          0.03 if direction == last_trend else \
          0.015

    # 2. 多级限价开仓
    levels = [0.0025, 0.004, 0.006, 0.008, 0.016]
    for lvl in levels:
        price = entry * (1 + (lvl if direction == 'BUY' else -lvl))
        await rest_order(direction, 'LIMIT', qty, price=price)

    # 3. 止盈
    if strength == 'strong':
        tp_lv = [0.0102, 0.0123, 0.015, 0.018, 0.022]
        pct = 0.2
    else:
        tp_lv = [0.0123, 0.018]
        pct = 0.5
    for tp in tp_lv:
        await rest_order(
            'SELL' if direction == 'BUY' else 'BUY',
            'LIMIT',
            qty * pct,
            price=entry * (1 + (tp if direction == 'BUY' else -tp))
        )

    # 4. 初始止损
    sl = entry * (0.98 if direction == 'BUY' else 1.02)
    await rest_order(
        'SELL' if direction == 'BUY' else 'BUY',
        'STOP_MARKET',
        stopPrice=sl
    )

# —— 指标更新 ——
def update_indicators():
    for tf, df in klines.items():
        if df.shape[0] < 20:
            continue
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_up'] = bb.bollinger_hband()
        df['bb_dn'] = bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi'] = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()

# —— 市场数据 WebSocket ——
async def market_ws():
    global latest_price
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    stream, d = data['stream'], data['data']
                    if stream.endswith('@markPrice'):
                        latest_price = float(d['p'])
                    if 'kline' in stream:
                        tf = stream.split('@')[1].split('_')[1]
                        k = d['k']
                        rec = {
                            'open': float(k['o']),
                            'high': float(k['h']),
                            'low': float(k['l']),
                            'close': float(k['c'])
                        }
                        async with lock:
                            df = klines[tf]
                            if df.empty or int(k['t']) > df.index[-1]:
                                klines[tf] = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
                            else:
                                df.iloc[-1] = list(rec.values())
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(5)

# —— 用户账户 WebSocket ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                params = {
                    'apiKey': Config.ED25519_API,
                    'timestamp': int(time.time() * 1000 + time_offset),
                    'recvWindow': Config.RECV_WINDOW
                }
                params['signature'] = sign_ws(params)  # Ed25519 签名
                await ws.send(json.dumps({
                    'id': str(uuid.uuid4()),
                    'method': 'session.logon',
                    'params': params
                }))
                # 心跳
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({
                            'id': str(uuid.uuid4()),
                            'method': 'session.status'
                        }))
                asyncio.create_task(hb())
                async for _ in ws:
                    pass
        except Exception as e:
            logging.error("User WS error: %s", e)
            await asyncio.sleep(5)

# —— 趋势监控 ——
async def trend_watcher():
    global last_trend, last_side
    while True:
        await asyncio.sleep(0.1)
        async with lock:
            if 'st' not in klines['15m'] or latest_price is None:
                continue
            st = klines['15m']['st'].iloc[-1]
            trend = 'BUY' if latest_price > st else 'SELL'
            if trend != last_trend:
                last_trend, last_side = trend, None

# —— 主策略 ——
async def main_strategy():
    while latest_price is None:
        await asyncio.sleep(0.2)
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if any(klines[tf].shape[0] < 20 for tf in klines):
                continue
            p = latest_price
            bb1 = klines['1h']['bb_pct'].iloc[-1]
            bb3 = klines['3m']['bb_pct'].iloc[-1]
            strength = (
                'strong' if ((last_trend == 'BUY' and bb1 < 0.2)
                             or (last_trend == 'SELL' and bb1 > 0.8))
                else 'weak'
            )
            if (bb3 <= 0 and last_trend == 'BUY') or (bb3 >= 1 and last_trend == 'SELL'):
                await bracket(p, last_trend, strength)

# —— 子策略：MACD ——
async def macd_strategy():
    fired = False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.shape[0] < 26 or 'macd' not in df:
                continue
            prev, cur = df['macd'].iloc[-2], df['macd'].iloc[-1]
            if not fired and prev > 0 > cur:
                await bracket(latest_price, 'SELL', 'weak')
                fired = True
            elif fired and prev < cur:
                await bracket(latest_price, 'BUY', 'weak')
                fired = False

# —— 子策略：RVGI ——
async def rvgi_strategy():
    cnt_l = cnt_s = 0
    while True:
        await asyncio.sleep(10)
        async with lock:
            df = klines['3m']
            if df.shape[0] < 10 or 'rvgi' not in df:
                continue
            rv, sg = df['rvgi'].iloc[-1], df['rvsig'].iloc[-1]
            if rv > sg and cnt_l < 4:
                await bracket(latest_price, 'BUY', 'weak')
                cnt_l += 1
            if rv < sg and cnt_s < 4:
                await bracket(latest_price, 'SELL', 'weak')
                cnt_s += 1

# —— 子策略：Triple ST ——
async def triple_st_strategy():
    active = False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if df.shape[0] < 3 or 'st' not in df:
                continue
            stv = df['st']
            rise = stv.iloc[-3] < stv.iloc[-2] < stv.iloc[-1] < latest_price
            fall = stv.iloc[-3] > stv.iloc[-2] > stv.iloc[-1] > latest_price
            if rise and not active:
                await bracket(latest_price, 'BUY', 'weak')
                active = True
            elif fall and not active:
                await bracket(latest_price, 'SELL', 'weak')
                active = True
            prev = stv.iloc[-2]
            if active and ((rise and latest_price < prev) or (fall and latest_price > prev)):
                side = 'SELL' if rise else 'BUY'
                await bracket(latest_price, side, 'weak')
                active = False

# —— 程序入口 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    await asyncio.gather(
        market_ws(),
        user_ws(),
        trend_watcher(),
        main_strategy(),
        macd_strategy(),
        rvgi_strategy(),
        triple_st_strategy(),
    )
    await session.close()

if __name__ == '__main__':
    asyncio.run(main())