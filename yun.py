#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 带心跳 & 日志环缓存（1000条）ETHUSDC 合约策略
  • 每次 tick 开始/结束打印 DEBUG 心跳
  • 自定义 RingBufferHandler 仅保留最近 1000 条日志
  • 主策略：3m %B + 15m SuperTrend + 1h %B
  • 子策略：15m MACD + 3因子 SuperTrend
  • OTOCO 一次性限价入场 + 止盈/止损
"""

import os, time, asyncio, hmac, hashlib, urllib.parse
from itertools import cycle
from collections import deque

import aiohttp
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import logging

# ========== 环境 & 参数 ==========
load_dotenv('/root/zhibai/.env')
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_SECRET_KEY', '')
SYMBOL = 'ETHUSDC'

FAPI_DOMAINS = cycle(['https://fapi.binance.com'])
RECV_WINDOW = 5000
TIME_UNIT = 'MILLISECOND'

# 策略参数
SUPER_LEN, SUPER_FAC = 10, 3.0
BB_H_P, BB_H_S = 20, 2
BB_3_P, BB_3_S = 20, 2
MACD_F, MACD_S, MACD_SIG = 12, 26, 9


# ========== 自定义 RingBufferHandler ==========
class RingBufferHandler(logging.Handler):
    """仅保留最近 maxlen 条日志消息。"""

    def __init__(self, inner_handler, maxlen=1000):
        super().__init__()
        self.buffer = deque(maxlen=maxlen)
        self.inner = inner_handler

    def emit(self, record):
        msg = self.format(record)
        # 存入环形缓冲区
        self.buffer.append(msg)
        # 同时交给内置 handler（通常为控制台或文件）输出
        self.inner.emit(record)


# ========== 日志配置 ==========
# 1) 控制台输出所有 INFO+
console_handler = logging.StreamHandler()
console_fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_fmt)
console_handler.setLevel(logging.INFO)

# 2) 环缓存 handler，DEBUG 起作用
ring_inner = logging.StreamHandler()  # 也输出到控制台
ring_inner.setFormatter(console_fmt)
ring_inner.setLevel(logging.DEBUG)
ring_handler = RingBufferHandler(ring_inner, maxlen=1000)
ring_handler.setLevel(logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(ring_handler)


# ========== 签名 & HTTP ==========
def sign(params: dict) -> str:
    qs = '&'.join(f"{k}={params[k]}" for k in sorted(params))
    return urllib.parse.quote_plus(
        hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    )


async def api_request(session, method, path, params=None, private=True):
    base = next(FAPI_DOMAINS)
    url = base + path
    headers = {'X-MBX-TIME-UNIT': TIME_UNIT}
    p = (params or {}).copy()
    if private:
        ts = int(time.time() * 1000)
        p.update({'timestamp': ts, 'recvWindow': RECV_WINDOW})
        p['signature'] = sign(p)
        headers['X-MBX-APIKEY'] = API_KEY

    if method.upper() == 'GET':
        async with session.get(url, params=p, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400: raise Exception(f"{resp.status}, {text}")
            return await resp.json()
    else:
        # OTOCO 必须放在 body(data)
        async with session.post(url, data=p, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400: raise Exception(f"{resp.status}, {text}")
            return await resp.json()


# ========== 市场数据 & 指标 ==========
class MarketData:
    def __init__(self):
        self.cache = {}

    async def fetch_klines(self, session, interval, limit=100):
        async with session.get(
                next(FAPI_DOMAINS) + '/fapi/v1/klines',
                params={'symbol': SYMBOL, 'interval': interval, 'limit': limit}
        ) as resp:
            data = await resp.json()
        df = pd.DataFrame(data, columns=range(12))
        df = df.rename(columns={2: 'high', 3: 'low', 4: 'close'})
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].astype(float)
        return df[['high', 'low', 'close']]

    async def update(self, session):
        tasks = {tf: asyncio.create_task(self.fetch_klines(session, tf))
                 for tf in ('3m', '15m', '1h')}
        for tf, task in tasks.items():
            self.cache[tf] = await task

    def supertrend(self, df, length, mult):
        s = ta.supertrend(df['high'], df['low'], df['close'],
                          length=length, multiplier=mult)
        return s[f"SUPERTd_{length}_{mult}"].iloc[-1]

    def bbp(self, df, period, dev):
        mb = df['close'].rolling(period).mean()
        sd = df['close'].rolling(period).std()
        up, lo = mb + dev * sd, mb - dev * sd
        return ((df['close'] - lo) / (up - lo)).iloc[-1]

    def macd_hist(self, df):
        m = ta.macd(df['close'], fast=MACD_F, slow=MACD_S, signal=MACD_SIG)
        return m[f"MACDh_{MACD_F}_{MACD_S}_{MACD_SIG}"]


# ========== 策略 & 下单 ==========
class Strategy:
    def __init__(self):
        self.md = MarketData()
        self.last_main = self.last_macd = self.last_st3 = None

    async def place_otoco(self, session, side, qty, price, slp, tpp):
        params = {
            'symbol': SYMBOL,
            'side': side, 'quantity': qty,
            'price': price,
            'stopPrice': slp, 'stopLimitPrice': slp, 'stopLimitTimeInForce': 'GTC',
            'takeProfitPrice': tpp, 'takeProfitLimitPrice': tpp, 'takeProfitTimeInForce': 'GTC',
            'newOrderRespType': 'RESULT'
        }
        logging.info(f"下 OTOCO → {side} {qty}@{price}, SL={slp}, TP={tpp}")
        return await api_request(session, 'POST', '/fapi/v1/orderList/otoco', params)

    async def tick(self, session):
        logging.debug("⚡ Tick 开始")
        await self.md.update(session)
        df3, df15, df1h = self.md.cache['3m'], self.md.cache['15m'], self.md.cache['1h']

        # — 主策略 —
        st15 = self.md.supertrend(df15, SUPER_LEN, SUPER_FAC)
        bb1h = self.md.bbp(df1h, BB_H_P, BB_H_S)
        bb3 = self.md.bbp(df3, BB_3_P, BB_3_S)
        side = qty = None
        if bb3 <= 0:
            side, qty = 'BUY', 0.12 if (bb1h < 0.2 and st15 > 0) else 0.03
        elif bb3 >= 1:
            side, qty = 'SELL', 0.12 if (bb1h > 0.8 and st15 < 0) else 0.03
        if side and side != self.last_main:
            price = round(df3['close'].iat[-1], 2)
            slp = round(price * (0.98 if side == 'BUY' else 1.02), 2)
            tpp = round(price * (1.02 if side == 'BUY' else 0.98), 2)
            await self.place_otoco(session, side, qty, price, slp, tpp)
            self.last_main = side

        # — MACD 子策略 —
        hist = self.md.macd_hist(df15)
        cur, prv = hist.iat[-1], hist.iat[-2]
        mside = 'BUY' if (prv < 0 < cur) else 'SELL' if (prv > 0 > cur) else None
        if mside and mside != self.last_macd:
            price = round(df15['close'].iat[-1], 2)
            slp = round(price * (0.97 if mside == 'BUY' else 1.03), 2)
            tpp = round(price * (1.02 if mside == 'BUY' else 0.98), 2)
            await self.place_otoco(session, mside, 0.15, price, slp, tpp)
            self.last_macd = mside

        # — 三因子 SuperTrend —
        s1 = self.md.supertrend(df15, 10, 1.0) > 0
        s2 = self.md.supertrend(df15, 11, 2.0) > 0
        s3 = self.md.supertrend(df15, 12, 3.0) > 0
        st3 = 'BUY' if all([s1, s2, s3]) else 'SELL' if not any([s1, s2, s3]) else None
        if st3 and st3 != self.last_st3:
            price = round(df15['close'].iat[-1], 2)
            slp = round(price * (0.97 if st3 == 'BUY' else 1.03), 2)
            tpp = round(price * (1.02 if st3 == 'BUY' else 0.98), 2)
            await self.place_otoco(session, st3, 0.15, price, slp, tpp)
            self.last_st3 = st3

        logging.debug("⚡ Tick 完成")

    async def run(self):
        logging.info("▶ 策略启动")
        async with aiohttp.ClientSession() as s:
            while True:
                try:
                    await self.tick(s)
                except Exception as e:
                    logging.error("策略异常：%s", e)
                await asyncio.sleep(3)


if __name__ == '__main__':
    try:
        asyncio.run(Strategy().run())
    except KeyboardInterrupt:
        logging.info("✋ 策略终止")