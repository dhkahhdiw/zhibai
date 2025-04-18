#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — ETHUSDC 合约交易策略（REST 版，实时轮询）
  • 主策略：3m %B + 15m SuperTrend + 1h %B 强弱
  • 子策略①：15m MACD（12,26,9）
  • 子策略②：15m 三因子 SuperTrend (10,1),(11,2),(12,3)
  • 一次性提交限价入场 + OCO 止盈/止损
  • 严格轮流触发，防止同方向重复下单
"""

import os, time, asyncio, hmac, hashlib, urllib.parse
from itertools import cycle

import aiohttp
import pandas as pd
import numpy as np
import pandas_ta as ta  # 需安装 pandas-ta
from dotenv import load_dotenv

# ========== 配置 ==========
# 强制指定 .env 路径
load_dotenv('/root/zhibai/.env')
API_KEY    = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_SECRET_KEY', '')
SYMBOL     = 'ETHUSDC'

# 接口与签名
FAPI_DOMAINS = cycle(['https://fapi.binance.com'])
RECV_WINDOW  = 5000
TIME_UNIT    = 'MILLISECOND'

# 策略参数
SUPER_LEN, SUPER_FACTOR = 10, 3.0
BB_PERIOD_H, BB_STD_H   = 20, 2
BB_PERIOD_3, BB_STD_3   = 20, 2
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# ========== 签名 & 请求 ==========
def sign(params: dict) -> str:
    qs = '&'.join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return urllib.parse.quote_plus(sig)

async def api_request(session, method, path, params=None, private=True):
    base = next(FAPI_DOMAINS)
    url  = base + path
    headers = {'X-MBX-TIME-UNIT': TIME_UNIT}
    p = params.copy() if params else {}
    if private:
        ts = int(time.time()*1000)
        p.update({'timestamp': ts, 'recvWindow': RECV_WINDOW})
        p['signature'] = sign(p)
        headers['X-MBX-APIKEY'] = API_KEY
    async with session.request(method, url, params=p, headers=headers) as r:
        data = await r.json()
        if r.status in (418, 429):
            await asyncio.sleep(int(r.headers.get('Retry-After',1)))
            return await api_request(session, method, path, params, private)
        return data

# ========== 市场数据 & 指标 ==========
class MarketData:
    def __init__(self):
        self.cache = {}

    async def fetch_klines(self, session, interval, limit=100):
        resp = await session.get(
            next(FAPI_DOMAINS) + '/fapi/v1/klines',
            params={'symbol':SYMBOL,'interval':interval,'limit':limit}
        )
        data = await resp.json()
        df = pd.DataFrame(data, columns=range(12))
        df = df.rename(columns={2:'high',3:'low',4:'close'})
        df[['high','low','close']] = df[['high','low','close']].astype(float)
        return df[['high','low','close']]

    async def update(self, session):
        tasks = {
            '3m':  asyncio.create_task(self.fetch_klines(session,'3m')),
            '15m': asyncio.create_task(self.fetch_klines(session,'15m')),
            '1h':  asyncio.create_task(self.fetch_klines(session,'1h')),
        }
        for k,t in tasks.items():
            self.cache[k] = await t

    def supertrend(self, df, length, factor):
        st = ta.supertrend(high=df['high'], low=df['low'], close=df['close'],
                           length=length, multiplier=factor)
        # 趋势方向列名 f"SUPERTd_{length}_{factor}"
        col = f"SUPERTd_{length}_{factor}"
        return st[col].iloc[-1]  # +1 上升, -1 下降

    def bb_percent(self, df, period, std):
        mb = df['close'].rolling(period).mean()
        sd = df['close'].rolling(period).std()
        up = mb + std*sd
        lo = mb - std*sd
        return ((df['close'] - lo)/(up - lo)).iloc[-1]

    def macd_hist(self, df):
        macd = ta.macd(close=df['close'],
                       fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        # pandas_ta 返回列 MACDh_<fast>_<slow>_<signal>
        col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
        return macd[col]

# ========== 策略与下单 ==========
class Strategy:
    def __init__(self):
        self.md = MarketData()
        self.last_side = None
        self.last_main = None
        self.last_macd = None
        self.last_st3  = None

    async def place_otoco(self, session, side, qty, entry, slp, tpp):
        """
        一次性提交 OTOCO 订单：
         - leg1: limit entry
         - leg2: stop-market 止损
         - leg3: take-profit-market 止盈
        """
        params = {
            'symbol': SYMBOL,
            'side': side,
            'quantity': qty,
            'price': entry,
            'orderListId': int(time.time()),       # 保证唯一
            'stopPrice': slp,
            'stopLimitPrice': slp,                 # 市价触发即可
            'stopLimitTimeInForce': 'GTC',
            'listClientOrderId': f"otoco_{int(time.time()*1000)}",
            'limitClientOrderId': f"leg1_{int(time.time()*1000)}",
            'stopClientOrderId':  f"leg2_{int(time.time()*1000)}",
            'takeProfitClientOrderId': f"leg3_{int(time.time()*1000)}"
        }
        # futures OTOCO 在 /fapi/v1/orderList/otoco
        return await api_request(session, 'POST', '/fapi/v1/orderList/otoco', params)

    async def on_tick(self, session):
        await self.md.update(session)
        df3  = self.md.cache['3m']
        df15 = self.md.cache['15m']
        df1h = self.md.cache['1h']

        # ─── 主策略信号 ─────────────────────
        st15 = self.md.supertrend(df15, SUPER_LEN, SUPER_FACTOR)
        trend = 1 if st15 > 0 else -1
        bb1h = self.md.bb_percent(df1h, BB_PERIOD_H, BB_STD_H)
        bb3  = self.md.bb_percent(df3, BB_PERIOD_3, BB_STD_3)
        main_side = None; strength=None; qty=0

        if bb3 <= 0:  # 多信号
            main_side = 'BUY'; strength = 'strong' if bb1h < 0.2 else 'weak'
            qty = 0.12 if (strength=='strong' and trend==1) else 0.03
        elif bb3 >= 1:  # 空信号
            main_side = 'SELL'; strength = 'strong' if bb1h > 0.8 else 'weak'
            qty = 0.12 if (strength=='strong' and trend==-1) else 0.03

        # 只触发一次，并且防止同方向重复
        if main_side and main_side != self.last_main:
            price = df3['close'].iloc[-1]
            # 初始止损／止盈按2%预估
            slp = price * (0.98 if main_side=='BUY' else 1.02)
            tpp = price * (1.02 if main_side=='BUY' else 0.98)
            await self.place_otoco(session, main_side, qty, price, slp, tpp)
            self.last_main = main_side

        # ─── MACD 子策略 ────────────────────
        hist = self.md.macd_hist(df15)
        cur, prev = hist.iloc[-1], hist.iloc[-2]
        macd_side = None
        if prev < 0 < cur:
            macd_side = 'BUY'
            m_qty = 0.15
        elif prev > 0 > cur:
            macd_side = 'SELL'
            m_qty = 0.15

        if macd_side and macd_side != self.last_macd:
            price = df15['close'].iloc[-1]
            slp   = price * (0.97 if macd_side=='BUY' else 1.03)
            tpp   = price * (1.02 if macd_side=='BUY' else 0.98)
            await self.place_otoco(session, macd_side, m_qty, price, slp, tpp)
            self.last_macd = macd_side

        # ─── 三因子 SuperTrend 子策略 ──────
        s1 = self.md.supertrend(df15,10,1.0) > 0
        s2 = self.md.supertrend(df15,11,2.0) > 0
        s3 = self.md.supertrend(df15,12,3.0) > 0
        st3_side = None
        if all([s1,s2,s3]):
            st3_side, st3_qty = 'BUY', 0.15
        elif not any([s1,s2,s3]):
            st3_side, st3_qty = 'SELL', 0.15

        if st3_side and st3_side != self.last_st3:
            price = df15['close'].iloc[-1]
            slp   = price * (0.97 if st3_side=='BUY' else 1.03)
            tpp   = price * (1.02 if st3_side=='BUY' else 0.98)
            await self.place_otoco(session, st3_side, st3_qty, price, slp, tpp)
            self.last_st3 = st3_side

    async def run(self):
        async with aiohttp.ClientSession() as s:
            while True:
                try:
                    await self.on_tick(s)
                except Exception as e:
                    print("策略异常：", e)
                # 3 秒钟拉一次最新价格/指标
                await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(Strategy().run())