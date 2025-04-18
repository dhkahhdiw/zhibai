#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import hmac
import hashlib
import urllib.parse
import asyncio
from itertools import cycle
from dotenv import load_dotenv

import aiohttp
import pandas as pd
import numpy as np
import ta
from ta.trend import MACDIndicator  # 用于 MACD 计算

# ========== 配置 ==========
load_dotenv('/root/zhibai/.env')
API_KEY     = os.getenv('BINANCE_API_KEY', '')
API_SECRET  = os.getenv('BINANCE_SECRET_KEY', '')
SYMBOL      = 'ETHUSDC'

# 合约私有接口域名列表（支持容灾）
FAPI_DOMAINS = cycle([
    'https://fapi.binance.com',
    # 如果有备用域名可加入：
    # 'https://fapi1.binance.com',
])

# 接收窗口、时间单位
RECV_WINDOW = 5000
TIME_UNIT   = 'MILLISECOND'

# 策略参数
SUPER_LEN     = 10
SUPER_FACT    = 3.0
BB_PERIOD_H   = 20
BB_STD_H      = 2
BB_PERIOD_3   = 20
BB_STD_3      = 2
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
LADDER_OFFSETS = [0.25, 0.4, 0.6, 0.8, 1.6]  # 阶梯偏移百分比
LADDER_RATIO   = 0.2  # 每阶仓位占比

# ========== 交易对参数 & 格式化 ==========
class ExchangeInfoFut:
    def __init__(self):
        self.filters = {}
        self.default_stp = 'NONE'

    async def load(self, session):
        url = next(FAPI_DOMAINS) + '/fapi/v1/exchangeInfo'
        async with session.get(url) as r:
            data = await r.json()
        for s in data.get('symbols', []):
            if s.get('symbol') == SYMBOL:
                for f in s.get('filters', []):
                    self.filters[f['filterType']] = f
                self.default_stp = data.get('defaultSelfTradePreventionMode', 'NONE')
                break

    def fmt_price(self, price: float) -> str:
        tick = float(self.filters['PRICE_FILTER']['tickSize'])
        p = np.floor(price / tick) * tick
        prec = abs(int(np.log10(tick)))
        return f"{p:.{prec}f}"

    def fmt_qty(self, qty: float) -> str:
        step = float(self.filters['LOT_SIZE']['stepSize'])
        q = np.floor(qty / step) * step
        prec = abs(int(np.log10(step)))
        return f"{q:.{prec}f}"

# ========== 签名 & HTTP 请求 ==========
def sign(params: dict) -> str:
    qs = '&'.join(f"{k}={params[k]}" for k in sorted(params))
    raw = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).digest()
    return urllib.parse.quote_plus(raw.hex())

async def api_request(session, method: str, path: str, params=None, private=True):
    base = next(FAPI_DOMAINS) if private else 'https://fapi.binance.com'
    url = base + path
    headers = {}
    p = params.copy() if params else {}
    if private:
        ts = int(time.time() * 1000)
        p.update({'timestamp': ts, 'recvWindow': RECV_WINDOW})
        p['signature'] = sign(p)
        headers['X-MBX-APIKEY'] = API_KEY
    headers['X-MBX-TIME-UNIT'] = TIME_UNIT

    async with session.request(method, url, params=p, headers=headers) as resp:
        if resp.status in (418, 429):
            retry = int(resp.headers.get('Retry-After', 1))
            await asyncio.sleep(retry)
            return await api_request(session, method, path, params, private)
        return await resp.json()

# ========== 市场数据 & 指标 ==========
class MarketDataFut:
    def __init__(self):
        self._cache = {}

    async def fetch_klines(self, session, interval: str, limit=500):
        url = next(FAPI_DOMAINS) + '/fapi/v1/klines'
        async with session.get(url, params={'symbol': SYMBOL, 'interval': interval, 'limit': limit}) as r:
            data = await r.json()
        df = pd.DataFrame(data, columns=range(12))
        df = df.rename(columns={2: 'high', 3: 'low', 4: 'close'})
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].astype(float)
        return df[['high', 'low', 'close']]

    async def update(self, session):
        tasks = {tf: asyncio.create_task(self.fetch_klines(session, tf))
                 for tf in ('15m', '1h', '3m')}
        for tf, t in tasks.items():
            self._cache[tf] = await t

    def supertrend(self, df: pd.DataFrame, length: int, factor: float) -> pd.Series:
        hl2 = (df['high'] + df['low']) / 2
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], length)
        up = hl2 - factor * atr
        dn = hl2 + factor * atr
        trend = np.ones(len(df))
        for i in range(1, len(df)):
            if df['close'].iat[i] > up.iat[i-1]:
                trend[i] = 1
            elif df['close'].iat[i] < dn.iat[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                up.iat[i] = min(up.iat[i], up.iat[i-1]) if trend[i]==1 else up.iat[i]
                dn.iat[i] = max(dn.iat[i], dn.iat[i-1]) if trend[i]==-1 else dn.iat[i]
        return pd.Series(trend, index=df.index)

    def bb_percent(self, df: pd.DataFrame, period: int, std: float) -> pd.Series:
        bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std)
        return (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    def macd_diff(self, df: pd.DataFrame) -> pd.Series:
        macd = MACDIndicator(
            close=df['close'],
            window_fast=MACD_FAST,
            window_slow=MACD_SLOW,
            window_sign=MACD_SIGNAL
        )
        return macd.macd_diff()

# ========== 策略核心 ==========
class StrategyFut:
    def __init__(self):
        self.md    = MarketDataFut()
        self.exi   = ExchangeInfoFut()
        self.last_side  = None
        self.last_trend = None

    def compute_ladder(self, base_price: float, side: str, total_qty: float):
        sign_ = 1 if side=='SELL' else -1
        return [(base_price * (1 + sign_*off/100), total_qty * LADDER_RATIO)
                for off in LADDER_OFFSETS]

    async def place_bracket(self, session, side: str, qty: float, entry: float, slp: float, tpp: float):
        # 1) 限价入场
        ep = self.exi.fmt_price(entry)
        qf = self.exi.fmt_qty(qty)
        await api_request(session, 'POST', '/fapi/v1/order', {
            'symbol': SYMBOL, 'side': side, 'type': 'LIMIT',
            'timeInForce': 'GTC', 'quantity': qf, 'price': ep,
            'selfTradePreventionMode': self.exi.default_stp
        })
        # 2) 止损
        slp_f = self.exi.fmt_price(slp)
        await api_request(session, 'POST', '/fapi/v1/order', {
            'symbol': SYMBOL,
            'side': 'SELL' if side=='BUY' else 'BUY',
            'type': 'STOP_MARKET',
            'closePosition': 'true',
            'stopPrice': slp_f,
            'workingType': 'CONTRACT_PRICE',
            'selfTradePreventionMode': self.exi.default_stp
        })
        # 3) 止盈
        tpp_f = self.exi.fmt_price(tpp)
        await api_request(session, 'POST', '/fapi/v1/order', {
            'symbol': SYMBOL,
            'side': 'SELL' if side=='BUY' else 'BUY',
            'type': 'TAKE_PROFIT_MARKET',
            'closePosition': 'true',
            'stopPrice': tpp_f,
            'workingType': 'CONTRACT_PRICE',
            'selfTradePreventionMode': self.exi.default_stp
        })

    async def main_trend(self, session):
        df15, df1h, df3m = (self.md._cache[k] for k in ('15m','1h','3m'))
        st_val = self.md.supertrend(df15, SUPER_LEN, SUPER_FACT).iat[-1]
        trend = 'up' if st_val > 0 else 'down'
        if trend == self.last_trend:
            return

        bb1h = self.md.bb_percent(df1h, BB_PERIOD_H, BB_STD_H).iat[-1]
        bb3m = self.md.bb_percent(df3m, BB_PERIOD_3, BB_STD_3).iat[-1]

        if trend == 'up' and bb3m <= 0:
            side, qty = 'BUY', 0.12 if bb1h < 0.2 else 0.03
        elif trend == 'down' and bb3m >= 1:
            side, qty = 'SELL', 0.12 if bb1h > 0.8 else 0.03
        else:
            return

        if side == self.last_side:
            return

        price = df3m['close'].iat[-1]
        slp   = price * (0.98 if side=='BUY' else 1.02)
        tpp   = price * (1.02 if side=='BUY' else 0.98)

        await self.place_bracket(session, side, qty, price, slp, tpp)
        self.last_side  = side
        self.last_trend = trend

    async def macd_sub(self, session):
        df15 = self.md._cache['15m']
        diff = self.md.macd_diff(df15)
        cur, prev = diff.iat[-1], diff.iat[-2]
        if not ((cur < 0 <= prev) or (cur > 0 >= prev)):
            return

        side = 'SELL' if cur < 0 else 'BUY'
        if side == self.last_side:
            return

        qty   = 0.15
        price = df15['close'].iat[-1]
        slp   = price * (0.97 if side=='BUY' else 1.03)
        tpp   = price

        await self.place_bracket(session, side, qty, price, slp, tpp)
        self.last_side = side

    async def super_sub(self, session):
        df15 = self.md._cache['15m']
        s1 = self.md.supertrend(df15, 10, 1.0).iat[-1]
        s2 = self.md.supertrend(df15, 11, 2.0).iat[-1]
        s3 = self.md.supertrend(df15, 12, 3.0).iat[-1]
        if not ((s1>0 and s2>0 and s3>0) or (s1<0 and s2<0 and s3<0)):
            return

        side = 'BUY' if s1>0 else 'SELL'
        if side == self.last_side:
            return

        price = df15['close'].iat[-1]
        slp   = price * (0.97 if side=='BUY' else 1.03)
        tpp   = price * (1.01 if side=='BUY' else 0.99)

        await self.place_bracket(session, side, 0.15, price, slp, tpp)
        self.last_side = side

    async def tick(self, session):
        if not self.exi.filters:
            await self.exi.load(session)
        await self.md.update(session)
        await self.main_trend(session)
        await self.macd_sub(session)
        await self.super_sub(session)

    async def run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    await self.tick(session)
                except Exception as e:
                    print("策略异常：", e)
                await asyncio.sleep(3 * 60)

if __name__ == '__main__':
    asyncio.run(StrategyFut().run())