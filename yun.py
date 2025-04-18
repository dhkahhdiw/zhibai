#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, hmac, hashlib, urllib.parse, asyncio
from itertools import cycle
from dotenv import load_dotenv
import aiohttp, pandas as pd, numpy as np, ta

# ========== 配置 ==========
load_dotenv('/root/zhibai/.env')
API_KEY    = os.getenv('BINANCE_API_KEY','')
API_SECRET = os.getenv('BINANCE_SECRET_KEY','')
SYMBOL     = 'ETHUSDC'
# Futures 私有接口（可拓展多域名）
FAPI_DOMAINS = cycle(['https://fapi.binance.com'])
RECV_WINDOW   = 5000
TIME_UNIT     = 'MILLISECOND'  # 支持 MILLISECOND 或 MICROSECOND

# 策略参数
SUPER_LEN, SUPER_FACT       = 10, 3.0
BB_PERIOD_H, BB_STD_H       = 20, 2
BB_PERIOD_3, BB_STD_3       = 20, 2
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# ========== ExchangeInfo 预加载 & 格式化 ==========
class ExchangeInfoFut:
    def __init__(self):
        self.filters = {}
        self.default_stp = 'NONE'

    async def load(self, session):
        url = next(FAPI_DOMAINS) + '/fapi/v1/exchangeInfo'
        async with session.get(url) as r:
            data = await r.json()
        # 某合约的 filters
        for s in data['symbols']:
            if s['symbol']==SYMBOL:
                for f in s['filters']:
                    self.filters[f['filterType']] = f
                # triggerProtect 用于条件单保护
                self.trigger_protect = s.get('triggerProtect', 0.0)
                break
        # selfTradePrevention
        self.default_stp = data.get('defaultSelfTradePreventionMode','NONE')

    def fmt_price(self, p: float) -> str:
        tick = float(self.filters['PRICE_FILTER']['tickSize'])
        qty = (np.floor(p / tick) * tick)
        prec = abs(int(np.log10(tick)))
        return f"{qty:.{prec}f}"

    def fmt_qty(self, q: float) -> str:
        step = float(self.filters['LOT_SIZE']['stepSize'])
        qty = (np.floor(q / step) * step)
        prec = abs(int(np.log10(step)))
        return f"{qty:.{prec}f}"

# ========== 签名 & 请求 ==========
def sign(params: dict) -> str:
    qs = '&'.join(f"{k}={params[k]}" for k in sorted(params))
    raw = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).digest()
    return urllib.parse.quote_plus(raw.hex())

async def api_request(session, method, path, params=None, private=True):
    base = next(FAPI_DOMAINS) if private else 'https://fapi.binance.com'
    url  = base + path
    headers = {}
    if private:
        ts = int(time.time()*1000)
        p  = params.copy() if params else {}
        p.update({'timestamp': ts, 'recvWindow': RECV_WINDOW})
        p['signature'] = sign(p)
        headers['X-MBX-APIKEY'] = API_KEY
    headers['X-MBX-TIME-UNIT'] = TIME_UNIT
    async with session.request(method, url, params=(p if private else params), headers=headers) as r:
        if r.status in (418,429):
            retry = int(r.headers.get('Retry-After', 1))
            await asyncio.sleep(retry)
            return await api_request(session, method, path, params, private)
        return await r.json()

# ========== 市场数据 ==========
class MarketDataFut:
    def __init__(self):
        self._cache = {}

    async def fetch_klines(self, session, interval, limit=500):
        url = next(FAPI_DOMAINS) + '/fapi/v1/klines'
        params = {'symbol': SYMBOL, 'interval': interval, 'limit': limit}
        async with session.get(url, params=params) as r:
            data = await r.json()
        df = pd.DataFrame(data, columns=range(12))
        df['close'] = df[4].astype(float)
        df['high']  = df[2].astype(float)
        df['low']   = df[3].astype(float)
        return df[['close','high','low']]

    async def update(self, session):
        tasks = [asyncio.create_task(self.fetch_klines(session, tf))
                 for tf in ('15m','1h','3m')]
        res = await asyncio.gather(*tasks)
        self._cache = dict(zip(('15m','1h','3m'), res))

# ========== 策略 ==========
class StrategyFut:
    def __init__(self):
        self.md   = MarketDataFut()
        self.exi  = ExchangeInfoFut()
        self.last_side = None
        self.last_trend = None

    async def place_bracket(self, session, side, qty, entry, sl, tp):
        # 1) 限价入场
        ep = self.exi.fmt_price(entry)
        qf = self.exi.fmt_qty(qty)
        await api_request(session, 'POST', '/fapi/v1/order', {
            'symbol': SYMBOL, 'side': side, 'type': 'LIMIT',
            'timeInForce':'GTC','quantity':qf,'price':ep,
            'selfTradePreventionMode': self.exi.default_stp
        })
        # 2) 止损 (STOP_MARKET)
        slp = self.exi.fmt_price(sl)
        await api_request(session, 'POST','/fapi/v1/order', {
            'symbol': SYMBOL, 'side': 'SELL' if side=='BUY' else 'BUY',
            'type':'STOP_MARKET','closePosition':'true',
            'stopPrice':slp,'workingType':'CONTRACT_PRICE',
            'selfTradePreventionMode': self.exi.default_stp
        })
        # 3) 止盈 (TAKE_PROFIT_MARKET)
        tpp = self.exi.fmt_price(tp)
        await api_request(session,'POST','/fapi/v1/order', {
            'symbol': SYMBOL, 'side': 'SELL' if side=='BUY' else 'BUY',
            'type':'TAKE_PROFIT_MARKET','closePosition':'true',
            'stopPrice':tpp,'workingType':'CONTRACT_PRICE',
            'selfTradePreventionMode': self.exi.default_stp
        })

    async def main_trend(self, session):
        df15 = self.md._cache['15m']
        # SuperTrend
        st = ta.volatility.average_true_range(df15['high'],df15['low'],df15['close'],SUPER_LEN)
        trend = 'up' if df15['close'].iloc[-1]>st.iloc[-1] else 'down'
        if trend==self.last_trend: return
        bb1h = ta.volatility.BollingerBands(self.md._cache['1h']['close'],BB_PERIOD_H,BB_STD_H).bollinger_pband().iloc[-1]
        bb3m = ta.volatility.BollingerBands(self.md._cache['3m']['close'],BB_PERIOD_3,BB_STD_3).bollinger_pband().iloc[-1]
        # 信号判定
        if trend=='up' and bb3m<=0:
            side, qty = 'BUY', 0.12 if bb1h<0.2 else 0.03
        elif trend=='down' and bb3m>=1:
            side, qty = 'SELL', 0.12 if bb1h>0.8 else 0.03
        else:
            return
        if side==self.last_side: return
        price = self.md._cache['3m']['close'].iloc[-1]
        # 止损 / 止盈
        if side=='BUY':
            sl, tp = price*0.98, price*1.02
        else:
            sl, tp = price*1.02, price*0.98
        await self.place_bracket(session, side, qty, price, sl, tp)
        self.last_side  = side
        self.last_trend = trend

    async def macd_sub(self, session):
        df15 = self.md._cache['15m']
        diff = ta.trend.MACD(df15['close'],MACD_FAST,MACD_SLOW,MACD_SIGNAL).macd_diff()
        cur, prev = diff.iloc[-1], diff.iloc[-2]
        price = df15['close'].iloc[-1]
        if cur<0<=prev and abs(cur)>=11:
            side, qty = 'SELL', 0.15
        elif cur>0>=prev and abs(cur)>=11:
            side, qty = 'BUY',  0.15
        else:
            return
        if side==self.last_side: return
        sl = price*(0.97 if side=='BUY' else 1.03)
        tp = price*(1.00)
        await self.place_bracket(session, side, qty, price, sl, tp)
        self.last_side = side

    async def super_sub(self, session):
        df15 = self.md._cache['15m']
        sts = [ta.trend.STC(df15['close'],fast=10,slow=26,short=23).stc() for _ in (1,2,3)]
        # 简化：当三条 STC 都极端时触发
        buy  = all(s.iloc[-1]>0.8 for s in sts)
        sell = all(s.iloc[-1]<0.2 for s in sts)
        if not (buy or sell): return
        side = 'BUY' if buy else 'SELL'
        if side==self.last_side: return
        price = df15['close'].iloc[-1]
        sl = price*(0.97 if side=='BUY' else 1.03)
        tp = price*(1.01 if side=='BUY' else 0.99)
        await self.place_bracket(session, side, 0.15, price, sl, tp)
        self.last_side = side

    async def tick(self, session):
        if not self.exi.filters:
            await self.exi.load(session)
        await self.md.update(session)
        await self.main_trend(session)
        await self.macd_sub(session)
        await self.super_sub(session)

    async def run(self):
        async with aiohttp.ClientSession() as s:
            while True:
                try:    await self.tick(s)
                except Exception as e:
                    print("Error:", e)
                await asyncio.sleep(3*60)

if __name__=='__main__':
    asyncio.run(StrategyFut().run())
