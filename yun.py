#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — ETHUSDC 合约策略（REST 轮询 + 一次性 OTOCO）
  • 每 3 秒轮询一次最新指标 & 价格
  • 主策略：3m %B + 15m SuperTrend + 1h %B
  • 子策略：15m MACD + 三因子 SuperTrend
  • 一次性提交入场限价 + 止盈 / 止损 条件单（OCO）
  • 严格轮流触发，防止同方向重复下单
"""

import os, time, asyncio, hmac, hashlib, urllib.parse
from itertools import cycle

import aiohttp
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import logging

# ========== 配置 ==========
load_dotenv('/root/zhibai/.env')
API_KEY    = os.getenv('BINANCE_API_KEY','')
API_SECRET = os.getenv('BINANCE_SECRET_KEY','')
SYMBOL     = 'ETHUSDC'

FAPI_DOMAINS = cycle(['https://fapi.binance.com'])
RECV_WINDOW  = 5000
TIME_UNIT    = 'MILLISECOND'

# 参数
SUPER_LEN, SUPER_FAC     = 10, 3.0
BB_H_P, BB_H_S           = 20, 2
BB_3_P, BB_3_S           = 20, 2
MACD_F, MACD_S, MACD_SIG = 12, 26, 9

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ========== 签名 & HTTP ==========
def sign(params: dict) -> str:
    qs = '&'.join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return urllib.parse.quote_plus(sig)

async def api_request(session, method, path, params=None, private=True):
    base = next(FAPI_DOMAINS)
    url  = base + path
    headers = {'X-MBX-TIME-UNIT': TIME_UNIT}
    p = (params or {}).copy()
    if private:
        ts = int(time.time()*1000)
        p.update({'timestamp':ts,'recvWindow':RECV_WINDOW})
        p['signature'] = sign(p)
        headers['X-MBX-APIKEY'] = API_KEY
    async with session.request(method, url, params=p, headers=headers) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise Exception(f"{resp.status}, {text}")
        return await resp.json()

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
        tasks = {tf: asyncio.create_task(self.fetch_klines(session,tf))
                 for tf in ('3m','15m','1h')}
        for tf,t in tasks.items():
            self.cache[tf] = await t

    def supertrend(self, df, length, mult):
        st = ta.supertrend(df['high'], df['low'], df['close'],
                           length=length, multiplier=mult)
        col = f"SUPERTd_{length}_{mult}"
        return st[col].iloc[-1]  # +1 up, -1 down

    def bbp(self, df, period, dev):
        mb = df['close'].rolling(period).mean()
        sd = df['close'].rolling(period).std()
        up, lo = mb+dev*sd, mb-dev*sd
        return ((df['close']-lo)/(up-lo)).iloc[-1]

    def macd_hist(self, df):
        mac = ta.macd(df['close'], fast=MACD_F, slow=MACD_S, signal=MACD_SIG)
        col = f"MACDh_{MACD_F}_{MACD_S}_{MACD_SIG}"
        return mac[col]

# ========== 策略 & 下单 ==========
class Strategy:
    def __init__(self):
        self.md = MarketData()
        self.last_main = self.last_macd = self.last_st3 = None

    async def place_otoco(self, session, side, qty, price, sl_price, tp_price):
        params = {
            'symbol': SYMBOL,
            'side': side,
            'quantity': qty,
            'price': price,
            'stopPrice': sl_price,
            'stopLimitPrice': sl_price,
            'stopLimitTimeInForce': 'GTC',
            'takeProfitPrice': tp_price,
            'takeProfitLimitPrice': tp_price,
            'takeProfitTimeInForce': 'GTC',
            'newOrderRespType': 'RESULT'
        }
        logging.info(f"下 OTOCO: side={side} qty={qty}@{price}, SL={sl_price}, TP={tp_price}")
        return await api_request(session, 'POST', '/fapi/v1/orderList/otoco', params)

    async def tick(self, session):
        await self.md.update(session)
        df3  = self.md.cache['3m']
        df15 = self.md.cache['15m']
        df1h = self.md.cache['1h']

        # 主策略
        st15   = self.md.supertrend(df15, SUPER_LEN, SUPER_FAC)
        bb1h   = self.md.bbp(df1h, BB_H_P, BB_H_S)
        bb3    = self.md.bbp(df3,  BB_3_P, BB_3_S)
        trend  = 1 if st15>0 else -1
        main_side = None; qty=0
        if bb3 <= 0:
            main_side = 'BUY'
            qty = 0.12 if (bb1h<0.2 and trend==1) else 0.03
        elif bb3 >= 1:
            main_side = 'SELL'
            qty = 0.12 if (bb1h>0.8 and trend==-1) else 0.03

        if main_side and main_side != self.last_main:
            price  = round(df3['close'].iloc[-1], 2)
            slp    = round(price*(0.98 if main_side=='BUY' else 1.02),2)
            tpp    = round(price*(1.02 if main_side=='BUY' else 0.98),2)
            await self.place_otoco(session, main_side, qty, price, slp, tpp)
            self.last_main = main_side

        # MACD
        hist    = self.md.macd_hist(df15)
        cur,prv = hist.iloc[-1], hist.iloc[-2]
        macd_side = 'BUY' if (prv<0<cur) else 'SELL' if (prv>0>cur) else None
        if macd_side and macd_side != self.last_macd:
            price = round(df15['close'].iloc[-1],2)
            slp   = round(price*(0.97 if macd_side=='BUY' else 1.03),2)
            tpp   = round(price*(1.02 if macd_side=='BUY' else 0.98),2)
            await self.place_otoco(session, macd_side, 0.15, price, slp, tpp)
            self.last_macd = macd_side

        # 三因子 SuperTrend
        s1 = self.md.supertrend(df15,10,1.0)>0
        s2 = self.md.supertrend(df15,11,2.0)>0
        s3 = self.md.supertrend(df15,12,3.0)>0
        st3_side = 'BUY' if all([s1,s2,s3]) else 'SELL' if not any([s1,s2,s3]) else None
        if st3_side and st3_side != self.last_st3:
            price = round(df15['close'].iloc[-1],2)
            slp   = round(price*(0.97 if st3_side=='BUY' else 1.03),2)
            tpp   = round(price*(1.02 if st3_side=='BUY' else 0.98),2)
            await self.place_otoco(session, st3_side, 0.15, price, slp, tpp)
            self.last_st3 = st3_side

    async def run(self):
        logging.info("策略启动，开始轮询…")
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    await self.tick(session)
                except Exception as e:
                    logging.error("策略异常：%s", e)
                await asyncio.sleep(1)
if __name__ == '__main__':
    try:
        asyncio.run(Strategy().run())
    except KeyboardInterrupt:
        logging.info("策略已收到停止信号，退出。")