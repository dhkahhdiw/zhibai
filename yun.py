#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import hmac
import hashlib
import asyncio
import logging
import uuid
import urllib.parse
import base64

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
load_dotenv('/root/zhibai/.env')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    API_KEY          = os.getenv('BINANCE_API_KEY')
    SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API      = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET_URL    = (
        'wss://fstream.binance.com/stream?streams='
        f'{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice'
    )
    WS_USER_URL      = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE        = 'https://fapi.binance.com'
    RECV_WINDOW      = 5000
    MIN_NOTIONAL_USD = 20.0
    SYNC_INTERVAL    = 300  # 5分钟

# 全局状态
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None

# 加载 Ed25519 私钥
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— DataManager：本地 K 线增量维护 & 指标计算 ——
class DataManager:
    def __init__(self):
        self.klines = {'3m': pd.DataFrame(columns=['open','high','low','close']),
                       '15m': pd.DataFrame(columns=['open','high','low','close']),
                       '1h': pd.DataFrame(columns=['open','high','low','close'])}
        self.last_ts = {'3m': 0, '15m': 0, '1h': 0}
        self.lock = asyncio.Lock()

    async def update_kline(self, tf, rec):
        async with self.lock:
            df = self.klines[tf]
            # rec['t'] 是当前 K 线的开时刻毫秒
            if rec['t'] > self.last_ts[tf]:
                # 新周期：追加新行
                new_row = pd.DataFrame([{
                    'open': rec['o'], 'high': rec['h'],
                    'low': rec['l'], 'close': rec['c']
                }])
                self.klines[tf] = pd.concat([df, new_row], ignore_index=True)
                self.last_ts[tf] = rec['t']
            else:
                # 同周期内更新最后一行
                self.klines[tf].iloc[-1] = [rec['o'], rec['h'], rec['l'], rec['c']]
            # 计算指标
            self._update_indicators(tf)

    def _update_indicators(self, tf):
        df = self.klines[tf]
        if len(df) < 20:
            return
        # Bollinger Bands
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st']   = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi']  = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df.columns and len(df) > 0:
                return df[col].iat[-1]
        return None

data_mgr = DataManager()

# —— OrderManager：集中下单、撤单、查询管理 ——
class OrderManager:
    def __init__(self):
        self.active = {}
        self.lock = asyncio.Lock()

    async def place(self, side, otype, qty=None, price=None, stopPrice=None, reduceOnly=False):
        ts = int(time.time() * 1000 + time_offset)
        params = {
            'symbol': Config.SYMBOL,
            'side': side,
            'type': otype,
            'timestamp': ts,
            'recvWindow': Config.RECV_WINDOW
        }
        if is_hedge_mode and otype in ('LIMIT','MARKET'):
            params['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'
        if otype == 'LIMIT':
            params.update({
                'timeInForce':'GTC',
                'quantity':f"{qty:.6f}",
                'price':f"{price:.2f}"
            })
            if reduceOnly:
                params['reduceOnly'] = 'true'
        elif otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
            params.update({
                'closePosition':'true',
                'stopPrice':f"{stopPrice:.2f}"
            })
        else:
            params['quantity'] = f"{qty:.6f}"
        qs = urllib.parse.urlencode(sorted(params.items()))
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        async with self.lock:
            start = time.time()
            res = await (await session.post(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
            elapsed = (time.time() - start)*1000
            logging.info("Order %s %s sent in %.1fms", otype, side, elapsed)
            if res.get('code'):
                logging.error("Order ERR %s %s: %s", otype, side, res)
                return None
            oid = res['orderId']
            self.active[oid] = res
            # 名义值检查
            if otype == 'LIMIT':
                notional = float(res['origQty'])*float(res['price'])
                if notional < Config.MIN_NOTIONAL_USD:
                    logging.warning("Notional %.2f < %.2f, cancel %s", notional, Config.MIN_NOTIONAL_USD, oid)
                    await self.cancel(oid)
                    return None
            logging.info("Order OK %s %s qty=%s id=%s", otype, side, qty or '', oid)
            return oid

    async def cancel(self, oid):
        qs = urllib.parse.urlencode({
            'symbol': Config.SYMBOL,
            'orderId': oid,
            'timestamp': int(time.time()*1000 + time_offset),
            'recvWindow': Config.RECV_WINDOW
        })
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        async with self.lock:
            res = await (await session.delete(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
            if res.get('code'):
                logging.error("Cancel ERR %s: %s", oid, res)
            else:
                logging.info("Cancel OK %s", oid)
                self.active.pop(oid, None)

    async def sync(self):
        qs = urllib.parse.urlencode({
            'timestamp': int(time.time()*1000 + time_offset),
            'recvWindow': Config.RECV_WINDOW
        })
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/openOrders?{qs}&signature={sig}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
        async with self.lock:
            self.active = {o['orderId']:o for o in res}
            logging.info("Synced %d open orders", len(self.active))

order_mgr = OrderManager()

# —— SignalRouter：同趋势单轮触发控制 ——
class SignalRouter:
    def __init__(self):
        self.last_trend = None
        self.fired = False

    def reset(self, trend):
        if trend != self.last_trend:
            self.last_trend = trend
            self.fired = False

    def allow(self, trend):
        if not self.fired and trend == self.last_trend:
            self.fired = True
            return True
        return False

router = SignalRouter()

# 周期性维护：校准时间 & 同步挂单
async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await order_mgr.sync()

# 时间同步 & 模式检测
async def sync_time():
    global time_offset
    srv = (await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json())['serverTime']
    time_offset = srv - int(time.time()*1000)
    logging.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    is_hedge_mode = (await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()).get('dualSidePosition',False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# WebSocket：市场数据
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o = json.loads(msg); s,d=o['stream'],o['data']
                    if s.endswith('@markPrice'):
                        latest_price, price_ts = float(d['p']), time.time()
                    if 'kline' in s:
                        tf = s.split('@')[1].split('_')[1]
                        k = d['k']
                        rec = {'t':k['t'],'o':float(k['o']),'h':float(k['h']),'l':float(k['l']),'c':float(k['c'])}
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(2)

# WebSocket：用户流
async def user_ws():
    global ed_priv
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params={'apiKey':Config.ED25519_API,'timestamp':int(time.time()*1000)}
                payload='&'.join(f"{k}={v}" for k,v in sorted(params.items()))
                sig=base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params['signature']=sig
                await ws.send(json.dumps({'id':str(uuid.uuid4()),'method':'session.logon','params':params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({'id':str(uuid.uuid4()),'method':'session.status'}))
                asyncio.create_task(hb())
                async for _ in ws: pass
        except Exception as e:
            logging.error("User WS error: %s", e)
            await asyncio.sleep(5)

# 趋势监控
async def trend_watcher():
    while True:
        await asyncio.sleep(0.5)
        if latest_price is None: continue
        st = await data_mgr.get('15m','st')
        if st is None: continue
        trend = 'UP' if latest_price>st else 'DOWN'
        router.reset(trend)

# 主策略（3m BB%B 挂单/止盈/止损）
async def main_strategy():
    levels  = [0.0025,0.0040,0.0060,0.0080,0.0160]
    tp_offs = [0.0102,0.0123,0.0150,0.0180,0.0220]
    sl_up, sl_dn = 0.98, 1.02

    while price_ts is None:
        await asyncio.sleep(0.1)
    while True:
        await asyncio.sleep(0.5)
        p    = latest_price
        bb1  = await data_mgr.get('1h','bb_pct')
        bb3  = await data_mgr.get('3m','bb_pct')
        stv  = await data_mgr.get('15m','st')
        if None in (bb1,bb3,stv): continue
        trend = 'UP' if p>stv else 'DOWN'
        if (bb3<=0 or bb3>=1) and router.allow(trend):
            strong = (trend=='UP' and bb1<0.2) or (trend=='DOWN' and bb1>0.8)
            qty = 0.12 if (trend=='UP' and strong) else 0.03
            if trend=='DOWN': qty = 0.07 if strong else 0.015
            side, rev = ('BUY','SELL') if trend=='UP' else ('SELL','BUY')
            # 挂单
            for off in levels:
                po = p*(1+off if side=='BUY' else 1-off)
                await order_mgr.place(side,'LIMIT',qty=qty,price=po)
            # 止盈
            for off in tp_offs:
                pt = p*(1+off if rev=='BUY' else 1-off)
                await order_mgr.place(rev,'LIMIT',qty=qty*0.2,price=pt,reduceOnly=True)
            # 止损
            slp = p*(sl_up if trend=='UP' else sl_dn)
            ttype = 'STOP_MARKET' if trend=='UP' else 'TAKE_PROFIT_MARKET'
            await order_mgr.place(rev,ttype,stopPrice=slp)

# 子策略：15m MACD
macd_cycle = None
async def macd_strategy():
    global macd_cycle
    while True:
        await asyncio.sleep(15)
        df = data_mgr.klines['15m']
        if len(df)<27 or 'macd' not in df.columns: continue
        prev, cur = df['macd'].iat[-2], df['macd'].iat[-1]
        osc = abs(cur)
        if prev>0 and cur<prev and osc>=11 and macd_cycle!='DOWN':
            macd_cycle='DOWN'; await order_mgr.place('SELL','MARKET',qty=0.017)
        if prev<0 and cur>prev and osc>=11 and macd_cycle!='UP':
            macd_cycle='UP'; await order_mgr.place('BUY','MARKET',qty=0.017)

# 子策略：15m RVGI
rvgi_cycle = None
async def rvgi_strategy():
    global rvgi_cycle
    while True:
        await asyncio.sleep(10)
        df = data_mgr.klines['15m']
        if len(df)<11 or 'rvgi' not in df.columns: continue
        rv, sg = df['rvgi'].iat[-1], df['rvsig'].iat[-1]
        if rv>sg and rvgi_cycle!='UP':
            rvgi_cycle='UP'
            await order_mgr.place('BUY','MARKET',qty=0.016)
            await order_mgr.place('SELL','LIMIT',qty=0.016,price=latest_price*1.06,reduceOnly=True)
            await order_mgr.place('SELL','STOP_MARKET',stopPrice=latest_price*0.98)
        if rv<sg and rvgi_cycle!='DOWN':
            rvgi_cycle='DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.016)
            await order_mgr.place('BUY','LIMIT',qty=0.016,price=latest_price*0.94,reduceOnly=True)
            await order_mgr.place('BUY','TAKE_PROFIT_MARKET',stopPrice=latest_price*1.02)

# 子策略：15m 三重 SuperTrend
triple_cycle = None
def supertrend(df, period, mult):
    hl2 = (df['high']+df['low'])/2
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    up, dn = hl2+mult*atr, hl2-mult*atr
    st = pd.Series(index=df.index); dirc = pd.Series(True,index=df.index)
    for i in range(len(df)):
        if i==0:
            st.iat[0] = up.iat[0]
        else:
            prev=st.iat[i-1]; price=df['close'].iat[i]
            if price>prev:
                st.iat[i]=max(dn.iat[i],prev); dirc.iat[i]=True
            else:
                st.iat[i]=min(up.iat[i],prev); dirc.iat[i]=False
    return st, dirc

async def triple_st_strategy():
    global triple_cycle
    while True:
        await asyncio.sleep(30)
        df = data_mgr.klines['15m']
        if len(df)<12: continue
        s1,d1 = supertrend(df,10,1)
        s2,d2 = supertrend(df,11,2)
        s3,d3 = supertrend(df,12,3)
        up = d1.iat[-1] and d2.iat[-1] and d3.iat[-1]
        dn = not (d1.iat[-1] or d2.iat[-1] or d3.iat[-1])
        if up and triple_cycle!='UP':
            triple_cycle='UP'; await order_mgr.place('BUY','MARKET',qty=0.015)
        if dn and triple_cycle!='DOWN':
            triple_cycle='DOWN'; await order_mgr.place('SELL','MARKET',qty=0.015)
        if triple_cycle=='UP' and not up:
            await order_mgr.place('SELL','MARKET',qty=0.015)
        if triple_cycle=='DOWN' and not dn:
            await order_mgr.place('BUY','MARKET',qty=0.015)

# 启动
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    await detect_mode()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            maintenance(),
            trend_watcher(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy()
        )
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())