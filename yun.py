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
    SYNC_INTERVAL    = 300        # 每5分钟同步一次

# —— 全局状态 ——
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None

# DataManager: 存储并增量更新本地K线和指标
class DataManager:
    def __init__(self):
        self.klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
        self.lock = asyncio.Lock()

    async def update_kline(self, tf, rec):
        async with self.lock:
            df = self.klines[tf]
            if df.empty or rec['t'] > df.index[-1]:
                row = pd.Series({
                    'open': rec['o'], 'high': rec['h'],
                    'low': rec['l'], 'close': rec['c']
                }, name=rec['t'])
                self.klines[tf] = pd.concat([df, pd.DataFrame([row])])
            else:
                self.klines[tf].iat[-1, 0:4] = [rec['o'], rec['h'], rec['l'], rec['c']]
            self._update_indicators(tf)

    def _update_indicators(self, tf):
        df = self.klines[tf]
        if len(df) < 20: return
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

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df.columns and len(df) > 0:
                return df[col].iat[-1]
        return None

data_mgr = DataManager()

# OrderManager: 集中管理订单
class OrderManager:
    def __init__(self):
        self.active_orders = {}
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
        if is_hedge_mode and otype in ('LIMIT', 'MARKET'):
            params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
        if otype == 'LIMIT':
            params.update({
                'timeInForce': 'GTC',
                'quantity': f"{qty:.6f}",
                'price': f"{price:.2f}"
            })
            if reduceOnly:
                params['reduceOnly'] = 'true'
        elif otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET'):
            params.update({
                'closePosition': 'true',
                'stopPrice': f"{stopPrice:.2f}"
            })
        else:
            params['quantity'] = f"{qty:.6f}"
        qs = urllib.parse.urlencode(sorted(params.items()))
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        async with self.lock:
            start = time.time()
            res = await (await session.post(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
            duration = (time.time() - start) * 1000
            logging.info("Order %s %s send took %.1fms", otype, side, duration)
            if res.get('code'):
                logging.error("Order ERR %s %s: %s", otype, side, res)
                return None
            oid = res['orderId']
            self.active_orders[oid] = res
            # 名义值检查
            if otype == 'LIMIT':
                notional = float(res['origQty']) * float(res['price'])
                if notional < Config.MIN_NOTIONAL_USD:
                    logging.warning("Notional %.2f < %.2f, canceling", notional, Config.MIN_NOTIONAL_USD)
                    await self.cancel(oid)
                    return None
            logging.info("Order OK %s %s qty=%s id=%s", otype, side, qty or '', oid)
            return oid

    async def cancel(self, orderId):
        qs = urllib.parse.urlencode({
            'symbol': Config.SYMBOL,
            'orderId': orderId,
            'timestamp': int(time.time() * 1000 + time_offset),
            'recvWindow': Config.RECV_WINDOW
        })
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        async with self.lock:
            res = await (await session.delete(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
            if res.get('code'):
                logging.error("Cancel ERR %s: %s", orderId, res)
            else:
                logging.info("Cancel OK %s", orderId)
                self.active_orders.pop(orderId, None)

    async def sync_positions(self):
        # 定期拉取仓位和挂单状态（示例：获取所有 open orders）
        url = f"{Config.REST_BASE}/fapi/v1/openOrders?timestamp={int(time.time()*1000+time_offset)}&recvWindow={Config.RECV_WINDOW}"
        sig = hmac.new(Config.SECRET_KEY, url.split('?')[1].encode(), hashlib.sha256).hexdigest()
        url = url + f"&signature={sig}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
        # 去重本地 active_orders
        async with self.lock:
            self.active_orders = {o['orderId']: o for o in res}
            logging.info("Synced %d open orders", len(self.active_orders))

order_mgr = OrderManager()

# SignalRouter: 轮换信号，防止重复
class SignalRouter:
    def __init__(self):
        self.last_trend = None
        self.last_signal = None

    def reset_on_trend_change(self, trend):
        if trend != self.last_trend:
            self.last_trend = trend
            self.last_signal = None

    def allow(self, trend, new_signal):
        # 同一趋势每轮只触发一次
        if new_signal == trend and self.last_signal is None:
            self.last_signal = new_signal
            return True
        return False

signal_router = SignalRouter()

# —— 时间漂移校准、仓位同步定时任务 ——
async def maintenance_tasks():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await order_mgr.sync_positions()

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    res = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    srv = res['serverTime']
    time_offset = srv - int(time.time() * 1000)
    logging.info("Time offset: %d ms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    is_hedge_mode = res.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# —— WebSocket：市场数据 & 账户流 ——
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o = json.loads(msg)
                    s, d = o['stream'], o['data']
                    if s.endswith('@markPrice'):
                        latest_price = float(d['p'])
                        price_ts = time.time()
                    if 'kline' in s:
                        tf = s.split('@')[1].split('_')[1]
                        k = d['k']
                        rec = {'t': k['t'], 'o': float(k['o']),
                               'h': float(k['h']), 'l': float(k['l']),
                               'c': float(k['c'])}
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(2)

async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params = {'apiKey': Config.ED25519_API, 'timestamp': int(time.time()*1000)}
                payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params['signature'] = sig
                await ws.send(json.dumps({'id': str(uuid.uuid4()),
                                          'method': 'session.logon',
                                          'params': params}))
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

# —— 趋势监控 & 信号复位 ——
async def trend_watcher():
    while True:
        await asyncio.sleep(0.5)
        if latest_price is None: continue
        st = await data_mgr.get('15m', 'st')
        if st is None: continue
        trend = 'UP' if latest_price > st else 'DOWN'
        signal_router.reset_on_trend_change(trend)

# —— 主策略 ———
async def main_strategy():
    levels    = [0.0025, 0.0040, 0.0060, 0.0080, 0.0160]
    tp_offs   = [0.0102, 0.0123, 0.0150, 0.0180, 0.0220]
    sl_mul_up = 0.98
    sl_mul_dn = 1.02

    while price_ts is None:
        await asyncio.sleep(0.1)

    while True:
        await asyncio.sleep(0.5)
        p = latest_price
        bb1 = await data_mgr.get('1h', 'bb_pct')
        bb3 = await data_mgr.get('3m', 'bb_pct')
        st  = await data_mgr.get('15m', 'st')
        if None in (bb1, bb3, st): continue

        trend = 'UP' if p > st else 'DOWN'
        # 仅在本轮未下单且3m BB%B突破时触发
        if bb3 <= 0 or bb3 >= 1:
            if signal_router.allow(trend, trend):
                strong = (trend == 'UP' and bb1 < 0.2) or (trend == 'DOWN' and bb1 > 0.8)
                qty = 0.12 if strong else 0.03
                if trend == 'DOWN':
                    qty = 0.07 if strong else 0.015
                side, rev = ('BUY','SELL') if trend=='UP' else ('SELL','BUY')
                # 分批挂单
                for off in levels:
                    po = p * (1+off if side=='BUY' else 1-off)
                    await order_mgr.place(side, 'LIMIT', qty=qty, price=po)
                # 止盈
                for off in tp_offs:
                    pt = p * (1+off if rev=='BUY' else 1-off)
                    await order_mgr.place(rev, 'LIMIT', qty=qty*0.2, price=pt, reduceOnly=True)
                # 止损
                slp = p * (sl_mul_up if trend=='UP' else sl_mul_dn)
                ttype = 'STOP_MARKET' if trend=='UP' else 'TAKE_PROFIT_MARKET'
                await order_mgr.place(rev, ttype, stopPrice=slp)

# —— 15m MACD 子策略 ——
async def macd_strategy():
    global macd_cycle
    while True:
        await asyncio.sleep(15)
        prev = await data_mgr.get('15m', 'macd')
        # require two bars back
        df = data_mgr.klines['15m']
        if len(df) < 27: continue
        cur = df['macd'].iat[-1]
        osc = abs(cur)
        if prev > 0 > cur and osc >= 11 and macd_cycle != 'DOWN':
            macd_cycle = 'DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.017)
        if prev < 0 < cur and osc >= 11 and macd_cycle != 'UP':
            macd_cycle = 'UP'
            await order_mgr.place('BUY','MARKET',qty=0.017)

# —— 15m RVGI 子策略 ——
async def rvgi_strategy():
    global rvgi_cycle
    while True:
        await asyncio.sleep(10)
        df = data_mgr.klines['15m']
        if len(df) < 11 or 'rvgi' not in df.columns: continue
        rv  = df['rvgi'].iat[-1]
        sg  = df['rvsig'].iat[-1]
        if rv > sg and rvgi_cycle != 'UP':
            rvgi_cycle = 'UP'
            await order_mgr.place('BUY','MARKET',qty=0.016)
            await order_mgr.place('SELL','LIMIT',qty=0.016,price=latest_price*1.06,reduceOnly=True)
            await order_mgr.place('SELL','STOP_MARKET',stopPrice=latest_price*0.98)
        if rv < sg and rvgi_cycle != 'DOWN':
            rvgi_cycle = 'DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.016)
            await order_mgr.place('BUY','LIMIT',qty=0.016,price=latest_price*0.94,reduceOnly=True)
            await order_mgr.place('BUY','TAKE_PROFIT_MARKET',stopPrice=latest_price*1.02)

# —— 15m 三重 SuperTrend 子策略 ——
def supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['high']+df['low'])/2
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    up, dn = hl2+multiplier*atr, hl2-multiplier*atr
    st   = pd.Series(index=df.index)
    dirc = pd.Series(True, index=df.index)
    for i in range(len(df)):
        if i==0: st.iat[0]=up.iat[0]
        else:
            prev = st.iat[i-1]; price=df['close'].iat[i]
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
        if len(df) < 12: continue
        s1,d1 = supertrend(df,10,1)
        s2,d2 = supertrend(df,11,2)
        s3,d3 = supertrend(df,12,3)
        up = d1.iat[-1] and d2.iat[-1] and d3.iat[-1]
        dn = not (d1.iat[-1] or d2.iat[-1] or d3.iat[-1])
        if up and triple_cycle != 'UP':
            triple_cycle = 'UP'
            await order_mgr.place('BUY','MARKET',qty=0.015)
        if dn and triple_cycle != 'DOWN':
            triple_cycle = 'DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.015)
        if triple_cycle=='UP' and not up:
            await order_mgr.place('SELL','MARKET',qty=0.015)
        if triple_cycle=='DOWN' and not dn:
            await order_mgr.place('BUY','MARKET',qty=0.015)

# —— 启动 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    await detect_mode()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            maintenance_tasks(),
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