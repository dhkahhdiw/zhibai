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
import watchfiles
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# —— 高性能事件循环 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv('/root/zhibai/.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
    SYNC_INTERVAL    = 300   # 5分钟
    MAX_POSITION     = 2.0   # 最多持仓量

# 全局状态
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    js = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    srv = js['serverTime']            # 毫秒
    local_ms = int(time.time() * 1000)
    time_offset = srv - local_ms      # 毫秒偏差
    logging.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    is_hedge_mode = res.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# —— PositionTracker：仓位同步 ——
class PositionTracker:
    def __init__(self):
        self.long = 0.0
        self.short = 0.0
        self.lock = asyncio.Lock()

    async def sync(self):
        ts = int(time.time() * 1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
        async with self.lock:
            for pos in res:
                if pos['symbol'] == Config.SYMBOL:
                    amt = abs(float(pos['positionAmt']))
                    if pos['positionSide'] == 'LONG':
                        self.long = amt
                    else:
                        self.short = amt

    async def get_available(self, side: str) -> float:
        async with self.lock:
            if side == 'BUY':
                return max(0.0, Config.MAX_POSITION - (self.long - self.short))
            else:
                return max(0.0, Config.MAX_POSITION - (self.short - self.long))

pos_tracker = PositionTracker()

# —— OrderBatcher：批量下单 & 限速 ——
class OrderBatcher:
    def __init__(self):
        self.queue = []
        self.rate_limit = 10   # 每秒不超过10单
        self.last_ts = 0.0

    async def add(self, url, headers=None):
        self.queue.append((url, headers))
        await self.flush()

    async def flush(self):
        now = time.time()
        wait = 1.0 / self.rate_limit - (now - self.last_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        for url, headers in self.queue:
            await _do_order(url, headers)
        self.queue.clear()
        self.last_ts = time.time()

order_batcher = OrderBatcher()

# —— 底层下单 & 重试 ——
async def _do_order(url: str, headers: dict):
    for i in range(3):
        try:
            resp = await (await session.post(url, headers=headers)).json()
            if resp.get('code'):
                if resp['code'] in (-2039, -2038):
                    logging.warning("Order conflict %s, retry %d...", resp['code'], i)
                    await asyncio.sleep(1)
                    continue
                logging.error("Order ERR %s %s", resp['code'], resp.get('msg'))
                return None
            return resp
        except aiohttp.ClientResponseError as e:
            if e.status == 418:
                logging.critical("IP Banned! Exiting.")
                os._exit(1)
            await asyncio.sleep(1)
    return None

# —— OrderManager：下单/撤单/同步 ——
class OrderManager:
    def __init__(self):
        self.active = {}
        self.lock = asyncio.Lock()

    def _sign(self, params: dict) -> str:
        qs = urllib.parse.urlencode(sorted(params.items()))
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        return qs + "&signature=" + sig

    async def place(self, side: str, otype: str, qty=None, price=None, stopPrice=None, reduceOnly=False):
        ts = int(time.time() * 1000 + time_offset)
        params = {
            'symbol': Config.SYMBOL, 'side': side, 'type': otype,
            'timestamp': ts, 'recvWindow': Config.RECV_WINDOW
        }
        if is_hedge_mode and otype in ('LIMIT', 'MARKET'):
            params['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
        if otype == 'LIMIT':
            params.update({'timeInForce': 'GTC', 'quantity': f"{qty:.6f}", 'price': f"{price:.2f}"})
            if reduceOnly:
                params['reduceOnly'] = 'true'
        elif otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET'):
            params.update({'closePosition': 'true', 'stopPrice': f"{stopPrice:.2f}"})
        else:
            params['quantity'] = f"{qty:.6f}"

        url = f"{Config.REST_BASE}/fapi/v1/order?{self._sign(params)}"
        await order_batcher.add(url, headers={'X-MBX-APIKEY': Config.API_KEY})
        # 结果会被异步处理到 active 中
        # 这里只简单返回 None/ID，可根据需求改为等待返回
        return None

    async def cancel(self, oid: int):
        ts = int(time.time() * 1000 + time_offset)
        params = {'symbol': Config.SYMBOL, 'orderId': oid, 'timestamp': ts, 'recvWindow': Config.RECV_WINDOW}
        url = f"{Config.REST_BASE}/fapi/v1/order?{self._sign(params)}"
        await _do_order(url, {'X-MBX-APIKEY': Config.API_KEY})
        async with self.lock:
            self.active.pop(oid, None)
        logging.info("Cancel OK %s", oid)

    async def sync(self):
        ts = int(time.time() * 1000 + time_offset)
        params = {'timestamp': ts, 'recvWindow': Config.RECV_WINDOW}
        url = f"{Config.REST_BASE}/fapi/v1/openOrders?{self._sign(params)}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
        async with self.lock:
            self.active = {o['orderId']: o for o in res}
        logging.info("Synced %d orders", len(self.active))

order_mgr = OrderManager()

# —— OrderCleaner：智能撤单 ——
class OrderCleaner:
    async def check(self):
        now = time.time()
        async with order_mgr.lock:
            for oid, o in list(order_mgr.active.items()):
                t0 = o.get('updateTime', now*1000)/1000
                if o.get('type') == 'LIMIT' and now - t0 > 600:
                    await order_mgr.cancel(oid)
                if o.get('type') in ('STOP_MARKET', 'TAKE_PROFIT_MARKET'):
                    if abs(o.get('stopPrice', 0) - latest_price) / latest_price > 0.02:
                        await order_mgr.cancel(oid)

order_cleaner = OrderCleaner()

# —— SignalRouter：单趋势防重 ——
class SignalRouter:
    def __init__(self):
        self.last = None
        self.fired = False
    def reset(self, t):
        if t != self.last:
            self.last, self.fired = t, False
    def allow(self):
        if not self.fired:
            self.fired = True
            return True
        return False

router = SignalRouter()

# —— DataManager：K线增量 + 指标 ——
class BollingerInc:
    def __init__(self, w=20, d=2):
        self.win, self.dev = w, d
        self.prices = []
    def update(self, p):
        self.prices.append(p)
        if len(self.prices) > self.win:
            self.prices.pop(0)
        if len(self.prices) == self.win:
            sma = sum(self.prices) / self.win
            std = (sum((x - sma)**2 for x in self.prices) / self.win)**0.5
            return sma + self.dev*std, sma - self.dev*std
        return None, None

class DataManager:
    def __init__(self):
        cols = ['open', 'high', 'low', 'close']
        self.klines = {tf: pd.DataFrame(columns=cols) for tf in ('3m','15m','1h')}
        self.last_ts = {tf: 0 for tf in ('3m','15m','1h')}
        self.lock = asyncio.Lock()
        self.bb3 = BollingerInc()

    async def update_kline(self, tf, rec):
        async with self.lock:
            df = self.klines[tf]
            nr = pd.DataFrame([[rec['o'], rec['h'], rec['l'], rec['c']]], columns=df.columns)
            if df.empty:
                self.klines[tf] = nr
            elif rec['t'] > self.last_ts[tf]:
                self.klines[tf] = pd.concat([df, nr], ignore_index=True)
            else:
                self.klines[tf].iloc[-1] = [rec['o'], rec['h'], rec['l'], rec['c']]
            self.last_ts[tf] = rec['t']
            self._update(tf)

    def _update(self, tf):
        df = self.klines[tf]
        if len(df) < 20:
            return
        if tf == '3m':
            up, dn = self.bb3.update(df['close'].iat[-1])
            if up:
                df.at[len(df)-1, 'bb_up'] = up
                df.at[len(df)-1, 'bb_dn'] = dn
            df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        else:
            bb = BollingerBands(df['close'], 20, 2)
            df['bb_up'] = bb.bollinger_hband()
            df['bb_dn'] = bb.bollinger_lband()
            df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df.columns and len(df) > 0:
                return df[col].iat[-1]
        return None

data_mgr = DataManager()

# —— WebSocket：市场数据 ——
async def market_ws():
    global latest_price, price_ts
    retry = 0
    while True:
        try:
            async with websockets.connect(
                Config.WS_MARKET_URL,
                ping_interval=15, ping_timeout=10, close_timeout=5
            ) as ws:
                logging.info("Market WS connected")
                retry = 0
                async for msg in ws:
                    o = json.loads(msg)
                    s, d = o['stream'], o['data']
                    if s.endswith('@markPrice'):
                        latest_price, price_ts = float(d['p']), time.time()
                    if 'kline' in s:
                        tf = s.split('@')[1].split('_')[1]
                        k = d['k']
                        rec = {'t': k['t'], 'o': float(k['o']),
                               'h': float(k['h']), 'l': float(k['l']),
                               'c': float(k['c'])}
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            delay = min(2**retry, 30)
            logging.error("Market WS err %s, retry in %ds", e, delay)
            await asyncio.sleep(delay)
            retry += 1

# —— WebSocket：用户流 ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params = {'apiKey': Config.ED25519_API, 'timestamp': int(time.time()*1000 + time_offset)}
                payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params['signature'] = sig
                await ws.send(json.dumps({'id': str(uuid.uuid4()), 'method': 'session.logon', 'params': params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({'id': str(uuid.uuid4()), 'method': 'session.status'}))
                asyncio.create_task(hb())
                async for _ in ws:
                    pass
        except Exception as e:
            logging.error("User WS err %s", e)
            await asyncio.sleep(5)

# —— 周期任务 & 热加载 ——
async def config_watcher():
    async for _ in watchfiles.awatch('/root/zhibai/'):
        logging.info("Config changed, reloading…")
        load_dotenv('/root/zhibai/.env')
        await pos_tracker.sync()
        await order_mgr.sync()

async def maintenance():
    asyncio.create_task(config_watcher())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await detect_mode()
        await pos_tracker.sync()
        await order_mgr.sync()
        await order_cleaner.check()

# —— 趋势监控 ——
async def trend_watcher():
    while True:
        await asyncio.sleep(0.5)
        if latest_price is None:
            continue
        st = await data_mgr.get('15m', 'st')
        if st is None:
            continue
        trend = 'UP' if latest_price > st else 'DOWN'
        router.reset(trend)

# —— 主策略（3m BB%B） ——
async def main_strategy():
    tp_offs = [0.0102, 0.0123, 0.0150, 0.0180, 0.0220]
    while price_ts is None:
        await asyncio.sleep(0.1)
    while True:
        await asyncio.sleep(0.5)
        p   = latest_price
        bb1 = await data_mgr.get('1h', 'bb_pct')
        bb3 = await data_mgr.get('3m', 'bb_pct')
        st  = await data_mgr.get('15m', 'st')
        if None in (bb1, bb3, st):
            continue
        if (bb3 <= 0 or bb3 >= 1) and router.allow():
            strong = (p > st and bb1 < 0.2) or (p < st and bb1 > 0.8)
            levels = dyn_params.update(st)
            qty = 0.12 if (p > st and strong) else 0.03
            if p < st:
                qty = 0.07 if strong else 0.015
            side, rev = ('BUY', 'SELL') if p > st else ('SELL', 'BUY')
            for off in levels:
                po = p * (1 + off if side == 'BUY' else 1 - off)
                await order_mgr.place(side, 'LIMIT', qty=qty, price=po)
            for off in tp_offs:
                pt = p * (1 + off if rev == 'BUY' else 1 - off)
                await order_mgr.place(rev, 'LIMIT', qty=qty*0.2, price=pt, reduceOnly=True)
            slp = p * (0.98 if p > st else 1.02)
            ttype = 'STOP_MARKET' if p > st else 'TAKE_PROFIT_MARKET'
            await order_mgr.place(rev, ttype, stopPrice=slp)

# —— 子策略：MACD ——
macd_cycle = None
async def macd_strategy():
    global macd_cycle
    while True:
        await asyncio.sleep(15)
        df = data_mgr.klines['15m']
        if len(df) < 27 or 'macd' not in df:
            continue
        prev, cur = df['macd'].iat[-2], df['macd'].iat[-1]
        osc = abs(cur)
        if prev > 0 and cur < prev and osc >= 11 and macd_cycle != 'DOWN':
            macd_cycle = 'DOWN'
            await order_mgr.place('SELL', 'MARKET', qty=0.017)
        if prev < 0 and cur > prev and osc >= 11 and macd_cycle != 'UP':
            macd_cycle = 'UP'
            await order_mgr.place('BUY', 'MARKET', qty=0.017)

# —— 子策略：RVGI ——
rvgi_cycle = None
async def rvgi_strategy():
    global rvgi_cycle
    while True:
        await asyncio.sleep(10)
        df = data_mgr.klines['15m']
        if len(df) < 11 or 'rvgi' not in df:
            continue
        rv, sg = df['rvgi'].iat[-1], df['rvsig'].iat[-1]
        if rv > sg and rvgi_cycle != 'UP':
            rvgi_cycle = 'UP'
            await order_mgr.place('BUY', 'MARKET', qty=0.016)
            await order_mgr.place('SELL', 'LIMIT', qty=0.016, price=latest_price*1.06, reduceOnly=True)
            await order_mgr.place('SELL', 'STOP_MARKET', stopPrice=latest_price*0.98)
        if rv < sg and rvgi_cycle != 'DOWN':
            rvgi_cycle = 'DOWN'
            await order_mgr.place('SELL', 'MARKET', qty=0.016)
            await order_mgr.place('BUY', 'LIMIT', qty=0.016, price=latest_price*0.94, reduceOnly=True)
            await order_mgr.place('BUY', 'TAKE_PROFIT_MARKET', stopPrice=latest_price*1.02)

# —— 子策略：三重 SuperTrend ——
triple_cycle = None
def supertrend(df, period, mult):
    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    up, dn = hl2 + mult*atr, hl2 - mult*atr
    st = pd.Series(index=df.index); dirc = pd.Series(True, index=df.index)
    for i in range(len(df)):
        if i == 0:
            st.iat[0] = up.iat[0]
        else:
            prev = st.iat[i-1]; price = df['close'].iat[i]
            if price > prev:
                st.iat[i] = max(dn.iat[i], prev); dirc.iat[i] = True
            else:
                st.iat[i] = min(up.iat[i], prev); dirc.iat[i] = False
    return st, dirc

async def triple_st_strategy():
    global triple_cycle
    while True:
        await asyncio.sleep(30)
        df = data_mgr.klines['15m']
        if len(df) < 12:
            continue
        s1, d1 = supertrend(df, 10, 1)
        s2, d2 = supertrend(df, 11, 2)
        s3, d3 = supertrend(df, 12, 3)
        up = d1.iat[-1] and d2.iat[-1] and d3.iat[-1]
        dn = not (d1.iat[-1] or d2.iat[-1] or d3.iat[-1])
        if up and triple_cycle != 'UP':
            triple_cycle = 'UP'
            await order_mgr.place('BUY', 'MARKET', qty=0.015)
        if dn and triple_cycle != 'DOWN':
            triple_cycle = 'DOWN'
            await order_mgr.place('SELL', 'MARKET', qty=0.015)
        if triple_cycle == 'UP' and not up:
            await order_mgr.place('SELL', 'MARKET', qty=0.015)
        if triple_cycle == 'DOWN' and not dn:
            await order_mgr.place('BUY', 'MARKET', qty=0.015)

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