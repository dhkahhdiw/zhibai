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
from collections import defaultdict
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

import uvloop
import aiohttp
import pandas as pd
import numpy as np
import websockets
import watchfiles
from ta.trend import MACD
from ta.momentum import ROCIndicator
from numba import jit

# —— 高性能事件循环 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# —— 环境变量 ——
load_dotenv()
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
    SYNC_INTERVAL    = 300
    MAX_POSITION     = 2.0

# —— 日志配置 ——
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)
class LineCountRotatingFileHandler(logging.Handler):
    def __init__(self, filename, max_lines=1000, encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.max_lines = max_lines
        open(self.filename, 'a', encoding=encoding).close()
    def emit(self, record):
        msg = self.format(record)
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) > self.max_lines:
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.writelines(lines[-self.max_lines:])
fh = LineCountRotatingFileHandler('/root/zhibai/yun.log')
fh.setFormatter(fmt)
logger.addHandler(fh)

# —— 全局状态 ——
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
logging.info("Loaded Ed25519 private key")

# —— DataManager ——
class DataManager:
    def __init__(self):
        self.klines   = {tf: pd.DataFrame(columns=["open","high","low","close"])
                         for tf in ("3m","15m","1h")}
        self.last_ts  = {tf:0 for tf in ("3m","15m","1h")}
        self.lock     = asyncio.Lock()
        self._evt     = asyncio.Event()
        self.realtime = {'price':None,'ts':0,'lock':asyncio.Lock()}

    async def update_kline(self, tf, rec):
        async with self.lock:
            df = self.klines[tf]
            row = {"open":rec["o"],"high":rec["h"],"low":rec["l"],"close":rec["c"]}
            if df.empty or rec["t"] > self.last_ts[tf]:
                df.loc[len(df)] = row
            else:
                df.iloc[-1] = list(row.values())
            self.last_ts[tf] = rec["t"]
            self._compute(tf)
            logging.debug(f"Kline updated {tf} at {rec['t']}")
            self._evt.set()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df) < 20: return
        m = df["close"].rolling(20).mean()
        s = df["close"].rolling(20).std()
        df["bb_up"]  = m + 2*s
        df["bb_dn"]  = m - 2*s
        df["bb_pct"] = (df["close"] - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"])
        if tf == "15m":
            hl2 = (df["high"] + df["low"]) / 2
            atr = df["high"].rolling(10).max() - df["low"].rolling(10).min()
            df["st"]   = hl2 - 3*atr
            df["macd"] = MACD(df["close"],12,26,9).macd_diff()
            df["rvgi"] = ROCIndicator(close=(df["close"]-df["open"]), window=10).roc()
            df["rvsig"]= df["rvgi"].rolling(4).mean()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df.columns and not df.empty:
                return df[col].iat[-1]
        return None

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

    async def track_price(self, price, ts):
        async with self.realtime['lock']:
            self.realtime['price'], self.realtime['ts'] = price, ts
        logging.debug(f"Price update: {price} at {ts}")
        self._evt.set()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(high, low, close, period, mult):
    n = len(close)
    st = [0.0]*n; dirc=[True]*n
    hl2 = [(high[i]+low[i])/2 for i in range(n)]
    atr = [max(high[i-period+1:i+1]) - min(low[i-period+1:i+1]) for i in range(n)]
    up  = [hl2[i] + mult*atr[i] for i in range(n)]
    dn  = [hl2[i] - mult*atr[i] for i in range(n)]
    st[0] = up[0]
    for i in range(1,n):
        if close[i] > st[i-1]:
            st[i], dirc[i] = max(dn[i],st[i-1]), True
        else:
            st[i], dirc[i] = min(up[i],st[i-1]), False
    return st, dirc

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    js = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = js['serverTime'] - int(time.time()*1000)
    logging.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge_mode = res.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# —— 持仓追踪 ——
class PositionTracker:
    def __init__(self):
        self.long = self.short = 0.0
        self.lock = asyncio.Lock()
    async def sync(self):
        ts= int(time.time()*1000 + time_offset)
        qs= urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
        sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url=f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        try:
            res= await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
            if 'code' in res:
                logging.error("Position sync failed: %s", res)
                return
            async with self.lock:
                for p in res:
                    if p['symbol']==Config.SYMBOL:
                        amt=abs(float(p['positionAmt']))
                        if p.get('positionSide','BOTH')=='LONG':
                            self.long=amt
                        else:
                            self.short=amt
        except Exception as e:
            logging.error("Position sync error: %s", e)
            await asyncio.sleep(5)
    async def get_available(self, side):
        async with self.lock:
            if side=='BUY':
                return max(0.0, Config.MAX_POSITION - (self.long - self.short))
            else:
                return max(0.0, Config.MAX_POSITION - (self.short - self.long))

pos_tracker = PositionTracker()

# —— 限流 & 批量下单 ——
class RateLimiter:
    def __init__(self, rate, per):
        self._tokens=rate; self._rate=rate; self._per=per; self._last=time.time()
    async def acquire(self):
        now=time.time()
        self._tokens=min(self._rate,
            self._tokens+(now-self._last)*(self._rate/self._per))
        self._last=now
        if self._tokens<1:
            await asyncio.sleep((1-self._tokens)*(self._per/self._rate))
        self._tokens-=1

class BatchOrderManager:
    def __init__(self):
        self.batch=[]; self.lock=asyncio.Lock(); self.rl=RateLimiter(1,1)
    async def add(self, order):
        async with self.lock:
            self.batch.append(order)
            await self.flush()
    async def flush(self):
        async with self.lock:
            if not self.batch: return
            await self.rl.acquire()
            for i in range(3):
                try:
                    r=await session.post(
                        f"{Config.REST_BASE}/fapi/v1/batchOrders",
                        headers={'X-MBX-APIKEY':Config.API_KEY},
                        data={'batchOrders':json.dumps(self.batch)}
                    )
                    r.raise_for_status()
                    logging.info("Batch flush success (%d orders)", len(self.batch))
                    self.batch.clear()
                    return
                except aiohttp.ClientResponseError as e:
                    if e.status==429:
                        await asyncio.sleep(2**i)
                        continue
                    logging.error("Batch flush error: %s", e)
                    break
            logging.error("Batch final failure (%d orders)", len(self.batch))

batch_mgr = BatchOrderManager()

# —— 下单管理 & 防护 ——
class OrderManager:
    def _sign(self, params):
        qs = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

    def _build(self, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        ts = int(time.time()*1000 + time_offset)
        params = {
            "symbol": Config.SYMBOL,
            "side": side,
            "type": otype,
            "recvWindow": Config.RECV_WINDOW,
            "timestamp": ts,
        }
        if is_hedge_mode and otype in ("LIMIT", "MARKET"):
            params["positionSide"] = "LONG" if side=="BUY" else "SHORT"
        if otype == "LIMIT":
            params.update({
                "timeInForce": "GTC",
                "quantity": f"{qty:.6f}",
                "price":    f"{price:.2f}",
            })
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            params.update({
                "stopPrice":    f"{stop:.2f}",
                "closePosition":"true",
            })
        else:
            params["quantity"] = f"{qty:.6f}"
        if reduceOnly:
            params["reduceOnly"] = "true"
        params["signature"] = self._sign(params)
        return params

    async def place(self, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        avail = await pos_tracker.get_available(side)
        if qty and qty > avail:
            logging.warning("Insufficient pos: need %.6f, avail %.6f", qty, avail)
            return
        order = self._build(side, otype, qty, price, stop, reduceOnly)
        logging.info("Enqueue order: %s", order)
        await batch_mgr.add(order)

class OrderGuard:
    def __init__(self):
        self.states=defaultdict(lambda:{'last_ts':0,'last_dir':None,'waiting_cross':False})
        self.pos   =defaultdict(lambda:{'long':0.0,'short':0.0})
        self.lock  =asyncio.Lock()

    def _cd(self, s):
        return {"main":60,"macd":30,"rvgi":10,"triple":300}.get(s,0)

    async def check(self, s, side):
        async with self.lock:
            st = self.states[s]; now = time.time()
            if now - st['last_ts'] < self._cd(s): return False
            if s=="main" and side==st['last_dir']: return False
            if s=="macd" and st['waiting_cross']:  return False
            if s=="triple":
                ts = await self.get_triple_status()
                if side=="BUY" and not ts['up']:   return False
                if side=="SELL" and not ts['down']: return False
            return True

    async def limit(self, s, side, qty):
        lim={"main":0.6,"macd":0.2,"rvgi":0.2,"triple":0.3}[s]
        async with self.lock:
            p = self.pos[s]
            return (p['long']+qty<=lim) if side=="BUY" else (p['short']+qty<=lim)

    async def update(self, s, side, qty):
        async with self.lock:
            p = self.pos[s]
            if side=="BUY":  p['long'] += qty
            else:            p['short'] += qty

    async def stamp(self, s, side):
        async with self.lock:
            st = self.states[s]
            st.update({
                'last_ts': time.time(),
                'last_dir': side,
                'waiting_cross': (s=="macd")
            })

    async def get_triple_status(self):
        df = data_mgr.klines["15m"]
        if len(df) < 12: return {'up':False,'down':False}
        h,l,c = df["high"].values, df["low"].values, df["close"].values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not (d1[-1] or d2[-1] or d3[-1])
        return {'up':up,'down':dn}

class EnhancedOrderManager(OrderManager):
    def __init__(self, guard):
        super().__init__()
        self.guard = guard

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        logging.debug(f"{strat}→{side} {otype} qty={qty} price={price} stop={stop}")
        if not await self.guard.check(strat, side): return
        if qty and not await self.guard.limit(strat, side, qty): return
        if strat=="rvgi":
            df = data_mgr.klines["15m"]
            if len(df)>=100:
                ma7, ma25, ma99 = (
                    df["close"].rolling(7).mean().iat[-1],
                    df["close"].rolling(25).mean().iat[-1],
                    df["close"].rolling(99).mean().iat[-1],
                )
                ok = (latest_price>ma7>ma25>ma99) if side=="SELL" else (latest_price<ma7<ma25<ma99)
                if not ok: return
        await self.place(side, otype, qty, price, stop, reduceOnly)
        await self.guard.update(strat, side, qty or 0)
        await self.guard.stamp(strat, side)

order_guard = OrderGuard()
mgr         = EnhancedOrderManager(order_guard)

# —— 策略引擎 ——
class StrategyEngine:
    def __init__(self):
        self.strats=[]
    def register(self, s):
        self.strats.append(s)
    async def run(self):
        while True:
            await data_mgr.wait_update()
            async with data_mgr.realtime['lock']:
                price, ts = data_mgr.realtime['price'], data_mgr.realtime['ts']
            if price is None or time.time()-ts>60:
                continue
            for strat in self.strats:
                await strat.check_signal(price)

engine = StrategyEngine()

# —— 动态参数 ——
class DynamicParameters:
    def __init__(self):
        self.base=[0.0025,0.0040,0.0060,0.0080,0.0160]
        self.vol, self.last = 1.0,0
    async def update(self):
        if time.time()-self.last<60: return
        df=data_mgr.klines["15m"]
        if len(df)<14 or latest_price is None: return
        atr=(df["high"].rolling(14).max()-df["low"].rolling(14).min()).iat[-1]
        self.vol = max(min(atr/latest_price,2.0),0.5)
        self.last= time.time()
    def levels(self):
        return [l*self.vol for l in self.base]
    def tps(self):
        return [0.0102*self.vol,0.0123*self.vol,
                0.0150*self.vol*1.2,0.0180*self.vol*1.5,
                0.0220*self.vol*2.0]

# —— 各策略 ——
class MainStrategy:
    def __init__(self):
        self.last=0; self.dyn=DynamicParameters()
    async def check_signal(self, price):
        if time.time()-self.last<1: return
        bb3=await data_mgr.get("3m","bb_pct"); st=await data_mgr.get("15m","st")
        if bb3 is None or st is None: return
        if bb3<=0 or bb3>=1:
            logging.info("MainStrategy triggered")
            await self._run(price, st, bb3)
            self.last=time.time()

    async def _run(self, price, st, bb3):
        await self.dyn.update()
        side = "BUY" if price>st else "SELL"
        strength = abs(bb3-0.5)*2
        qty = min(0.1*strength*self.dyn.vol, Config.MAX_POSITION*0.3)

        # 开仓限价单
        for lvl in self.dyn.levels():
            pr = price*(1 + lvl if side=="BUY" else 1 - lvl)
            await mgr.safe_place("main", side, "LIMIT", qty, pr)

        # 多级止盈限价
        rev = "SELL" if side=="BUY" else "BUY"
        for tp in self.dyn.tps():
            pr = price*(1 + tp if side=="BUY" else 1 - tp)
            await mgr.safe_place("main", rev, "LIMIT", qty*0.2, pr, reduceOnly=True)

        # 止损/止盈市价全平
        slp = price*(0.98 if side=="BUY" else 1.02)*self.dyn.vol
        ot = "STOP_MARKET" if side=="BUY" else "TAKE_PROFIT_MARKET"
        await mgr.safe_place("main", rev, ot, stop=slp)

class MACDStrategy:
    def __init__(self): self._cd=0
    async def check_signal(self, price):
        if time.time()-self._cd<5: return
        df=data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        m = df["macd"]
        if m.iat[-2]<0< m.iat[-1]:
            await self._enter("BUY")
        elif m.iat[-2]>0> m.iat[-1]:
            await self._enter("SELL")
    async def _enter(self, side):
        for p in (0.3,0.5,0.2):
            await mgr.safe_place("macd", side, "MARKET", 0.15*p)
            await asyncio.sleep(0.5)
        self._cd = time.time()

class RVGIStrategy:
    def __init__(self): self._cd=0
    async def check_signal(self, price):
        if time.time()-self._cd<5: return
        df=data_mgr.klines["15m"]
        if len(df)<11 or not {"rvgi","rvsig"}.issubset(df.columns): return
        rv, sg = df["rvgi"].iat[-1], df["rvsig"].iat[-1]
        ma7, ma25, ma99 = (
            df["close"].rolling(7).mean().iat[-1],
            df["close"].rolling(25).mean().iat[-1],
            df["close"].rolling(99).mean().iat[-1]
        )
        if rv>sg and price>ma7>ma25>ma99:
            await mgr.safe_place("rvgi","BUY","MARKET",0.05)
            self._cd=time.time()
        elif rv<sg and price<ma7<ma25<ma99:
            await mgr.safe_place("rvgi","SELL","MARKET",0.05)
            self._cd=time.time()

class TripleTrendStrategy:
    def __init__(self): self.state=None
    async def check_signal(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<12: return
        h,l,c = df["high"].values, df["low"].values, df["close"].values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not (d1[-1] or d2[-1] or d3[-1])
        if up and self.state!="UP":
            self.state="UP"; await mgr.safe_place("triple","BUY","MARKET",0.15)
        if dn and self.state!="DOWN":
            self.state="DOWN"; await mgr.safe_place("triple","SELL","MARKET",0.15)
        if self.state=="UP" and not up:
            await mgr.safe_place("triple","SELL","MARKET",0.15)
        if self.state=="DOWN" and not dn:
            await mgr.safe_place("triple","BUY","MARKET",0.15)

engine.register(MainStrategy())
engine.register(MACDStrategy())
engine.register(RVGIStrategy())
engine.register(TripleTrendStrategy())

# —— WebSocket 市场流 ——
async def market_ws():
    global latest_price
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL, ping_interval=20, ping_timeout=60) as ws:
                retry = 0
                async for msg in ws:
                    o=json.loads(msg)
                    s, d = o["stream"], o["data"]
                    if s.endswith("@markPrice"):
                        latest_price = float(d["p"])
                        await data_mgr.track_price(latest_price, time.time())
                    elif "@kline_" in s:
                        tf = s.split("@")[1].split("_")[1]
                        k = d["k"]
                        rec = {"t":k["t"],"o":float(k["o"]),
                               "h":float(k["h"]),"l":float(k["l"]),
                               "c":float(k["c"])}
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            delay = min(2**retry, 30)
            logging.error("Market WS err %s, retry in %ds", e, delay)
            await asyncio.sleep(delay)
            retry += 1

# —— WebSocket 用户流 ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                params = {
                    "apiKey": Config.ED25519_API,
                    "timestamp": int(time.time()*1000 + time_offset)
                }
                payload = "&".join(f"{k}={v}" for k,v in sorted(params.items()))
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params["signature"] = sig
                await ws.send(json.dumps({
                    "id": str(uuid.uuid4()),
                    "method": "session.logon",
                    "params": params
                }))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({
                            "id": str(uuid.uuid4()),
                            "method": "session.status"
                        }))
                asyncio.create_task(hb())
                async for _ in ws: pass
        except Exception as e:
            logging.error("User WS err %s, retry in 5s", e)
            await asyncio.sleep(5)

# —— 配置热加载 & 定期维护 ——
async def config_watcher():
    async for changes in watchfiles.awatch('.'):
        for change, path in changes:
            if path.endswith('.py') or path.endswith('.env'):
                logging.info("Config changed, reloading…")
                load_dotenv()
                await pos_tracker.sync()

async def maintenance():
    asyncio.create_task(config_watcher())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await detect_mode()
        await pos_tracker.sync()
        await batch_mgr.flush()

# —— 主入口 ——
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
            engine.run()
        )
    finally:
        await session.close()

if __name__=='__main__':
    asyncio.run(main())