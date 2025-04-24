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

import uvloop
import aiohttp
import pandas as pd
import websockets
import watchfiles
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.trend import MACD
from ta.momentum import ROCIndicator
from numba import jit

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
    SYNC_INTERVAL    = 300   # 5分钟
    MAX_POSITION     = 2.0   # 每策略最大持仓量上限

# 全局状态
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— 时间同步 & 持仓模式检测 ——
async def sync_time():
    global time_offset
    js = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = js['serverTime'] - int(time.time() * 1000)
    logging.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge_mode = res.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# —— PositionTracker ——
class PositionTracker:
    def __init__(self):
        self.long = self.short = 0.0
        self.lock = asyncio.Lock()

    async def sync(self):
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts,'recvWindow':Config.RECV_WINDOW})
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
        async with self.lock:
            for pos in res:
                if pos['symbol']==Config.SYMBOL:
                    amt = abs(float(pos['positionAmt']))
                    if pos['positionSide']=='LONG':
                        self.long = amt
                    else:
                        self.short = amt

    async def get_available(self, side:str)->float:
        async with self.lock:
            if side=='BUY':
                return max(0.0, Config.MAX_POSITION - (self.long - self.short))
            return max(0.0, Config.MAX_POSITION - (self.short - self.long))

pos_tracker = PositionTracker()

# —— RateLimiter（令牌桶） ——
class RateLimiter:
    def __init__(self, rate, per):
        self._tokens = rate
        self._rate = rate
        self._per = per
        self._last = time.time()
    async def acquire(self):
        now = time.time()
        self._tokens = min(self._rate,
            self._tokens + (now-self._last)*(self._rate/self._per))
        self._last = now
        if self._tokens < 1:
            await asyncio.sleep((1-self._tokens)*(self._per/self._rate))
        self._tokens -= 1

batch_rate = RateLimiter(rate=1, per=1)  # 每秒最多 1 次批量下单

# —— BatchOrderManager ——
class BatchOrderManager:
    BATCH_SIZE = 1   # 一条订单即刻下单
    def __init__(self):
        self.batch = []
        self.lock = asyncio.Lock()

    async def add(self, order):
        async with self.lock:
            self.batch.append(order)
            await self.flush()

    async def flush(self):
        async with self.lock:
            if not self.batch:
                return
            await batch_rate.acquire()
            payload = {'batchOrders': json.dumps(self.batch)}
            try:
                resp = await session.post(
                    f"{Config.REST_BASE}/fapi/v1/batchOrders",
                    headers={'X-MBX-APIKEY':Config.API_KEY},
                    data=payload
                )
                resp.raise_for_status()
                logging.info("Batch flush success: %s", self.batch)
                self.batch.clear()
            except Exception as e:
                logging.error("Batch flush failed: %s, batch=%s", e, self.batch)
                # TODO: 持久化 self.batch 或 报警

batch_mgr = BatchOrderManager()

# —— OrderManager ——
class OrderManager:
    def __init__(self):
        pass

    def _sign(self, params: dict) -> str:
        qs = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

    def _build_order(self, side, otype, qty=None, price=None, stopPrice=None, reduceOnly=False):
        ts = int(time.time()*1000 + time_offset)
        base = {
            "symbol":Config.SYMBOL,
            "side":side, "type":otype,
            "recvWindow":Config.RECV_WINDOW,
            "timestamp":ts
        }
        if is_hedge_mode and otype in ("LIMIT","MARKET"):
            base["positionSide"] = "LONG" if side=="BUY" else "SHORT"
        if otype=="LIMIT":
            base.update({
                "timeInForce":"GTC",
                "quantity":f"{qty:.6f}",
                "price":   f"{price:.2f}"
            })
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            base.update({
                "closePosition":"true",
                "stopPrice":    f"{stopPrice:.2f}"
            })
        else:
            base["quantity"] = f"{qty:.6f}"
        if reduceOnly:
            base["reduceOnly"] = "true"
        base["signature"] = self._sign(base)
        return base

    async def place(self, side, otype, qty=None, price=None, stopPrice=None, reduceOnly=False):
        avail = await pos_tracker.get_available(side)
        if qty and qty>avail:
            logging.warning("仓位不足: 需 %.6f, 可用 %.6f", qty, avail)
            return
        order = self._build_order(side, otype, qty, price, stopPrice, reduceOnly)
        logging.info("Enqueue order: %s", order)
        await batch_mgr.add(order)

order_mgr = OrderManager()

# —— OrderGuard & EnhancedOrderManager ——
class OrderGuard:
    def __init__(self):
        self.strategy_states = defaultdict(dict)
        self.position_tracking = defaultdict(lambda: {'long':0.0,'short':0.0})
        self.lock = asyncio.Lock()

    async def check_strategy_state(self, strategy: str, direction: str) -> bool:
        async with self.lock:
            state = self.strategy_states.get(strategy, {})
            now = time.time()
            # 主策略：同方向冷却60s
            if strategy=="main":
                if direction==state.get('last_direction'): return False
                if now-state.get('last_ts',0)<60:              return False
            return True

    async def check_position_limit(self, strategy: str, direction: str, qty: float) -> bool:
        limits = {"main":1.0,"macd":0.3,"rvgi":0.3,"triple":0.3}
        async with self.lock:
            pos = self.position_tracking[strategy]
            if direction=="BUY":
                return pos['long']+qty<=limits[strategy]
            return pos['short']+qty<=limits[strategy]

    async def update_position(self, strategy: str, direction: str, qty: float):
        async with self.lock:
            if direction=="BUY":
                self.position_tracking[strategy]['long'] += qty
            else:
                self.position_tracking[strategy]['short'] += qty

    async def update_strategy_state(self, strategy: str, direction: str):
        async with self.lock:
            self.strategy_states[strategy] = {
                'last_ts': time.time(),
                'last_direction': direction
            }

class EnhancedOrderManager(OrderManager):
    def __init__(self, guard: OrderGuard):
        super().__init__()
        self.guard = guard

    async def safe_place(self, strategy: str, side: str, otype: str,
                         qty=None, price=None, stopPrice=None, reduceOnly=False):
        if not await self.guard.check_strategy_state(strategy, side):
            logging.warning("策略 %s: 状态冲突，中止下单", strategy)
            return
        if qty and not await self.guard.check_position_limit(strategy, side, qty):
            logging.warning("策略 %s: 仓位超限，中止下单", strategy)
            return
        await super().place(side, otype, qty, price, stopPrice, reduceOnly)
        await self.guard.update_position(strategy, side, qty or 0)
        await self.guard.update_strategy_state(strategy, side)

order_guard = OrderGuard()
mgr = EnhancedOrderManager(order_guard)

# —— DataManager ——
class DataManager:
    def __init__(self):
        self.klines = {
            tf: pd.DataFrame(columns=["open","high","low","close"])
            for tf in ("3m","15m","1h")
        }
        self.last_ts = {tf:0 for tf in ("3m","15m","1h")}
        self.lock = asyncio.Lock()

    async def update_kline(self, tf, rec):
        async with self.lock:
            df = self.klines[tf]
            new = [rec["o"], rec["h"], rec["l"], rec["c"]]
            if df.empty or rec["t"]>self.last_ts[tf]:
                # 新行直接 loc
                df.loc[len(df)] = new
            else:
                df.iloc[-1] = new
            self.last_ts[tf] = rec["t"]
            self._update(tf)

    def _update(self, tf):
        df = self.klines[tf]
        if len(df)<20: return
        m = df["close"].rolling(20).mean()
        s = df["close"].rolling(20).std()
        df["bb_up"]  = m + 2*s
        df["bb_dn"]  = m - 2*s
        df["bb_pct"] = (df["close"] - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"])
        if tf=="15m":
            hl2 = (df["high"] + df["low"])/2
            atr = df["high"].rolling(10).max() - df["low"].rolling(10).min()
            df["st"]   = hl2 - 3*atr
            df["macd"] = MACD(df["close"],12,26,9).macd_diff()
            df["rvgi"] = ROCIndicator(close=df["close"]-df["open"], window=10).roc()
            df["rvsig"]= df["rvgi"].rolling(4).mean()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df.columns and not df.empty:
                return df[col].iat[-1]
        return None

data_mgr = DataManager()

# —— Numba SuperTrend ——
@jit(nopython=True)
def numba_supertrend(high, low, close, period, mult):
    n=len(close)
    st=[0.0]*n; dirc=[True]*n
    hl2=[(high[i]+low[i])/2 for i in range(n)]
    atr=[max(high[i-period+1:i+1]) - min(low[i-period+1:i+1]) for i in range(n)]
    up=[hl2[i] + mult*atr[i] for i in range(n)]
    dn=[hl2[i] - mult*atr[i] for i in range(n)]
    st[0]=up[0]
    for i in range(1,n):
        if close[i]>st[i-1]:
            st[i]=max(dn[i],st[i-1]); dirc[i]=True
        else:
            st[i]=min(up[i],st[i-1]); dirc[i]=False
    return st, dirc

# —— WebSocket: 市场数据 ——
async def market_ws():
    global latest_price, price_ts
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL,
                                          ping_interval=15, ping_timeout=10, close_timeout=5) as ws:
                logging.info("Market WS connected")
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        latest_price, price_ts = float(d["p"]), time.time()
                    if "kline" in s:
                        tf=s.split("@")[1].split("_")[1]
                        k=d["k"]
                        rec={"t":k["t"],"o":float(k["o"]),
                             "h":float(k["h"]),"l":float(k["l"]),
                             "c":float(k["c"])}
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            delay=min(2**retry,30)
            logging.error("Market WS err %s, retry in %ds", e, delay)
            await asyncio.sleep(delay); retry+=1

# —— WebSocket: 用户流 ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params={"apiKey":Config.ED25519_API,
                        "timestamp":int(time.time()*1000+time_offset)}
                payload="&".join(f"{k}={v}" for k,v in sorted(params.items()))
                sig=base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params["signature"]=sig
                await ws.send(json.dumps({
                    "id":str(uuid.uuid4()),
                    "method":"session.logon",
                    "params":params
                }))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({
                            "id":str(uuid.uuid4()),
                            "method":"session.status"
                        }))
                asyncio.create_task(hb())
                async for _ in ws: pass
        except Exception as e:
            logging.error("User WS err %s, retry 5s", e)
            await asyncio.sleep(5)

# —— 配置热加载 & 定期维护 ——
async def config_watcher():
    async for _ in watchfiles.awatch('/root/zhibai/'):
        logging.info("Config changed, reloading…")
        load_dotenv('/root/zhibai/.env')
        await pos_tracker.sync()

async def maintenance():
    asyncio.create_task(config_watcher())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode()
        await pos_tracker.sync(); await batch_mgr.flush()

# —— 主策略 ——
async def main_strategy():
    tp_offs=[0.0102,0.0123,0.0150,0.0180,0.0220]
    while price_ts is None: await asyncio.sleep(0.1)
    while True:
        await asyncio.sleep(0.5)
        p   = latest_price
        bb1 = await data_mgr.get("1h","bb_pct")
        bb3 = await data_mgr.get("3m","bb_pct")
        st  = await data_mgr.get("15m","st")
        if None in (bb1,bb3,st): continue
        if bb3<=0 or bb3>=1:
            side="BUY" if p>st else "SELL"
            # 主策略方向登记（不立即下市价单）
            await mgr.safe_place("main", side, "MARKET", 0.000001)  # 触发方向，仅记录
            strong=(p>st and bb1<0.2) or (p<st and bb1>0.8)
            levels=dyn_params.update(st)
            qty=0.12 if (p>st and strong) else 0.03
            if p<st: qty=0.07 if strong else 0.015
            rev="SELL" if side=="BUY" else "BUY"
            # 分级限价挂单
            for off in levels:
                price_off=p*(1+off if side=="BUY" else 1-off)
                await mgr.safe_place("main", side, "LIMIT", qty, price_off)
            # 止盈限价挂单
            for off in tp_offs:
                pt=p*(1+off if rev=="BUY" else 1-off)
                await mgr.safe_place("main", rev, "LIMIT", qty*0.2, pt, reduceOnly=True)
            # 止损/止盈市价单
            slp=p*(0.98 if side=="BUY" else 1.02)
            ttype="STOP_MARKET" if side=="BUY" else "TAKE_PROFIT_MARKET"
            await mgr.safe_place("main", rev, ttype, None, None, slp)

# —— 子策略：MACD ——
macd_cycle=None
async def macd_strategy():
    global macd_cycle
    while True:
        await asyncio.sleep(15)
        df=data_mgr.klines["15m"]
        if len(df)<27 or "macd" not in df: continue
        prev,cur=df["macd"].iat[-2],df["macd"].iat[-1]; osc=abs(cur)
        if prev>0 and cur<prev and osc>=11 and macd_cycle!="DOWN":
            macd_cycle="DOWN"
            await mgr.safe_place("macd","SELL","MARKET",0.017)
        if prev<0 and cur>prev and osc>=11 and macd_cycle!="UP":
            macd_cycle="UP"
            await mgr.safe_place("macd","BUY","MARKET",0.017)

# —— 子策略：RVGI ——
rvgi_cycle=None
async def rvgi_strategy():
    global rvgi_cycle
    while True:
        await asyncio.sleep(10)
        df=data_mgr.klines["15m"]
        if len(df)<11 or not {"rvgi","rvsig"}.issubset(df.columns): continue
        rv,sg=df["rvgi"].iat[-1],df["rvsig"].iat[-1]
        if rv>sg and rvgi_cycle!="UP":
            rvgi_cycle="UP"
            await mgr.safe_place("rvgi","BUY","MARKET",0.016)
            await mgr.safe_place("rvgi","SELL","LIMIT",0.016,latest_price*1.06)
            await mgr.safe_place("rvgi","STOP_MARKET",None,None,latest_price*0.98)
        if rv<sg and rvgi_cycle!="DOWN":
            rvgi_cycle="DOWN"
            await mgr.safe_place("rvgi","SELL","MARKET",0.016)
            await mgr.safe_place("rvgi","BUY","LIMIT",0.016,latest_price*0.94)
            await mgr.safe_place("rvgi","TAKE_PROFIT_MARKET",None,None,latest_price*1.02)

# —— 子策略：三重 SuperTrend ——
triple_cycle=None
async def triple_st_strategy():
    global triple_cycle
    while True:
        await asyncio.sleep(30)
        df=data_mgr.klines["15m"]
        if len(df)<12: continue
        high,low,close=df["high"].values,df["low"].values,df["close"].values
        s1,d1=numba_supertrend(high,low,close,10,1)
        s2,d2=numba_supertrend(high,low,close,11,2)
        s3,d3=numba_supertrend(high,low,close,12,3)
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not(up or d2[-1] or d3[-1])
        if up and triple_cycle!="UP":
            triple_cycle="UP";   await mgr.safe_place("triple","BUY","MARKET",0.015)
        if dn and triple_cycle!="DOWN":
            triple_cycle="DOWN"; await mgr.safe_place("triple","SELL","MARKET",0.015)
        if triple_cycle=="UP"   and not up:
            await mgr.safe_place("triple","SELL","MARKET",0.015)
        if triple_cycle=="DOWN" and not dn:
            await mgr.safe_place("triple","BUY","MARKET",0.015)

# —— 启动 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time(); await detect_mode()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            maintenance(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy()
        )
    finally:
        await session.close()

if __name__=='__main__':
    asyncio.run(main())