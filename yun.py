#!/usr/bin/env python3
# coding: utf-8

import os, time, json, hmac, hashlib, asyncio, logging, uuid, urllib.parse, base64
from collections import defaultdict

import uvloop, aiohttp, pandas as pd, websockets, watchfiles
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
    SYNC_INTERVAL    = 300   # seconds
    MAX_POSITION     = 2.0   # per-strategy max

# —— 全局状态 ——
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
    time_offset = js['serverTime'] - int(time.time()*1000)
    logging.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
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
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
        async with self.lock:
            for p in res:
                if p['symbol']==Config.SYMBOL:
                    amt = abs(float(p['positionAmt']))
                    if p.get('positionSide','BOTH')=='LONG':
                        self.long = amt
                    else:
                        self.short = amt

    async def get_available(self, side:str) -> float:
        async with self.lock:
            if side=='BUY':
                return max(0.0, Config.MAX_POSITION - (self.long - self.short))
            return max(0.0, Config.MAX_POSITION - (self.short - self.long))

pos_tracker = PositionTracker()

# —— 令牌桶限流 ——
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

batch_rate = RateLimiter(rate=1, per=1)

# —— 批量下单管理 ——
class BatchOrderManager:
    def __init__(self):
        self.batch = []
        self.lock = asyncio.Lock()
    async def add(self, order):
        async with self.lock:
            self.batch.append(order)
            await self.flush()
    async def flush(self):
        async with self.lock:
            if not self.batch: return
            await batch_rate.acquire()
            payload = {'batchOrders': json.dumps(self.batch)}
            try:
                r = await session.post(
                    f"{Config.REST_BASE}/fapi/v1/batchOrders",
                    headers={'X-MBX-APIKEY':Config.API_KEY},
                    data=payload
                )
                r.raise_for_status()
                logging.info("Batch flush success: %s", self.batch)
                self.batch.clear()
            except Exception as e:
                logging.error("Batch flush failed: %s batch=%s", e, self.batch)

batch_mgr = BatchOrderManager()

# —— 基础下单管理 ——
class OrderManager:
    def _sign(self, params):
        qs = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

    def _build(self, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        ts = int(time.time()*1000 + time_offset)
        p = {
            "symbol": Config.SYMBOL,
            "side": side,
            "type": otype,
            "recvWindow": Config.RECV_WINDOW,
            "timestamp": ts
        }
        if is_hedge_mode and otype in ("LIMIT","MARKET"):
            p["positionSide"] = "LONG" if side=="BUY" else "SHORT"
        if otype=="LIMIT":
            p.update({"timeInForce":"GTC",
                      "quantity":f"{qty:.6f}","price":f"{price:.2f}"})
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            p.update({"closePosition":"true","stopPrice":f"{stop:.2f}"})
        else:
            p["quantity"] = f"{qty:.6f}"
        if reduceOnly:
            p["reduceOnly"] = "true"
        p["signature"] = self._sign(p)
        return p

    async def place(self, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        avail = await pos_tracker.get_available(side)
        if qty and qty > avail:
            logging.warning("Insufficient free position: need %.6f, avail %.6f", qty, avail)
            return
        o = self._build(side, otype, qty, price, stop, reduceOnly)
        logging.info("Enqueue order: %s", o)
        await batch_mgr.add(o)

order_mgr = OrderManager()

# —— 订单防护 ——
class OrderGuard:
    def __init__(self):
        self.states = defaultdict(lambda: {'last_dir':None,'last_ts':0,'waiting_cross':False})
        self.pos    = defaultdict(lambda:{'long':0.0,'short':0.0})
        self.lock   = asyncio.Lock()

    def _cooldown(self, strat):
        return {"main":60,"macd":30,"rvgi":10,"triple":300}.get(strat,0)

    async def check(self, strat, direction):
        async with self.lock:
            s = self.states[strat]
            now = time.time()
            if now - s['last_ts'] < self._cooldown(strat):
                return False
            if strat=="main" and direction==s['last_dir']:
                return False
            if strat=="macd" and s['waiting_cross']:
                return False
            if strat=="triple":
                st = await self.get_triple_status()
                if direction=="BUY" and not st['up']: return False
                if direction=="SELL" and not st['down']: return False
            return True

    async def limit(self, strat, direction, qty):
        limits={"main":2.0,"macd":0.2,"rvgi":0.2,"triple":0.3}
        async with self.lock:
            p = self.pos[strat]
            if direction=="BUY":
                return p['long']+qty<=limits[strat]
            return p['short']+qty<=limits[strat]

    async def update(self, strat, direction, qty):
        async with self.lock:
            p = self.pos[strat]
            if direction=="BUY": p['long']+=qty
            else:                p['short']+=qty

    async def stamp(self, strat, direction):
        async with self.lock:
            self.states[strat].update({
                'last_ts':time.time(),
                'last_dir':direction,
                'waiting_cross': strat in ("macd",)
            })

    @jit(nopython=True)
    def _st_dirs(self, high, low, close):
        n=len(close)
        d1=[True]*n; d2=[True]*n; d3=[True]*n
        # reuse numba_supertrend logic externally
        return d1, d2, d3  # placeholder

    async def get_triple_status(self):
        df = data_mgr.klines["15m"]
        if len(df)<12: return {'up':False,'down':False}
        h,l,c = df["high"].values, df["low"].values, df["close"].values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not (d1[-1] or d2[-1] or d3[-1])
        return {'up':up,'down':dn}

order_guard = OrderGuard()

class EnhancedOrderManager(OrderManager):
    def __init__(self, guard: OrderGuard):
        super().__init__(); self.guard = guard

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        if not await self.guard.check(strat, side): return
        if qty and not await self.guard.limit(strat, side, qty): return
        if strat=="rvgi" and not await self._rvgi_ma(side): return
        await super().place(side, otype, qty, price, stop, reduceOnly)
        await self.guard.update(strat, side, qty or 0)
        await self.guard.stamp(strat, side)

    async def _rvgi_ma(self, side):
        df = data_mgr.klines["15m"]
        if len(df)<100: return False
        ma7  = df["close"].rolling(7).mean().iat[-1]
        ma25 = df["close"].rolling(25).mean().iat[-1]
        ma99 = df["close"].rolling(99).mean().iat[-1]
        if side=="SELL": return latest_price>ma7>ma25>ma99
        else:            return latest_price<ma7<ma25<ma99

mgr = EnhancedOrderManager(order_guard)

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.klines  = {tf:pd.DataFrame(columns=["open","high","low","close"])
                        for tf in ("3m","15m","1h")}
        self.last_ts = {tf:0 for tf in ("3m","15m","1h")}
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.realtime= {'price':None,'ts':0,'lock':asyncio.Lock()}

    async def update_kline(self, tf, rec):
        async with self.lock:
            df = self.klines[tf]
            row = {"open":rec["o"],"high":rec["h"],"low":rec["l"],"close":rec["c"]}
            if df.empty or rec["t"]>self.last_ts[tf]:
                df.loc[len(df)] = row
            else:
                df.iloc[-1] = list(row.values())
            self.last_ts[tf]=rec["t"]
            self._update(tf)
            self._evt.set()

    def _update(self, tf):
        df=self.klines[tf]
        if len(df)<20: return
        m=df["close"].rolling(20).mean(); s=df["close"].rolling(20).std()
        df["bb_up"]=m+2*s; df["bb_dn"]=m-2*s
        df["bb_pct"]=(df["close"]-df["bb_dn"])/(df["bb_up"]-df["bb_dn"])
        if tf=="15m":
            hl2=(df["high"]+df["low"])/2
            atr=df["high"].rolling(10).max()-df["low"].rolling(10).min()
            df["st"]=hl2-3*atr
            df["macd"]=MACD(df["close"],12,26,9).macd_diff()
            df["rvgi"]=ROCIndicator(close=(df["close"]-df["open"]),window=10).roc()
            df["rvsig"]=df["rvgi"].rolling(4).mean()

    async def get(self, tf, col):
        async with self.lock:
            df=self.klines[tf]
            if col in df.columns and not df.empty:
                return df[col].iat[-1]
        return None

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

    async def track_price(self, price, ts):
        async with self.realtime['lock']:
            self.realtime['price']=price
            self.realtime['ts']=ts

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(high, low, close, period, mult):
    n=len(close)
    st=[0.0]*n; dirc=[True]*n
    hl2=[(high[i]+low[i])/2 for i in range(n)]
    atr=[max(high[i-period+1:i+1]) - min(low[i-period+1:i+1]) for i in range(n)]
    up=[hl2[i]+mult*atr[i] for i in range(n)]
    dn=[hl2[i]-mult*atr[i] for i in range(n)]
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
                                          ping_interval=15, ping_timeout=10) as ws:
                logging.info("Market WS connected")
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        latest_price, price_ts = float(d["p"]), time.time()
                        await data_mgr.track_price(latest_price, price_ts)
                    if "kline" in s:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        rec={"t":k["t"],"o":float(k["o"]),
                             "h":float(k["h"]),"l":float(k["l"]),
                             "c":float(k["c"])}
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            delay=min(2**retry,30)
            logging.error("Market WS err %s, retry %ds", e, delay)
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
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
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

# —— 事件驱动策略引擎 ——
class StrategyEngine:
    def __init__(self):
        self.strategies = []
    def register(self, strat):
        self.strategies.append(strat)
    async def run(self):
        while True:
            await data_mgr.wait_update()
            price = await self._price()
            for strat in self.strategies:
                await strat.check_signal(price)
    async def _price(self):
        async with data_mgr.realtime['lock']:
            return data_mgr.realtime['price']

engine = StrategyEngine()

# —— 主策略 ——
class MainStrategy:
    def __init__(self):
        self.last = 0
        self.dyn = DynamicParameters()
    async def check_signal(self, price):
        if time.time()-self.last<1: return
        bb3 = await data_mgr.get("3m","bb_pct")
        st  = await data_mgr.get("15m","st")
        if bb3 is None or st is None: return
        if bb3<=0 or bb3>=1:
            await self.process(price, st, bb3)
            self.last=time.time()
    async def process(self, price, st, bb3):
        await self.dyn.update_parameters()
        side = "BUY" if price>st else "SELL"
        strength = abs(bb3-0.5)*2
        base_qty=0.1*strength
        qty = min(base_qty*self.dyn.volatility, Config.MAX_POSITION*0.3)
        for lvl in self.dyn.get_order_levels():
            pr=price*(1+lvl if side=="BUY" else 1-lvl)
            await mgr.safe_place("main", side, "LIMIT", qty, pr)
        for tp in self.dyn.get_tp_levels():
            pr=price*(1+tp if side=="BUY" else 1-tp)
            rev="SELL" if side=="BUY" else "BUY"
            await mgr.safe_place("main", rev, "LIMIT", qty*0.2, pr, reduceOnly=True)
        slp=price*(0.98 if side=="BUY" else 1.02)*self.dyn.volatility
        ot="STOP_MARKET" if side=="BUY" else "TAKE_PROFIT_MARKET"
        await mgr.safe_place("main", rev, ot, stop=slp)

# —— MACD 策略 ——
class MACDStrategy:
    def __init__(self): self.cycle=None
    async def check_signal(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        last3=df["macd"].iat[-3:]
        # 三连跌且穿零轴
        if (last3[0]>last3[1]>last3[2]<0):
            await self.enter("SELL")
        if last3[0]<last3[1]<last3[2]>0:
            await self.enter("BUY")
    async def enter(self, side):
        steps=[0.3,0.5,0.2]
        for s in steps:
            await mgr.safe_place("macd", side, "MARKET", 0.15*s)
            await asyncio.sleep(0.5)

# —— RVGI 策略 ——
class RVGIStrategy:
    def __init__(self): self.cycle=None
    async def check_signal(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<11 or not {"rvgi","rvsig"}.issubset(df.columns): return
        rv,sg=df["rvgi"].iat[-1],df["rvsig"].iat[-1]
        ma7=df["close"].rolling(7).mean().iat[-1]
        ma25=df["close"].rolling(25).mean().iat[-1]
        ma99=df["close"].rolling(99).mean().iat[-1]
        if rv>sg and price>ma7>ma25>ma99:
            await mgr.safe_place("rvgi","BUY","MARKET",0.05)
        if rv<sg and price<ma7<ma25<ma99:
            await mgr.safe_place("rvgi","SELL","MARKET",0.05)

# —— 三重 SuperTrend 策略 ——
class TripleTrendStrategy:
    def __init__(self): self.state=None
    async def check_signal(self, price):
        await order_guard.check("triple","")  # 触发平仓
        df=data_mgr.klines["15m"]
        if len(df)<12: return
        h,l,c=df["high"].values,df["low"].values,df["close"].values
        _,d1=numba_supertrend(h,l,c,10,1)
        _,d2=numba_supertrend(h,l,c,11,2)
        _,d3=numba_supertrend(h,l,c,12,3)
        up=d1[-1] and d2[-1] and d3[-1]
        dn=not(up or d2[-1] or d3[-1])
        if up and self.state!="UP":
            self.state="UP"; await mgr.safe_place("triple","BUY","MARKET",0.15)
        if dn and self.state!="DOWN":
            self.state="DOWN"; await mgr.safe_place("triple","SELL","MARKET",0.15)
        if self.state=="UP" and not up:
            await mgr.safe_place("triple","SELL","MARKET",0.15)
        if self.state=="DOWN" and not dn:
            await mgr.safe_place("triple","BUY","MARKET",0.15)

# —— 动态参数 ——
class DynamicParameters:
    def __init__(self):
        self.base_levels=[0.0025,0.0040,0.0060,0.0080,0.0160]
        self.volatility=1.0; self.last=0
    async def update_parameters(self):
        if time.time()-self.last<60: return
        df=data_mgr.klines["15m"]
        if len(df)<14: return
        atr=(df["high"].rolling(14).max()-df["low"].rolling(14).min()).iat[-1]
        self.volatility=max(min(atr/latest_price if latest_price else 1,2.0),0.5)
        self.last=time.time()
    def get_order_levels(self):
        return [l*self.volatility for l in self.base_levels]
    def get_tp_levels(self):
        return [0.0102*self.volatility,
                0.0123*self.volatility,
                0.0150*self.volatility*1.2,
                0.0180*self.volatility*1.5,
                0.0220*self.volatility*2]

# —— 注册策略 & 启动 ——
engine.register(MainStrategy())
engine.register(MACDStrategy())
engine.register(RVGIStrategy())
engine.register(TripleTrendStrategy())

async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time(); await detect_mode()
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