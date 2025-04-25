#!/usr/bin/env python3
# coding: utf-8

import os, time, json, uuid, base64, hmac, hashlib, asyncio, logging, urllib.parse
from collections import defaultdict
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

import uvloop, aiohttp, websockets, pandas as pd, numpy as np
from ta.trend import MACD
from ta.momentum import ROCIndicator
from numba import jit
import watchfiles

# —— 高性能事件循环 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# —— 环境 & 配置 ——
load_dotenv()
class Config:
    SYMBOL, PAIR = 'ETHUSDC', 'ethusdc'
    API_KEY = os.getenv('BINANCE_API_KEY')
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET = f"wss://fstream.binance.com/stream?streams={PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    WS_USER   = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST      = 'https://fapi.binance.com'
    RECV_W    = 5000
    SYNC_INT  = 300
    MAX_POS   = 2.0

# —— 日志 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
h = logging.StreamHandler(); h.setFormatter(fmt); LOG.addHandler(h)
class SimpleRotator(logging.Handler):
    def __init__(self, path, max_lines=1000):
        super().__init__(); self.path, self.max = path, max_lines
        open(path, 'a').close()
    def emit(self, r):
        msg = self.format(r)
        with open(self.path,'a') as f: f.write(msg+'\n')
        lines = open(self.path).read().splitlines()
        if len(lines)>self.max:
            open(self.path,'w').write('\n'.join(lines[-self.max:])+'\n')
fh = SimpleRotator('bot.log'); fh.setFormatter(fmt); LOG.addHandler(fh)

# —— 全局状态 ——
session: aiohttp.ClientSession
time_offset = 0
latest_price = None
is_hedge = False

# —— Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Loaded Ed25519 key")

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.klines = {tf: pd.DataFrame(columns="open high low close".split()) for tf in ("3m","15m","1h")}
        self.last_ts= dict.fromkeys(self.klines, 0)
        self.lock = asyncio.Lock()
        self.evt  = asyncio.Event()
        self.price = None; self.ptime = 0

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            row = {"open":o,"high":h,"low":l,"close":c}
            if df.empty or ts>self.last_ts[tf]:
                df.loc[len(df)] = row
            else:
                df.iloc[-1] = list(row.values())
            self.last_ts[tf] = ts
            self._compute(tf)
            LOG.debug(f"{tf}@{ts} updated")
            self.evt.set()

    def _compute(self, tf):
        df=self.klines[tf]
        if len(df)<20: return
        m,s = df.close.rolling(20).mean(), df.close.rolling(20).std()
        df["bb_up"],df["bb_dn"]=m+2*s,m-2*s
        df["bb_pct"]=(df.close-df.bb_dn)/(df.bb_up-df.bb_dn)
        if tf=="15m":
            hl2=(df.high+df.low)/2
            atr=df.high.rolling(10).max()-df.low.rolling(10).min()
            df["st"]=hl2-3*atr
            df["macd"]=MACD(df.close,12,26,9).macd_diff()
            rv=ROCIndicator(close=df.close-df.open,window=10).roc()
            df["rvgi"],df["rvsig"]=rv,rv.rolling(4).mean()

    async def get(self, tf, col):
        async with self.lock:
            df=self.klines[tf]
            return df[col].iat[-1] if col in df and not df.empty else None

    async def wait(self):
        await self.evt.wait(); self.evt.clear()

    async def track_price(self, p, ts):
        async with self.lock:
            self.price,self.ptime = p,ts
        LOG.debug(f"Price {p}@{ts}")
        self.evt.set()

dm = DataManager()

@jit(nopython=True)
def supertrend(h,l,c,per,mul):
    n=len(c); st=[0]*n; d=[True]*n
    hl2=[(h[i]+l[i])/2 for i in range(n)]
    atr=[max(h[i-per+1:i+1])-min(l[i-per+1:i+1]) for i in range(n)]
    up=[hl2[i]+mul*atr[i] for i in range(n)]
    dn=[hl2[i]-mul*atr[i] for i in range(n)]
    st[0]=up[0]
    for i in range(1,n):
        if c[i]>st[i-1]:
            st[i],d[i] = max(dn[i],st[i-1]), True
        else:
            st[i],d[i] = min(up[i],st[i-1]), False
    return st,d

# —— 时间 & 模式 ——
async def sync_time():
    global time_offset
    res = await (await session.get(f"{Config.REST}/fapi/v1/time")).json()
    time_offset = res["serverTime"]-int(time.time()*1000)
    LOG.info("Time offset %d", time_offset)

async def detect_mode():
    global is_hedge
    ts=int(time.time()*1000+time_offset)
    qs=urllib.parse.urlencode({"timestamp":ts,"recvWindow":Config.RECV_W})
    sig=hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
    u=f"{Config.REST}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res=await (await session.get(u, headers={"X-MBX-APIKEY":Config.API_KEY})).json()
    is_hedge=res.get("dualSidePosition",False)
    LOG.info("Hedge mode %s", is_hedge)

# —— 持仓 & 下单 ——
class PositionTracker:
    def __init__(self):
        self.long=self.short=0; self.lock=asyncio.Lock()
    async def sync(self):
        ts=int(time.time()*1000+time_offset)
        qs=urllib.parse.urlencode({"timestamp":ts,"recvWindow":Config.RECV_W})
        sig=hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
        u=f"{Config.REST}/fapi/v2/positionRisk?{qs}&signature={sig}"
        try:
            res=await (await session.get(u,headers={"X-MBX-APIKEY":Config.API_KEY})).json()
            if "code" in res: return
            async with self.lock:
                for p in res:
                    if p["symbol"]==Config.SYMBOL:
                        amt=abs(float(p["positionAmt"]))
                        if p.get("positionSide","BOTH")=="LONG": self.long=amt
                        else: self.short=amt
        except: await asyncio.sleep(1)
    async def avail(self,side):
        async with self.lock:
            used = self.long-self.short if side=="BUY" else self.short-self.long
            return max(0, Config.MAX_POS - used)

pos = PositionTracker()

class OrderManager:
    def __init__(self):
        self.batch=[]; self.lock=asyncio.Lock(); self.tokens=1; self.last=time.time()
    def _sign(self,params):
        qs=urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
    def _build(self,side,otype,qty=None,price=None,stop=None,reduce=False):
        ts=int(time.time()*1000+time_offset)
        p={"symbol":Config.SYMBOL,"side":side,"type":otype,
           "recvWindow":Config.RECV_W,"timestamp":ts}
        if is_hedge and otype in ("LIMIT","MARKET"):
            p["positionSide"]="LONG" if side=="BUY" else "SHORT"
        if otype=="LIMIT":
            p.update(timeInForce="GTC",quantity=f"{qty:.6f}",price=f"{price:.2f}")
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            p.update(stopPrice=f"{stop:.2f}",closePosition="true")
        else:
            p["quantity"]=f"{qty:.6f}"
        if reduce: p["reduceOnly"]="true"
        p["signature"]=self._sign(p)
        return p

    async def place(self,side,otype,qty=None,price=None,stop=None,reduce=False):
        if qty and qty>await pos.avail(side):
            LOG.warning("No pos for %s %.4f",side,qty); return
        o=self._build(side,otype,qty,price,stop,reduce)
        LOG.debug("Order→ %s",o)
        async with self.lock:
            self.batch.append(o)
            await self.flush()

    async def flush(self):
        # 简易令牌桶
        now=time.time()
        self.tokens=min(1, self.tokens+(now-self.last))
        self.last=now
        if self.tokens<1 or not self.batch: return
        self.tokens-=1
        data=json.dumps(self.batch); self.batch=[]
        try:
            r=await session.post(f"{Config.REST}/fapi/v1/batchOrders",
                headers={"X-MBX-APIKEY":Config.API_KEY},
                data={"batchOrders":data})
            r.raise_for_status(); LOG.info("Flushed batch")
        except aiohttp.ClientResponseError as e:
            LOG.error("BatchErr %s",e)

mgr = OrderManager()

# —— 策略基类 & 子类 ——
class BaseStrategy:
    def __init__(self, name): self.name=name; self.cd=0
    async def check(self, price):
        if time.time()-self.cd< self.interval: return
        want = await self.signal(price)
        if want:
            await self.execute(want, price)
            self.cd=time.time()
    async def signal(self, price): pass
    async def execute(self, s, price): pass

class MainStrategy(BaseStrategy):
    interval=1
    def __init__(self):
        super().__init__("main"); self.dyn=1; self.last_dyn=0
    async def signal(self, price):
        bb3=await dm.get("3m","bb_pct"); st=await dm.get("15m","st")
        return ("BUY" if price>st else "SELL") if bb3 not in (None,) and (bb3<=0 or bb3>=1) else None
    async def execute(self, side, price):
        # 更新动态
        if time.time()-self.last_dyn>60:
            df=dm.klines["15m"]; atr=(df.high.rolling(14).max()-df.low.rolling(14).min()).iat[-1]
            self.dyn = max(0.5,min(2, atr/price)); self.last_dyn=time.time()
        strength=abs((await dm.get("3m","bb_pct"))-0.5)*2
        qty=min(0.1*strength*self.dyn, Config.MAX_POS*0.3)
        # 开仓
        for lvl in [0.0025,0.0040,0.0060,0.0080,0.0160]:
            p=price*(1+(lvl*self.dyn) if side=="BUY" else 1-(lvl*self.dyn))
            await mgr.place(side,"LIMIT",qty,p)
        # 止盈限仓
        rev="SELL" if side=="BUY" else "BUY"
        for tp_mul in (0.0102,0.0123,0.0150*1.2,0.0180*1.5,0.0220*2.0):
            tp=tp_mul*self.dyn
            p=price*(1+tp if side=="BUY" else 1-tp)
            await mgr.place(rev,"LIMIT",qty*0.2,p,reduce=True)
        # 止损/市价平仓
        sl=price*(0.98 if side=="BUY" else 1.02)*self.dyn
        ot="STOP_MARKET" if side=="BUY" else "TAKE_PROFIT_MARKET"
        await mgr.place(rev,ot,stop=sl)

class MACDStrategy(BaseStrategy):
    interval=5
    async def signal(self, price):
        df=dm.klines["15m"]
        if len(df)<30 or "macd" not in df: return None
        mac=df.macd
        return "BUY" if mac.iat[-2]<0<mac.iat[-1] else "SELL" if mac.iat[-2]>0>mac.iat[-1] else None
    async def execute(self, side, price):
        for p in (0.3,0.5,0.2):
            await mgr.place(side,"MARKET",0.15*p)
            await asyncio.sleep(0.5)

class RVGIStrategy(BaseStrategy):
    interval=5
    async def signal(self, price):
        df=dm.klines["15m"]
        if len(df)<11 or not {"rvgi","rvsig"}.issubset(df): return None
        rv,sg = df.rvgi.iat[-1], df.rvsig.iat[-1]
        ma7,ma25,ma99 = (df.close.rolling(7).mean().iat[-1],
                         df.close.rolling(25).mean().iat[-1],
                         df.close.rolling(99).mean().iat[-1])
        if rv>sg and price>ma7>ma25>ma99: return "BUY"
        if rv<sg and price<ma7<ma25<ma99: return "SELL"
        return None
    async def execute(self, side, price):
        await mgr.place(side,"MARKET",0.05)

class TripleTrendStrategy(BaseStrategy):
    interval=1
    def __init__(self):
        super().__init__("triple"); self.state=None
    async def signal(self, price):
        df=dm.klines["15m"]
        if len(df)<12: return None
        h,l,c = df.high.values, df.low.values, df.close.values
        _,d1=supertrend(h,l,c,10,1); _,d2=supertrend(h,l,c,11,2); _,d3=supertrend(h,l,c,12,3)
        up=d1[-1] and d2[-1] and d3[-1]
        dn=not (d1[-1] or d2[-1] or d3[-1])
        if up and self.state!="UP": self.state="UP"; return "BUY"
        if dn and self.state!="DOWN": self.state="DOWN"; return "SELL"
        if self.state=="UP" and not up: self.state="DOWN"; return "SELL"
        if self.state=="DOWN" and not dn: self.state="UP"; return "BUY"
        return None
    async def execute(self, side, price):
        await mgr.place(side,"MARKET",0.15)

# 注册策略
strategies = [MainStrategy(), MACDStrategy(), RVGIStrategy(), TripleTrendStrategy()]

# —— WebSocket 市场流 ——
async def market_ws():
    global latest_price
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=20) as ws:
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        latest_price=float(d["p"])
                        await dm.track_price(latest_price, time.time())
                    else:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        await dm.update_kline(tf, float(k["o"]),float(k["h"]),
                                              float(k["l"]),float(k["c"]), k["t"])
        except Exception as e:
            await asyncio.sleep(min(2**retry,30)); retry+=1

# —— WebSocket 用户流 ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER) as ws:
                ts=int(time.time()*1000+time_offset)
                params={"apiKey":Config.ED25519_API,"timestamp":ts}
                pay="&".join(f"{k}={v}" for k,v in sorted(params.items()))
                params["signature"]=base64.b64encode(ed_priv.sign(pay.encode())).decode()
                await ws.send(json.dumps({"id":str(uuid.uuid4()),"method":"session.logon","params":params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({"id":str(uuid.uuid4()),"method":"session.status"}))
                asyncio.create_task(hb())
                await ws.wait_closed()
        except Exception:
            await asyncio.sleep(5)

# —— 维护 & 主循环 ——
async def maintenance():
    asyncio.create_task(watch_reload())
    while True:
        await asyncio.sleep(Config.SYNC_INT)
        await sync_time(); await detect_mode(); await pos.sync(); await mgr.flush()

async def watch_reload():
    async for ch in watchfiles.awatch('.'):
        for _,p in ch:
            if p.endswith(('.env','py')):
                LOG.info("Reloading env/py"); load_dotenv(); await pos.sync()

async def engine():
    while True:
        await dm.wait()
        if dm.price is None or time.time()-dm.ptime>60: continue
        for strat in strategies:
            await strat.check(dm.price)

async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time(); await detect_mode()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())