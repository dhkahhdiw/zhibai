#!/usr/bin/env python3
# coding: utf-8

import os, time, json, uuid, base64, hmac, hashlib, asyncio, logging, urllib.parse
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
    API_KEY       = os.getenv('BINANCE_API_KEY')
    SECRET_KEY    = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API   = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET     = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_USER       = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE     = 'https://fapi.binance.com'
    RECV_WINDOW   = 5000
    SYNC_INTERVAL = 300
    MAX_POSITION  = 2.0

# —— 日志 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)
class Rotator(logging.Handler):
    def __init__(self, path, max_lines=1000):
        super().__init__(); self.path, self.max = path, max_lines
        open(path,'a').close()
    def emit(self, record):
        msg = self.format(record)
        with open(self.path,'a') as f: f.write(msg+'\n')
        lines = open(self.path).read().splitlines()
        if len(lines) > self.max:
            open(self.path,'w').write('\n'.join(lines[-self.max:])+'\n')
fh = Rotator('bot.log'); fh.setFormatter(fmt); LOG.addHandler(fh)

# —— 全局状态 ——
session: aiohttp.ClientSession
time_offset = 0
latest_price = None
is_hedge_mode = False

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Loaded Ed25519 private key")

# —— DataManager：K 线 & 实时价 ——
class DataManager:
    def __init__(self):
        self.klines   = {tf: pd.DataFrame(columns=["open","high","low","close"])
                         for tf in ("3m","15m","1h")}
        self.last_ts  = {tf:0 for tf in ("3m","15m","1h")}
        self.lock     = asyncio.Lock()
        self._evt     = asyncio.Event()
        self.price    = None
        self.ptime    = 0

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
            # 关键指标日志
            if tf=="3m":
                LOG.debug(f"3m bb_pct={df['bb_pct'].iat[-1]:.4f}")
            if tf=="15m":
                LOG.debug(
                    f"15m st={df['st'].iat[-1]:.2f} "
                    f"macd={df['macd'].iat[-1]:.4f} "
                    f"rvgi={df['rvgi'].iat[-1]:.4f} "
                    f"rvsig={df['rvsig'].iat[-1]:.4f}"
                )
            self._evt.set()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df) < 20: return
        m = df["close"].rolling(20).mean()
        s = df["close"].rolling(20).std()
        df["bb_up"], df["bb_dn"] = m+2*s, m-2*s
        df["bb_pct"] = (df["close"]-df["bb_dn"]) / (df["bb_up"]-df["bb_dn"])
        if tf == "15m":
            hl2 = (df["high"]+df["low"])/2
            atr = df["high"].rolling(10).max()-df["low"].rolling(10).min()
            df["st"]    = hl2 - 3*atr
            df["macd"]  = MACD(df["close"],12,26,9).macd_diff()
            rv = ROCIndicator(close=(df["close"]-df["open"]), window=10).roc()
            df["rvgi"], df["rvsig"] = rv, rv.rolling(4).mean()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df and not df.empty:
                return df[col].iat[-1]
        return None

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

    async def track_price(self, price, ts):
        async with self.lock:
            self.price, self.ptime = price, ts
        LOG.debug(f"Price update: {price:.2f}@{ts}")
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
    res = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = res['serverTime'] - int(time.time()*1000)
    LOG.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url=f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res= await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge_mode = res.get('dualSidePosition', False)
    LOG.info("Hedge mode: %s", is_hedge_mode)

# —— 持仓追踪 ——
class PositionTracker:
    def __init__(self):
        self.long = self.short = 0.0
        self.lock = asyncio.Lock()

    async def sync(self):
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
        sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url=f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        try:
            res = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
            if 'code' in res:
                LOG.error("Position sync failed: %s", res)
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
            LOG.error("Position sync error: %s", e)
            await asyncio.sleep(1)

    async def avail(self, side):
        async with self.lock:
            used = (self.long-self.short) if side=='BUY' else (self.short-self.long)
            return max(0.0, Config.MAX_POSITION - used)

pos_tracker = PositionTracker()

# —— 下单管理 & 限流 ——
class OrderManager:
    def __init__(self):
        self.batch=[]; self.lock=asyncio.Lock()
        self.tokens=1; self.last=time.time()

    def _sign(self, params):
        qs = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

    def _build(self, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        ts=int(time.time()*1000+time_offset)
        p={"symbol":Config.SYMBOL,"side":side,"type":otype,
           "recvWindow":Config.RECV_WINDOW,"timestamp":ts}
        if is_hedge_mode and otype in ("LIMIT","MARKET"):
            p["positionSide"]="LONG" if side=="BUY" else "SHORT"
        if otype=="LIMIT":
            p.update(timeInForce="GTC",quantity=f"{qty:.6f}",price=f"{price:.2f}")
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            p.update(stopPrice=f"{stop:.2f}",closePosition="true")
        else:
            p["quantity"]=f"{qty:.6f}"
        if reduceOnly:
            p["reduceOnly"]="true"
        p["signature"]=self._sign(p)
        return p

    async def place(self, side, otype, qty=None, price=None, stop=None, reduceOnly=False):
        if qty and qty>await pos_tracker.avail(side):
            LOG.warning("Insufficient pos for %s %.6f", side, qty)
            return
        order = self._build(side, otype, qty, price, stop, reduceOnly)
        LOG.info("Enqueue order: %s", order)
        async with self.lock:
            self.batch.append(order)
            await self.flush()

    async def flush(self):
        now = time.time()
        self.tokens = min(1, self.tokens + (now-self.last))
        self.last = now
        if self.tokens < 1 or not self.batch:
            return
        self.tokens -= 1
        data=json.dumps(self.batch); self.batch.clear()
        try:
            r = await session.post(
                f"{Config.REST_BASE}/fapi/v1/batchOrders",
                headers={'X-MBX-APIKEY':Config.API_KEY},
                data={'batchOrders':data}
            )
            r.raise_for_status()
            LOG.info("Batch flushed")
        except aiohttp.ClientResponseError as e:
            LOG.error("Batch flush error: %s", e)

mgr = OrderManager()

# —— 策略基类 & 子类 ——
class BaseStrategy:
    interval=1
    def __init__(self):
        self._last=0
    async def check_signal(self, price):
        if time.time()-self._last < self.interval:
            return
        sig = await self.signal(price)
        if sig:
            await self.execute(sig, price)
            self._last = time.time()
    async def signal(self, price): ...
    async def execute(self, side, price): ...

class MainStrategy(BaseStrategy):
    interval=1
    def __init__(self):
        super().__init__(); self._dyn=1; self._last_dyn=0
    async def signal(self, price):
        bb3 = await data_mgr.get("3m","bb_pct")
        st  = await data_mgr.get("15m","st")
        if bb3 is None or st is None:
            return None
        return "BUY" if price>st else "SELL" if (bb3<=0 or bb3>=1) else None
    async def execute(self, side, price):
        # 更新动态参数
        if time.time()-self._last_dyn>60:
            df=data_mgr.klines["15m"]
            atr=(df.high.rolling(14).max()-df.low.rolling(14).min()).iat[-1]
            self._dyn = max(0.5, min(2.0, atr/price))
            self._last_dyn = time.time()
            LOG.debug(f"Dynamic vol={self._dyn:.4f}")
        # 开仓 & 止盈止损
        bb3=await data_mgr.get("3m","bb_pct")
        strength=abs(bb3-0.5)*2
        qty=min(0.1*strength*self._dyn, Config.MAX_POSITION*0.3)
        # 限价开仓
        for lvl in [0.0025,0.004,0.006,0.008,0.016]:
            pr = price*(1+(lvl*self._dyn) if side=="BUY" else 1-(lvl*self._dyn))
            await mgr.place(side,"LIMIT",qty,pr)
        # 多级止盈
        rev="SELL" if side=="BUY" else "BUY"
        for tp_mul in (0.0102,0.0123,0.015*1.2,0.018*1.5,0.022*2.0):
            pr = price*(1+(tp_mul*self._dyn) if side=="BUY" else 1-(tp_mul*self._dyn))
            await mgr.place(rev,"LIMIT",qty*0.2,pr,reduceOnly=True)
        # 止损市价
        sl = price*(0.98 if side=="BUY" else 1.02)*self._dyn
        ot = "STOP_MARKET" if side=="BUY" else "TAKE_PROFIT_MARKET"
        await mgr.place(rev,ot,stop=sl)

class MACDStrategy(BaseStrategy):
    interval=5
    async def signal(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df:
            return None
        m = df.macd
        return "BUY" if m.iat[-2]<0<m.iat[-1] else "SELL" if m.iat[-2]>0>m.iat[-1] else None
    async def execute(self, side, price):
        LOG.debug(f"MACD → {side}")
        for p in (0.3,0.5,0.2):
            await mgr.place(side,"MARKET",0.15*p)
            await asyncio.sleep(0.5)

class RVGIStrategy(BaseStrategy):
    interval=5
    async def signal(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<11 or not {"rvgi","rvsig"}.issubset(df):
            return None
        rv, sg = df.rvgi.iat[-1], df.rvsig.iat[-1]
        ma7,ma25,ma99 = (df.close.rolling(7).mean().iat[-1],
                         df.close.rolling(25).mean().iat[-1],
                         df.close.rolling(99).mean().iat[-1])
        if rv>sg and price>ma7>ma25>ma99:
            return "BUY"
        if rv<sg and price<ma7<ma25<ma99:
            return "SELL"
        return None
    async def execute(self, side, price):
        LOG.debug(f"RVGI → {side}")
        await mgr.place(side,"MARKET",0.05)

class TripleTrendStrategy(BaseStrategy):
    interval=1
    def __init__(self):
        super().__init__(); self._state=None
    async def signal(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<12:
            return None
        h,l,c = df.high.values, df.low.values, df.close.values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not (d1[-1] or d2[-1] or d3[-1])
        if up and self._state!="UP":
            self._state="UP"; return "BUY"
        if dn and self._state!="DOWN":
            self._state="DOWN"; return "SELL"
        if self._state=="UP" and not up:
            self._state="DOWN"; return "SELL"
        if self._state=="DOWN" and not dn:
            self._state="UP"; return "BUY"
        return None
    async def execute(self, side, price):
        LOG.debug(f"TripleTrend → {side}")
        await mgr.place(side,"MARKET",0.15)

strategies = [MainStrategy(), MACDStrategy(), RVGIStrategy(), TripleTrendStrategy()]

# —— WebSocket 市场流 ——
async def market_ws():
    global latest_price
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry = 0
                async for msg in ws:
                    o = json.loads(msg)
                    s,d = o["stream"], o["data"]
                    if s.endswith("@markPrice"):
                        latest_price = float(d["p"])
                        await data_mgr.track_price(latest_price, time.time())
                    else:
                        tf = s.split("@")[1].split("_")[1]
                        k = d["k"]
                        await data_mgr.update_kline(
                            tf,
                            float(k["o"]), float(k["h"]),
                            float(k["l"]), float(k["c"]),
                            k["t"]
                        )
        except Exception as e:
            LOG.error("Market WS error: %s, reconnect in %ds", e, min(2**retry,30))
            await asyncio.sleep(min(2**retry,30))
            retry += 1

# —— WebSocket 用户流 ——
async def user_ws():
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_USER, ping_interval=None) as ws:
                retry = 0
                # 登录
                ts = int(time.time()*1000+time_offset)
                params={"apiKey":Config.ED25519_API,"timestamp":ts}
                payload="&".join(f"{k}={v}" for k,v in sorted(params.items()))
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params["signature"] = sig
                await ws.send(json.dumps({
                    "id": str(uuid.uuid4()),
                    "method": "session.logon",
                    "params": params
                }))
                # 自动回应服务器 ping
                async for message in ws:
                    # 服务器通常发送 ping 帧，无需手动处理，库会自动 pong
                    await asyncio.sleep(0)  # placeholder 保持循环
        except Exception as e:
            LOG.error("User WS error: %s, reconnect in 5s", e)
            await asyncio.sleep(5)
            retry += 1

# —— 配置热加载 & 定期维护 ——
async def watch_reload():
    async for changes in watchfiles.awatch('.'):
        for _, path in changes:
            if path.endswith(('.env','py')):
                LOG.info("Reloading env/py")
                load_dotenv()
                await pos_tracker.sync()

async def maintenance():
    asyncio.create_task(watch_reload())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await detect_mode()
        await pos_tracker.sync()
        await mgr.flush()

# —— 策略引擎 ——
async def engine():
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60:
            continue
        for strat in strategies:
            await strat.check_signal(data_mgr.price)

# —— 主入口 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time(); await detect_mode()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())