#!/usr/bin/env python3
# coding: utf-8

import os, time, json, uuid, math, base64, hmac, hashlib, asyncio, logging, urllib.parse, signal
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import uvloop, aiohttp, websockets, pandas as pd
from ta.trend import MACD
from numba import jit
import watchfiles

# —— 高性能事件循环 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv('/root/zhibai/.env')

# —— 日志配置 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)

# —— 配置 ——
class Config:
    SYMBOL          = 'ETHUSDC'
    PAIR            = SYMBOL.lower()
    API_KEY         = os.getenv('BINANCE_API_KEY')
    SECRET_KEY      = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API     = os.getenv('ED25519_API_KEY')
    ED25519_KEY     = os.getenv('ED25519_KEY_PATH')
    REST_BASE       = 'https://fapi.binance.com'
    WS_MARKET       = f"wss://fstream.binance.com/stream?streams={PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    WS_USER         = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    RECV_WINDOW     = 5000
    SYNC_INTERVAL   = 300
    MAX_POS         = 2.0
    HIST_LIMIT      = 1000
    ORDER_COOLDOWN  = 360        # 同一信号同方向最小间隔（秒）

# —— 全局状态 ——
session: aiohttp.ClientSession = None
time_offset = 0
is_hedge = False
price_step = qty_step = None
price_prec = qty_prec = 0

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 key loaded")

# —— 向下取整 ——
def quantize(val: float, step: float) -> float:
    return math.floor(val/step) * step

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    res = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = res['serverTime'] - int(time.time()*1000)
    LOG.info(f"Time offset {time_offset}ms")

async def detect_mode():
    global is_hedge
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    data = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge = data.get('dualSidePosition', False)
    LOG.info(f"Hedge mode: {is_hedge}")

# —— 加载精度过滤器 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    info = await (await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")).json()
    sym = next(s for s in info['symbols'] if s['symbol']==Config.SYMBOL)
    pf = next(f for f in sym['filters'] if f['filterType']=='PRICE_FILTER')
    ls = next(f for f in sym['filters'] if f['filterType']=='LOT_SIZE')
    price_step, qty_step = float(pf['tickSize']), float(ls['stepSize'])
    price_prec = int(round(-math.log10(price_step)))
    qty_prec   = int(round(-math.log10(qty_step)))
    LOG.info(f"Filters loaded: price_step={price_step}, qty_step={qty_step}, price_prec={price_prec}, qty_prec={qty_prec}")

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        tfs = ("3m","15m","1h")
        self.klines  = {tf: pd.DataFrame(columns=["open","high","low","close"]) for tf in tfs}
        self.last_ts = {tf:0 for tf in tfs}
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.price   = None; self.ptime = 0

    async def load_history(self):
        async with self.lock:
            for tf in self.klines:
                res = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                                        params={"symbol":Config.SYMBOL,"interval":tf,"limit":Config.HIST_LIMIT},
                                        headers={"X-MBX-APIKEY":Config.API_KEY})
                data = await res.json()
                df = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),"low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf], self.last_ts[tf] = df, int(data[-1][0])
                self._compute(tf)
                LOG.info(f"{tf} history loaded: {len(df)}")

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            if ts>self.last_ts[tf]:
                df.loc[len(df), ["open","high","low","close"]] = [o,h,l,c]
            else:
                df.iloc[-1, df.columns.get_indexer(["open","high","low","close"])] = [o,h,l,c]
            self.last_ts[tf] = ts
            self._compute(tf)
            if tf=="3m": LOG.debug(f"3m bb_pct={df.bb_pct.iat[-1]:.4f}")
            if tf=="15m" and {"st","macd"}.issubset(df):
                LOG.debug(f"15m st={df.st.iat[-1]:.2f} macd={df.macd.iat[-1]:.4f}")
            self._evt.set()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df)<20: return
        m, s = df.close.rolling(20).mean(), df.close.rolling(20).std()
        df["bb_up"], df["bb_dn"] = m+2*s, m-2*s
        df["bb_pct"] = (df.close - df.bb_dn)/(df.bb_up-df.bb_dn)
        if tf=="15m":
            hl2 = (df.high+df.low)/2
            atr = df.high.rolling(10).max() - df.low.rolling(10).min()
            df["st"]   = hl2 - 3*atr
            df["macd"] = MACD(df.close,12,26,9).macd_diff()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            return df[col].iat[-1] if not df.empty and col in df else None

    async def wait_update(self):
        await self._evt.wait(); self._evt.clear()

    async def track_price(self, p, ts):
        async with self.lock:
            self.price, self.ptime = p, ts
        LOG.debug(f"Price update: {p:.2f}@{ts}")
        self._evt.set()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n=len(c); st=[0.0]*n; dirc=[False]*n
    hl2=[(h[i]+l[i])/2 for i in range(n)]
    atr=[max(h[i-per+1:i+1])-min(l[i-per+1:i+1]) for i in range(n)]
    up=[hl2[i]+mult*atr[i] for i in range(n)]
    dn=[hl2[i]-mult*atr[i] for i in range(n)]
    st[0],dirc[0]=up[0],True
    for i in range(1,n):
        if c[i]>st[i-1]:
            st[i],dirc[i]=max(dn[i],st[i-1]),True
        else:
            st[i],dirc[i]=min(up[i],st[i-1]),False
    return st,dirc

# —— 持仓追踪 ——
class PositionTracker:
    def __init__(self):
        self.long=self.short=0.0; self.lock=asyncio.Lock()
    async def sync(self):
        ts=int(time.time()*1000+time_offset)
        qs=urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
        sig=hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
        url=f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res=await (await session.get(url,headers={'X-MBX-APIKEY':Config.API_KEY})).json()
        if isinstance(res,list):
            async with self.lock:
                for p in res:
                    if p['symbol']==Config.SYMBOL:
                        amt=abs(float(p['positionAmt']))
                        if p.get('positionSide','BOTH')=='LONG': self.long=amt
                        else: self.short=amt
    async def avail(self, side):
        async with self.lock:
            used=(self.long-self.short) if side=='BUY' else (self.short-self.long)
            return max(0.0, Config.MAX_POS - used)

pos_tracker = PositionTracker()

# —— 会话管理 ——
async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 下单管理 + 守卫 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.last_order = {}  # { sig_key: timestamp }

    def _guard(self, sig):
        now = time.time()
        if now - self.last_order.get(sig, 0) < Config.ORDER_COOLDOWN:
            return False
        self.last_order[sig] = now
        return True

    async def place(self, side, otype, qty=None, price=None, stop=None, sig_key=None):
        if sig_key and not self._guard(sig_key):
            LOG.debug(f"Guard skip {sig_key}")
            return
        await ensure_session()
        ts=int(time.time()*1000+time_offset)
        params={"symbol":Config.SYMBOL,"side":side,"type":otype,"timestamp":ts,"recvWindow":Config.RECV_WINDOW}
        if qty is not None:
            q=quantize(qty,qty_step); params["quantity"]=f"{q:.{qty_prec}f}"
        if price is not None:
            p=quantize(price,price_step); params["price"]=f"{p:.{price_prec}f}"
        if stop is not None:
            sp=quantize(stop,price_step); params["stopPrice"]=f"{sp:.{price_prec}f}"
        if otype=="LIMIT": params["timeInForce"]="GTC"
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"): params["closePosition"]="true"
        if is_hedge and otype in ("LIMIT","MARKET"):
            params["positionSide"]="LONG" if side=="BUY" else "SHORT"
        qs=urllib.parse.urlencode(sorted(params.items()))
        sig=hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
        url=f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        headers={'X-MBX-APIKEY':Config.API_KEY}
        async with self.lock:
            r=await session.post(url,headers=headers); txt=await r.text()
            if r.status==200:
                LOG.info(f"Order OK {otype} {side} [{sig_key}]")
            else:
                LOG.error(f"Order ERR {otype} {side} [{sig_key}] {txt}")

mgr = OrderManager()

# —— 策略：Main ——
class MainStrategy:
    def __init__(self):
        self.interval=1; self._last=0
    async def check(self, price):
        now=time.time()
        if now-self._last<self.interval: return

        # 趋势判定 (15m Supertrend)
        df15 = data_mgr.klines["15m"]
        h15,l15,c15 = df15.high.values, df15.low.values, df15.close.values
        st_line, st_dir = numba_supertrend(h15,l15,c15,10,3)
        up_trend = price>st_line[-1] and st_dir[-1]
        dn_trend = price<st_line[-1] and not st_dir[-1]

        # 强弱信号 (1h BB%)
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong_short = bb1>0.8; strong_long = bb1<0.2
        weak_short   = not strong_short; weak_long   = not strong_long

        # 3m BB%B
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not (bb3<=0 or bb3>=1): return

        # 决定方向与仓位
        if up_trend:
            side="BUY"
            qty = 0.12 if strong_long else 0.03
            strength = "STRONG" if strong_long else "WEAK"
        elif dn_trend:
            side="SELL"
            qty = 0.12 if strong_short else 0.03
            strength = "STRONG" if strong_short else "WEAK"
        else:
            return

        self._last = now
        levels = [0.0025,0.004,0.006,0.008,0.016]
        tp_levels = [0.0102,0.0123,0.015,0.018,0.022]

        # 分级挂限价
        for lvl in levels:
            price0 = price * (1+lvl if side=="BUY" else 1-lvl)
            sig=f"MAIN_{side}_{strength}_{lvl}"
            await mgr.place(side,"LIMIT",qty=qty,price=price0,sig_key=sig)

        # 止盈
        for tp in tp_levels:
            sig=f"MAIN_TP_{side}_{strength}_{tp}"
            await mgr.place("TAKE_PROFIT_MARKET","TAKE_PROFIT_MARKET",sig_key=sig)

        # 初始止损
        sl = price * (0.97 if side=="BUY" else 1.02)
        sig=f"MAIN_SL_{side}"
        await mgr.place("STOP_MARKET","STOP_MARKET",stop=sl,sig_key=sig)

# —— 策略：15 级 MACD ——
class MACDStrategy:
    def __init__(self):
        self._in_round = False

    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        prev, cur = df.macd.iat[-2], df.macd.iat[-1]
        if prev>0>cur and not self._in_round:
            offset=0.005; sp=price*(1+offset)
            await mgr.place("SELL","LIMIT",qty=0.15,price=sp,sig_key="MACD_SELL")
            await mgr.place("STOP_MARKET","STOP_MARKET",stop=sp*1.03,sig_key="MACD_SELL_SL")
            await mgr.place("TAKE_PROFIT_MARKET","TAKE_PROFIT_MARKET",sig_key="MACD_SELL_TP")
            self._in_round=True
        elif prev<0<cur and self._in_round:
            offset=0.005; bp=price*(1-offset)
            await mgr.place("BUY","LIMIT",qty=0.15,price=bp,sig_key="MACD_BUY")
            await mgr.place("STOP_MARKET","STOP_MARKET",stop=bp*0.97,sig_key="MACD_BUY_SL")
            await mgr.place("TAKE_PROFIT_MARKET","TAKE_PROFIT_MARKET",sig_key="MACD_BUY_TP")
            self._in_round=False

# —— 策略：TripleTrend ——
class TripleTrendStrategy:
    def __init__(self):
        self.interval=1; self._last=0; self._state=None
    async def check(self, price):
        now=time.time()
        if now-self._last<self.interval: return
        df=data_mgr.klines["15m"]
        if len(df)<12: return
        h,l,c = df.high.values, df.low.values, df.close.values
        try:
            _,d1=numba_supertrend(h,l,c,10,1)
            _,d2=numba_supertrend(h,l,c,11,2)
            _,d3=numba_supertrend(h,l,c,12,3)
        except: return
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not (d1[-1] or d2[-1] or d3[-1])
        side=None
        if up and self._state!="UP": side, self._state="BUY","UP"
        elif dn and self._state!="DOWN": side, self._state="SELL","DOWN"
        elif self._state=="UP" and not up: side, self._state="SELL","DOWN"
        elif self._state=="DOWN" and not dn: side, self._state="BUY","UP"
        if side:
            self._last=now; LOG.debug(f"Triple→{side}")
            await mgr.place(side,"MARKET",qty=0.15,sig_key=f"TRIPLE_{side}")

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket 市场 ——
async def market_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        p=float(d["p"]); await data_mgr.track_price(p,int(time.time()*1000))
                    else:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        await data_mgr.update_kline(tf,
                            float(k["o"]),float(k["h"]),float(k["l"]),float(k["c"]),k["t"])
        except Exception as e:
            delay=min(2**retry,30); LOG.error("MarketWS %s, retry %ds",e,delay)
            await asyncio.sleep(delay); retry+=1

# —— WebSocket 用户 ——
async def user_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_USER, ping_interval=None) as ws:
                retry=0
                ts=int(time.time()*1000+time_offset)
                params={"apiKey":Config.ED25519_API,"timestamp":ts}
                payload="&".join(f"{k}={v}" for k,v in sorted(params.items()))
                sig=base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params["signature"]=sig
                await ws.send(json.dumps({"id":str(uuid.uuid4()),"method":"session.logon","params":params}))
                async for msg in ws:
                    r=json.loads(msg)
                    if r.get("result",{}).get("orderId"):
                        LOG.info("ExecReport %s",r["result"])
        except Exception as e:
            LOG.error("UserWS %s, retry 5s",e)
            await asyncio.sleep(5); retry+=1

# —— 热重载 & 维护 ——
async def watch_reload():
    async for _ in watchfiles.awatch('/root/zhibai'):
        LOG.info("Reloading…"); load_dotenv('/root/zhibai/.env'); await pos_tracker.sync()

async def maintenance():
    asyncio.create_task(watch_reload())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode(); await pos_tracker.sync()

# —— 引擎 ——
async def engine():
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60: continue
        for strat in strategies:
            try: await strat.check(data_mgr.price)
            except Exception as e: LOG.exception("Strategy %s failed: %s", strat.__class__.__name__, e)

# —— 入口 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    await sync_time(); await detect_mode(); await load_symbol_filters()
    await data_mgr.load_history(); await pos_tracker.sync()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())