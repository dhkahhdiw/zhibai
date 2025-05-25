#!/usr/bin/env python3
# coding: utf-8

import os, time, json, math, asyncio, logging, signal, uuid, base64
from collections import defaultdict

import uvloop, pandas as pd, aiohttp, websockets
from dotenv import load_dotenv
from ta.trend import MACD, ADXIndicator
from numba import jit
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# —— 高性能事件循环 & 环境加载 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv()

# —— 日志配置 ——
LOG = logging.getLogger('vhf_bot')
LOG.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— 全局配置 ——
class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    WS_MARKET_URL    = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_TRADE_URL     = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    ED25519_API_KEY  = os.getenv('YZ_ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('YZ_ED25519_KEY_PATH')
    RECV_WINDOW      = 5000
    ROTATION_COOLDOWN= 1800
    HIST_LIMIT       = 1000

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    _ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 key loaded")

def sign_payload(params: dict) -> str:
    items = sorted(params.items())
    payload = "&".join(f"{k}={v}" for k,v in items)
    sig = _ed_priv.sign(payload.encode('ascii'))
    return base64.urlsafe_b64encode(sig).decode('ascii').rstrip('=')

def quantize(val, step):
    return math.floor(val/step) * step

# —— 全局精度变量 ——
price_step = qty_step = price_prec = qty_prec = 0

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs = ("3m","15m","1h")
        self.klines = {tf: pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = dict.fromkeys(self.tfs, 0)
        self.price = None; self.ptime = 0
        self.lock = asyncio.Lock(); self._evt = asyncio.Event()

    async def load_history(self):
        async with aiohttp.ClientSession() as sess:
            for tf in self.tfs:
                data = await (await sess.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={"symbol":Config.SYMBOL,"interval":tf,"limit":Config.HIST_LIMIT},
                    timeout=10
                )).json()
                df = pd.DataFrame([{
                    "open":float(x[1]),"high":float(x[2]),
                    "low":float(x[3]),"close":float(x[4])
                } for x in data])
                self.klines[tf] = df
                self.last_ts[tf] = int(data[-1][0])
                self._compute(tf)
        LOG.info("历史K线加载完毕")

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            if ts > self.last_ts[tf]:
                idx = len(df)
                df.loc[idx, ["open","high","low","close"]] = [o,h,l,c]
                self.last_ts[tf] = ts
            else:
                last = df.index[-1]
                df.loc[last, ["open","high","low","close"]] = [o,h,l,c]
            self._compute(tf)
            self._evt.set()

    async def track_price(self, p, ts):
        async with self.lock:
            self.price, self.ptime = p, ts
            self._evt.set()

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df) < 20: return
        m = df.close.rolling(20).mean(); s = df.close.rolling(20).std()
        df["bb_up"], df["bb_dn"] = m + 2*s, m - 2*s
        df["bb_pct"] = (df.close - df.bb_dn) / (df.bb_up - df.bb_dn)
        adx = ADXIndicator(df.high, df.low, df.close, window=14)
        df["adx"], df["dmp"], df["dmn"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
        if tf == "15m":
            df["macd"] = MACD(df.close,12,26,9).macd_diff()
            df["ma7"]  = df.close.rolling(7).mean()
            df["ma25"] = df.close.rolling(25).mean()
            df["ma99"] = df.close.rolling(99).mean()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n=len(c); st=[0.0]*n; dirc=[False]*n
    hl2=[(h[i]+l[i])/2 for i in range(n)]
    atr=[max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up=[hl2[i]+mult*atr[i] for i in range(n)]
    dn=[hl2[i]-mult*atr[i] for i in range(n)]
    st[0],dirc[0]=up[0],True
    for i in range(1,n):
        if c[i] > st[i-1]:
            st[i],dirc[i] = max(dn[i],st[i-1]), True
        else:
            st[i],dirc[i] = min(up[i],st[i-1]), False
    return st,dirc

# —— 下单管理 via WebSocket ——
class OrderManager:
    def __init__(self):
        self.ws = None
        self.lock = asyncio.Lock()
        self.guard = defaultdict(lambda:{'ts':0,'fp':None,'trend':None})
        self.cooldown = Config.ROTATION_COOLDOWN
        self.ready = asyncio.Event()

    async def init_ws(self):
        retry = 0
        while True:
            try:
                self.ws = await websockets.connect(
                    Config.WS_TRADE_URL, ping_interval=20, ping_timeout=60
                )
                ts = int(time.time()*1000)
                params = {"apiKey":str(Config.ED25519_API_KEY),"timestamp":ts}
                sig = sign_payload(params)
                req_id = str(uuid.uuid4())
                req = {"id":req_id,"method":"session.logon",
                       "params":{**params,"signature":sig}}
                await self.ws.send(json.dumps(req))
                LOG.info("Trade WS: session.logon sent, waiting ack…")
                # 等待 ACK
                while True:
                    msg = await self.ws.recv()
                    resp = json.loads(msg)
                    if resp.get("id")==req_id and resp.get("status")==200:
                        LOG.info("Trade WS: session.logon successful")
                        self.ready.set()
                        return
            except Exception as e:
                delay = min(2**retry, 30)
                LOG.error(f"[WS TRD INIT] {e}, reconnect in {delay}s")
                await asyncio.sleep(delay)
                retry += 1

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, extra=None):
        await self.ready.wait()
        fp = f"{strat}|{side}|{otype}|{hash(frozenset((extra or {}).items()))}"
        trend = 'LONG' if side=='BUY' else 'SHORT'
        st = self.guard[strat]; now=time.time()
        if st['fp']==fp or (st['trend']==trend and now-st['ts']<self.cooldown):
            return
        await self.place(side, otype, qty, price, stop, extra or {})
        self.guard[strat] = {'fp':fp,'trend':trend,'ts':now}

    async def place(self, side, otype, qty=None, price=None, stop=None, extra=None):
        async with self.lock:
            params = {
                "apiKey":str(Config.ED25519_API_KEY),
                "symbol":Config.SYMBOL,"side":side,"type":otype,
                "timestamp":int(time.time()*1000),
                "recvWindow":Config.RECV_WINDOW
            }
            if price is not None:
                params["price"] = f"{quantize(price,price_step):.{price_prec}f}"
            if qty is not None:
                params["quantity"] = f"{quantize(qty,qty_step):.{qty_prec}f}"
            if otype in ("STOP_MARKET","TAKE_PROFIT_MARKET") and stop is not None:
                params["stopPrice"]   = f"{quantize(stop,price_step):.{price_prec}f}"
                params["workingType"] = "MARK_PRICE"
                params["reduceOnly"]  = "true"
            if extra:
                params.update(extra)
            sig = sign_payload(params)
            req = {"id":str(uuid.uuid4()),"method":"order.place",
                   "params":{**params,"signature":sig}}
            await self.ws.send(json.dumps(req))
            LOG.debug(f"[ORDER] {otype} {side} sent → {req['id']}")

mgr = OrderManager()

# —— 本地 OCO 管理 ——
class PositionTracker:
    class Pos:
        __slots__ = ('side','qty','sl','tp','cloid','active')
        def __init__(self, side, qty, sl, tp, cloid):
            self.side, self.qty, self.sl, self.tp = side,qty,sl,tp
            self.cloid = cloid; self.active = True

    def __init__(self):
        self.pos = {}; self.lock = asyncio.Lock(); self.next_cloid = 1

    async def on_fill(self, data):
        if data.get('status') != "FILLED": return
        side = data['side']
        qty  = float(data.get('executedQty',0))
        price= float(data.get('avgPrice',0)) or float(data.get('price',0))
        sl   = price * (0.98 if side=="BUY" else 1.02)
        tp   = price * (1.02 if side=="BUY" else 0.98)
        cloid= self.next_cloid; self.next_cloid+=1
        async with self.lock:
            self.pos[cloid] = self.Pos(side,qty,sl,tp,cloid)
            LOG.info(f"[PT] New pos {cloid} {side}@{price:.4f}, SL={sl:.4f}, TP={tp:.4f}")
        # 主动下 SL/TP 单
        await mgr.place(
            side=("SELL" if side=="BUY" else "BUY"),
            otype="STOP_MARKET",
            extra={"reduceOnly":"true","clientOrderId":f"sl_{cloid}"}
        )
        await mgr.place(
            side=("SELL" if side=="BUY" else "BUY"),
            otype="TAKE_PROFIT_MARKET",
            extra={"reduceOnly":"true","clientOrderId":f"tp_{cloid}"}
        )

    async def check(self, price):
        async with self.lock:
            for cloid, p in list(self.pos.items()):
                if not p.active: continue
                hit_sl = price <= p.sl if p.side=='BUY' else price >= p.sl
                hit_tp = price >= p.tp if p.side=='BUY' else price <= p.tp
                if hit_sl or hit_tp:
                    # 撤对手单
                    other_cid = f"{'tp' if hit_sl else 'sl'}_{cloid}"
                    await mgr.place(
                        side=("BUY" if p.side=='BUY' else "SELL"),
                        otype="STOP_MARKET",
                        extra={"cancelClientOrderId":other_cid}
                    )
                    # 市价平仓
                    await mgr.place(
                        side=("SELL" if p.side=='BUY' else "BUY"),
                        otype="MARKET", qty=p.qty
                    )
                    p.active = False
                    LOG.info(f"[PT] Closed {cloid} via {'SL' if hit_sl else 'TP'}")

pos_tracker = PositionTracker()

# —— 策略实现 ——
class MainStrategy:
    def __init__(self):
        self._last = 0; self.intv = 1

    async def check(self, price):
        now = time.time()
        if now-self._last<self.intv: return
        df = data_mgr.klines["15m"]
        df15 = data_mgr.klines["15m"]
        if len(df15)<99 or df15.adx.iat[-1]<=25: return
        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99): return
        h,l,c = df15.high.values, df15.low.values, df15.close.values
        st,sd = numba_supertrend(h,l,c,10,3)
        up = price>st[-1] and sd[-1]
        dn = price<st[-1] and not sd[-1]
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not(bb3<=0 or bb3>=1): return
        side = "BUY" if bb3<=0 else "SELL"
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong = bb1<0.2 or bb1>0.8

        qty = 0.12 if ((side=="BUY" and up) or (side=="SELL" and dn)) else 0.07
        if strong:
            levels = [0.0025,0.014,0.026,0.038,0.056]; sizes=[qty*0.2]*5
        else:
            if side=="BUY":
                levels = [-0.0155,-0.0255] if up else [-0.0075]
            else:
                levels = [0.0155,0.0255] if dn else [0.0075]
            sizes=[0.015]*len(levels)

        self._last = now
        for lvl, sz in zip(levels, sizes):
            p0 = price*(1+(lvl if side=="BUY" else -lvl))
            sl = price*(0.98 if side=="BUY" else 1.02)
            tp = price*(1.02 if side=="BUY" else 0.98)
            await mgr.safe_place("main", side, "LIMIT",
                                 qty=sz, price=p0,
                                 extra={'sl':sl,'tp':tp})
        # 防护止损
        slp = price*(0.98 if side=="BUY" else 1.02)
        await mgr.safe_place("main", side, "STOP_MARKET",
                             stop=slp, extra={'closePosition':'true'})

class MACDStrategy:
    def __init__(self): self._in=False
    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df)<30 or df.adx.iat[-1]<=25: return
        prev,curr = df.macd.iat[-2],df.macd.iat[-1]
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]
        if not(price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        if prev>0>curr and not self._in:
            sp=price*1.005
            await mgr.safe_place("macd","SELL","LIMIT",
                qty=0.06, price=sp, extra={'sl':sp*1.03,'tp':sp*0.97})
            self._in=True
        elif prev<0<curr and self._in:
            bp=price*0.995
            await mgr.safe_place("macd","BUY","LIMIT",
                qty=0.06, price=bp, extra={'sl':bp*0.97,'tp':bp*1.03})
            self._in=False

class TripleTrendStrategy:
    def __init__(self): self.last=0; self.active=False
    async def check(self, price):
        now=time.time()
        if now-self.last<1: return
        df1h=data_mgr.klines["1h"]
        if len(df1h)<15 or df1h.adx.iat[-1]<=25: return
        df15=data_mgr.klines["15m"]
        if len(df15)<99: return
        ma7,ma25,ma99 = df15.ma7.iat[-1],df15.ma25.iat[-1],df15.ma99.iat[-1]
        if not(price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        h,l,c = df15.high.values, df15.low.values, df15.close.values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not(d1[-1] or d2[-1] or d3[-1])
        prev=(d1[-2],d2[-2],d3[-2]); curr=(d1[-1],d2[-1],d3[-1])
        flip_dn = self.active and any(p and not c2 for p,c2 in zip(prev,curr))
        flip_up = self.active and any(not p and c2 for p,c2 in zip(prev,curr))

        self.last=now
        if up_all and not self.active:
            self.active=True
            p0,sl,tp = price*0.996, price*0.97, price*1.02
            await mgr.safe_place("triple","BUY","LIMIT",
                qty=0.015, price=p0, extra={'sl':sl,'tp':tp})
        elif dn_all and not self.active:
            self.active=True
            p0,sl,tp = price*1.004, price*1.03, price*0.98
            await mgr.safe_place("triple","SELL","LIMIT",
                qty=0.015, price=p0, extra={'sl':sl,'tp':tp})
        elif flip_dn:
            await mgr.safe_place("triple","SELL","MARKET"); self.active=False
        elif flip_up:
            await mgr.safe_place("triple","BUY","MARKET"); self.active=False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 主循环 ——
async def market_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(
                Config.WS_MARKET_URL, ping_interval=20, ping_timeout=60
            ) as ws:
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        await data_mgr.track_price(float(d["p"]), int(time.time()*1000))
                    else:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        await data_mgr.update_kline(
                            tf, float(k["o"]),float(k["h"]),
                            float(k["l"]),float(k["c"]),k["t"]
                        )
        except Exception as e:
            delay = min(2**retry,30); retry+=1
            LOG.error(f"[WS MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay)

async def trade_ws():
    await mgr.init_ws()
    retry=0
    while True:
        try:
            msg = await mgr.ws.recv()
            data = json.loads(msg)
            if data.get("result",{}).get("orderId"):
                await pos_tracker.on_fill(data["result"])
        except Exception as e:
            delay = min(2**retry,30); retry+=1
            LOG.error(f"[WS TRD] {e}, reconnect in {delay}s")
            await asyncio.sleep(delay)
            mgr.ready.clear()
            await mgr.init_ws()

async def engine():
    while data_mgr.price is None:
        await asyncio.sleep(0.5)
    while True:
        await data_mgr.wait_update()
        if time.time()-data_mgr.ptime>60: continue
        for strat in strategies:
            try: await strat.check(data_mgr.price)
            except: LOG.exception(f"策略 {strat} 错误")
        await pos_tracker.check(data_mgr.price)

async def main():
    global price_step, qty_step, price_prec, qty_prec
    await data_mgr.load_history()
    async with aiohttp.ClientSession() as sess:
        info = await (await sess.get(
            "https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10
        )).json()
        sym = next(s for s in info["symbols"] if s["symbol"]==Config.SYMBOL)
        pf = next(f for f in sym["filters"] if f["filterType"]=="PRICE_FILTER")
        ls = next(f for f in sym["filters"] if f["filterType"]=="LOT_SIZE")
        price_step, qty_step = float(pf["tickSize"]), float(ls["stepSize"])
        price_prec   = int(-math.log10(price_step)+0.5)
        qty_prec     = int(-math.log10(qty_step)+0.5)
    LOG.info(f"Precision loaded: price_step={price_step}, qty_step={qty_step}")

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(loop.shutdown_asyncgens()))
    await asyncio.gather(market_ws(), trade_ws(), engine())

if __name__=='__main__':
    asyncio.run(main())