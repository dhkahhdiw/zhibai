#!/usr/bin/env python3
# coding: utf-8

import os, time, json, uuid, math, base64, hmac, hashlib, asyncio, logging, signal, urllib.parse
from collections import defaultdict

import uvloop, aiohttp, websockets, pandas as pd
from ta.trend import MACD
from numba import jit
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# —— 事件循环 & 环境 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv('/root/zhibai/.env')

# —— 日志 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— 配置 ——
class Config:
    SYMBOL       = 'ETHUSDC'
    PAIR         = SYMBOL.lower()
    API_KEY      = os.getenv('BINANCE_API_KEY')
    SECRET_KEY   = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API  = os.getenv('ED25519_API_KEY')
    ED25519_KEY  = os.getenv('ED25519_KEY_PATH')
    REST_BASE    = 'https://fapi.binance.com'
    WS_MARKET    = f"wss://fstream.binance.com/stream?streams={PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    WS_USER      = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    RECV_WINDOW  = 5000
    SYNC_INTERVAL= 300

# —— 全局 ——
session: aiohttp.ClientSession = None
time_offset = 0
is_hedge = False
price_step = qty_step = None
price_prec = qty_prec = 0

# —— 加载私钥 ——
with open(Config.ED25519_KEY, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 key loaded")

def quantize(val: float, step: float) -> float:
    return math.floor(val/step) * step

# —— 时间 & 模式 ——
async def sync_time():
    global time_offset
    data = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = data['serverTime'] - int(time.time()*1000)
    LOG.info(f"Time offset {time_offset}ms")

async def detect_mode():
    global is_hedge
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    r = await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})
    data = await r.json()
    is_hedge = data.get('dualSidePosition', False)
    LOG.info(f"Hedge mode: {is_hedge}")

# —— 精度过滤 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    info = await (await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")).json()
    sym = next(s for s in info['symbols'] if s['symbol']==Config.SYMBOL)
    pf = next(f for f in sym['filters'] if f['filterType']=='PRICE_FILTER')
    ls = next(f for f in sym['filters'] if f['filterType']=='LOT_SIZE')
    price_step, qty_step = float(pf['tickSize']), float(ls['stepSize'])
    price_prec = int(-math.log10(price_step))
    qty_prec   = int(-math.log10(qty_step))
    LOG.info(f"Filters loaded: price_step={price_step}, qty_step={qty_step}")

# —— 持仓跟踪 & 双重止盈止损 ——
class PositionTracker:
    class Position:
        __slots__ = ('cloid','side','qty','entry_price','sl_price','tp_price','active')
        def __init__(self, cloid, side, qty, entry, sl, tp):
            self.cloid = cloid; self.side = side; self.qty = qty
            self.entry_price = entry; self.sl_price = sl; self.tp_price = tp
            self.active = True

    def __init__(self):
        self.positions = {}  # cloid -> Position
        self.orders    = {}  # order_id -> cloid
        self.lock      = asyncio.Lock()
        self.next_cloid= 1

    async def on_fill(self, order_id, side, qty, price, sl, tp):
        """成交后：记录+下远程止损止盈"""
        async with self.lock:
            cloid = self.next_cloid; self.next_cloid +=1
            pos = self.Position(cloid, side, qty, price, sl, tp)
            self.positions[cloid] = pos
            self.orders[order_id] = cloid
            LOG.info(f"[PT] Opened cloid={cloid}, side={side}, qty={qty}, entry={price}, SL={sl}, TP={tp}")
        # 远程止损
        close_side = 'SELL' if side=='LONG' else 'BUY'
        await mgr.place(close_side, 'STOP_MARKET', qty=qty, stop=sl, extra_params={'closePosition':'true'})
        await mgr.place(close_side, 'TAKE_PROFIT_MARKET', qty=qty, stop=tp, extra_params={'closePosition':'true'})

    async def on_order_update(self, order_id, status):
        async with self.lock:
            if order_id not in self.orders: return
            cloid = self.orders[order_id]; pos = self.positions.get(cloid)
            if not pos: return
            if status in ('FILLED','CANCELED'):
                pos.active = False
                LOG.info(f"[PT] Closed cloid={cloid} via remote update")

    async def check_trigger(self, price):
        """本地监控触发"""
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active: continue
                if (pos.side=='LONG'  and (price <= pos.sl_price or price >= pos.tp_price)) or \
                   (pos.side=='SHORT' and (price >= pos.sl_price or price <= pos.tp_price)):
                    await self.close_position(pos, price)

    async def close_position(self, pos, trigger_price):
        side = 'SELL' if pos.side=='LONG' else 'BUY'
        LOG.info(f"[PT] Local trigger close cloid={pos.cloid} via MARKET {side} @price={trigger_price}")
        try:
            await mgr.place(side, 'MARKET', qty=pos.qty, extra_params={'closePosition':'true'})
            pos.active = False
        except Exception as e:
            LOG.error(f"[PT] close_position failed: {e}")

    async def sync(self):
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
        sig= hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res = await (await session.get(url,headers={'X-MBX-APIKEY':Config.API_KEY})).json()
        for p in res:
            if p['symbol']==Config.SYMBOL:
                LOG.debug(f"[PT] Remote pos: {p['positionAmt']} @{p.get('positionSide')}")

pos_tracker = PositionTracker()

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs = ("3m","15m","1h")
        self.klines  = {tf: pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = {tf:0 for tf in self.tfs}
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.price, self.ptime = None,0

    async def load_history(self):
        async with self.lock:
            for tf in self.tfs:
                res = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                    params={"symbol":Config.SYMBOL,"interval":tf,"limit":1000},
                    headers={"X-MBX-APIKEY":Config.API_KEY})
                data = await res.json()
                df = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),
                                    "low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf]=df; self.last_ts[tf]=int(data[-1][0]); self._compute(tf)
                LOG.info(f"[DM] {tf} loaded {len(df)} bars")

    async def update_kline(self, tf,o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            if ts>self.last_ts[tf]:
                idx=len(df); df.loc[idx,["open","high","low","close"]]=[o,h,l,c]
            else:
                idx=df.index[-1]; df.loc[idx,["open","high","low","close"]]=[o,h,l,c]
            self.last_ts[tf]=ts; self._compute(tf); self._evt.set()

    async def track_price(self,p,ts):
        async with self.lock:
            self.price, self.ptime = p, ts
        self._evt.set()
        await pos_tracker.check_trigger(p)

    async def wait_update(self):
        await self._evt.wait(); self._evt.clear()

    def _compute(self, tf):
        df=self.klines[tf]
        if len(df)<20: return
        m, s = df.close.rolling(20).mean(), df.close.rolling(20).std()
        df["bb_up"],df["bb_dn"]=m+2*s,m-2*s
        df["bb_pct"]=(df.close-df.bb_dn)/(df.bb_up-df.bb_dn)
        if tf=="15m":
            hl2=(df.high+df.low)/2
            atr=df.high.rolling(10).max()-df.low.rolling(10).min()
            df["st"]=hl2-3*atr
            df["macd"]=MACD(df.close,12,26,9).macd_diff()
            df["ma7"],df["ma25"],df["ma99"]=df.close.rolling(7).mean(),df.close.rolling(25).mean(),df.close.rolling(99).mean()

data_mgr = DataManager()

# —— Supertrend ——
@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n=len(c); st=[0.0]*n; dirc=[False]*n
    hl2=[(h[i]+l[i])/2 for i in range(n)]
    atr=[max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up=[hl2[i]+mult*atr[i] for i in range(n)]
    dn=[hl2[i]-mult*atr[i] for i in range(n)]
    st[0],dirc[0]=up[0],True
    for i in range(1,n):
        if c[i]>st[i-1]:
            st[i],dirc[i]=max(dn[i],st[i-1]),True
        else:
            st[i],dirc[i]=min(up[i],st[i-1]),False
    return st,dirc

# —— 订单守卫 ——
class OrderGuard:
    def __init__(self):
        self.states, self.lock = defaultdict(dict), asyncio.Lock()
        self.cooldown = {"main":60,"macd":30,"triple":300}
    async def check(self,strat,fp):
        async with self.lock:
            st=self.states[strat]; now=time.time()
            if st.get('fp')==fp and now-st.get('ts',0)<self.cooldown[strat]: return False
            if now-st.get('ts',0)<self.cooldown[strat]: return False
            return True
    async def update(self,strat,fp):
        async with self.lock:
            self.states[strat]={'fp':fp,'ts':time.time()}

guard = OrderGuard()

async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 下单管理 ——
class OrderManager:
    def __init__(self): self.lock=asyncio.Lock()
    async def safe_place(self,strat,side,otype,qty=None,price=None,stop=None,extra_params=None):
        fp=f"{strat}|{side}|{otype}|{hash(frozenset((extra_params or {}).items()))}"
        if not await guard.check(strat,fp): return
        await self.place(side,otype,qty,price,stop,sig_key=fp,extra_params=extra_params or {})
        await guard.update(strat,fp)

    async def place(self,side,otype,qty=None,price=None,stop=None,
                    sig_key=None,extra_params=None):
        await ensure_session()
        ts=int(time.time()*1000+time_offset)
        params={"symbol":Config.SYMBOL,"side":side,"type":otype,
                "timestamp":ts,"recvWindow":Config.RECV_WINDOW}
        if qty   is not None: params["quantity"]=f"{quantize(qty,qty_step):.{qty_prec}f}"
        if price is not None: params["price"]=f"{quantize(price,price_step):.{price_prec}f}"
        if stop  is not None: params["stopPrice"]=f"{quantize(stop,price_step):.{price_prec}f}"
        if otype=="LIMIT":    params["timeInForce"]="GTC"
        if otype in ("STOP_MARKET","TAKE_PROFIT_MARKET","MARKET"):
            params["closePosition"]="true"
        if is_hedge and otype in ("LIMIT","MARKET"):
            params["positionSide"]="LONG" if side=="BUY" else "SHORT"
        params.update(extra_params or {})

        qs=urllib.parse.urlencode(sorted(params.items()))
        sig=hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
        url=f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        headers={'X-MBX-APIKEY':Config.API_KEY}
        async with self.lock:
            r=await session.post(url,headers=headers)
            text=await r.text()
            tag=f"[{sig_key}]" if sig_key else ""
            if r.status==200:
                data=await r.json()
                LOG.info(f"Order OK {otype} {side} {tag}")
                # 仅限价策略单成交后 on_fill
                if sig_key and otype=="LIMIT":
                    exec_qty=float(data.get('executedQty',0))
                    sl=extra_params.get('sl'); tp=extra_params.get('tp')
                    if exec_qty>0 and sl is not None and tp is not None:
                        await pos_tracker.on_fill(data['orderId'], side, exec_qty,
                                                  float(data.get('price',0)), sl, tp)
            else:
                LOG.error(f"Order ERR {otype} {side} {tag}: {text}")

mgr = OrderManager()

# —— 策略：TripleTrend ——
class TripleTrendStrategy:
    def __init__(self):
        self.round_active = False
        self.entry_price = None
        self._last = 0

    async def check(self, price):
        now = time.time()
        if now - self._last < 1:
            return
        df = data_mgr.klines["15m"]
        if len(df) < 99:
            return
        h, l, c = df.high.values, df.low.values, df.close.values
        try:
            _, d1 = numba_supertrend(h, l, c, 10, 1)
            _, d2 = numba_supertrend(h, l, c, 11, 2)
            _, d3 = numba_supertrend(h, l, c, 12, 3)
        except Exception:
            LOG.error("Numba supertrend error in TripleTrendStrategy", exc_info=True)
            return

        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not (d1[-1] or d2[-1] or d3[-1])
        prev = (d1[-2], d2[-2], d3[-2])
        curr = (d1[-1], d2[-1], d3[-1])
        flip_to_dn = self.round_active and any(p and not c2 for p, c2 in zip(prev, curr))
        flip_to_up = self.round_active and any(not p and c2 for p, c2 in zip(prev, curr))

        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]

        if up_all and not self.round_active and price < ma7 < ma25 < ma99:
            self._last = now
            self.round_active = True
            self.entry_price = price
            p0 = price * 0.995
            LOG.info("TRIPLE → LIMIT BUY @%.2f", p0)
            await mgr.safe_place("triple", "BUY", "LIMIT", qty=0.15, price=p0, extra_params={'sl': price*0.97, 'tp': price*1.02})
        elif dn_all and not self.round_active and price > ma7 > ma25 > ma99:
            self._last = now
            self.round_active = True
            self.entry_price = price
            p0 = price * 1.005
            LOG.info("TRIPLE → LIMIT SELL @%.2f", p0)
            await mgr.safe_place("triple", "SELL", "LIMIT", qty=0.15, price=p0, extra_params={'sl': price*1.03, 'tp': price*0.98})
        elif flip_to_dn:
            LOG.info("TRIPLE take profit long → MARKET SELL")
            await mgr.safe_place("triple", "SELL", "MARKET")
            self.round_active = False
        elif flip_to_up:
            LOG.info("TRIPLE take profit short → MARKET BUY")
            await mgr.safe_place("triple", "BUY", "MARKET")
            self.round_active = False

# —— 策略：Main ——
class MainStrategy:
    def __init__(self):
        self._last = 0
        self.interval = 1

    async def check(self, price):
        now = time.time()
        if now - self._last < self.interval:
            return
        df15 = data_mgr.klines["15m"]
        if len(df15) < 99:
            return
        h15, l15, c15 = df15.high.values, df15.low.values, df15.close.values
        try:
            st_line, st_dir = numba_supertrend(h15, l15, c15, 10, 3)
        except Exception:
            LOG.error("Numba supertrend error in MainStrategy", exc_info=True)
            return
        trend_up = price > st_line[-1] and st_dir[-1]
        trend_dn = price < st_line[-1] and not st_dir[-1]

        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong_long  = bb1 < 0.2
        strong_short = bb1 > 0.8
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not (bb3 <= 0 or bb3 >= 1):
            return

        side = "BUY" if bb3 <= 0 else "SELL"
        # 挂单层级与手数
        if strong_long or strong_short:
            qty = 0.12 if ((side=="BUY" and trend_up) or (side=="SELL" and trend_dn)) else 0.07
            levels = [0.0025,0.004,0.006,0.008,0.016]
            sizes  = [qty*0.2]*5
        else:
            if side=="BUY":
                if trend_up:
                    levels, sizes = [-0.0055,-0.0155], [0.03*0.5]*2
                else:
                    levels, sizes = [-0.0075], [0.015]
            else:
                if trend_dn:
                    levels, sizes = [0.0055,0.0155], [0.03*0.5]*2
                else:
                    levels, sizes = [0.0075], [0.015]

        self._last = now
        # 挂网格入场
        for lvl, sz in zip(levels, sizes):
            p0 = price * (1+lvl if side=="BUY" else 1-lvl)
            await mgr.safe_place("main", side, "LIMIT", qty=sz, price=p0, extra_params={'sl': price*0.98 if side=="BUY" else price*1.02, 'tp': price*(1+0.01 if side=="BUY" else 1-0.01)})
        # 止损单
        sl = price*0.98 if side=="BUY" else price*1.02
        await mgr.safe_place("main", side, "STOP_MARKET", stop=sl)

# —— 策略：MACD ——
class MACDStrategy:
    def __init__(self):
        self._in = False

    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df) < 30 or "macd" not in df:
            return
        prev, curr = df.macd.iat[-2], df.macd.iat[-1]
        if prev > 0 > curr and not self._in:
            sp = price * 1.005
            LOG.info("MACD death cross SELL")
            await mgr.safe_place("macd", "SELL", "LIMIT", qty=0.15, price=sp, extra_params={'sl': sp*1.03, 'tp': sp*0.97})
            self._in = True
        elif prev < 0 < curr and self._in:
            bp = price * 0.995
            LOG.info("MACD golden cross BUY")
            await mgr.safe_place("macd", "BUY", "LIMIT", qty=0.15, price=bp, extra_params={'sl': bp*0.97, 'tp': bp*1.03})
            self._in = False

strategies = [ MainStrategy(), MACDStrategy(), TripleTrendStrategy() ]

# —— 市场 WS ——
async def market_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        await data_mgr.track_price(float(d["p"]),int(time.time()*1000))
                    else:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        await data_mgr.update_kline(tf,
                            float(k["o"]),float(k["h"]),float(k["l"]),float(k["c"]),k["t"]
                        )
        except Exception as e:
            delay=min(2**retry,30); LOG.error(f"[WS MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay); retry+=1

# —— 用户 WS ——
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
                    method=r.get("method") or r.get("e")
                    if method in("executionReport","order.update"):
                        o=r.get("params",r.get("result",r.get("o",{})))
                        oid=o.get('orderId') or o.get('i')
                        st=o.get('status') or o.get('X')
                        await pos_tracker.on_order_update(oid,st)
        except Exception as e:
            LOG.error(f"[WS USER] {e}, reconnect in 5s"); await asyncio.sleep(5); retry+=1

# —— 维护 & 执行 ——
async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode(); await pos_tracker.sync()

async def engine():
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60: continue
        for strat in strategies:
            try: await strat.check(data_mgr.price)
            except Exception: LOG.exception(f"Strat {strat.__class__.__name__} failed")

async def main():
    global session
    session = aiohttp.ClientSession()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    await sync_time(); await detect_mode(); await load_symbol_filters()
    await data_mgr.load_history(); await pos_tracker.sync()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())