#!/usr/bin/env python3
# coding: utf-8

import os, time, json, math, asyncio, logging, signal, urllib.parse, base64, hmac, hashlib
from collections import defaultdict

import uvloop, aiohttp, websockets, pandas as pd
from ta.trend import MACD
from numba import jit
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# —— 高性能事件循环 & 环境加载 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv('/root/zhibai/.env')

# —— 日志配置 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— 全局配置 ——
class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    API_KEY          = os.getenv('YZ_BINANCE_API_KEY')
    SECRET_KEY       = os.getenv('YZ_BINANCE_SECRET_KEY').encode()
    ED25519_API      = os.getenv('YZ_ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('YZ_ED25519_KEY_PATH')
    REST_BASE        = 'https://fapi.binance.com'
    WS_MARKET        = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_USER_BASE     = 'wss://fstream.binance.com/ws/'
    RECV_WINDOW      = 5000
    SYNC_INTERVAL    = 60
    ROTATION_CD      = 1800  # 同方向冷却 30 分钟

# —— 全局状态 ——
session     = None
listen_key  = None
time_offset = 0
is_hedge    = False

price_step = qty_step = None
price_prec = qty_prec = 0

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 key loaded")

def quantize(val: float, step: float) -> float:
    return math.floor(val/step) * step

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    data = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = data['serverTime'] - int(time.time() * 1000)
    LOG.debug(f"Time offset {time_offset}ms")

async def detect_mode():
    global is_hedge
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url= f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    data = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge = data.get('dualSidePosition', False)
    LOG.info(f"Hedge mode: {is_hedge}")

# —— 精度过滤 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    info = await (await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")).json()
    sym = next(s for s in info['symbols'] if s['symbol']==Config.SYMBOL)
    pf  = next(f for f in sym['filters'] if f['filterType']=='PRICE_FILTER')
    ls  = next(f for f in sym['filters'] if f['filterType']=='LOT_SIZE')
    price_step, qty_step = float(pf['tickSize']), float(ls['stepSize'])
    price_prec = int(round(-math.log10(price_step)))
    qty_prec   = int(round(-math.log10(qty_step)))
    LOG.info(f"Filters loaded: price_step={price_step}, qty_step={qty_step}")

# —— 下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def place(self, side, otype, qty=None, price=None, stop=None,
                    extra_params=None, return_id=False):
        await ensure_session()
        ts = int(time.time()*1000 + time_offset)
        params = {"symbol":Config.SYMBOL,"side":side,"type":otype,
                  "timestamp":ts,"recvWindow":Config.RECV_WINDOW}
        if qty is not None:    params["quantity"]   = f"{quantize(qty,qty_step):.{qty_prec}f}"
        if price is not None:  params["price"]      = f"{quantize(price,price_step):.{price_prec}f}"
        if stop is not None:   params["stopPrice"]  = f"{quantize(stop,price_step):.{price_prec}f}"
        if otype=="LIMIT":     params["timeInForce"] = "GTC"

        if otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            params.update({
                "workingType":  "MARK_PRICE",
                "priceProtect": "FALSE",
                "closePosition":"true"
            })
            if is_hedge:
                params["positionSide"] = "LONG" if side=="BUY" else "SHORT"
                params["reduceOnly"]   = "true"

        if is_hedge and otype in ("LIMIT","MARKET"):
            params["positionSide"] = "LONG" if side=="BUY" else "SHORT"

        if extra_params:
            params.update(extra_params)

        qs  = urllib.parse.urlencode(sorted(params.items()), safe=',')
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        headers = {'X-MBX-APIKEY':Config.API_KEY}

        async with self.lock:
            r = await session.post(url, headers=headers)
            data = await r.json()
            if r.status != 200:
                LOG.error(f"[Mgr] ERR {otype} {side}: {data}")
                return None
            oid = data.get('orderId')
            LOG.info(f"[Mgr] OK {otype} {side} oid={oid}")
            return oid if return_id else None

mgr = OrderManager()

async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 持仓 & SL/TP 管理 ——
class PositionTracker:
    class Pos:
        __slots__ = ('cloid','side','qty','sl','tp','sl_id','tp_id','active')
        def __init__(self, cloid, side, qty, sl, tp, sl_id, tp_id):
            self.cloid, self.side, self.qty = cloid, side, qty
            self.sl, self.tp = sl, tp
            self.sl_id, self.tp_id = sl_id, tp_id
            self.active = True

    def __init__(self):
        self.lock       = asyncio.Lock()
        self.next_cloid = 1
        self.positions  = {}
        self.ord2cloid  = {}

    async def on_fill(self, order, side, qty, price, sl, tp):
        async with self.lock:
            cloid = self.next_cloid; self.next_cloid += 1
            sl_q = quantize(sl, price_step)
            tp_q = quantize(tp, price_step)
            close = 'SELL' if side=='BUY' else 'BUY'
            sl_id = await mgr.place(close, 'STOP_MARKET', stop=sl_q, return_id=True)
            tp_id = await mgr.place(close, 'TAKE_PROFIT_MARKET', stop=tp_q, return_id=True)
            pos = self.Pos(cloid, side, qty, sl_q, tp_q, sl_id, tp_id)
            self.positions[cloid] = pos
            self.ord2cloid[order['orderId']] = cloid
            LOG.info(f"[PT] Open cloid={cloid} SL={sl_q}({sl_id}) TP={tp_q}({tp_id})")

    async def on_order_update(self, oid, status):
        async with self.lock:
            # SL/TP 远程触发
            for pos in self.positions.values():
                if oid in (pos.sl_id, pos.tp_id):
                    pos.active = False
                    LOG.info(f"[PT] Remote SL/TP triggered cloid={pos.cloid}")
                    return
            # 原始仓位单关闭
            cloid = self.ord2cloid.get(oid)
            if cloid and status in ('FILLED','CANCELED'):
                self.positions[cloid].active = False
                LOG.info(f"[PT] Remote position close cloid={cloid}")

    async def check_trigger(self, price):
        eps = price_step * 0.5
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active: continue
                hit_sl = (pos.side=='BUY' and price<=pos.sl+eps) or \
                         (pos.side=='SELL'and price>=pos.sl-eps)
                hit_tp = (pos.side=='BUY' and price>=pos.tp-eps) or \
                         (pos.side=='SELL'and price<=pos.tp+eps)
                if hit_sl or hit_tp:
                    LOG.info(f"[PT] Local trigger cloid={pos.cloid} price={price}")
                    # 撤 SL/TP 挂单
                    if pos.sl_id: await mgr.place('','','', extra_params={'orderId':pos.sl_id})
                    if pos.tp_id: await mgr.place('','','', extra_params={'orderId':pos.tp_id})
                    # 市价平仓
                    close = 'SELL' if pos.side=='BUY' else 'BUY'
                    await mgr.place(close, 'MARKET', qty=pos.qty)
                    pos.active = False

    async def sync(self):
        # 可选：同步远程持仓状况，目前留空
        pass

pos_tracker = PositionTracker()

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs     = ("3m","15m","1h")
        self.klines  = {tf:pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = {tf:0 for tf in self.tfs}
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.price   = None
        self.ptime   = 0

    async def load_history(self):
        async with self.lock:
            for tf in self.tfs:
                r = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                    params={"symbol":Config.SYMBOL,"interval":tf,"limit":1000},
                    headers={"X-MBX-APIKEY":Config.API_KEY})
                data = await r.json()
                df   = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),
                                      "low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf]   = df
                self.last_ts[tf]  = int(data[-1][0])
                self._compute(tf)
                LOG.info(f"[DM] {tf} loaded {len(df)} bars")

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            idx= len(df) if ts>self.last_ts[tf] else df.index[-1]
            df.loc[idx, ["open","high","low","close"]] = [o,h,l,c]
            self.last_ts[tf] = ts
            self._compute(tf)
            self._evt.set()

    async def track_price(self, p, ts):
        async with self.lock:
            self.price, self.ptime = p, ts
        self._evt.set()
        await pos_tracker.check_trigger(p)

    async def wait_update(self):
        await self._evt.wait(); self._evt.clear()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df)<20: return
        m = df.close.rolling(20).mean()
        s = df.close.rolling(20).std()
        df["bb_up"], df["bb_dn"] = m+2*s, m-2*s
        df["bb_pct"] = (df.close - df.bb_dn)/(df.bb_up - df.bb_dn)
        if tf=="15m":
            hl2 = (df.high+df.low)/2
            atr = df.high.rolling(10).max() - df.low.rolling(10).min()
            df["st"]   = hl2 - 3*atr
            df["macd"] = MACD(df.close,12,26,9).macd_diff()
            df["ma7"]  = df.close.rolling(7).mean()
            df["ma25"] = df.close.rolling(25).mean()
            df["ma99"] = df.close.rolling(99).mean()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n   = len(c)
    st  = [0.0]*n; dirc=[False]*n
    hl2 = [(h[i]+l[i])/2 for i in range(n)]
    atr = [max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up  = [hl2[i]+mult*atr[i] for i in range(n)]
    dn  = [hl2[i]-mult*atr[i] for i in range(n)]
    st[0],dirc[0] = up[0],True
    for i in range(1,n):
        if c[i]>st[i-1]:
            st[i],dirc[i] = max(dn[i],st[i-1]),True
        else:
            st[i],dirc[i] = min(up[i],st[i-1]),False
    return st,dirc

# —— 订单守卫 ——
class OrderGuard:
    def __init__(self):
        self.states   = defaultdict(lambda:{'ts':0,'trend':None,'fp':None})
        self.lock     = asyncio.Lock()
        self.cooldown = {
            'main':   Config.ROTATION_CD,
            'macd':   Config.ROTATION_CD,
            'triple': Config.ROTATION_CD
        }

    async def check(self, strat, fp, trend):
        async with self.lock:
            st  = self.states[strat]
            now = time.time()
            if trend==st['trend'] and now-st['ts']<self.cooldown[strat]:
                return False
            if st['fp']==fp:
                return False
            return True

    async def update(self, strat, fp, trend):
        async with self.lock:
            self.states[strat] = {'fp':fp,'trend':trend,'ts':time.time()}

guard = OrderGuard()

# —— 用户流 ListenKey & 保活 ——
async def get_listen_key():
    async with session.post(
        f"{Config.REST_BASE}/fapi/v1/listenKey",
        headers={'X-MBX-APIKEY':Config.ED25519_API}
    ) as resp:
        data = await resp.json()
        return data['listenKey']

async def keepalive_listen_key():
    global listen_key
    while True:
        await asyncio.sleep(1800)
        try:
            listen_key = await get_listen_key()
            LOG.info("ListenKey renewed")
        except Exception as e:
            LOG.error(f"listenKey renewal failed: {e}")

# —— 策略实现 ——
class MainStrategy:
    def __init__(self): self._last=0; self.interval=1
    async def check(self, price):
        now=time.time()
        if now-self._last<self.interval: return
        df=data_mgr.klines["15m"]
        if len(df)<30: return

        h,l,c = df.high.values,df.low.values,df.close.values
        st_line,st_dir = numba_supertrend(h,l,c,10,3)
        trend_up = price>st_line[-1] and st_dir[-1]
        trend_dn = price<st_line[-1] and not st_dir[-1]
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not(bb3<=0 or bb3>=1): return
        side="BUY" if bb3<=0 else "SELL"
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong = bb1<0.2 or bb1>0.8

        if strong:
            qty = 0.12 if ((side=="BUY" and trend_up) or (side=="SELL" and trend_dn)) else 0.07
            levels,sizes = [0.0025,0.004,0.006,0.008,0.016], [qty*0.2]*5
        else:
            if side=="BUY":
                levels,sizes = ([-0.0055,-0.0155],[0.015]) if trend_up else ([-0.0075],[0.015])
            else:
                levels,sizes = ([0.0055,0.0155],[0.015]) if trend_dn else ([0.0075],[0.015])

        self._last=now
        for lvl,sz in zip(levels,sizes):
            p0 = price*(1+lvl if side=="BUY" else 1-lvl)
            sl = price*(0.98 if side=="BUY" else 1.02)
            tp = price*(1.02 if side=="BUY" else 0.98)
            await mgr.place(side,"LIMIT",qty=sz,price=p0,extra_params={'sl':sl,'tp':tp})
        sl = price*(0.98 if side=="BUY" else 1.02)
        await mgr.place(side,"STOP_MARKET",stop=sl)

class MACDStrategy:
    def __init__(self): self._in=False
    async def check(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]
        if not(price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        prev,curr = df.macd.iat[-2],df.macd.iat[-1]
        if prev>0>curr and not self._in:
            sp=price*1.005; sl, tp = sp*1.03, sp*0.97
            await mgr.place("SELL","LIMIT",qty=0.06,price=sp,extra_params={'sl':sl,'tp':tp})
            self._in=True
        elif prev<0<curr and self._in:
            bp=price*0.995; sl, tp = bp*0.97, bp*1.03
            await mgr.place("BUY","LIMIT",qty=0.06,price=bp,extra_params={'sl':sl,'tp':tp})
            self._in=False

class TripleTrendStrategy:
    def __init__(self): self.round_active=False; self._last=0
    async def check(self, price):
        now=email=time.time()
        if now-self._last<1: return
        df=data_mgr.klines["15m"]
        if len(df)<99: return
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]
        if not(price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        h,l,c=df.high.values,df.low.values,df.close.values
        _,d1=numba_supertrend(h,l,c,10,1)
        _,d2=numba_supertrend(h,l,c,11,2)
        _,d3=numba_supertrend(h,l,c,12,3)
        up_all=d1[-1] and d2[-1] and d3[-1]
        dn_all=not (d1[-1] or d2[-1] or d3[-1])
        prev,curr=(d1[-2],d2[-2],d3[-2]),(d1[-1],d2[-1],d3[-1])
        flip_dn=self.round_active and any(p and not c2 for p,c2 in zip(prev,curr))
        flip_up=self.round_active and any(not p and c2 for p,c2 in zip(prev,curr))
        self._last=now

        if up_all and not self.round_active:
            p0,sl,tp=price*0.999,price*0.97,price*1.03
            await mgr.place("BUY","LIMIT",qty=0.05,price=p0,extra_params={'sl':sl,'tp':tp})
            self.round_active=True
        elif dn_all and not self.round_active:
            p0,sl,tp=price*1.001,price*1.03,price*0.98
            await mgr.place("SELL","LIMIT",qty=0.05,price=p0,extra_params={'sl':sl,'tp':tp})
            self.round_active=True
        elif flip_dn:
            await mgr.place("SELL","MARKET"); self.round_active=False
        elif flip_up:
            await mgr.place("BUY","MARKET"); self.round_active=False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 运行逻辑 ——
async def market_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        await data_mgr.track_price(float(d["p"]), int(time.time()*1000))
                    else:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        await data_mgr.update_kline(tf,
                            float(k["o"]),float(k["h"]),float(k["l"]),float(k["c"]),k["t"])
        except Exception as e:
            delay=min(2**retry,30)
            LOG.error(f"[WS MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay); retry+=1

async def user_ws():
    global listen_key
    retry=0
    while True:
        try:
            listen_key=await get_listen_key()
            async with websockets.connect(Config.WS_USER_BASE+listen_key) as ws:
                LOG.info(f"User stream connected")
                async for msg in ws:
                    data=json.loads(msg)
                    if data.get('e')=='ORDER_TRADE_UPDATE':
                        o=data['o']
                        await pos_tracker.on_order_update(o['i'], o['X'])
        except Exception as e:
            LOG.error(f"[WS USER] {e}, reconnect in 5s")
            await asyncio.sleep(5); retry+=1

async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode(); await pos_tracker.sync()
        if data_mgr.price is not None:
            await pos_tracker.check_trigger(data_mgr.price)

async def engine():
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60:
            continue
        for strat in strategies:
            try:
                await strat.check(data_mgr.price)
            except Exception:
                LOG.exception(f"Strat {strat.__class__.__name__} failed")

async def main():
    global session
    session = aiohttp.ClientSession()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    await sync_time(); await detect_mode(); await load_symbol_filters()
    await data_mgr.load_history(); await pos_tracker.sync()

    await asyncio.gather(
        market_ws(),
        user_ws(),
        maintenance(),
        engine(),
        keepalive_listen_key()
    )

if __name__=='__main__':
    asyncio.run(main())
