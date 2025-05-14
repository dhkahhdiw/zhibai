#!/usr/bin/env python3
# coding: utf-8

import os, time, json, math, hmac, hashlib, asyncio, logging, signal, urllib.parse
from collections import defaultdict

import uvloop, aiohttp, websockets, pandas as pd
from ta.trend import MACD, ADXIndicator
from numba import jit
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# —— 高性能事件循环 & 环境加载 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv('/root/zhibai/.env')

# —— 日志配置 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— 全局配置 ——
class Config:
    SYMBOL            = 'ETHUSDC'
    PAIR              = SYMBOL.lower()
    API_KEY           = os.getenv('YZ_BINANCE_API_KEY')
    SECRET_KEY        = os.getenv('YZ_BINANCE_SECRET_KEY').encode()
    ED25519_API       = os.getenv('YZ_ED25519_API_KEY')
    ED25519_KEY_PATH  = os.getenv('YZ_ED25519_KEY_PATH')
    REST_BASE         = 'https://fapi.binance.com'
    WS_MARKET         = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_USER_BASE      = 'wss://fstream.binance.com/ws/'
    RECV_WINDOW       = 5000
    SYNC_INTERVAL     = 60
    ROTATION_COOLDOWN = 1800  # 30 分钟

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
    data = (await (await session.get(f"{Config.REST_BASE}/fapi/v1/time", timeout=5)).json())
    time_offset = data['serverTime'] - int(time.time() * 1000)
    LOG.debug(f"Time offset {time_offset}ms")

async def detect_mode():
    global is_hedge
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url= f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    data = (await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY}, timeout=5)).json())
    is_hedge = data.get('dualSidePosition', False)
    LOG.debug(f"Hedge mode: {is_hedge}")

# —— 精度过滤 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    info = (await (await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo", timeout=5)).json())
    sym = next(s for s in info['symbols'] if s['symbol']==Config.SYMBOL)
    pf  = next(f for f in sym['filters'] if f['filterType']=='PRICE_FILTER')
    ls  = next(f for f in sym['filters'] if f['filterType']=='LOT_SIZE')
    price_step, qty_step = float(pf['tickSize']), float(ls['stepSize'])
    price_prec = int(-math.log10(price_step) + 0.5)
    qty_prec   = int(-math.log10(qty_step) + 0.5)
    LOG.debug(f"Filters loaded: price_step={price_step}, qty_step={qty_step}")

# —— 持仓 & 止盈止损 ——
class PositionTracker:
    class Position:
        __slots__ = ('cloid','side','qty','sl_price','tp_price','active')
        def __init__(self, cloid, side, qty, sl, tp):
            self.cloid    = cloid
            self.side     = side
            self.qty      = qty
            self.sl_price = sl
            self.tp_price = tp
            self.active   = True

    def __init__(self):
        self.positions = {}; self.orders = {}
        self.lock      = asyncio.Lock(); self.next_id = 1

    async def on_fill(self, oid, side, qty, price, sl, tp):
        async with self.lock:
            cid = self.next_id; self.next_id += 1
            sl_q = quantize(sl, price_step)
            tp_q = quantize(tp, price_step)
            pos = self.Position(cid, side, qty, sl_q, tp_q)
            self.positions[cid] = pos
            self.orders[oid]    = cid
            LOG.info(f"[PT] Opened {cid} {side}@{price:.4f}, SL={sl_q:.4f}, TP={tp_q:.4f}")

        # 下止损
        try:
            await mgr.place(
                'SELL' if side=='BUY' else 'BUY',
                'STOP_MARKET', stop=sl_q,
                extra_params={
                    'closePosition':'true',
                    'workingType':'MARK_PRICE',
                    'priceProtect':'FALSE',
                    **({'positionSide':'LONG' if side=='BUY' else 'SHORT','reduceOnly':'true'} if is_hedge else {})
                }
            )
        except Exception as e:
            LOG.error(f"[PT] Failed to send SL: {e}")

        # 下止盈
        try:
            await mgr.place(
                'SELL' if side=='BUY' else 'BUY',
                'TAKE_PROFIT_MARKET', stop=tp_q,
                extra_params={
                    'closePosition':'true',
                    'workingType':'MARK_PRICE',
                    'priceProtect':'FALSE',
                    **({'positionSide':'LONG' if side=='BUY' else 'SHORT','reduceOnly':'true'} if is_hedge else {})
                }
            )
        except Exception as e:
            LOG.error(f"[PT] Failed to send TP: {e}")

    async def on_order_update(self, oid, status):
        async with self.lock:
            if oid not in self.orders: return
            cid = self.orders[oid]
            if status in ('FILLED','CANCELED'):
                self.positions[cid].active = False
                LOG.info(f"[PT] Closed cloid={cid} via {status}")

    async def check_trigger(self, price):
        eps = price_step * 0.5
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active: continue
                hit_sl = (price <= pos.sl_price+eps if pos.side=='BUY' else price >= pos.sl_price-eps)
                hit_tp = (price >= pos.tp_price-eps if pos.side=='BUY' else price <= pos.tp_price+eps)
                if hit_sl or hit_tp:
                    LOG.info(f"[PT] Local close cloid={pos.cloid} at {price:.4f}")
                    await mgr.place(
                        'SELL' if pos.side=='BUY' else 'BUY',
                        'MARKET', qty=pos.qty
                    )
                    pos.active = False

    async def sync(self):
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
        sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url= f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res= await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY}, timeout=5)).json()
        for p in res:
            if p['symbol']==Config.SYMBOL:
                LOG.debug(f"[PT] Remote pos amt={p['positionAmt']}")

pos_tracker = PositionTracker()

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs     = ("3m","15m","1h")
        self.klines  = {tf:pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = dict.fromkeys(self.tfs, 0)
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.price   = None
        self.ptime   = 0

    async def load_history(self):
        async with self.lock:
            for tf in self.tfs:
                params = {"symbol":Config.SYMBOL,"interval":tf,"limit":1000}
                hdrs   = {"X-MBX-APIKEY":Config.API_KEY}
                data   = await (await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                            params=params, headers=hdrs, timeout=5)).json()
                df     = pd.DataFrame([{
                    "open":float(x[1]),"high":float(x[2]),
                    "low": float(x[3]),"close":float(x[4])
                } for x in data])
                self.klines[tf]   = df
                self.last_ts[tf] = int(data[-1][0])
                self._compute(tf)

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df  = self.klines[tf]
            idx = len(df) if ts>self.last_ts[tf] else df.index[-1]
            df.loc[idx,["open","high","low","close"]] = [o,h,l,c]
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
        # 布林带
        m = df.close.rolling(20).mean(); s = df.close.rolling(20).std()
        df["bb_up"], df["bb_dn"] = m+2*s, m-2*s
        df["bb_pct"]             = (df.close - df.bb_dn) / (df.bb_up - df.bb_dn)
        # ADX
        adx = ADXIndicator(df.high, df.low, df.close, window=14)
        df["adx"], df["dmp"], df["dmn"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
        if tf=="15m":
            df["st"]    = (df.high+df.low)/2 - 3*(df.high.rolling(10).max()-df.low.rolling(10).min())
            df["macd"]  = MACD(df.close,12,26,9).macd_diff()
            df["ma7"]   = df.close.rolling(7).mean()
            df["ma25"]  = df.close.rolling(25).mean()
            df["ma99"]  = df.close.rolling(99).mean()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n = len(c)
    st = [0.0]*n; dirc=[False]*n
    hl2= [(h[i]+l[i])/2 for i in range(n)]
    atr= [max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up = [hl2[i] + mult*atr[i] for i in range(n)]
    dn = [hl2[i] - mult*atr[i] for i in range(n)]
    st[0],dirc[0] = up[0], True
    for i in range(1,n):
        if c[i] > st[i-1]:
            st[i],dirc[i] = max(dn[i], st[i-1]), True
        else:
            st[i],dirc[i] = min(up[i], st[i-1]), False
    return st, dirc

# —— 订单守卫 ——
class OrderGuard:
    def __init__(self):
        self.states   = defaultdict(lambda:{'ts':0,'fp':None,'trend':None})
        self.lock     = asyncio.Lock()
        self.cooldown = dict.fromkeys(('main','macd','triple'), Config.ROTATION_COOLDOWN)

    async def check(self, strat, fp, trend):
        async with self.lock:
            st = self.states[strat]
            now= time.time()
            if st['fp']==fp or (st['trend']==trend and now-st['ts']<self.cooldown[strat]):
                return False
            return True

    async def update(self, strat, fp, trend):
        async with self.lock:
            self.states[strat] = {'fp':fp,'trend':trend,'ts':time.time()}

guard = OrderGuard()

async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 用户流 ListenKey 与 Keepalive ——
async def get_listen_key():
    data = await (await session.post(f"{Config.REST_BASE}/fapi/v1/listenKey",
                headers={'X-MBX-APIKEY':Config.API_KEY}, timeout=5)).json()
    return data['listenKey']

async def keepalive_listen_key():
    global listen_key
    while True:
        await asyncio.sleep(1800)
        try:
            listen_key = await get_listen_key()
            LOG.debug("ListenKey renewed")
        except Exception as e:
            LOG.error(f"Keepalive failed: {e}")

# —— 下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, extra_params=None):
        fp    = f"{strat}|{side}|{otype}|{hash(frozenset((extra_params or {}).items()))}"
        trend = 'LONG' if side=='BUY' else 'SHORT'
        if not await guard.check(strat, fp, trend):
            return
        await self.place(side, otype, qty, price, stop, sig_key=fp, extra_params=extra_params or {})
        await guard.update(strat, fp, trend)

    async def place(self, side, otype, qty=None, price=None, stop=None, sig_key=None, extra_params=None):
        await ensure_session()
        ts     = int(time.time()*1000 + time_offset)
        params = {"symbol":Config.SYMBOL,"side":side,"type":otype,
                  "timestamp":ts,"recvWindow":Config.RECV_WINDOW}

        # 普通参数
        if otype=="LIMIT":
            if qty is not None:
                params["quantity"] = f"{quantize(qty,qty_step):.{qty_prec}f}"
            if price is not None:
                params["price"] = f"{quantize(price,price_step):.{price_prec}f}"
            params["timeInForce"]="GTC"
        elif otype in ("MARKET",):
            if qty is not None:
                params["quantity"] = f"{quantize(qty,qty_step):.{qty_prec}f}"
        elif otype in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            # stopPrice 与 extra_params 中的 closePosition 互斥 quantity
            params["stopPrice"] = f"{quantize(stop,price_step):.{price_prec}f}"
            params["workingType"] = "MARK_PRICE"
            params["priceProtect"] = "FALSE"
            if extra_params.get("closePosition")=='true':
                # 挂全平单
                params["closePosition"]="true"
            else:
                # 按 qty 平仓
                params["quantity"] = f"{quantize(qty,qty_step):.{qty_prec}f}"
            extra_params.pop("closePosition",None)

        # Hedge 模式下增加 reduceOnly / positionSide
        if is_hedge and otype in ("LIMIT","MARKET","STOP_MARKET","TAKE_PROFIT_MARKET"):
            params["positionSide"] = "LONG" if side=="BUY" else "SHORT"
            if otype!="LIMIT":
                params["reduceOnly"] = "true"

        # 合并额外参数（sl/tp 仅做记录传递）
        if extra_params:
            params.update(extra_params)

        # 签名与请求
        qs  = urllib.parse.urlencode(sorted(params.items()))
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"

        async with self.lock:
            r    = await session.post(url, headers={'X-MBX-APIKEY':Config.API_KEY}, timeout=5)
            text = await r.text()
            try:
                data = await r.json()
            except:
                LOG.error(f"[Mgr] Non-JSON resp: {text}")
                return

            if r.status != 200:
                LOG.error(f"[Mgr] ERR {otype} {side} {r.status}: {data}")
            else:
                LOG.debug(f"[Mgr] OK  {otype} {side}: {data}")
                # 仅 LIMIT 单成交回调开仓
                if sig_key and otype=="LIMIT" and float(data.get('executedQty',0))>0:
                    await pos_tracker.on_fill(
                        data['orderId'], side,
                        float(data['executedQty']),
                        float(data.get('price',0)),
                        extra_params.get('sl'),
                        extra_params.get('tp')
                    )

mgr = OrderManager()

# —— 策略实现 ——
class MainStrategy:
    def __init__(self):
        self._last = 0; self.interval = 1

    async def check(self, price):
        now = time.time()
        if now - self._last < self.interval: return
        df15 = data_mgr.klines["15m"]
        if len(df15) < 99: return

        if df15['adx'].iat[-1] <= 35:
            return

        h,l,c = df15.high.values, df15.low.values, df15.close.values
        st,sd = numba_supertrend(h,l,c,10,3)
        trend_up = price > st[-1] and sd[-1]
        trend_dn = price < st[-1] and not sd[-1]

        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not (bb3 <= 0 or bb3 >= 1): return
        side = "BUY" if bb3 <= 0 else "SELL"

        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong = bb1 < 0.2 or bb1 > 0.8

        qty = 0.12 if ((side=="BUY" and trend_up) or (side=="SELL" and trend_dn)) else 0.07
        if strong:
            levels = [0.0025,0.004,0.006,0.008,0.016]
            sizes  = [qty*0.2]*5
        else:
            if side=="BUY":
                levels = [-0.0055,-0.0155] if trend_up else [-0.0075]
            else:
                levels = [0.0055,0.0155] if trend_dn else [0.0075]
            sizes = [0.015]*len(levels)

        self._last = now
        for lvl, sz in zip(levels, sizes):
            p0 = price * (1 + (lvl if side=="BUY" else -lvl))
            sl = price * (0.98 if side=="BUY" else 1.02)
            tp = price * (1.02 if side=="BUY" else 0.98)
            await mgr.safe_place("main", side, "LIMIT", qty=sz, price=p0, extra_params={'sl':sl,'tp':tp})

        # 防护挂单
        sl = price * (0.98 if side=="BUY" else 1.02)
        await mgr.safe_place("main", side, "STOP_MARKET", stop=sl, extra_params={'closePosition':'true'})

class MACDStrategy:
    def __init__(self):
        self._in = False

    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df) < 30 or "macd" not in df.columns: return
        if df['adx'].iat[-1] <= 25: return

        prev, curr = df.macd.iat[-2], df.macd.iat[-1]
        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return

        if prev>0>curr and not self._in:
            sp = price * 1.005
            await mgr.safe_place("macd","SELL","LIMIT",qty=0.016,price=sp,
                                 extra_params={'sl':sp*1.03,'tp':sp*0.97})
            self._in = True
        elif prev<0<curr and self._in:
            bp = price * 0.995
            await mgr.safe_place("macd","BUY","LIMIT",qty=0.016,price=bp,
                                 extra_params={'sl':bp*0.97,'tp':bp*1.03})
            self._in = False

class TripleTrendStrategy:
    def __init__(self):
        self._last = 0
        self.round_active = False

    async def check(self, price):
        now = time.time()
        if now - self._last < 1: return
        df1h = data_mgr.klines["1h"]
        if len(df1h) < 15 or df1h['adx'].iat[-1] <= 25: return

        df15 = data_mgr.klines["15m"]
        if len(df15) < 99: return
        ma7,ma25,ma99 = df15.ma7.iat[-1], df15.ma25.iat[-1], df15.ma99.iat[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return

        h,l,c = df15.high.values, df15.low.values, df15.close.values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)

        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not (d1[-1] or d2[-1] or d3[-1])
        prev = (d1[-2], d2[-2], d3[-2])
        curr = (d1[-1], d2[-1], d3[-1])

        flip_dn = self.round_active and any(p and not c2 for p,c2 in zip(prev,curr))
        flip_up = self.round_active and any(not p and c2 for p,c2 in zip(prev,curr))

        self._last = now
        if up_all and not self.round_active:
            self.round_active = True
            p0,sl,tp = price*0.996, price*0.97, price*1.02
            await mgr.safe_place("triple","BUY","LIMIT",qty=0.015,price=p0,
                                 extra_params={'sl':sl,'tp':tp})
        elif dn_all and not self.round_active:
            self.round_active = True
            p0,sl,tp = price*1.004, price*1.03, price*0.98
            await mgr.safe_place("triple","SELL","LIMIT",qty=0.015,price=p0,
                                 extra_params={'sl':sl,'tp':tp})
        elif flip_dn:
            await mgr.safe_place("triple","SELL","MARKET");  self.round_active = False
        elif flip_up:
            await mgr.safe_place("triple","BUY","MARKET");   self.round_active = False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 主循环 ——
async def market_ws():
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry = 0
                async for msg in ws:
                    o = json.loads(msg)
                    stream, d = o["stream"], o["data"]
                    if stream.endswith("@markPrice"):
                        await data_mgr.track_price(float(d["p"]), int(time.time()*1000))
                    else:
                        tf = stream.split("@")[1].split("_")[1]
                        k  = d["k"]
                        await data_mgr.update_kline(tf,
                            float(k["o"]), float(k["h"]),
                            float(k["l"]), float(k["c"]), k["t"])
        except Exception as e:
            delay = min(2**retry, 30)
            LOG.error(f"[WS MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay); retry += 1

async def user_ws():
    global listen_key
    retry = 0
    while True:
        try:
            listen_key = await get_listen_key()
            async with websockets.connect(Config.WS_USER_BASE + listen_key, ping_interval=None) as ws:
                LOG.debug(f"User stream connect: {listen_key}")
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get('e')=='ORDER_TRADE_UPDATE':
                        o = data['o']
                        await pos_tracker.on_order_update(o['i'], o['X'])
        except Exception as e:
            LOG.error(f"[WS USER] {e}, reconnect in 5s")
            await asyncio.sleep(5); retry += 1

async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        try:
            await sync_time()
            await detect_mode()
            await pos_tracker.sync()
            if data_mgr.price is not None:
                await pos_tracker.check_trigger(data_mgr.price)
        except Exception:
            LOG.exception("Maintenance failed")

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
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    await sync_time()
    await detect_mode()
    await load_symbol_filters()
    await data_mgr.load_history()
    await pos_tracker.sync()
    await asyncio.gather(
        market_ws(),
        user_ws(),
        maintenance(),
        engine(),
        keepalive_listen_key()
    )

if __name__=='__main__':
    asyncio.run(main())