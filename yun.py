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
LOG.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
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

# —— 确保 session 可用 ——
async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 时间同步 & 模式检测 ——
async def sync_time():
    await ensure_session()
    data = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    global time_offset
    time_offset = data['serverTime'] - int(time.time() * 1000)
    LOG.debug(f"Time offset: {time_offset} ms")

async def detect_mode():
    await ensure_session()
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    data = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    global is_hedge
    is_hedge = data.get('dualSidePosition', False)
    LOG.info(f"Hedge mode: {is_hedge}")

# —— 实时账户净值获取 ——
async def fetch_equity():
    await ensure_session()
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v2/account?{qs}&signature={sig}"
    try:
        data = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
        bal = float(data.get('totalWalletBalance', 0))
        up  = float(data.get('totalUnrealizedProfit', 0))
        LOG.debug(f"Equity fetched: wallet={bal:.4f}, unrealized={up:.4f}")
        return bal + up
    except Exception as e:
        LOG.warning(f"fetch_equity error: {e}")
        return None

# —— 精度过滤 ——
async def load_symbol_filters():
    await ensure_session()
    info = await (await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")).json()
    sym = next(s for s in info['symbols'] if s['symbol'] == Config.SYMBOL)
    pf  = next(f for f in sym['filters'] if f['filterType'] == 'PRICE_FILTER')
    ls  = next(f for f in sym['filters'] if f['filterType'] == 'LOT_SIZE')
    global price_step, qty_step, price_prec, qty_prec
    price_step, qty_step = float(pf['tickSize']), float(ls['stepSize'])
    price_prec = int(round(-math.log10(price_step)))
    qty_prec   = int(round(-math.log10(qty_step)))
    LOG.info(f"Filters: price_step={price_step}, qty_step={qty_step}")

# —— 凯利仓位优化 ——
class KellyOptimizer:
    def __init__(self, window=100):
        self.window = window
        self.trades = defaultdict(list)
    def update(self, strat: str, profit: float):
        lst = self.trades[strat]
        lst.append(profit)
        if len(lst) > self.window:
            lst.pop(0)
    def calculate(self, strat: str, price: float, equity: float) -> float:
        if equity is None or price <= 0:
            return 0
        profs = self.trades[strat]
        if len(profs) < 10:
            base = 0.02 * equity / price
            LOG.debug(f"Kelly[{strat}]: insufficient history, base size={base:.6f}")
            return base
        wins = [p for p in profs if p>0]
        loss = [p for p in profs if p<=0]
        p = len(wins)/len(profs)
        b = (sum(wins)/len(wins)) / abs(sum(loss)/len(loss)) if loss else 0
        f = max(0, (p*b - (1-p))/b)*0.5 if b>0 else 0
        size = f * equity / price
        LOG.debug(f"Kelly[{strat}]: p={p:.3f}, b={b:.3f}, f={f:.3f}, size={size:.6f}")
        return size

optimizer = KellyOptimizer()

# —— 下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def cancel_order(self, oid):
        await ensure_session()
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({
            'symbol': Config.SYMBOL,
            'orderId': oid,
            'timestamp': ts,
            'recvWindow': Config.RECV_WINDOW
        })
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        async with self.lock:
            r = await session.delete(url, headers={'X-MBX-APIKEY': Config.API_KEY})
            LOG.info(f"cancel_order {oid}: status={r.status}")

    async def place(self, side, otype, qty=None, price=None, stop=None,
                    extra_params=None, return_id=False):
        await ensure_session()
        if qty is not None and qty <= qty_step:
            LOG.debug(f"Skipping too-small qty={qty}")
            return None
        ts = int(time.time()*1000 + time_offset)
        params = {
            'symbol': Config.SYMBOL,
            'side': side,
            'type': otype,
            'timestamp': ts,
            'recvWindow': Config.RECV_WINDOW
        }
        if qty   is not None: params['quantity']  = f"{quantize(qty,qty_step):.{qty_prec}f}"
        if price is not None: params['price']     = f"{quantize(price,price_step):.{price_prec}f}"
        if stop  is not None: params['stopPrice'] = f"{quantize(stop,price_step):.{price_prec}f}"
        if otype == 'LIMIT':  params['timeInForce'] = 'GTC'
        if otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
            params.update({'workingType':'MARK_PRICE','priceProtect':'FALSE','closePosition':'true'})
            if is_hedge:
                params['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'
                params['reduceOnly']   = 'true'
        if is_hedge and otype in ('LIMIT','MARKET'):
            params['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'
        if extra_params:
            params.update(extra_params)

        qs = urllib.parse.urlencode(sorted(params.items()), safe=',')
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        headers = {'X-MBX-APIKEY': Config.API_KEY}

        async with self.lock:
            r = await session.post(url, headers=headers)
            data = await r.json()
            if r.status != 200:
                LOG.error(f"[Mgr] ERR {otype} {side}: {data}")
                return None
            oid = data.get('orderId')
            LOG.info(f"[Mgr] OK {otype} {side} oid={oid}")
            exec_qty = float(data.get('executedQty', 0))
            if otype=='LIMIT' and exec_qty>0:
                # partial fill on initial entry
                price_ = float(data.get('price', 0))
                sl = extra_params.get('sl'); tp = extra_params.get('tp')
                await pos_tracker.on_fill(oid, side, exec_qty, price_, sl, tp)
            return oid if return_id else None

mgr = OrderManager()

# —— 持仓 & 止盈止损 ——
class PositionTracker:
    class Pos:
        __slots__ = ('cloid','side','qty','sl','tp','sl_id','tp_id','active')
        def __init__(self, cloid, side, qty, sl, tp, sl_id, tp_id):
            self.cloid, self.side, self.qty = cloid, side, qty
            self.sl, self.tp = sl, tp
            self.sl_id, self.tp_id = sl_id, tp_id
            self.active = True

        def __repr__(self):
            return (f"Pos(cloid={self.cloid}, side={self.side}, qty={self.qty}, "
                    f"sl={self.sl}, tp={self.tp}, sl_id={self.sl_id}, tp_id={self.tp_id}, active={self.active})")

    def __init__(self):
        self.lock       = asyncio.Lock()
        self.next_cloid = 1
        self.positions  = {}
        self.ord2cloid  = {}

    async def on_fill(self, oid, side, qty, price, sl, tp):
        async with self.lock:
            cloid = self.next_cloid; self.next_cloid += 1
            sl_q = quantize(sl, price_step); tp_q = quantize(tp, price_step)
            close = 'SELL' if side=='BUY' else 'BUY'
            sl_id = await mgr.place(close, 'STOP_MARKET', stop=sl_q, return_id=True)
            tp_id = await mgr.place(close, 'TAKE_PROFIT_MARKET', stop=tp_q, return_id=True)
            pos = self.Pos(cloid, side, qty, sl_q, tp_q, sl_id, tp_id)
            self.positions[cloid] = pos
            self.ord2cloid[oid]   = cloid
            LOG.info(f"[PT] Opened {pos}")

    async def on_order_update(self, oid, status):
        async with self.lock:
            # SL/TP triggered remotely
            for pos in self.positions.values():
                if oid in (pos.sl_id, pos.tp_id):
                    pos.active = False
                    LOG.info(f"[PT] Remote SL/TP triggered for cloid={pos.cloid}")
                    return
            # Main position closed
            cloid = self.ord2cloid.get(oid)
            if cloid and status in ('FILLED','CANCELED'):
                self.positions[cloid].active = False
                LOG.info(f"[PT] Remote position close for cloid={cloid}")

    async def check_trigger(self, price):
        eps = price_step * 0.5
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active: continue
                hit_sl = (pos.side=='BUY' and price <= pos.sl+eps) or (pos.side=='SELL' and price >= pos.sl-eps)
                hit_tp = (pos.side=='BUY' and price >= pos.tp-eps) or (pos.side=='SELL' and price <= pos.tp+eps)
                if hit_sl or hit_tp:
                    LOG.info(f"[PT] Local trigger for cloid={pos.cloid} at price={price:.4f}")
                    if pos.sl_id: await mgr.cancel_order(pos.sl_id)
                    if pos.tp_id: await mgr.cancel_order(pos.tp_id)
                    close = 'SELL' if pos.side=='BUY' else 'BUY'
                    await mgr.place(close, 'MARKET', qty=pos.qty)
                    pos.active = False

    async def sync(self):
        # optional: fetch remote positions & reconcile
        async with self.lock:
            LOG.debug("Current open positions:")
            for pos in self.positions.values():
                LOG.debug("  " + repr(pos))

pos_tracker = PositionTracker()

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs     = ("3m","15m","1h")
        self.klines  = {tf: pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = dict.fromkeys(self.tfs, 0)
        self.lock    = asyncio.Lock()
        self.evt     = asyncio.Event()
        self.price   = None
        self.ptime   = 0

    async def load_history(self):
        await ensure_session()
        async with self.lock:
            for tf in self.tfs:
                r = await session.get(
                    f"{Config.REST_BASE}/fapi/v1/klines",
                    params={"symbol":Config.SYMBOL,"interval":tf,"limit":1000},
                    headers={'X-MBX-APIKEY':Config.API_KEY}
                )
                data = await r.json()
                df = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),
                                    "low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf]  = df
                self.last_ts[tf] = int(data[-1][0])
                self._compute(tf)
                LOG.info(f"[DM] {tf} loaded {len(df)} bars")

    async def update_kline(self, tf, o, h, l, c, ts):
        async with self.lock:
            df = self.klines[tf]
            idx = len(df) if ts > self.last_ts[tf] else df.index[-1]
            df.loc[idx, ["open","high","low","close"]] = [o,h,l,c]
            self.last_ts[tf] = ts
            self._compute(tf)
            self.evt.set()
            LOG.debug(f"[DM] {tf} kline updated @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts/1000))}")

    async def track_price(self, p, ts):
        async with self.lock:
            self.price, self.ptime = p, ts
        LOG.info(f"[PRICE] {p:.4f} @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts/1000))}")
        self.evt.set()
        await pos_tracker.check_trigger(p)

    async def wait_update(self):
        await self.evt.wait()
        self.evt.clear()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df) < 20: return
        m = df.close.rolling(20).mean()
        s = df.close.rolling(20).std()
        df["bb_up"], df["bb_dn"] = m+2*s, m-2*s
        df["bb_pct"] = (df.close - df.bb_dn) / (df.bb_up - df.bb_dn)
        if tf == "15m":
            hl2 = (df.high + df.low) / 2
            atr = df.high.rolling(10).max() - df.low.rolling(10).min()
            df["st"]   = hl2 - 3*atr
            df["macd"] = MACD(df.close,12,26,9).macd_diff()
            df["ma7"], df["ma25"], df["ma99"] = df.close.rolling(7).mean(), df.close.rolling(25).mean(), df.close.rolling(99).mean()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h, l, c, per, mult):
    n = len(c)
    st = [0.0]*n; dirc=[False]*n
    hl2 = [(h[i]+l[i])/2 for i in range(n)]
    atr = [max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up, dn = [hl2[i]+mult*atr[i] for i in range(n)], [hl2[i]-mult*atr[i] for i in range(n)]
    st[0], dirc[0] = up[0], True
    for i in range(1,n):
        if c[i] > st[i-1]:
            st[i], dirc[i] = max(dn[i], st[i-1]), True
        else:
            st[i], dirc[i] = min(up[i], st[i-1]), False
    return st, dirc

# —— 策略实现 ——
class MainStrategy:
    name="main"
    def __init__(self): self._last=0; self.interval=1
    async def check(self, price):
        now = time.time()
        if now - self._last < self.interval: return
        df = data_mgr.klines["15m"]
        if len(df) < 30: return

        h,l,c = df.high.values, df.low.values, df.close.values
        st_line, st_dir = numba_supertrend(h,l,c,10,3)
        trend_ok = (price>st_line[-1] and st_dir[-1]) or (price<st_line[-1] and not st_dir[-1])
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not (bb3<=0 or bb3>=1): return
        side = "BUY" if bb3<=0 else "SELL"
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong = (bb1<0.2 or bb1>0.8)

        equity = await fetch_equity() or 0
        size    = optimizer.calculate(self.name, price, equity)
        if size <= qty_step:
            LOG.debug(f"[{self.name}] size too small: {size}")
            return

        self._last = now
        levels = [0.0025,0.004,0.006,0.008,0.016] if strong else (
                 [-0.0055,-0.0155] if (side=="BUY" and trend_ok) or (side=="SELL" and trend_ok)
                 else ([-0.0075] if side=="BUY" else [0.0075]))
        sl = price*(0.98 if side=="BUY" else 1.02)
        tp = price*(1.02 if side=="BUY" else 0.98)
        LOG.info(f"[{self.name}] placing {side} ladders: levels={levels}, size_each={size/len(levels):.6f}")
        for lvl in levels:
            p0 = price*(1+lvl if side=="BUY" else 1-lvl)
            await mgr.place(side,"LIMIT",qty=size/len(levels),price=p0,extra_params={'sl':sl,'tp':tp})
        await mgr.place(side,"STOP_MARKET",stop=sl)

class MACDStrategy:
    name="macd"
    def __init__(self): self._in=False
    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]
        if not(price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        prev,curr = df.macd.iat[-2],df.macd.iat[-1]
        equity = await fetch_equity() or 0
        size = optimizer.calculate(self.name, price, equity)
        if size<=qty_step:
            LOG.debug(f"[{self.name}] size too small: {size}")
            return
        if prev>0>curr and not self._in:
            sp,sl,tp=price*1.005,price*1.005*1.03,price*1.005*0.97
            LOG.info(f"[{self.name}] SELL signal @ {price:.4f}")
            await mgr.place("SELL","LIMIT",qty=size,price=sp,extra_params={'sl':sl,'tp':tp})
            self._in=True
        elif prev<0<curr and self._in:
            bp,sl,tp=price*0.995,price*0.995*0.97,price*0.995*1.03
            LOG.info(f"[{self.name}] BUY signal @ {price:.4f}")
            await mgr.place("BUY","LIMIT",qty=size,price=bp,extra_params={'sl':sl,'tp':tp})
            self._in=False

class TripleTrendStrategy:
    name="triple"
    def __init__(self): self.active=False; self._last=0
    async def check(self, price):
        now=time.time()
        if now-self._last<1: return
        df=data_mgr.klines["15m"]
        if len(df)<99: return
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]
        if not(price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        h,l,c=df.high.values,df.low.values,df.close.values
        _,d1=numba_supertrend(h,l,c,10,1)
        _,d2=numba_supertrend(h,l,c,11,2)
        _,d3=numba_supertrend(h,l,c,12,3)
        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not (d1[-1] or d2[-1] or d3[-1])
        prev,curr=(d1[-2],d2[-2],d3[-2]),(d1[-1],d2[-1],d3[-1])
        flip_dn = self.active and any(p and not c2 for p,c2 in zip(prev,curr))
        flip_up = self.active and any(not p and c2 for p,c2 in zip(prev,curr))
        equity=await fetch_equity() or 0
        size=optimizer.calculate(self.name, price, equity)
        if size<=qty_step:
            LOG.debug(f"[{self.name}] size too small: {size}")
            return

        self._last=now
        if up_all and not self.active:
            p0,sl,tp=price*0.995,price*0.995*0.97,price*0.995*1.03
            LOG.info(f"[{self.name}] BUY triple @ {price:.4f}")
            await mgr.place("BUY","LIMIT",qty=size,price=p0,extra_params={'sl':sl,'tp':tp})
            self.active=True
        elif dn_all and not self.active:
            p0,sl,tp=price*1.005,price*1.005*1.03,price*1.005*0.98
            LOG.info(f"[{self.name}] SELL triple @ {price:.4f}")
            await mgr.place("SELL","LIMIT",qty=size,price=p0,extra_params={'sl':sl,'tp':tp})
            self.active=True
        elif flip_dn:
            LOG.info(f"[{self.name}] Flip down, MARKET SELL @ {price:.4f}")
            await mgr.place("SELL","MARKET"); self.active=False
        elif flip_up:
            LOG.info(f"[{self.name}] Flip up, MARKET BUY @ {price:.4f}")
            await mgr.place("BUY","MARKET"); self.active=False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 主循环 ——
async def market_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry=0
                LOG.info("Market ws connected")
                async for msg in ws:
                    o = json.loads(msg); s,d = o["stream"], o["data"]
                    if s.endswith("@markPrice"):
                        await data_mgr.track_price(float(d["p"]), int(time.time()*1000))
                    else:
                        tf = s.split("@")[1].split("_")[1]; k=d["k"]
                        await data_mgr.update_kline(tf,
                            float(k["o"]),float(k["h"]),float(k["l"]),float(k["c"]),k["t"])
        except Exception as e:
            delay=min(2**retry,30)
            LOG.warning(f"[WS MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay); retry+=1

async def user_ws():
    global listen_key
    retry=0
    while True:
        try:
            listen_key = await get_listen_key()
            async with websockets.connect(Config.WS_USER_BASE + listen_key) as ws:
                LOG.info("User ws connected")
                async for msg in ws:
                    data=json.loads(msg)
                    if data.get('e')=='ORDER_TRADE_UPDATE':
                        o=data['o']
                        await pos_tracker.on_order_update(o['i'], o['X'])
        except Exception as e:
            LOG.warning(f"[WS USER] {e}, reconnect in 5s")
            await asyncio.sleep(5); retry+=1

async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode(); await pos_tracker.sync()

async def engine():
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60:
            LOG.warning("Stale price, skipping")
            continue
        for strat in strategies:
            try:
                await strat.check(data_mgr.price)
            except Exception as e:
                LOG.exception(f"Strategy {strat.name} failed: {e}")

async def get_listen_key():
    await ensure_session()
    ts = int(time.time()*1000 + time_offset)
    payload = f"timestamp={ts}"
    sig     = base64.b64encode(ed_priv.sign(payload.encode())).decode()
    async with session.post(
        f"{Config.REST_BASE}/fapi/v1/listenKey",
        headers={'X-MBX-APIKEY': Config.ED25519_API},
        params={'signature': sig, 'timestamp': ts}
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

async def main():
    global session
    await ensure_session()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: LOG.info(f"Received {sig.name}, shutting down..."))

    # 初始化
    await sync_time(); await detect_mode(); await load_symbol_filters()
    await data_mgr.load_history()

    # 并行
    await asyncio.gather(
        market_ws(),
        user_ws(),
        maintenance(),
        engine(),
        keepalive_listen_key()
    )

if __name__=='__main__':
    asyncio.run(main())