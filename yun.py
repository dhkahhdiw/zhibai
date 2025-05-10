#!/usr/bin/env python3
# coding: utf-8

import os, time, json, math, base64, hmac, hashlib, asyncio, logging, signal, urllib.parse
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
    ROTATION_COOLDOWN = 1800  # 同方向冷却 30 分钟
    RATE_LIMIT_RESET  = 60    # REST 权重重置间隔（秒）

# —— 全局状态 ——
session     = None
listen_key  = None
time_offset = 0
is_hedge    = False
price_step = qty_step = None
price_prec = qty_prec = 0
last_weight = {'ip':0,'uid':0,'ts':0}

# —— Ed25519 私钥加载 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 key loaded")

def quantize(val: float, step: float) -> float:
    return math.floor(val/step) * step

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    for _ in range(3):
        try:
            r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
            data = await r.json()
            time_offset = data['serverTime'] - int(time.time()*1000)
            LOG.debug(f"Time offset {time_offset}ms")
            return
        except Exception as e:
            LOG.warning(f"sync_time error: {e}")
            await asyncio.sleep(1)
    LOG.error("Failed sync_time after retries")

async def detect_mode():
    global is_hedge
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url= f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    try:
        r = await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})
        j = await r.json()
        is_hedge = j.get('dualSidePosition', False)
        LOG.debug(f"Hedge mode: {is_hedge}")
    except Exception as e:
        LOG.warning(f"detect_mode error: {e}")

# —— 实时账户净值 (equity) 获取 ——
async def fetch_equity():
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url= f"{Config.REST_BASE}/fapi/v2/account?{qs}&signature={sig}"
    try:
        r = await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})
        data = await r.json()
        bal = float(data.get('totalWalletBalance',0))
        up  = float(data.get('totalUnrealizedProfit',0))
        return bal + up
    except Exception as e:
        LOG.warning(f"fetch_equity error: {e}")
        return None

# —— 精度过滤 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")
    info = await r.json()
    sym = next(s for s in info['symbols'] if s['symbol']==Config.SYMBOL)
    pf  = next(f for f in sym['filters'] if f['filterType']=='PRICE_FILTER')
    ls  = next(f for f in sym['filters'] if f['filterType']=='LOT_SIZE')
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
        if len(lst)>self.window: lst.pop(0)
    def calculate(self, strat: str, price: float, equity: float) -> float:
        profs = self.trades[strat]
        if equity is None or price<=0: return 0
        if len(profs)<10:
            return 0.02 * equity / price
        wins = [p for p in profs if p>0]
        loss = [p for p in profs if p<=0]
        p = len(wins)/len(profs)
        if not wins or not loss:
            return 0.02 * equity / price
        b = (sum(wins)/len(wins)) / abs(sum(loss)/len(loss))
        f = max(0,(p*b - (1-p))/b)*0.5
        return f * equity / price

optimizer = KellyOptimizer(window=100)

# —— 持仓追踪 & 止盈止损 ——
class PositionTracker:
    class Pos:
        __slots__=('id','side','qty','entry','sl','tp','active')
        def __init__(self,i,side,qty,entry,sl,tp):
            self.id,self.side,self.qty,self.entry,self.sl,self.tp,self.active = i,side,qty,entry,sl,tp,True

    def __init__(self):
        self.lock      = asyncio.Lock()
        self.positions = {}
        self.orders    = {}
        self.next_id   = 1

    async def on_fill(self, oid, side, qty, price, sl, tp):
        async with self.lock:
            pid = self.next_id; self.next_id+=1
            sl_q = quantize(sl, price_step); tp_q = quantize(tp, price_step)
            pos = self.Pos(pid,side,qty,price,sl_q,tp_q)
            self.positions[pid] = pos
            self.orders[oid]    = pid
            LOG.info(f"[PT] Open #{pid} {side}@{price:.2f} SL={sl_q:.2f} TP={tp_q:.2f}")

    async def on_update(self, oid, status):
        async with self.lock:
            pid = self.orders.get(oid)
            if pid and status in ('FILLED','CANCELED'):
                self.positions[pid].active=False
                LOG.info(f"[PT] Closed #{pid} via {status}")

    async def check(self, price):
        eps = price_step*0.5
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active: continue
                hit = (price<=pos.sl+eps if pos.side=='BUY' else price>=pos.sl-eps) \
                   or (price>=pos.tp-eps if pos.side=='BUY' else price<=pos.tp+eps)
                if hit:
                    await self._close_local(pos, price)

    async def _close_local(self, pos, price):
        side = 'SELL' if pos.side=='BUY' else 'BUY'
        LOG.info(f"[PT] Local close #{pos.id} {side}@{price:.2f}")
        try:
            await mgr.place(side,'MARKET',qty=pos.qty)
        except Exception as e:
            LOG.error(f"Local close failed: {e}")
        pos.active=False

pos_tracker = PositionTracker()

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs     = ("3m","15m","1h")
        self.klines  = {tf:pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = dict.fromkeys(self.tfs,0)
        self.lock    = asyncio.Lock()
        self.evt     = asyncio.Event()
        self.price   = None; self.ptime=0

    async def load(self):
        async with self.lock:
            for tf in self.tfs:
                r = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                    params={"symbol":Config.SYMBOL,"interval":tf,"limit":1000},
                    headers={'X-MBX-APIKEY':Config.API_KEY})
                data = await r.json()
                df = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),
                    "low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf]=df; self.last_ts[tf]=int(data[-1][0])
                self._comp(tf)
            LOG.info("[DM] History loaded")

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df=self.klines[tf]
            idx = len(df) if ts>self.last_ts[tf] else df.index[-1]
            df.loc[idx, ["open","high","low","close"]] = [o,h,l,c]
            self.last_ts[tf]=ts; self._comp(tf); self.evt.set()

    async def track(self, p, ts):
        async with self.lock: self.price,self.ptime=p,ts
        self.evt.set(); await pos_tracker.check(p)

    async def wait(self):
        await self.evt.wait(); self.evt.clear()

    def _comp(self, tf):
        df=self.klines[tf]
        if len(df)<20: return
        m,s = df.close.rolling(20).mean(), df.close.rolling(20).std()
        df["bb_up"],df["bb_dn"]=m+2*s, m-2*s
        df["bb_pct"]=(df.close-df.bb_dn)/(df.bb_up-df.bb_dn)
        if tf=="15m":
            hl2=(df.high+df.low)/2
            atr=df.high.rolling(10).max()-df.low.rolling(10).min()
            df["st"]=hl2-3*atr
            df["macd"]=MACD(df.close,12,26,9).macd_diff()
            df["ma7"]=df.close.rolling(7).mean()
            df["ma25"]=df.close.rolling(25).mean()
            df["ma99"]=df.close.rolling(99).mean()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n=len(c); st=[0.0]*n; dirc=[False]*n
    hl2=[(h[i]+l[i])/2 for i in range(n)]
    atr=[max(h[max(0,i-per+1):i+1])-min(l[max(0,i-per+1):i+1]) for i in range(n)]
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
        self.states = defaultdict(lambda:{'ts':0,'fp':None,'trend':None})
        self.lock   = asyncio.Lock()
        self.cd     = {'main':Config.ROTATION_COOLDOWN,'macd':Config.ROTATION_COOLDOWN,'triple':Config.ROTATION_COOLDOWN}
    async def check(self,strat,fp,trend):
        async with self.lock:
            st=self.states[strat]; now=time.time()
            if st['fp']==fp or (trend==st['trend'] and now-st['ts']<self.cd[strat]):
                return False
            return True
    async def update(self,strat,fp,trend):
        async with self.lock:
            self.states[strat]={'fp':fp,'trend':trend,'ts':time.time()}

guard = OrderGuard()

async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 用户流 ListenKey & 保活 ——
async def get_listen_key():
    ts = int(time.time()*1000 + time_offset)
    payload = f"timestamp={ts}"
    sig     = base64.b64encode(ed_priv.sign(payload.encode())).decode()
    r = await session.post(f"{Config.REST_BASE}/fapi/v1/listenKey",
        headers={'X-MBX-APIKEY':Config.ED25519_API},
        params={'signature':sig,'timestamp':ts})
    return (await r.json())['listenKey']

async def keepalive_listen_key():
    global listen_key
    while True:
        await asyncio.sleep(1800)
        try:
            listen_key = await get_listen_key(); LOG.debug("ListenKey renewed")
        except Exception as e:
            LOG.error(f"listenKey renew fail: {e}")

# —— 下单管理 ——
class OrderManager:
    def __init__(self, api_key, secret_key):
        self.lock       = asyncio.Lock()
        self.API_KEY    = api_key
        self.SECRET_KEY = secret_key

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, extra_params=None):
        fp    = f"{strat}|{side}|{otype}|{hash(frozenset((extra_params or {}).items()))}"
        trend = 'LONG' if side=='BUY' else 'SHORT'
        if not await guard.check(strat,fp,trend): return
        await self.place(side,otype,qty,price,stop,extra_params or {})
        await guard.update(strat,fp,trend)

    async def place(self, side, otype, qty=None, price=None, stop=None, extra_params={}):
        await ensure_session()
        ts=int(time.time()*1000+time_offset)
        params={"symbol":Config.SYMBOL,"side":side,"type":otype,
                "timestamp":ts,"recvWindow":Config.RECV_WINDOW}
        if qty   is not None: params["quantity"]=f"{quantize(qty,qty_step):.{qty_prec}f}"
        if price is not None: params["price"]=f"{quantize(price,price_step):.{price_prec}f}"
        if stop  is not None: params["stopPrice"]=f"{quantize(stop,price_step):.{price_prec}f}"
        if otype=="LIMIT":    params["timeInForce"]="GTC"
        if otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            params["closePosition"]="true"; params.pop("quantity",None)
        if is_hedge:
            params["positionSide"] = 'LONG' if side=='BUY' else 'SHORT'
            if otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
                params["reduceOnly"]="true"
        params.update(extra_params)
        qs = urllib.parse.urlencode(sorted(params.items()))
        sig= hmac.new(self.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url= f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        hdr={'X-MBX-APIKEY':self.API_KEY}
        async with self.lock:
            r = await session.post(url, headers=hdr)
            data = await r.json()
            if r.status!=200:
                LOG.error(f"[Order ERR] {otype} {side}: {data}")
            else:
                LOG.debug(f"[Order OK] {otype} {side}: {data}")
                if otype=="LIMIT" and float(data.get('executedQty',0))>0:
                    await pos_tracker.on_fill(
                        data['orderId'], side, float(data['executedQty']),
                        float(data.get('price',0)), extra_params.get('sl'), extra_params.get('tp')
                    )

# 实例化下单管理
mgr = OrderManager(Config.API_KEY, Config.SECRET_KEY)

# —— 策略：主、MACD、三重趋势 ——
class MainStrategy:
    name="main"
    def __init__(self): self._last=0; self.interval=1
    async def check(self, price):
        now=time.time()
        if now-self._last<self.interval: return
        df=data_mgr.klines["15m"]
        if len(df)<30: return
        bb3 = df.bb_pct.iat[-1]
        if not (bb3<=0 or bb3>=1): return
        side = "BUY" if bb3<=0 else "SELL"
        strong = data_mgr.klines["1h"].bb_pct.iat[-1]<0.2 or data_mgr.klines["1h"].bb_pct.iat[-1]>0.8
        h,l,c = df.high.values,df.low.values,df.close.values
        st_line,st_dir = numba_supertrend(h,l,c,10,3)
        trend_ok = (side=="BUY" and price>st_line[-1] and st_dir[-1]) or \
                   (side=="SELL" and price<st_line[-1] and not st_dir[-1])
        equity = await fetch_equity() or 0
        total_size = optimizer.calculate(self.name, price, equity)
        self._last=now

        # 分档挂单
        if strong:
            levels = [0.0025,0.004,0.006,0.008,0.016]
        else:
            levels = [0.0025,-0.0055] if trend_ok else [-0.0075]
        per = total_size/len(levels) if levels else 0
        sl = price*(0.98 if side=="BUY" else 1.02)
        tp = price*(1.01 if side=="BUY" else 0.99)
        for lvl in levels:
            p0 = price*(1+lvl if side=="BUY" else 1-lvl)
            await mgr.safe_place(self.name, side, "LIMIT",
                                 qty=per, price=p0, extra_params={'sl':sl,'tp':tp})
        # 挂止损触发单
        await mgr.safe_place(self.name, side, "STOP_MARKET", stop=sl)

class MACDStrategy:
    name="macd"
    def __init__(self): self._in=False
    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        prev,curr = df.macd.iat[-2], df.macd.iat[-1]
        equity = await fetch_equity() or 0
        size   = optimizer.calculate(self.name, price, equity)
        if prev>0>curr and not self._in:
            sp=price*1.005; sl, tp = sp*1.03, sp*0.97
            await mgr.safe_place(self.name,"SELL","LIMIT",qty=size,price=sp,extra_params={'sl':sl,'tp':tp})
            self._in=True
        elif prev<0<curr and self._in:
            bp=price*0.995; sl, tp = bp*0.97, bp*1.03
            await mgr.safe_place(self.name,"BUY","LIMIT",qty=size,price=bp,extra_params={'sl':sl,'tp':tp})
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
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        h,l,c = df.high.values,df.low.values,df.close.values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not (d1[-1] or d2[-1] or d3[-1])
        prev,curr=(d1[-2],d2[-2],d3[-2]),(d1[-1],d2[-1],d3[-1])
        flip_dn = self.active and any(p and not c2 for p,c2 in zip(prev,curr))
        flip_up = self.active and any(not p and c2 for p,c2 in zip(prev,curr))
        equity = await fetch_equity() or 0
        size   = optimizer.calculate(self.name, price, equity)
        self._last=now

        if up_all and not self.active:
            p0,sl,tp = price*0.995,price*0.97,price*1.02
            await mgr.safe_place(self.name,"BUY","LIMIT",qty=size,price=p0,extra_params={'sl':sl,'tp':tp})
            self.active=True
        elif dn_all and not self.active:
            p0,sl,tp = price*1.005,price*1.03,price*0.98
            await mgr.safe_place(self.name,"SELL","LIMIT",qty=size,price=p0,extra_params={'sl':sl,'tp':tp})
            self.active=True
        elif flip_dn:
            await mgr.safe_place(self.name,"SELL","MARKET"); self.active=False
        elif flip_up:
            await mgr.safe_place(self.name,"BUY","MARKET"); self.active=False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 主循环 ——
async def market_ws():
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry = 0
                async for msg in ws:
                    o = json.loads(msg); s,d = o['stream'],o['data']
                    if s.endswith('@markPrice'):
                        await data_mgr.track(float(d['p']), int(time.time()*1000))
                    else:
                        tf = s.split('@')[1].split('_')[1]
                        k  = d['k']
                        await data_mgr.update_kline(tf,
                            float(k['o']),float(k['h']),float(k['l']),float(k['c']),k['t'])
        except Exception as e:
            delay = min(2**retry,30)
            LOG.warning(f"[WS MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay); retry+=1

async def user_ws():
    global listen_key
    retry = 0
    while True:
        try:
            listen_key = await get_listen_key()
            async with websockets.connect(Config.WS_USER_BASE+listen_key) as ws:
                LOG.info(f"User stream connected")
                async for msg in ws:
                    data=json.loads(msg)
                    if data.get('e')=='ORDER_TRADE_UPDATE':
                        o=data['o']; await pos_tracker.on_update(o['i'], o['X'])
        except Exception as e:
            LOG.warning(f"[WS USER] {e}, reconnect in 5s")
            await asyncio.sleep(5); retry+=1

async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode()

async def engine():
    while True:
        await data_mgr.wait()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60: continue
        for strat in strategies:
            try: await strat.check(data_mgr.price)
            except Exception: LOG.exception(f"{strat.name} fail")

async def main():
    global session
    session = aiohttp.ClientSession()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    await sync_time(); await detect_mode(); await load_symbol_filters()
    await data_mgr.load()

    await asyncio.gather(
        asyncio.shield(market_ws()),
        asyncio.shield(user_ws()),
        asyncio.shield(maintenance()),
        asyncio.shield(engine()),
        asyncio.shield(keepalive_listen_key())
    )

if __name__=='__main__':
    asyncio.run(main())