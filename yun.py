#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import uuid
import math
import base64
import hmac
import hashlib
import asyncio
import logging
import signal
import urllib.parse
from collections import defaultdict

import uvloop
import aiohttp
import websockets
import pandas as pd
from ta.trend import MACD
from numba import jit
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

# —— 高性能事件循环 & 环境加载 ——
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
    SYMBOL = 'ETHUSDC'
    PAIR = SYMBOL.lower()
    API_KEY = os.getenv('BINANCE_API_KEY')
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API = os.getenv('ED25519_API_KEY')
    ED25519_KEY = os.getenv('ED25519_KEY_PATH')
    REST_BASE = 'https://fapi.binance.com'
    WS_MARKET = f"wss://fstream.binance.com/stream?streams={PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    WS_USER = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    RECV_WINDOW = 5000
    SYNC_INTERVAL = 300


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


def quantize(val: float, step: float) -> float:
    return math.floor(val / step) * step


# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    data = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    time_offset = data['serverTime'] - int(time.time() * 1000)
    LOG.debug(f"Time offset: {time_offset}ms")


async def detect_mode():
    global is_hedge
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    data = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    is_hedge = data.get('dualSidePosition', False)
    LOG.info(f"Hedge mode: {is_hedge}")


# —— 精度加载 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    info = await (await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")).json()
    sym = next(s for s in info['symbols'] if s['symbol'] == Config.SYMBOL)
    pf = next(f for f in sym['filters'] if f['filterType'] == 'PRICE_FILTER')
    ls = next(f for f in sym['filters'] if f['filterType'] == 'LOT_SIZE')
    price_step, qty_step = float(pf['tickSize']), float(ls['stepSize'])
    price_prec = int(round(-math.log10(price_step)))
    qty_prec = int(round(-math.log10(qty_step)))
    LOG.info(f"Filters: price_step={price_step}, qty_step={qty_step}")


# —— 持仓＆止盈止损管理 ——
class PositionTracker:
    class Pos:
        __slots__ = ('cloid', 'side', 'qty', 'entry', 'sl', 'tp', 'active', 'sl_id', 'tp_id')

        def __init__(self, cloid, side, qty, entry, sl, tp, sl_id, tp_id):
            self.cloid, self.side, self.qty = cloid, side, qty
            self.entry, self.sl, self.tp = entry, sl, tp
            self.active = True
            self.sl_id, self.tp_id = sl_id, tp_id

    def __init__(self):
        self.positions = {}  # cloid -> Pos
        self.order_map = {}  # orderId -> cloid
        self.lock = asyncio.Lock()
        self.next_cloid = 1

    async def on_fill(self, order: dict, side, qty, price, sl, tp):
        """限价单成交回调，下挂远程 SL/TP 并本地注册"""
        async with self.lock:
            cloid = self.next_cloid;
            self.next_cloid += 1
            close_side = 'SELL' if side == 'BUY' else 'BUY'
            sl_id = await mgr.place(close_side, 'STOP_MARKET',
                                    qty=qty, stop=sl,
                                    extra_params={'closePosition': 'true'},
                                    return_id=True)
            tp_id = await mgr.place(close_side, 'TAKE_PROFIT_MARKET',
                                    qty=qty, stop=tp,
                                    extra_params={'closePosition': 'true'},
                                    return_id=True)
            pos = self.Pos(cloid, side, qty, price, sl, tp, sl_id, tp_id)
            self.positions[cloid] = pos
            self.order_map[order['orderId']] = cloid
            LOG.info(f"[PT] Opened cloid={cloid} {side}@{price:.4f} SL={sl:.4f}({sl_id}) TP={tp:.4f}({tp_id})")

    async def on_order_update(self, order_id, status):
        async with self.lock:
            # 远程 SL/TP 执行或取消
            for pos in self.positions.values():
                if order_id in (pos.sl_id, pos.tp_id):
                    pos.active = False
                    LOG.info(f"[PT] Remote SL/TP triggered for cloid={pos.cloid}")
                    return
            # 限价单被成交/取消
            if order_id in self.order_map and status in ('FILLED', 'CANCELED'):
                cl = self.order_map[order_id]
                self.positions[cl].active = False
                LOG.info(f"[PT] Position closed remote for cloid={cl}")

    async def check_trigger(self, price):
        """本地触发备份：先 cancel 远程单，再市价平仓"""
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active: continue
                hit_sl = (pos.side == 'BUY' and price <= pos.sl) or (pos.side == 'SELL' and price >= pos.sl)
                hit_tp = (pos.side == 'BUY' and price >= pos.tp) or (pos.side == 'SELL' and price <= pos.tp)
                if hit_sl or hit_tp:
                    LOG.info(f"[PT] Local trigger cloid={pos.cloid} price={price:.4f}")
                    await mgr.cancel(pos.sl_id);
                    await mgr.cancel(pos.tp_id)
                    close = 'SELL' if pos.side == 'BUY' else 'BUY'
                    await mgr.place(close, 'MARKET', qty=pos.qty)
                    pos.active = False

    async def sync(self):
        """同步远程仓位风险；保证返回是 List，再迭代"""
        ts = int(time.time() * 1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        data = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
        if isinstance(data, list):
            for p in data:
                if p.get('symbol') == Config.SYMBOL:
                    LOG.debug(f"[PT] Remote pos amt={p['positionAmt']} side={p.get('positionSide')}")
        else:
            LOG.error(f"[PT] positionRisk unexpected response: {data}")


pos_tracker = PositionTracker()


# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs     = ("3m","15m","1h")
        self.klines  = {tf: pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts = {tf: 0 for tf in self.tfs}
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.price   = None; self.ptime=0

    async def load_history(self):
        async with self.lock:
            for tf in self.tfs:
                res = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                    params={"symbol":Config.SYMBOL,"interval":tf,"limit":1000},
                    headers={"X‑MBX‑APIKEY":Config.API_KEY})
                data = await res.json()
                df = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),
                                    "low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf] = df; self.last_ts[tf] = int(data[-1][0])
                self._compute(tf)
                LOG.info(f"[DM] {tf} loaded {len(df)} bars")

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            if ts>self.last_ts[tf]:
                df.loc[len(df)] = [o,h,l,c]
                self.last_ts[tf] = ts
            else:
                df.loc[len(df)-1] = [o,h,l,c]
            self._compute(tf)
            self._evt.set()

    async def track_price(self, p, ts):
        async with self.lock:
            self.price, self.ptime = p, ts
        self._evt.set()
        await pos_tracker.check_trigger(p)

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

    def _compute(self, tf):
        df = self.klines[tf]
        if len(df)<20: return
        m = df.close.rolling(20).mean(); s = df.close.rolling(20).std()
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

# —— Numba Supertrend ——
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
    return st, dirc

# —— 订单守卫 ——
class OrderGuard:
    def __init__(self):
        self.states = {}
        self.lock   = asyncio.Lock()
        self.cooldown = {"main":360,"macd":360,"triple":360}

    async def check(self, strat, fp, trend):
        async with self.lock:
            st = self.states.get(strat, {})
            now = time.time()
            if now - st.get('ts',0) < self.cooldown[strat]: return False
            if st.get('trend')==trend and st.get('fp')==fp: return False
            return True

    async def update(self, strat, fp, trend):
        async with self.lock:
            self.states[strat] = {'fp':fp,'trend':trend,'ts':time.time()}

guard = OrderGuard()

async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, extra_params=None):
        fp = f"{strat}|{side}|{otype}|{price}|{stop}"
        trend = 'LONG' if side=='BUY' else 'SHORT'
        if not await guard.check(strat, fp, trend): return
        LOG.info(f"[Mgr] {strat} → {side} {otype}")
        oid = await self.place(side, otype, qty, price, stop, extra_params=extra_params or {})
        await guard.update(strat, fp, trend)
        return oid

    async def place(self, side, otype, qty=None, price=None, stop=None,
                    extra_params=None, return_id=False):
        await ensure_session()
        ts = int(time.time()*1000 + time_offset)
        params = {"symbol":Config.SYMBOL,"side":side,"type":otype,
                  "timestamp":ts,"recvWindow":Config.RECV_WINDOW}
        if qty   is not None: params["quantity"]  = f"{quantize(qty,qty_step):.{qty_prec}f}"
        if price is not None: params["price"]     = f"{quantize(price,price_step):.{price_prec}f}"
        if stop  is not None: params["stopPrice"] = f"{quantize(stop,price_step):.{price_prec}f}"
        if otype=="LIMIT":    params["timeInForce"]="GTC"
        if otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"):
            params["closePosition"]="true"
        if is_hedge and otype in ("LIMIT","MARKET"):
            params["positionSide"] = "LONG" if side=="BUY" else "SHORT"
        params.update(extra_params or {})
        qs = urllib.parse.urlencode(sorted(params.items()), safe=',')
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        async with self.lock:
            r = await session.post(url, headers={'X‑MBX‑APIKEY':Config.API_KEY})
            text = await r.text()
            if r.status==200:
                data = await r.json()
                oid = data['orderId']
                # 如果是限价成交，触发 on_fill
                if otype=="LIMIT" and float(data.get('executedQty',0))>0:
                    sl = params.get('stopPrice'); tp = params.get('tp')
                    await pos_tracker.on_fill(data, side,
                                              float(data['executedQty']),
                                              float(data['price']), sl, tp)
                LOG.debug(f"[Mgr] OK {otype} {side} id={oid}")
                return oid if return_id else None
            else:
                LOG.error(f"[Mgr] ERR {otype} {side}: {text}")
                return None

    async def cancel(self, order_id):
        if order_id is None: return
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'symbol':Config.SYMBOL,
                                     'orderId':order_id,
                                     'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
        sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url=f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        await session.delete(url, headers={'X‑MBX‑APIKEY':Config.API_KEY})
        LOG.debug(f"[Mgr] Cancelled {order_id}")

mgr = OrderManager()

# —— 策略：Main ——
class MainStrategy:
    def __init__(self):
        self._last=0; self.interval=1
    async def check(self, price):
        now=time.time()
        if now-self._last<self.interval: return
        df15 = data_mgr.klines["15m"]
        if len(df15)<99: return
        # MA 前置过滤
        ma7,ma25,ma99 = df15.ma7.iat[-1], df15.ma25.iat[-1], df15.ma99.iat[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        # Supertrend 判断趋势
        h,l,c = df15.high.values, df15.low.values, df15.close.values
        st_line, st_dir = numba_supertrend(h,l,c,10,3)
        trend_up = price>st_line[-1] and st_dir[-1]
        trend_dn = price<st_line[-1] and not st_dir[-1]
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not (bb3<=0 or bb3>=1): return
        side = "BUY" if bb3<=0 else "SELL"
        # 网格+SL/TP
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong = bb1<0.2 or bb1>0.8
        if strong:
            qty = 0.12 if ((side=="BUY" and trend_up) or (side=="SELL" and trend_dn)) else 0.07
            levels=[0.0025,0.004,0.006,0.008,0.016]; sizes=[qty*0.2]*5
        else:
            if side=="BUY":
                levels, sizes = ([-0.0055,-0.0155],[0.015]) if trend_up else ([-0.0075],[0.015])
            else:
                levels, sizes = ([0.0055,0.0155],[0.015]) if trend_dn else ([0.0075],[0.015])
        self._last=now
        for lvl,sz in zip(levels,sizes):
            p0=price*(1+lvl if side=="BUY" else 1-lvl)
            sl=price*0.98 if side=="BUY" else price*1.02
            tp=price*(1+0.01 if side=="BUY" else 1-0.01)
            await mgr.safe_place("main", side, "LIMIT", qty=sz, price=p0, extra_params={'sl':sl,'tp':tp})
        # 止损
        sl = price*0.98 if side=="BUY" else price*1.02
        await mgr.safe_place("main", side, "STOP_MARKET", stop=sl)

# —— 策略：MACD ——
class MACDStrategy:
    def __init__(self): self._in=False
    async def check(self, price):
        df = data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        # MA 前置
        ma7,ma25,ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        prev, curr = df.macd.iat[-2], df.macd.iat[-1]
        if prev>0>curr and not self._in:
            sp=price*1.005
            await mgr.safe_place("macd","SELL","LIMIT",qty=0.15,price=sp,extra_params={'sl':sp*1.03,'tp':sp*0.97})
            self._in=True
        elif prev<0<curr and self._in:
            bp=price*0.995
            await mgr.safe_place("macd","BUY","LIMIT",qty=0.15,price=bp,extra_params={'sl':bp*0.97,'tp':bp*1.03})
            self._in=False

# —— 策略：TripleTrend ——
class TripleTrendStrategy:
    def __init__(self):
        self.round_active=False; self._last=0
    async def check(self, price):
        now=time.time()
        if now-self._last<1: return
        df = data_mgr.klines["15m"]
        if len(df)<99: return
        # MA 前置
        ma7,ma25,ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        h,l,c = df.high.values, df.low.values, df.close.values
        _,d1 = numba_supertrend(h,l,c,10,1)
        _,d2 = numba_supertrend(h,l,c,11,2)
        _,d3 = numba_supertrend(h,l,c,12,3)
        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not (d1[-1] or d2[-1] or d3[-1])
        prev, curr = (d1[-2],d2[-2],d3[-2]), (d1[-1],d2[-1],d3[-1])
        flip_dn = self.round_active and any(p and not c2 for p,c2 in zip(prev,curr))
        flip_up = self.round_active and any(not p and c2 for p,c2 in zip(prev,curr))
        self._last=now
        if up_all and not self.round_active:
            self.round_active=True
            p0,sl,tp = price*0.995, price*0.97, price*1.02
            await mgr.safe_place("triple","BUY","LIMIT",qty=0.015,price=p0,extra_params={'sl':sl,'tp':tp})
        elif dn_all and not self.round_active:
            self.round_active=True
            p0,sl,tp = price*1.005, price*1.03, price*0.98
            await mgr.safe_place("triple","SELL","LIMIT",qty=0.015,price=p0,extra_params={'sl':sl,'tp':tp})
        elif flip_dn:
            await mgr.safe_place("triple","SELL","MARKET"); self.round_active=False
        elif flip_up:
            await mgr.safe_place("triple","BUY","MARKET"); self.round_active=False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WS 市场 ——
async def market_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=20) as ws:
                retry=0
                async for msg in ws:
                    o=json.loads(msg); s,d=o["stream"],o["data"]
                    if s.endswith("@markPrice"):
                        await data_mgr.track_price(float(d["p"]), int(time.time()*1000))
                    else:
                        tf = s.split("@")[1].split("_")[1]
                        k = d["k"]
                        await data_mgr.update_kline(tf,
                            float(k["o"]), float(k["h"]),
                            float(k["l"]), float(k["c"]), k["t"])
        except:
            await asyncio.sleep(min(2**retry,30)); retry+=1

# —— WS 用户 ——
async def user_ws():
    retry=0
    while True:
        try:
            async with websockets.connect(Config.WS_USER, ping_interval=None) as ws:
                retry=0
                ts=int(time.time()*1000+time_offset)
                p={"apiKey":Config.ED25519_API,"timestamp":ts}
                payload="&".join(f"{k}={v}" for k,v in sorted(p.items()))
                sig=base64.b64encode(ed_priv.sign(payload.encode())).decode()
                p["signature"]=sig
                await ws.send(json.dumps({"id":str(uuid.uuid4()),
                                          "method":"session.logon",
                                          "params":p}))
                async for msg in ws:
                    r=json.loads(msg)
                    if r.get("method") in ("order.update","executionReport"):
                        o=r.get("params",r.get("result",{}))
                        oid = o.get('orderId') or o.get('i')
                        st  = o.get('status')  or o.get('X')
                        await pos_tracker.on_order_update(oid, st)
        except:
            await asyncio.sleep(5); retry+=1

# —— 启动 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    await sync_time()
    await detect_mode()
    await load_symbol_filters()
    await data_mgr.load_history()
    await pos_tracker.sync()

    # 并行运行
    await asyncio.gather(
        market_ws(),
        user_ws(),
        maintenance(),
        engine(),
    )


if __name__ == '__main__':
    asyncio.run(main())