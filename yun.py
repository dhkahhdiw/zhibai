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
import watchfiles
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key

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
    SYMBOL         = 'ETHUSDC'
    PAIR           = SYMBOL.lower()
    API_KEY        = os.getenv('BINANCE_API_KEY')
    SECRET_KEY     = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API    = os.getenv('ED25519_API_KEY')
    ED25519_KEY    = os.getenv('ED25519_KEY_PATH')
    REST_BASE      = 'https://fapi.binance.com'
    WS_MARKET      = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_USER        = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    RECV_WINDOW    = 5000
    SYNC_INTERVAL  = 300
    MAX_POS        = 2.0
    HIST_LIMIT     = 1000
    ORDER_COOLDOWN = 60

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
    return math.floor(val/step) * step

# —— 时间同步 & 模式检测 ——
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
    data = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge = data.get('dualSidePosition', False)
    LOG.info(f"Hedge mode: {is_hedge}")

# —— 精度过滤器 ——
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
                res  = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                                         params={"symbol":Config.SYMBOL,"interval":tf,"limit":Config.HIST_LIMIT},
                                         headers={"X-MBX-APIKEY":Config.API_KEY})
                data = await res.json()
                df   = pd.DataFrame([{"open":float(x[1]),"high":float(x[2]),
                                      "low":float(x[3]),"close":float(x[4])} for x in data])
                self.klines[tf], self.last_ts[tf] = df, int(data[-1][0])
                self._compute(tf)
                LOG.info(f"{tf} history loaded: {len(df)}")

    async def update_kline(self, tf, o,h,l,c,ts):
        async with self.lock:
            df = self.klines[tf]
            if ts > self.last_ts[tf]:
                df.loc[len(df), ["open","high","low","close"]] = [o,h,l,c]
            else:
                df.iloc[-1, df.columns.get_indexer(["open","high","low","close"])] = [o,h,l,c]
            self.last_ts[tf] = ts
            self._compute(tf)
            # 日志
            if tf=="3m":  LOG.debug(f"3m BB%={df.bb_pct.iat[-1]:.4f}")
            if tf=="15m": LOG.debug(f"15m Supertrend={df.st.iat[-1]:.2f} MACD diff={df.macd.iat[-1]:.4f}")
            if tf=="1h":  LOG.debug(f"1h BB%={df.bb_pct.iat[-1]:.4f}")
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
            # 新增 MA7/25/99 SMA9 计算
            df["ma7"]  = df.close.rolling(7).mean()
            df["ma25"] = df.close.rolling(25).mean()
            df["ma99"] = df.close.rolling(99).mean()

    async def wait_update(self):
        await self._evt.wait(); self._evt.clear()

    async def track_price(self, p, ts):
        async with self.lock:
            self.price, self.ptime = p, ts
        LOG.debug(f"Price update: {p:.2f}@{ts}")
        self._evt.set()

data_mgr = DataManager()

# —— Supertrend via Numba ——
@jit(nopython=True)
def numba_supertrend(h,l,c,per,mult):
    n=len(c); st=[0.0]*n; dirc=[False]*n
    hl2=[(h[i]+l[i])/2 for i in range(n)]
    atr=[max(h[i-per+1:i+1]) - min(l[i-per+1:i+1]) for i in range(n)]
    up=[hl2[i] + mult*atr[i] for i in range(n)]
    dn=[hl2[i] - mult*atr[i] for i in range(n)]
    st[0], dirc[0] = up[0], True
    for i in range(1,n):
        if c[i] > st[i-1]:
            st[i], dirc[i] = max(dn[i], st[i-1]), True
        else:
            st[i], dirc[i] = min(up[i], st[i-1]), False
    return st, dirc

# —— 订单守卫 ——
class OrderGuard:
    def __init__(self):
        self.states   = defaultdict(dict)
        self.lock     = asyncio.Lock()
        self.cooldown = {
            "main": Config.ORDER_COOLDOWN,
            "macd": Config.ORDER_COOLDOWN//2,
            "triple": Config.ORDER_COOLDOWN*5
        }

    async def check(self, strat, fp):
        async with self.lock:
            st = self.states[strat]; now=time.time()
            if fp==st.get('last_fp') and now-st.get('ts',0)<self.cooldown[strat]:
                LOG.debug(f"{strat} 信号冷却中 {(self.cooldown[strat]-(now-st['ts'])):.1f}s")
                return False
            if now-st.get('ts',0)<self.cooldown[strat]:
                return False
            return True

    async def update(self, strat, fp):
        async with self.lock:
            self.states[strat]={'last_fp':fp,'ts':time.time()}

guard = OrderGuard()

# —— 会话管理 ——
async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def safe_place(self, strat, side, otype, qty=None, price=None, stop=None, params=None):
        fp = f"{strat}|{side}|{hash(frozenset((params or {}).items()))}"
        if not await guard.check(strat,fp): return
        await self.place(side,otype,qty,price,stop,sig_key=fp)
        await guard.update(strat,fp)

    async def place(self, side, otype, qty=None, price=None, stop=None, sig_key=None):
        await ensure_session()
        ts = int(time.time()*1000 + time_offset)
        params={"symbol":Config.SYMBOL,"side":side,"type":otype,
                "timestamp":ts,"recvWindow":Config.RECV_WINDOW}
        if qty   is not None: params["quantity"]  = f"{quantize(qty,qty_step):.{qty_prec}f}"
        if price is not None: params["price"]     = f"{quantize(price,price_step):.{price_prec}f}"
        if stop  is not None: params["stopPrice"] = f"{quantize(stop,price_step):.{price_prec}f}"
        if otype=="LIMIT": params["timeInForce"]="GTC"
        elif otype in ("STOP_MARKET","TAKE_PROFIT_MARKET"): params["closePosition"]="true"
        if is_hedge and otype in ("LIMIT","MARKET"):
            params["positionSide"]="LONG" if side=="BUY" else "SHORT"
        qs = urllib.parse.urlencode(sorted(params.items()))
        sig= hmac.new(Config.SECRET_KEY,qs.encode(),hashlib.sha256).hexdigest()
        url= f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"
        headers={'X-MBX-APIKEY':Config.API_KEY}
        async with self.lock:
            r  = await session.post(url,headers=headers)
            txt= await r.text()
            tag=f"[{sig_key}]" if sig_key else ""
            if r.status==200: LOG.info(f"Order OK {otype} {side} {tag}")
            else:            LOG.error(f"Order ERR {otype} {side} {tag} {txt}")

mgr = OrderManager()

# —— 策略：更新版 TripleTrend ——
class TripleTrendStrategy:
    def __init__(self):
        self.round_active = False    # 一轮买入卖出标志
        self.entry_price  = None
        self._last        = 0

    async def check(self, price):
        now=time.time()
        if now-self._last<1: return
        df=data_mgr.klines["15m"]
        if len(df)<99: return
        h,l,c=df.high.values,df.low.values,df.close.values
        # 三个 Supertrend 指标
        try:
            _,d1=numba_supertrend(h,l,c,10,1)
            _,d2=numba_supertrend(h,l,c,11,2)
            _,d3=numba_supertrend(h,l,c,12,3)
        except StopIteration:
            LOG.error("Numba supertrend StopIteration in TripleTrendStrategy, skip")
            return

        up_all   = d1[-1] and d2[-1] and d3[-1]
        dn_all   = not(d1[-1] or d2[-1] or d3[-1])
        flip_to_dn = self.round_active and any(prev and not cur for prev,cur in zip((d1[-2],d2[-2],d3[-2]), (d1[-1],d2[-1],d3[-1])))
        flip_to_up = self.round_active and any(not prev and cur for prev,cur in zip((d1[-2],d2[-2],d3[-2]), (d1[-1],d2[-1],d3[-1])))

        # MA 过滤条件
        ma7,ma25,ma99 = df.ma7.iat[-1],df.ma25.iat[-1],df.ma99.iat[-1]

        # 开盘条件：三线同时上升，且尚未激活本轮；并且价格<MA7<MA25<MA99
        if up_all and not self.round_active and price<ma7<ma25<ma99:
            self._last=now; self.round_active=True; self.entry_price=price
            order_price = price * 0.995
            LOG.info("TRIPLE enter long → LIMIT BUY @%.2f", order_price)
            await mgr.safe_place("triple","BUY","LIMIT",qty=0.15,price=order_price,params={"enter":"long"})
            # 固定止损
            sl = price*0.97
            await mgr.safe_place("triple","BUY","STOP_MARKET",stop=sl,params={"stop":"long"})

        # 开空条件：三线同时下降，且尚未激活本轮；且价格>MA7>MA25>MA99
        elif dn_all and not self.round_active and price>ma7>ma25>ma99:
            self._last=now; self.round_active=True; self.entry_price=price
            order_price = price * 1.005
            LOG.info("TRIPLE enter short → LIMIT SELL @%.2f", order_price)
            await mgr.safe_place("triple","SELL","LIMIT",qty=0.15,price=order_price,params={"enter":"short"})
            sl = price*1.03
            await mgr.safe_place("triple","SELL","STOP_MARKET",stop=sl,params={"stop":"short"})

        # 止盈多单：本轮已开多，任一指标转向下降，立即市价卖出
        elif self.round_active and self.entry_price and any(prev and not cur for prev,cur in zip((d1[-2],d2[-2],d3[-2]), (d1[-1],d2[-1],d3[-1]))):
            LOG.info("TRIPLE take profit long → MARKET SELL")
            await mgr.safe_place("triple","SELL","MARKET",params={"tp":"long"})
            self.round_active=False

        # 止盈空单：本轮已开空，任一指标转向上升，立即市价买入
        elif self.round_active and self.entry_price and any(not prev and cur for prev,cur in zip((d1[-2],d2[-2],d3[-2]), (d1[-1],d2[-1],d3[-1]))):
            LOG.info("TRIPLE take profit short → MARKET BUY")
            await mgr.safe_place("triple","BUY","MARKET",params={"tp":"short"})
            self.round_active=False

# —— 策略：主策略 ——
class MainStrategy:
    def __init__(self):
        self._last=0; self.interval=1
        self.ended=False

    async def check(self, price):
        now=time.time()
        if now-self._last<self.interval: return

        df15=data_mgr.klines["15m"]
        if len(df15)<99: return

        # 主趋势：15m supertrend
        h15,l15,c15=df15.high.values,df15.low.values,df15.close.values
        try:
            st_line,st_dir = numba_supertrend(h15,l15,c15,10,3)
        except StopIteration:
            LOG.error("Numba supertrend StopIteration in MainStrategy, skip"); return
        trend_up = price>st_line[-1] and st_dir[-1]
        trend_dn = price<st_line[-1] and not st_dir[-1]

        # 强弱：1h BB%
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong_long  = bb1<0.2
        strong_short = bb1>0.8

        # 3m BB% 信号
        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        if not (bb3<=0 or bb3>=1): return


        # 信号方向
        side=None
        if bb3<=0:  side="BUY"
        if bb3>=1:  side="SELL"

        # 仓位大小
        if strong_long or strong_short:
            # 强信号
            if side=="BUY":
                qty = 0.12 if side=="BUY" and trend_up else 0.07
            else:
                qty = 0.12 if side=="SELL" and trend_dn else 0.07
            levels = [0.0025,0.004,0.006,0.008,0.016]
            sizes = [qty*0.2]*5
            offsets = [(-l if side=="BUY" else l) for l in levels]
        else:
            # 弱信号
            if side=="BUY":
                qty = 0.03 if trend_up else 0.015
                if trend_up:
                    offsets = [-0.0055, -0.0155]; sizes=[qty*0.5]*2
                else:
                    offsets = [-0.0075]; sizes=[qty]
            else:
                qty = 0.03 if trend_dn else 0.015
                if trend_dn:
                    offsets = [0.0055,0.0155]; sizes=[qty*0.5]*2
                else:
                    offsets = [0.0075]; sizes=[qty]

        # 挂单
        for off,sz in zip(offsets,sizes):
            price0 = price*(1+off if side=="BUY" else 1-off)
            await mgr.safe_place("main",side,"LIMIT",qty=sz,price=price0)

        # 止盈挂单
        tps = [0.0102,0.0123,0.025,0.028,0.032]
        for tp in tps:
            price_tp = price*(1+tp if side=="BUY" else 1-tp)
            await mgr.safe_place("main",side,"LIMIT",qty=qty*0.2,price=price_tp,params={"tp":tp})

        # 止损
        sl = price*0.98 if side=="BUY" else price*1.02
        await mgr.safe_place("main",side,"STOP_MARKET",stop=sl,params={"SL":True})

        self._last=now

# —— 策略：15m MACD ——
class MACDStrategy:
    def __init__(self): self._in=False

    async def check(self, price):
        df=data_mgr.klines["15m"]
        if len(df)<30 or "macd" not in df: return
        prev,cur = df.macd.iat[-2],df.macd.iat[-1]
        if prev>0>cur and not self._in:
            sp=price*1.005; LOG.info("MACD death cross→SELL")
            await mgr.safe_place("macd","SELL","LIMIT",qty=0.15,price=sp)
            await mgr.safe_place("macd","SELL","STOP_MARKET",stop=sp*1.03)
            self._in=True
        elif prev<0<cur and self._in:
            bp=price*0.995; LOG.info("MACD golden cross→BUY")
            await mgr.safe_place("macd","BUY","LIMIT",qty=0.15,price=bp)
            await mgr.safe_place("macd","BUY","STOP_MARKET",stop=bp*0.97)
            self._in=False

strategies = [
    MainStrategy(),
    MACDStrategy(),
    TripleTrendStrategy(),
]

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
                        await data_mgr.track_price(float(d["p"]),int(time.time()*1000))
                    else:
                        tf=s.split("@")[1].split("_")[1]; k=d["k"]
                        await data_mgr.update_kline(tf,float(k["o"]),float(k["h"]),
                                                   float(k["l"]),float(k["c"]),k["t"])
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
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params["signature"]=sig
                await ws.send(json.dumps({"id":str(uuid.uuid4()),
                                           "method":"session.logon","params":params}))
                async for msg in ws:
                    r=json.loads(msg)
                    if r.get("result",{}).get("orderId"):
                        LOG.info("ExecReport %s",r["result"])
        except Exception as e:
            LOG.error("UserWS %s, retry in 5s",e); await asyncio.sleep(5); retry+=1

# —— 热重载 & 维护 ——
async def watch_reload():
    async for _ in watchfiles.awatch('/root/zhibai'):
        LOG.info("Reloading…"); load_dotenv('/root/zhibai/.env')
        await pos_tracker.sync()
async def maintenance():
    asyncio.create_task(watch_reload())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time(); await detect_mode(); await pos_tracker.sync()

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

# —— 策略引擎 & 入口 ——
async def engine():
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or time.time()-data_mgr.ptime>60: continue
        for strat in strategies:
            try: await strat.check(data_mgr.price)
            except Exception as e: LOG.exception("Strategy %s failed: %s", strat.__class__.__name__, e)

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