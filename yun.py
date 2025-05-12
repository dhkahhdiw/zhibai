#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import math
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
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519

# —— 高性能事件循环 & 环境加载 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv()  # 默认读取当前路径下的 .env

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
    BINANCE_API_KEY   = os.getenv('YZ_BINANCE_API_KEY', '').strip()
    BINANCE_SECRET_KEY= os.getenv('YZ_BINANCE_SECRET_KEY', '').strip().encode()
    ED25519_API_KEY   = os.getenv('YZ_ED25519_API_KEY', '').strip()
    ED25519_KEY_PATH  = os.getenv('YZ_ED25519_KEY_PATH', '').strip()
    REST_BASE         = 'https://fapi.binance.com'
    WS_MARKET         = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_USER_BASE      = 'wss://fstream.binance.com/ws/'
    RECV_WINDOW       = 5000
    SYNC_INTERVAL     = 60          # 每分钟对齐一次时间、模式和持仓
    ROTATION_COOLDOWN = 1800        # 策略同向冷却时间（秒）

# —— 全局状态 ——
session      = None
listen_key   = None
time_offset  = 0
is_hedge     = False
price_step   = qty_step = None
price_prec   = qty_prec = 0

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("✔ Ed25519 key loaded")

def quantize(val: float, step: float) -> float:
    """向下取整到合约精度"""
    return math.floor(val / step) * step

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    resp = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    data = await resp.json()
    server_ms = int(data['serverTime'])
    local_ms  = int(time.time() * 1000)
    time_offset = server_ms - local_ms
    LOG.info(f"✔ Time offset: {time_offset} ms")

async def detect_mode():
    """检测当前账户是否为对冲模式"""
    global is_hedge
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sign = hmac.new(Config.BINANCE_SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sign}"
    resp = await session.get(url, headers={'X-MBX-APIKEY': Config.BINANCE_API_KEY})
    data = await resp.json()
    # 部分账户接口返回 {'dualSidePosition': True/False}
    is_hedge = bool(data.get('dualSidePosition', False))
    LOG.info(f"✔ Hedge mode: {is_hedge}")

# —— 加载合约精度 ——
async def load_symbol_filters():
    global price_step, qty_step, price_prec, qty_prec
    resp = await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo")
    data = await resp.json()
    sym = next(s for s in data['symbols'] if s['symbol'] == Config.SYMBOL)
    pf  = next(f for f in sym['filters'] if f['filterType']=='PRICE_FILTER')
    lf  = next(f for f in sym['filters'] if f['filterType']=='LOT_SIZE')
    price_step = float(pf['tickSize'])
    qty_step   = float(lf['stepSize'])
    price_prec = int(round(-math.log10(price_step)))
    qty_prec   = int(round(-math.log10(qty_step)))
    LOG.info(f"✔ Filters: price_step={price_step}, qty_step={qty_step}")

# —— 持仓 & 止盈止损追踪 ——
class PositionTracker:
    class Position:
        __slots__ = ('cloid','side','qty','entry','sl_price','tp_price','active')
        def __init__(self, cloid, side, qty, entry, sl, tp):
            self.cloid     = cloid
            self.side      = side
            self.qty       = qty
            self.entry     = entry
            self.sl_price  = sl
            self.tp_price  = tp
            self.active    = True

    def __init__(self):
        self.positions = {}         # cloid -> Position
        self.order2cl  = {}         # orderId -> cloid
        self.lock       = asyncio.Lock()
        self.next_cloid = 1

    async def on_fill(self, order_id, side, qty, price, sl, tp):
        """限价单成交后，保存仓位并下止损/止盈单"""
        async with self.lock:
            cloid = self.next_cloid
            self.next_cloid += 1
            sl_q = quantize(sl, price_step)
            tp_q = quantize(tp, price_step)
            pos = self.Position(cloid, side, qty, price, sl_q, tp_q)
            self.positions[cloid] = pos
            self.order2cl[order_id] = cloid
            LOG.info(f"[PT] Opened cloid={cloid} side={side} entry={price:.5f} SL={sl_q:.5f} TP={tp_q:.5f}")

        # 根据模式准备止损/止盈单的参数
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        common = {
            'closePosition': 'true',
            'workingType':    'MARK_PRICE',
            'priceProtect':  'FALSE',
        }
        if is_hedge:
            common.update({
                'positionSide': 'LONG' if side=='BUY' else 'SHORT',
                'reduceOnly':   'true'
            })

        # 下 STOP_MARKET 单
        await mgr.place(
            close_side, 'STOP_MARKET',
            stop=pos.sl_price,
            extra_params=common
        )
        # 下 TAKE_PROFIT_MARKET 单
        await mgr.place(
            close_side, 'TAKE_PROFIT_MARKET',
            stop=pos.tp_price,
            extra_params=common
        )

    async def on_order_update(self, order_id, status):
        """监听远程订单更新，若仓位平掉则标记失效"""
        async with self.lock:
            if order_id not in self.order2cl:
                return
            cloid = self.order2cl[order_id]
            pos   = self.positions.get(cloid)
            if pos and status in ('FILLED','CANCELED','REJECTED'):
                pos.active = False
                LOG.info(f"[PT] Closed cloid={cloid} via {status}")

    async def check_trigger(self, price):
        """本地触发检查（备用）"""
        eps = price_step * 0.5
        async with self.lock:
            for pos in list(self.positions.values()):
                if not pos.active:
                    continue
                trig = False
                if pos.side == 'BUY':
                    if price <= pos.sl_price + eps:
                        LOG.info(f"[PT] Local SL trigger: {price:.5f} ≤ {pos.sl_price:.5f}")
                        trig = True
                    elif price >= pos.tp_price - eps:
                        LOG.info(f"[PT] Local TP trigger: {price:.5f} ≥ {pos.tp_price:.5f}")
                        trig = True
                else:
                    if price >= pos.sl_price - eps:
                        LOG.info(f"[PT] Local SL(short): {price:.5f} ≥ {pos.sl_price:.5f}")
                        trig = True
                    elif price <= pos.tp_price + eps:
                        LOG.info(f"[PT] Local TP(short): {price:.5f} ≤ {pos.tp_price:.5f}")
                        trig = True
                if trig:
                    await self._local_close(pos)

    async def _local_close(self, pos):
        side = 'SELL' if pos.side=='BUY' else 'BUY'
        LOG.info(f"[PT] Executing local MARKET close {side}@qty={pos.qty}")
        try:
            await mgr.place(side, 'MARKET', qty=pos.qty)
            pos.active = False
        except Exception as e:
            LOG.error(f"[PT] Local close failed: {e}")

    async def sync(self):
        """定期拉取远程持仓状态，用于日志和纠偏"""
        ts = int(time.time()*1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
        sig = hmac.new(Config.BINANCE_SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        resp = await session.get(url, headers={'X-MBX-APIKEY': Config.BINANCE_API_KEY})
        data = await resp.json()
        for p in data:
            if p['symbol'] == Config.SYMBOL:
                LOG.debug(f"[PT] Remote posAmt={p['positionAmt']} entry={p['entryPrice']}")

pos_tracker = PositionTracker()

# —— K 线数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs      = ["3m","15m","1h"]
        self.klines   = {tf: pd.DataFrame(columns=["open","high","low","close"]) for tf in self.tfs}
        self.last_ts  = {tf: 0 for tf in self.tfs}
        self.lock     = asyncio.Lock()
        self._evt     = asyncio.Event()
        self.price    = None
        self.ptime    = 0

    async def load_history(self):
        """启动时拉取历史 K 线"""
        async with self.lock:
            for tf in self.tfs:
                resp = await session.get(
                    f"{Config.REST_BASE}/fapi/v1/klines",
                    params={"symbol": Config.SYMBOL, "interval": tf, "limit": 1000},
                    headers={"X-MBX-APIKEY": Config.BINANCE_API_KEY}
                )
                arr = await resp.json()
                df = pd.DataFrame([{
                    "open":  float(x[1]),
                    "high":  float(x[2]),
                    "low":   float(x[3]),
                    "close": float(x[4])
                } for x in arr])
                self.klines[tf]  = df
                self.last_ts[tf]= int(arr[-1][0])
                self._compute(tf)
                LOG.info(f"[DM] Loaded {tf} {len(df)} bars")

    async def update_kline(self, tf, o,h,l,c,ts):
        """收到 WebSocket 最新 K 线数据后更新 DataFrame"""
        async with self.lock:
            df = self.klines[tf]
            if ts > self.last_ts[tf]:
                # 新开一根
                df.loc[len(df)] = [o,h,l,c]
            else:
                # 更新最后一根
                df.loc[len(df)-1, ["open","high","low","close"]] = [o,h,l,c]
            self.last_ts[tf] = ts
            self._compute(tf)
            self._evt.set()

    async def track_price(self, price, ts):
        """最新标记价"""
        async with self.lock:
            self.price = price
            self.ptime = ts
        self._evt.set()
        await pos_tracker.check_trigger(price)

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

    def _compute(self, tf):
        """计算指标：布林带、超级趋势、MACD、移动均线等"""
        df = self.klines[tf]
        if len(df) < 20:
            return
        # 布林带
        m = df.close.rolling(20).mean()
        s = df.close.rolling(20).std()
        df["bb_up"]  = m + 2*s
        df["bb_dn"]  = m - 2*s
        df["bb_pct"] = (df.close - df.bb_dn)/(df.bb_up - df.bb_dn)
        if tf == "15m":
            hl2 = (df.high + df.low) / 2
            atr = df.high.rolling(10).max() - df.low.rolling(10).min()
            df["st_line"], df["st_dir"] = numba_supertrend(
                df.high.values, df.low.values, df.close.values, 10, 3
            )
            macd = MACD(df.close, 12, 26, 9)
            df["macd"]  = macd.macd_diff()
            df["ma7"]   = df.close.rolling(7).mean()
            df["ma25"]  = df.close.rolling(25).mean()
            df["ma99"]  = df.close.rolling(99).mean()

@jit(nopython=True)
def numba_supertrend(h, l, c, per, mult):
    n = len(c)
    st_line = [0.0]*n
    dirc    = [False]*n
    hl2     = [(h[i]+l[i])/2 for i in range(n)]
    atr     = [max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up      = [hl2[i] + mult*atr[i] for i in range(n)]
    dn      = [hl2[i] - mult*atr[i] for i in range(n)]
    st_line[0], dirc[0] = up[0], True
    for i in range(1,n):
        if c[i] > st_line[i-1]:
            st_line[i] = max(dn[i], st_line[i-1])
            dirc[i]     = True
        else:
            st_line[i] = min(up[i], st_line[i-1])
            dirc[i]     = False
    return st_line, dirc

data_mgr = DataManager()

# —— 策略防护：去重 & 冷却 ——
class OrderGuard:
    def __init__(self):
        self.states   = defaultdict(lambda: {'ts':0,'fp':None,'trend':None})
        self.lock     = asyncio.Lock()
        self.cooldown = Config.ROTATION_COOLDOWN

    async def check(self, strat, fp, trend):
        async with self.lock:
            st = self.states[strat]
            now = time.time()
            if st['fp']==fp and now - st['ts'] < self.cooldown:
                return False
            if st['trend']==trend and now - st['ts'] < self.cooldown:
                return False
            return True

    async def update(self, strat, fp, trend):
        async with self.lock:
            self.states[strat] = {'fp':fp,'trend':trend,'ts':time.time()}

guard = OrderGuard()

# —— HTTP 会话保活 ——
async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 用户流 ListenKey 获取 & 保活 ——
async def get_listen_key():
    """创建或刷新 listenKey"""
    resp = await session.post(
        f"{Config.REST_BASE}/fapi/v1/listenKey",
        headers={'X-MBX-APIKEY': Config.BINANCE_API_KEY}
    )
    data = await resp.json()
    return data['listenKey']

async def keepalive_listen_key():
    global listen_key
    while True:
        await asyncio.sleep(1800)
        try:
            await session.put(
                f"{Config.REST_BASE}/fapi/v1/listenKey",
                headers={'X-MBX-APIKEY': Config.BINANCE_API_KEY},
                json={'listenKey': listen_key}
            )
            LOG.info("✔ listenKey refreshed")
        except Exception as e:
            LOG.error(f"Failed refresh listenKey: {e}")

# —— 订单下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def safe_place(self, strat, side, otype, qty=None, price=None,
                         stop=None, extra_params=None):
        """带去重 & 冷却保护的下单入口"""
        fp    = f"{side}:{otype}:{price or stop}"
        trend = 'LONG' if side=='BUY' else 'SHORT'
        if not await guard.check(strat, fp, trend):
            return
        LOG.info(f"[Mgr][{strat}] {side} {otype} price={price} stop={stop} qty={qty}")
        await self.place(side, otype, qty, price, stop, extra_params or {}, fp)
        await guard.update(strat, fp, trend)

    async def place(self, side, otype, qty=None, price=None,
                    stop=None, extra_params=None, sig_key=None):
        """直接调用 Binance REST 下单"""
        await ensure_session()
        ts     = int(time.time()*1000 + time_offset)
        params = {
            'symbol': Config.SYMBOL,
            'side':   side,
            'type':   otype,
            'timestamp': ts,
            'recvWindow': Config.RECV_WINDOW
        }
        if qty   is not None:
            params['quantity'] = f"{quantize(qty, qty_step):.{qty_prec}f}"
        if price is not None:
            params['price']    = f"{quantize(price, price_step):.{price_prec}f}"
        if stop  is not None:
            params['stopPrice']= f"{quantize(stop, price_step):.{price_prec}f}"

        # 附加双向持仓或关闭仓位标志
        if otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
            params.update({
                'closePosition': 'true',
                'workingType':   'MARK_PRICE',
                'priceProtect': 'FALSE'
            })
            if is_hedge:
                params['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'
                params['reduceOnly']   = 'true'
        elif otype in ('LIMIT','MARKET') and is_hedge:
            params['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'

        # 合并额外参数 sl/tp 用于 on_fill
        if extra_params:
            params.update(extra_params)

        # 签名
        qs  = urllib.parse.urlencode(sorted(params.items()))
        sig = hmac.new(Config.BINANCE_SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"

        async with self.lock:
            resp = await session.post(url, headers={'X-MBX-APIKEY': Config.BINANCE_API_KEY})
            text = await resp.text()
            try:
                data = await resp.json()
            except:
                LOG.error(f"[Mgr] 非 JSON 返回: {text}")
                return

        if resp.status != 200:
            LOG.error(f"[Mgr] 下单失败 {otype} {side}: {data}")
            return

        LOG.info(f"[Mgr] 下单成功 id={data.get('orderId')}")
        # 若为限价单且部分或完全成交，回调 on_fill
        executed = float(data.get('executedQty', 0))
        if sig_key and otype=='LIMIT' and executed>0:
            sl = extra_params.get('slPrice') or extra_params.get('stopPrice')
            tp = extra_params.get('tpPrice') or extra_params.get('stopPrice')
            await pos_tracker.on_fill(
                data['orderId'], side, executed,
                float(data.get('price', 0)), sl, tp
            )

mgr = OrderManager()

# —— 策略实现 ——
class MainStrategy:
    def __init__(self):
        self.interval = 1
        self.last_run = 0

    async def check(self, price):
        now = time.time()
        if now - self.last_run < self.interval:
            return
        df15 = data_mgr.klines['15m']
        if len(df15) < 100:
            return

        bb3 = data_mgr.klines['3m'].bb_pct.iloc[-1]
        if not (bb3 <= 0 or bb3 >= 1):
            return
        side = 'BUY' if bb3 <= 0 else 'SELL'
        bb1 = data_mgr.klines['1h'].bb_pct.iloc[-1]
        strong = bb1 < 0.2 or bb1 > 0.8

        # 计算下单档位与数量
        if strong:
            base_qty = 0.12 if ((side=='BUY' and price>df15['st_line'].iloc[-1]) or
                                (side=='SELL' and price<df15['st_line'].iloc[-1])) else 0.07
            levels   = [0.0025, 0.004, 0.006, 0.008, 0.016]
            sizes    = [base_qty*0.2]*5
        else:
            if side=='BUY':
                if price > df15['st_line'].iloc[-1]:
                    levels, sizes = [-0.0055, -0.0155], [0.015, 0.015]
                else:
                    levels, sizes = [-0.0075], [0.015]
            else:
                if price < df15['st_line'].iloc[-1]:
                    levels, sizes = [0.0055, 0.0155], [0.015, 0.015]
                else:
                    levels, sizes = [0.0075], [0.015]

        # 依次下限价单，并统一设置 sl/tp
        for lvl, sz in zip(levels, sizes):
            px = price * (1 + lvl if side=='BUY' else 1 - lvl)
            sl = price * (0.98 if side=='BUY' else 1.02)
            tp = price * (1.01 if side=='BUY' else 0.99)
            await mgr.safe_place(
                'main', side, 'LIMIT',
                qty=sz, price=px,
                extra_params={'slPrice': sl, 'tpPrice': tp}
            )

        # 最后一档止损止损单
        sl0 = price * (0.98 if side=='BUY' else 1.02)
        await mgr.safe_place('main', side, 'STOP_MARKET', stop=sl0)

        self.last_run = now

class MACDStrategy:
    def __init__(self):
        self.in_pos = False

    async def check(self, price):
        df = data_mgr.klines['15m']
        if len(df) < 30 or 'macd' not in df:
            return
        c7, c25, c99 = df.ma7.iloc[-1], df.ma25.iloc[-1], df.ma99.iloc[-1]
        if not ((price < c7 < c25 < c99) or (price > c7 > c25 > c99)):
            return
        prev, curr = df.macd.iloc[-2], df.macd.iloc[-1]
        if prev > 0 > curr and not self.in_pos:
            px = price * 1.005
            await mgr.safe_place(
                'macd', 'SELL', 'LIMIT',
                qty=0.026, price=px,
                extra_params={'slPrice': px*1.03, 'tpPrice': px*0.97}
            )
            self.in_pos = True
        elif prev < 0 < curr and self.in_pos:
            px = price * 0.995
            await mgr.safe_place(
                'macd', 'BUY', 'LIMIT',
                qty=0.026, price=px,
                extra_params={'slPrice': px*0.97, 'tpPrice': px*1.03}
            )
            self.in_pos = False

class TripleTrendStrategy:
    def __init__(self):
        self.active = False
        self.last_run= 0

    async def check(self, price):
        now = time.time()
        if now - self.last_run < 1:
            return
        df = data_mgr.klines['15m']
        if len(df) < 100:
            return
        c7, c25, c99 = df.ma7.iloc[-1], df.ma25.iloc[-1], df.ma99.iloc[-1]
        if not ((price < c7 < c25 < c99) or (price > c7 > c25 > c99)):
            return

        h, l, c = df.high.values, df.low.values, df.close.values
        _, d1 = numba_supertrend(h,l,c,10,1)
        _, d2 = numba_supertrend(h,l,c,11,2)
        _, d3 = numba_supertrend(h,l,c,12,3)
        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = not (d1[-1] or d2[-1] or d3[-1])
        prev = (d1[-2], d2[-2], d3[-2])
        curr = (d1[-1], d2[-1], d3[-1])
        flip_dn = self.active and any(p and not c2 for p, c2 in zip(prev,curr))
        flip_up = self.active and any(not p and c2 for p, c2 in zip(prev,curr))

        if up_all and not self.active:
            px = price * 0.999
            sl = price * 0.97
            tp = price * 1.02
            await mgr.safe_place(
                'triple','BUY','LIMIT', qty=0.015, price=px,
                extra_params={'slPrice': sl, 'tpPrice': tp}
            )
            self.active = True
        elif dn_all and not self.active:
            px = price * 1.001
            sl = price * 1.03
            tp = price * 0.98
            await mgr.safe_place(
                'triple','SELL','LIMIT', qty=0.015, price=px,
                extra_params={'slPrice': sl, 'tpPrice': tp}
            )
            self.active = True
        elif flip_dn:
            await mgr.safe_place('triple','SELL','MARKET')
            self.active = False
        elif flip_up:
            await mgr.safe_place('triple','BUY','MARKET')
            self.active = False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 主循环 ——
async def market_ws():
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry = 0
                async for msg in ws:
                    msg = json.loads(msg)
                    stream = msg['stream']
                    data   = msg['data']
                    if stream.endswith('@markPrice'):
                        price = float(data['p'])
                        await data_mgr.track_price(price, int(time.time()*1000))
                    else:
                        tf = stream.split('@')[1].split('_')[1]
                        k  = data['k']
                        await data_mgr.update_kline(
                            tf,
                            float(k['o']), float(k['h']),
                            float(k['l']), float(k['c']),
                            k['t']
                        )
        except Exception as e:
            delay = min(2**retry, 30)
            LOG.error(f"[WS MKT] {e}, reconnect in {delay}s")
            await asyncio.sleep(delay)
            retry += 1

async def user_ws():
    global listen_key
    retry = 0
    while True:
        try:
            listen_key = await get_listen_key()
            url = Config.WS_USER_BASE + listen_key
            async with websockets.connect(url, ping_interval=None) as ws:
                LOG.info("✔ Connected user stream")
                retry = 0
                async for msg in ws:
                    evt = json.loads(msg)
                    if evt.get('e') == 'ORDER_TRADE_UPDATE':
                        o = evt['o']
                        await pos_tracker.on_order_update(o['i'], o['X'])
        except Exception as e:
            LOG.error(f"[WS USER] {e}, reconnect in 5s")
            await asyncio.sleep(5)
            retry += 1

async def maintenance():
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await detect_mode()
        await pos_tracker.sync()
        if data_mgr.price is not None:
            await pos_tracker.check_trigger(data_mgr.price)

async def engine():
    while True:
        await data_mgr.wait_update()
        # 避免指标过旧
        if data_mgr.price is None or time.time() - data_mgr.ptime > 60:
            continue
        for strat in strategies:
            try:
                await strat.check(data_mgr.price)
            except Exception:
                LOG.exception(f"Strategy {strat.__class__.__name__} error")

async def main():
    global session
    session = aiohttp.ClientSession()
    # 优雅关闭
    loop = asyncio.get_running_loop()
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

if __name__ == '__main__':
    asyncio.run(main())