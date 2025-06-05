#!/usr/bin/env python3
# coding: utf-8

"""
Quant Trading Bot — 完整脚本（含子策略、止盈止损、OCO 逻辑）
环境：Ubuntu 22.04，Python 3.10+
依赖：
    pip install aiohttp websockets uvloop python-dotenv ta pandas numba cryptography
"""

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

import uvloop
import aiohttp
import websockets
import pandas as pd

from ta.trend import MACD, ADXIndicator
from numba import jit
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from collections import defaultdict

# —— 高性能事件循环 & 环境加载 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
# 修改为你的 .env 文件路径
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
    API_KEY           = os.getenv('YZ_BINANCE_API_KEY', '')
    SECRET_KEY        = os.getenv('YZ_BINANCE_SECRET_KEY', '').encode()
    ED25519_API       = os.getenv('YZ_ED25519_API_KEY', '')
    ED25519_KEY_PATH  = os.getenv('YZ_ED25519_KEY_PATH', '')
    REST_BASE         = 'https://fapi.binance.com'
    WS_MARKET         = (
        f"wss://fstream.binance.com/stream?streams="
        f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
    )
    WS_USER_BASE      = 'wss://fstream.binance.com/ws/'
    RECV_WINDOW       = 5000
    SYNC_INTERVAL     = 60            # 同步间隔（秒）
    ROTATION_COOLDOWN = 1800          # 子策略轮换冷却时间：30 分钟

# —— 全局状态 ——
session     = None
listen_key  = None
time_offset = 0
is_hedge    = False     # 对冲模式开关
price_step  = qty_step  = None
price_prec  = qty_prec  = 0

# —— 加载 Ed25519 私钥（如有需要签名 WebSocket 上层消息） ——
ed_priv = None
if Config.ED25519_KEY_PATH:
    try:
        with open(Config.ED25519_KEY_PATH, 'rb') as f:
            ed_priv = load_pem_private_key(f.read(), password=None)
        LOG.info("Ed25519 key loaded successfully.")
    except Exception as e:
        LOG.error(f"Failed to load Ed25519 key: {e}")
else:
    LOG.warning("No ED25519_KEY_PATH specified; skipping Ed25519 key load.")

def quantize(val: float, step: float) -> float:
    """
    将 val 向下取整到最接近 step 的整数倍。
    """
    return math.floor(val / step) * step

# —— 时间同步 & 模式检测 ——
async def sync_time():
    """
    同步服务器时间，计算并保存 time_offset（毫秒级）。
    """
    global time_offset
    try:
        await ensure_session()
        r = await session.get(f"{Config.REST_BASE}/fapi/v1/time", timeout=5)
        data = await r.json()
        time_offset = data['serverTime'] - int(time.time() * 1000)
        LOG.debug(f"[sync_time] Time offset: {time_offset} ms")
    except Exception as e:
        LOG.error(f"[sync_time] Error fetching server time: {e}")

async def detect_mode():
    """
    检测当前账户是否为对冲模式（dual-side position）。
    """
    global is_hedge
    try:
        await ensure_session()
        ts = int(time.time() * 1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
        r = await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY}, timeout=5)
        data = await r.json()
        # dualSidePosition 为 True 表示对冲模式
        is_hedge = data.get('dualSidePosition', False)
        LOG.debug(f"[detect_mode] Hedge mode: {is_hedge}")
    except Exception as e:
        LOG.error(f"[detect_mode] Error detecting hedge mode: {e}")

# —— 加载交易对精度（PRICE_FILTER & LOT_SIZE） ——
async def load_symbol_filters():
    """
    从 /exchangeInfo 拿到 SYMBOL 对应的 tickSize 和 stepSize，以便下单时正确量化价格和数量。
    """
    global price_step, qty_step, price_prec, qty_prec
    try:
        await ensure_session()
        r = await session.get(f"{Config.REST_BASE}/fapi/v1/exchangeInfo", timeout=5)
        info = await r.json()
        sym = next(s for s in info['symbols'] if s['symbol'] == Config.SYMBOL)
        pf  = next(f for f in sym['filters'] if f['filterType'] == 'PRICE_FILTER')
        ls  = next(f for f in sym['filters'] if f['filterType'] == 'LOT_SIZE')
        price_step = float(pf['tickSize'])
        qty_step   = float(ls['stepSize'])
        price_prec = int(-math.log10(price_step) + 0.5)
        qty_prec   = int(-math.log10(qty_step) + 0.5)
        LOG.debug(f"[load_symbol_filters] price_step={price_step}, qty_step={qty_step}, "
                  f"price_prec={price_prec}, qty_prec={qty_prec}")
    except Exception as e:
        LOG.error(f"[load_symbol_filters] Error loading symbol filters: {e}")

# —— 下单管理 ——
class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    async def safe_place(self, strat: str, side: str, otype: str,
                         qty: float = None, price: float = None,
                         stop: float = None, extra_params: dict = None):
        """
        “安全下单”：每个策略同一方向+类型+参数组合，需要冷却时间，不重复发单。
        strat: 策略名（比如 "main"）
        otype: "LIMIT", "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET" 等
        extra_params: 除数量/价格外的额外参数（必须包含 'sl'/'tp' 或 closePosition）
        """
        if extra_params is None:
            extra_params = {}
        fp = f"{strat}|{side}|{otype}|{hash(frozenset(extra_params.items()))}"
        trend = 'LONG' if side == 'BUY' else 'SHORT'

        if not await guard.check(strat, fp, trend):
            return

        await self.place(side, otype, qty, price, stop, extra_params=extra_params)
        await guard.update(strat, fp, trend)

    async def place(self, side: str, otype: str,
                    qty: float = None, price: float = None,
                    stop: float = None, extra_params: dict = None):
        """
        直接下单，对应 /fapi/v1/order 接口。
        extra_params 中可包含 'sl', 'tp' 或 'closePosition': 'true' 等字段。
        """
        try:
            await ensure_session()
            ts = int(time.time() * 1000 + time_offset)
            params = {
                "symbol": Config.SYMBOL,
                "side": side,
                "type": otype,
                "timestamp": ts,
                "recvWindow": Config.RECV_WINDOW
            }

            # —— 参数拼装 —— #
            if otype == "LIMIT":
                if qty is not None:
                    params["quantity"] = f"{quantize(qty, qty_step):.{qty_prec}f}"
                if price is not None:
                    params["price"] = f"{quantize(price, price_step):.{price_prec}f}"
                params["timeInForce"] = "GTC"

            elif otype == "MARKET":
                if qty is not None:
                    params["quantity"] = f"{quantize(qty, qty_step):.{qty_prec}f}"

            elif otype in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                # STOP_MARKET / TAKE_PROFIT_MARKET 必须有 stopPrice，并且要么 closePosition=true 要么 quantity
                if stop is not None:
                    params["stopPrice"] = f"{quantize(stop, price_step):.{price_prec}f}"
                    params["workingType"] = "MARK_PRICE"
                    params["priceProtect"] = "FALSE"
                if extra_params.get("closePosition") == 'true':
                    params["closePosition"] = "true"
                else:
                    if qty is not None:
                        params["quantity"] = f"{quantize(qty, qty_step):.{qty_prec}f}"
                extra_params.pop("closePosition", None)

            # —— 对冲模式下，追加 positionSide & reduceOnly —— #
            if is_hedge and otype in ("LIMIT", "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"):
                params["positionSide"] = "LONG" if side == "BUY" else "SHORT"
                if otype != "LIMIT":
                    params["reduceOnly"] = "true"

            # —— 合并额外参数 —— #
            if extra_params:
                params.update(extra_params)

            LOG.debug(f"[Mgr.place] Order params: {json.dumps(params)}")
            qs = urllib.parse.urlencode(sorted(params.items()))
            sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
            url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"

            async with self.lock:
                r = await session.post(url, headers={'X-MBX-APIKEY': Config.API_KEY}, timeout=5)
                text = await r.text()
                try:
                    data = await r.json()
                except:
                    LOG.error(f"[Mgr.place] Non-JSON resp: {text}")
                    return

                if r.status != 200:
                    LOG.error(f"[Mgr.place] ERR {otype} {side} {r.status}: {data}")
                    return

                LOG.debug(f"[Mgr.place] OK  {otype} {side}: {data}")
                # 如果是限价单，且已经有部分成交，则当作开仓完成，调用 pos_tracker.on_fill()
                if otype == "LIMIT" and float(data.get('executedQty', 0)) > 0:
                    await pos_tracker.on_fill(
                        data['orderId'], side,
                        float(data['executedQty']),
                        float(data.get('price', 0)),
                        extra_params.get('sl'),
                        extra_params.get('tp')
                    )
        except Exception as e:
            LOG.error(f"[Mgr.place] Exception in place(): {e}")

    async def batch_place(self, orders: list[dict]) -> list:
        """
        同步提交多笔订单：POST /fapi/v1/batchOrders
        orders: List[dict]，每个 dict 至少要有：
          - type, side, stopPrice（如果是止盈/止损挂单）或 price（如果是限价单）、closePosition 或 quantity
          - newClientOrderId: 自定义 clientOrderId，用于后续 OCO 逻辑
        """
        try:
            await ensure_session()
            ts = int(time.time() * 1000 + time_offset)
            # 先在每个 order 中填充 symbol、positionSide、reduceOnly、workingType、priceProtect
            payload_list = []
            for o in orders:
                base = {
                    "symbol": Config.SYMBOL,
                    "side": o["side"],
                    "type": o["type"],
                    "newClientOrderId": o.get("newClientOrderId", ""),
                    "timestamp": ts,
                    "recvWindow": Config.RECV_WINDOW
                }
                # 如果对冲模式，需加 positionSide 和 reduceOnly
                if is_hedge:
                    base["positionSide"] = "LONG" if o["side"] == "BUY" else "SHORT"
                    if o["type"] in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                        base["reduceOnly"] = "true"
                # 添加 stopPrice 或 price 或 quantity
                if o["type"] in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                    base["stopPrice"] = o["stopPrice"]
                    base["workingType"] = "MARK_PRICE"
                    base["priceProtect"] = "FALSE"
                    if o.get("closePosition") == "true":
                        base["closePosition"] = "true"
                    else:
                        base["quantity"] = o.get("quantity", "")
                else:
                    # LIMIT 或 MARKET
                    if o.get("price"):
                        base["price"] = o["price"]
                    if o.get("quantity"):
                        base["quantity"] = o["quantity"]
                    if o["type"] == "LIMIT":
                        base["timeInForce"] = "GTC"
                payload_list.append(base)

            payload = {
                "batchOrders": json.dumps(payload_list)
            }
            # 构造签名：batchOrders、recvWindow、timestamp 一起签
            qs = urllib.parse.urlencode({
                "batchOrders": payload["batchOrders"],
                "timestamp": ts,
                "recvWindow": Config.RECV_WINDOW
            })
            sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
            url = f"{Config.REST_BASE}/fapi/v1/batchOrders?{qs}&signature={sig}"

            async with self.lock:
                r = await session.post(url, headers={'X-MBX-APIKEY': Config.API_KEY}, timeout=5)
                text = await r.text()
                try:
                    data = await r.json()
                except:
                    LOG.error(f"[Mgr.batch_place] Non-JSON resp: {text}")
                    return None

                if r.status != 200:
                    LOG.error(f"[Mgr.batch_place] ERR {r.status}: {data}")
                    return None

                LOG.debug(f"[Mgr.batch_place] OK: {data}")
                return data
        except Exception as e:
            LOG.error(f"[Mgr.batch_place] Exception: {e}")
            return None

    async def cancel(self, order_id: int = None, client_order_id: str = None):
        """
        撤销单笔订单：DELETE /fapi/v1/order
        必须传入 orderId 或 origClientOrderId
        """
        if order_id is None and client_order_id is None:
            LOG.error("[Mgr.cancel] order_id & client_order_id cannot both be None")
            return

        try:
            await ensure_session()
            ts = int(time.time() * 1000 + time_offset)
            params = {
                "symbol": Config.SYMBOL,
                "timestamp": ts,
                "recvWindow": Config.RECV_WINDOW
            }
            if order_id is not None:
                params["orderId"] = order_id
            else:
                params["origClientOrderId"] = client_order_id

            qs = urllib.parse.urlencode(params)
            sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
            url = f"{Config.REST_BASE}/fapi/v1/order?{qs}&signature={sig}"

            async with self.lock:
                r = await session.delete(url, headers={'X-MBX-APIKEY': Config.API_KEY}, timeout=5)
                text = await r.text()
                try:
                    data = await r.json()
                except:
                    LOG.error(f"[Mgr.cancel] Non-JSON resp: {text}")
                    return

                if r.status != 200:
                    LOG.error(f"[Mgr.cancel] ERR CANCEL {r.status}: {data}")
                    return

                LOG.debug(f"[Mgr.cancel] OK: {data}")
                return data
        except Exception as e:
            LOG.error(f"[Mgr.cancel] Exception: {e}")
            return

mgr = OrderManager()

# —— 持仓 & 止盈止损 追踪 ——
class PositionTracker:
    class Position:
        __slots__ = ('cloid', 'side', 'qty', 'sl_price', 'tp_price',
                     'active', 'sl_oid', 'tp_oid', 'entry_price')

        def __init__(self, cloid, side, qty, entry_price, sl, tp):
            self.cloid       = cloid         # 我们自己生成的“客户端委托 ID”
            self.side        = side          # 'BUY' / 'SELL'
            self.qty         = qty           # 成交数量
            self.entry_price = entry_price   # 开仓价格
            self.sl_price    = sl            # 预设止损价（已量化）
            self.tp_price    = tp            # 预设止盈价（已量化）
            self.active      = True
            self.sl_oid      = None          # 记录 STOP_MARKET 订单 ID
            self.tp_oid      = None          # 记录 TAKE_PROFIT_MARKET 订单 ID

    def __init__(self):
        self.positions = {}       # cloid -> Position
        self.orders    = {}       # 交易所 orderId -> cloid
        self.lock      = asyncio.Lock()
        self.next_id   = 1

    async def on_fill(self, oid: int, side: str,
                      qty: float, price: float,
                      sl: float, tp: float):
        """
        有一笔限价单(主单)在撮合成交后调用：
        1. 记录新仓位（cloid 自增）
        2. 发起对应 SL/TP 的 STOP_MARKET / TAKE_PROFIT_MARKET 挂单（OCO 逻辑）
        """
        async with self.lock:
            cid = self.next_id
            self.next_id += 1

            # 量化 sl, tp 到 price_step 的整数倍
            sl_q = quantize(sl, price_step)
            tp_q = quantize(tp, price_step)

            pos = self.Position(cid, side, qty, price, sl_q, tp_q)
            self.positions[cid] = pos
            self.orders[oid]    = cid

            LOG.info(f"[PT.on_fill] Opened cloid={cid} side={side} entry={price:.4f} sl={sl_q:.4f} tp={tp_q:.4f}")

        # 异步去挂 SL/TP 挂单
        asyncio.create_task(self._place_sl_tp(side, sl_q, tp_q, cid))

    async def _place_sl_tp(self, side: str,
                           sl_price: float, tp_price: float,
                           cloid: int):
        """
        为一个新仓位下止损和止盈挂单。使用 OCO 逻辑：如果一单成交，就自动撤销另一单。
        """
        base = {
            "side": "SELL" if side == "BUY" else "BUY",  # 平仓方向
        }
        # 如果对冲模式，给出 positionSide 和 reduceOnly
        if is_hedge:
            base["positionSide"] = "LONG" if side == "BUY" else "SHORT"
            base["reduceOnly"]   = "true"

        # 组装两个子单
        orders = [
            {
                "type": "STOP_MARKET",
                **base,
                "stopPrice": f"{sl_price:.{price_prec}f}",
                "closePosition": "true",
                "newClientOrderId": f"sl_{cloid}"
            },
            {
                "type": "TAKE_PROFIT_MARKET",
                **base,
                "stopPrice": f"{tp_price:.{price_prec}f}",
                "closePosition": "true",
                "newClientOrderId": f"tp_{cloid}"
            }
        ]
        result = await mgr.batch_place(orders)
        if result:
            for o in result:
                cid_str = o.get("clientOrderId", "")
                oid     = o.get("orderId")
                if cid_str.startswith(f"sl_{cloid}"):
                    self.positions[cloid].sl_oid = oid
                elif cid_str.startswith(f"tp_{cloid}"):
                    self.positions[cloid].tp_oid = oid

    async def on_order_update(self, oid: int, status: str):
        """
        用户流回调：某个子单（止损或止盈）状态更新时触发：
        1. 如果子单 FILLED，那么当作 OCO，撤销另一边挂单；
        2. 如果是主单（限价开仓单）被 CANCELED/FILLED，标记 Position 为非活跃。
        """
        async with self.lock:
            # —— 首先，看是否是 SL/TP 子单 → OCO 逻辑 —— #
            for cid, pos in self.positions.items():
                if oid == pos.sl_oid and status == "FILLED":
                    pos.active = False
                    other = pos.tp_oid
                    if other:
                        await mgr.cancel(order_id=other)
                    LOG.info(f"[PT.on_order_update] OCO triggered cloid={cid}, SL filled ({oid}), canceled TP ({other})")
                    return
                if oid == pos.tp_oid and status == "FILLED":
                    pos.active = False
                    other = pos.sl_oid
                    if other:
                        await mgr.cancel(order_id=other)
                    LOG.info(f"[PT.on_order_update] OCO triggered cloid={cid}, TP filled ({oid}), canceled SL ({other})")
                    return

            # —— 再看是否是主单（限价开仓单）的状态更新 —— #
            if oid in self.orders and status in ('FILLED', 'CANCELED'):
                cid = self.orders[oid]
                self.positions[cid].active = False
                LOG.info(f"[PT.on_order_update] Main order closed cloid={cid} via status={status}")

    async def check_trigger(self, price: float):
        """
        每次收到最新标记价格时调用。对于所有“仍然活跃”的本地持仓，检查是否触及 SL/TP：
        如果触及，就：
          1) 撤销正在排队的那一边挂单（SL 或 TP）
          2) 立刻发一个市价单去平仓，确保确实出场。
        """
        eps = price_step * 0.5
        async with self.lock:
            for cid, pos in list(self.positions.items()):
                if not pos.active:
                    continue

                # BUY 方向仓位，如果 price <= SL；或 price >= TP，则触发；SELL 方向相反
                hit_sl = (price <= pos.sl_price + eps) if pos.side == 'BUY' else (price >= pos.sl_price - eps)
                hit_tp = (price >= pos.tp_price - eps) if pos.side == 'BUY' else (price <= pos.tp_price + eps)

                if hit_sl or hit_tp:
                    LOG.info(f"[PT.check_trigger] Local trigger cloid={cid} side={pos.side} price={price:.4f}")
                    # 撤销另一边挂单
                    if hit_sl and pos.tp_oid:
                        await mgr.cancel(order_id=pos.tp_oid)
                    elif hit_tp and pos.sl_oid:
                        await mgr.cancel(order_id=pos.sl_oid)

                    # 市价平仓
                    close_side = "SELL" if pos.side == "BUY" else "BUY"
                    await mgr.safe_place("ptm", close_side, "MARKET", qty=pos.qty)
                    pos.active = False

    async def sync(self):
        """
        定期同步远端仓位信息，仅打印当前持仓量，方便调试。
        """
        try:
            ts = int(time.time() * 1000 + time_offset)
            qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
            sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
            url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
            r = await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY}, timeout=5)
            res = await r.json()
            for p in res:
                if p['symbol'] == Config.SYMBOL:
                    LOG.debug(f"[PT.sync] Remote pos amount={p['positionAmt']}, entryPrice={p.get('entryPrice')}")
        except Exception as e:
            LOG.error(f"[PT.sync] Error syncing remote positions: {e}")

pos_tracker = PositionTracker()

# —— 订单防护 & 冷却管理 ——
class OrderGuard:
    def __init__(self):
        # states[strat] = {'ts': 上一次下单时间, 'fp': 指纹, 'trend': 'LONG'/'SHORT'}
        self.states   = defaultdict(lambda: {'ts': 0, 'fp': None, 'trend': None})
        self.lock     = asyncio.Lock()
        self.cooldown = dict.fromkeys(('main', 'macd', 'triple', 'ptm'), Config.ROTATION_COOLDOWN)

    async def check(self, strat: str, fp: str, trend: str) -> bool:
        """
        判断策略 strat、方向 trend、指纹 fp 的组合是否在冷却期外。
        如果在冷却期内或指纹重复，返回 False；否则 True。
        """
        async with self.lock:
            st = self.states[strat]
            now = time.time()
            # 如果“上一次 fp 与这次 fp 相同”，或者“同一趋势但距离上次下单时间 < 冷却”，都不下单
            if st['fp'] == fp:
                return False
            if st['trend'] == trend and (now - st['ts'] < self.cooldown[strat]):
                return False
            return True

    async def update(self, strat: str, fp: str, trend: str):
        """
        在实际发单后，更新该策略的状态：记录新的 fp、趋势、时间戳。
        """
        async with self.lock:
            self.states[strat] = {'fp': fp, 'trend': trend, 'ts': time.time()}

guard = OrderGuard()

# —— 确保 aiohttp.ClientSession 存在 ——
async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession()

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.tfs     = ("3m", "15m", "1h")
        # 初始化三个 K 线 DataFrame：包含 open, high, low, close，以及后面计算的技术指标列
        self.klines  = {
            tf: pd.DataFrame(columns=["open", "high", "low", "close"]) for tf in self.tfs
        }
        self.last_ts = dict.fromkeys(self.tfs, 0)
        self.lock    = asyncio.Lock()
        self._evt    = asyncio.Event()
        self.price   = None
        self.ptime   = 0

    async def load_history(self):
        """
        启动时拉取各周期最近 1000 根历史 K 线，并计算初始指标。
        """
        async with self.lock:
            for tf in self.tfs:
                params = {"symbol": Config.SYMBOL, "interval": tf, "limit": 1000}
                hdrs   = {"X-MBX-APIKEY": Config.API_KEY}
                r      = await session.get(f"{Config.REST_BASE}/fapi/v1/klines",
                                           params=params, headers=hdrs, timeout=5)
                data   = await r.json()
                df = pd.DataFrame([{
                    "open": float(x[1]), "high": float(x[2]),
                    "low": float(x[3]), "close": float(x[4])
                } for x in data])
                self.klines[tf]   = df
                self.last_ts[tf] = int(data[-1][0])
                self._compute(tf)

    async def update_kline(self, tf: str, o: float, h: float, l: float, c: float, ts: int):
        """
        收到 WebSocket 推送的最新 K 线数据时，维护 DataFrame：
        如果 ts > last_ts[tf] → 新开一个 bar；否则更新最后一根 bar 的 o/h/l/c；
        然后重新计算该周期的指标；最后 set 事件，让 engine 可以取到新的 price 和指标做策略判断。
        """
        async with self.lock:
            df  = self.klines[tf]
            # 新 K 线：append；否则就更新最后一个 bar
            if ts > self.last_ts[tf]:
                df.loc[len(df), ["open", "high", "low", "close"]] = [o, h, l, c]
            else:
                idx = df.index[-1]
                df.loc[idx, ["open", "high", "low", "close"]] = [o, h, l, c]
            self.last_ts[tf] = ts
            self._compute(tf)
            self._evt.set()

    async def track_price(self, p: float, ts: int):
        """
        收到标记价格更新时，记录最新 price + ptime，触发策略和本地 SL/TP 检测。
        """
        async with self.lock:
            self.price  = p
            self.ptime  = ts
        self._evt.set()
        # 检查本地 SL/TP
        await pos_tracker.check_trigger(p)

    async def wait_update(self):
        """
        engine() 中循环：等到 _evt.set() 再继续下发策略检查。
        """
        await self._evt.wait()
        self._evt.clear()

    def _compute(self, tf: str):
        """
        计算技术指标：BB（布林带百分比）、ADX + DMP/DMN，15m 周期额外计算 SuperTrend、MACD、MA7/25/99。
        """
        df = self.klines[tf]
        if len(df) < 20:
            return

        # 布林带：20 周期
        m = df.close.rolling(20).mean()
        s = df.close.rolling(20).std()
        df["bb_up"]  = m + 2 * s
        df["bb_dn"]  = m - 2 * s
        df["bb_pct"] = (df.close - df.bb_dn) / (df.bb_up - df.bb_dn)

        # ADX 指标
        adx = ADXIndicator(df.high, df.low, df.close, window=14)
        df["adx"]     = adx.adx()
        df["dmp"]     = adx.adx_pos()
        df["dmn"]     = adx.adx_neg()

        if tf == "15m":
            # SuperTrend 简单实现
            df["st"]   = (df.high + df.low) / 2 - 3 * (
                df.high.rolling(10).max() - df.low.rolling(10).min()
            )
            df["macd"] = MACD(df.close, 12, 26, 9).macd_diff()
            df["ma7"]  = df.close.rolling(7).mean()
            df["ma25"] = df.close.rolling(25).mean()
            df["ma99"] = df.close.rolling(99).mean()

data_mgr = DataManager()

@jit(nopython=True)
def numba_supertrend(h: list, l: list, c: list, per: int, mult: float):
    """
    用 Numba 加速计算 SuperTrend：返回 st 与 bool 数组 dirc（True 表示上涨趋势）。
    简化实现，仅供参考。
    """
    n = len(c)
    st = [0.0] * n
    dirc = [False] * n
    hl2 = [(h[i] + l[i]) / 2 for i in range(n)]
    # 简单 ATR = 高低差的 rolling max-min
    atr = [
        max(h[max(0, i - per + 1):i + 1]) - min(l[max(0, i - per + 1):i + 1])
        for i in range(n)
    ]
    up = [hl2[i] + mult * atr[i] for i in range(n)]
    dn = [hl2[i] - mult * atr[i] for i in range(n)]
    st[0], dirc[0] = up[0], True
    for i in range(1, n):
        if c[i] > st[i - 1]:
            st[i] = max(dn[i], st[i - 1])
            dirc[i] = True
        else:
            st[i] = min(up[i], st[i - 1])
            dirc[i] = False
    return st, dirc

# —— 策略实现 —— #
class MainStrategy:
    def __init__(self):
        self._last = 0
        self.interval = 1  # 最少 1 秒才查一次

    async def check(self, price: float):
        """
        主策略：
        1. 15m 周期 ADX > 25，且 price 与 MA7/MA25/MA99 顺序符合多头/空头；
        2. 3m 周期布林带突破（bb_pct <= 0 或 bb_pct >= 1）；
        3. 再看 1h 周期布林带是否很“强”（bb_pct < 0.2 或 > 0.8），
           强的话用更密集的小仓位；否则用一般仓位；
        4. 计算要挂几档限价单，每档都附带 sl, tp；
        5. 为避免“挂单后价格跳过挂单直接爆仓”，额外在当前市价附近挂一个 STOP_MARKET 止损挂单（无 tp）。
        """
        now = time.time()
        if now - self._last < self.interval:
            return

        df15 = data_mgr.klines["15m"]
        if len(df15) < 99 or df15['adx'].iat[-1] <= 25:
            return

        ma7  = df15.ma7.iat[-1]
        ma25 = df15.ma25.iat[-1]
        ma99 = df15.ma99.iat[-1]
        # 不满足 MA7 < MA25 < MA99 或 MA7 > MA25 > MA99 的顺序，就跳过
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99):
            return

        # 计算 SuperTrend 判断大趋势
        h = df15.high.values
        l = df15.low.values
        c = df15.close.values
        st, sd = numba_supertrend(h, l, c, 10, 3)
        trend_up = (price > st[-1] and sd[-1])
        trend_dn = (price < st[-1] and not sd[-1])

        bb3 = data_mgr.klines["3m"].bb_pct.iat[-1]
        # 布林带突破 3m
        if not (bb3 <= 0 or bb3 >= 1):
            return

        side = "BUY" if bb3 <= 0 else "SELL"
        bb1 = data_mgr.klines["1h"].bb_pct.iat[-1]
        strong = (bb1 < 0.2 or bb1 > 0.8)

        # 如果趋势与 bb3 的方向匹配，则开 0.12；否则 0.07
        qty = 0.12 if ((side == "BUY" and trend_up) or (side == "SELL" and trend_dn)) else 0.07

        if strong:
            levels = [0.0025, 0.014, 0.026, 0.038, 0.056]
            sizes  = [qty * 0.2] * 5
        else:
            if side == "BUY":
                levels = [-0.0155, -0.0255] if trend_up else [-0.0075]
            else:
                levels = [0.0155, 0.0255] if trend_dn else [0.0075]
            sizes = [0.015] * len(levels)

        self._last = now

        # 挂多个限价单（附带 sl, tp）
        for lvl, sz in zip(levels, sizes):
            p0 = price * (1 + (lvl if side == "BUY" else -lvl))
            sl = price * (0.98 if side == "BUY" else 1.02)
            tp = price * (1.02 if side == "BUY" else 0.98)
            # 直接传递已量化的 price、sl、tp；在 OrderManager 中会再一次 quantize
            await mgr.safe_place("main", side, "LIMIT",
                                 qty=sz, price=p0,
                                 extra_params={'sl': sl, 'tp': tp})

        # —— 防护挂单止损：在当前市价附近挂一个 STOP_MARKET —— #
        sl_price = price * (0.98 if side == "BUY" else 1.02)
        await mgr.safe_place("main", side, "STOP_MARKET",
                             stop=sl_price, extra_params={'closePosition': 'true'})

class MACDStrategy:
    def __init__(self):
        self._in = False  # 是否已经开过仓

    async def check(self, price: float):
        """
        MACD 策略：
        1. 15m 周期 ADX > 25，且 price 与 MA7/MA25/MA99 顺序匹配；
        2. 检查前两根 MACD diff 是否穿越零轴：prev > 0 > curr → 做空限价单；prev < 0 < curr → 做多限价单；
        3. 每次只开 0.016 ETH 的小单，附带固定 sl=3%, tp=3%。
        """
        df = data_mgr.klines["15m"]
        if len(df) < 30 or df['adx'].iat[-1] <= 25:
            return

        prev = df.macd.iat[-2]
        curr = df.macd.iat[-1]
        ma7  = df.ma7.iat[-1]
        ma25 = df.ma25.iat[-1]
        ma99 = df.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99):
            return

        # prev > 0 > curr：死叉 → 做空限价；prev < 0 < curr：金叉 → 做多限价
        if prev > 0 > curr and not self._in:
            sp = price * 1.005
            await mgr.safe_place("macd", "SELL", "LIMIT",
                                 qty=0.016, price=sp,
                                 extra_params={'sl': sp * 1.03, 'tp': sp * 0.97})
            self._in = True

        elif prev < 0 < curr and self._in:
            bp = price * 0.995
            await mgr.safe_place("macd", "BUY", "LIMIT",
                                 qty=0.016, price=bp,
                                 extra_params={'sl': bp * 0.97, 'tp': bp * 1.03})
            self._in = False

class TripleTrendStrategy:
    def __init__(self):
        self._last = 0
        self.round_active = False

    async def check(self, price: float):
        """
        三重 SuperTrend 策略：
        1. 1h 周期 ADX > 25；15m 周期有足够的数据；
        2. 15m 周期上同时计算三种参数的 SuperTrend（(10,1)、(11,2)、(12,3)）；
        3. 三者都向上 → “波段做多”；三者都向下 → “波段做空”；
        4. 进入“波段仓”后，当任意一个指标产生拐点 → 反向市价平仓；
        5. 新一轮三者齐向上/齐向下时，重新限价开仓，并附带 sl、tp。
        """
        now = time.time()
        if now - self._last < 1:
            return

        df1h = data_mgr.klines["1h"]
        if len(df1h) < 15 or df1h['adx'].iat[-1] <= 25:
            return

        df15 = data_mgr.klines["15m"]
        if len(df15) < 99:
            return

        ma7  = df15.ma7.iat[-1]
        ma25 = df15.ma25.iat[-1]
        ma99 = df15.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99):
            return

        h = df15.high.values
        l = df15.low.values
        c = df15.close.values
        _, d1 = numba_supertrend(h, l, c, 10, 1)
        _, d2 = numba_supertrend(h, l, c, 11, 2)
        _, d3 = numba_supertrend(h, l, c, 12, 3)

        up_all = d1[-1] and d2[-1] and d3[-1]
        dn_all = (not d1[-1]) and (not d2[-1]) and (not d3[-1])
        prev   = (d1[-2], d2[-2], d3[-2])
        curr   = (d1[-1], d2[-1], d3[-1])

        flip_dn = self.round_active and any(p and (not c2) for p, c2 in zip(prev, curr))
        flip_up = self.round_active and any((not p) and c2 for p, c2 in zip(prev, curr))

        self._last = now

        # 三线都向上 & 当前没有持仓 → 做多限价
        if up_all and not self.round_active:
            self.round_active = True
            p0 = price * 0.996
            sl = price * 0.97
            tp = price * 1.02
            await mgr.safe_place("triple", "BUY", "LIMIT",
                                 qty=0.015, price=p0,
                                 extra_params={'sl': sl, 'tp': tp})

        # 三线都向下 & 当前没有持仓 → 做空限价
        elif dn_all and not self.round_active:
            self.round_active = True
            p0 = price * 1.004
            sl = price * 1.03
            tp = price * 0.98
            await mgr.safe_place("triple", "SELL", "LIMIT",
                                 qty=0.015, price=p0,
                                 extra_params={'sl': sl, 'tp': tp})

        # 如果三线中的任意一条变向且 round_active=True，则市价平仓
        elif flip_dn:
            await mgr.safe_place("triple", "SELL", "MARKET")
            self.round_active = False

        elif flip_up:
            await mgr.safe_place("triple", "BUY", "MARKET")
            self.round_active = False

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket & 主循环 —— #
async def market_ws():
    """
    连接市场行情 WebSocket：订阅 3m/15m/1h K 线与 markPrice 流。
    收到 markPrice 时，调用 DataManager.track_price() → 触发本地 SL/TP 检测与策略；
    收到 K 线更新时，更新 DataManager 中对应周期的 K 线数据。
    """
    retry = 0
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET, ping_interval=None) as ws:
                retry = 0
                async for msg in ws:
                    o = json.loads(msg)
                    stream = o["stream"]
                    data   = o["data"]
                    if stream.endswith("@markPrice"):
                        # 标记价格更新
                        price = float(data["p"])
                        ts    = int(time.time() * 1000)
                        await data_mgr.track_price(price, ts)
                    else:
                        # K 线更新
                        tf = stream.split("@")[1].split("_")[1]   # 例如 "3m"/"15m"/"1h"
                        k  = data["k"]
                        await data_mgr.update_kline(
                            tf,
                            float(k["o"]), float(k["h"]),
                            float(k["l"]), float(k["c"]),
                            int(k["t"])
                        )
        except Exception as e:
            delay = min(2 ** retry, 30)
            LOG.error(f"[WS_MKT] {e}, retry in {delay}s")
            await asyncio.sleep(delay)
            retry += 1

async def get_listen_key():
    """
    创建新的用户数据流 listenKey：POST /fapi/v1/listenKey
    """
    await ensure_session()
    r = await session.post(
        f"{Config.REST_BASE}/fapi/v1/listenKey",
        headers={'X-MBX-APIKEY': Config.API_KEY}, timeout=5
    )
    data = await r.json()
    return data['listenKey']

async def user_ws():
    """
    用户数据流 WebSocket：收到 ORDER_TRADE_UPDATE 事件，就调用 pos_tracker.on_order_update()。
    """
    global listen_key
    retry = 0
    while True:
        try:
            listen_key = await get_listen_key()
            url = f"{Config.WS_USER_BASE}{listen_key}"
            async with websockets.connect(url, ping_interval=None) as ws:
                LOG.debug(f"[WS_USER] User stream connected: {listen_key}")
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get('e') == 'ORDER_TRADE_UPDATE':
                        o = data['o']
                        oid    = o['i']      # 交易所上的 orderId
                        status = o['X']      # ORDER STATUS (NEW, FILLED, CANCELED, etc)
                        await pos_tracker.on_order_update(oid, status)
        except Exception as e:
            LOG.error(f"[WS_USER] {e}, reconnect in 5s")
            await asyncio.sleep(5)
            retry += 1

async def keepalive_listen_key():
    """
    每隔 30 分钟，调用一次 get_listen_key()，保持 listenKey 有效。
    """
    global listen_key
    while True:
        await asyncio.sleep(1800)
        try:
            listen_key = await get_listen_key()
            LOG.debug("[keepalive_listen_key] ListenKey renewed.")
        except Exception as e:
            LOG.error(f"[keepalive_listen_key] Failed to renew listenKey: {e}")

async def maintenance():
    """
    每隔 SYNC_INTERVAL，执行：
      1. sync_time()
      2. detect_mode()
      3. pos_tracker.sync()
      4. 本地 SL/TP 再次 check_trigger（以防漏掉某些时机）
    """
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        try:
            await sync_time()
            await detect_mode()
            await pos_tracker.sync()
            if data_mgr.price is not None:
                await pos_tracker.check_trigger(data_mgr.price)
        except Exception as e:
            LOG.exception(f"[Maintenance] Exception: {e}")

async def engine():
    """
    核心引擎：等待任意市场数据更新（price 或 K 线），获得最新 data_mgr.price 后，
    依次调用每个策略的 check() 。如果 price 超过 60 秒未更新，则跳过。
    """
    while True:
        await data_mgr.wait_update()
        if data_mgr.price is None or (time.time() - data_mgr.ptime > 60):
            continue
        for strat in strategies:
            try:
                await strat.check(data_mgr.price)
            except Exception as e:
                LOG.exception(f"[Engine] Strategy {strat.__class__.__name__} error: {e}")

async def main():
    """
    程序主入口：
      1. 创建 aiohttp session
      2. 注册 SIGINT/SIGTERM 的清理 handler
      3. 同步一次时间、模式、加载精度、加载历史数据
      4. 并行启动：market_ws, user_ws, maintenance, engine, keepalive_listen_key
    """
    global session
    session = aiohttp.ClientSession()
    loop = asyncio.get_event_loop()

    # 退出时，先关闭 aiohttp Session
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(session.close()))

    # 1. 同步时间 & 检测模式
    await sync_time()
    await detect_mode()

    # 2. 加载交易对精度
    await load_symbol_filters()

    # 3. 加载历史 K 线，并计算指标
    await data_mgr.load_history()

    # 4. 同步一下远端仓位（主要为了日志查看）
    await pos_tracker.sync()

    # 5. 并行执行各协程
    await asyncio.gather(
        market_ws(),
        user_ws(),
        maintenance(),
        engine(),
        keepalive_listen_key()
    )

if __name__ == '__main__':
    asyncio.run(main())