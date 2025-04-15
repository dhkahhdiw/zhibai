#!/usr/bin/env python3
"""
ETH/USDC 高频交易引擎 v7.4（REST实时指标版）— 更新后的策略逻辑

【数据与预处理】
  - 通过REST API实时获取15m、1h、3m K线数据；
  - 使用 ffilling (df.ffill()) 处理缺失数据；
  - 各周期数据将用于计算不同指标，确保下单时使用最新数据。

【趋势判断与信号确认】
  1. 主趋势判断：采用15m超级趋势指标（20周期MA+3×ATR），
     - 当最新价高于上轨（绿色）则判定为上升趋势 (LONG)；
     - 当最新价低于下轨（红色）则判定为下降趋势 (SHORT)；
     - 介于两者之间时辅以15m MACD（12,26,9）的判断。
  2. 强弱信号：基于1h布林带%B (20,2)判断，
     - 对于多单：若%B < 0.2为强信号，否则为弱信号；
     - 对于空单：若%B > 0.8为强信号，否则为弱信号。
  3. 触发条件：采用3m布林带%B (20,2)计算，
     - 公式：%B = (最新价 - 下轨) / (上轨 - 下轨)；
     - 当 %B ≤ 0 触发 BUY 信号，当 %B ≥ 1 触发 SELL 信号。
  4. 信号融合：要求15m与1h指标支持同一方向，3m信号作为具体触发依据。

【订单生成与挂单】
  - 入场挂单：根据信号与趋势一致性确定单笔仓位；
      • 同趋势下：强信号下单仓位 0.12 ETH，弱信号 0.03 ETH；
      • 趋势反向下：强信号 0.07 ETH，弱信号 0.015 ETH；
  - 挂单价格以触发时最新价为基准，采用预设偏移档位（例如：±0.25%、0.40%、±0.60%、±0.80%、±1.60%，各20%）；
  - 止盈挂单：以入场价为基准提前挂单（强信号偏移依次为 ±1.02%、±1.23%、±1.50%、±1.80%、±2.20%，各20%；弱信号采用另一组）；
  - 止损：初始按买入价×0.98（多单）／卖出价×1.02（空单）设定，并结合3m布林带带宽动态调整。

【信号轮换】
  - 同一轮内仅允许触发一次订单。若上一次触发为 BUY 信号，则本轮内若再次收到 BUY 信号则忽略，
    只有当收到 SELL 信号时才触发下单；反之亦然。
  - 通过变量 last_triggered_side 保存本轮已触发订单方向。

【并行子策略】
  - 15m MACD 策略：基于15m数据计算 MACD（EMA12, EMA26, DEA=EMA9），离轴值 = 2×(DIF - DEA)，
      根据设定条件触发平仓（例如：空单：离轴值在 [11,20) 下平 0.1 ETH，≥20 下平 0.15 ETH；多单类似）。
  - 超级趋势策略：采用三个不同参数的15m超级趋势指标（长度10、11、12，因子均为3），
      当三者一致上升则市价多单 0.15 ETH，否则当三者一致下降则市价空单 0.15 ETH，同时触发平仓操作。

【仓位控制与资金管理】
  - 每次下单或平仓后通过 update_local_position() 更新本地仓位（current_long、current_short）；
  - 根据当前趋势设定仓位上限（例如：趋势上升时，多仓允许100%，空仓不超过75%；趋势下降时，多仓不超过75%，空仓允许100%）；
  - 超出上限时暂停对应方向下单（本示例仅输出仓位信息，可进一步扩展）。

所有API调用均采用币安合约 REST API (fapi/v1)，USDC合约时附加参数 marginCoin=USDC。

"""

import uvloop
uvloop.install()

import os, asyncio, time, hmac, hashlib, urllib.parse, logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from aiohttp import ClientTimeout, TCPConnector, ClientConnectorError, ServerDisconnectedError
from aiohttp_retry import RetryClient, ExponentialRetry
from aiohttp.resolver import AsyncResolver

from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server

# ------------------- 环境变量初始化 -------------------
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY    = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL     = 'ETHUSDC'
LEVERAGE   = 50

# 仓位上限（本地仓位记录及下单控制参考）
BULL_LONG_LIMIT  = 0.49
BULL_SHORT_LIMIT = 0.35
BEAR_LONG_LIMIT  = 0.35
BEAR_SHORT_LIMIT = 0.49

# 订单基础规模（单位：ETH）
STRONG_SIZE_SAME  = 0.12
STRONG_SIZE_DIFF  = 0.07
WEAK_SIZE_SAME    = 0.03
WEAK_SIZE_DIFF    = 0.015

# 入场挂单方案（基于触发时价格偏移）
def get_entry_order_list(strong: bool) -> List[Dict[str, Any]]:
    if strong:
        return [
            {'offset': 0.0025, 'ratio': 0.20},
            {'offset': 0.0040, 'ratio': 0.20},
            {'offset': 0.0060, 'ratio': 0.20},
            {'offset': 0.0080, 'ratio': 0.20},
            {'offset': 0.0160, 'ratio': 0.20},
        ]
    else:
        return [
            {'offset': 0.0025, 'ratio': 0.50},
            {'offset': 0.0160, 'ratio': 0.50},
        ]

# 止盈挂单方案
def get_take_profit_orders_strong() -> List[Dict[str, Any]]:
    return [
        {'offset': 0.0102, 'ratio': 0.20},
        {'offset': 0.0123, 'ratio': 0.20},
        {'offset': 0.0150, 'ratio': 0.20},
        {'offset': 0.0180, 'ratio': 0.20},
        {'offset': 0.0220, 'ratio': 0.20},
    ]

def get_take_profit_orders_weak() -> List[Dict[str, Any]]:
    return [
        {'offset': 0.0123, 'ratio': 0.50},
        {'offset': 0.0180, 'ratio': 0.50},
    ]

# 交易与限频配置
RECV_WINDOW = 10000
MIN_NOTIONAL = 20.0
# 采用严格轮换逻辑，不使用固定冷却期
RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

start_http_server(8001)
METRICS = {'throughput': Gauge('api_throughput', '请求数/秒')}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/eth_usdc_hft.log", encoding="utf-8", mode='a')
    ]
)
logger = logging.getLogger('ETH-USDC-REST')

@dataclass
class TradingConfig:
    st_period: int = 20
    st_multiplier: float = 3.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    max_retries: int = 7
    order_timeout: float = 2.0
    network_timeout: float = 5.0
    price_precision: int = 2
    quantity_precision: int = 3
    order_adjust_interval: float = 1.0
    dual_side_position: bool = False

# ------------------- REST API 客户端 -------------------
class BinanceRestClient:
    def __init__(self) -> None:
        self.config = TradingConfig()
        self.connector = TCPConnector(limit=512,
                                      resolver=AsyncResolver(),
                                      ttl_dns_cache=300,
                                      force_close=True,
                                      ssl=True)
        self._init_session()
        self.recv_window = RECV_WINDOW
        self.request_timestamps = defaultdict(lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 200))
        self._time_diff = 0

    def _init_session(self) -> None:
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.5,
            max_timeout=3.0,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={TimeoutError, ClientConnectorError, ServerDisconnectedError}
        )
        self.session = RetryClient(connector=self.connector,
                                    retry_options=retry_opts,
                                    timeout=ClientTimeout(total=self.config.network_timeout,
                                                          sock_connect=self.config.order_timeout))

    async def sync_server_time(self) -> None:
        url = "https://fapi.binance.com/fapi/v1/time"
        for retry in range(5):
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                    data = await resp.json()
                    server_time = data.get('serverTime')
                    if not server_time:
                        raise ValueError("缺少 serverTime")
                    local_time = int(time.time() * 1000)
                    self._time_diff = server_time - local_time
                    logger.info(f"时间同步成功，差值：{self._time_diff}ms")
                    return
            except Exception as e:
                logger.error(f"时间同步失败(重试 {retry+1}): {e}")
                await asyncio.sleep(2 ** retry)
        logger.warning("时间同步失败，采用本地时间")
        self._time_diff = 0

    async def _signed_request(self, method: str, path: str, params: dict) -> dict:
        params.update({
            "timestamp": int(time.time() * 1000 + self._time_diff),
            "recvWindow": self.recv_window
        })
        if params.get("symbol", "") == SYMBOL:
            params["marginCoin"] = "USDC"
        sorted_params = sorted(params.items())
        query = urllib.parse.urlencode(sorted_params, doseq=True)
        signature = hmac.new(SECRET_KEY.encode('utf-8'),
                             query.encode('utf-8'),
                             hashlib.sha256).hexdigest()
        url = f"https://fapi.binance.com/fapi/v1{path}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": API_KEY,
                   "Accept": "application/json",
                   "Content-Type": "application/x-www-form-urlencoded"}
        logger.debug(f"请求: {url.split('?')[0]} 参数: {sorted_params}")
        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)
                async with self.session.request(method, url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status} 返回: {await resp.text()}")
                    if "application/json" not in resp.headers.get("Content-Type", ""):
                        raise Exception(f"响应异常: {await resp.text()}")
                    data = await resp.json()
                    if isinstance(data, dict) and data.get("code", 0) < 0:
                        raise Exception(f"接口错误，Code: {data.get('code')}, Msg: {data.get('msg')}")
                    return data
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt+1}): {e}")
                if attempt >= self.config.max_retries:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise Exception("超过最大重试次数")

    async def _rate_limit_check(self, endpoint: str) -> None:
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]
        now = time.monotonic()
        while dq and dq[0] < now - period:
            dq.popleft()
        if len(dq) >= limit:
            wait_time = max(dq[0] + period - now + np.random.uniform(0, 0.05), 0)
            logger.warning(f"接口 {endpoint} 限频，等待 {wait_time:.3f}s")
            METRICS['throughput'].set(0)
            await asyncio.sleep(wait_time)
        dq.append(now)
        METRICS['throughput'].inc()

    async def manage_leverage(self) -> dict:
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE}
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self, interval: str, limit: int = 100) -> pd.DataFrame:
        params = {'symbol': SYMBOL, 'interval': interval, 'limit': limit}
        data = await self._signed_request('GET', '/klines', params)
        if not isinstance(data, list):
            logger.error("K线格式异常")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # 使用 ffilling 替代 fillna(method='ffill')
        df = df.ffill()
        return df

# ------------------- 策略模块 -------------------
@dataclass
class Signal:
    action: bool
    side: str   # 'BUY' 或 'SELL'
    order_details: dict = None

class ETHUSDCStrategy:
    def __init__(self, client: BinanceRestClient) -> None:
        self.client = client
        self.config = TradingConfig()
        # 当前趋势：'LONG' 或 'SHORT'
        self.last_trade_side: str = None
        # 轮换触发机制：本轮内仅允许触发一个订单，下单信号必须与上一次触发相反
        self.last_triggered_side: str = None
        self.entry_price: float = None
        # 本地仓位记录（单位：ETH）
        self.current_long: float = 0.0
        self.current_short: float = 0.0
        self.prev_macd_off: float = None

    def update_local_position(self, side: str, quantity: float, closing: bool = False) -> None:
        if closing:
            if side.upper() == 'BUY':  # 平空仓
                self.current_short = max(0, self.current_short - quantity)
            elif side.upper() == 'SELL':  # 平多仓
                self.current_long = max(0, self.current_long - quantity)
        else:
            if side.upper() == 'BUY':
                self.current_long += quantity
            elif side.upper() == 'SELL':
                self.current_short += quantity
        logger.info(f"[本地仓位更新] 多仓: {self.current_long:.4f} ETH，空仓: {self.current_short:.4f} ETH")

    def build_order_params(self, base: Dict[str, Any], pos_side: str = None) -> Dict[str, Any]:
        if self.config.dual_side_position and pos_side is not None:
            base['positionSide'] = pos_side
        return base

    # ------------------ 趋势判断：15m超级趋势+MACD辅助 ------------------
    async def analyze_trend_15m(self) -> str:
        df = await self.client.fetch_klines(interval='15m', limit=100)
        if df.empty:
            return 'LONG'
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        tr = np.maximum.reduce([
            high.diff().abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ])
        atr = pd.Series(tr).rolling(window=self.config.st_period).mean().iloc[-1]
        hl2 = (high + low) / 2
        last_hl2 = hl2.iloc[-1]
        basic_upper_val = last_hl2 + self.config.st_multiplier * atr
        basic_lower_val = last_hl2 - self.config.st_multiplier * atr
        latest = close.iloc[-1]
        if latest > basic_upper_val:
            return 'LONG'
        elif latest < basic_lower_val:
            return 'SHORT'
        else:
            ema_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            return 'LONG' if macd_line.iloc[-1] >= 0 else 'SHORT'

    # ------------------ 1h布林带%B判断信号强弱 ------------------
    async def get_hourly_strength(self, side: str) -> bool:
        df = await self.client.fetch_klines(interval='1h', limit=50)
        if df.empty:
            return False
        close = df['close'].astype(float)
        sma = close.rolling(window=self.config.bb_period).mean()
        std = close.rolling(window=self.config.bb_period).std()
        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std
        percent_b = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1]-lower.iloc[-1]) > 0 else 0.5
        logger.info(f"[1h%B] {side.upper()}信号: {percent_b:.3f}")
        if side.upper() == 'BUY':
            return percent_b < 0.2
        elif side.upper() == 'SELL':
            return percent_b > 0.8
        return False

    # ------------------ 3m布林带%B强化下单信号 ------------------
    async def analyze_order_signals_3m(self) -> Tuple[Signal, dict]:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return Signal(False, 'NONE'), {}
        close = df['close'].astype(float)
        sma = close.rolling(window=self.config.bb_period).mean()
        std = close.rolling(window=self.config.bb_period).std()
        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std
        percent_b = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1]-lower.iloc[-1])>0 else 0.5
        logger.info(f"[3m%B] 当前 %B: {percent_b:.3f}")
        if percent_b <= 0:
            return Signal(True, 'BUY', {'trigger_price': close.iloc[-1]}), {}
        elif percent_b >= 1:
            return Signal(True, 'SELL', {'trigger_price': close.iloc[-1]}), {}
        return Signal(False, 'NONE'), {}

    # ------------------ 挂单与止盈止损：入场 ------------------
    async def place_dynamic_limit_orders(self, side: str, order_list: List[Dict[str, Any]], trigger_price: float, order_size: float) -> None:
        pos_side = "LONG" if side == "BUY" else "SHORT"
        for order in order_list:
            offset = order['offset']
            ratio = order['ratio']
            qty = round(order_size * ratio, self.config.quantity_precision)
            if qty <= 0:
                continue
            limit_price = round(trigger_price * (1 - offset) if side=="BUY" else trigger_price * (1 + offset), self.config.price_precision)
            if limit_price <= 0:
                continue
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': limit_price,
                'quantity': qty,
                'timeInForce': 'GTC'
            }
            params = self.build_order_params(params, pos_side)
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"[下单] {side} @ {limit_price}，数量: {qty}，偏移: {offset*100:.2f}% 成功，返回: {data}")
                self.update_local_position(side, qty, closing=False)
            except Exception as e:
                logger.error(f"[下单] {side}挂单失败：{e}")

    async def place_take_profit_orders(self, side: str, tp_orders: List[Dict[str, Any]], entry_price: float) -> None:
        # 为止盈订单，订单方向通常与入场单相反
        pos_side = None  # 如果未使用双向持仓则不传
        for order in tp_orders:
            offset = order['offset']
            tp_price = round(entry_price * (1 + offset) if side.upper() == 'BUY' else entry_price * (1 - offset), self.config.price_precision)
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': tp_price,
                'quantity': 0,
                'timeInForce': 'GTC'
            }
            params = self.build_order_params(params, pos_side)
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"[止盈] {side} TP订单 @ {tp_price}，返回: {data}")
            except Exception as e:
                logger.error(f"[止盈] {side} TP订单挂单失败：{e}")

    def build_order_params(self, base: Dict[str, Any], pos_side: str = None) -> Dict[str, Any]:
        if self.config.dual_side_position and pos_side is not None:
            base['positionSide'] = pos_side
        return base

    # ------------------ 市价平仓 ------------------
    async def close_position(self, side: str, quantity: float, strategy: str = "normal") -> None:
        pos_side = None
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity
        }
        params = self.build_order_params(params, pos_side)
        try:
            data = await self.client._signed_request('POST', '/order', params)
            logger.info(f"[平仓] 市价平仓 {side}，数量: {quantity}，返回: {data}")
            self.update_local_position(side, quantity, closing=True)
        except Exception as e:
            logger.error(f"[平仓] 失败: {e}")

    # ------------------ 15m MACD策略平仓 ------------------
    async def madc_strategy_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                df = df.ffill()
                close = df['close'].astype(float)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9, adjust=False).mean()
                divergence = 2 * (dif - dea)
                trigger_val = divergence.iloc[-1]
                logger.info(f"[MADC] 离轴值: {trigger_val:.2f}")
                # 空单平仓条件：离轴值在 [11,20) 平0.1 ETH，≥20平0.15 ETH
                if trigger_val >= 11 and trigger_val < 20:
                    logger.info("[MADC] 空单触发弱信号，平多仓0.1ETH")
                    await self.close_position('SELL', 0.1, strategy="madc")
                elif trigger_val >= 20:
                    logger.info("[MADC] 空单触发强信号，平多仓0.15ETH")
                    await self.close_position('SELL', 0.15, strategy="madc")
                # 多单平仓条件：离轴值在 (-20, -11] 平0.1 ETH，≤ -20 平0.15 ETH
                if trigger_val <= -11 and trigger_val > -20:
                    logger.info("[MADC] 多单触发弱信号，平空仓0.1ETH")
                    await self.close_position('BUY', 0.1, strategy="madc")
                elif trigger_val <= -20:
                    logger.info("[MADC] 多单触发强信号，平空仓0.15ETH")
                    await self.close_position('BUY', 0.15, strategy="madc")
            except Exception as e:
                logger.error(f"[MADC] 异常: {e}")
            await asyncio.sleep(60 * 15)

    # ------------------ 超级趋势策略 ------------------
    async def supertrend_strategy_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                df = df.ffill()
                high = df['high'].astype(float)
                low = df['low'].astype(float)
                close = df['close'].astype(float)
                def supertrend(length: int, factor: float) -> str:
                    atr_val = pd.Series(np.maximum.reduce([
                        high.diff().abs(),
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()
                    ])).rolling(window=length).mean().iloc[-1]
                    hl2 = ((high + low) / 2).iloc[-1]
                    basic_upper = hl2 + factor * atr_val
                    basic_lower = hl2 - factor * atr_val
                    latest = close.iloc[-1]
                    if latest > basic_upper:
                        return 'UP'
                    elif latest < basic_lower:
                        return 'DOWN'
                    else:
                        return 'NEUTRAL'
                trend1 = supertrend(10, 1)
                trend2 = supertrend(11, 2)
                trend3 = supertrend(12, 3)
                logger.info(f"[超级趋势] 指标趋势: {trend1}, {trend2}, {trend3}")
                if trend1 == trend2 == trend3 == 'UP':
                    logger.info("[超级趋势] 指标均上升，触发市价多单0.15ETH")
                    self.update_local_position('BUY', 0.15, closing=False)
                elif trend1 == trend2 == trend3 == 'DOWN':
                    logger.info("[超级趋势] 指标均下降，触发市价空单0.15ETH")
                    self.update_local_position('SELL', 0.15, closing=False)
            except Exception as e:
                logger.error(f"[超级趋势] 异常: {e}")
            await asyncio.sleep(60 * 15)

    # ------------------ 信号下单逻辑（入场） ------------------
    async def order_signal_loop(self) -> None:
        while True:
            try:
                signal, _ = await self.analyze_order_signals_3m()
                if signal.action:
                    # 轮换触发：如果本轮内已有订单触发，则仅当当前信号与上次触发方向相反时才允许下单
                    if self.last_triggered_side is not None:
                        if signal.side.upper() == self.last_triggered_side:
                            logger.info("[信号下单] 本轮内同方向信号已触发，忽略本次下单")
                            await asyncio.sleep(self.config.order_adjust_interval)
                            continue
                    # 获取1h布林带强弱信号
                    strong = await self.get_hourly_strength(signal.side)
                    # 根据 last_trade_side 决定下单仓位
                    if self.last_trade_side:
                        if signal.side.upper() == self.last_trade_side.upper():
                            order_size = STRONG_SIZE_SAME if strong else WEAK_SIZE_SAME
                        else:
                            order_size = STRONG_SIZE_DIFF if strong else WEAK_SIZE_DIFF
                    else:
                        order_size = STRONG_SIZE_SAME if strong else WEAK_SIZE_SAME
                    orders = get_entry_order_list(strong)
                    self.entry_price = signal.order_details.get("trigger_price")
                    await self.place_dynamic_limit_orders(signal.side, orders, self.entry_price, order_size)
                    # 更新轮换状态：本轮内记录已触发信号方向
                    self.last_triggered_side = signal.side.upper()
                    # 更新当前趋势，下单方向作为本轮最后触发方向
                    self.last_trade_side = 'LONG' if signal.side.upper()=='BUY' else 'SHORT'
                    logger.info(f"[信号下单] 触发订单: {signal.side.upper()}，本轮结束")
                await asyncio.sleep(self.config.order_adjust_interval)
            except Exception as e:
                logger.error(f"[信号下单] 异常: {e}")
                await asyncio.sleep(self.config.order_adjust_interval)
            await asyncio.sleep(self.config.order_adjust_interval)

    # ------------------ 止盈止损管理（动态止损） ------------------
    async def stop_loss_profit_management_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='3m', limit=50)
                if df.empty or not self.entry_price:
                    await asyncio.sleep(0.5)
                    continue
                df = df.ffill()
                latest = float(df['close'].iloc[-1])
                sma = df['close'].astype(float).rolling(window=self.config.bb_period).mean().iloc[-1]
                std = df['close'].astype(float).rolling(window=self.config.bb_period).std().iloc[-1]
                band_width = (sma + self.config.bb_std * std) - (sma - self.config.bb_std * std)
                dynamic_stop = (latest - band_width * 0.5) if self.last_trade_side=='LONG' else (latest + band_width * 0.5)
                logger.info(f"[止盈止损] 当前价={latest:.2f}, 动态止损={dynamic_stop:.2f}")
                if self.last_trade_side == 'LONG' and latest < dynamic_stop:
                    logger.info("[止损] 多单止损触发")
                    await self.close_position('SELL', self.current_long, strategy="normal")
                elif self.last_trade_side == 'SHORT' and latest > dynamic_stop:
                    logger.info("[止损] 空单止损触发")
                    await self.close_position('BUY', self.current_short, strategy="normal")
            except Exception as e:
                logger.error(f"[止盈止损] 异常: {e}")
            await asyncio.sleep(0.5)

    async def execute(self) -> None:
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        await asyncio.gather(
            self.analyze_trend_15m(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.madc_strategy_loop(),
            self.supertrend_strategy_loop()
        )

# ------------------- 主函数 -------------------
async def main() -> None:
    client = BinanceRestClient()
    strategy = ETHUSDCStrategy(client)
    try:
        strategy.last_trade_side = await strategy.analyze_trend_15m()
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")

if __name__ == "__main__":
    asyncio.run(main())