#!/usr/bin/env python3
"""
ETH/USDT 高频交易引擎 v7.4
  基于15m超趋势与3m级BB%B信号策略，并平行新增15m级MACD离轴交叉策略（仓位独立计算）

【原策略说明】
1. 15分钟超趋势策略、3分钟订单信号、止盈止损、仓位统计与控制、趋势转换风控（滑点控制与阶梯报价）
   ——仓位比例动态采用 1:0.5（震荡时1:0.3），并设定仓位上限（如趋势为多：多仓上限0.64，空仓上限0.42）
2. 【新增MACD策略】（独立计算，不参与原策略仓位控制）
   - 使用15分钟K线数据计算EMA12与EMA26，离轴值 = 2*(EMA12–EMA26)
   - 当检测到交叉时记录触发点：
       • 上穿（由负转正）：若离轴值在 [11,20) 下单市价 SELL 订单 0.07 ETH；离轴值 ≥20 下单 0.14 ETH
       • 下穿（由正转负）：若离轴值在 (-20, -11] 下单市价 BUY 订单 0.07 ETH；离轴值 ≤ -20 下单 0.14 ETH
   - 固定止损：多单止损 = 成交价×0.96，空单止损 = 成交价×1.04

【下单信号轮流触发机制】
   - 当原策略下单信号触发后，只允许触发单笔订单；若上次下单方向为 BUY，则连续相同方向的信号将被忽略，
     必须等待方向改变后才能触发下单。

【优化目标】
   - 增强各信号响应频率和速度，特别是止盈和买入信号的响应；
   - 下单信号检测间隔修改为1秒，以便更及时捕捉信号。

适配环境：Vultr high frequency (vhf-1c-1gb ubuntu22.04)
要求：
  - 使用 REST API（https://fapi.binance.com）
  - 异步 aiohttp 与 uvloop 实现低延迟
  - 内置限频、重试、时间同步与仓位统计
  - 原策略与MACD策略各自独立运行，不互相干扰

请确保 API Key 与 Secret 配置正确。
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

# ==================== 环境配置 ====================
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)

API_KEY    = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL     = 'ETHUSDT'
LEVERAGE   = 50

# 原策略仓位目标
BULL_LONG_QTY  = 0.56
BULL_SHORT_QTY = 0.35
BEAR_LONG_QTY  = 0.35
BEAR_SHORT_QTY = 0.56

BULL_LONG_QTY_SHAKE  = 0.49
BULL_SHORT_QTY_SHAKE = 0.35
BEAR_LONG_QTY_SHAKE  = 0.35
BEAR_SHORT_QTY_SHAKE = 0.49

MAX_POS_LONG_TREND_LONG = 0.64
MAX_POS_SHORT_TREND_LONG = 0.42
MAX_POS_LONG_TREND_SHORT = 0.42
MAX_POS_SHORT_TREND_SHORT = 0.64

# 单笔基础订单数量
QUANTITY = 0.07

# 新增MACD策略订单大小（固定值）
MACD_SMALL_ORDER = 0.07
MACD_BIG_ORDER   = 0.14

REST_URL = 'https://fapi.binance.com'

# ==================== 高频参数 ====================
RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}
RECV_WINDOW = 10000
MIN_NOTIONAL = 20.0

# ==================== Prometheus监控 ====================
start_http_server(8001)
METRICS = {
    'memory': Gauge('hft_memory', '内存(MB)'),
    'latency': Gauge('order_latency', '延迟(ms)'),
    'throughput': Gauge('api_throughput', '请求数/秒'),
    'position': Gauge('eth_position', '持仓量'),
    'errors': Gauge('api_errors', 'API错误')
}

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/eth_usdt_hft.log", encoding="utf-8", mode='a')
    ]
)
logger = logging.getLogger('ETH-USDT-HFT')

@dataclass
class TradingConfig:
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    vol_multiplier: float = 2.5
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
    max_position: float = 10.0
    slippage: float = 0.001
    dual_side_position: str = "true"
    # 下单信号检测间隔设为1秒以增强响应
    order_adjust_interval: float = 1.0
    max_slippage_market: float = 0.0015
    daily_drawdown_limit: float = 0.20

class BinanceHFTClient:
    def __init__(self) -> None:
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=512,
            resolver=AsyncResolver(),
            ttl_dns_cache=300,
            force_close=True,
            ssl=True
        )
        self._init_session()
        self.recv_window = RECV_WINDOW
        self.request_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 200))
        self._time_diff = 0

    def _init_session(self) -> None:
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.5,
            max_timeout=3.0,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={TimeoutError, ClientConnectorError, ServerDisconnectedError}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=self.config.network_timeout, sock_connect=self.config.order_timeout)
        )

    async def sync_server_time(self) -> None:
        url = f"{REST_URL}/fapi/v1/time"
        for retry in range(5):
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                    data = await resp.json()
                    if 'serverTime' not in data:
                        raise ValueError("缺少 serverTime")
                    server_time = data['serverTime']
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
        sorted_params = sorted(params.items())
        query = urllib.parse.urlencode(sorted_params, doseq=True)
        signature = hmac.new(SECRET_KEY.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
        url = f"{REST_URL}/fapi/v1{path}?{query}&signature={signature}"
        headers = {
            "X-MBX-APIKEY": API_KEY,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        logger.debug(f"请求: {url.split('?')[0]} 参数: {sorted_params}")
        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)
                async with self.session.request(method, url, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"HTTP {resp.status} 返回: {error_text}")
                    if "application/json" not in resp.headers.get("Content-Type", ""):
                        text = await resp.text()
                        raise Exception(f"响应异常: {text}")
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
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE, 'dualSidePosition': self.config.dual_side_position}
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self, interval: str, limit: int = 100) -> pd.DataFrame:
        params = {'symbol': SYMBOL, 'interval': interval, 'limit': limit}
        data = await self._signed_request('GET', '/klines', params)
        if not isinstance(data, list):
            logger.error("K线格式异常")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'quote_asset_volume', 'trades',
                                          'taker_buy_base', 'taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    async def fetch_weekly_data(self, limit: int = 100) -> pd.DataFrame:
        params = {'symbol': SYMBOL, 'interval': '1w', 'limit': limit}
        data = await self._signed_request('GET', '/klines', params)
        if not isinstance(data, list):
            logger.error("周K线格式异常")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'quote_asset_volume', 'trades',
                                          'taker_buy_base', 'taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

@dataclass
class Signal:
    action: bool
    side: str
    tp: float
    sl: float
    order_details: dict = None

class ETHUSDTStrategy:
    def __init__(self, client: BinanceHFTClient) -> None:
        self.client = client
        self.config = TradingConfig()
        self._indicator_cache = defaultdict(lambda: None)
        self.last_trade_side: str = None      # 原策略主趋势
        self.last_triggered_side: str = None    # 用于轮流触发，下单后记录方向
        self.entry_price: float = None
        self.hard_stop: float = None
        self.trailing_stop: float = None
        self.atr_baseline: float = None
        self.tp_triggered_30: bool = False
        self.tp_triggered_20: bool = False
        self.target_long: float = 0.0
        self.target_short: float = 0.0
        self.tp_base_price: float = None
        # 原策略仓位统计
        self.current_long: float = 0.0
        self.current_short: float = 0.0
        self.max_long: float = MAX_POS_LONG_TREND_LONG
        self.max_short: float = MAX_POS_SHORT_TREND_LONG
        # 新增MACD策略仓位（独立计算）
        self.macd_long: float = 0.0
        self.macd_short: float = 0.0
        self.prev_macd_off: float = None

    def update_position(self, side: str, qty: float, is_entry: bool, strategy: str = "normal") -> None:
        if strategy == "normal":
            if is_entry:
                if side.upper() == "BUY":
                    self.current_long += qty
                elif side.upper() == "SELL":
                    self.current_short += qty
            else:
                if side.upper() == "SELL":
                    self.current_long = max(0.0, self.current_long - qty)
                elif side.upper() == "BUY":
                    self.current_short = max(0.0, self.current_short - qty)
            logger.info(f"[原策略] 仓位更新: 多仓={self.current_long:.4f} ETH, 空仓={self.current_short:.4f} ETH")
        elif strategy == "macd":
            if is_entry:
                if side.upper() == "BUY":
                    self.macd_long += qty
                elif side.upper() == "SELL":
                    self.macd_short += qty
            else:
                if side.upper() == "SELL":
                    self.macd_long = max(0.0, self.macd_long - qty)
                elif side.upper() == "BUY":
                    self.macd_short = max(0.0, self.macd_short - qty)
            logger.info(f"[MACD策略] 仓位更新: 多仓={self.macd_long:.4f} ETH, 空仓={self.macd_short:.4f} ETH")

    async def trend_monitoring_loop(self) -> None:
        while True:
            try:
                trend = await self.analyze_trend_15m()
                logger.info(f"15分钟趋势: {trend}")
                await self.adjust_position_ratio(trend)
                if self.last_trade_side and trend != self.last_trade_side:
                    await self.handle_trend_reversal(trend)
                self.last_trade_side = trend
            except Exception as e:
                logger.error(f"趋势监控异常: {e}")
            await asyncio.sleep(60)

    async def analyze_trend_15m(self) -> str:
        df = await self.client.fetch_klines(interval='15m', limit=100)
        if df.empty:
            return 'LONG'
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low  = df['low'].values.astype(np.float64)
        atr = self._vectorized_atr(high, low, close, period=self.config.st_period)
        last_atr = atr[-1]
        hl2 = (df['high'] + df['low']) / 2
        basic_upper = hl2 + self.config.st_multiplier * last_atr
        basic_lower = hl2 - self.config.st_multiplier * last_atr
        final_upper = [basic_upper.iloc[0]]
        final_lower = [basic_lower.iloc[0]]
        for i in range(1, len(close)):
            curr_upper = basic_upper.iloc[i] if (basic_upper.iloc[i] < final_upper[-1] or close[i-1] > final_upper[-1]) else final_upper[-1]
            curr_lower = basic_lower.iloc[i] if (basic_lower.iloc[i] > final_lower[-1] or close[i-1] < final_lower[-1]) else final_lower[-1]
            final_upper.append(curr_upper)
            final_lower.append(curr_lower)
        final_upper = np.array(final_upper)
        final_lower = np.array(final_lower)
        if close[-1] > final_upper[-1]:
            return 'LONG'
        elif close[-1] < final_lower[-1]:
            return 'SHORT'
        else:
            ema_fast = pd.Series(close).ewm(span=self.config.macd_fast, adjust=False).mean().values
            ema_slow = pd.Series(close).ewm(span=self.config.macd_slow, adjust=False).mean().values
            macd_line = ema_fast - ema_slow
            return 'LONG' if macd_line[-1] >= 0 else 'SHORT'

    def _vectorized_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        ])
        atr = np.convolve(tr, np.ones(period), 'valid') / period
        return atr

    async def adjust_position_ratio(self, trend: str) -> None:
        df = await self.client.fetch_klines(interval='15m', limit=50)
        if df.empty:
            return
        atr_arr = self._vectorized_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        atr_val = atr_arr[-1]
        recent_atr = np.mean(atr_arr)
        if trend == 'LONG':
            if atr_val < 0.3 * recent_atr:
                self.target_long = BULL_LONG_QTY_SHAKE
                self.target_short = BULL_SHORT_QTY_SHAKE
            else:
                self.target_long = BULL_LONG_QTY
                self.target_short = BULL_SHORT_QTY
            self.max_long = MAX_POS_LONG_TREND_LONG
            self.max_short = MAX_POS_SHORT_TREND_LONG
        else:
            if atr_val < 0.3 * recent_atr:
                self.target_long = BEAR_LONG_QTY_SHAKE
                self.target_short = BEAR_SHORT_QTY_SHAKE
            else:
                self.target_long = BEAR_LONG_QTY
                self.target_short = BEAR_SHORT_QTY
            self.max_long = MAX_POS_LONG_TREND_SHORT
            self.max_short = MAX_POS_SHORT_TREND_SHORT
        logger.info(f"[原策略] 目标仓位: 多仓={self.target_long} ETH, 空仓={self.target_short} ETH")
        logger.info(f"[原策略] 仓位上限: 多仓上限={self.max_long} ETH, 空仓上限={self.max_short} ETH")

    async def handle_trend_reversal(self, new_trend: str) -> None:
        logger.info(f"趋势反转，当前新趋势: {new_trend}")
        if self.last_trade_side == 'LONG' and new_trend == 'SHORT':
            if self.current_short > 0:
                await self.close_position(side='BUY', ratio=0.5, strategy="normal")
        elif self.last_trade_side == 'SHORT' and new_trend == 'LONG':
            if self.current_long > 0:
                await self.close_position(side='SELL', ratio=0.5, strategy="normal")
        asyncio.create_task(self.gradual_position_adjustment())

    async def gradual_position_adjustment(self) -> None:
        for _ in range(12):
            await self.rebalance_hedge()
            await asyncio.sleep(300)

    async def rebalance_hedge(self) -> None:
        logger.info("[原策略] 执行仓位再平衡：逐步调整为1:1")
        diff = self.current_long - self.current_short
        if abs(diff) < 1e-4:
            logger.info("[原策略] 仓位接近1:1，无需调整")
            return
        if diff > 0:
            exit_qty = diff * 0.5
            if exit_qty * self.entry_price < MIN_NOTIONAL:
                logger.info("平仓金额较小，跳过再平衡")
                return
            await self.close_position(side='SELL', ratio=exit_qty / self.current_long, strategy="normal")
        else:
            exit_qty = (-diff) * 0.5
            if exit_qty * self.entry_price < MIN_NOTIONAL:
                logger.info("平仓金额较小，跳过再平衡")
                return
            await self.close_position(side='BUY', ratio=exit_qty / self.current_short, strategy="normal")

    async def analyze_order_signals_3m(self) -> Tuple[Signal, List[Dict[str, Any]]]:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return Signal(False, 'NONE', 0, 0), []
        close_prices = df['close'].values.astype(np.float64)
        sma = pd.Series(close_prices).rolling(window=self.config.bb_period).mean().values
        std = pd.Series(close_prices).rolling(window=self.config.bb_period).std().values
        upper_band = sma + self.config.bb_std * std
        lower_band = sma - self.config.bb_std * std
        bb_range = upper_band[-1] - lower_band[-1]
        latest_price = close_prices[-1]
        percent_b = 0.0 if bb_range == 0 else (latest_price - lower_band[-1]) / bb_range
        logger.info(f"[原策略] 3m %B: {percent_b:.3f}")
        if percent_b <= 0.0:
            return Signal(True, 'BUY', 0, 0, order_details={'trigger_price': latest_price, 'percent_b': percent_b}), [
                {'offset': 0.0005, 'ratio': 0.20},
                {'offset': 0.0015, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.20},
                {'offset': 0.0075, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
        elif percent_b >= 1.0:
            return Signal(True, 'SELL', 0, 0, order_details={'trigger_price': latest_price, 'percent_b': percent_b}), [
                {'offset': 0.0005, 'ratio': 0.20},
                {'offset': 0.0015, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.20},
                {'offset': 0.0075, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
        return Signal(False, 'NONE', 0, 0), []

    async def place_dynamic_limit_orders(self, side: str, order_list: List[Dict[str, Any]], trigger_price: float, strategy: str = "normal") -> None:
        pos_side = "LONG" if side == "BUY" else "SHORT"
        if strategy == "normal":
            if self.last_triggered_side is not None and self.last_triggered_side.upper() == side.upper():
                logger.info("[原策略] 同方向信号连续触发，忽略此次下单")
                return
        for order in order_list:
            offset = order['offset']
            ratio = order['ratio']
            order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
            if order_qty <= 0:
                logger.error("无效订单数量，跳过")
                continue
            limit_price = round(trigger_price * (1 - offset) if side == "BUY" else trigger_price * (1 + offset), self.config.price_precision)
            if limit_price <= 0:
                logger.error("无效挂单价格，跳过")
                continue
            notional = limit_price * order_qty
            if notional < MIN_NOTIONAL:
                logger.error(f"订单名义 {notional} USDT 小于 {MIN_NOTIONAL} USDT，跳过")
                continue
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': limit_price,
                'quantity': order_qty,
                'timeInForce': 'GTC',
                'positionSide': pos_side
            }
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"[{strategy.upper()}] 挂单成功: {side}@{limit_price}，数量: {order_qty}（偏移 {offset*100:.2f}%） 返回: {data}")
                self.update_position(side, order_qty, is_entry=True, strategy=strategy)
                if strategy == "normal":
                    self.last_triggered_side = side.upper()
            except Exception as e:
                logger.error(f"[{strategy.upper()}] 下单失败: {e}")

    async def get_current_percentb(self) -> float:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return 0.5
        close_prices = df['close'].values.astype(np.float64)
        sma = pd.Series(close_prices).rolling(window=self.config.bb_period).mean().values
        std = pd.Series(close_prices).rolling(window=self.config.bb_period).std().values
        upper_band = sma + self.config.bb_std * std
        lower_band = sma - self.config.bb_std * std
        bb_range = upper_band[-1] - lower_band[-1]
        percent_b = 0.0 if bb_range == 0 else (close_prices[-1] - lower_band[-1]) / bb_range
        logger.info(f"[原策略] 3m %B: {percent_b:.3f}")
        return percent_b

    async def cancel_all_orders(self) -> None:
        logger.info("撤销所有订单")
        params = {'symbol': SYMBOL}
        try:
            await self.client._signed_request('DELETE', '/openOrders', params)
        except Exception as e:
            logger.error(f"撤单异常: {e}")

    async def update_dynamic_stop_loss(self) -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or not self.last_trade_side:
            return
        latest_price = df['close'].values[-1]
        close_prices = df['close'].values.astype(np.float64)
        sma = pd.Series(close_prices).rolling(window=self.config.bb_period).mean().values
        std = pd.Series(close_prices).rolling(window=self.config.bb_period).std().values
        upper_band = sma + self.config.bb_std * std
        lower_band = sma - self.config.bb_std * std
        bb_width = upper_band[-1] - lower_band[-1]
        dynamic_offset = bb_width * 0.5
        if self.last_trade_side == 'LONG':
            new_stop = latest_price - dynamic_offset
            if self.hard_stop is None or new_stop > self.hard_stop:
                self.hard_stop = new_stop
            self.trailing_stop = latest_price - dynamic_offset
            logger.info(f"[原策略] 多单止损更新: 当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")
        elif self.last_trade_side == 'SHORT':
            new_stop = latest_price + dynamic_offset
            if self.hard_stop is None or new_stop < self.hard_stop:
                self.hard_stop = new_stop
            self.trailing_stop = latest_price + dynamic_offset
            logger.info(f"[原策略] 空单止损更新: 当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")

    async def manage_profit_targets(self) -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.entry_price is None:
            return
        current_price = df['close'].values[-1]
        current_percentb = await self.get_current_percentb()
        if self.last_trade_side == 'LONG':
            if current_percentb >= 0.80 and not self.tp_triggered_30:
                logger.info("[原策略] 多单止盈信号：市价平30%")
                await self.close_position(side='SELL', ratio=0.3, strategy="normal")
                self.tp_triggered_30 = True
                self.tp_base_price = current_price
            elif current_percentb >= 0.90 and not self.tp_triggered_20:
                logger.info("[原策略] 多单止盈信号：市价平20%")
                await self.close_position(side='SELL', ratio=0.2, strategy="normal")
                self.tp_triggered_20 = True
                self.tp_base_price = current_price
            if self.tp_base_price is not None:
                await self.place_take_profit_orders(side='SELL', base_price=self.tp_base_price)
        elif self.last_trade_side == 'SHORT':
            if current_percentb <= 0.20 and not self.tp_triggered_30:
                logger.info("[原策略] 空单止盈信号：市价平30%")
                await self.close_position(side='BUY', ratio=0.3, strategy="normal")
                self.tp_triggered_30 = True
                self.tp_base_price = current_price
            elif current_percentb <= 0.10 and not self.tp_triggered_20:
                logger.info("[原策略] 空单止盈信号：市价平20%")
                await self.close_position(side='BUY', ratio=0.2, strategy="normal")
                self.tp_triggered_20 = True
                self.tp_base_price = current_price
            if self.tp_base_price is not None:
                await self.place_take_profit_orders(side='BUY', base_price=self.tp_base_price)
        if self.last_trade_side == 'LONG' and current_price < self.hard_stop:
            logger.info("[原策略] 多单：当前价低于止损，触发平仓")
            await self.close_position(side='SELL', ratio=1.0, strategy="normal")
        elif self.last_trade_side == 'SHORT' and current_price > self.hard_stop:
            logger.info("[原策略] 空单：当前价高于止损，触发平仓")
            await self.close_position(side='BUY', ratio=1.0, strategy="normal")

    async def place_take_profit_orders(self, side: str, base_price: float) -> None:
        pos_side = self.last_trade_side
        tp_offsets = [0.0040, 0.0120]
        remaining_qty = QUANTITY
        qty_each = round(remaining_qty / (len(tp_offsets) + 1), self.config.quantity_precision)
        for offset in tp_offsets:
            tp_price = round(base_price * (1 + offset), self.config.price_precision) if side == 'SELL' else round(base_price * (1 - offset), self.config.price_precision)
            notional = tp_price * qty_each
            if notional < MIN_NOTIONAL:
                logger.error(f"止盈订单名义 {notional} USDT 小于 {MIN_NOTIONAL} USDT，跳过")
                continue
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': tp_price,
                'quantity': qty_each,
                'timeInForce': 'GTC',
                'positionSide': pos_side
            }
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"[原策略] 止盈单成功: {side}@{tp_price}，数量: {qty_each}（偏移 {offset*100:.2f}%） 返回: {data}")
                self.update_position(side, qty_each, is_entry=False, strategy="normal")
            except Exception as e:
                logger.error(f"[原策略] 止盈单失败: {e}")

    async def close_position(self, side: str, ratio: float, strategy: str = "normal") -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.entry_price is None:
            return
        current_price = df['close'].values[-1]
        price_limit = current_price * (1 + self.config.max_slippage_market) if side == 'BUY' else current_price * (1 - self.config.max_slippage_market)
        logger.info(f"[{strategy.upper()}] 市价平仓请求: side={side}, ratio={ratio}, 价格限制={price_limit:.2f}")
        pos_side = self.last_trade_side
        order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
        if current_price * order_qty < MIN_NOTIONAL:
            logger.error(f"市价单名义 {current_price * order_qty} USDT 小于 {MIN_NOTIONAL} USDT，跳过")
            return
        if side.upper() == "SELL" and current_price < self.entry_price * (1 - self.config.max_slippage_market):
            logger.warning(f"[{strategy.upper()}] SELL 平仓滑点过大，采用阶梯报价")
            ladder_orders = [{'offset': 0.0010, 'ratio': 0.5}, {'offset': 0.0020, 'ratio': 0.5}]
            await self.place_dynamic_limit_orders(side, ladder_orders, trigger_price=current_price, strategy=strategy)
            return
        elif side.upper() == "BUY" and current_price > self.entry_price * (1 + self.config.max_slippage_market):
            logger.warning(f"[{strategy.upper()}] BUY 平仓滑点过大，采用阶梯报价")
            ladder_orders = [{'offset': 0.0010, 'ratio': 0.5}, {'offset': 0.0020, 'ratio': 0.5}]
            await self.place_dynamic_limit_orders(side, ladder_orders, trigger_price=current_price, strategy=strategy)
            return
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'MARKET',
            'quantity': order_qty,
            'positionSide': pos_side
        }
        try:
            data = await self.client._signed_request('POST', '/order', params)
            logger.info(f"[{strategy.upper()}] 市价平仓成功，返回: {data}")
            self.update_position(side, order_qty, is_entry=False, strategy=strategy)
        except Exception as e:
            logger.error(f"[{strategy.upper()}] 平仓失败: {e}")

    async def order_signal_loop(self) -> None:
        while True:
            try:
                signal, order_list = await self.analyze_order_signals_3m()
                if signal.action:
                    # 优化：若上次下单方向与本次信号相同则忽略此次下单
                    if self.last_triggered_side is not None and self.last_triggered_side.upper() == signal.side.upper():
                        logger.info("[原策略] 信号方向同上次，忽略此次下单")
                    else:
                        if signal.side.upper() == 'BUY':
                            if self.current_long < self.target_long and self.current_long < self.max_long:
                                await self.place_dynamic_limit_orders(signal.side, order_list, trigger_price=signal.order_details.get("trigger_price"), strategy="normal")
                                self.last_triggered_side = signal.side.upper()
                            else:
                                logger.info("[原策略] 多仓已达到目标或上限，暂停买单")
                        elif signal.side.upper() == 'SELL':
                            if self.current_short < self.target_short and self.current_short < self.max_short:
                                await self.place_dynamic_limit_orders(signal.side, order_list, trigger_price=signal.order_details.get("trigger_price"), strategy="normal")
                                self.last_triggered_side = signal.side.upper()
                            else:
                                logger.info("[原策略] 空仓已达到目标或上限，暂停卖单")
                else:
                    # 无信号时清空记录，保证轮换条件持续
                    self.last_triggered_side = None
            except Exception as e:
                logger.error(f"[原策略] 下单信号异常: {e}")
            # 提高检测频率为1秒
            await asyncio.sleep(1)

    async def stop_loss_profit_management_loop(self) -> None:
        while True:
            try:
                await self.update_dynamic_stop_loss()
                await self.manage_profit_targets()
            except Exception as e:
                logger.error(f"[原策略] 止盈止损异常: {e}")
            await asyncio.sleep(0.5)

    # 新增MACD离轴策略，独立运行，不采用信号轮换
    async def macd_strategy_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                close_prices = df['close'].values.astype(np.float64)
                ema12 = pd.Series(close_prices).ewm(span=12, adjust=False).mean().values
                ema26 = pd.Series(close_prices).ewm(span=26, adjust=False).mean().values
                curr_off = 2 * (ema12[-1] - ema26[-1])
                logger.info(f"[MACD] 当前离轴值: {curr_off:.2f}")
                if self.prev_macd_off is not None:
                    if self.prev_macd_off <= 0 and curr_off > 0:
                        trigger = curr_off
                        logger.info(f"[MACD] 上穿触发，条件: {trigger:.2f}")
                        if trigger >= 11 and trigger < 20:
                            order_size = MACD_SMALL_ORDER
                        elif trigger >= 20:
                            order_size = MACD_BIG_ORDER
                        else:
                            order_size = 0.0
                        if order_size > 0:
                            entry_price = close_prices[-1]
                            stop_loss = entry_price * 1.04
                            logger.info(f"[MACD] 触发空单，下单 {order_size} ETH，止损 {stop_loss:.2f}")
                            await self.close_position(side='SELL', ratio=order_size / QUANTITY, strategy="macd")
                        else:
                            logger.info("[MACD] 信号条件不足，不下空单")
                    elif self.prev_macd_off >= 0 and curr_off < 0:
                        trigger = curr_off
                        logger.info(f"[MACD] 下穿触发，条件: {trigger:.2f}")
                        if trigger <= -11 and trigger > -20:
                            order_size = MACD_SMALL_ORDER
                        elif trigger <= -20:
                            order_size = MACD_BIG_ORDER
                        else:
                            order_size = 0.0
                        if order_size > 0:
                            entry_price = close_prices[-1]
                            stop_loss = entry_price * 0.96
                            logger.info(f"[MACD] 触发多单，下单 {order_size} ETH，止损 {stop_loss:.2f}")
                            await self.close_position(side='BUY', ratio=order_size / QUANTITY, strategy="macd")
                        else:
                            logger.info("[MACD] 信号条件不足，不下多单")
                self.prev_macd_off = curr_off
            except Exception as e:
                logger.error(f"[MACD] 策略异常: {e}")
            await asyncio.sleep(60*15)

    async def execute(self) -> None:
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        await asyncio.gather(
            self.trend_monitoring_loop(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.macd_strategy_loop()
        )

async def main() -> None:
    client = BinanceHFTClient()
    strategy = ETHUSDTStrategy(client)
    try:
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")

if __name__ == "__main__":
    asyncio.run(main())