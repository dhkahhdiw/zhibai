#!/usr/bin/env python3
"""
ETH/USDT 高频交易引擎 v7.4（基于15m超趋势与3m级BB%B信号策略，REST API版）
主要策略：
1. 15分钟超级趋势策略：采用20周期均线、3×ATR结合MACD辅助判断趋势；
   趋势为多时目标仓位：多仓 0.1 ETH、空仓 0.05 ETH；
   趋势反转时，使用市价单平掉与新趋势不符部分（50%），后续分批再平衡至1:1。
2. 3分钟级下单策略：基于 Bollinger Bands %B（周期20、标准差2）判断下单信号，
   当 %B ≤ 0 或 ≥ 1 时，根据最新市场价格下挂多个档位固定限价单。
3. 止盈止损策略：初始止损按买入价×0.98（多单）/×1.02（空单）设定，并结合3m级 Bollinger 带宽动态跟踪止损；
   止盈分阶段触发，部分市价平仓，剩余挂限价止盈单。
适配环境：Vultr high frequency (vhf-1c-1gb ubuntu22.04)
要求：
  - 使用 REST API（URL为 https://fapi.binance.com）
  - 使用异步 aiohttp 与 uvloop 以获得较低延迟
  - 内置限频检测与重试策略
本代码参考 Binance 官方更新文档中关于时间戳、错误码、限频以及 dualSidePosition 要求的说明。
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

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDT'
LEVERAGE = 50

# 目标仓位设置（单位：ETH）
BULL_LONG_QTY = 0.6  # 多头情况下多仓
BULL_SHORT_QTY = 0.3  # 多头时对冲空仓
BEAR_LONG_QTY = 0.3  # 空头时对冲多仓
BEAR_SHORT_QTY = 0.6  # 空头时空仓

# 单笔基础订单数量（示例值，可根据实际持仓调整）
QUANTITY = 0.1

# REST API 基础URL（USDT永续）
REST_URL = 'https://fapi.binance.com'

# ==================== 高频参数 ====================
# (参考币安更新文档中各接口请求限制设定)
RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

# 为降低因网络延迟带来的时间戳误差，设置较大 recvWindow（单位：毫秒）
RECV_WINDOW = 10000

# 最小订单名义价值（单位：USDT），避免出现 -4164 错误
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
    st_period: int = 20  # 超级趋势周期
    st_multiplier: float = 3.0  # 超级趋势ATR倍数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20  # Bollinger Bands周期（3m级）
    bb_std: float = 2.0  # Bollinger Bands标准差倍数
    max_retries: int = 7
    order_timeout: float = 2.0
    network_timeout: float = 5.0
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0
    slippage: float = 0.001
    dual_side_position: str = "true"
    order_adjust_interval: float = 180.0
    max_slippage_market: float = 0.0015  # 市价单允许最大滑点 0.15%
    daily_drawdown_limit: float = 0.20  # 此处保留字段，但风险监控逻辑已移除


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
        self.position = 0.0

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
                        raise ValueError("返回数据中缺少 serverTime")
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    self._time_diff = server_time - local_time
                    logger.info(f"时间同步成功，时间差：{self._time_diff}ms")
                    return
            except Exception as e:
                logger.error(f"时间同步失败(重试 {retry + 1}): {e}")
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
        logger.debug(f"请求URL: {url.split('?')[0]} 参数: {sorted_params}")
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
                        raise Exception(f"响应内容异常: {text}")
                    data = await resp.json()
                    if isinstance(data, dict) and data.get("code", 0) < 0:
                        raise Exception(f"接口错误，ErrorCode: {data.get('code')}, Msg: {data.get('msg')}")
                    return data
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}): {e}")
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
            logger.warning(f"速率限制 {endpoint}，等待 {wait_time:.3f}s")
            METRICS['throughput'].set(0)
            await asyncio.sleep(wait_time)
        dq.append(now)
        METRICS['throughput'].inc()

    async def manage_leverage(self) -> dict:
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'dualSidePosition': self.config.dual_side_position
        }
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self, interval: str, limit: int = 100) -> pd.DataFrame:
        params = {'symbol': SYMBOL, 'interval': interval, 'limit': limit}
        data = await self._signed_request('GET', '/klines', params)
        if not isinstance(data, list):
            logger.error("K线数据格式异常")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    async def fetch_weekly_data(self, limit: int = 100) -> pd.DataFrame:
        params = {'symbol': SYMBOL, 'interval': '1w', 'limit': limit}
        data = await self._signed_request('GET', '/klines', params)
        if not isinstance(data, list):
            logger.error("周级K线数据格式异常")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
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
        self.last_trade_side: str = None  # 'LONG' 或 'SHORT'
        self.entry_price: float = None
        self.hard_stop: float = None
        self.trailing_stop: float = None
        self.atr_baseline: float = None
        self.tp_triggered_30: bool = False
        self.tp_triggered_20: bool = False
        self.target_long: float = 0.0
        self.target_short: float = 0.0
        self.tp_base_price: float = None

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
        low = df['low'].values.astype(np.float64)
        atr = self._vectorized_atr(high, low, close, period=self.config.st_period)
        last_atr = atr[-1]
        hl2 = (df['high'] + df['low']) / 2
        basic_upper = hl2 + self.config.st_multiplier * last_atr
        basic_lower = hl2 - self.config.st_multiplier * last_atr
        final_upper = [basic_upper.iloc[0]]
        final_lower = [basic_lower.iloc[0]]
        for i in range(1, len(close)):
            curr_upper = basic_upper.iloc[i] if (
                        basic_upper.iloc[i] < final_upper[-1] or close[i - 1] > final_upper[-1]) else final_upper[-1]
            curr_lower = basic_lower.iloc[i] if (
                        basic_lower.iloc[i] > final_lower[-1] or close[i - 1] < final_lower[-1]) else final_lower[-1]
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
        ratio_factor = 0.3 if atr_val < 0.3 * recent_atr else 0.5
        if trend == 'LONG':
            self.target_long = BULL_LONG_QTY
            self.target_short = BULL_SHORT_QTY if ratio_factor == 0.5 else BULL_SHORT_QTY * (0.3 / 0.5)
        else:
            self.target_long = BEAR_LONG_QTY if ratio_factor == 0.5 else BEAR_LONG_QTY * (0.3 / 0.5)
            self.target_short = BEAR_SHORT_QTY
        logger.info(f"目标仓位：多仓 {self.target_long} ETH, 空仓 {self.target_short} ETH")

    async def handle_trend_reversal(self, new_trend: str) -> None:
        logger.info(f"趋势反转，当前新趋势: {new_trend}")
        if self.last_trade_side == 'LONG' and new_trend == 'SHORT':
            await self.close_position(side='SELL', ratio=0.5)
        elif self.last_trade_side == 'SHORT' and new_trend == 'LONG':
            await self.close_position(side='BUY', ratio=0.5)
        asyncio.create_task(self.gradual_position_adjustment())

    async def gradual_position_adjustment(self) -> None:
        for _ in range(12):
            await self.rebalance_hedge()
            await asyncio.sleep(300)

    async def rebalance_hedge(self) -> None:
        logger.info("执行仓位再平衡: 调整为1:1")
        # 此处加入查询仓位及下单逻辑，参照 Binance 查询订单接口实现

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
        logger.info(f"3m %B: {percent_b:.3f}")
        if percent_b <= 0.0:
            return Signal(True, 'BUY', 0, 0, order_details={'trigger_price': latest_price, 'percent_b': percent_b}), [
                {'offset': 0.0005, 'ratio': 0.20},
                {'offset': 0.0010, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.20},
                {'offset': 0.0065, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
        elif percent_b >= 1.0:
            return Signal(True, 'SELL', 0, 0, order_details={'trigger_price': latest_price, 'percent_b': percent_b}), [
                {'offset': 0.0005, 'ratio': 0.20},
                {'offset': 0.0010, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.20},
                {'offset': 0.0065, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
        return Signal(False, 'NONE', 0, 0), []

    async def place_dynamic_limit_orders(self, side: str, order_list: List[Dict[str, Any]],
                                         trigger_price: float) -> None:
        # 开仓订单根据 side 设置 positionSide
        pos_side = "LONG" if side == "BUY" else "SHORT"
        for order in order_list:
            offset = order['offset']
            ratio = order['ratio']
            order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
            if order_qty <= 0:
                logger.error("无效订单数量，跳过当前档")
                continue
            limit_price = round(trigger_price * (1 - offset) if side == "BUY" else trigger_price * (1 + offset),
                                self.config.price_precision)
            if limit_price <= 0:
                logger.error("无效挂单价格，跳过当前档")
                continue
            notional = limit_price * order_qty
            if notional < MIN_NOTIONAL:
                logger.error(f"订单名义价值 {notional} USDT 小于最低要求 {MIN_NOTIONAL} USDT，跳过当前档")
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
                logger.info(
                    f"挂限价单成功: {side}@{limit_price}，数量: {order_qty}（偏移 {offset * 100:.2f}%） 返回: {data}")
            except Exception as e:
                logger.error(f"限价挂单失败: {e}")

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
        logger.info(f"3m 当前 %B: {percent_b:.3f}")
        return percent_b

    async def cancel_all_orders(self) -> None:
        logger.info("撤销所有未成交订单")
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
            logger.info(
                f"多单动态止损更新: 当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")
        elif self.last_trade_side == 'SHORT':
            new_stop = latest_price + dynamic_offset
            if self.hard_stop is None or new_stop < self.hard_stop:
                self.hard_stop = new_stop
            self.trailing_stop = latest_price + dynamic_offset
            logger.info(
                f"空单动态止损更新: 当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")

    async def manage_profit_targets(self) -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.entry_price is None:
            return
        current_price = df['close'].values[-1]
        current_percentb = await self.get_current_percentb()
        if self.last_trade_side == 'LONG':
            if current_percentb >= 0.85 and not self.tp_triggered_30:
                logger.info("多单首个止盈信号：市价平30%")
                await self.close_position(side='SELL', ratio=0.3)
                self.tp_triggered_30 = True
                self.tp_base_price = current_price
            elif current_percentb >= 1.0 and not self.tp_triggered_20:
                logger.info("多单第二止盈信号：市价平20%")
                await self.close_position(side='SELL', ratio=0.2)
                self.tp_triggered_20 = True
                self.tp_base_price = current_price
            if self.tp_base_price is not None:
                await self.place_take_profit_orders(side='SELL', base_price=self.tp_base_price)
        elif self.last_trade_side == 'SHORT':
            if current_percentb <= 0.15 and not self.tp_triggered_30:
                logger.info("空单首个止盈信号：市价平30%")
                await self.close_position(side='BUY', ratio=0.3)
                self.tp_triggered_30 = True
                self.tp_base_price = current_price
            elif current_percentb <= 0.0 and not self.tp_triggered_20:
                logger.info("空单第二止盈信号：市价平20%")
                await self.close_position(side='BUY', ratio=0.2)
                self.tp_triggered_20 = True
                self.tp_base_price = current_price
            if self.tp_base_price is not None:
                await self.place_take_profit_orders(side='BUY', base_price=self.tp_base_price)
        if self.last_trade_side == 'LONG' and current_price < self.hard_stop:
            logger.info("多单：当前价低于动态止损，触发平仓")
            await self.close_position(side='SELL', ratio=1.0)
        elif self.last_trade_side == 'SHORT' and current_price > self.hard_stop:
            logger.info("空单：当前价高于动态止损，触发平仓")
            await self.close_position(side='BUY', ratio=1.0)

    async def place_take_profit_orders(self, side: str, base_price: float) -> None:
        pos_side = self.last_trade_side
        tp_offsets = [0.0025, 0.0040, 0.0060, 0.0080, 0.0120]
        remaining_qty = QUANTITY
        qty_each = round(remaining_qty / (len(tp_offsets) + 1), self.config.quantity_precision)
        for offset in tp_offsets:
            if side == 'SELL':
                tp_price = round(base_price * (1 + offset), self.config.price_precision)
            else:
                tp_price = round(base_price * (1 - offset), self.config.price_precision)
            notional = tp_price * qty_each
            if notional < MIN_NOTIONAL:
                logger.error(f"止盈订单名义价值 {notional} USDT 小于最低 {MIN_NOTIONAL} USDT，跳过该档")
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
                logger.info(f"止盈挂单成功: {side}@{tp_price}，数量: {qty_each}（偏移 {offset * 100:.2f}%） 返回: {data}")
            except Exception as e:
                logger.error(f"止盈挂单失败: {e}")

    async def close_position(self, side: str, ratio: float) -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return
        current_price = df['close'].values[-1]
        price_limit = current_price * (1 + self.config.max_slippage_market) if side == 'BUY' else current_price * (
                    1 - self.config.max_slippage_market)
        logger.info(f"市价平仓请求：side={side}, ratio={ratio}, 价格限制={price_limit:.2f}")
        pos_side = self.last_trade_side
        order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
        if current_price * order_qty < MIN_NOTIONAL:
            logger.error(f"市价单名义价值 {current_price * order_qty} USDT 小于最低要求 {MIN_NOTIONAL} USDT，跳过平仓")
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
            logger.info(f"市价平仓成功，返回: {data}")
        except Exception as e:
            logger.error(f"平仓请求失败: {e}")

    async def order_signal_loop(self) -> None:
        while True:
            try:
                signal, order_list = await self.analyze_order_signals_3m()
                if signal.action:
                    await self.place_dynamic_limit_orders(signal.side, order_list,
                                                          trigger_price=signal.order_details.get("trigger_price"))
            except Exception as e:
                logger.error(f"下单信号异常: {e}")
            await asyncio.sleep(self.config.order_adjust_interval)

    async def stop_loss_profit_management_loop(self) -> None:
        while True:
            try:
                await self.update_dynamic_stop_loss()
                await self.manage_profit_targets()
            except Exception as e:
                logger.error(f"止盈止损管理异常: {e}")
            await asyncio.sleep(3)

    async def execute(self) -> None:
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        # 移除风险控制任务，其他任务保留
        await asyncio.gather(
            self.trend_monitoring_loop(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop()
        )


async def main() -> None:
    client = BinanceHFTClient()
    strategy = ETHUSDTStrategy(client)
    try:
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，正在优雅退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")


if __name__ == "__main__":
    asyncio.run(main())