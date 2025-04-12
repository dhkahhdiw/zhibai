#!/usr/bin/env python3
"""
ETH/USDT 高频交易引擎 v7.4
  基于15m超趋势与3m级BB%B信号策略，REST API版

【策略说明】
1. 15分钟超级趋势策略
   - 主趋势为多时，目标仓位比例默认为 1:0.5（如做多0.1 ETH，对应做空0.05 ETH）；
   - 若ATR（14周期）低于近期均值30%，则自动调整为1:0.3（如0.1对0.03）；
   - 趋势为空时则反向设置。
   - 同时设定ETH持仓上限：若主趋势为多，则多仓上限0.6、空仓上限0.3；若趋势为空则反之。
2. 趋势转换风控机制
   - 当15分钟趋势反转时，立即市价平掉不符趋势半仓（如原趋势为多，则平空仓50%）；
   - 余下仓位在1小时内分批通过限价单逐步调整为1:1；
   - 市价单执行时允许最大0.15%滑点，若超限则采用阶梯报价下单。
3. 3分钟级下单策略
   - 基于 Bollinger Bands %B（周期20、标准差2）判断信号，
     当 %B ≤ 0 或 ≥ 1 时，下挂多个档位限价单。
4. 止盈止损策略
   - 初始止损设为买入价×0.98（多）/×1.02（空），并结合3m Bollinger 带宽动态跟踪止损；
   - 止盈分阶段触发后部分市价平仓，其余挂限价止盈单。

适配环境：Vultr high frequency (vhf-1c-1gb ubuntu22.04)
要求：
  - 使用 REST API（https://fapi.binance.com）
  - 异步 aiohttp 与 uvloop 实现低延迟
  - 内置限频、重试、时间同步
  - 严格统计及控制仓位比例
  - 趋势转换时进行风控，市价单附带滑点控制（超0.15%则采用阶梯报价）

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

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDT'
LEVERAGE = 50

# 目标仓位设置（单位：ETH）【常规目标】
BULL_LONG_QTY = 0.1  # 趋势为多时，多仓目标
BULL_SHORT_QTY = 0.583  # 趋势为多时，空仓目标（1:0.5）
BEAR_LONG_QTY = 0.583  # 趋势为空时，多仓目标
BEAR_SHORT_QTY = 0.1  # 趋势为空时，空仓目标

# 震荡市场下仓位目标调整为1:0.3
BULL_LONG_QTY_SHAKE = 0.1
BULL_SHORT_QTY_SHAKE = 0.3
BEAR_LONG_QTY_SHAKE = 0.3
BEAR_SHORT_QTY_SHAKE = 0.1

# 仓位上限配置（根据主趋势决定）
# 当主趋势为多时，多仓上限0.6，空仓上限0.3；趋势为空时反之
MAX_POS_LONG_TREND_LONG = 0.6
MAX_POS_SHORT_TREND_LONG = 0.35
MAX_POS_LONG_TREND_SHORT = 0.35
MAX_POS_SHORT_TREND_SHORT = 0.6

# 单笔基础订单数量
QUANTITY = 0.1

# REST API 基础URL
REST_URL = 'https://fapi.binance.com'

# ==================== 高频参数 ====================
RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

# 将 recvWindow 设置为 10000 毫秒以降低时间戳误差
RECV_WINDOW = 10000

# 最小订单名义价值（USDT），避免 -4164 错误
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
    st_multiplier: float = 3.0  # ATR 倍数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20  # Bollinger Bands 周期（3m级）
    bb_std: float = 2.0  # Bollinger Bands 标准差
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
    daily_drawdown_limit: float = 0.20  # 保留字段（风险监控逻辑已移除）


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
        self.last_trade_side: str = None  # 当前主趋势：'LONG' 或 'SHORT'
        self.entry_price: float = None
        self.hard_stop: float = None
        self.trailing_stop: float = None
        self.atr_baseline: float = None
        self.tp_triggered_30: bool = False
        self.tp_triggered_20: bool = False
        self.target_long: float = 0.0  # 目标多仓（策略订单量）
        self.target_short: float = 0.0  # 目标空仓（策略订单量）
        self.tp_base_price: float = None
        # 新增仓位统计
        self.current_long: float = 0.0  # 当前持有多仓（ETH）
        self.current_short: float = 0.0  # 当前持有空仓（ETH）
        # 新增仓位上限，根据趋势不同设定
        self.max_long: float = 0.6
        self.max_short: float = 0.3

    def update_position(self, side: str, qty: float, is_entry: bool) -> None:
        if is_entry:
            if side.upper() == "BUY":
                self.current_long += qty
            elif side.upper() == "SELL":
                self.current_short += qty
        else:
            if side.upper() == "SELL":  # 平多仓
                self.current_long = max(0.0, self.current_long - qty)
            elif side.upper() == "BUY":  # 平空仓
                self.current_short = max(0.0, self.current_short - qty)
        logger.info(f"更新仓位统计: 多仓={self.current_long:.4f} ETH, 空仓={self.current_short:.4f} ETH")

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
        # 根据ATR判断震荡市场：若ATR低于近期平均的30%，则调整目标仓位比例
        if trend == 'LONG':
            if atr_val < 0.3 * recent_atr:
                self.target_long = BULL_LONG_QTY_SHAKE
                self.target_short = BULL_SHORT_QTY_SHAKE
            else:
                self.target_long = BULL_LONG_QTY
                self.target_short = BULL_SHORT_QTY
            # 趋势为多时仓位上限
            self.max_long = MAX_POS_LONG_TREND_LONG
            self.max_short = MAX_POS_SHORT_TREND_LONG
        else:  # trend == 'SHORT'
            if atr_val < 0.3 * recent_atr:
                self.target_long = BEAR_LONG_QTY_SHAKE
                self.target_short = BEAR_SHORT_QTY_SHAKE
            else:
                self.target_long = BEAR_LONG_QTY
                self.target_short = BEAR_SHORT_QTY
            # 趋势为空时仓位上限反向
            self.max_long = MAX_POS_LONG_TREND_SHORT
            self.max_short = MAX_POS_SHORT_TREND_SHORT
        logger.info(f"目标仓位调整: 多仓目标={self.target_long} ETH, 空仓目标={self.target_short} ETH")
        logger.info(f"仓位上限设定: 多仓上限={self.max_long} ETH, 空仓上限={self.max_short} ETH")

    async def handle_trend_reversal(self, new_trend: str) -> None:
        logger.info(f"趋势反转，当前新趋势: {new_trend}")
        # 当趋势反转时，立即平掉与当前趋势不符的较大仓位50%
        if self.last_trade_side == 'LONG' and new_trend == 'SHORT':
            if self.current_short > 0:
                await self.close_position(side='BUY', ratio=0.5)
        elif self.last_trade_side == 'SHORT' and new_trend == 'LONG':
            if self.current_long > 0:
                await self.close_position(side='SELL', ratio=0.5)
        # 然后在1小时内分批逐步调整至1:1
        asyncio.create_task(self.gradual_position_adjustment())

    async def gradual_position_adjustment(self) -> None:
        # 每5分钟执行一次，共12次
        for _ in range(12):
            await self.rebalance_hedge()
            await asyncio.sleep(300)

    async def rebalance_hedge(self) -> None:
        logger.info("执行仓位再平衡：逐步调整为1:1比例")
        diff = self.current_long - self.current_short
        if abs(diff) < 1e-4:
            logger.info("当前仓位已接近1:1，无需调整")
            return
        if diff > 0:
            exit_qty = diff * 0.5
            if exit_qty * self.entry_price < MIN_NOTIONAL:
                logger.info("平仓金额较小，跳过逐步调整")
                return
            await self.close_position(side='SELL', ratio=exit_qty / self.current_long)
        else:
            exit_qty = (-diff) * 0.5
            if exit_qty * self.entry_price < MIN_NOTIONAL:
                logger.info("平仓金额较小，跳过逐步调整")
                return
            await self.close_position(side='BUY', ratio=exit_qty / self.current_short)

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
                {'offset': 0.0010, 'ratio': 0.20},
                {'offset': 0.0030, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.20},
                {'offset': 0.0065, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
        elif percent_b >= 1.0:
            return Signal(True, 'SELL', 0, 0, order_details={'trigger_price': latest_price, 'percent_b': percent_b}), [
                {'offset': 0.0010, 'ratio': 0.20},
                {'offset': 0.0030, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.20},
                {'offset': 0.0065, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
        return Signal(False, 'NONE', 0, 0), []

    async def place_dynamic_limit_orders(self, side: str, order_list: List[Dict[str, Any]],
                                         trigger_price: float) -> None:
        # 根据开仓信号设置 positionSide
        pos_side = "LONG" if side == "BUY" else "SHORT"
        # 仓位控制：检查对应仓位是否已达到上限
        if side.upper() == "BUY":
            if self.current_long >= self.max_long:
                logger.info(f"多仓已达到上限 {self.max_long} ETH，暂停买单")
                return
        elif side.upper() == "SELL":
            if self.current_short >= self.max_short:
                logger.info(f"空仓已达到上限 {self.max_short} ETH，暂停卖单")
                return
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
                # 更新仓位记录，认为成功下单为持仓增加
                self.update_position(side, order_qty, is_entry=True)
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
            tp_price = round(base_price * (1 + offset), self.config.price_precision) if side == 'SELL' else round(
                base_price * (1 - offset), self.config.price_precision)
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
                self.update_position(side, qty_each, is_entry=False)
            except Exception as e:
                logger.error(f"止盈挂单失败: {e}")

    async def close_position(self, side: str, ratio: float) -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.entry_price is None:
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
        # 滑点控制：若实际市场价格偏离开仓价超过0.15%，则转换为阶梯报价执行平仓
        if side.upper() == "SELL" and current_price < self.entry_price * (1 - self.config.max_slippage_market):
            logger.warning("市场滑点超过允许范围（SELL），采用阶梯报价平仓")
            ladder_order_list = [{'offset': 0.0010, 'ratio': 0.5}, {'offset': 0.0020, 'ratio': 0.5}]
            await self.place_dynamic_limit_orders(side, ladder_order_list, trigger_price=current_price)
            return
        elif side.upper() == "BUY" and current_price > self.entry_price * (1 + self.config.max_slippage_market):
            logger.warning("市场滑点超过允许范围（BUY），采用阶梯报价平仓")
            ladder_order_list = [{'offset': 0.0010, 'ratio': 0.5}, {'offset': 0.0020, 'ratio': 0.5}]
            await self.place_dynamic_limit_orders(side, ladder_order_list, trigger_price=current_price)
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
            self.update_position(side, order_qty, is_entry=False)
        except Exception as e:
            logger.error(f"平仓请求失败: {e}")

    async def order_signal_loop(self) -> None:
        while True:
            try:
                signal, order_list = await self.analyze_order_signals_3m()
                if signal.action:
                    # 根据仓位目标与当前持仓严格控制下单
                    if signal.side.upper() == 'BUY':
                        if self.current_long < self.target_long and self.current_long < self.max_long:
                            await self.place_dynamic_limit_orders(signal.side, order_list,
                                                                  trigger_price=signal.order_details.get(
                                                                      "trigger_price"))
                        else:
                            logger.info("多仓已达到目标或上限，暂停买单")
                    elif signal.side.upper() == 'SELL':
                        if self.current_short < self.target_short and self.current_short < self.max_short:
                            await self.place_dynamic_limit_orders(signal.side, order_list,
                                                                  trigger_price=signal.order_details.get(
                                                                      "trigger_price"))
                        else:
                            logger.info("空仓已达到目标或上限，暂停卖单")
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