#!/usr/bin/env python3
# ETH/USDC 高频交易引擎 v7.4（整合优化版）

import uvloop

uvloop.install()

import os, asyncio, time, hmac, hashlib, urllib.parse, logging, datetime
from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector, ClientConnectorError, ServerDisconnectedError
from aiohttp_retry import RetryClient, ExponentialRetry
from aiohttp.resolver import AsyncResolver

from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server

# ==================== 环境配置 ====================
_env_path = '/root/zhibai/.env'
load_dotenv(_env_path)

# 合约参数
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDC'
LEVERAGE = 10
QUANTITY = 0.06
REST_URL = 'https://fapi.binance.com'

# ==================== 高频参数 ====================
RATE_LIMITS = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

# ==================== 监控指标 ====================
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
        logging.FileHandler("/var/log/eth_usdc_hft.log", encoding="utf-8", mode='a')
    ]
)
logger = logging.getLogger('ETH-USDC-HFT')


@dataclass
class TradingConfig:
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    vol_multiplier: float = 2.5
    st_period: int = 20  # 超级趋势MA周期
    st_multiplier: float = 3.0  # ATR倍数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20  # Bollinger Bands周期20
    bb_std: float = 2.0  # 2倍标准差
    max_retries: int = 7
    order_timeout: float = 2.0
    network_timeout: float = 5.0
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0
    slippage: float = 0.001
    dual_side_position: str = "true"
    order_adjust_interval: float = 180.0
    max_slippage_market: float = 0.0015
    daily_drawdown_limit: float = 0.20


class BinanceHFTClient:
    def __init__(self):
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=512,
            resolver=AsyncResolver(),
            ttl_dns_cache=300,
            force_close=True,
            ssl=False
        )
        self._init_session()
        self.recv_window = 6000
        self.request_timestamps = defaultdict(lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 200))
        self._time_diff = 0
        self.position = 0.0  # 实际持仓量

    def _init_session(self):
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

    async def sync_server_time(self):
        url = REST_URL + "/fapi/v1/time"
        for retry in range(5):
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                    data = await resp.json()
                    if 'serverTime' not in data:
                        raise ValueError("返回数据中缺少 serverTime 字段")
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    self._time_diff = server_time - local_time
                    logger.info(f"时间同步成功，时间差: {self._time_diff}ms")
                    return
            except Exception as e:
                logger.error(f"时间同步失败(重试 {retry + 1}): {str(e)}")
                await asyncio.sleep(2 ** retry)
        logger.warning("无法同步服务器时间，回退使用本地时间")
        self._time_diff = 0

    async def _signed_request(self, method: str, path: str, params: dict) -> dict:
        params.update({
            "timestamp": int(time.time() * 1000 + self._time_diff),
            "recvWindow": self.recv_window
        })
        sorted_params = sorted(params.items())
        query = urllib.parse.urlencode(sorted_params, doseq=True)
        signature = hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = REST_URL + "/fapi/v1" + path + "?" + query + "&signature=" + signature
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
                        raise Exception(f"HTTP Status {resp.status}. Response: {error_text}")
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" not in content_type:
                        text = await resp.text()
                        raise Exception(f"Unexpected Content-Type: {content_type}. Response text: {text}")
                    return await resp.json()
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}): {type(e).__name__} {str(e)}")
                if attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise Exception("超过最大重试次数")

    async def _rate_limit_check(self, endpoint: str):
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]
        now = time.monotonic()
        while dq and dq[0] < now - period:
            dq.popleft()
        if len(dq) >= limit:
            wait_time = max(dq[0] + period - now + np.random.uniform(0, 0.05), 0)
            logger.warning(f"速率限制触发: {endpoint} 等待 {wait_time:.3f}s")
            METRICS['throughput'].set(0)
            await asyncio.sleep(wait_time)
        dq.append(now)
        METRICS['throughput'].inc()

    async def manage_leverage(self):
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE, 'dualSidePosition': self.config.dual_side_position}
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self, interval: str, limit: int = 100) -> pd.DataFrame:
        try:
            data = await self._signed_request('GET', '/klines', {
                'symbol': SYMBOL,
                'interval': interval,
                'limit': limit,
                'contractType': 'PERPETUAL'
            })
            if not isinstance(data, list):
                raise ValueError("K线数据格式异常")
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"K线获取失败: {str(e)}")
            return pd.DataFrame()

    async def fetch_weekly_data(self, limit: int = 100) -> pd.DataFrame:
        try:
            data = await self._signed_request('GET', '/klines', {
                'symbol': SYMBOL,
                'interval': '1w',
                'limit': limit,
                'contractType': 'PERPETUAL'
            })
            if not isinstance(data, list):
                raise ValueError("K线数据格式异常")
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"周级K线获取失败: {str(e)}")
            return pd.DataFrame()


@dataclass
class Signal:
    action: bool
    side: str
    tp: float
    sl: float
    order_details: dict = None


class ETHUSDCStrategy:
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()
        self._indicator_cache = defaultdict(lambda: None)
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_side = None
        self.entry_price = None
        self.hard_stop = None
        self.trailing_stop = None
        self.atr_baseline = None
        self.tp_triggered_30 = False  # 首次TP 30%是否触发
        self.tp_triggered_20 = False  # 首次TP 20%是否触发

    async def execute(self):
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        await asyncio.gather(
            self.trend_monitoring_loop(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.risk_control_loop()
        )

    async def trend_monitoring_loop(self):
        while True:
            try:
                trend = await self.analyze_trend_15m()
                logger.info(f"15分钟趋势判断：{trend}")
                if trend in ['LONG', 'SHORT']:
                    await self.adjust_position_ratio(trend)
                if self.last_trade_side and trend != self.last_trade_side:
                    await self.handle_trend_reversal(new_trend=trend)
                self.last_trade_side = trend
            except Exception as e:
                logger.error(f"趋势监控异常: {str(e)}")
            await asyncio.sleep(60)

    async def order_signal_loop(self):
        while True:
            try:
                signal, order_list = await self.analyze_order_signals_3m()
                if signal.action:
                    if self.last_trade_side == 'LONG':
                        long_qty = QUANTITY * 1
                        short_qty = QUANTITY * 0.5
                    elif self.last_trade_side == 'SHORT':
                        long_qty = QUANTITY * 0.5
                        short_qty = QUANTITY * 1
                    else:
                        long_qty = short_qty = QUANTITY
                    await self.place_dynamic_limit_orders(signal.side, order_list, long_qty, short_qty)
                else:
                    # 没有下单信号时撤销未成交订单
                    await self.cancel_all_orders()
            except Exception as e:
                logger.error(f"下单信号异常: {str(e)}")
            await asyncio.sleep(self.config.order_adjust_interval)

    async def stop_loss_profit_management_loop(self):
        while True:
            try:
                await self.update_dynamic_stop_loss()
                await self.manage_profit_targets()
            except Exception as e:
                logger.error(f"止盈止损管理异常: {str(e)}")
            await asyncio.sleep(3)

    async def risk_control_loop(self):
        while True:
            try:
                await self.monitor_risk()
            except Exception as e:
                logger.error(f"风险监控异常: {str(e)}")
            await asyncio.sleep(10)

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
        final_upper = []
        for i, price in enumerate(close):
            if i == 0:
                final_upper.append(basic_upper.iloc[i])
            else:
                if price > final_upper[-1]:
                    final_upper.append(basic_upper.iloc[i])
                else:
                    final_upper.append(final_upper[-1])
        last_final_upper = final_upper[-1]
        if close[-1] > last_final_upper:
            st_trend = 'LONG'
        elif close[-1] < basic_lower.iloc[-1]:
            st_trend = 'SHORT'
        else:
            ema_fast = pd.Series(close).ewm(span=self.config.macd_fast, adjust=False).mean().values
            ema_slow = pd.Series(close).ewm(span=self.config.macd_slow, adjust=False).mean().values
            macd_line = ema_fast - ema_slow
            st_trend = 'LONG' if macd_line[-1] >= 0 else 'SHORT'
        return st_trend

    def _vectorized_atr(self, high, low, close, period: int) -> np.ndarray:
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        ])
        atr = np.convolve(tr, np.ones(period), 'valid') / period
        return atr

    async def adjust_position_ratio(self, trend: str):
        df = await self.client.fetch_klines(interval='15m', limit=50)
        if df.empty:
            return
        atr_val = \
        self._vectorized_atr(df['high'].values, df['low'].values, df['close'].values, period=self.config.atr_window)[-1]
        recent_atr = np.mean(self._vectorized_atr(df['high'].values, df['low'].values, df['close'].values,
                                                  period=self.config.atr_window))
        if atr_val < 0.3 * recent_atr:
            ratio = (1, 0.3)
        else:
            ratio = (1, 0.5)
        logger.info(f"调整仓位比例: 多仓 {ratio[0]}, 空仓 {ratio[1]}")

    async def handle_trend_reversal(self, new_trend: str):
        logger.info(f"触发趋势反转，新趋势：{new_trend}")
        try:
            if new_trend == 'LONG':
                await self.close_position(side='SELL', ratio=0.5)
            elif new_trend == 'SHORT':
                await self.close_position(side='BUY', ratio=0.5)
            asyncio.create_task(self.gradual_position_adjustment())
        except Exception as e:
            logger.error(f"趋势反转平仓异常：{str(e)}")

    async def gradual_position_adjustment(self):
        for _ in range(12):
            await self.rebalance_hedge()
            await asyncio.sleep(300)

    async def rebalance_hedge(self):
        logger.info("执行持仓再平衡，调整对冲仓位至1:1")
        # 实际实现时需要查询持仓情况，并下单对冲

    async def close_position(self, side: str, ratio: float):
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return
        current_price = df['close'].values[-1]
        if side == 'BUY':
            price_limit = current_price * (1 + self.config.max_slippage_market)
        else:
            price_limit = current_price * (1 - self.config.max_slippage_market)
        logger.info(f"平仓 {side}，比例：{ratio}，价格限制：{price_limit:.2f}")
        params = {'symbol': SYMBOL, 'side': side, 'type': 'MARKET', 'quantity': QUANTITY * ratio}
        if not params['quantity'] or params['quantity'] <= 0:
            logger.error("无效订单数量，跳过平仓")
            return
        if not price_limit or price_limit <= 0:
            logger.error("无效订单价格，跳过平仓")
            return
        await self.client._signed_request('POST', '/order', params)

    async def analyze_order_signals_3m(self):
        """
        使用实时3分钟 Bollinger Bands %B（周期20, 标准差2）判断下单信号：
          - 当 %B 处于极端状态（≥1 或 ≤0）时触发下单；
          - 补充条件：当前持仓为 LONG 时，当 %B 从极端值回落至约0.95以下，
            或 SHORT 时当 %B 从极端值上升至约0.05以上，也触发信号；
          - 返回订单参数列表，每档订单基于最新价格偏移固定比例。
        """
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
        logger.info(f"实时3m %B: {percent_b:.3f}")

        order_list = []
        signal_trigger = False
        signal_side = 'NONE'

        if percent_b >= 1.0:
            signal_side = 'SELL'
            signal_trigger = True
        elif percent_b <= 0.0:
            signal_side = 'BUY'
            signal_trigger = True
        elif self.last_trade_side == 'LONG' and percent_b < 0.95:
            signal_side = 'SELL'
            signal_trigger = True
        elif self.last_trade_side == 'SHORT' and percent_b > 0.05:
            signal_side = 'BUY'
            signal_trigger = True

        if signal_trigger:
            order_list = [
                {'offset': 0.0025, 'ratio': 0.30},
                {'offset': 0.0040, 'ratio': 0.20},
                {'offset': 0.0060, 'ratio': 0.10},
                {'offset': 0.0080, 'ratio': 0.20},
                {'offset': 0.0160, 'ratio': 0.20},
            ]
            logger.info(f"触发下单信号: side={signal_side}, %B={percent_b:.3f}")
            return Signal(True, signal_side, 0, 0, order_details={'percent_b': percent_b}), order_list
        else:
            return Signal(False, 'NONE', 0, 0), []

    async def place_dynamic_limit_orders(self, side: str, order_list: list, long_qty: float, short_qty: float):
        """
        基于最新3m数据及预设档位偏移挂单，下单价格依据最新3m均价计算。
        """
        df_now = await self.client.fetch_klines(interval='3m', limit=50)
        if df_now.empty:
            return
        current_price = df_now['close'].values[-1]
        for order in order_list:
            dynamic_offset = order['offset']
            ratio = order['ratio']
            order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
            if not order_qty or order_qty <= 0:
                logger.error("计算得到无效订单数量，跳过当前档")
                continue
            if side == 'BUY':
                limit_price = round(current_price * (1 - dynamic_offset), self.config.price_precision)
            else:
                limit_price = round(current_price * (1 + dynamic_offset), self.config.price_precision)
            if not limit_price or limit_price <= 0:
                logger.error("计算得到无效挂单价格，跳过当前档")
                continue
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': limit_price,
                'quantity': order_qty,
                'timeInForce': 'GTC'
            }
            try:
                await self.client._signed_request('POST', '/order', params)
                logger.info(
                    f"挂单成功 {side}@{limit_price}，数量: {order_qty}（偏移 {dynamic_offset * 100:.2f}%, ratio {ratio}）")
            except Exception as e:
                logger.error(f"挂单失败: {str(e)}")
        asyncio.create_task(self.adjust_pending_orders(side))

    async def adjust_pending_orders(self, side: str):
        await asyncio.sleep(180)
        current_b = await self.get_current_percentb()
        if (side == 'BUY' and current_b > 0.3) or (side == 'SELL' and current_b < 0.7):
            await self.cancel_all_orders()
            logger.info(f"根据%B={current_b:.3f}条件撤销未成交订单")

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
        logger.info(f"实时3m %B = {percent_b:.3f}")
        return percent_b

    async def cancel_all_orders(self):
        logger.info("撤销所有未成交订单")
        params = {'symbol': SYMBOL}
        try:
            await self.client._signed_request('DELETE', '/openOrders', params)
        except Exception as e:
            logger.error(f"撤单异常: {str(e)}")

    async def update_dynamic_stop_loss(self):
        """
        根据最新3m K线数据更新动态止损：
          - 以最新3m价格为基准；
          - 计算 Bollinger Bands 带宽（周期20，标准差2）；
          - 对于 LONG 仓位：止损价 = 最新价 - (带宽×系数)，仅当新止损高于旧值时更新；
          - 对于 SHORT 仓位：止损价 = 最新价 + (带宽×系数)。
        """
        df_bb = await self.client.fetch_klines(interval='3m', limit=50)
        if df_bb.empty or self.last_trade_side is None:
            return
        latest_price = df_bb['close'].values[-1]
        close_prices = df_bb['close'].values.astype(np.float64)
        sma = pd.Series(close_prices).rolling(window=self.config.bb_period).mean().values
        std = pd.Series(close_prices).rolling(window=self.config.bb_period).std().values
        upper_band = sma + self.config.bb_std * std
        lower_band = sma - self.config.bb_std * std
        bb_width = upper_band[-1] - lower_band[-1]
        coeff = 0.5
        dynamic_offset = bb_width * coeff

        if self.last_trade_side == 'LONG':
            new_stop = latest_price - dynamic_offset
            if self.hard_stop is None or new_stop > self.hard_stop:
                self.hard_stop = new_stop
            self.trailing_stop = latest_price - dynamic_offset
            logger.info(
                f"LONG 动态止损更新：当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")
        elif self.last_trade_side == 'SHORT':
            new_stop = latest_price + dynamic_offset
            if self.hard_stop is None or new_stop < self.hard_stop:
                self.hard_stop = new_stop
            self.trailing_stop = latest_price + dynamic_offset
            logger.info(
                f"SHORT 动态止损更新：当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")

    async def manage_profit_targets(self):
        """
        管理止盈逻辑：
          - 利用最新3m数据和 %B 值判断是否首次触发止盈信号；
          - 同时挂分档止盈订单；
          - 检测是否突破动态止损水平，从而触发市价全平。
        """
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.entry_price is None:
            return
        current_price = df['close'].values[-1]
        current_b = await self.get_current_percentb()

        if self.last_trade_side == 'LONG':
            if current_b >= 0.82 and not self.tp_triggered_30:
                logger.info("触发首次止盈信号：LONG，市价平仓30%")
                await self.close_position(side='SELL', ratio=0.3)
                self.tp_triggered_30 = True
            elif current_b >= 1 and not self.tp_triggered_20:
                logger.info("触发首次止盈信号：LONG，市价平仓20%")
                await self.close_position(side='SELL', ratio=0.2)
                self.tp_triggered_20 = True
            await self.place_take_profit_orders(side='SELL')
        elif self.last_trade_side == 'SHORT':
            if current_b <= 0.18 and not self.tp_triggered_30:
                logger.info("触发首次止盈信号：SHORT，市价平仓30%")
                await self.close_position(side='BUY', ratio=0.3)
                self.tp_triggered_30 = True
            elif current_b <= 0 and not self.tp_triggered_20:
                logger.info("触发首次止盈信号：SHORT，市价平仓20%")
                await self.close_position(side='BUY', ratio=0.2)
                self.tp_triggered_20 = True
            await self.place_take_profit_orders(side='BUY')

        if self.last_trade_side == 'LONG' and current_price < self.hard_stop:
            logger.info("LONG 价格低于动态止损，触发平仓")
            await self.close_position(side='SELL', ratio=1.0)
        elif self.last_trade_side == 'SHORT' and current_price > self.hard_stop:
            logger.info("SHORT 价格高于动态止损，触发平仓")
            await self.close_position(side='BUY', ratio=1.0)

    async def place_take_profit_orders(self, side: str):
        """
        针对剩余仓位，基于最新3m数据，分别按固定偏移挂止盈单，
        每个订单数量为剩余仓位的平均分配。
        """
        df_now = await self.client.fetch_klines(interval='3m', limit=50)
        if df_now.empty:
            return
        current_price = df_now['close'].values[-1]
        tp_offsets = [0.0025, 0.004, 0.006, 0.008, 0.012]
        remaining_qty = QUANTITY
        qty_each = round(remaining_qty / len(tp_offsets), self.config.quantity_precision)
        for offset in tp_offsets:
            if side == 'SELL':  # LONG仓位止盈挂卖单
                tp_price = round(current_price * (1 + offset), self.config.price_precision)
            else:  # SHORT仓位止盈挂买单
                tp_price = round(current_price * (1 - offset), self.config.price_precision)
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': tp_price,
                'quantity': qty_each,
                'timeInForce': 'GTC'
            }
            try:
                await self.client._signed_request('POST', '/order', params)
                logger.info(f"止盈挂单成功 {side}@{tp_price}，数量: {qty_each}（TP offset {offset * 100:.2f}%)")
            except Exception as e:
                logger.error(f"止盈挂单失败: {str(e)}")

    async def monitor_risk(self):
        if self.daily_pnl < -self.config.daily_drawdown_limit:
            logger.warning("日内亏损超20%，进入只平仓模式")
            await self.close_position(side='SELL' if self.last_trade_side == 'LONG' else 'BUY', ratio=1.0)
        if self.consecutive_losses >= 3:
            logger.warning("连续亏损3次，降低仓位至50%")
            # 根据实际情况调整仓位

    async def compute_atr_baseline(self):
        if self.atr_baseline is None:
            df_weekly = await self.client.fetch_weekly_data(limit=50)
            if not df_weekly.empty:
                high = df_weekly['high'].values.astype(np.float64)
                low = df_weekly['low'].values.astype(np.float64)
                close = df_weekly['close'].values.astype(np.float64)
                atr = self._vectorized_atr(high, low, close, period=self.config.atr_window)
                self.atr_baseline = atr[-1]
                logger.info(f"重置周ATR基准：{self.atr_baseline:.4f}")

    async def on_order(self, is_buy: bool, price: float):
        self.entry_price = price
        if is_buy:
            self.hard_stop = price * 0.98
            self.last_trade_side = 'LONG'
        else:
            self.hard_stop = price * 1.02
            self.last_trade_side = 'SHORT'
        self.tp_triggered_30 = False
        self.tp_triggered_20 = False
        logger.info(f"记录开仓：{price:.2f}, 初始硬止损：{self.hard_stop:.2f}")


async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，优雅退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")


if __name__ == "__main__":
    asyncio.run(main())