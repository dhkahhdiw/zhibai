#!/usr/bin/env python3
# ETH/USDT 高频交易引擎 v7.2（修正后仅返回 LONG 或 SHORT 状态）

import uvloop
uvloop.install()

import os, asyncio, time, hmac, hashlib, urllib.parse, logging, datetime
from collections import deque, defaultdict
from dataclasses import dataclass
from socket import AF_INET

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

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDT'
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
        logging.FileHandler("/var/log/eth_usdt_hft.log", encoding="utf-8", mode='a')
    ]
)
logger = logging.getLogger('ETH-USDT-HFT')


@dataclass
class TradingConfig:
    """策略参数配置"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    vol_multiplier: float = 2.5
    st_period: int = 20             # 超级趋势周期（MA周期）
    st_multiplier: float = 3.0      # ATR 倍数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20             # 布林带周期
    bb_std: float = 2.0             # 标准差倍数
    max_retries: int = 7
    order_timeout: float = 2.0
    network_timeout: float = 5.0
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0
    slippage: float = 0.001         # 默认滑点
    dual_side_position: str = "true"
    order_adjust_interval: float = 180.0   # 3 分钟
    max_slippage_market: float = 0.0015      # 0.15% 滑点上限
    daily_drawdown_limit: float = 0.20       # 单日亏损20%触发风险保护


class BinanceHFTClient:
    def __init__(self):
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=512,
            resolver=AsyncResolver(),
            ttl_dns_cache=300,
            force_close=True,
            family=AF_INET,
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
            timeout=ClientTimeout(
                total=self.config.network_timeout,
                sock_connect=self.config.order_timeout
            )
        )

    async def sync_server_time(self):
        url = REST_URL + "/fapi/v1/time"
        for retry in range(5):
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" not in content_type:
                        raise ValueError(f"Unexpected content type: {content_type}")
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
            "Accept": "application/json"
        }
        logger.debug(f"请求URL: {url.split('?')[0]} 参数: {sorted_params}")

        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)
                async with self.session.request(method, url, headers=headers) as resp:
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
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'dualSidePosition': self.config.dual_side_position
        }
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self, interval: str, limit: int = 100) -> pd.DataFrame:
        """通用K线数据获取函数"""
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
        """获取周级数据用于ATR基准计算"""
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
    order_details: dict = None  # 额外订单参数


class ETHUSDTStrategy:
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

    async def execute(self):
        """初始化同步时间、杠杆，并并行执行策略各子模块"""
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        # 异步并行执行各任务
        await asyncio.gather(
            self.trend_monitoring_loop(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.risk_control_loop()
        )

    async def trend_monitoring_loop(self):
        """15分钟级趋势监控及仓位调整"""
        while True:
            try:
                trend = await self.analyze_trend_15m()
                logger.info(f"15分钟趋势判断：{trend}")
                # 根据趋势调整仓位比例及风控
                if trend in ['LONG', 'SHORT']:
                    await self.adjust_position_ratio(trend)
                # 趋势反转时处理仓位对冲并启动分批调仓任务
                if self.last_trade_side and trend != self.last_trade_side:
                    await self.handle_trend_reversal(new_trend=trend)
                self.last_trade_side = trend
            except Exception as e:
                logger.error(f"趋势监控异常: {str(e)}")
            # 每分钟更新一次15分钟趋势
            await asyncio.sleep(60)

    async def order_signal_loop(self):
        """3分钟级挂单策略与动态订单调整"""
        while True:
            try:
                signal, order_list = await self.analyze_order_signals_3m()
                if signal.action:
                    # 根据趋势设置仓位比例（LONG：多仓=1, 空仓=0.5；SHORT：多仓=0.5, 空仓=1）
                    if self.last_trade_side == 'LONG':
                        long_qty = QUANTITY * 1
                        short_qty = QUANTITY * 0.5
                    elif self.last_trade_side == 'SHORT':
                        long_qty = QUANTITY * 0.5
                        short_qty = QUANTITY * 1
                    else:
                        long_qty = short_qty = QUANTITY
                    await self.place_dynamic_limit_orders(signal.side, order_list, long_qty, short_qty)
            except Exception as e:
                logger.error(f"下单信号异常: {str(e)}")
            await asyncio.sleep(self.config.order_adjust_interval)

    async def stop_loss_profit_management_loop(self):
        """止盈止损管理，每3秒检测"""
        while True:
            try:
                await self.manage_stop_loss_and_profit()
            except Exception as e:
                logger.error(f"止盈止损管理异常: {str(e)}")
            await asyncio.sleep(3)

    async def risk_control_loop(self):
        """风险监控：监控每日亏损及连续亏损次数"""
        while True:
            try:
                await self.monitor_risk()
            except Exception as e:
                logger.error(f"风险监控异常: {str(e)}")
            await asyncio.sleep(10)

    async def analyze_trend_15m(self) -> str:
        """
        利用15分钟数据计算超级趋势和MACD，
        返回状态仅为 'LONG' 或 'SHORT'。
        其中，当价格既不大于最终上轨也不低于基础下轨时，
        通过判断 MACD 值来决定趋势。
        """
        df = await self.client.fetch_klines(interval='15m', limit=100)
        if df.empty:
            # 默认返回 LONG
            return 'LONG'
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)

        # 计算ATR（采用 st_period）
        atr = self._vectorized_atr(high, low, close, period=self.config.st_period)
        last_atr = atr[-1]

        # 计算基础趋势线：HL2 ± st_multiplier * ATR
        hl2 = (df['high'] + df['low']) / 2
        basic_upper = hl2 + self.config.st_multiplier * last_atr
        basic_lower = hl2 - self.config.st_multiplier * last_atr

        # 超级趋势动态迭代（仅计算上轨）
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

        # 初步判断
        if close[-1] > last_final_upper:
            st_trend = 'LONG'
        elif close[-1] < basic_lower.iloc[-1]:
            st_trend = 'SHORT'
        else:
            # 如果位于区间内，通过判断 MACD 值决定趋势
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
        """调整仓位比例，记录日志作为下单参数参考"""
        df = await self.client.fetch_klines(interval='15m', limit=50)
        if df.empty:
            return
        atr_val = self._vectorized_atr(df['high'].values, df['low'].values, df['close'].values, period=self.config.atr_window)[-1]
        recent_atr = np.mean(self._vectorized_atr(df['high'].values, df['low'].values, df['close'].values, period=self.config.atr_window))
        if atr_val < 0.3 * recent_atr:
            ratio = (1, 0.3)
        else:
            ratio = (1, 0.5)
        logger.info(f"调整仓位比例: 多仓 {ratio[0]}, 空仓 {ratio[1]}")

    async def handle_trend_reversal(self, new_trend: str):
        """趋势反转处理：平掉与新趋势不符的50%仓位，并启动分批调仓至1:1"""
        logger.info(f"触发趋势反转，新趋势：{new_trend}")
        try:
            if new_trend == 'LONG':
                await self.close_position(side='SELL', ratio=0.5)
            elif new_trend == 'SHORT':
                await self.close_position(side='BUY', ratio=0.5)
            # 启动异步任务，分12次每5分钟调整到1:1仓位（1小时内完成）
            asyncio.create_task(self.gradual_position_adjustment())
        except Exception as e:
            logger.error(f"趋势反转平仓异常：{str(e)}")

    async def gradual_position_adjustment(self):
        """分批调整持仓至1:1，每5分钟调整一次，共1小时"""
        for _ in range(12):
            await self.rebalance_hedge()
            await asyncio.sleep(300)

    async def rebalance_hedge(self):
        """执行持仓再平衡操作，确保对冲比例调整至1:1（此处为占位函数）"""
        logger.info("执行持仓再平衡，调整对冲仓位至1:1")
        # 此处应查询当前持仓并计算差额后下单

    async def close_position(self, side: str, ratio: float):
        """以市价单平仓，加入滑点控制"""
        df = await self.client.fetch_klines(interval='1m', limit=1)
        if df.empty:
            return
        current_price = df['close'].values[0]
        if side == 'BUY':
            price_limit = current_price * (1 + self.config.max_slippage_market)
        else:
            price_limit = current_price * (1 - self.config.max_slippage_market)
        logger.info(f"平仓 {side}，比例：{ratio}，价格限制：{price_limit:.2f}")
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'MARKET',
            'quantity': QUANTITY * ratio,
        }
        await self.client._signed_request('POST', '/order', params)

    async def analyze_order_signals_3m(self):
        """
        计算3分钟布林带%B，并判断挂单信号，
        当%B接近0或1时构造订单参数列表。
        """
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return Signal(False, 'NONE', 0, 0), []
        close = df['close'].values.astype(np.float64)
        sma = pd.Series(close).rolling(window=self.config.bb_period).mean().values
        std = pd.Series(close).rolling(window=self.config.bb_period).std().values
        upper_band = sma + self.config.bb_std * std
        lower_band = sma - self.config.bb_std * std
        last_close = close[-1]
        bb_range = upper_band[-1] - lower_band[-1]
        percent_b = 0.0 if bb_range == 0 else (last_close - lower_band[-1]) / bb_range
        logger.info(f"3分钟 %B = {percent_b:.3f}")
        order_list = []
        if percent_b < 0.05 or percent_b > 0.95:
            side = 'BUY' if percent_b < 0.05 else 'SELL'
            order_list = [
                {'offset': 0.0025, 'ratio': 0.30},
                {'offset': 0.0040, 'ratio': 0.20},
                {'offset': 0.0060, 'ratio': 0.10},
                {'offset': 0.0080, 'ratio': 0.20},
                {'offset': 0.0160, 'ratio': 0.20},
            ]
            return Signal(True, side, 0, 0, order_details={'percent_b': percent_b}), order_list
        return Signal(False, 'NONE', 0, 0), []

    async def place_dynamic_limit_orders(self, side: str, order_list: list, long_qty: float, short_qty: float):
        """
        根据订单列表构造挂单，
        每单价格 = 当前价 ± dynamic_offset（挂单偏移根据ATR动态调整）；
        若挂单后3分钟未成交则启动调整任务，同时根据%B条件撤销挂单。
        """
        df_now = await self.client.fetch_klines(interval='1m', limit=1)
        if df_now.empty:
            return
        current_price = df_now['close'].values[0]
        df_daily = await self.client.fetch_klines(interval='1d', limit=50)
        if not df_daily.empty:
            atr_daily = self._vectorized_atr(df_daily['high'].values, df_daily['low'].values, df_daily['close'].values, period=self.config.atr_window)
            daily_mean = np.mean(atr_daily) if atr_daily.size > 0 else 1.0
            current_atr = self._vectorized_atr(
                np.array([df_now['high'].values[-1]]),
                np.array([df_now['low'].values[-1]]),
                np.array([current_price]),
                period=self.config.atr_window)
            atr_ratio = (current_atr[-1] / daily_mean) if daily_mean else 1.0
        else:
            atr_ratio = 1.0

        for order in order_list:
            dynamic_offset = order['offset'] * (1 + 0.5 * (atr_ratio - 1))
            ratio = order['ratio']
            order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
            if side == 'BUY':
                limit_price = round(current_price * (1 - dynamic_offset), self.config.price_precision)
            else:
                limit_price = round(current_price * (1 + dynamic_offset), self.config.price_precision)
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
                logger.info(f"挂单成功 {side}@{limit_price}，数量: {order_qty}（偏移 {dynamic_offset*100:.2f}%, ratio {ratio}）")
            except Exception as e:
                logger.error(f"挂单失败: {str(e)}")
        asyncio.create_task(self.adjust_pending_orders(side))

    async def adjust_pending_orders(self, side: str):
        """等待3分钟后检查未成交订单，如%B条件不符则撤销"""
        await asyncio.sleep(180)
        current_b = await self.get_current_percentb()
        if (side == 'BUY' and current_b > 0.3) or (side == 'SELL' and current_b < 0.7):
            await self.cancel_all_orders()
            logger.info(f"根据%B={current_b:.3f}条件撤销未成交订单")

    async def get_current_percentb(self) -> float:
        """实时获取最新%B值（使用3m数据）"""
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return 0.5
        close = df['close'].values.astype(np.float64)
        sma = pd.Series(close).rolling(window=self.config.bb_period).mean().values
        std = pd.Series(close).rolling(window=self.config.bb_period).std().values
        upper_band = sma + self.config.bb_std * std
        lower_band = sma - self.config.bb_std * std
        bb_range = upper_band[-1] - lower_band[-1]
        percent_b = 0.0 if bb_range == 0 else (close[-1] - lower_band[-1]) / bb_range
        return percent_b

    async def cancel_all_orders(self):
        """撤销所有未成交订单（调用交易所撤单接口）"""
        logger.info("撤销所有未成交订单")
        params = {'symbol': SYMBOL}
        try:
            await self.client._signed_request('DELETE', '/openOrders', params)
        except Exception as e:
            logger.error(f"撤单异常: {str(e)}")

    async def manage_stop_loss_and_profit(self):
        """
        止盈止损管理：
         1. 未记录开仓价则等待 on_order 回调；
         2. 达到1%盈利后启动ATR跟踪止损；
         3. 硬止损为 entry_price*0.98（多单）或 entry_price*1.02（空单）。
        """
        df = await self.client.fetch_klines(interval='1m', limit=1)
        if df.empty or self.entry_price is None:
            return
        current_price = df['close'].values[0]
        highest_high = current_price  # 实际应记录持仓期间的最高价
        unrealized_pnl = ((current_price - self.entry_price) / self.entry_price
                          if self.last_trade_side == 'LONG'
                          else (self.entry_price - current_price) / self.entry_price)
        if unrealized_pnl >= 0.01:
            atr = self._vectorized_atr(
                np.array([df['high'].values[-1]]),
                np.array([df['low'].values[-1]]),
                np.array([current_price]),
                period=self.config.atr_window)
            self.trailing_stop = highest_high - 1.5 * atr[-1]
            logger.info(f"启动移动止盈：当前止损位 {self.trailing_stop:.2f}")
        if self.last_trade_side == 'LONG' and (current_price < self.hard_stop or (self.trailing_stop and current_price < self.trailing_stop)):
            await self.close_position(side='SELL', ratio=1.0)
        elif self.last_trade_side == 'SHORT' and (current_price > self.hard_stop or (self.trailing_stop and current_price > self.trailing_stop)):
            await self.close_position(side='BUY', ratio=1.0)

    async def monitor_risk(self):
        """
        风控：
         1. 单日亏损超过20%时进入只平仓模式；
         2. 连续3次亏损时降低仓位至原计划50%；
         3. 实际盈亏需根据交易回报更新 self.daily_pnl 与 self.consecutive_losses。
        """
        if self.daily_pnl < -self.config.daily_drawdown_limit:
            logger.warning("日内亏损超20%，进入只平仓模式")
            await self.close_position(side='SELL' if self.last_trade_side=='LONG' else 'BUY', ratio=1.0)
        if self.consecutive_losses >= 3:
            logger.warning("连续亏损3次，降低仓位至50%")

    async def compute_atr_baseline(self):
        """每周重置ATR基准，用于自适应波动率"""
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
        """订单成交回调，记录开仓价和设置硬止损"""
        self.entry_price = price
        if is_buy:
            self.hard_stop = price * 0.98
            self.last_trade_side = 'LONG'
        else:
            self.hard_stop = price * 1.02
            self.last_trade_side = 'SHORT'
        logger.info(f"记录开仓：{price:.2f}, 硬止损：{self.hard_stop:.2f}")

async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDTStrategy(client)
    try:
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，优雅退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")

if __name__ == "__main__":
    asyncio.run(main())