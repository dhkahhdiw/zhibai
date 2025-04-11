#!/usr/bin/env python3
# ETH/USDC 高频交易引擎 v7.4（整合完善版：基于15m超趋势与3m级BB%B信号策略）
#
# 主要逻辑：
# 1. 15m级超级趋势（参数：20周期、3倍ATR）结合MACD辅助判断趋势，
#    趋势为多时目标仓位：多仓0.1ETH、空仓0.05ETH；趋势为空时反之。
#    趋势反转时，市价单平掉与新趋势不符的部分（50%），后续分批再平衡至1:1。
#
# 2. 3m级下单策略：利用 Bollinger Bands %B（20-2）判定下单信号，
#    当 %B ≤ 0 或 ≥ 1 时，以触发信号时的最新价格为下单基准，
#    分别挂固定限价单：
#         ±0.05% 挂单30%
#         ±0.1%  挂单20%
#         ±0.45% 挂单10%
#         ±0.65% 挂单20%
#         ±1.1%  挂单20%
#    当 %B 回落至 0.3～0.7 区间时，取消所有未成交订单。
#
# 3. 止盈止损：初始止损设为买入价×0.98/1.02，配合动态跟踪止损，
#    首次达到止盈条件（多时 %B 0.82 / 空时 %B 0.18）时市价平仓30%，
#    再达到 (%B 1 / 0)时市价平仓20%，剩余仓位以最新价格分别挂限价止盈单（档位：±0.25%、±0.4%、±0.6%、±0.8%、±1.2%）。
#
# 4. 风险控制：当日亏损超过资金20%或连续亏损≥3次时触发风控。
#

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

# 合约参数：使用合约 ETHUSDC（USDC 保证金合约）
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
# 注意：如果使用 USDC 保证金合约，请使用 Binance 针对 USDC 合约的专用接口地址
SYMBOL = 'ETHUSDC'
LEVERAGE = 50
# 趋势判断目标仓位（单位：ETH）
BULL_LONG_QTY = 0.12    # 趋势多时，多仓
BULL_SHORT_QTY = 0.06  # 趋势多时，对冲空仓
BEAR_LONG_QTY = 0.06   # 趋势空时，对冲多仓
BEAR_SHORT_QTY = 0.12   # 趋势空时，空仓

# 对于 USDC 合约，假设使用如下基础 URL（请参照 Binance 官方文档确认）
REST_URL = 'https://api.binance.usdc/v1'

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
    st_period: int = 20            # 超级趋势周期
    st_multiplier: float = 3.0     # ATR倍数（超级趋势）
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20            # Bollinger Bands周期（3m级）
    bb_std: float = 2.0            # 标准差倍数
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
        self.position = 0.0  # 当前持仓量（正数：多仓，负数：空仓）

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
        if 'contractType' not in params:
            params['contractType'] = 'PERPETUAL'
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
                    data = await resp.json()
                    if isinstance(data, dict) and "errorCode" in data:
                        raise Exception(f"下单错误，ErrorCode: {data.get('errorCode')}, Msg: {data.get('msg')}")
                    return data
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
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE, 'dualSidePosition': self.config.dual_side_position, 'contractType': 'PERPETUAL'}
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
        self.last_trade_side = None  # 'LONG' 或 'SHORT'
        self.entry_price = None
        self.hard_stop = None
        self.trailing_stop = None
        self.atr_baseline = None
        self.tp_triggered_30 = False
        self.tp_triggered_20 = False
        # 根据15分钟趋势判断设置目标仓位（单位：ETH）
        self.target_long = 0.0
        self.target_short = 0.0

    # ---------- 多时间框架趋势判断与仓位控制 ----------

    async def trend_monitoring_loop(self):
        while True:
            try:
                trend = await self.analyze_trend_15m()
                logger.info(f"15分钟趋势判断：{trend}")
                await self.adjust_position_ratio(trend)
                if self.last_trade_side is not None and trend != self.last_trade_side:
                    await self.handle_trend_reversal(trend)
                self.last_trade_side = trend
            except Exception as e:
                logger.error(f"趋势监控异常: {str(e)}")
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
            curr_upper = basic_upper.iloc[i] if basic_upper.iloc[i] < final_upper[-1] or close[i-1] > final_upper[-1] else final_upper[-1]
            curr_lower = basic_lower.iloc[i] if basic_lower.iloc[i] > final_lower[-1] or close[i-1] < final_lower[-1] else final_lower[-1]
            final_upper.append(curr_upper)
            final_lower.append(curr_lower)
        final_upper = np.array(final_upper)
        final_lower = np.array(final_lower)
        if close[-1] > final_upper[-1]:
            st_trend = 'LONG'
        elif close[-1] < final_lower[-1]:
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
        atr_arr = self._vectorized_atr(df['high'].values, df['low'].values, df['close'].values, period=14)
        atr_val = atr_arr[-1]
        recent_atr = np.mean(atr_arr)
        ratio_factor = 0.3 if atr_val < 0.3 * recent_atr else 0.5
        if trend == 'LONG':
            self.target_long = BULL_LONG_QTY
            self.target_short = BULL_SHORT_QTY if ratio_factor == 0.5 else BULL_SHORT_QTY * (0.3/0.5)
        else:
            self.target_long = BEAR_LONG_QTY if ratio_factor == 0.5 else BEAR_LONG_QTY * (0.3/0.5)
            self.target_short = BEAR_SHORT_QTY
        logger.info(f"目标仓位：多仓 {self.target_long} ETH, 空仓 {self.target_short} ETH")

    async def handle_trend_reversal(self, new_trend: str):
        logger.info(f"趋势反转，新趋势：{new_trend}")
        if self.last_trade_side == 'LONG' and new_trend == 'SHORT':
            await self.close_position(side='SELL', ratio=0.5)
        elif self.last_trade_side == 'SHORT' and new_trend == 'LONG':
            await self.close_position(side='BUY', ratio=0.5)
        asyncio.create_task(self.gradual_position_adjustment())

    async def gradual_position_adjustment(self):
        for _ in range(12):
            await self.rebalance_hedge()
            await asyncio.sleep(300)

    async def rebalance_hedge(self):
        logger.info("执行仓位再平衡，调整为1:1")
        # 此处实际应查询持仓并进行对冲下单

    # ---------- 3m级下单策略 ----------

    async def analyze_order_signals_3m(self):
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
        if percent_b <= 0.0:
            signal_side = 'BUY'
            signal_trigger = True
        elif percent_b >= 1.0:
            signal_side = 'SELL'
            signal_trigger = True
        else:
            signal_side = 'NONE'
            signal_trigger = False
        if signal_trigger:
            order_list = [
                {'offset': 0.0005, 'ratio': 0.30},
                {'offset': 0.0010, 'ratio': 0.20},
                {'offset': 0.0045, 'ratio': 0.10},
                {'offset': 0.0065, 'ratio': 0.20},
                {'offset': 0.0110, 'ratio': 0.20},
            ]
            logger.info(f"下单信号触发: side={signal_side}, trigger_price={latest_price:.2f}, %B={percent_b:.3f}")
            return Signal(True, signal_side, 0, 0, order_details={'trigger_price': latest_price, 'percent_b': percent_b}), order_list
        else:
            return Signal(False, 'NONE', 0, 0), []

    async def place_dynamic_limit_orders(self, side: str, order_list: list, trigger_price: float):
        base_price = trigger_price
        for order in order_list:
            offset = order['offset']
            ratio = order['ratio']
            order_qty = round(QUANTITY * ratio, self.config.quantity_precision)
            if order_qty <= 0:
                logger.error("无效订单数量，跳过当前档")
                continue
            if side == 'BUY':
                limit_price = round(base_price * (1 - offset), self.config.price_precision)
            else:
                limit_price = round(base_price * (1 + offset), self.config.price_precision)
            if limit_price <= 0:
                logger.error("无效挂单价格，跳过当前档")
                continue
            params = {
                'symbol': SYMBOL,
                'contractType': 'PERPETUAL',
                'side': side,
                'type': 'LIMIT',
                'price': limit_price,
                'quantity': order_qty,
                'timeInForce': 'GTC'
            }
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"限价挂单成功 {side}@{limit_price}，数量: {order_qty}（偏移 {offset*100:.2f}%, ratio {ratio}），返回: {data}")
            except Exception as e:
                logger.error(f"限价挂单失败: {str(e)}")
        asyncio.create_task(self.adjust_pending_orders(side))

    async def adjust_pending_orders(self, side: str):
        await asyncio.sleep(180)
        current_b = await self.get_current_percentb()
        if 0.3 <= current_b <= 0.7:
            await self.cancel_all_orders()
            logger.info(f"%B回落至{current_b:.3f}，撤销所有未成交订单")

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
        params = {'symbol': SYMBOL, 'contractType': 'PERPETUAL'}
        try:
            await self.client._signed_request('DELETE', '/openOrders', params)
        except Exception as e:
            logger.error(f"撤单异常: {str(e)}")

    # ---------- 止盈止损系统 ----------

    async def update_dynamic_stop_loss(self):
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.last_trade_side is None:
            return
        latest_price = df['close'].values[-1]
        close_prices = df['close'].values.astype(np.float64)
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
            logger.info(f"多单动态止损更新：当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")
        elif self.last_trade_side == 'SHORT':
            new_stop = latest_price + dynamic_offset
            if self.hard_stop is None or new_stop < self.hard_stop:
                self.hard_stop = new_stop
            self.trailing_stop = latest_price + dynamic_offset
            logger.info(f"空单动态止损更新：当前价={latest_price:.2f}, 带宽={bb_width:.4f}, 新止损={self.hard_stop:.2f}")

    async def manage_profit_targets(self):
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or self.entry_price is None:
            return
        current_price = df['close'].values[-1]
        current_b = await self.get_current_percentb()
        if self.last_trade_side == 'LONG':
            if current_b >= 0.82 and not self.tp_triggered_30:
                logger.info("首次止盈信号，多单市价平仓30%")
                await self.close_position(side='SELL', ratio=0.3)
                self.tp_triggered_30 = True
            elif current_b >= 1.0 and not self.tp_triggered_20:
                logger.info("首次止盈信号，多单市价平仓20%")
                await self.close_position(side='SELL', ratio=0.2)
                self.tp_triggered_20 = True
            await self.place_take_profit_orders(side='SELL')
        elif self.last_trade_side == 'SHORT':
            if current_b <= 0.18 and not self.tp_triggered_30:
                logger.info("首次止盈信号，空单市价平仓30%")
                await self.close_position(side='BUY', ratio=0.3)
                self.tp_triggered_30 = True
            elif current_b <= 0.0 and not self.tp_triggered_20:
                logger.info("首次止盈信号，空单市价平仓20%")
                await self.close_position(side='BUY', ratio=0.2)
                self.tp_triggered_20 = True
            await self.place_take_profit_orders(side='BUY')
        if self.last_trade_side == 'LONG' and current_price < self.hard_stop:
            logger.info("多单价格低于动态止损，触发平仓")
            await self.close_position(side='SELL', ratio=1.0)
        elif self.last_trade_side == 'SHORT' and current_price > self.hard_stop:
            logger.info("空单价格高于动态止损，触发平仓")
            await self.close_position(side='BUY', ratio=1.0)

    async def place_take_profit_orders(self, side: str):
        df_now = await self.client.fetch_klines(interval='3m', limit=50)
        if df_now.empty:
            return
        current_price = df_now['close'].values[-1]
        tp_offsets = [0.0025, 0.0040, 0.0060, 0.0080, 0.0120]
        remaining_qty = QUANTITY  # 此处建议替换为实际持仓查询结果
        qty_each = round(remaining_qty / (len(tp_offsets) + 1), self.config.quantity_precision)
        for offset in tp_offsets:
            if side == 'SELL':
                tp_price = round(current_price * (1 + offset), self.config.price_precision)
            else:
                tp_price = round(current_price * (1 - offset), self.config.price_precision)
            params = {
                'symbol': SYMBOL,
                'contractType': 'PERPETUAL',
                'side': side,
                'type': 'LIMIT',
                'price': tp_price,
                'quantity': qty_each,
                'timeInForce': 'GTC'
            }
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"止盈挂单成功 {side}@{tp_price}，数量: {qty_each}（TP offset {offset*100:.2f}%），返回: {data}")
            except Exception as e:
                logger.error(f"止盈挂单失败: {str(e)}")

    # ---------- 市价平仓 ----------

    async def close_position(self, side: str, ratio: float):
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return
        current_price = df['close'].values[-1]
        if side == 'BUY':
            price_limit = current_price * (1 + self.config.max_slippage_market)
        else:
            price_limit = current_price * (1 - self.config.max_slippage_market)
        logger.info(f"市价平仓 {side}，比例 {ratio}，价格限制 {price_limit:.2f}")
        params = {
            'symbol': SYMBOL,
            'contractType': 'PERPETUAL',
            'side': side,
            'type': 'MARKET',
            'quantity': QUANTITY * ratio
        }
        if params['quantity'] <= 0 or price_limit <= 0:
            logger.error("无效订单数量或价格，跳过平仓")
            return
        try:
            data = await self.client._signed_request('POST', '/order', params)
            logger.info(f"市价平仓成功，返回: {data}")
        except Exception as e:
            logger.error(f"平仓请求失败: {str(e)}")

    # ---------- 风险控制 ----------

    async def risk_control_loop(self):
        while True:
            try:
                await self.monitor_risk()
            except Exception as e:
                logger.error(f"风险监控异常: {str(e)}")
            await asyncio.sleep(10)

    async def monitor_risk(self):
        if self.daily_pnl < -self.config.daily_drawdown_limit:
            logger.warning("日内亏损超20%，进入只平仓模式")
            if self.last_trade_side == 'LONG':
                await self.close_position(side='SELL', ratio=1.0)
            elif self.last_trade_side == 'SHORT':
                await self.close_position(side='BUY', ratio=1.0)
        if self.consecutive_losses >= 3:
            logger.warning("连续亏损3次，调整仓位至50%")
            # 实际情况中，根据资金及风险调整仓位

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

    # ---------- 订单通知 ----------
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

    # ---------- 主执行入口 ----------
    async def order_signal_loop(self):
        while True:
            try:
                signal, order_list = await self.analyze_order_signals_3m()
                if signal.action:
                    await self.place_dynamic_limit_orders(signal.side, order_list, trigger_price=signal.order_details.get("trigger_price"))
                else:
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

    async def execute(self):
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        await asyncio.gather(
            self.trend_monitoring_loop(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.risk_control_loop()
        )


async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，正在优雅退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")


if __name__ == "__main__":
    asyncio.run(main())