#!/usr/bin/env python3
# ETH/USDC 合约高频交易引擎 v4.2 (整合多周期指标与限价挂单示例)

import uvloop

uvloop.install()

import os
import asyncio
import time
import hmac
import hashlib
import urllib.parse
import logging
from typing import Dict, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from socket import AF_INET

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry

# 尝试导入 aiodns，如果未安装则使用默认解析器
try:
    from aiohttp.resolver import AsyncResolver

    resolver = AsyncResolver(nameservers=["8.8.8.8", "1.1.1.1"])
except Exception:
    resolver = None
    print("Warning: aiodns 库未安装，使用默认解析器")

from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server
import psutil

# ==================== 加载环境变量 ====================
_env_path = '/root/zhibai/.env'
if not os.path.exists(_env_path):
    raise FileNotFoundError(f"未找到环境文件: {_env_path}")
load_dotenv(_env_path)

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
# 这里交易对为 USDC 合约交易对，示例使用 "ETHUSDC"
SYMBOL = os.getenv('TRADING_PAIR', 'ETHUSDC').strip()
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.06))
# Binance Futures USDC-M 基础 URL 使用正式地址
REST_URL = 'https://api.binance.com'

# 检查 API 密钥
if not API_KEY or not SECRET_KEY:
    raise Exception("请在 /root/zhibai/.env 中正确配置 BINANCE_API_KEY 与 BINANCE_SECRET_KEY")
if len(API_KEY) != 64 or len(SECRET_KEY) != 64:
    raise ValueError("API密钥格式错误，应为64位字符，请检查 BINANCE_API_KEY 与 BINANCE_SECRET_KEY")

# ==================== 高频参数 ====================
RATE_LIMITS = {
    'klines': (60, 5),  # 每5秒最多60次请求
    'orders': (300, 10),
    'leverage': (30, 60)
}

# ==================== 监控指标 ====================
start_http_server(8000)
METRICS = {
    'memory': Gauge('hft_memory', '内存(MB)'),
    'latency': Gauge('order_latency', '延迟(ms)'),
    'throughput': Gauge('api_throughput', '请求数/秒'),
    'position': Gauge('eth_position', '持仓量'),
    'errors': Gauge('api_errors', 'API错误')
}

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,  # 如需调试可设为 DEBUG
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/eth_usdc_hft.log", encoding="utf-8", mode='a')
    ]
)
logger = logging.getLogger('ETH-USDC-HFT')
DEBUG = False


@dataclass
class TradingConfig:
    """交易参数配置"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 2.5
    max_retries: int = 3
    order_timeout: float = 0.5  # 500ms连接超时
    network_timeout: float = 2.0  # 2秒总超时
    price_precision: int = int(os.getenv('PRICE_PRECISION', '2'))
    quantity_precision: int = int(os.getenv('QUANTITY_PRECISION', '3'))
    max_position: float = 10.0


# ==================== 数据获取及下单模块 ====================
class BinanceHFTClient:
    """获取K线数据、发送签名请求以及下单"""

    def __init__(self):
        self.config = TradingConfig()
        connector_args = {
            "limit": 128,
            "ssl": True,
            "ttl_dns_cache": 60,
            "force_close": True,
            "family": AF_INET
        }
        if resolver is not None:
            connector_args["resolver"] = resolver
        self.connector = TCPConnector(**connector_args)
        self._init_session()
        self.recv_window = 5000
        self.request_timestamps = defaultdict(lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 150))
        self._time_diff = 0
        self.position = 0.0

    def _init_session(self):
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.1,
            max_timeout=0.5,
            statuses={408, 429, 500, 502, 503, 504}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=self.config.network_timeout, sock_connect=self.config.order_timeout)
        )

    async def _rate_limit_check(self, endpoint: str):
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]
        now = time.monotonic()
        while dq and dq[0] < now - period:
            dq.popleft()
        if len(dq) >= limit:
            wait_time = max(dq[0] + period - now + np.random.uniform(0, 0.01), 0)
            logger.warning(f"速率限制触发: {endpoint} 等待 {wait_time:.3f}s")
            await asyncio.sleep(wait_time)
        dq.append(now)
        METRICS['throughput'].inc()

    async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
        params.update({
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': self.recv_window
        })
        sorted_params = sorted(params.items())
        query = urllib.parse.urlencode(sorted_params)
        try:
            signature = hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
        except Exception as e:
            logger.error(f"签名生成失败: {str(e)}")
            await self.sync_server_time()
            raise
        full_query = f"{query}&signature={signature}"
        # 使用 USDC-M Futures 专用路径前缀 /sapi/v1/futures/usdc
        url = f"{REST_URL}/sapi/v1/futures/usdc{path}?{full_query}"
        headers = {"X-MBX-APIKEY": API_KEY}
        if DEBUG:
            logger.debug(f"请求 {method} {url}")
        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)
                async with self.session.request(method, url, headers=headers) as resp:
                    resp_text = await resp.text()
                    if DEBUG:
                        logger.debug(f"返回状态 {resp.status}, 内容: {resp_text}")
                    if resp.status == 401:
                        logger.error(f"API错误 401: {resp_text}")
                        raise Exception("API-key format invalid，请检查API密钥格式")
                    if resp.status != 200:
                        logger.error(f"API错误 {resp.status}: {resp_text}")
                        METRICS['errors'].inc()
                        continue
                    return await resp.json()
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))
                else:
                    raise
        raise Exception("超过最大重试次数")

    async def sync_server_time(self):
        url = f"{REST_URL}/sapi/v1/futures/usdc/time"
        time_diffs = []
        for _ in range(7):
            try:
                async with self.session.get(url) as resp:
                    data = await resp.json()
                    server_time = data.get('serverTime')
                    if server_time is None:
                        raise Exception("serverTime 未返回")
                    local_time = int(time.time() * 1000)
                    time_diffs.append(server_time - local_time)
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"时间同步失败: {str(e)}")
        if time_diffs:
            self._time_diff = int(np.mean(time_diffs[-3:]))
            if abs(self._time_diff) > 500:
                logger.warning(f"时间偏差警告: {self._time_diff}ms")
        else:
            raise Exception("无法同步服务器时间")

    async def manage_leverage(self):
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE, 'dualSidePosition': 'true'}
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        params = {'symbol': SYMBOL, 'interval': '1m', 'limit': 100, 'contractType': 'PERPETUAL'}
        try:
            data = await self._signed_request('GET', '/klines', params)
            arr = np.empty((len(data), 6), dtype=np.float64)
            for i, candle in enumerate(data):
                # 根据文档字段，下标调整为：1=open, 2=high, 3=low, 4=close, 5=volume, 7=结束时间
                arr[i] = [float(candle[j]) for j in [1, 2, 3, 4, 5, 7]]
            return pd.DataFrame({
                'open': arr[:, 0],
                'high': arr[:, 1],
                'low': arr[:, 2],
                'close': arr[:, 3],
                'volume': arr[:, 4],
                'timestamp': arr[:, 5].astype('uint64')
            }).iloc[-100:]
        except Exception as e:
            logger.error(f"获取K线失败: {str(e)}")
            return None


# ==================== 多周期数据与指标计算模块 ====================
def resample_data(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    重采样 K 线数据到指定周期
    interval 格式如 '3T'(3分钟), '15T'(15分钟), '1H', '6H'
    """
    df.index = pd.to_datetime(df['timestamp'], unit='ms')
    resampled = df.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled


def SMA(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': histogram})


def KDJ(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, k_period: int = 3,
        d_period: int = 3) -> pd.DataFrame:
    low_min = low.rolling(n, min_periods=n).min()
    high_max = high.rolling(n, min_periods=n).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=k_period - 1, adjust=False).mean()
    d = k.ewm(com=d_period - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({'K': k, 'D': d, 'J': j})


def BBI(series: pd.Series) -> pd.Series:
    return (SMA(series, 3) + SMA(series, 6) + SMA(series, 12) + SMA(series, 24)) / 4


def MTM(series: pd.Series, period: int = 10) -> pd.Series:
    return series.diff(period)


def LWR(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    highest = high.rolling(n, min_periods=n).max()
    lowest = low.rolling(n, min_periods=n).min()
    return 100 * (close - lowest) / (highest - lowest)


def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def PSR(returns: pd.Series, window: int = 30) -> float:
    """
    简化的概率夏普比率计算：利用滚动窗口收益均值/标准差计算 Sharpe Ratio,
    并转化为概率。这里只给出一个示例，实际计算可参考论文方法。
    """
    r = returns.rolling(window).mean().iloc[-1]
    sigma = returns.rolling(window).std().iloc[-1]
    if sigma == 0:
        return 0.0
    sharpe = r / sigma
    # 概率值的简化版本：归一化 Sharpe (此处仅为示例)
    psr = 1 / (1 + np.exp(-sharpe))
    return psr


# ==================== 策略模块 ====================
@dataclass
class CombinedSignals:
    ea_long: bool = False
    ea_short: bool = False
    rsi_long: bool = False
    rsi_short: bool = False
    macd_long: bool = False
    macd_short: bool = False
    kdj_long: bool = False
    kdj_short: bool = False
    bbi_long: bool = False
    bbi_short: bool = False
    mtm_long: bool = False
    mtm_short: bool = False
    lwr_long: bool = False
    lwr_short: bool = False
    psr_long: bool = False
    psr_short: bool = False


class ETHUSDCStrategy:
    """集成多指标策略，挂限价单并设置动态止盈止损"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()

    async def execute(self):
        # 设置杠杆并同步时间
        await self.client.manage_leverage()
        await self.client.sync_server_time()

        while True:
            try:
                start_cycle = time.monotonic()
                # 获取1分钟K线数据
                df = await self.client.fetch_klines()
                if df is None or df.empty:
                    await asyncio.sleep(0.2)
                    continue

                # 计算 ATR 用于止盈止损，使用1分钟数据
                atr = ATR(df['high'], df['low'], df['close'], period=14).iloc[-1]

                # EA信号（使用SMA交叉）在各周期上，重采样数据
                signals = CombinedSignals()
                # 1分钟
                s1 = self.compute_ea_signal(df, short_period=5, long_period=20)
                # 3分钟
                df3m = resample_data(df, '3T')
                s3 = self.compute_ea_signal(df3m, short_period=5, long_period=30)
                # 15分钟
                df15m = resample_data(df, '15T')
                s15 = self.compute_ea_signal(df15m, short_period=7, long_period=50)
                # 1小时
                df1h = resample_data(df, '1H')
                s1h = self.compute_ea_signal(df1h, short_period=10, long_period=60)
                # 6小时
                df6h = resample_data(df, '6H')
                s6h = self.compute_ea_signal(df6h, short_period=20, long_period=120)
                # 综合 EA 信号：多数周期金叉看多，多数周期死叉看空
                signals.ea_long = sum([s1, s3, s15, s1h, s6h]) >= 3
                signals.ea_short = sum([not s1, not s3, not s15, not s1h, not s6h]) >= 3

                # RSI（14）
                rsi = RSI(df['close'], period=14)
                signals.rsi_long = rsi.iloc[-1] > 60
                signals.rsi_short = rsi.iloc[-1] < 40

                # MACD
                macd_df = MACD(df['close'], fast=12, slow=26, signal=9)
                signals.macd_long = macd_df['histogram'].iloc[-1] > 0
                signals.macd_short = macd_df['histogram'].iloc[-1] < 0

                # KDJ
                kdj_df = KDJ(df['high'], df['low'], df['close'], n=9, k_period=3, d_period=3)
                signals.kdj_long = kdj_df['J'].iloc[-1] > 50
                signals.kdj_short = kdj_df['J'].iloc[-1] < 50

                # BBI（大周期均线综合）
                signals.bbi_long = BBI(df['close']).iloc[-1] < df['close'].iloc[-1]
                signals.bbi_short = BBI(df['close']).iloc[-1] > df['close'].iloc[-1]

                # MTM（动量指标）
                mtm = MTM(df['close'], period=10)
                signals.mtm_long = mtm.iloc[-1] > 0
                signals.mtm_short = mtm.iloc[-1] < 0

                # LWR（Williams %R）
                lwr = LWR(df['high'], df['low'], df['close'], n=14)
                signals.lwr_long = lwr.iloc[-1] < 20  # 超买区
                signals.lwr_short = lwr.iloc[-1] > 80  # 超卖区

                # PSR，基于最近30分钟的收益（简化示例）
                returns = df['close'].pct_change().dropna()
                psr_val = PSR(returns, window=30)
                signals.psr_long = psr_val > 0.7
                signals.psr_short = psr_val < 0.3

                # 综合信号：如多仓信号：至少EA、RSI、MACD、KDJ、PSR 五项多数信号看多
                long_count = sum(
                    [signals.ea_long, signals.rsi_long, signals.macd_long, signals.kdj_long, signals.psr_long])
                short_count = sum(
                    [signals.ea_short, signals.rsi_short, signals.macd_short, signals.kdj_short, signals.psr_short])
                # 简单逻辑：多仓信号大于3项则看多，空仓同理
                if long_count >= 3:
                    order_side = 'BUY'
                    # 限价单价格设为当前价减去 ATR 的一定比例
                    limit_price = df['close'].iloc[-1] - atr * 0.5
                    take_profit = df['close'].iloc[-1] + atr * 2.5
                    stop_loss = df['close'].iloc[-1] - atr * 1.5
                    print("多仓信号：", long_count)
                    await self.place_limit_order(order_side, limit_price, take_profit, stop_loss)
                elif short_count >= 3:
                    order_side = 'SELL'
                    limit_price = df['close'].iloc[-1] + atr * 0.5
                    take_profit = df['close'].iloc[-1] - atr * 2.5
                    stop_loss = df['close'].iloc[-1] + atr * 1.5
                    print("空仓信号：", short_count)
                    await self.place_limit_order(order_side, limit_price, take_profit, stop_loss)
                else:
                    print("综合信号不足，多空双方信号数量：", long_count, short_count)
                # 计算并记录本周期耗时
                cycle_time = (time.monotonic() - start_cycle) * 1000
                METRICS['latency'].set(cycle_time)
                await asyncio.sleep(max(0.5, 1 - cycle_time / 1000))
            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(1)

    async def place_limit_order(self, side: str, limit_price: float, tp: float, sl: float):
        # 下单时只使用限价单挂单, 此处示例直接挂单，不含自动取消逻辑，可根据需要扩展
        # 格式化数量及价格
        formatted_price = float(f"{limit_price:.{self.config.price_precision}f}")
        formatted_qty = float(f"{QUANTITY:.{self.config.quantity_precision}f}")
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'LIMIT',
            'price': formatted_price,
            'quantity': formatted_qty,
            'timeInForce': 'GTC'
        }
        order_resp = await self.client._signed_request('POST', '/order', params)
        print(f"挂单 {side} 限价 {formatted_price} 返回:", order_resp)
        # 动态止盈止损逻辑（示例仅打印，如可接入追单模块）
        print(f"设置止盈: {tp:.{self.config.price_precision}f} ，止损: {sl:.{self.config.price_precision}f}")

    def compute_ea_signal(self, df: pd.DataFrame, short_period: int, long_period: int) -> bool:
        """利用 SMA 均线计算 EA 信号：短期上穿长期为看多"""
        short_sma = df['close'].rolling(short_period).mean()
        long_sma = df['close'].rolling(long_period).mean()
        # 如果最后一个点短均线上穿长期均线，则信号为 True
        return short_sma.iloc[-1] > long_sma.iloc[-1]


# ==================== 主函数 ====================
async def main():
    # 初始化 Binance 客户端
    from GPT0 import BinanceHFTClient  # 若是单文件运行，此步不需要
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        await strategy.execute()
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())


# ==================== 辅助函数 ====================
def resample_data(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """利用 1 分钟数据重采样到其他周期，interval 格式如 '3T','15T','1H','6H'"""
    df.index = pd.to_datetime(df['timestamp'], unit='ms')
    resampled = df.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled