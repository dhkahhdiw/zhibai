#!/usr/bin/env python3
# 高频交易引擎 v3.61 (Vultr VHF-1C-1GB 终极优化版)

import uvloop

uvloop.install()

import asyncio
import os
import time
import hmac
import hashlib
import urllib.parse
import logging
from typing import Optional, Dict, Deque, Any
from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server
import psutil

# ========== 环境配置 ==========
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
SYMBOL = os.getenv('TRADING_PAIR', 'ETHUSDT')
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.01))
REST_URL = 'https://fapi.binance.com'

# ========== 高频参数 ==========
# 各接口的 (请求上限, 时间周期[s])
RATE_LIMITS = {
    'klines': (40, 5),  # K线请求：最多40次，每5秒
    'orders': (160, 10),  # 订单请求：最多160次，每10秒
    'leverage': (15, 60)
}

# ========== 监控指标 ==========
start_http_server(8000)
METRICS = {
    'memory': Gauge('hft_memory', '内存使用(MB)'),
    'latency': Gauge('order_latency', '订单延迟(ms)'),
    'throughput': Gauge('requests_sec', 'API请求数/秒')
}

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/hft_engine.log",
                            encoding="utf-8",
                            mode='a',
                            delay=True)
    ]
)
logger = logging.getLogger('HFT-Core')


@dataclass
class TradingConfig:
    """交易策略参数（Vultr终极优化）"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 2.0  # 提高波动率系数
    max_retries: int = 7
    order_timeout: float = 0.25  # 超时时间（秒）
    price_precision: int = 2
    position_ratio: float = 0.95  # 仓位利用率


class BinanceHFTClient:
    """高频交易客户端（Vultr终极优化版）"""

    def __init__(self):
        # 先初始化配置，确保后续使用 self._config 时不会出错
        self._config = TradingConfig()

        self.connector = TCPConnector(
            limit=64,  # 增大连接池
            ssl=False,
            force_close=True,  # 强制关闭空闲连接
            use_dns_cache=True,
            ttl_dns_cache=180,  # DNS缓存时间（秒）
            enable_cleanup_closed=True
        )
        # 初始化 session 依赖 _config.order_timeout
        self._init_session()
        self.recv_window = 5000
        self.request_timestamps = defaultdict(
            lambda: deque(maxlen=RATE_LIMITS.get('klines', (60, 1))[0] + 60)
        )
        self._time_diff = 0

    def _init_session(self):
        """初始化智能会话"""
        retry_opts = ExponentialRetry(
            attempts=self._config.max_retries,
            start_timeout=0.03,  # 更激进超时设置
            max_timeout=1.5,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={asyncio.TimeoutError}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=self._config.order_timeout)
        )

    async def _rate_limit_check(self, endpoint: str):
        """动态速率限制（带自适应调整）"""
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]
        now = time.time()
        while dq and dq[0] < now - period:
            dq.popleft()

        if len(dq) >= limit:
            wait_time = dq[0] + period - now + np.random.uniform(0, 0.05)
            logger.warning(f"Rate limit hit: {endpoint} - Waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)

        dq.append(now)
        METRICS['throughput'].inc()

    async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
        """签名请求（完全修复版）"""
        params.update({
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': self.recv_window
        })
        query = urllib.parse.urlencode(sorted(params.items()))
        try:
            signature = hmac.new(
                SECRET_KEY.encode(),
                query.encode(),
                hashlib.sha256
            ).hexdigest()
        except Exception as e:
            logger.error(f"签名生成失败: {str(e)}")
            await self.sync_server_time()
            raise

        headers = {"X-MBX-APIKEY": API_KEY}
        for attempt in range(self._config.max_retries):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)

                post_data = {**params, 'signature': signature}

                async with self.session.request(
                        method,
                        REST_URL + path,
                        headers=headers,
                        data=post_data
                ) as resp:
                    if resp.status != 200:
                        error = await resp.json()
                        logger.error(f"API Error {resp.status}: {error}")
                        continue
                    return await resp.json()
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}): {str(e)}")
                await asyncio.sleep(0.25 ** attempt)
        raise Exception("超过最大重试次数")

    async def sync_server_time(self):
        """增强时间同步（带误差补偿）"""
        time_diffs = []
        for _ in range(7):
            try:
                data = await self._signed_request('GET', '/fapi/v1/time', {})
                server_time = data['serverTime']
                local_time = int(time.time() * 1000)
                time_diffs.append(server_time - local_time)
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"时间同步失败: {str(e)}")

        if time_diffs:
            self._time_diff = int(np.mean(time_diffs[-3:]))  # 使用最后三次平均值
            if abs(self._time_diff) > 500:
                logger.warning(f"时间偏差警告: {self._time_diff}ms")
        else:
            raise Exception("无法同步服务器时间")

    async def manage_leverage(self):
        """杠杆管理（带仓位控制）"""
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE}
        return await self._signed_request('POST', '/fapi/v1/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """极速K线获取（内存优化版）"""
        params = {'symbol': SYMBOL, 'interval': '1m', 'limit': 100}
        try:
            data = await self._signed_request('GET', '/fapi/v1/klines', params)
            # 预分配内存 + 向量化处理
            arr = np.empty((len(data), 6), dtype=np.float64)
            for i, candle in enumerate(data):
                arr[i] = [float(candle[j]) for j in [0, 1, 2, 3, 4, 5]]
            return pd.DataFrame({
                'timestamp': arr[:, 0].astype('uint64'),
                'open': arr[:, 1],
                'high': arr[:, 2],
                'low': arr[:, 3],
                'close': arr[:, 4],
                'volume': arr[:, 5]
            }).iloc[-100:]
        except Exception as e:
            logger.error(f"获取K线失败: {str(e)}")
            return None


class VolatilityStrategy:
    """波动率策略（终极优化版）"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()  # 使用独立配置或共享客户端配置
        self._init_indicators()

    def _init_indicators(self):
        """预编译加速（通过lambda调用pandas内置函数）"""
        self.ema_fast = lambda x: x.ewm(span=self.config.ema_fast, adjust=False).mean().values
        self.ema_slow = lambda x: x.ewm(span=self.config.ema_slow, adjust=False).mean().values

    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """SIMD加速ATR计算"""
        prev_close = close[:-1]
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close)
        )
        return np.mean(tr[-self.config.atr_window:])

    async def execute_strategy(self):
        """策略主循环（超频模式）"""
        await self.client.manage_leverage()
        await self.client.sync_server_time()

        while True:
            try:
                # 实时监控内存使用
                METRICS['memory'].set(psutil.Process().memory_info().rss // 1024 // 1024)

                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(0.25)
                    continue

                close_view = df['close'].values.astype(np.float64, copy=False)
                ema_f = self.ema_fast(pd.Series(close_view))[-1]
                ema_s = self.ema_slow(pd.Series(close_view))[-1]

                if ema_f > ema_s * 1.002:  # 提高触发阈值
                    atr_val = self.calculate_atr(
                        df['high'].values,
                        df['low'].values,
                        close_view
                    )
                    stop_price = close_view[-1] - atr_val * self.config.volatility_multiplier
                    stop_price = round(stop_price, self.config.price_precision)
                    await self._execute_order('BUY', stop_price)

                await asyncio.sleep(0.25)  # 250ms循环周期

            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(1)

    async def _execute_order(self, side: str, price: float):
        """极速订单执行"""
        start_time = time.monotonic()
        try:
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'STOP_MARKET',
                'stopPrice': price,
                'quantity': round(QUANTITY * self.config.position_ratio, 3),
                'timeInForce': 'GTC'
            }
            resp = await self.client._signed_request('POST', '/fapi/v1/order', params)
            latency = (time.monotonic() - start_time) * 1000
            METRICS['latency'].set(latency)
            logger.info(f"订单成交 {side}@{resp.get('avgPrice')} 延迟:{latency:.2f}ms")
        except Exception as e:
            logger.error(f"下单失败: {str(e)}")
            if "Signature" in str(e):
                await self.client.sync_server_time()


async def main():
    client = BinanceHFTClient()
    strategy = VolatilityStrategy(client)
    try:
        await strategy.execute_strategy()
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())