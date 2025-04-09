#!/usr/bin/env python3
# 高频交易引擎 v3.3 (Vultr VHF-1C-1GB 特化版)

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
RATE_LIMITS = {
    'klines': (30, 5),  # 30次/5秒
    'orders': (120, 10),  # 120次/10秒
    'leverage': (10, 60)  # 10次/分钟
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
    """交易策略参数"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 1.5
    max_retries: int = 3
    order_timeout: float = 1.2  # 秒


class BinanceHFTClient:
    """高频交易客户端（Vultr优化版）"""

    def __init__(self):
        self.connector = TCPConnector(
            limit=12,
            ssl=False,
            force_close=True,
            use_dns_cache=True,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        self._init_session()
        self.recv_window = 5000
        self.request_timestamps = defaultdict(
            lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 10)
        )
        self._time_diff = 0
        self._config = TradingConfig()

    def _init_session(self):
        """初始化带重试的会话"""
        retry_opts = ExponentialRetry(
            attempts=3,
            start_timeout=0.3,
            max_timeout=1.5,
            statuses={408, 500, 502, 503, 504}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=self._config.order_timeout)
        )

    async def _rate_limit_check(self, endpoint: str):
        """动态速率限制检查"""
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]

        now = time.time()
        while dq and dq[0] < now - period:
            dq.popleft()

        if len(dq) >= limit:
            wait_time = dq[0] + period - now
            logger.warning(f"Rate limit hit: {endpoint} - Waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        dq.append(now)
        METRICS['throughput'].inc()

    async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
        """带签名的API请求"""
        params.update({
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': self.recv_window
        })
        query = urllib.parse.urlencode(sorted(params.items()))
        signature = hmac.new(
            SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {"X-MBX-APIKEY": API_KEY}
        for attempt in range(self._config.max_retries):
            try:
                await self._rate_limit_check(path.split('/')[-1])
                async with self.session.request(
                        method,
                        REST_URL + path,
                        headers=headers,
                        data={​ ** ​params, 'signature': signature}  # 修复零宽字符问题[5](@ref)
                ) as resp:
                if resp.status != 200:
                    error = await resp.json()
                    logger.error(f"API Error {resp.status}: {error}")
                    continue
                return await resp.json()
        except Exception as e:
        logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
        await asyncio.sleep(0.5 ​ ** ​ attempt)

        raise Exception("Max retries exceeded")

    async def sync_server_time(self):
        """增强型时间同步"""
        for _ in range(3):
            try:
                data = await self._signed_request('GET', '/fapi/v1/time', {})
                server_time = data['serverTime']
                local_time = int(time.time() * 1000)
                self._time_diff = server_time - local_time
                if abs(self._time_diff) > 3000:
                    logger.warning(f"时间偏差过大: {self._time_diff}ms")
                return
            except Exception as e:
                logger.error(f"时间同步失败: {str(e)}")
                await asyncio.sleep(1)
        raise Exception("无法同步服务器时间")

    async def manage_leverage(self):
        """杠杆管理"""
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE}
        return await self._signed_request('POST', '/fapi/v1/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """内存优化版K线获取"""
        params = {'symbol': SYMBOL, 'interval': '1m', 'limit': 100}
        try:
            data = await self._signed_request('GET', '/fapi/v1/klines', params)
            # 使用内存视图优化[1](@ref)
            arr = np.array(data, dtype=np.float32)
            return pd.DataFrame({
                'timestamp': arr[:, 0],
                'open': arr[:, 1].astype('float32'),
                'high': arr[:, 2].astype('float32'),
                'low': arr[:, 3].astype('float32'),
                'close': arr[:, 4].astype('float32'),
                'volume': arr[:, 5].astype('float32')
            }).iloc[-100:]
        except Exception as e:
            logger.error(f"获取K线失败: {str(e)}")
            return None

    class VolatilityStrategy:
        """波动率策略（Vultr优化版）"""

        def __init__(self, client: BinanceHFTClient):
            self.client = client
            self.config = TradingConfig()
            self._init_indicators()

        def _init_indicators(self):
            """向量化指标计算"""
            self.ema_fast = lambda x: x.ewm(span=self.config.ema_fast).mean().values
            self.ema_slow = lambda x: x.ewm(span=self.config.ema_slow).mean().values

        def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
            """向量化ATR计算"""
            tr = np.maximum(
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
            return np.mean(tr[-self.config.atr_window:])

        async def execute_strategy(self):
            """策略执行引擎"""
            await self.client.manage_leverage()
            await self.client.sync_server_time()

            while True:
                try:
                    METRICS['memory'].set(
                        psutil.Process().memory_info().rss // 1024 // 1024
                    )

                    df = await self.client.fetch_klines()
                    if df is None:
                        await asyncio.sleep(0.5)
                        continue

                    # 内存视图优化[1](@ref)
                    close_view = df['close'].values.astype(np.float32, copy=False)
                    ema_f = self.ema_fast(close_view)[-1]
                    ema_s = self.ema_slow(close_view)[-1]

                    if ema_f > ema_s * 1.001:
                        atr_val = self.calculate_atr(
                            df['high'].values,
                            df['low'].values,
                            close_view
                        )
                        stop_price = close_view[-1] - atr_val * self.config.volatility_multiplier
                        await self._execute_order('BUY', stop_price)

                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"策略异常: {str(e)}")
                    await asyncio.sleep(5)

        async def _execute_order(self, side: str, price: float):
            """订单执行（带延迟监控）"""
            start_time = time.monotonic()
            try:
                params = {
                    'symbol': SYMBOL,
                    'side': side,
                    'type': 'STOP_MARKET',
                    'stopPrice': round(price, 2),
                    'quantity': QUANTITY,
                    'timeInForce': 'GTC'
                }
                resp = await self.client._signed_request('POST', '/fapi/v1/order', params)
                METRICS['latency'].set((time.monotonic() - start_time) * 1000)
                logger.info(f"订单成交 {side}@{resp.get('avgPrice')}")
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