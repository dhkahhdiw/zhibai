#!/usr/bin/env python3
# ETH/USDC高频交易引擎 v4.9 (Vultr生产环境稳定版)

import uvloop

uvloop.install()

import os
import asyncio
import time
import hmac
import hashlib
import urllib.parse
import logging
from typing import Dict, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from socket import AF_INET

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from aiohttp.resolver import AsyncResolver

from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server

# ==================== 环境配置 ====================
_env_path = '/root/zhibai/.env'
load_dotenv(_env_path)

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDC'
LEVERAGE = 10
QUANTITY = 0.06
REST_URL = 'https://api.binance.com'

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
        logging.FileHandler("/var/log/eth_usdc_hft.log",
                            encoding="utf-8",
                            mode='a')
    ]
)
logger = logging.getLogger('ETH-USDC-HFT')


@dataclass
class TradingConfig:
    """交易参数配置"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 2.5
    max_retries: int = 5
    order_timeout: float = 0.3  # 300ms超时
    network_timeout: float = 1.5
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0
    slippage: float = 0.001


class BinanceHFTClient:
    def __init__(self):
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=512,
            resolver=AsyncResolver(),
            ttl_dns_cache=300,
            force_close=True,
            family=AF_INET
        )
        self._init_session()
        self.recv_window = 6000
        self.request_timestamps = defaultdict(
            lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 200)
        )
        self._time_diff = 0
        self.position = 0.0

    def _init_session(self):
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.1,
            max_timeout=1.0,
            statuses={408, 429, 500, 502, 503, 504}
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
        """增强型时间同步（带空值检测）"""
        for retry in range(5):
            try:
                data = await self._signed_request('GET', '/sapi/v1/futures/usdc/time', {})
                if not data or 'serverTime' not in data:
                    raise ValueError("无效的服务器响应")

                server_time = data['serverTime']
                local_time = int(time.time() * 1000)
                self._time_diff = server_time - local_time

                if abs(self._time_diff) > 500:
                    logger.warning(f"时间偏差警告: {self._time_diff}ms")
                return
            except Exception as e:
                logger.error(f"时间同步失败(重试{retry + 1}): {str(e)}")
                await asyncio.sleep(0.5 ​ ** retry)
                raise Exception("无法同步服务器时间")

            async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
                params.update({
                    'timestamp': int(time.time() * 1000 + self._time_diff),
                    'recvWindow': self.recv_window
                })
                sorted_params = sorted(params.items())
                query = urllib.parse.urlencode(sorted_params)
                signature = hmac.new(
                    SECRET_KEY.encode(),
                    query.encode(),
                    hashlib.sha256
                ).hexdigest()
                url = f"{REST_URL}/sapi/v1/futures/usdc{path}?{query}&signature={signature}"
                headers = {"X-MBX-APIKEY": API_KEY}

                for attempt in range(self.config.max_retries + 1):
                    try:
                        endpoint = path.split('/')[-1]
                        await self._rate_limit_check(endpoint)
                        async with self.session.request(method, url, headers=headers) as resp:
                            if resp.status != 200:
                                error = await resp.json(content_type=None)
                                logger.error(f"API错误 {resp.status}: {error.get('msg', '未知错误')}")
                                continue
                            return await resp.json(content_type=None)
                    except Exception as e:
                        logger.error(f"请求失败 (尝试 {attempt + 1}): {str(e)}")
                        await asyncio.sleep(0.1 * (2 ​ ** attempt))
                raise Exception("超过最大重试次数")

            # 其他方法保持原有逻辑...

        class ETHUSDCStrategy:
            def __init__(self, client: BinanceHFTClient):
                self.client = client
                self.config = TradingConfig()
                self._indicator_cache = defaultdict(lambda: None)

            # 策略执行方法保持原有逻辑...

        async def main():
            client = BinanceHFTClient()
            strategy = ETHUSDCStrategy(client)
            try:
                await strategy.execute()
            finally:
                await client.session.close()

        if __name__ == "__main__":
            asyncio.run(main())