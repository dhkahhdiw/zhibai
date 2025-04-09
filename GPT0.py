#!/usr/bin/env python3
# ETH/USDC高频交易引擎 v4.2 (全问题修复版)

import uvloop

uvloop.install()

import os
import time
import hmac
import hashlib
import urllib.parse
import logging
import asyncio
from typing import Optional, Dict
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
_env_path = '/root/zhibai/.env'
if not os.path.exists(_env_path):
    raise FileNotFoundError(f"未找到环境文件: {_env_path}")

load_dotenv(_env_path)

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = os.getenv('TRADING_PAIR', 'ETHUSD_PERP').strip()
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.06))
REST_URL = 'https://dapi.binance.com'

# 密钥格式验证
if len(API_KEY) != 64 or len(SECRET_KEY) != 64:
    raise ValueError("API密钥格式错误，应为64位字符")

# ========== 高频参数 ==========
RATE_LIMITS = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

# ========== 监控指标 ==========
start_http_server(8000)
METRICS = {
    'memory': Gauge('hft_memory', '内存(MB)'),
    'latency': Gauge('order_latency', '延迟(ms)'),
    'throughput': Gauge('api_throughput', '请求数/秒'),
    'position': Gauge('eth_position', '持仓量'),
    'errors': Gauge('api_errors', 'API错误')
}

# ========== 日志配置 ==========
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
    max_retries: int = 3
    order_timeout: float = 0.15  # 150ms超时
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0
    network_timeout: float = 1.0  # 网络超时


class BinanceHFTClient:
    """高频交易客户端（终极修复版）"""

    def __init__(self):
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=128,
            ssl=True,
            ttl_dns_cache=60,
            force_close=True
        )
        self._init_session()
        self.recv_window = 5000
        self.request_timestamps = defaultdict(
            lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 150))
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
            timeout=ClientTimeout(
                total=self.config.network_timeout,
                sock_connect=self.config.order_timeout
            )
        )

    async def _rate_limit_check(self, endpoint: str):
        """增强速率限制检查"""
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
        """增强型签名请求"""
        params = {
            **params,
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': self.recv_window
        }

        try:
            query = urllib.parse.urlencode(sorted(params.items()))
            signature = hmac.new(
                SECRET_KEY.encode(),
                query.encode(),
                hashlib.sha256
            ).hexdigest()
        except Exception as e:
            logger.error(f"签名失败: {e}")
            await self.sync_server_time()
            raise

        headers = {"X-MBX-APIKEY": API_KEY}
        url = f"{REST_URL}{path}"

        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)

                async with self.session.request(
                        method,
                        url,
                        headers=headers,
                        params={**params, 'signature': signature}
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"API错误 {resp.status}: {error}")
                        METRICS['errors'].inc()
                        continue
                    return await resp.json()
            except Exception as e:
                logger.error(f"请求失败 第{attempt + 1}次尝试: {str(e)}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))
                else:
                    raise
        raise Exception("超过最大重试次数")

    async def sync_server_time(self):
        """高精度时间同步"""
        try:
            async with self.session.get(f"{REST_URL}/dapi/v1/time") as resp:
                data = await resp.json()
                server_time = data['serverTime']
                local_time = int(time.time() * 1000)
                self._time_diff = server_time - local_time
                if abs(self._time_diff) > 500:
                    logger.warning(f"时间偏差警告: {self._time_diff}ms")
        except Exception as e:
            logger.error(f"时间同步失败: {str(e)}")

    async def manage_leverage(self):
        """设置杠杆（USDC合约专用）"""
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'dualSidePosition': 'true'
        }
        return await self._signed_request('POST', '/dapi/v1/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """获取K线数据（优化内存）"""
        params = {
            'symbol': SYMBOL,
            'interval': '1m',
            'limit': 100,
            'contractType': 'PERPETUAL'
        }
        try:
            data = await self._signed_request('GET', '/dapi/v1/klines', params)
            arr = np.zeros((len(data), 6), dtype=np.float64)
            for i, candle in enumerate(data):
                arr[i] = [float(candle[1]), float(candle[2]),
                          float(candle[3]), float(candle[4]),
                          float(candle[5]), float(candle[7])]
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


class ETHUSDCStrategy:
    """交易策略实现（修复版）"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()

    async def execute(self):
        """策略主循环"""
        await self.client.manage_leverage()
        await self.client.sync_server_time()

        while True:
            try:
                cycle_start = time.monotonic()
                METRICS['memory'].set(psutil.Process().memory_info().rss // 1024 // 1024)

                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(0.15)
                    continue

                close_prices = df['close'].values.astype(np.float64)
                ema_fast = pd.Series(close_prices).ewm(
                    span=self.config.ema_fast, adjust=False).mean().values
                ema_slow = pd.Series(close_prices).ewm(
                    span=self.config.ema_slow, adjust=False).mean().values
                atr = self._calculate_atr(df)

                if ema_fast[-1] > ema_slow[-1] * 1.003:
                    stop_price = close_prices[-1] - (atr * self.config.volatility_multiplier)
                    await self._execute_order('BUY', stop_price)
                elif ema_fast[-1] < ema_slow[-1] * 0.997:
                    stop_price = close_prices[-1] + (atr * self.config.volatility_multiplier)
                    await self._execute_order('SELL', stop_price)

                cycle_time = (time.monotonic() - cycle_start) * 1000
                METRICS['latency'].set(cycle_time)
                await asyncio.sleep(max(0.05, 0.15 - cycle_time / 1000))

            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """SIMD加速ATR计算"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        prev_close = close[:-1]
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close)
        ])
        return np.mean(tr[-self.config.atr_window:])

    async def _execute_order(self, side: str, price: float):
        """执行订单（带风险控制）"""
        try:
            available = self.config.max_position - abs(self.client.position)
            qty = min(QUANTITY, available)
            if qty <= 0:
                logger.warning("已达最大持仓限制")
                return

            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'STOP',
                'stopPrice': round(price, self.config.price_precision),
                'quantity': round(qty, self.config.quantity_precision),
                'timeInForce': 'GTC',
                'workingType': 'MARK_PRICE',
                'priceProtect': 'true'
            }

            response = await self.client._signed_request('POST', '/dapi/v1/order', params)
            self.client.position += qty if side == 'BUY' else -qty
            METRICS['position'].set(self.client.position)
            logger.info(f"订单成功: {response}")
        except Exception as e:
            logger.error(f"订单失败: {str(e)}")
            METRICS['errors'].inc()


async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        await strategy.execute()
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())