#!/usr/bin/env python3
# 高频交易引擎 v3.7 (ETH/USDC合约专版)

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
load_dotenv('/root/zhibai/.env')

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = os.getenv('TRADING_PAIR', 'ETHUSDC').strip()
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.06))
REST_URL = 'https://dapi.binance.com'  # USDC合约专用端点

if not all([API_KEY, SECRET_KEY]):
    raise ValueError("Missing API credentials in .env file")

# ========== 高频参数 ==========
RATE_LIMITS = {
    'klines': (45, 5),
    'orders': (180, 10),
    'leverage': (20, 60)
}

# ========== 监控指标 ==========
start_http_server(8000)
METRICS = {
    'memory': Gauge('hft_memory_usage', 'Memory usage (MB)'),
    'latency': Gauge('order_exec_latency', 'Order latency (ms)'),
    'throughput': Gauge('api_throughput', 'API requests/sec'),
    'position': Gauge('eth_position', 'Current ETH position')
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
    """ETH/USDC交易参数优化"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 2.2
    max_retries: int = 5
    order_timeout: float = 0.2
    price_precision: int = 2
    quantity_precision: int = 3
    risk_ratio: float = 0.02


class BinanceHFTClient:
    """USDC合约高频客户端"""

    def __init__(self):
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=48,
            ssl=True,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        self._init_session()
        self.recv_window = 5000
        self.request_timestamps = defaultdict(
            lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 50))
        self._time_diff = 0

    def _init_session(self):
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.05,
            max_timeout=1.0,
            statuses={408, 429, 500, 502, 503, 504}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=self.config.order_timeout)
        )

    async def _signed_request(self, method: str, endpoint: str, params: Dict) -> Dict:
        """增强型签名请求"""
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
        url = f"{REST_URL}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                await self._rate_limit_check(endpoint)
                async with self.session.request(
                        method,
                        url,
                        headers=headers,
                        params={**params, 'signature': signature}
                ) as resp:
                    if resp.status != 200:
                        error = await resp.json()
                        logger.error(f"API Error {resp.status}: {error}")
                        continue
                    return await resp.json()
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                await asyncio.sleep(0.2 ** attempt)
        raise Exception("Max retries exceeded")

    async def sync_server_time(self):
        """精准时间同步"""
        time_diffs = []
        for _ in range(5):
            try:
                async with self.session.get(f"{REST_URL}/dapi/v1/time") as resp:
                    data = await resp.json()
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    time_diffs.append(server_time - local_time)
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Time sync error: {str(e)}")

        if time_diffs:
            self._time_diff = int(np.mean(time_diffs))
            if abs(self._time_diff) > 1000:
                logger.warning(f"Time deviation: {self._time_diff}ms")

    async def manage_leverage(self):
        """杠杆设置"""
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'marginType': 'ISOLATED'
        }
        return await self._signed_request('POST', '/dapi/v1/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """优化K线获取"""
        params = {
            'symbol': SYMBOL,
            'interval': '1m',
            'limit': 100,
            'contractType': 'PERPETUAL'
        }
        try:
            data = await self._signed_request('GET', '/dapi/v1/klines', params)
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            return df.iloc[-100:]
        except Exception as e:
            logger.error(f"Klines error: {str(e)}")
            return None


class ETHUSDCTradingStrategy:
    """ETH/USDC高频策略"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()
        self.position = 0.0

    async def execute(self):
        """策略执行主循环"""
        await self.client.manage_leverage()
        await self.client.sync_server_time()

        while True:
            try:
                start_time = time.monotonic()

                # 获取市场数据
                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(0.3)
                    continue

                # 计算指标
                close_prices = df['close'].values.astype(np.float64)
                ema_fast = self._calculate_ema(close_prices, self.config.ema_fast)
                ema_slow = self._calculate_ema(close_prices, self.config.ema_slow)
                atr = self._calculate_atr(df)

                # 交易信号
                if ema_fast[-1] > ema_slow[-1] * 1.0015:
                    stop_price = close_prices[-1] - (atr * self.config.volatility_multiplier)
                    await self._place_order('BUY', stop_price)
                elif ema_fast[-1] < ema_slow[-1] * 0.9985:
                    stop_price = close_prices[-1] + (atr * self.config.volatility_multiplier)
                    await self._place_order('SELL', stop_price)

                # 性能监控
                METRICS['latency'].set((time.monotonic() - start_time) * 1000)
                await asyncio.sleep(0.25)

            except Exception as e:
                logger.error(f"Strategy error: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """向量化EMA计算"""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """优化ATR计算"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        prev_close = close[:-1]
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close)
        )
        return np.mean(tr[-self.config.atr_window:])

    async def _place_order(self, side: str, price: float):
        """执行订单"""
        try:
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'STOP_MARKET',
                'stopPrice': round(price, self.config.price_precision),
                'quantity': round(QUANTITY, self.config.quantity_precision),
                'timeInForce': 'GTC',
                'closePosition': 'false'
            }
            response = await self.client._signed_request('POST', '/dapi/v1/order', params)
            logger.info(f"Order executed: {response}")
            self.position = float(response['executedQty'])
            METRICS['position'].set(self.position)
        except Exception as e:
            logger.error(f"Order failed: {str(e)}")


async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCTradingStrategy(client)
    try:
        await strategy.execute()
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())