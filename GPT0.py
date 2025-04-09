#!/usr/bin/env python3
# ETH/USDC高频交易引擎 v3.8 (Vultr VHF专项优化版)

import uvloop

uvloop.install()

import asyncio
import os
import time
import hmac
import hashlib
import urllib.parse
import logging
from typing import Optional, Dict, Deque
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
SYMBOL = os.getenv('TRADING_PAIR', 'ETHUSD_PERP').strip()  # USDC合约格式
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.06))
REST_URL = 'https://dapi.binance.com'  # USDC合约专用端点

# ========== 高频参数 ==========
RATE_LIMITS = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

# ========== 监控指标 ==========
start_http_server(8000)
METRICS = {
    'memory': Gauge('hft_memory', '内存使用(MB)'),
    'latency': Gauge('order_latency', '订单延迟(ms)'),
    'throughput': Gauge('api_throughput', 'API请求数/秒'),
    'position': Gauge('eth_position', '当前持仓')
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
    volatility_multiplier: float = 2.5
    max_retries: int = 5
    order_timeout: float = 0.15  # 150ms超时
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0  # 最大持仓量


class BinanceHFTClient:
    """USDC合约高频客户端（修复速率限制）"""

    def __init__(self):
        self.config = TradingConfig()
        self.connector = TCPConnector(
            limit=64,
            ssl=True,
            ttl_dns_cache=180,
            enable_cleanup_closed=True
        )
        self._init_session()
        self.recv_window = 5000
        self.request_timestamps = defaultdict(
            lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 100))
        self._time_diff = 0
        self.position = 0.0

    def _init_session(self):
        """初始化高性能会话"""
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.02,
            max_timeout=0.5,
            statuses={408, 429, 500, 502, 503, 504}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=self.config.order_timeout)
        )

    async def _rate_limit_check(self, endpoint: str):
        """动态速率限制检查（修复版本）"""
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]

        now = time.monotonic()
        while dq and dq[0] < now - period:
            dq.popleft()

        if len(dq) >= limit:
            wait_time = dq[0] + period - now + np.random.uniform(0, 0.02)
            logger.warning(f"Rate limit hit: {endpoint} - Waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)

        dq.append(now)
        METRICS['throughput'].inc()

    async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
        """修复的签名请求方法"""
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
        url = f"{REST_URL}{path}"

        for attempt in range(self.config.max_retries):
            try:
                await self._rate_limit_check(path.split('/')[-1])
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
                await asyncio.sleep(0.15 ** attempt)
        raise Exception("Max retries exceeded")

    async def sync_server_time(self):
        """精准时间同步"""
        time_diffs = []
        for _ in range(3):
            try:
                async with self.session.get(f"{REST_URL}/dapi/v1/time") as resp:
                    data = await resp.json()
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    time_diffs.append(server_time - local_time)
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.error(f"Time sync error: {str(e)}")

        if time_diffs:
            self._time_diff = int(np.median(time_diffs))
            if abs(self._time_diff) > 500:
                logger.warning(f"时间偏差较大: {self._time_diff}ms")

    async def manage_leverage(self):
        """杠杆管理"""
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'marginType': 'ISOLATED'
        }
        return await self._signed_request('POST', '/dapi/v1/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """极速K线获取"""
        params = {
            'symbol': SYMBOL,
            'interval': '1m',
            'limit': 100,
            'contractType': 'PERPETUAL'
        }
        try:
            data = await self._signed_request('GET', '/dapi/v1/klines', params)
            arr = np.empty((len(data), 6), dtype=np.float64)
            for i, candle in enumerate(data):
                arr[i] = [float(candle[j]) for j in [1, 2, 3, 4, 5, 7]]  # USDC合约数据结构
            return pd.DataFrame({
                'open': arr[:, 0],
                'high': arr[:, 1],
                'low': arr[:, 2],
                'close': arr[:, 3],
                'volume': arr[:, 4],
                'timestamp': arr[:, 5].astype('uint64')
            }).iloc[-100:]
        except Exception as e:
            logger.error(f"K线获取失败: {str(e)}")
            return None


class ETHUSDCTradingStrategy:
    """ETH/USDC高频策略（修复版）"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()

    async def execute(self):
        """策略主循环"""
        await self.client.manage_leverage()
        await self.client.sync_server_time()

        while True:
            try:
                start_time = time.monotonic()
                METRICS['memory'].set(psutil.Process().memory_info().rss // 1024 // 1024)

                # 获取市场数据
                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(0.2)
                    continue

                # 计算指标
                close_prices = df['close'].values.astype(np.float64)
                ema_fast = pd.Series(close_prices).ewm(span=self.config.ema_fast).mean().values
                ema_slow = pd.Series(close_prices).ewm(span=self.config.ema_slow).mean().values
                atr = self._calculate_atr(df)

                # 生成交易信号
                if ema_fast[-1] > ema_slow[-1] * 1.002:
                    stop_price = close_prices[-1] - (atr * self.config.volatility_multiplier)
                    await self._place_order('BUY', stop_price)
                elif ema_fast[-1] < ema_slow[-1] * 0.998:
                    stop_price = close_prices[-1] + (atr * self.config.volatility_multiplier)
                    await self._place_order('SELL', stop_price)

                # 更新延迟指标
                METRICS['latency'].set((time.monotonic() - start_time) * 1000)
                await asyncio.sleep(0.15)  # 150ms循环周期

            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """向量化ATR计算"""
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
            qty = min(QUANTITY, self.config.max_position - self.client.position)
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'STOP',
                'stopPrice': round(price, self.config.price_precision),
                'quantity': round(qty, self.config.quantity_precision),
                'timeInForce': 'GTC',
                'workingType': 'MARK_PRICE'
            }

            response = await self.client._signed_request('POST', '/dapi/v1/order', params)
            self.client.position += qty if side == 'BUY' else -qty
            METRICS['position'].set(self.client.position)
            logger.info(f"订单执行成功: {response}")
        except Exception as e:
            logger.error(f"订单执行失败: {str(e)}")


async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCTradingStrategy(client)
    try:
        await strategy.execute()
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())