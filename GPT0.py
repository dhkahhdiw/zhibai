#!/usr/bin/env python3
# ETH/USDC高频交易引擎 v5.0 (Vultr生产环境稳定版)

import uvloop
uvloop.install()

import os
import re
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
        logging.FileHandler("/var/log/eth_usdc_hft.log", encoding="utf-8", mode='a')
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
                cleaned_msg = re.sub(r'[\x00-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202E]', '', str(e))
                logger.error(f"时间同步失败(重试 {retry+1}): {cleaned_msg}")
                await asyncio.sleep(0.5 * (retry + 1))
        raise Exception("无法同步服务器时间")

    async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
        # 确保参数字符串无隐藏字符
        params.update({
            "timestamp": int(time.time() * 1000 + self._time_diff),
            "recvWindow": self.recv_window
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
                logger.error(f"请求失败 (尝试 {attempt+1}): {str(e)}")
                await asyncio.sleep(0.1 * (2 ** attempt))
        raise Exception("超过最大重试次数")

    async def _rate_limit_check(self, endpoint: str):
        """动态速率限制管理"""
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

    async def manage_leverage(self):
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE, 'dualSidePosition': 'true'}
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """优化K线解析（向量化加速）"""
        try:
            data = await self._signed_request('GET', '/klines', {
                'symbol': SYMBOL,
                'interval': '1m',
                'limit': 100,
                'contractType': 'PERPETUAL'
            })
            dtype = [('open', 'f8'), ('high', 'f8'), ('low', 'f8'),
                     ('close', 'f8'), ('volume', 'f8'), ('timestamp', 'u8')]
            arr = np.array([(c[1], c[2], c[3], c[4], c[5], c[6]) for c in data], dtype=dtype)
            return pd.DataFrame(arr)[-100:]
        except Exception as e:
            logger.error(f"K线获取失败: {str(e)}")
            return None

@dataclass
class Signal:
    action: bool
    side: str
    tp: float
    sl: float

class ETHUSDCStrategy:
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()
        self._indicator_cache = defaultdict(lambda: None)

    async def execute(self):
        # 管理杠杆与时间同步
        await self.client.manage_leverage()
        await self.client.sync_server_time()
        while True:
            try:
                start_time = time.monotonic()
                df = await self.client.fetch_klines()
                if df is None or df.empty:
                    await asyncio.sleep(0.5)
                    continue

                close_prices = df['close'].values.astype(np.float32)
                high_prices = df['high'].values.astype(np.float32)
                low_prices = df['low'].values.astype(np.float32)

                # 若未计算ATR则计算一次
                if self._indicator_cache['atr'] is None:
                    self._indicator_cache['atr'] = self._vectorized_atr(high_prices, low_prices, close_prices)
                atr_array = self._indicator_cache['atr']
                atr_val = atr_array[-1]

                signals = await self.generate_signals(df, atr_val)
                current_price = close_prices[-1]
                # 根据方向调整价格（规避滑点）
                adjusted_price = current_price * (1 + self.config.slippage * (-1 if signals.side == 'BUY' else 1))
                if signals.action:
                    await self.place_limit_order(
                        side=signals.side,
                        limit_price=adjusted_price,
                        take_profit=signals.tp,
                        stop_loss=signals.sl
                    )
                METRICS['latency'].set((time.monotonic() - start_time) * 1000)
                await asyncio.sleep(max(0.2, 0.8 - (time.monotonic() - start_time)))
            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(5)

    def _vectorized_atr(self, high, low, close) -> np.ndarray:
        """向量化ATR计算"""
        prev_close = close[:-1]
        high_cut = high[1:]
        low_cut = low[1:]
        tr = np.maximum(high_cut - low_cut,
                        np.abs(high_cut - prev_close),
                        np.abs(low_cut - prev_close))
        atr = np.convolve(tr, np.ones(self.config.atr_window) / self.config.atr_window, mode='valid')
        return atr

    async def generate_signals(self, df: pd.DataFrame, atr_val: float) -> Signal:
        ema_fast = df['close'].ewm(span=self.config.ema_fast).mean().values
        ema_slow = df['close'].ewm(span=self.config.ema_slow).mean().values
        ema_cross = ema_fast[-1] > ema_slow[-1]
        volatility_ratio = atr_val / df['close'].values[-1]
        orderbook_imbalance = ((df['high'].values[-1] - df['low'].values[-1]) / df['close'].values[-1]) < 0.005
        action = ema_cross and (volatility_ratio > 0.02) and orderbook_imbalance
        return Signal(
            action=action,
            side='BUY' if action else 'SELL',
            tp=df['close'].values[-1] + atr_val * 2,
            sl=df['close'].values[-1] - atr_val * 1.5
        )

    async def place_limit_order(self, side: str, limit_price: float, tp: float, sl: float):
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
        try:
            order_resp = await self.client._signed_request('POST', '/order', params)
            logger.info(f"挂单 {side}@{formatted_price} 成功 | 止盈: {tp:.2f} 止损: {sl:.2f}")
            if abs(self.client.position) > self.config.max_position * 0.8:
                await self.client.manage_leverage()
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