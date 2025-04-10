#!/usr/bin/env python3
# ETH/USDC高频交易引擎 v4.2 (全问题修复版)

import uvloop
uvloop.install()

import asyncio
import os
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
from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server
import psutil

# ==================== 环境变量加载 ====================
_env_path = '/root/zhibai/.env'
if not os.path.exists(_env_path):
    raise FileNotFoundError(f"未找到环境文件: {_env_path}")
load_dotenv(_env_path)

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = os.getenv('TRADING_PAIR', 'ETHUSDC').strip()  # 对于 USDC-M Futures 通常为 "ETHUSDC"
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.06))
# 使用 Binance 官方基础 URL
REST_URL = 'https://api.binance.com'

if not API_KEY or not SECRET_KEY:
    raise Exception("请在 /root/zhibai/.env 中正确配置 BINANCE_API_KEY 与 BINANCE_SECRET_KEY")
# 此处示例要求密钥长度为64字符（请根据实际情况调整）
if len(API_KEY) != 64 or len(SECRET_KEY) != 64:
    raise ValueError("API密钥格式错误，应为64位字符，请检查 BINANCE_API_KEY 与 BINANCE_SECRET_KEY")

# ==================== 高频参数 ====================
RATE_LIMITS = {
    'klines': (60, 5),
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
    level=logging.INFO,  # 若需更详细调试信息，可将 level 设置为 DEBUG
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/eth_usdc_hft.log", encoding="utf-8", mode='a')
    ]
)
logger = logging.getLogger('ETH-USDC-HFT')
DEBUG = False  # 调试开关

# ==================== 交易参数配置 ====================
@dataclass
class TradingConfig:
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 2.5
    max_retries: int = 3
    order_timeout: float = 0.5      # 500ms连接超时
    network_timeout: float = 2.0    # 2秒总超时
    price_precision: int = int(os.getenv('PRICE_PRECISION', '2'))
    quantity_precision: int = int(os.getenv('QUANTITY_PRECISION', '3'))
    max_position: float = 10.0

# ==================== Binance高频交易客户端 ====================
class BinanceHFTClient:
    def __init__(self):
        self.config = TradingConfig()
        # 使用默认解析器，强制使用IPv4
        self.connector = TCPConnector(
            limit=128,
            ssl=True,
            ttl_dns_cache=60,
            force_close=True,
            family=AF_INET
        )
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
                # 调整字段下标请参考 Binance USDC-M Futures 最新文档，示例中使用 [1, 2, 3, 4, 5, 7]
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

class ETHUSDCStrategy:
    """策略交易实现（全问题修复版）"""
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.config = TradingConfig()

    async def execute(self):
        await self.client.manage_leverage()
        await self.client.sync_server_time()
        while True:
            try:
                cycle_start = time.monotonic()
                METRICS['memory'].set(psutil.Process().memory_info().rss // 1024 // 1024)
                df = await self.client.fetch_klines()
                if df is None or df.empty:
                    await asyncio.sleep(0.15)
                    continue
                close_prices = df['close'].values.astype(np.float64)
                ema_fast = pd.Series(close_prices).ewm(span=self.config.ema_fast, adjust=False).mean().values
                ema_slow = pd.Series(close_prices).ewm(span=self.config.ema_slow, adjust=False).mean().values
                atr = self._calculate_atr(df)
                if ema_fast[-1] > ema_slow[-1] * 1.003:
                    stop_price = close_prices[-1] - (atr * self.config.volatility_multiplier)
                    formatted_stop = float(f"{stop_price:.{self.config.price_precision}f}")
                    await self._execute_order('BUY', formatted_stop)
                elif ema_fast[-1] < ema_slow[-1] * 0.997:
                    stop_price = close_prices[-1] + (atr * self.config.volatility_multiplier)
                    formatted_stop = float(f"{stop_price:.{self.config.price_precision}f}")
                    await self._execute_order('SELL', formatted_stop)
                cycle_time = (time.monotonic() - cycle_start) * 1000
                METRICS['latency'].set(cycle_time)
                await asyncio.sleep(max(0.05, 0.15 - cycle_time / 1000))
            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_atr(self, df: pd.DataFrame) -> float:
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
        try:
            available = self.config.max_position - abs(self.client.position)
            qty = min(QUANTITY, available)
            if qty <= 0:
                logger.warning("已达最大持仓限制")
                return
            formatted_qty = float(f"{qty:.{self.config.quantity_precision}f}")
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'STOP',  # 如官方要求使用 'STOP_MARKET'，请调整此处
                'stopPrice': float(f"{price:.{self.config.price_precision}f}"),
                'quantity': formatted_qty,
                'timeInForce': 'GTC',
                'workingType': 'MARK_PRICE',
                'priceProtect': 'true'
            }
            response = await self.client._signed_request('POST', '/order', params)
            if side == 'BUY':
                self.client.position += qty
            else:
                self.client.position -= qty
            METRICS['position'].set(self.client.position)
            logger.info(f"订单成功: {response}")
        except Exception as e:
            logger.error(f"订单失败: {str(e)}")
            METRICS['errors'].inc()
            if "Signature" in str(e):
                await self.client.sync_server_time()

async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        await strategy.execute()
    finally:
        await client.session.close()

if __name__ == "__main__":
    asyncio.run(main())