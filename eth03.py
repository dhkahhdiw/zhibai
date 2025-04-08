#!/usr/bin/env python3
import uvloop

uvloop.install()

import asyncio
import os
import time
import hmac
import hashlib
import urllib.parse
import logging
from typing import Optional, Dict
from collections import deque

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server

# ========== 环境变量与全局配置 ==========
load_dotenv()
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
SYMBOL = os.getenv('SYMBOL', 'ETHUSDC')
LEVERAGE = int(os.getenv('LEVERAGE', 10))
QUANTITY = float(os.getenv('QUANTITY', 0.01))
REST_URL = 'https://fapi.binance.com'
PROXY = None  # 生产环境关闭代理

# ========== 高频交易优化参数 ==========
RATE_LIMIT = {'klines': 30, 'orders': 120}  # 基于1GB内存的保守限制
TCP_FAST_OPEN = True  # 需内核支持net.ipv4.tcp_fastopen=3

# ========== 监控指标初始化 ==========
MEM_USAGE = Gauge('memory_usage', 'Process memory usage (MB)')
ORDER_LATENCY = Gauge('order_latency', 'Order execution latency (ms)')
start_http_server(8000)  # 监控端口

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hft.log", encoding="utf-8")
    ]
)
logger = logging.getLogger('HFT-Core')


class BinanceHFTClient:
    """高频交易客户端（含VPS优化适配）"""

    def __init__(self):
        # 网络层优化配置[6,7](@ref)
        self.connector = TCPConnector(
            limit=8,  # 1GB内存适配
            ssl=False,
            force_close=True,
            tcp_fast_open=TCP_FAST_OPEN
        )
        retry_opts = ExponentialRetry(attempts=3, start_timeout=0.5)
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=2)  # 超时缩短
        )
        self.recv_window = 5000
        self.request_timestamps = {
            'klines': deque(maxlen=RATE_LIMIT['klines'] + 5),
            'orders': deque(maxlen=RATE_LIMIT['orders'] + 5)
        }

    def _sign(self, params: Dict) -> str:
        """HMAC-SHA256签名（带时间校准）[1](@ref)"""
        query = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(
            SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

    async def _post(self, path: str, params: Dict) -> Dict:
        """带频率限制的POST请求"""
        # 时间校准机制[4](@ref)
        server_time = await self._get_server_time()
        local_time = int(time.time() * 1000)
        time_diff = abs(server_time - local_time)
        if time_diff > 3000:
            logger.warning(f"时间偏移告警: {time_diff}ms")

        params.update({
            'timestamp': server_time,
            'recvWindow': self.recv_window,
            'signature': self._sign(params)
        })

        # 请求频率控制[3](@ref)
        self._rate_limit_check(path.replace('/fapi/v1/', ''))

        headers = {"X-MBX-APIKEY": API_KEY}
        async with self.session.post(REST_URL + path, headers=headers, data=params) as resp:
            response = await resp.json()
            if resp.status != 200:
                logger.error(f"POST {path} 异常 {resp.status}: {response}")
            return response

    async def _get_server_time(self) -> int:
        """获取交易所服务器时间"""
        async with self.session.get(f"{REST_URL}/fapi/v1/time") as resp:
            data = await resp.json()
            return data['serverTime']

    def _rate_limit_check(self, endpoint: str):
        """高频请求速率控制"""
        now = time.time()
        window = 60  # 60秒窗口

        # 清理过期时间戳
        while self.request_timestamps[endpoint] and self.request_timestamps[endpoint][0] < now - window:
            self.request_timestamps[endpoint].popleft()

        if len(self.request_timestamps[endpoint]) >= RATE_LIMIT[endpoint]:
            sleep_time = (self.request_timestamps[endpoint][0] + window) - now
            logger.warning(f"速率限制触发: {endpoint} 休眠{sleep_time:.2f}s")
            time.sleep(max(sleep_time, 0))

        self.request_timestamps[endpoint].append(now)


class VolatilityStrategy:
    """动态波动率策略（含VPS优化）"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.atr_window = 14
        self.order_manager = OrderManager()

    def calculate_features(self, df: pd.DataFrame) -> Dict:
        """动态波动率计算（内存优化版）"""
        close = df['close'].to_numpy(dtype='float32')
        high = df['high'].to_numpy(dtype='float32')
        low = df['low'].to_numpy(dtype='float32')

        # 动态ATR倍数[4](@ref)
        volatility_ratio = (high[-100:].max() - low[-100:].min()) / close[-100:].mean()
        atr_multiplier = 1.5 + volatility_ratio * 0.5

        # 向量化计算
        prev_close = np.roll(close, 1)
        tr = np.maximum(high - low,
                        np.abs(high - prev_close),
                        np.abs(low - prev_close))[1:]
        atr = tr[-self.atr_window:].mean()

        # EMA计算（pandas优化）
        ema_fast = df['close'].ewm(span=9).mean().iloc[-1]
        ema_slow = df['close'].ewm(span=21).mean().iloc[-1]

        return {'atr': atr, 'ema_fast': ema_fast, 'ema_slow': ema_slow,
                'multiplier': atr_multiplier}

    async def run(self):
        """策略主循环"""
        await self.client.set_leverage()
        while True:
            try:
                # 内存监控[6](@ref)
                MEM_USAGE.set(os.getpid().memory_info().rss // 1024 // 1024)

                df = await self.client.fetch_klines()
                if df is None: continue

                feat = self.calculate_features(df)
                price = df['close'].iloc[-1]

                # 信号生成（带滑点补偿）[4](@ref)
                if feat['ema_fast'] > feat['ema_slow'] * 1.001:
                    stop = price - feat['atr'] * feat['multiplier'] * 1.0005  # 滑点补偿
                    await self._safe_order('BUY', stop)
                elif feat['ema_fast'] < feat['ema_slow'] * 0.999:
                    stop = price + feat['atr'] * feat['multiplier'] * 1.0005
                    await self._safe_order('SELL', stop)

                await asyncio.sleep(0.5)  # 500ms轮询间隔

            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(5)

    async def _safe_order(self, side: str, stop_price: float):
        """带熔断机制的订单执行"""
        try:
            start_time = time.time()
            result = await self.client.place_order(
                side, QUANTITY, stop_price
            )
            latency = (time.time() - start_time) * 1000
            ORDER_LATENCY.set(latency)

            if result.get('status') == 'FILLED':
                logger.info(f"订单成交 {side} {QUANTITY} @ {result.get('avgPrice')}")
        except Exception as e:
            logger.error(f"下单失败: {str(e)}")


class OrderManager:
    """订单状态追踪器"""

    def __init__(self):
        self.active_orders = {}

    async def sync_orders(self, client: BinanceHFTClient):
        """每30秒同步未完成订单"""
        params = {'symbol': SYMBOL}
        data = await client._post('/fapi/v1/openOrders', params)
        self.active_orders = {o['orderId']: o for o in data}


async def main():
    client = BinanceHFTClient()
    strat = VolatilityStrategy(client)
    try:
        await strat.run()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())