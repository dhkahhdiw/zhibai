#!/usr/bin/env python3
# 高频交易核心引擎 v2.1 完整修正版本（Vultr 1GB内存优化版）

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
from collections import deque

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server
import psutil  # 用于内存监控

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
TCP_FAST_OPEN = True  # 需内核支持 net.ipv4.tcp_fastopen=3

# ========== 监控指标初始化 ==========
MEM_USAGE = Gauge('memory_usage', 'Process memory usage (MB)')
ORDER_LATENCY = Gauge('order_latency', 'Order execution latency (ms)')
start_http_server(8000)  # 监控端口

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hft.log", encoding="utf-8", mode='a', delay=True)
    ]
)
logger = logging.getLogger('HFT-Core')


class BinanceHFTClient:
    """高频交易客户端（含VPS优化适配）"""

    def __init__(self):
        # 网络层优化配置
        self.connector = TCPConnector(
            limit=8,  # 针对 1GB 内存的优化
            ssl=False,
            force_close=True,
            tcp_fast_open=TCP_FAST_OPEN,
            keepalive_timeout=10
        )
        retry_opts = ExponentialRetry(attempts=3, start_timeout=0.5, max_timeout=2)
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=1.5)  # 总超时缩短至1.5秒
        )
        self.recv_window = 5000
        self.request_timestamps: Dict[str, Deque[float]] = {
            'klines': deque(maxlen=RATE_LIMIT['klines'] + 10),
            'orders': deque(maxlen=RATE_LIMIT['orders'] + 10)
        }
        self._time_diff = 0  # 本地与服务器时间差

    def _hmac_sign(self, params: Dict) -> str:
        """生成 HMAC 签名"""
        query = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(
            SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

    async def _post(self, path: str, params: Dict) -> Dict:
        """带时间校准的 POST 请求"""
        params.update({
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': self.recv_window,
            'signature': self._hmac_sign(params)
        })

        endpoint = path.split('/')[-1]
        self._rate_limit_check(endpoint)

        headers = {"X-MBX-APIKEY": API_KEY}
        async with self.session.post(REST_URL + path, headers=headers, data=params) as resp:
            response = await resp.json()
            if resp.status != 200:
                logger.error(f"POST {path} 异常 {resp.status}: {response}")
                raise Exception(f"API Error: {response}")
            return response

    async def _get(self, path: str, params: Dict) -> Dict:
        """GET 请求封装（适配 Binance 官方要求）"""
        self._rate_limit_check('klines')
        async with self.session.get(REST_URL + path, params=params) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error(f"GET {path} 异常 {resp.status}: {data}")
                raise Exception(f"API Error: {data}")
            return data

    async def place_order(self, side: str, quantity: float, stop_price: float) -> Dict:
        """
        下单接口：调用 STOP_MARKET 单，适用于 Binance 期货
        注：实际下单类型和参数请参照 Binance 官方文档进行调整
        """
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'STOP_MARKET',
            'stopPrice': round(stop_price, 2),
            'closePosition': False,
            'quantity': quantity,
            'timeInForce': 'GTC'
        }
        return await self._post('/fapi/v1/order', params)

    async def sync_server_time(self):
        """时间同步校准（每分钟执行）"""
        async with self.session.get(f"{REST_URL}/fapi/v1/time") as resp:
            data = await resp.json()
            server_time = data['serverTime']
            self._time_diff = server_time - int(time.time() * 1000)
            logger.info(f"时间校准完成, 偏差: {self._time_diff}ms")

    def _rate_limit_check(self, endpoint: str):
        """简单速率限制控制（基于60秒窗口）"""
        now = time.time()
        window = 60  # 60秒窗口

        # 清理过期时间戳
        while self.request_timestamps[endpoint] and self.request_timestamps[endpoint][0] < now - window:
            self.request_timestamps[endpoint].popleft()

        if len(self.request_timestamps[endpoint]) >= RATE_LIMIT[endpoint]:
            sleep_time = (self.request_timestamps[endpoint][0] + window) - now
            logger.warning(f"速率限制触发: {endpoint} 休眠 {sleep_time:.2f}s")
            time.sleep(max(sleep_time, 0))
        self.request_timestamps[endpoint].append(now)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """获取 K 线数据（内存敏感型优化）"""
        params = {
            'symbol': SYMBOL,
            'interval': '1m',
            'limit': 100
        }
        try:
            data = await self._get('/fapi/v1/klines', params)
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ]).astype({
                'open': 'float32', 'high': 'float32',
                'low': 'float32', 'close': 'float32'
            })
            return df.iloc[-100:]  # 仅保留最近100条记录
        except Exception as e:
            logger.error(f"获取 K 线失败: {str(e)}")
            return None

    async def reset_session_periodically(self):
        """定时重建 session，防止长时间运行内存泄漏（可选）"""
        while True:
            await asyncio.sleep(600)  # 每10分钟重建一次
            logger.info("重建 aiohttp session")
            await self.session.close()
            self.session = RetryClient(
                connector=self.connector,
                retry_options=ExponentialRetry(attempts=3, start_timeout=0.5, max_timeout=2),
                timeout=ClientTimeout(total=1.5)
            )


class OrderManager:
    """订单状态追踪器"""

    def __init__(self):
        self.active_orders = {}

    async def sync_orders(self, client: BinanceHFTClient):
        """订单状态同步"""
        params = {'symbol': SYMBOL}
        try:
            data = await client._post('/fapi/v1/openOrders', params)
            self.active_orders = {o['orderId']: o for o in data}
        except Exception as e:
            logger.error(f"同步订单失败: {str(e)}")


class VolatilityStrategy:
    """动态波动率策略（含内存优化）"""

    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.atr_window = 14
        self.order_manager = OrderManager()

    def _vectorized_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """向量化 ATR 计算"""
        prev_close = np.roll(close, 1)
        tr = np.maximum(high - low,
                        np.abs(high - prev_close),
                        np.abs(low - prev_close))[1:]
        return tr[-self.atr_window:].mean()

    def calculate_features(self, df: pd.DataFrame) -> Dict:
        """计算策略所需特征（内存优化版）"""
        close = df['close'].to_numpy(dtype='float32')
        high = df['high'].to_numpy(dtype='float32')
        low = df['low'].to_numpy(dtype='float32')

        # 动态波动率因子计算
        volatility_ratio = (high[-100:].max() - low[-100:].min()) / close[-100:].mean()
        atr_multiplier = 1.5 + volatility_ratio * 0.5

        # 简单 EMA 计算（用均值近似）
        ema_fast = close[-9:].mean()
        ema_slow = close[-21:].mean()

        return {
            'atr': self._vectorized_atr(high, low, close),
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'multiplier': atr_multiplier
        }

    async def run(self):
        """策略主循环（含熔断机制）"""
        await self.client.set_leverage()
        await self.client.sync_server_time()  # 初次时间同步

        # 启动定时任务：重建 session、同步订单、定时同步时间
        asyncio.create_task(self.client.reset_session_periodically())
        asyncio.create_task(self._schedule_tasks())

        while True:
            try:
                # 内存监控：使用 psutil 获取当前进程内存使用（MB）
                MEM_USAGE.set(psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024)
                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(0.5)
                    continue

                feat = self.calculate_features(df)
                price = df['close'].iloc[-1]

                # 根据 EMA 快慢线判断下单信号（带滑点补偿）
                if feat['ema_fast'] > feat['ema_slow'] * 1.001:
                    stop = price - feat['atr'] * feat['multiplier'] * 1.0005
                    await self._safe_order('BUY', stop)
                elif feat['ema_fast'] < feat['ema_slow'] * 0.999:
                    stop = price + feat['atr'] * feat['multiplier'] * 1.0005
                    await self._safe_order('SELL', stop)

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"策略异常: {str(e)}")
                await asyncio.sleep(5)

    async def _schedule_tasks(self):
        """定时任务调度，每分钟同步时间和订单状态"""
        while True:
            try:
                await self.client.sync_server_time()
                await self.order_manager.sync_orders(self.client)
            except Exception as e:
                logger.error(f"定时任务执行失败: {str(e)}")
            await asyncio.sleep(60)

    async def _safe_order(self, side: str, stop_price: float):
        """熔断型订单执行，记录下单延迟"""
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
            if "Signature" in str(e):
                await self.client.sync_server_time()  # 签名错误时强制同步时间

    async def set_leverage(self):
        """杠杆设置（带重试机制）"""
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE
        }
        return await self.client._post('/fapi/v1/leverage', params)


async def main():
    client = BinanceHFTClient()
    strat = VolatilityStrategy(client)
    try:
        await strat.run()
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())