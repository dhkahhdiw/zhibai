#!/usr/bin/env python3
# 高频交易核心引擎 v2.1 完整修正版本（Vultr 1GB 内存优化版）

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

# ========== 高频交易优化参数 ==========
RATE_LIMIT = {'klines': 30, 'orders': 120}

# ========== 监控指标初始化 ==========
MEM_USAGE = Gauge('memory_usage', 'Process memory usage (MB)')
ORDER_LATENCY = Gauge('order_latency', 'Order execution latency (ms)')
start_http_server(8000)

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
    """高频交易客户端（含 VPS 优化）"""

    def __init__(self):
        # 移除 keepalive_timeout，保留 force_close=True 避免冲突  [oai_citation:2‡Client Reference — aiohttp 3.11.16 documentation](https://docs.aiohttp.org/en/stable/client_reference.html?utm_source=chatgpt.com)
        self.connector = TCPConnector(
            limit=8,
            ssl=False,
            force_close=True
        )
        retry_opts = ExponentialRetry(attempts=3, start_timeout=0.5, max_timeout=2)
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=1.5)
        )
        self.recv_window = 5000
        self.request_timestamps: Dict[str, Deque[float]] = {
            'klines': deque(maxlen=RATE_LIMIT['klines'] + 10),
            'orders': deque(maxlen=RATE_LIMIT['orders'] + 10)
        }
        self._time_diff = 0  # 本地与服务器时间差

    def _hmac_sign(self, params: Dict) -> str:
        query = urllib.parse.urlencode(sorted(params.items()))
        return hmac.new(
            SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

    async def _rate_limit_check(self, endpoint: str):
        """异步速率限制（60s 窗口）"""
        now = time.time()
        window = 60
        dq = self.request_timestamps[endpoint]
        while dq and dq[0] < now - window:
            dq.popleft()
        if len(dq) >= RATE_LIMIT[endpoint]:
            wait = dq[0] + window - now
            logger.warning(f"速率限制触发: {endpoint}，等待 {wait:.2f}s")
            await asyncio.sleep(wait)
        dq.append(time.time())

    async def _post(self, path: str, params: Dict) -> Dict:
        params.update({
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': self.recv_window,
            'signature': self._hmac_sign(params)
        })
        await self._rate_limit_check(path.split('/')[-1])
        headers = {"X-MBX-APIKEY": API_KEY}
        async with self.session.post(REST_URL + path, headers=headers, data=params) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error(f"POST {path} 错误 {resp.status}: {data}")
                raise Exception(f"API Error: {data}")
            return data

    async def _get(self, path: str, params: Dict) -> Dict:
        """GET /fapi/v1/klines 等  [oai_citation:3‡Kline Candlestick Data | Binance Open Platform](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data?utm_source=chatgpt.com)"""
        await self._rate_limit_check('klines')
        async with self.session.get(REST_URL + path, params=params) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error(f"GET {path} 错误 {resp.status}: {data}")
                raise Exception(f"API Error: {data}")
            return data

    async def place_order(self, side: str, quantity: float, stop_price: float) -> Dict:
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
        async with self.session.get(f"{REST_URL}/fapi/v1/time") as resp:
            data = await resp.json()
            self._time_diff = data['serverTime'] - int(time.time() * 1000)
            logger.info(f"时间校准: 偏差 {self._time_diff}ms")

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        params = {'symbol': SYMBOL, 'interval': '1m', 'limit': 100}
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
            return df.iloc[-100:]
        except Exception as e:
            logger.error(f"K 线获取失败: {e}")
            return None

    async def reset_session_periodically(self):
        while True:
            await asyncio.sleep(600)
            logger.info("重建 aiohttp session")
            await self.session.close()
            retry_opts = ExponentialRetry(attempts=3, start_timeout=0.5, max_timeout=2)
            self.session = RetryClient(
                connector=self.connector,
                retry_options=retry_opts,
                timeout=ClientTimeout(total=1.5)
            )


class OrderManager:
    def __init__(self):
        self.active_orders = {}

    async def sync_orders(self, client: BinanceHFTClient):
        try:
            data = await client._post('/fapi/v1/openOrders', {'symbol': SYMBOL})
            self.active_orders = {o['orderId']: o for o in data}
        except Exception as e:
            logger.error(f"订单同步失败: {e}")


class VolatilityStrategy:
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.atr_window = 14
        self.order_manager = OrderManager()

    def _vectorized_atr(self, high, low, close) -> float:
        prev = np.roll(close, 1)
        tr = np.maximum(high - low, np.abs(high - prev), np.abs(low - prev))[1:]
        return tr[-self.atr_window:].mean()

    def calculate_features(self, df: pd.DataFrame) -> Dict:
        close = df['close'].to_numpy(dtype='float32')
        high = df['high'].to_numpy(dtype='float32')
        low = df['low'].to_numpy(dtype='float32')
        vol_ratio = (high[-100:].max() - low[-100:].min()) / close[-100:].mean()
        atr_mul = 1.5 + vol_ratio * 0.5
        ema_f = close[-9:].mean()
        ema_s = close[-21:].mean()
        return {'atr': self._vectorized_atr(high, low, close),
                'ema_fast': ema_f, 'ema_slow': ema_s, 'multiplier': atr_mul}

    async def run(self):
        await self.client._post('/fapi/v1/leverage', {'symbol': SYMBOL, 'leverage': LEVERAGE})
        await self.client.sync_server_time()
        asyncio.create_task(self.client.reset_session_periodically())
        asyncio.create_task(self._schedule_tasks())

        while True:
            try:
                MEM_USAGE.set(psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024)
                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(0.5)
                    continue

                feat = self.calculate_features(df)
                price = df['close'].iloc[-1]
                if feat['ema_fast'] > feat['ema_slow'] * 1.001:
                    stop = price - feat['atr'] * feat['multiplier'] * 1.0005
                    await self._safe_order('BUY', stop)
                elif feat['ema_fast'] < feat['ema_slow'] * 0.999:
                    stop = price + feat['atr'] * feat['multiplier'] * 1.0005
                    await self._safe_order('SELL', stop)

                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"策略异常: {e}")
                await asyncio.sleep(5)

    async def _schedule_tasks(self):
        while True:
            try:
                await self.client.sync_server_time()
                await self.order_manager.sync_orders(self.client)
            except Exception as e:
                logger.error(f"定时任务失败: {e}")
            await asyncio.sleep(60)

    async def _safe_order(self, side: str, stop_price: float):
        try:
            start = time.time()
            res = await self.client.place_order(side, QUANTITY, stop_price)
            ORDER_LATENCY.set((time.time() - start) * 1000)
            if res.get('status') == 'FILLED':
                logger.info(f"订单成交 {side} {QUANTITY} @ {res.get('avgPrice')}")
        except Exception as e:
            logger.error(f"下单失败: {e}")
            if "Signature" in str(e):
                await self.client.sync_server_time()

async def main():
    client = BinanceHFTClient()
    strat = VolatilityStrategy(client)
    try:
        await strat.run()
    finally:
        await client.session.close()

if __name__ == "__main__":
    asyncio.run(main())