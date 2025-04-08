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
from typing import Optional

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from dotenv import load_dotenv

# ========== 环境变量加载 ==========
load_dotenv()
API_KEY    = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
SYMBOL     = os.getenv('SYMBOL', 'ETHUSDC')
LEVERAGE   = int(os.getenv('LEVERAGE', 10))
QUANTITY   = float(os.getenv('QUANTITY', 0.01))
REST_URL   = 'https://fapi.binance.com'
PROXY      = None   # 生产环境建议关闭代理

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger('HFT-Bot')

class BinanceHFTClient:
    def __init__(self):
        # 并发连接数根据 1GB 内存适当调低
        self.connector = TCPConnector(limit=20, ssl=False, force_close=True)
        retry_opts = ExponentialRetry(attempts=3, start_timeout=0.5)
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(total=3)
        )
        self.recv_window = 5000

    def _sign(self, params: dict) -> str:
        query = urllib.parse.urlencode(params)
        return hmac.new(
            SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

    async def _post(self, path: str, params: dict) -> dict:
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = self.recv_window
        params['signature'] = self._sign(params)
        headers = {"X-MBX-APIKEY": API_KEY}
        url = REST_URL + path
        async with self.session.post(url, headers=headers, data=params, proxy=PROXY) as resp:
            data = await resp.json()
            if resp.status != 200:
                logger.error(f"POST {path} 异常 {resp.status}: {data}")
            return data

    async def set_leverage(self):
        """启动时设定杠杆"""
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE}
        result = await self._post('/fapi/v1/leverage', params)
        logger.info(f"Set leverage: {result}")

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """获取 1m K 线，返回 DataFrame"""
        url = f"{REST_URL}/fapi/v1/klines"
        params = {'symbol': SYMBOL, 'interval': '1m', 'limit': 100}
        try:
            async with self.session.get(url, params=params, proxy=PROXY) as resp:
                if resp.status != 200:
                    logger.error(f"K 线接口异常 HTTP {resp.status}")
                    return None
                data = await resp.json()
        except Exception as e:
            logger.error(f"获取 K 线失败: {e}")
            return None

        arr = np.array(data, dtype=float)[:, 1:6]  # O,H,L,C,V
        df = pd.DataFrame(arr, columns=['open','high','low','close','vol'])
        return df

    async def place_order(self, side: str, quantity: float, stop_price: float) -> dict:
        """下 STOP_MARKET 单"""
        params = {
            'symbol': SYMBOL,
            'side': side.upper(),
            'type': 'STOP_MARKET',
            'quantity': round(quantity, 3),
            'stopPrice': round(stop_price, 2),
        }
        result = await self._post('/fapi/v1/order', params)
        if result.get('status') == 'FILLED':
            logger.info(f"订单成交 {side} {quantity} @ {result.get('avgPrice')}")
        return result

    async def close(self):
        await self.session.close()

class VolatilityStrategy:
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.atr_window = 14

    def calculate_features(self, df: pd.DataFrame) -> dict:
        close = df['close'].to_numpy()
        high  = df['high'].to_numpy()
        low   = df['low'].to_numpy()

        # True Range
        prev_close = np.roll(close, 1)
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low  - prev_close)
        ])[1:]  # 丢掉第一行
        atr = tr[-self.atr_window:].mean()

        # EMA 快慢线（用 pandas 仅做一次）
        ema = df['close'].ewm
        ema_fast = ema(span=9).mean().iloc[-1]
        ema_slow = ema(span=21).mean().iloc[-1]

        return {'atr': atr, 'ema_fast': ema_fast, 'ema_slow': ema_slow}

    async def run(self):
        # 启动时设定杠杆
        await self.client.set_leverage()

        while True:
            try:
                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(1)
                    continue

                feat = self.calculate_features(df)
                price = df['close'].iloc[-1]

                # 买入信号
                if feat['ema_fast'] > feat['ema_slow'] * 1.001:
                    stop = price - feat['atr'] * 1.5
                    await self.client.place_order('BUY', QUANTITY, stop)
                # 卖出信号
                elif feat['ema_fast'] < feat['ema_slow'] * 0.999:
                    stop = price + feat['atr'] * 1.5
                    await self.client.place_order('SELL', QUANTITY, stop)

                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"策略异常: {e}")
                await asyncio.sleep(10)

async def main():
    client = BinanceHFTClient()
    strat  = VolatilityStrategy(client)
    try:
        await strat.run()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())