#!/usr/bin/env python3
import uvloop

uvloop.install()

import asyncio
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from dotenv import load_dotenv
import os
import hmac
import hashlib
import urllib.parse
import logging
from typing import Optional

# 环境变量加载
load_dotenv()
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# ========== 全局配置 ==========
SYMBOL = 'ETHUSDC'
LEVERAGE = 10
REST_BASE_URL = 'https://fapi.binance.com'
PROXY = None  # 生产环境建议关闭代理

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
        self.connector = TCPConnector(limit=100, ssl=False, force_close=True)
        self.session = RetryClient(
            connector=self.connector,
            retry_options=ExponentialRetry(attempts=3),
            timeout=ClientTimeout(total=2)  # 高频交易需要更短超时
        )
        self.recv_window = 5000

    def _sign(self, params: dict) -> str:
        query = urllib.parse.urlencode(params)
        return hmac.new(
            SECRET_KEY.encode('utf-8'),
            query.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """获取优化后的K线数据"""
        url = f"{REST_BASE_URL}/fapi/v1/klines"
        params = {'symbol': SYMBOL, 'interval': '1m', 'limit': 100}

        try:
            async with self.session.get(url, params=params, proxy=PROXY) as resp:
                if resp.status != 200:
                    logger.error(f"K线接口异常: HTTP {resp.status}")
                    return None

                data = await resp.json()
                if len(data) < 10:
                    logger.warning("K线数据不足")
                    return None

                # 使用NumPy加速处理
                arr = np.array(data)[:, 1:6].astype(float)  # O,H,L,C,V
                return pd.DataFrame(arr, columns=['open', 'high', 'low', 'close', 'vol'])

        except Exception as e:
            logger.error(f"获取K线失败: {str(e)}")
            return None

    async def place_order(self, side: str, quantity: float, stop_price: float) -> dict:
        """下单接口优化版"""
        params = {
            'symbol': SYMBOL,
            'side': side.upper(),
            'type': 'STOP_MARKET',
            'quantity': round(quantity, 3),
            'stopPrice': round(stop_price, 2),
            'timestamp': int(time.time() * 1000),
            'recvWindow': self.recv_window
        }
        params['signature'] = self._sign(params)

        headers = {"X-MBX-APIKEY": API_KEY}
        url = f"{REST_BASE_URL}/fapi/v1/order"

        try:
            async with self.session.post(url, headers=headers, data=params) as resp:
                result = await resp.json()
                if result.get('status') == 'FILLED':
                    logger.info(f"订单成交: {side} {quantity} @ {result['avgPrice']}")
                return result
        except Exception as e:
            logger.error(f"下单失败: {str(e)}")
            return {}


class VolatilityStrategy:
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.position = 0.0
        self.atr_window = 14

    def calculate_features(self, df: pd.DataFrame) -> dict:
        """使用向量化计算优化指标"""
        high, low, close = df['high'], df['low'], df['close']

        # 计算ATR
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean().iloc[-1]

        # 计算EMA交叉
        ema_fast = close.ewm(span=9).mean().iloc[-1]
        ema_slow = close.ewm(span=21).mean().iloc[-1]

        return {
            'atr': atr,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'spread': (high - low).iloc[-1]
        }

    async def run(self):
        """策略主循环"""
        while True:
            try:
                df = await self.client.fetch_klines()
                if df is None:
                    await asyncio.sleep(1)
                    continue

                features = self.calculate_features(df)
                current_price = df['close'].iloc[-1]

                # 生成信号
                if features['ema_fast'] > features['ema_slow'] * 1.001:
                    stop_loss = current_price - features['atr'] * 1.5
                    await self.client.place_order('BUY', QUANTITY, stop_loss)
                elif features['ema_fast'] < features['ema_slow'] * 0.999:
                    stop_loss = current_price + features['atr'] * 1.5
                    await self.client.place_order('SELL', QUANTITY, stop_loss)

                await asyncio.sleep(5)  # 1分钟周期

            except Exception as e:
                logger.error(f"策略运行异常: {str(e)}")
                await asyncio.sleep(10)


async def main():
    client = BinanceHFTClient()
    strategy = VolatilityStrategy(client)

    try:
        await strategy.run()
    except KeyboardInterrupt:
        logger.info("用户终止操作")
    finally:
        await client.session.close()


if __name__ == "__main__":
    asyncio.run(main())