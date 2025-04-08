#!/usr/bin/env python3
import asyncio
import uvloop  # 高性能事件循环
uvloop.install()

import logging
import hmac
import hashlib
import urllib.parse
import time
import math
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry
from config import API_KEY, SECRET_KEY
from scipy.stats import norm  # 如需 PSR

# ========== 系统配置 ==========
SYMBOL = 'ETHUSDC'
LEVERAGE = 10
QUANTITY = 0.06
REST_BASE_URL = 'https://fapi.binance.com'
PROXY = None  # Linux 上通常不走本地代理，若需要可填写 http://127.0.0.1:7897

SL_PERCENT = 0.02
TP_PERCENT = 0.02

KLINE_INTERVAL = '1h'
MAX_KLINES = 100

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger('RestAPI-Bot')


class BinanceRESTAPI:
    def __init__(self):
        connector = TCPConnector(limit=100, ssl=True)  # 启用 SSL 验证
        self.session = RetryClient(
            raise_for_status=False,
            retry_options=ExponentialRetry(attempts=3),
            connector=connector,
            timeout=ClientTimeout(total=30)
        )

    def _sign(self, params: dict) -> str:
        query = urllib.parse.urlencode(params)
        return hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()

    async def get_kline_data(self):
        url = f"{REST_BASE_URL}/fapi/v1/klines"
        params = {'symbol': SYMBOL, 'interval': KLINE_INTERVAL, 'limit': MAX_KLINES}
        try:
            async with self.session.get(url, params=params, proxy=PROXY) as resp:
                data = await resp.json()
                if not isinstance(data, list) or len(data) < 2:
                    logger.warning(f"{KLINE_INTERVAL} Kline 数据异常: {data}")
                    return []
                return data
        except Exception as e:
            logger.error(f"获取 Kline 数据失败: {e}")
            return []

    async def place_order(self, params: dict):
        url = f"{REST_BASE_URL}/fapi/v1/order"
        params['signature'] = self._sign(params)
        headers = {"X-MBX-APIKEY": API_KEY}
        try:
            async with self.session.post(url, headers=headers, data=params, proxy=PROXY) as resp:
                result = await resp.json()
                logger.info(f"订单响应: {result}")
                return result
        except Exception as e:
            logger.error(f"下单异常: {e}")
            return {}

    async def close(self):
        await self.session.close()


def calculate_rsi(df: pd.DataFrame, period=14) -> pd.Series:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calculate_atr(df: pd.DataFrame, period=14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().fillna(0)


class TradingStrategy:
    def __init__(self, api: BinanceRESTAPI):
        self.api = api

    async def run(self):
        while True:
            data = await self.api.get_kline_data()
            if not data:
                await asyncio.sleep(5)
                continue

            df = pd.DataFrame(data, columns=[
                'ts', 'open', 'high', 'low', 'close', 'vol',
                'cts', 'qav', 'nt', 'tbbav', 'tbqav', 'ignore'
            ]).astype(float)
            df['rsi'] = calculate_rsi(df)
            df['atr'] = calculate_atr(df)

            price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            rsi = df['rsi'].iloc[-1]

            # 示例信号：RSI<30买入，>70卖出
            if rsi < 30:
                sl = round(price - atr * 1.5, 2)
                tp = round(price + atr * 1.5, 2)
                params = {
                    'symbol': SYMBOL, 'side': 'BUY', 'type': 'MARKET',
                    'quantity': QUANTITY, 'stopPrice': sl, 'timeInForce': 'GTC',
                    'timestamp': int(time.time()*1000)
                }
                await self.api.place_order(params)
                logger.info(f"BUY 市价单, SL={sl}, TP={tp}")
            elif rsi > 70:
                sl = round(price + atr * 1.5, 2)
                tp = round(price - atr * 1.5, 2)
                params = {
                    'symbol': SYMBOL, 'side': 'SELL', 'type': 'MARKET',
                    'quantity': QUANTITY, 'stopPrice': sl, 'timeInForce': 'GTC',
                    'timestamp': int(time.time()*1000)
                }
                await self.api.place_order(params)
                logger.info(f"SELL 市价单, SL={sl}, TP={tp}")

            await asyncio.sleep(60)


async def main():
    api = BinanceRESTAPI()
    strat = TradingStrategy(api)
    try:
        await strat.run()
    except KeyboardInterrupt:
        logger.info("退出")
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())