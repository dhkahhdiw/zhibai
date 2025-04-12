#!/usr/bin/env python3
"""
ETH/USDT 高频交易引擎 v8.1（基于NautilusTrader架构思想的优化版）
"""
import uvloop

uvloop.install()

import os
import asyncio
import time
import hmac
import hashlib
import urllib.parse
import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector
from aiohttp_retry import RetryClient, ExponentialRetry

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/var/log/eth_usdt_hft.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('ETH-HFT')


# 环境配置
class EnvConfig:
    API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
    SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
    SYMBOL = 'ETHUSDT'
    LEVERAGE = 50
    BASE_URL = 'https://fapi.binance.com'


# 交易参数
class TradingParameters:
    BULL_LONG_QTY = Decimal('0.36')
    BULL_SHORT_QTY = Decimal('0.18')
    BEAR_LONG_QTY = Decimal('0.18')
    BEAR_SHORT_QTY = Decimal('0.36')
    ORDER_QUANTITY = Decimal('0.12')
    PRICE_PRECISION = 2
    QTY_PRECISION = 3
    MAX_RETRIES = 5
    RATE_LIMITS = {
        'klines': (60, 5),
        'orders': (300, 10),
        'leverage': (30, 60)
    }


# 异步HTTP客户端
class BinanceHFTClient:
    def __init__(self):
        self.connector = TCPConnector(
            limit=1000,
            ttl_dns_cache=300,
            force_close=True,
            ssl=False
        )
        self.retry_options = ExponentialRetry(
            attempts=TradingParameters.MAX_RETRIES,
            start_timeout=0.5,
            max_timeout=3.0,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={asyncio.TimeoutError, ConnectionError}
        )
        self.session = RetryClient(
            connector=self.connector,
            retry_options=self.retry_options,
            timeout=ClientTimeout(total=5)
        )
        self.time_diff = 0

    async def sync_time(self):
        url = f"{EnvConfig.BASE_URL}/fapi/v1/time"
        async with self.session.get(url) as resp:
            data = await resp.json()
            self.time_diff = data['serverTime'] - int(time.time() * 1000)

    async def signed_request(self, method: str, endpoint: str, params: dict) -> dict:
        params = {
            **params,
            'timestamp': int(time.time() * 1000 + self.time_diff),
            'recvWindow': 6000
        }
        query = urllib.parse.urlencode(sorted(params.items()))
        signature = hmac.new(
            EnvConfig.SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "X-MBX-APIKEY": EnvConfig.API_KEY,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        url = f"{EnvConfig.BASE_URL}/fapi/v1{endpoint}?{query}&signature={signature}"

        for attempt in range(TradingParameters.MAX_RETRIES + 1):
            try:
                async with self.session.request(method, url, headers=headers) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get('Retry-After', 2 ** attempt))
                        await asyncio.sleep(retry_after)
                        continue
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {str(e)}")
                await asyncio.sleep(2 ** attempt)
        raise Exception("Max retries exceeded")


# 策略引擎
class HFTStrategyEngine:
    def __init__(self, client: BinanceHFTClient):
        self.client = client
        self.position = Decimal('0')
        self.daily_pnl = Decimal('0')
        self.consecutive_losses = 0

    async def supertrend_signal(self) -> str:
        df = await self._get_klines('15m')
        if df.empty:
            return 'NEUTRAL'

        hl2 = (df['high'] + df['low']) / 2
        atr = self._calc_atr(df, 20)
        upper = hl2 + 3 * atr
        lower = hl2 - 3 * atr

        st_upper, st_lower = [], []
        for i in range(len(df)):
            if i == 0 or df['close'][i - 1] > st_upper[-1]:
                st_upper.append(min(upper[i], st_upper[-1] if i > 0 else upper[i]))
            else:
                st_upper.append(upper[i])

            if i == 0 or df['close'][i - 1] < st_lower[-1]:
                st_lower.append(max(lower[i], st_lower[-1] if i > 0 else lower[i]))
            else:
                st_lower.append(lower[i])

        return 'BULL' if df['close'].iloc[-1] > st_upper[-1] else 'BEAR'

    async def bb_percentb_signal(self) -> Tuple[bool, Optional[str]]:
        df = await self._get_klines('3m')
        if df.empty:
            return False, None

        close = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = close + 2 * std
        lower = close - 2 * std
        percent_b = (df['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

        if percent_b <= 0:
            return True, 'BUY'
        elif percent_b >= 1:
            return True, 'SELL'
        return False, None

    async def execute_strategy(self):
        trend = await self.supertrend_signal()
        bb_signal, side = await self.bb_percentb_signal()

        if bb_signal:
            await self.place_orders(side, trend)

        await self.risk_management()

    async def place_orders(self, side: str, trend: str):
        qty = self._calc_position(trend, side)
        orders = [
            {'offset': Decimal('0.0005'), 'ratio': Decimal('0.3')},
            {'offset': Decimal('0.001'), 'ratio': Decimal('0.2')},
            {'offset': Decimal('0.0045'), 'ratio': Decimal('0.1')}
        ]

        price = await self._get_current_price()
        for order in orders:
            limit_price = price * (1 - order['offset'] if side == 'BUY' else 1 + order['offset'])
            await self.client.signed_request(
                'POST',
                '/order',
                {
                    'symbol': EnvConfig.SYMBOL,
                    'side': side,
                    'type': 'LIMIT',
                    'quantity': round(qty * order['ratio'], TradingParameters.QTY_PRECISION),
                    'price': round(limit_price, TradingParameters.PRICE_PRECISION),
                    'timeInForce': 'GTC'
                }
            )

    async def risk_management(self):
        if self.daily_pnl < -Decimal('0.2'):
            await self.close_positions()
        elif self.consecutive_losses >= 3:
            await self.reduce_position(Decimal('0.5'))

    # 辅助方法
    async def _get_klines(self, interval: str) -> pd.DataFrame:
        data = await self.client.signed_request(
            'GET',
            '/klines',
            {'symbol': EnvConfig.SYMBOL, 'interval': interval, 'limit': 100}
        )
        return pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]).apply(pd.to_numeric)

    def _calc_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calc_position(self, trend: str, side: str) -> Decimal:
        if trend == 'BULL':
            return TradingParameters.BULL_LONG_QTY if side == 'BUY' else TradingParameters.BULL_SHORT_QTY
        else:
            return TradingParameters.BEAR_LONG_QTY if side == 'BUY' else TradingParameters.BEAR_SHORT_QTY

    async def _get_current_price(self) -> Decimal:
        data = await self.client.signed_request(
            'GET',
            '/ticker/price',
            {'symbol': EnvConfig.SYMBOL}
        )
        return Decimal(data['price'])


# 主程序
async def main():
    client = BinanceHFTClient()
    await client.sync_time()

    strategy = HFTStrategyEngine(client)

    while True:
        try:
            await strategy.execute_strategy()
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Main loop error: {str(e)}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Graceful shutdown")
