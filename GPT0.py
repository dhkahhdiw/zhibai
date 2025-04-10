#!/usr/bin/env python3
# ETH/USDC高频交易引擎 v5.1 (修复版)

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
from aiohttp import ClientTimeout, TCPConnector, ClientConnectorError, ServerDisconnectedError
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
# 对于 USDC 永续合约，基础 URL 使用币安官方地址
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
    """优化后的交易参数配置"""
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volatility_multiplier: float = 2.5
    max_retries: int = 7      # 增加重试次数
    order_timeout: float = 2.0  # 请求超时延长至2秒
    network_timeout: float = 5.0  # 网络总超时延长至5秒
    price_precision: int = 2
    quantity_precision: int = 3
    max_position: float = 10.0
    slippage: float = 0.001
    dual_side_position: str = "true"  # 根据API要求设置


class BinanceHFTClient:
    def __init__(self):
        self.config = TradingConfig()
        # 如使用代理或遇到SSL问题，可设置 ssl=False
        self.connector = TCPConnector(
            limit=512,
            resolver=AsyncResolver(),
            ttl_dns_cache=300,
            force_close=True,
            family=AF_INET,
            ssl=False
        )
        self._init_session()
        self.recv_window = 6000  # 与 Binance 接口参数保持一致
        # 修正括号闭合问题
        self.request_timestamps = defaultdict(lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 200))
        self._time_diff = 0
        self.position = 0.0

    def _init_session(self):
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.5,
            max_timeout=3.0,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={TimeoutError, ClientConnectorError, ServerDisconnectedError}
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
        """
        使用 USDC 永续接口同步服务器时间，检查返回的 Content-Type，
        若返回内容异常则重试，多次失败后回退使用本地时间。
        """
        url = REST_URL + "/sapi/v1/futures/usdc/time"
        for retry in range(5):
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" not in content_type:
                        raise ValueError(f"Unexpected content type: {content_type}")
                    data = await resp.json()
                    if 'serverTime' not in data:
                        raise ValueError("返回数据中缺少 serverTime 字段")
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    self._time_diff = server_time - local_time
                    logger.info(f"时间同步成功，时间差: {self._time_diff}ms")
                    return
            except Exception as e:
                logger.error(f"时间同步失败(重试 {retry + 1}): {str(e)}")
                await asyncio.sleep(2 ** retry)
        logger.warning("无法同步服务器时间，回退使用本地时间")
        self._time_diff = 0

    async def _signed_request(self, method: str, path: str, params: Dict) -> Dict:
        """
        采用签名请求，确保构造 URL 正确。该方法在请求时增加必要的参数并生成签名。
        """
        params.update({
            "timestamp": int(time.time() * 1000 + self._time_diff),
            "recvWindow": self.recv_window
        })
        sorted_params = sorted(params.items())
        query = urllib.parse.urlencode(sorted_params, doseq=True)
        signature = hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = REST_URL + "/sapi/v1/futures/usdc" + path + "?" + query + "&signature=" + signature
        headers = {
            "X-MBX-APIKEY": API_KEY,
            "Accept": "application/json"
        }

        logger.debug(f"请求URL: {url.split('?')[0]} 参数: {sorted_params}")

        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)
                async with self.session.request(method, url, headers=headers) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" not in content_type:
                        text = await resp.text()
                        raise Exception(f"Unexpected Content-Type: {content_type}. Response text: {text}")
                    return await resp.json()
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt + 1}): {type(e).__name__} {str(e)}")
                if attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise Exception("超过最大重试次数")

    async def _rate_limit_check(self, endpoint: str):
        """智能管理速率限制，根据预设参数检测请求频率"""
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]
        now = time.monotonic()
        while dq and dq[0] < now - period:
            dq.popleft()
        if len(dq) >= limit:
            wait_time = max(dq[0] + period - now + np.random.uniform(0, 0.05), 0)
            logger.warning(f"速率限制触发: {endpoint} 等待 {wait_time:.3f}s")
            METRICS['throughput'].set(0)
            await asyncio.sleep(wait_time)
        dq.append(now)
        METRICS['throughput'].inc()

    async def manage_leverage(self):
        """设置杠杆，依据配置提交请求"""
        params = {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'dualSidePosition': self.config.dual_side_position
        }
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self) -> Optional[pd.DataFrame]:
        """获取K线数据，并转换为 DataFrame 格式"""
        try:
            data = await self._signed_request('GET', '/klines', {
                'symbol': SYMBOL,
                'interval': '1m',
                'limit': 100,
                'contractType': 'PERPETUAL'
            })
            if not isinstance(data, list):
                raise ValueError("K线数据格式异常")
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
            return df.iloc[-100:]
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
        """先同步时间、设置杠杆，再进入策略主循环"""
        await self.client.sync_server_time()
        await self.client.manage_leverage()

        while True:
            try:
                start_time = time.monotonic()
                df = await self.client.fetch_klines()
                if df is None or df.empty:
                    await asyncio.sleep(1)
                    continue

                close_prices = df['close'].values.astype(np.float32)
                high_prices = df['high'].values.astype(np.float32)
                low_prices = df['low'].values.astype(np.float32)

                atr_array = self._vectorized_atr(high_prices, low_prices, close_prices)
                atr_val = atr_array[-1]

                signals = await self.generate_signals(df, atr_val)

                if signals.action:
                    current_price = close_prices[-1]
                    adjusted_price = current_price * (1 + self.config.slippage *
                                                      (-1 if signals.side == 'BUY' else 1))
                    await self.place_limit_order(
                        side=signals.side,
                        limit_price=adjusted_price,
                        take_profit=signals.tp,
                        stop_loss=signals.sl
                    )

                METRICS['latency'].set((time.monotonic() - start_time) * 1000)
                await asyncio.sleep(max(0.5, 1.0 - (time.monotonic() - start_time)))
            except Exception as e:
                logger.error(f"策略异常: {str(e)}", exc_info=True)
                await asyncio.sleep(5)

    def _vectorized_atr(self, high, low, close) -> np.ndarray:
        """向量化计算 ATR，避免除零异常"""
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr = np.convolve(tr, np.ones(self.config.atr_window), 'valid') / self.config.atr_window
        return atr

    async def generate_signals(self, df: pd.DataFrame, atr_val: float) -> Signal:
        """基于 EMA 金叉、ATR 和价格区间生成买卖信号"""
        ema_fast = df['close'].ewm(span=self.config.ema_fast, adjust=False).mean().values
        ema_slow = df['close'].ewm(span=self.config.ema_slow, adjust=False).mean().values
        ema_cross = ema_fast[-1] > ema_slow[-1]

        volatility_ratio = atr_val / df['close'].values[-1]
        price_range = df['high'].values[-1] - df['low'].values[-1]
        orderbook_imbalance = (price_range / df['close'].values[-1]) < 0.005

        action = ema_cross and (volatility_ratio > 0.02) and orderbook_imbalance

        return Signal(
            action=action,
            side='BUY' if action else 'SELL',
            tp=df['close'].values[-1] + atr_val * 2,
            sl=df['close'].values[-1] - atr_val * 1.5
        )

    async def place_limit_order(self, side: str, limit_price: float, tp: float, sl: float):
        """构造并提交限价单，校验价格及数量有效性"""
        formatted_price = round(limit_price, self.config.price_precision)
        formatted_qty = round(QUANTITY, self.config.quantity_precision)

        if formatted_price <= 0 or formatted_qty <= 0:
            logger.error("无效订单参数: 价格或数量<=0")
            return

        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'LIMIT',
            'price': formatted_price,
            'quantity': formatted_qty,
            'timeInForce': 'GTC'
        }

        try:
            await self.client._signed_request('POST', '/order', params)
            logger.info(f"挂单成功 {side}@{formatted_price} | TP: {tp:.2f} SL: {sl:.2f}")
            if abs(self.client.position) > self.config.max_position * 0.75:
                await self.client.manage_leverage()
        except Exception as e:
            logger.error(f"订单失败: {str(e)}")
            METRICS['errors'].inc()


async def main():
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，优雅退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")


if __name__ == "__main__":
    asyncio.run(main())