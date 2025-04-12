#!/usr/bin/env python3
# ETH/USDT 高频交易引擎 v8.1 (合规版)
import uvloop

uvloop.install()

import os, asyncio, time, hmac, hashlib, urllib.parse, logging, datetime
from collections import deque, defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector, ClientSession
from aiohttp_retry import RetryClient, ExponentialRetry
import json
import random

# ==================== 环境配置 ====================
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)

API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDT'
LEVERAGE = 50
REST_URL = 'https://fapi.binance.com'


# ==================== 交易参数 ====================
class TradingConfig:
    # 核心参数
    atr_window = 14
    ema_fast = 9
    ema_slow = 21
    st_period = 20
    st_multiplier = 3.0
    bb_period = 20
    bb_std = 2.0

    # 风险参数
    max_retries = 5
    order_timeout = 1.5
    network_timeout = 3.0
    price_precision = 2
    quantity_precision = 3
    max_slippage = 0.0015
    daily_drawdown_limit = 0.20

    # 高频参数
    rate_limits = {
        'klines': (60, 5),
        'order': (300, 10),
        'openOrders': (60, 60)
    }

    # 网络优化
    tcp_params = {
        'limit': 1000,
        'ttl_dns_cache': 300,
        'force_close': True,
        'use_dns_cache': True
    }


# ==================== 核心引擎 ====================
class BinanceHFEngine:
    def __init__(self, config: TradingConfig):
        self.config = config
        self._time_diff = 0
        self.session = None
        self.request_timestamps = defaultdict(lambda: deque(maxlen=1000))
        self.position = 0.0
        self._init_session()

    def _init_session(self):
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={asyncio.TimeoutError}
        )
        self.connector = TCPConnector(**self.config.tcp_params)
        self.session = RetryClient(
            connector=self.connector,
            retry_options=retry_opts,
            timeout=ClientTimeout(
                total=self.config.network_timeout,
                sock_connect=self.config.order_timeout
            )
        )

    async def sync_time(self):
        """精确时间同步（带时差校验）"""
        for _ in range(3):
            try:
                async with self.session.get(f"{REST_URL}/fapi/v1/time") as resp:
                    data = await resp.json()
                    server_time = data['serverTime']
                    local_time = int(time.time() * 1000)
                    self._time_diff = server_time - local_time
                    if abs(self._time_diff) > 5000:
                        logging.warning(f"大时差警告：{self._time_diff}ms")
                    return
            except Exception as e:
                await asyncio.sleep(2 ** _)
        raise ConnectionError("时间同步失败")

    async def signed_request(self, method: str, endpoint: str, params: dict):
        """带严格速率控制的签名请求"""
        # 速率检查
        await self._rate_limit_check(endpoint)

        # 构造签名
        params = {k: v for k, v in params.items() if v is not None}
        params.update({
            'timestamp': int(time.time() * 1000 + self._time_diff),
            'recvWindow': 6000
        })
        query = urllib.parse.urlencode(sorted(params.items()))
        signature = hmac.new(
            SECRET_KEY.encode(),
            query.encode(),
            hashlib.sha256
        ).hexdigest()

        # 构造请求
        url = f"{REST_URL}/fapi/v1/{endpoint}?{query}&signature={signature}"
        headers = {
            "X-MBX-APIKEY": API_KEY,
            "X-TS-Strategy": "HFTv8",
            "Accept-Encoding": "gzip"
        }

        async with self.session.request(method, url, headers=headers) as resp:
            if resp.status != 200:
                error = await resp.text()
                logging.error(f"API Error {resp.status}: {error}")
                raise Exception(f"API Error {resp.status}")
            return await resp.json()

    async def _rate_limit_check(self, endpoint: str):
        """动态速率限制管理"""
        limit, period = self.config.rate_limits.get(endpoint, (1200, 60))
        now = time.monotonic()
        dq = self.request_timestamps[endpoint]

        # 移除过期记录
        while dq and dq[0] < now - period:
            dq.popleft()

        # 计算等待时间
        if len(dq) >= limit:
            sleep_time = max(dq[0] + period - now + random.uniform(0.01, 0.05), 0)
            logging.warning(f"速率限制 {endpoint} 等待 {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)

        dq.append(now)

    # ==================== 交易接口 ====================
    async def set_leverage(self):
        """设置杠杆（符合币安2025新规）"""
        return await self.signed_request('POST', 'leverage', {
            'symbol': SYMBOL,
            'leverage': LEVERAGE,
            'dualSidePosition': 'false'
        })

    async def get_klines(self, interval: str, limit: int = 500):
        """获取K线数据（优化版本）"""
        try:
            data = await self.signed_request('GET', 'klines', {
                'symbol': SYMBOL,
                'interval': interval,
                'limit': limit
            })
            return pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count',
                'taker_buy_volume', 'taker_buy_quote', 'ignore'
            ])
        except Exception as e:
            logging.error(f"K线获取失败: {e}")
            return pd.DataFrame()

    async def place_order(self, side: str, qty: float, price: float = None):
        """智能下单（支持限价/市价）"""
        order_type = 'LIMIT' if price else 'MARKET'
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': order_type,
            'quantity': round(qty, self.config.quantity_precision),
            'newOrderRespType': 'FULL'
        }
        if price:
            params['price'] = round(price, self.config.price_precision)
            params['timeInForce'] = 'GTC'

        try:
            return await self.signed_request('POST', 'order', params)
        except Exception as e:
            if '-2039' in str(e) or '-2038' in str(e):  # 处理新错误代码
                logging.error("订单参数错误，触发撤单")
                await self.cancel_all_orders()
            raise

    async def cancel_all_orders(self):
        """批量撤单（优化版本）"""
        return await self.signed_request('DELETE', 'openOrders', {
            'symbol': SYMBOL
        })


# ==================== 策略引擎 ====================
class AlphaStrategy:
    def __init__(self, engine: BinanceHFEngine):
        self.engine = engine
        self.config = TradingConfig()
        self.position = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0

    async def super_trend_signal(self):
        """15分钟超级趋势信号"""
        df = await self.engine.get_klines('15m', 100)
        if df.empty:
            return None

        # 计算指标
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        # 计算ATR
        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))
        atr = np.convolve(tr, np.ones(14), 'valid') / 14

        # 计算超级趋势
        hl2 = (high + low) / 2
        upper = hl2 + 3 * atr
        lower = hl2 - 3 * atr

        # 生成信号
        last_close = close[-1]
        if last_close > upper[-1]:
            return 'LONG'
        elif last_close < lower[-1]:
            return 'SHORT'
        return None

    async def bb_percent_b(self):
        """3分钟布林带%B信号"""
        df = await self.engine.get_klines('3m', 20)
        if df.empty:
            return 0.5

        close = df['close'].values.astype(float)
        sma = pd.Series(close).rolling(20).mean().values
        std = pd.Series(close).rolling(20).std().values
        upper = sma + 2 * std
        lower = sma - 2 * std

        last_close = close[-1]
        if (upper[-1] - lower[-1]) == 0:
            return 0.5
        return (last_close - lower[-1]) / (upper[-1] - lower[-1])

    async def execute_strategy(self):
        """策略执行循环"""
        while True:
            try:
                # 获取信号
                trend = await self.super_trend_signal()
                bb = await self.bb_percent_b()

                # 趋势跟踪
                if trend == 'LONG' and self.position <= 0:
                    await self.adjust_position(0.1)
                elif trend == 'SHORT' and self.position >= 0:
                    await self.adjust_position(-0.1)

                # 短线信号
                if bb <= 0.05:
                    await self.place_twap_order('BUY', 0.05)
                elif bb >= 0.95:
                    await self.place_twap_order('SELL', 0.05)

                # 风险管理
                await self.risk_check()

                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"策略执行错误: {e}")
                await asyncio.sleep(10)

    async def place_twap_order(self, side: str, qty: float):
        """TWAP下单（支持动态滑点）"""
        # 获取市场深度
        depth = await self.engine.signed_request('GET', 'depth', {'symbol': SYMBOL, 'limit': 5})
        best_ask = float(depth['asks'][0][0])
        best_bid = float(depth['bids'][0][0])

        # 计算动态价格
        spread = best_ask - best_bid
        if side == 'BUY':
            price = best_bid + spread * 0.3
        else:
            price = best_ask - spread * 0.3

        # 分三笔下单
        for i in range(3):
            try:
                await self.engine.place_order(side, qty / 3, price)
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"TWAP分单失败: {e}")

    async def adjust_position(self, target: float):
        """仓位调整（带滑点控制）"""
        delta = target - self.position
        if abs(delta) < 0.01:
            return

        side = 'BUY' if delta > 0 else 'SELL'
        try:
            # 获取当前价格
            ticker = await self.engine.signed_request('GET', 'ticker/bookTicker', {'symbol': SYMBOL})
            price = float(ticker['bidPrice']) if side == 'SELL' else float(ticker['askPrice'])

            # 执行市价单
            result = await self.engine.place_order(side, abs(delta))
            self.position = target
            logging.info(f"仓位调整成功: {side} {abs(delta)} @ {price}")
        except Exception as e:
            logging.error(f"调仓失败: {e}")

    async def risk_check(self):
        """实时风险监控"""
        if self.daily_pnl < -self.config.daily_drawdown_limit:
            logging.critical("触发最大回撤限制！")
            await self.close_all_positions()

        if self.consecutive_losses >= 3:
            logging.warning("连续3次亏损，降低风险暴露")
            await self.adjust_position(self.position * 0.5)

    async def close_all_positions(self):
        """紧急平仓"""
        if self.position > 0:
            await self.engine.place_order('SELL', self.position)
        elif self.position < 0:
            await self.engine.place_order('BUY', abs(self.position))
        self.position = 0.0


# ==================== 主程序 ====================
async def main():
    # 初始化引擎
    config = TradingConfig()
    engine = BinanceHFEngine(config)
    strategy = AlphaStrategy(engine)

    try:
        # 环境准备
        await engine.sync_time()
        await engine.set_leverage()

        # 启动策略
        await strategy.execute_strategy()
    finally:
        await engine.session.close()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler('hft.log'), logging.StreamHandler()]
    )
    asyncio.run(main())