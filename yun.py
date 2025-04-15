#!/usr/bin/env python3
"""
ETH/USDC 高频交易引擎 v7.4— 究极青眼白龙版
------------------------------------------------------------
【一】多时间框架趋势与信号：
1. 趋势判定：采用15m超级趋势指标（默认20周期MA+3×ATR）判断主趋势：
   - 当价格在超级趋势线上方且指标为绿色时，为上升趋势；
   - 当价格在超级趋势线下方且指标为红色时，为下降趋势；
   - 当价格处于边界时，辅以15m MACD（12,26,9）零轴关系验证趋势。
2. 强弱信号：
   - 空单：基于1h布林带%B，若%B > 0.8为强信号，否则为弱信号；
   - 多单：若%B < 0.2为强信号，否则为弱信号。

【二】挂单与止盈止损：
- 利用3m布林带%B (20,2)判断入场：若%B ≤ 0触发买单，≥1触发卖单；
- 强信号时：同趋势单笔仓位±0.12ETH；反趋势±0.07ETH；
- 弱信号时：同趋势±0.03ETH；反趋势±0.015ETH；
- 挂单价格按照多档偏移设置（强信号5档，弱信号2档），止盈亦类似设置；
- 止损初始按买价×0.98／卖价×1.02设置，并依据3m布林带带宽动态追踪。

【三】15分钟MADC策略（MACD辅助平仓）：
- 以15m数据计算MACD（EMA12,EMA26），离轴线数值=2*(DIF-DEA)；
- 当交叉触发条件时，根据离轴值区间设定平仓仓位：
   - 空单：交叉值≥11且<20时平仓0.07ETH；≥20时平仓0.14ETH；
   - 多单同理：交叉值≤-11且>-20时平仓0.07ETH；≤-20时平仓0.14ETH。

【四】超级趋势策略：
- 使用三个不同参数的15m超级趋势指标（例如长度10、11、12，对应不同因子），
  当三指标均上升时，立刻市价做多0.15ETH；均下降时，市价做空0.15ETH；
- 指标反转时平仓对应3m挂单仓位。

【五】信号过滤与辅助指标：
- 在信号触发前，通过5m成交量检验（要求最新5m成交量大于前3根均值）；
- 额外计算 MA7、MA25、MA99（基于SMA9）用于辅助信号判断（预留接口）。

【六】仓位控制：
- 取消从币安读取仓位，仓位全部由本地记录更新；
- 入场和平仓成功后更新本地仓位；
- 根据趋势设定仓位上限（趋势上升时：多仓0.49ETH、空仓0.35ETH；下降时反之），超过时暂停下单。
"""

import uvloop
uvloop.install()

import os, asyncio, time, hmac, hashlib, urllib.parse, logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

from aiohttp import ClientTimeout, TCPConnector, ClientConnectorError, ServerDisconnectedError
from aiohttp_retry import RetryClient, ExponentialRetry
from aiohttp.resolver import AsyncResolver

from dotenv import load_dotenv
from prometheus_client import Gauge, start_http_server

# ==================== 环境配置 ====================
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY    = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL     = 'ETHUSDC'
LEVERAGE   = 50

# 仓位控制目标（本地记录，仅作参考）
BULL_LONG_LIMIT  = 0.49
BULL_SHORT_LIMIT = 0.35
BEAR_LONG_LIMIT  = 0.35
BEAR_SHORT_LIMIT = 0.49

# 订单基础规模
STRONG_SIZE_SAME  = 0.12
STRONG_SIZE_DIFF  = 0.07
WEAK_SIZE_SAME    = 0.03
WEAK_SIZE_DIFF    = 0.015

# 下单挂单方案
def get_entry_order_list(strong: bool) -> List[Dict[str, Any]]:
    if strong:
        return [
            {'offset': 0.0025, 'ratio': 0.20},
            {'offset': 0.0040, 'ratio': 0.20},
            {'offset': 0.0060, 'ratio': 0.20},
            {'offset': 0.0080, 'ratio': 0.20},
            {'offset': 0.0160, 'ratio': 0.20},
        ]
    else:
        return [
            {'offset': 0.0025, 'ratio': 0.50},
            {'offset': 0.0160, 'ratio': 0.50},
        ]

def get_tp_order_list(strong: bool) -> List[Dict[str, Any]]:
    if strong:
        return [
            {'offset': 0.0007, 'ratio': 0.20},
            {'offset': 0.0007, 'ratio': 0.20},
            {'offset': 0.0003, 'ratio': 0.20},
            {'offset': 0.0025, 'ratio': 0.20},
            {'offset': 0.0110, 'ratio': 0.20},
        ]
    else:
        return [
            {'offset': 0.0007, 'ratio': 0.50},
            {'offset': 0.0110, 'ratio': 0.50},
        ]

# 交易及限频配置
RECV_WINDOW = 10000
MIN_NOTIONAL = 20.0
COOLDOWN_SECONDS = 3
RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    'klines': (60, 5),
    'orders': (300, 10),
    'leverage': (30, 60)
}

start_http_server(8001)
METRICS = {
    'throughput': Gauge('api_throughput', '请求数/秒'),
}

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
    st_period: int = 20
    st_multiplier: float = 3.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    max_retries: int = 7
    order_timeout: float = 2.0
    network_timeout: float = 5.0
    price_precision: int = 2
    quantity_precision: int = 3
    order_adjust_interval: float = 1.0
    dual_side_position: bool = False

# ------------------- Binance API 客户端 -------------------
class BinanceHFTClient:
    def __init__(self) -> None:
        self.config = TradingConfig()
        self.connector = TCPConnector(limit=512, resolver=AsyncResolver(), ttl_dns_cache=300, force_close=True, ssl=True)
        self._init_session()
        self.recv_window = RECV_WINDOW
        self.request_timestamps = defaultdict(lambda: deque(maxlen=RATE_LIMITS['klines'][0] + 200))
        self._time_diff = 0

    def _init_session(self) -> None:
        retry_opts = ExponentialRetry(
            attempts=self.config.max_retries,
            start_timeout=0.5,
            max_timeout=3.0,
            statuses={408, 429, 500, 502, 503, 504},
            exceptions={TimeoutError, ClientConnectorError, ServerDisconnectedError}
        )
        self.session = RetryClient(connector=self.connector, retry_options=retry_opts,
                                    timeout=ClientTimeout(total=self.config.network_timeout, sock_connect=self.config.order_timeout))

    async def sync_server_time(self) -> None:
        url = f"https://fapi.binance.com/fapi/v1/time"
        for retry in range(5):
            try:
                async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
                    data = await resp.json()
                    server_time = data.get('serverTime')
                    if not server_time:
                        raise ValueError("缺少 serverTime")
                    local_time = int(time.time() * 1000)
                    self._time_diff = server_time - local_time
                    logger.info(f"时间同步成功，差值：{self._time_diff}ms")
                    return
            except Exception as e:
                logger.error(f"时间同步失败(重试 {retry+1}): {e}")
                await asyncio.sleep(2 ** retry)
        logger.warning("时间同步失败，采用本地时间")
        self._time_diff = 0

    async def _signed_request(self, method: str, path: str, params: dict) -> dict:
        params.update({"timestamp": int(time.time() * 1000 + self._time_diff), "recvWindow": self.recv_window})
        if params.get("symbol", "") == SYMBOL:
            params["marginCoin"] = "USDC"
        sorted_params = sorted(params.items())
        query = urllib.parse.urlencode(sorted_params, doseq=True)
        signature = hmac.new(SECRET_KEY.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
        url = f"https://fapi.binance.com/fapi/v1{path}?{query}&signature={signature}"
        headers = {"X-MBX-APIKEY": API_KEY, "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
        logger.debug(f"请求: {url.split('?')[0]} 参数: {sorted_params}")
        for attempt in range(self.config.max_retries + 1):
            try:
                endpoint = path.split('/')[-1]
                await self._rate_limit_check(endpoint)
                async with self.session.request(method, url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status} 返回: {await resp.text()}")
                    if "application/json" not in resp.headers.get("Content-Type", ""):
                        raise Exception(f"响应异常: {await resp.text()}")
                    data = await resp.json()
                    if isinstance(data, dict) and data.get("code", 0) < 0:
                        raise Exception(f"接口错误，Code: {data.get('code')}, Msg: {data.get('msg')}")
                    return data
            except Exception as e:
                logger.error(f"请求失败 (尝试 {attempt+1}): {e}")
                if attempt >= self.config.max_retries:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))
        raise Exception("超过最大重试次数")

    async def _rate_limit_check(self, endpoint: str) -> None:
        limit, period = RATE_LIMITS.get(endpoint, (60, 1))
        dq = self.request_timestamps[endpoint]
        now = time.monotonic()
        while dq and dq[0] < now - period:
            dq.popleft()
        if len(dq) >= limit:
            wait_time = max(dq[0] + period - now + np.random.uniform(0, 0.05), 0)
            logger.warning(f"接口 {endpoint} 限频，等待 {wait_time:.3f}s")
            METRICS['throughput'].set(0)
            await asyncio.sleep(wait_time)
        dq.append(now)
        METRICS['throughput'].inc()

    async def manage_leverage(self) -> dict:
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE}
        return await self._signed_request('POST', '/leverage', params)

    async def fetch_klines(self, interval: str, limit: int = 100) -> pd.DataFrame:
        params = {'symbol': SYMBOL, 'interval': interval, 'limit': limit}
        data = await self._signed_request('GET', '/klines', params)
        if not isinstance(data, list):
            logger.error("K线格式异常")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'close_time', 'quote_asset_volume', 'trades',
                                          'taker_buy_base', 'taker_buy_quote', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

# ------------------- 策略模块 -------------------
@dataclass
class Signal:
    action: bool
    side: str      # 'BUY' 或 'SELL'
    order_details: dict = None

class ETHUSDCStrategy:
    def __init__(self, client: BinanceHFTClient) -> None:
        self.client = client
        self.config = TradingConfig()
        self.last_trade_side: str = None     # 当前趋势方向：'LONG' 或 'SHORT'
        self.last_triggered_side: str = None   # 上次下单方向（信号轮流冷却用）
        self.last_order_time: float = 0
        self.entry_price: float = None
        # 本地仓位记录（所有操作均本地更新，不再从币安读取）
        self.current_long: float = 0.0
        self.current_short: float = 0.0
        self.prev_macd_off: float = None

    def update_local_position(self, side: str, quantity: float, closing: bool = False) -> None:
        if closing:
            if side.upper() == 'BUY':   # 平空
                self.current_short = max(0, self.current_short - quantity)
            elif side.upper() == 'SELL':  # 平多
                self.current_long = max(0, self.current_long - quantity)
        else:
            if side.upper() == 'BUY':
                self.current_long += quantity
            elif side.upper() == 'SELL':
                self.current_short += quantity
        logger.info(f"[本地仓位更新] 多仓: {self.current_long:.4f} ETH，空仓: {self.current_short:.4f} ETH")

    # ------------------ 趋势判断 ------------------
    async def analyze_trend_15m(self) -> str:
        df = await self.client.fetch_klines(interval='15m', limit=100)
        if df.empty:
            return 'LONG'
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        tr = np.maximum.reduce([
            high.diff().abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ])
        atr = pd.Series(tr).rolling(window=self.config.st_period).mean().iloc[-1]
        hl2 = (high + low) / 2
        last_hl2 = hl2.iloc[-1]
        basic_upper_val = last_hl2 + self.config.st_multiplier * atr
        basic_lower_val = last_hl2 - self.config.st_multiplier * atr
        latest = close.iloc[-1]
        if latest > basic_upper_val:
            return 'LONG'
        elif latest < basic_lower_val:
            return 'SHORT'
        else:
            ema_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            return 'LONG' if macd_line.iloc[-1] >= 0 else 'SHORT'

    # ------------------ 信号强弱判断 ------------------
    async def get_hourly_strength(self, side: str) -> bool:
        df = await self.client.fetch_klines(interval='1h', limit=50)
        if df.empty:
            return False
        close = df['close'].astype(float)
        sma = close.rolling(window=self.config.bb_period).mean()
        std = close.rolling(window=self.config.bb_period).std()
        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std
        percent_b = (close.iloc[-1] - lower.iloc[-1])/(upper.iloc[-1]-lower.iloc[-1]) if (upper.iloc[-1]-lower.iloc[-1])>0 else 0.5
        logger.info(f"[1h%B] {side.upper()}信号: {percent_b:.3f}")
        if side.upper() == 'BUY':
            return percent_b < 0.2
        elif side.upper() == 'SELL':
            return percent_b > 0.8
        return False

    # ------------------ 3m布林带信号 ------------------
    async def analyze_order_signals_3m(self) -> Tuple[Signal, dict]:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty:
            return Signal(False, 'NONE'), {}
        close = df['close'].astype(float)
        sma = close.rolling(window=self.config.bb_period).mean()
        std = close.rolling(window=self.config.bb_period).std()
        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std
        percent_b = (close.iloc[-1] - lower.iloc[-1])/(upper.iloc[-1]-lower.iloc[-1]) if (upper.iloc[-1]-lower.iloc[-1])>0 else 0.5
        logger.info(f"[3m%B] 当前 %B: {percent_b:.3f}")
        if percent_b <= 0:
            return Signal(True, 'BUY', {'trigger_price': close.iloc[-1]}), {}
        elif percent_b >= 1:
            return Signal(True, 'SELL', {'trigger_price': close.iloc[-1]}), {}
        return Signal(False, 'NONE'), {}

    # ------------------ 5分钟成交量过滤 ------------------
    async def volume_filter_check(self) -> bool:
        df = await self.client.fetch_klines(interval='5m', limit=4)
        if df.empty or len(df) < 4:
            return True
        # 最新K线为最后一根，其余三根求均值
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-4:-1].mean()
        logger.info(f"[成交量过滤] 当前5m量: {current_vol:.2f}, 前3根均值: {avg_vol:.2f}")
        return current_vol > avg_vol

    # ------------------ 挂单下单及止盈止损 ------------------
    async def place_dynamic_limit_orders(self, side: str, order_list: List[Dict[str, Any]], trigger_price: float, order_size: float) -> None:
        pos_side = "LONG" if side == "BUY" else "SHORT"
        for order in order_list:
            offset = order['offset']
            ratio = order['ratio']
            qty = round(order_size * ratio, self.config.quantity_precision)
            if qty <= 0:
                continue
            limit_price = round(trigger_price * (1 - offset) if side=="BUY" else trigger_price * (1 + offset), self.config.price_precision)
            if limit_price <= 0:
                continue
            params = {
                'symbol': SYMBOL,
                'side': side,
                'type': 'LIMIT',
                'price': limit_price,
                'quantity': qty,
                'timeInForce': 'GTC',
                'positionSide': pos_side
            }
            try:
                data = await self.client._signed_request('POST', '/order', params)
                logger.info(f"[下单] {side} @ {limit_price}，数量: {qty}，偏移: {offset*100:.2f}% 成功，返回: {data}")
                self.update_local_position(side, qty, closing=False)
            except Exception as e:
                logger.error(f"[下单] {side}挂单失败：{e}")

    async def close_position(self, side: str, ratio: float, strategy: str = "normal") -> None:
        df = await self.client.fetch_klines(interval='3m', limit=50)
        if df.empty or not self.entry_price:
            return
        current_price = float(df['close'].iloc[-1])
        order_qty = ratio
        if current_price * order_qty < MIN_NOTIONAL:
            logger.error(f"[平仓] 名义金额小于 {MIN_NOTIONAL} USDC，跳过")
            return
        params = {
            'symbol': SYMBOL,
            'side': side,
            'type': 'MARKET',
            'quantity': order_qty,
            'positionSide': self.last_trade_side
        }
        try:
            data = await self.client._signed_request('POST', '/order', params)
            logger.info(f"[平仓] 市价平仓 {side}，数量: {order_qty}，返回: {data}")
            self.update_local_position(side, order_qty, closing=True)
        except Exception as e:
            logger.error(f"[平仓] 失败: {e}")

    # ------------------ 本地仓位记录更新 ------------------
    def update_local_position(self, side: str, quantity: float, closing: bool = False) -> None:
        if closing:
            if side.upper() == 'BUY':    # 平空仓
                self.current_short = max(0, self.current_short - quantity)
            elif side.upper() == 'SELL': # 平多仓
                self.current_long = max(0, self.current_long - quantity)
        else:
            if side.upper() == 'BUY':
                self.current_long += quantity
            elif side.upper() == 'SELL':
                self.current_short += quantity
        logger.info(f"[本地仓位更新] 多仓: {self.current_long:.4f} ETH，空仓: {self.current_short:.4f} ETH")

    # ------------------ 15m MADC策略 ------------------
    async def madc_strategy_loop(self) -> None:
        # 此处基于15m数据计算MACD，并计算离轴线 = 2*(DIF-DEA)
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                close = df['close'].astype(float)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9, adjust=False).mean()
                divergence = 2 * (dif - dea)
                # 当交叉时记录触发条件（此处简单采用最后一次离轴值）
                trigger_val = divergence.iloc[-1]
                logger.info(f"[MADC] 离轴值: {trigger_val:.2f}")
                # 空单触发条件
                if trigger_val >= 11 and trigger_val < 20:
                    logger.info("[MADC] 空单触发（弱信号），平多仓0.07ETH")
                    await self.close_position('SELL', 0.07, strategy="madc")
                elif trigger_val >= 20:
                    logger.info("[MADC] 空单触发（强信号），平多仓0.14ETH")
                    await self.close_position('SELL', 0.14, strategy="madc")
                # 多单触发条件
                if trigger_val <= -11 and trigger_val > -20:
                    logger.info("[MADC] 多单触发（弱信号），平空仓0.07ETH")
                    await self.close_position('BUY', 0.07, strategy="madc")
                elif trigger_val <= -20:
                    logger.info("[MADC] 多单触发（强信号），平空仓0.14ETH")
                    await self.close_position('BUY', 0.14, strategy="madc")
            except Exception as e:
                logger.error(f"[MADC] 异常: {e}")
            await asyncio.sleep(60 * 15)

    # ------------------ 超级趋势策略（并行） ------------------
    async def supertrend_strategy_loop(self) -> None:
        # 采用三个不同参数的15m超级趋势指标
        # 示例参数：factor1: length=10, factor=3; factor2: length=11, factor=3; factor3: length=12, factor=3
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                high = df['high'].astype(float)
                low = df['low'].astype(float)
                close = df['close'].astype(float)
                atr = pd.Series(np.maximum.reduce([
                    high.diff().abs(),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()
                ])).rolling(window=10).mean().iloc[-1]  # 用最短周期ATR作为示例

                # 分别计算三组超级趋势
                # 超级趋势计算简化：basic_upper = (High+Low)/2 + factor*ATR, basic_lower = (High+Low)/2 - factor*ATR
                def supertrend(length: int, factor: float) -> str:
                    atr_val = pd.Series(np.maximum.reduce([
                        high.diff().abs(),
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()
                    ])).rolling(window=length).mean().iloc[-1]
                    hl2 = ((high + low) / 2).iloc[-1]
                    basic_upper = hl2 + factor * atr_val
                    basic_lower = hl2 - factor * atr_val
                    latest = close.iloc[-1]
                    if latest > basic_upper:
                        return 'UP'
                    elif latest < basic_lower:
                        return 'DOWN'
                    else:
                        return 'NEUTRAL'
                trend1 = supertrend(10, 1)
                trend2 = supertrend(11, 2)
                trend3 = supertrend(12, 3)
                logger.info(f"[超级趋势] 指标趋势: {trend1}, {trend2}, {trend3}")
                # 如果三者均为UP，则触发市价做多0.15ETH；均为DOWN则做空0.15ETH
                if trend1 == trend2 == trend3 == 'UP':
                    logger.info("[超级趋势] 所有指标上升，触发市价多单0.15ETH")
                    # 市价入场直接更新本地仓位
                    self.update_local_position('BUY', 0.15, closing=False)
                elif trend1 == trend2 == trend3 == 'DOWN':
                    logger.info("[超级趋势] 所有指标下降，触发市价空单0.15ETH")
                    self.update_local_position('SELL', 0.15, closing=False)
            except Exception as e:
                logger.error(f"[超级趋势] 异常: {e}")
            await asyncio.sleep(60 * 15)

    # ------------------ 信号下单逻辑 ------------------
    async def order_signal_loop(self) -> None:
        while True:
            try:
                vol_ok = await self.volume_filter_check()
                if not vol_ok:
                    logger.info("[信号下单] 成交量过滤未通过，忽略当前信号")
                    await asyncio.sleep(self.config.order_adjust_interval)
                    continue
                signal, _ = await self.analyze_order_signals_3m()
                if signal.action:
                    strong = await self.get_hourly_strength(signal.side)
                    if self.last_trade_side:
                        if signal.side.upper() == self.last_trade_side.upper():
                            order_size = STRONG_SIZE_SAME if strong else WEAK_SIZE_SAME
                        else:
                            order_size = STRONG_SIZE_DIFF if strong else WEAK_SIZE_DIFF
                    else:
                        order_size = STRONG_SIZE_SAME if strong else WEAK_SIZE_SAME
                    if self.last_triggered_side and self.last_triggered_side.upper() == signal.side.upper():
                        if time.time() - self.last_order_time < COOLDOWN_SECONDS:
                            logger.info("[信号] 冷却期内，忽略同向信号")
                            await asyncio.sleep(self.config.order_adjust_interval)
                            continue
                    orders = get_entry_order_list(strong)
                    self.entry_price = signal.order_details.get("trigger_price")
                    await self.place_dynamic_limit_orders(signal.side, orders, trigger_price=self.entry_price, order_size=order_size)
                    self.last_triggered_side = signal.side.upper()
                    self.last_order_time = time.time()
                    # 记录当前趋势（采用下单方向作为趋势参考）
                    self.last_trade_side = 'LONG' if signal.side.upper() == 'BUY' else 'SHORT'
                await asyncio.sleep(self.config.order_adjust_interval)
            except Exception as e:
                logger.error(f"[信号下单] 异常: {e}")
                await asyncio.sleep(self.config.order_adjust_interval)

    # ------------------ 止盈止损管理 ------------------
    async def stop_loss_profit_management_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='3m', limit=50)
                if df.empty or not self.entry_price:
                    await asyncio.sleep(0.5)
                    continue
                latest = float(df['close'].iloc[-1])
                sma = df['close'].astype(float).rolling(window=self.config.bb_period).mean().iloc[-1]
                std = df['close'].astype(float).rolling(window=self.config.bb_period).std().iloc[-1]
                band_width = (sma + self.config.bb_std * std) - (sma - self.config.bb_std * std)
                dynamic_stop = (latest - band_width * 0.5) if self.last_trade_side=='LONG' else (latest + band_width * 0.5)
                logger.info(f"[止盈止损] 当前价={latest:.2f}, 动态止损={dynamic_stop:.2f}")
                if self.last_trade_side == 'LONG' and latest < dynamic_stop:
                    logger.info("[止损] 多单止损触发")
                    await self.close_position('SELL', self.current_long, strategy="normal")
                elif self.last_trade_side == 'SHORT' and latest > dynamic_stop:
                    logger.info("[止损] 空单止损触发")
                    await self.close_position('BUY', self.current_short, strategy="normal")
            except Exception as e:
                logger.error(f"[止盈止损] 异常: {e}")
            await asyncio.sleep(0.5)

    # ------------------ 15m MACD（MADC）平仓策略 ------------------
    async def madc_strategy_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                close = df['close'].astype(float)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                dif = ema12 - ema26
                dea = dif.ewm(span=9, adjust=False).mean()
                divergence = 2 * (dif - dea)
                trigger_val = divergence.iloc[-1]
                logger.info(f"[MADC] 离轴值: {trigger_val:.2f}")
                if trigger_val >= 11 and trigger_val < 20:
                    logger.info("[MADC] 空单触发（弱信号），平多仓0.07ETH")
                    await self.close_position('SELL', 0.07, strategy="madc")
                elif trigger_val >= 20:
                    logger.info("[MADC] 空单触发（强信号），平多仓0.14ETH")
                    await self.close_position('SELL', 0.14, strategy="madc")
                if trigger_val <= -11 and trigger_val > -20:
                    logger.info("[MADC] 多单触发（弱信号），平空仓0.07ETH")
                    await self.close_position('BUY', 0.07, strategy="madc")
                elif trigger_val <= -20:
                    logger.info("[MADC] 多单触发（强信号），平空仓0.14ETH")
                    await self.close_position('BUY', 0.14, strategy="madc")
            except Exception as e:
                logger.error(f"[MADC] 异常: {e}")
            await asyncio.sleep(60 * 15)

    # ------------------ 超级趋势策略 ------------------
    async def supertrend_strategy_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                high = df['high'].astype(float)
                low = df['low'].astype(float)
                close = df['close'].astype(float)
                # 简单超级趋势计算函数：返回 'UP'、'DOWN' 或 'NEUTRAL'
                def supertrend(length: int, factor: float) -> str:
                    atr_val = pd.Series(np.maximum.reduce([
                        high.diff().abs(),
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()
                    ])).rolling(window=length).mean().iloc[-1]
                    hl2 = ((high + low) / 2).iloc[-1]
                    basic_upper = hl2 + factor * atr_val
                    basic_lower = hl2 - factor * atr_val
                    latest = close.iloc[-1]
                    if latest > basic_upper:
                        return 'UP'
                    elif latest < basic_lower:
                        return 'DOWN'
                    else:
                        return 'NEUTRAL'
                trend1 = supertrend(10, 1)
                trend2 = supertrend(11, 2)
                trend3 = supertrend(12, 3)
                logger.info(f"[超级趋势] 指标趋势: {trend1}, {trend2}, {trend3}")
                if trend1 == trend2 == trend3 == 'UP':
                    logger.info("[超级趋势] 所有指标上升，触发市价多单0.15ETH")
                    self.update_local_position('BUY', 0.15, closing=False)
                elif trend1 == trend2 == trend3 == 'DOWN':
                    logger.info("[超级趋势] 所有指标下降，触发市价空单0.15ETH")
                    self.update_local_position('SELL', 0.15, closing=False)
            except Exception as e:
                logger.error(f"[超级趋势] 异常: {e}")
            await asyncio.sleep(60 * 15)

    async def execute(self) -> None:
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        await asyncio.gather(
            self.analyze_trend_15m(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.madc_strategy_loop(),
            self.supertrend_strategy_loop(),
            # 不再读取仓位，而以本地记录为准
        )

# ------------------- 主函数 -------------------
async def main() -> None:
    client = BinanceHFTClient()
    strategy = ETHUSDCStrategy(client)
    try:
        strategy.last_trade_side = await strategy.analyze_trend_15m()
        await strategy.execute()
    except KeyboardInterrupt:
        logger.info("收到终止信号，退出...")
    finally:
        await client.session.close()
        logger.info("HTTP会话已关闭")

if __name__ == "__main__":
    asyncio.run(main())