#!/usr/bin/env python3
"""
ETH/USDC 高频交易引擎 v7.4（优化版）
------------------------------------------------------------
策略逻辑：
【一】多时间框架趋势判断：
  - 主趋势：15m超级趋势指标（20周期MA + 3×ATR）为主；当价格高于超级趋势线且为绿色，趋势为上升；低于且为红色，趋势为下降。
  - 辅助：当趋势边界时，叠加15m MACD（12,26,9）柱状线与零轴关系确认趋势。
【二】强弱信号订单机制：
  - 根据1h布林带%B(20,2)：多单若%B < 0.2为强信号，否则为弱；空单若%B > 0.8为强信号，否则为弱。
【三】挂单与止盈止损：
  - 3m布林带%B(20,2)触发下单：若%B ≤ 0触发买单，≥ 1触发卖单。
  - 下单规模按信号与趋势的关系：
      同趋势：强信号 0.12 ETH，弱信号 0.03 ETH；
      异趋势：强信号 0.07 ETH，弱信号 0.015 ETH。
  - 下单挂单价格：强信号使用5档（偏移 0.25%, 0.40%, 0.60%, 0.80%, 1.60%，各20%）；弱信号使用2档（0.25%，1.60%，各50%）。
  - 止盈挂单方案：强信号分5档（偏移 0.07%, 0.07%, 0.03%, 0.25%, 1.10%，各20%）；弱信号类似设置。
  - 止损：初始设置为买价×0.98 / 卖价×1.02，并以3m布林带带宽动态调整。
【四】信号轮换：连续同向信号忽略，下单后冷却3秒，特殊情况下首次强信号允许下单。
【五】保留15m MACD策略，用于在趋势反转时市价平仓。
【六】仓位控制：
  - 增强实时仓位同步，通过/fapi/v2/positionRisk获得仓位数据，用于指导平仓和仓位控制；
  - 趋势上升时多仓上限0.49 ETH、空仓上限0.35 ETH；趋势下降时多仓0.35 ETH、空仓0.49 ETH；超出部分及时平仓。
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

# 仓位控制目标（后续趋势中更新）
BULL_LONG_LIMIT  = 0.49
BULL_SHORT_LIMIT = 0.35
BEAR_LONG_LIMIT  = 0.35
BEAR_SHORT_LIMIT = 0.49

# 订单基础规模（根据信号与趋势方向决定）
STRONG_SIZE_SAME  = 0.12
STRONG_SIZE_DIFF  = 0.07
WEAK_SIZE_SAME    = 0.03
WEAK_SIZE_DIFF    = 0.015

# 下单挂单方案：返回挂单偏移及比例
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

# 交易及限频相关配置
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
    order_adjust_interval: float = 1.0  # 每秒检测下单信号

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
        params = {'symbol': SYMBOL, 'leverage': LEVERAGE, 'dualSidePosition': self.config.dual_side_position}
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

    async def fetch_position(self) -> Dict[str, float]:
        """通过 /fapi/v2/positionRisk 获取实时仓位数据；返回{'long': 数量, 'short': 数量}"""
        params = {'symbol': SYMBOL}
        data = await self._signed_request('GET', '/positionRisk', params)
        for pos in data:
            if pos.get("symbol") == SYMBOL:
                pos_amt = float(pos.get("positionAmt", "0"))
                return {'long': pos_amt if pos_amt > 0 else 0, 'short': -pos_amt if pos_amt < 0 else 0}
        return {'long': 0, 'short': 0}

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
        self.last_trade_side: str = None    # 当前趋势方向，'LONG'或'SHORT'
        self.last_triggered_side: str = None  # 上次下单方向，用于冷却判断
        self.last_order_time: float = 0       # 下单时间，用于冷却判断
        self.entry_price: float = None        # 上次入场价格
        # 实时仓位（由实时更新同步）
        self.current_long: float = 0.0
        self.current_short: float = 0.0
        # 趋势下单控制上限
        self.max_long: float = 0.0
        self.max_short: float = 0.0
        # 用于15m MACD策略平仓：保留原3m仓位数据
        self.macd_long: float = 0.0
        self.macd_short: float = 0.0
        self.prev_macd_off: float = None

    # ------------------ 趋势判断 ------------------
    async def analyze_trend_15m(self) -> str:
        df = await self.client.fetch_klines(interval='15m', limit=100)
        if df.empty:
            return 'LONG'
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        # 计算ATR：将 numpy 数组转换为 pd.Series 才能使用 rolling
        tr = np.maximum.reduce([
            high.diff().abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ])
        atr = pd.Series(tr).rolling(window=self.config.st_period).mean().iloc[-1]
        hl2 = (high + low) / 2
        basic_upper = hl2 + self.config.st_multiplier * atr
        basic_lower = hl2 - self.config.st_multiplier * atr

        latest = close.iloc[-1]
        if latest > basic_upper:
            return 'LONG'
        elif latest < basic_lower:
            return 'SHORT'
        else:
            ema_fast = close.ewm(span=self.config.macd_fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.config.macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            return 'LONG' if macd_line.iloc[-1] >= 0 else 'SHORT'

    # ------------------ 信号强弱判断 ------------------
    async def get_hourly_strength(self, side: str) -> bool:
        """
        利用1h布林带%B(20,2)判断信号强弱：
         - 对于BUY信号，若%B < 0.2则为强信号；
         - 对于SELL信号，若%B > 0.8则为强信号；
         返回True表示强信号，否则为弱信号。
        """
        df = await self.client.fetch_klines(interval='1h', limit=50)
        if df.empty:
            return False
        close = df['close'].astype(float)
        sma = close.rolling(window=self.config.bb_period).mean()
        std = close.rolling(window=self.config.bb_period).std()
        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std
        percent_b = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1]-lower.iloc[-1])>0 else 0.5
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
        percent_b = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1]-lower.iloc[-1]) if (upper.iloc[-1]-lower.iloc[-1])>0 else 0.5
        logger.info(f"[3m%B] 当前 %B: {percent_b:.3f}")
        if percent_b <= 0:
            return Signal(True, 'BUY', {'trigger_price': close.iloc[-1]}), {}
        elif percent_b >= 1:
            return Signal(True, 'SELL', {'trigger_price': close.iloc[-1]}), {}
        return Signal(False, 'NONE'), {}

    # ------------------ 实时仓位同步 ------------------
    async def update_positions_from_exchange(self) -> None:
        try:
            pos = await self.client.fetch_position()
            self.current_long = pos.get('long', 0)
            self.current_short = pos.get('short', 0)
            logger.info(f"[仓位同步] 多仓：{self.current_long:.4f} ETH，空仓：{self.current_short:.4f} ETH")
        except Exception as e:
            logger.error(f"[仓位同步] 获取异常：{e}")

    async def position_update_loop(self) -> None:
        while True:
            await self.update_positions_from_exchange()
            await asyncio.sleep(10)

    # ------------------ 挂单下单及止盈止损 ------------------
    async def place_dynamic_limit_orders(self, side: str, order_list: List[Dict[str, Any]], trigger_price: float, order_size: float) -> None:
        pos_side = "LONG" if side == "BUY" else "SHORT"
        for order in order_list:
            offset = order['offset']
            ratio = order['ratio']
            qty = round(order_size * ratio, self.config.quantity_precision)
            if qty <= 0:
                continue
            limit_price = round(trigger_price * (1 - offset) if side == "BUY" else trigger_price * (1 + offset), self.config.price_precision)
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
        except Exception as e:
            logger.error(f"[平仓] 失败: {e}")

    # ------------------ 趋势封控 ------------------
    async def trend_control_loop(self) -> None:
        while True:
            try:
                if self.last_trade_side == 'LONG' and self.current_short > (0.75 * self.current_long):
                    excess = self.current_short - (0.75 * self.current_long)
                    logger.info(f"[趋势封控] 空仓超限 {excess:.4f} ETH，平部分空仓")
                    await self.close_position(side='BUY', ratio=excess, strategy="normal")
                elif self.last_trade_side == 'SHORT' and self.current_long > (0.75 * self.current_short):
                    excess = self.current_long - (0.75 * self.current_short)
                    logger.info(f"[趋势封控] 多仓超限 {excess:.4f} ETH，平部分多仓")
                    await self.close_position(side='SELL', ratio=excess, strategy="normal")
            except Exception as e:
                logger.error(f"[趋势封控] 异常: {e}")
            await asyncio.sleep(30)

    # ------------------ 信号下单逻辑 ------------------
    async def order_signal_loop(self) -> None:
        while True:
            try:
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
                    await self.close_position(side='SELL', ratio=self.current_long, strategy="normal")
                elif self.last_trade_side == 'SHORT' and latest > dynamic_stop:
                    logger.info("[止损] 空单止损触发")
                    await self.close_position(side='BUY', ratio=self.current_short, strategy="normal")
            except Exception as e:
                logger.error(f"[止盈止损] 异常: {e}")
            await asyncio.sleep(0.5)

    # ------------------ 15m MACD平仓策略 ------------------
    async def macd_strategy_loop(self) -> None:
        while True:
            try:
                df = await self.client.fetch_klines(interval='15m', limit=100)
                if df.empty:
                    await asyncio.sleep(60)
                    continue
                close = df['close'].astype(float)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                curr = macd.iloc[-1]
                logger.info(f"[MACD] 当前MACD：{curr:.2f}")
                if self.prev_macd_off is not None:
                    if self.prev_macd_off < 0 and curr >= 0:
                        logger.info("[MACD] 由负转正，触发市价平空")
                        await self.close_position(side='BUY', ratio=self.current_short, strategy="macd")
                    elif self.prev_macd_off > 0 and curr <= 0:
                        logger.info("[MACD] 由正转负，触发市价平多")
                        await self.close_position(side='SELL', ratio=self.current_long, strategy="macd")
                self.prev_macd_off = curr
            except Exception as e:
                logger.error(f"[MACD] 异常: {e}")
            await asyncio.sleep(60 * 15)

    async def execute(self) -> None:
        await self.client.sync_server_time()
        await self.client.manage_leverage()
        await asyncio.gather(
            self.analyze_trend_15m(),
            self.order_signal_loop(),
            self.stop_loss_profit_management_loop(),
            self.macd_strategy_loop(),
            self.trend_control_loop(),
            self.position_update_loop()
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