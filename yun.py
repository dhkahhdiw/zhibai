#!/usr/bin/env python3
# yun.py — 主策略：3m%B + 15m超级趋势 + 1h%B 强弱信号 + 止盈预挂

import os
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import SuperTrend
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# ─── 配置 ──────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser('~/.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
API_SEC = os.getenv('BINANCE_SECRET_KEY')
SYMBOL  = 'ETHUSDC'
INTERVAL_3M  = Client.KLINE_INTERVAL_3MINUTE
INTERVAL_1H  = Client.KLINE_INTERVAL_1HOUR
INTERVAL_15M = Client.KLINE_INTERVAL_15MINUTE

# 多线程包装同步接口
executor = ThreadPoolExecutor(max_workers=4)
client   = Client(API_KEY, API_SEC, futures=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ─── 工具函数 ────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """
    同步拉取 K 线，lookback 单位：条数
    """
    data = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbav','tqav','ignore'
    ])
    df['close'] = df['close'].astype(float)
    return df

async def get_klines(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, fetch_klines, symbol, interval, lookback
    )

def compute_main_signal(df3m: pd.DataFrame, trend: int, pb1h: float):
    """
    3m %B 主策略信号：
    返回 (side, strength, qty) 或 (None, None, None)
      side in {'long','short'}
      strength in {'strong','weak'}
      qty: ETH 数量
    """
    bb3 = BollingerBands(df3m['close'], window=20, window_dev=2)
    close = df3m['close'].iat[-1]
    upper = bb3.bollinger_hband().iat[-1]
    lower = bb3.bollinger_lband().iat[-1]
    pct_b = (close - lower) / (upper - lower) if upper>lower else 0.5

    # 多单信号
    if pct_b <= 0:
        strength = 'strong' if pct_b<0.2 else 'weak'
        qty = 0.12 if strength=='strong' and trend==1 else 0.03
        return 'long', strength, qty if trend==1 else qty
    # 空单信号
    if pct_b >= 1:
        strength = 'strong' if pct_b>0.8 else 'weak'
        qty = 0.12 if strength=='strong' and trend==-1 else 0.03
        return 'short', strength, qty if trend==-1 else qty

    return None, None, None

async def place_order(side: str, qty: float, price: float = None, reduceOnly: bool = False):
    """
    下单：市价或限价；reduceOnly 用于止盈单
    """
    params = dict(
        symbol=SYMBOL,
        side='BUY' if side=='long' else 'SELL',
        type='LIMIT' if price else 'MARKET',
        quantity=round(qty,6),
        positionSide='LONG' if side=='long' else 'SHORT',
        reduceOnly=reduceOnly
    )
    if price:
        params.update(timeInForce='GTC', price=str(round(price,2)))
    try:
        res = client.futures_create_order(**params)
        logging.info(f"下单成功: {res['orderId']} {side} qty={qty}@{price or 'MKT'}")
        return res
    except BinanceAPIException as e:
        logging.error(f"下单失败: {e.code} {e.message}")
        return None

async def place_take_profits(side: str, entry_price: float, entry_qty: float, strength: str):
    """
    提前挂多路限价止盈单 (reduceOnly=True)
    强信号: offsets=[1.02%,1.23%,1.5%,1.8%,2.2%], props=20%
    弱信号: offsets=[1.23%,1.8%], props=50%
    """
    if strength=='strong':
        offsets = [0.0102,0.0123,0.0150,0.0180,0.0220]
        props   = [0.2]*5
    else:
        offsets = [0.0123,0.0180]
        props   = [0.5]*2

    for off, prop in zip(offsets, props):
        tp_qty = entry_qty * prop
        if side=='long':
            tp_price = entry_price * (1 + off)
            await place_order('short', tp_qty, price=tp_price, reduceOnly=True)
        else:
            tp_price = entry_price * (1 - off)
            await place_order('long', tp_qty, price=tp_price, reduceOnly=True)

# ─── 主循环 ─────────────────────────────────────────────────────────────
last_main_side = None  # 轮流触发：上一次入场方向

async def main_loop():
    global last_main_side
    logging.info("策略启动: 3m%B + 15m SuperTrend + 1h%B 强弱 + 提前止盈挂单")
    while True:
        # 并行拉三条 K 线
        df15, df1h, df3 = await asyncio.gather(
            get_klines(SYMBOL, INTERVAL_15M,  50),
            get_klines(SYMBOL, INTERVAL_1H,   50),
            get_klines(SYMBOL, INTERVAL_3M,   50)
        )

        # 1) 15m SuperTrend 判主趋势
        st = SuperTrend(
            high=df15['high'], low=df15['low'], close=df15['close'],
            length=10, multiplier=3
        )
        super_trend = st.super_trend()
        trend = 1 if super_trend.iat[-1] < df15['close'].iat[-1] else -1

        # 2) 1h %B 强弱信号
        bb1h = BollingerBands(df1h['close'], window=20, window_dev=2)
        upper1h = bb1h.bollinger_hband().iat[-1]
        lower1h = bb1h.bollinger_lband().iat[-1]
        close1h = df1h['close'].iat[-1]
        pb1h = (close1h - lower1h) / (upper1h - lower1h) if upper1h>lower1h else 0.5

        # 3) 3m 主信号
        side, strength, qty = compute_main_signal(df3, trend, pb1h)
        if side and side != last_main_side:
            entry_price = df3['close'].iat[-1]
            logging.info(f"主信号触发: {side} 强度={strength} qty={qty} 价格={entry_price}")
            order = await place_order(side, qty, price=entry_price)
            if order:
                await place_take_profits(side, entry_price, qty, strength)
                last_main_side = side

        await asyncio.sleep(1)

# ─── 启动 ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("手动终止")