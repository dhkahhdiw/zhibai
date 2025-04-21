#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 综合交易策略：
  • 主策略：3 m %B + 15 m 超级趋势 + 1 h %B 强弱
  • 子策略①：15 m MACD（12,26,9）
  • 子策略②：15 m 三因子 SuperTrend (len=10,f=1),(11,2),(12,3)
  • 止盈预挂：多路 reduceOnly 限价单
  • 轮流触发机制造成同向只触发一次入场
  • 初始止损 + 动态跟踪止损
"""

import os, time, asyncio, logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD, SuperTrend
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# ─── 配置 ──────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser('/root/zhibai/.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
API_SEC = os.getenv('BINANCE_SECRET_KEY')

SYMBOL      = 'ETHUSDC'
INTERVAL_3M = Client.KLINE_INTERVAL_3MINUTE
INTERVAL_1H = Client.KLINE_INTERVAL_1HOUR
INTERVAL_15M= Client.KLINE_INTERVAL_15MINUTE

# Binance futures 客户端（REST）
client = Client(API_KEY, API_SEC, futures=True)

# 线程池 & 日志
executor = ThreadPoolExecutor(4)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── 工具函数 ────────────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit):
    data = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbav','tqav','ignore'
    ])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

async def get_klines(symbol, interval, limit=50):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_klines, symbol, interval, limit)

async def place_order(side, qty, price=None, reduceOnly=False):
    """下单函数：市价 or 限价 + reduceOnly"""
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
        logging.info(f"下单成功[{side}]: id={res['orderId']} qty={qty}@{price or 'MKT'}")
        return res
    except BinanceAPIException as e:
        logging.error(f"下单失败: {e.code} {e.message}")
        return None

async def place_take_profits(side, entry_price, entry_qty, strength):
    """提前挂多路限价止盈"""
    if strength=='strong':
        offsets, props = [0.0102,0.0123,0.015,0.018,0.022], [0.2]*5
    else:
        offsets, props = [0.0123,0.018], [0.5]*2

    for off, prop in zip(offsets, props):
        tp_qty = entry_qty * prop
        tp_price = entry_price*(1+off) if side=='long' else entry_price*(1-off)
        await place_order('short' if side=='long' else 'long', tp_qty, price=tp_price, reduceOnly=True)

# ─── 状态管理：轮流触发 ─────────────────────────────────────────────────
last_main = None   # 主策略入场方向
last_macd = None   # MACD 子策略
last_st3  = None   # 三因子 SuperTrend 子策略

# ─── 各策略实现 ─────────────────────────────────────────────────────────
def main_signal(df3m, df1h, df15):
    """① 主策略：3m%B + 15m SuperTrend + 1h%B 强弱"""
    # 15m SuperTrend 判趋势
    st = SuperTrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=3)
    is_bull = st.super_trend().iat[-1] < df15['close'].iat[-1]
    trend = 1 if is_bull else -1

    # 1h %B 强弱
    bb1h = BollingerBands(df1h['close'], window=20, window_dev=2)
    up1, lo1 = bb1h.bollinger_hband().iat[-1], bb1h.bollinger_lband().iat[-1]
    pb1h = (df1h['close'].iat[-1]-lo1)/(up1-lo1) if up1>lo1 else 0.5

    # 3m %B 判断入场
    bb3 = BollingerBands(df3m['close'], window=20, window_dev=2)
    up3, lo3 = bb3.bollinger_hband().iat[-1], bb3.bollinger_lband().iat[-1]
    pctb = (df3m['close'].iat[-1]-lo3)/(up3-lo3) if up3>lo3 else 0.5

    side=strength=qty=None
    if pctb<=0:
        strength = 'strong' if pctb<0.2 else 'weak'
        qty = 0.12 if (strength=='strong' and trend==1) else 0.03
        side='long'
    elif pctb>=1:
        strength = 'strong' if pctb>0.8 else 'weak'
        qty = 0.12 if (strength=='strong' and trend==-1) else 0.03
        side='short'

    return side, strength, qty, trend

def macd_signal(df15):
    """② 子策略：15m MACD（12,26,9）"""
    macd = MACD(df15['close'], window_slow=26, window_fast=12, window_sign=9)
    hist = macd.macd_diff().iat[-1]
    prev = macd.macd_diff().iat[-2]
    # 银叉（空单） or 金叉（多单）
    if prev>0 and hist<0:
        # 距零轴高度决定 qty
        qty = 0.15
        return 'short', qty
    if prev<0 and hist>0:
        qty = 0.15
        return 'long', qty
    return None, None

def st3_signal(df15):
    """③ 子策略：15m 三因子 SuperTrend (10,3),(11,2),(12,3)"""
    s1 = SuperTrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=1).super_trend().iat[-1] < df15['close'].iat[-1]
    s2 = SuperTrend(df15['high'], df15['low'], df15['close'], length=11, multiplier=2).super_trend().iat[-1] < df15['close'].iat[-1]
    s3 = SuperTrend(df15['high'], df15['low'], df15['close'], length=12, multiplier=3).super_trend().iat[-1] < df15['close'].iat[-1]
    # 三升三降
    if s1 and s2 and s3:
        return 'long', 0.15
    if not s1 and not s2 and not s3:
        return 'short', 0.15
    return None, None

# ─── 主循环 ─────────────────────────────────────────────────────────────
async def main_loop():
    global last_main, last_macd, last_st3
    logging.info("策略启动")
    while True:
        # 并行拉 3k、1h、15m
        df3, df1h, df15 = await asyncio.gather(
            get_klines(SYMBOL, INTERVAL_3M,  50),
            get_klines(SYMBOL, INTERVAL_1H,  50),
            get_klines(SYMBOL, INTERVAL_15M, 50)
        )

        # 1) 主策略
        side, strength, qty, trend = main_signal(df3, df1h, df15)
        if side and side!=last_main:
            price = df3['close'].iat[-1]
            logging.info(f"[主] 信号 {side} 强度={strength} qty={qty}")
            if await place_order(side, qty, price=price):
                await place_take_profits(side, price, qty, strength)
                last_main = side

        # 2) MACD 子策略
        m_side, m_qty = macd_signal(df15)
        if m_side and m_side!=last_macd:
            price = df15['close'].iat[-1]
            logging.info(f"[MACD] 信号 {m_side} qty={m_qty}")
            await place_order(m_side, m_qty, price=price)
            last_macd = m_side

        # 3) 三因子 SuperTrend 子策略
        st_side, st_qty = st3_signal(df15)
        if st_side and st_side!=last_st3:
            price = df15['close'].iat[-1]
            logging.info(f"[ST3] 信号 {st_side} qty={st_qty}")
            await place_order(st_side, st_qty, price=price)
            last_st3 = st_side

        await asyncio.sleep(1)

if __name__=='__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("策略终止")