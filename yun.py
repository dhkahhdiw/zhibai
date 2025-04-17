#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 综合交易策略：
  • 主策略：3 m %B + 15 m SuperTrend + 1 h %B 强弱
  • 子策略①：15 m MACD（12,26,9）
  • 子策略②：15 m 三因子 SuperTrend (10,3),(11,2),(12,3)
  • 止盈预挂：多路 reduceOnly 限价单 + 固定止损
  • 轮流触发机制造成同向只触发一次入场
  • 主循环 1 秒刷新
"""

import os, asyncio, logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# ─── 配置 ──────────────────────────────────────────────────────────────
load_dotenv(os.path.expanduser('~/.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
API_SEC = os.getenv('BINANCE_SECRET_KEY')

SYMBOL       = 'ETHUSDC'
INT_3M       = Client.KLINE_INTERVAL_3MINUTE
INT_1H       = Client.KLINE_INTERVAL_1HOUR
INT_15M      = Client.KLINE_INTERVAL_15MINUTE

# Binance REST 客户端
client = Client(API_KEY, API_SEC)
# 防止析构时报 session 错误
client.session = None

# 日志 & 线程池
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
executor = ThreadPoolExecutor(4)

# ─── 数据拉取 & 指标 ────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit):
    data = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(data, columns=[
        'ot','open','high','low','close','vol','ct','qav','trades','tbav','tqav','ignore'
    ])
    for col in ('open','high','low','close','vol'):
        df[col] = df[col].astype(float)
    return df

async def get_klines(symbol, interval, limit=50):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_klines, symbol, interval, limit)

def supertrend(df, length, mult):
    """自定义 SuperTrend，返回 direction Series (+1 up / -1 down)"""
    hl2 = (df['high']+df['low'])/2
    atr = df['high'].combine(df['low'], max) - df['high'].combine(df['low'], min)
    atr = atr.rolling(length).mean()  # 简化 ATR
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr
    st = pd.Series(0, index=df.index)
    dir = pd.Series(1, index=df.index)
    for i in range(len(df)):
        if i==0:
            st.iat[0] = upper.iat[0]
            continue
        prev, up, lo, price = st.iat[i-1], upper.iat[i], lower.iat[i], df['close'].iat[i]
        st.iat[i] = up if price<=prev else lo
        dir.iat[i] = 1 if price>st.iat[i] else -1
    return dir

def bb_pct(df, length=20, dev=2.0):
    bb = BollingerBands(df['close'], window=length, window_dev=dev)
    bot, top = bb.bollinger_lower(), bb.bollinger_hband()
    return (df['close'] - bot) / (top - bot)

# ─── 下单 & 预挂止盈止损 ────────────────────────────────────────────────
async def place_bracket_order(side, qty, entry_price, strength):
    """
    • 先以限价下入场单，
    • 然后提前挂 reduceOnly 止盈多路限价单 + 固定比例止损
    """
    try:
        # 入场限价
        res = client.futures_create_order(
            symbol=SYMBOL,
            side='BUY' if side=='long' else 'SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity=qty,
            price=f"{entry_price:.4f}",
            positionSide='LONG' if side=='long' else 'SHORT'
        )
        logging.info(f"[ENTRY] {side} qty={qty}@{entry_price:.4f}")
        # 挂止盈
        if strength=='strong':
            offs = [0.0102,0.0123,0.015,0.018,0.022]; props=[0.2]*5
        else:
            offs = [0.0123,0.018];           props=[0.5]*2
        for off, prop in zip(offs, props):
            tp_price = entry_price*(1+off) if side=='long' else entry_price*(1-off)
            tp_qty   = qty * prop
            client.futures_create_order(
                symbol=SYMBOL,
                side='SELL' if side=='long' else 'BUY',
                type='LIMIT',
                timeInForce='GTC',
                quantity=round(tp_qty,6),
                price=f"{tp_price:.4f}",
                reduceOnly=True
            )
            logging.info(f"[TP] {side} qty={tp_qty}@{tp_price:.4f}")
        # 挂固定止损
        sl_off = -0.02 if side=='long' else +0.02
        sl_price = entry_price*(1+sl_off)
        client.futures_create_order(
            symbol=SYMBOL,
            side='SELL' if side=='long' else 'BUY',
            type='STOP_MARKET',
            stopPrice=f"{sl_price:.4f}",
            quantity=qty,
            reduceOnly=True
        )
        logging.info(f"[SL] {side} qty={qty}@{sl_price:.4f}")
    except BinanceAPIException as e:
        logging.error(f"Bracket 下单失败: {e.code} {e.message}")

# ─── 轮流触发 State ────────────────────────────────────────────────────
last_main = None
last_macd = None
last_st3 = None

# ─── 各策略信号 ────────────────────────────────────────────────────────
def main_signal(df3, df1h, df15):
    # 趋势：15m SuperTrend
    dir15 = supertrend(df15, 10, 3).iat[-1]
    trend = 1 if dir15>0 else -1

    # 1h %B 强弱
    pb1h = bb_pct(df1h).iat[-1]
    long_str = 'strong' if pb1h<0.2 else 'weak'
    short_str= 'strong' if pb1h>0.8 else 'weak'

    # 3m %B 入场
    pb3 = bb_pct(df3).iat[-1]
    if pb3<=0:
        side='long'; strength=long_str
        qty = 0.12 if (strength=='strong' and trend==1) else 0.03
    elif pb3>=1:
        side='short'; strength=short_str
        qty = 0.12 if (strength=='strong' and trend==-1) else 0.03
    else:
        return None, None, None, None

    return side, strength, qty, trend

def macd_signal(df15):
    macd = MACD(df15['close'], 26,12,9).macd_diff()
    prev, cur = macd.iat[-2], macd.iat[-1]
    if prev>0 and cur<0: return 'short', 0.15
    if prev<0 and cur>0: return 'long',  0.15
    return None, None

def st3_signal(df15):
    dirs = [
        supertrend(df15,10,1).iat[-1],
        supertrend(df15,11,2).iat[-1],
        supertrend(df15,12,3).iat[-1],
    ]
    if all(d>0 for d in dirs): return 'long',  0.15
    if all(d<0 for d in dirs): return 'short', 0.15
    return None, None

# ─── 主循环 ─────────────────────────────────────────────────────────────
async def main_loop():
    global last_main, last_macd, last_st3
    logging.info("策略启动，1s 刷新")
    while True:
        df3, df1h, df15 = await asyncio.gather(
            get_klines(SYMBOL, INT_3M,  50),
            get_klines(SYMBOL, INT_1H,  50),
            get_klines(SYMBOL, INT_15M, 50),
        )

        # ─ 主策略
        side, strength, qty, trend = main_signal(df3, df1h, df15)
        if side and side!=last_main:
            price = df3['close'].iat[-1]
            logging.info(f"[主] 信号 {side} 强度={strength} qty={qty}")
            await place_bracket_order(side, qty, price, strength)
            last_main = side

        # ─ MACD 子策略
        m_side, m_qty = macd_signal(df15)
        if m_side and m_side!=last_macd:
            price = df15['close'].iat[-1]
            logging.info(f"[MACD] 信号 {m_side} qty={m_qty}")
            await place_bracket_order(m_side, m_qty, price, 'strong')
            last_macd = m_side

        # ─ 三因子 SuperTrend
        st_side, st_qty = st3_signal(df15)
        if st_side and st_side!=last_st3:
            price = df15['close'].iat[-1]
            logging.info(f"[ST3] 信号 {st_side} qty={st_qty}")
            await place_bracket_order(st_side, st_qty, price, 'strong')
            last_st3 = st_side

        await asyncio.sleep(1)

if __name__=='__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("手动终止策略")