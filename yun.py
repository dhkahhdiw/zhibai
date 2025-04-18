#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 一次性 OTOCO 挂单版
  • 主策略、MACD、三因子 SuperTrend 均改为 OTOCO
  • 确保多空交替，单向每轮只进一次
  • 适配 Binance Futures REST OTOCO 接口
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
# 强制指定 .env 路径
load_dotenv('/root/zhibai/.env')
API_KEY, API_SEC = os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY')

SYMBOL      = 'ETHUSDC'
INTERVAL_3M = Client.KLINE_INTERVAL_3MINUTE
INTERVAL_1H = Client.KLINE_INTERVAL_1HOUR
INTERVAL_15M= Client.KLINE_INTERVAL_15MINUTE

# futures 客户端
client = Client(API_KEY, API_SEC, futures=True)

executor = ThreadPoolExecutor(4)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── 工具：拉 K 线 ───────────────────────────────────────────────────────
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

# ─── 一次性 OTOCO 挂单 ───────────────────────────────────────────────────
async def place_otoco(side, qty, entry, sl_off, tp_off):
    """
    一次性提交 OTOCO:
     - price: 入场限价
     - stopPrice + stopLimitPrice: 止损
     - takeProfitPrice + takeProfitLimitPrice: 止盈
    """
    side_s = 'BUY' if side=='long' else 'SELL'
    # 止损：挂单价稍优于触发价
    sl_price = round(entry * sl_off, 2)
    sl_limit = round(sl_price * (0.995 if side=='long' else 1.005), 2)
    # 止盈：挂单价稍优于触发价
    tp_price = round(entry * tp_off, 2)
    tp_limit = round(tp_price * (0.995 if side=='short' else 1.005), 2)

    params = {
        'symbol': SYMBOL,
        'side': side_s,
        'quantity': round(qty,6),
        'price':      str(round(entry,2)),
        'stopPrice':  str(sl_price),
        'stopLimitPrice': str(sl_limit),
        'stopLimitTimeInForce':'GTC',
        'takeProfitPrice':     str(tp_price),
        'takeProfitLimitPrice':str(tp_limit),
        'takeProfitTimeInForce':'GTC',
        'newOrderRespType':'RESULT',
    }
    try:
        res = client.futures_post('/fapi/v1/orderList/otoco', params)
        logging.info(f"OTOCO 下单[{side}]: entry={entry}@{qty}, SL@{sl_price}, TP@{tp_price}")
        return res
    except BinanceAPIException as e:
        logging.error(f"OTOCO 下单失败: {e.code} {e.message}")
        return None

# ─── 策略信号 ───────────────────────────────────────────────────────────
def main_signal(df3m, df1h, df15):
    # 15m SuperTrend 趋势
    st = SuperTrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=3)
    bull = st.super_trend().iat[-1] < df15['close'].iat[-1]
    trend = 1 if bull else -1

    # 1h %B
    bb1h = BollingerBands(df1h['close'], window=20, window_dev=2)
    up1, lo1 = bb1h.bollinger_hband().iat[-1], bb1h.bollinger_lband().iat[-1]
    p1 = (df1h['close'].iat[-1]-lo1)/(up1-lo1) if up1>lo1 else 0.5

    # 3m %B
    bb3 = BollingerBands(df3m['close'], window=20, window_dev=2)
    up3, lo3 = bb3.bollinger_hband().iat[-1], bb3.bollinger_lband().iat[-1]
    p3 = (df3m['close'].iat[-1]-lo3)/(up3-lo3) if up3>lo3 else 0.5

    if p3<=0:
        strength = 'strong' if p3<0.2 else 'weak'
        qty = 0.12 if (strength=='strong' and trend==1) else 0.03
        return 'long', strength, qty
    if p3>=1:
        strength = 'strong' if p3>0.8 else 'weak'
        qty = 0.12 if (strength=='strong' and trend==-1) else 0.03
        return 'short', strength, qty
    return None, None, None

def macd_signal(df15):
    macd = MACD(df15['close'], window_slow=26, window_fast=12, window_sign=9)
    h, p = macd.macd_diff().iat[-1], macd.macd_diff().iat[-2]
    if p>0 and h<0: return 'short', 0.15
    if p<0 and h>0: return 'long',  0.15
    return None, None

def st3_signal(df15):
    s1 = SuperTrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=1).super_trend().iat[-1] < df15['close'].iat[-1]
    s2 = SuperTrend(df15['high'], df15['low'], df15['close'], length=11, multiplier=2).super_trend().iat[-1] < df15['close'].iat[-1]
    s3 = SuperTrend(df15['high'], df15['low'], df15['close'], length=12, multiplier=3).super_trend().iat[-1] < df15['close'].iat[-1]
    if s1 and s2 and s3: return 'long',  0.15
    if not s1 and not s2 and not s3: return 'short', 0.15
    return None, None

# ─── 状态管理：轮流触发 ─────────────────────────────────────────────────
last_main = None
last_macd = None
last_st3  = None

# ─── 主循环 ─────────────────────────────────────────────────────────────
async def main_loop():
    global last_main, last_macd, last_st3
    logging.info("策略启动，使用 OTOCO 挂单")
    while True:
        df3, df1h, df15 = await asyncio.gather(
            get_klines(SYMBOL, INTERVAL_3M, 50),
            get_klines(SYMBOL, INTERVAL_1H, 50),
            get_klines(SYMBOL, INTERVAL_15M,50),
        )

        # 主策略
        side, strength, qty = main_signal(df3, df1h, df15)
        if side and side!=last_main:
            entry_price = df3['close'].iat[-1]
            off_sl = 0.98 if side=='long' else 1.02
            off_tp = 1.02 if side=='long' else 0.98
            logging.info(f"[主] 信号 {side} 强度={strength} qty={qty}")
            if await place_otoco(side, qty, entry_price, off_sl, off_tp):
                last_main = side

        # MACD 子策略
        m_side, m_qty = macd_signal(df15)
        if m_side and m_side!=last_macd:
            entry_price = df15['close'].iat[-1]
            off_sl = 0.97 if m_side=='long' else 1.03
            off_tp = 1.00
            logging.info(f"[MACD] 信号 {m_side} qty={m_qty}")
            if await place_otoco(m_side, m_qty, entry_price, off_sl, off_tp):
                last_macd = m_side

        # 三因子 SuperTrend 子策略
        st_side, st_qty = st3_signal(df15)
        if st_side and st_side!=last_st3:
            entry_price = df15['close'].iat[-1]
            off_sl = 0.97 if st_side=='long' else 1.03
            off_tp = 1.01 if st_side=='long' else 0.99
            logging.info(f"[ST3] 信号 {st_side} qty={st_qty}")
            if await place_otoco(st_side, st_qty, entry_price, off_sl, off_tp):
                last_st3 = st_side

        await asyncio.sleep(1)

if __name__=='__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("策略已终止")