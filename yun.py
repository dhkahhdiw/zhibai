#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 高频 ETHUSDC 合约策略
"""

import os, time, asyncio, logging
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from ta.volatility import BollingerBands, AverageTrueRange

# ─── 配置 ─────────────────────────────────────────────────────────────
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY    = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL     = 'ETHUSDC'
LEVERAGE   = 50
SYMBOL       = 'ETHUSDC'
INTERVAL_3M  = Client.KLINE_INTERVAL_3MINUTE
INTERVAL_1H  = Client.KLINE_INTERVAL_1HOUR
INTERVAL_15M = Client.KLINE_INTERVAL_15MINUTE

# 初始化客户端
client = Client(API_KEY, API_SEC, futures=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── 工具函数 ───────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int=100) -> pd.DataFrame:
    """拉取 K 线，返回 DataFrame with columns [open, high, low, close, volume]"""
    data = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','num_trades','taker_base','taker_quote','ignore'
    ])
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def supertrend(df: pd.DataFrame, length=10, mult=3.0):
    """自定义 SuperTrend，返回 (st_line, direction_series)"""
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=length).average_true_range()
    hl2 = (df['high'] + df['low'])/2
    upper = hl2 + mult*atr
    lower = hl2 - mult*atr

    st = pd.Series(index=df.index)
    direction = pd.Series(1, index=df.index)
    for i in range(len(df)):
        if i==0:
            st.iat[0] = upper.iat[0]
            continue
        prev = st.iat[i-1]
        ub, lb, price = upper.iat[i], lower.iat[i], df['close'].iat[i]
        # 1) 本期基础线
        st.iat[i] = ub if price <= prev else lb
        # 2) 平滑
        if prev==upper.iat[i-1] and ub<prev: st.iat[i]=prev
        if prev==lower.iat[i-1] and lb>prev: st.iat[i]=prev
        # 3) 方向
        direction.iat[i] = 1 if price>st.iat[i] else -1
    return st, direction

def bb_pctb(df: pd.DataFrame, length=20, std=2.0) -> pd.Series:
    """计算 %B = (price−lower)/(upper−lower)"""
    bb = BollingerBands(df['close'], window=length, window_dev=std)
    return (df['close'] - bb.bollinger_lower()) / (bb.bollinger_upper() - bb.bollinger_lower())

async def place_oco(symbol, side, qty, entry_price, tp_offsets, sl_offset):
    """下 OCO：先限价入场，再挂止盈/止损限价单"""
    try:
        # 1) 挂入场限价单
        client.futures_create_order(
            symbol=symbol, side=side, type='LIMIT',
            timeInForce='GTC', quantity=qty, price=f"{entry_price:.4f}"
        )
        logging.info(f"OCO 挂 entry {side} {qty}@{entry_price:.4f}")
        # 2) 挂止盈/止损 OCO
        tp_price = entry_price * (1 + tp_offsets[0])  # 取第一个止盈档
        sl_price = entry_price * (1 + sl_offset)
        client.futures_create_order(
            symbol=symbol,
            side='SELL' if side=='BUY' else 'BUY',
            type='TAKE_PROFIT_LIMIT',
            quantity=qty,
            price=f"{tp_price:.4f}",
            stopPrice=f"{sl_price:.4f}",
            timeInForce='GTC',
            reduceOnly=True,
            workingType='CONTRACT_PRICE'
        )
        logging.info(f"OCO 挂 TP@{tp_price:.4f} SL@{sl_price:.4f}")
    except BinanceAPIException as e:
        logging.error(f"OCO 下单失败: {e}")

# ─── 核心信号逻辑 ────────────────────────────────────────────────────────
last_direction = None  # +1 多下过，就要等 -1 空触发后再多；-1 同理

def signal_main(df3, df1h, st_dir):
    """返回 (side, qty, entry_price, tp_offsets, sl_offset) 或 None"""
    global last_direction
    pctb3 = bb_pctb(df3).iat[-1]
    pctb1h = bb_pctb(df1h).iat[-1]

    # 1) 强弱信号
    long_str  = 'strong' if pctb1h<0.2 else 'weak'
    short_str = 'strong' if pctb1h>0.8 else 'weak'

    # 2) 方向
    trend = 1 if st_dir.iat[-1]>0 else -1
    # 3) 判断入场
    side = None; qty = 0; entry = df3['close'].iat[-1]
    tp_offsets = [0.0102,0.0123,0.015,0.018,0.022]  # 1.02%...2.2%
    sl_offset  = -0.02 if side=='BUY' else 0.02

    # 同趋势轮流：若上次多，下次仅可空
    if trend==1:
        if pctb3<=0 or pctb3>=1:
            if long_str=='strong':
                side, qty = ('BUY', 0.12)
            elif long_str=='weak':
                side, qty = ('BUY', 0.03)
    else:
        if pctb3<=0 or pctb3>=1:
            if short_str=='strong':
                side, qty = ('SELL', 0.12)
            elif short_str=='weak':
                side, qty = ('SELL', 0.03)

    # 4) 轮流触发
    if side:
        dir_val = 1 if side=='BUY' else -1
        if last_direction==dir_val:
            return None
        last_direction = dir_val
        return side, qty, entry, tp_offsets, (-0.02 if side=='BUY' else +0.02)
    return None

# ─── 主循环 ────────────────────────────────────────────────────────────
async def main_loop():
    logging.info("策略启动，1s 刷新")
    while True:
        # 并行拿数据
        df3, df1h, df15 = await asyncio.gather(
            asyncio.to_thread(fetch_klines, SYMBOL, INTERVAL_3M, 50),
            asyncio.to_thread(fetch_klines, SYMBOL, INTERVAL_1H, 50),
            asyncio.to_thread(fetch_klines, SYMBOL, INTERVAL_15M, 50),
        )

        # SuperTrend 主趋势
        _, st_dir = supertrend(df15, length=10, mult=3.0)

        sig = signal_main(df3, df1h, st_dir)
        if sig:
            side, qty, entry, tps, slo = sig
            await asyncio.to_thread(place_oco, SYMBOL, side, qty, entry, tps, slo)

        await asyncio.sleep(1)

if __name__=="__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logging.info("手动终止策略")