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
load_dotenv(os.path.expanduser('~/.env'))
API_KEY = os.getenv('BINANCE_API_KEY')
API_SEC = os.getenv('BINANCE_SECRET_KEY')

SYMBOL       = 'ETHUSDC'
INTERVAL_3M  = Client.KLINE_INTERVAL_3MINUTE
INTERVAL_1H  = Client.KLINE_INTERVAL_1HOUR
INTERVAL_15M = Client.KLINE_INTERVAL_15MINUTE

# 初始化客户端（同步版）
client = Client(API_KEY, API_SEC)
# 防止析构时报 session 不存在
client.session = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── 工具函数 ───────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int=100) -> pd.DataFrame:
    data = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','num_trades','taker_base','taker_quote','ignore'
    ])
    return df[['open','high','low','close','volume']].astype(float)

def supertrend(df: pd.DataFrame, length=10, mult=3.0):
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
        st.iat[i] = ub if price <= prev else lb
        if prev==upper.iat[i-1] and ub<prev: st.iat[i]=prev
        if prev==lower.iat[i-1] and lb>prev: st.iat[i]=prev
        direction.iat[i] = 1 if price>st.iat[i] else -1
    return st, direction

def bb_pctb(df: pd.DataFrame, length=20, std=2.0) -> pd.Series:
    bb = BollingerBands(df['close'], window=length, window_dev=std)
    return (df['close'] - bb.bollinger_lower()) / (bb.bollinger_upper() - bb.bollinger_lower())

async def place_oco(symbol, side, qty, entry_price, tp_offsets, sl_offset):
    try:
        # 入场限价
        client.futures_create_order(
            symbol=symbol, side=side, type='LIMIT',
            timeInForce='GTC', quantity=qty, price=f"{entry_price:.4f}"
        )
        logging.info(f"OCO 挂 entry {side} {qty}@{entry_price:.4f}")
        # 止盈止损
        tp_price = entry_price * (1 + tp_offsets[0])
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

# ─── 信号逻辑 ───────────────────────────────────────────────────────────
last_direction = None  # 轮流触发

def signal_main(df3, df1h, st_dir):
    global last_direction
    pctb3  = bb_pctb(df3).iat[-1]
    pctb1h = bb_pctb(df1h).iat[-1]
    trend = 1 if st_dir.iat[-1]>0 else -1

    # 强弱判断
    long_str  = 'strong' if pctb1h<0.2 else 'weak'
    short_str = 'strong' if pctb1h>0.8 else 'weak'

    side, qty, entry = None, 0, df3['close'].iat[-1]
    tp_offsets = [0.0102,0.0123,0.015,0.018,0.022]

    # 3m %B≤0 或 ≥1
    if pctb3<=0 or pctb3>=1:
        if trend==1:
            if long_str=='strong': side, qty = 'BUY', 0.12
            elif long_str=='weak': side, qty = 'BUY', 0.03
        else:
            if short_str=='strong': side, qty = 'SELL', 0.12
            elif short_str=='weak': side, qty = 'SELL', 0.03

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
        df3, df1h, df15 = await asyncio.gather(
            asyncio.to_thread(fetch_klines, SYMBOL, INTERVAL_3M, 50),
            asyncio.to_thread(fetch_klines, SYMBOL, INTERVAL_1H, 50),
            asyncio.to_thread(fetch_klines, SYMBOL, INTERVAL_15M, 50),
        )
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