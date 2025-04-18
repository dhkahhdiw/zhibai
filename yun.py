#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 纯 REST 拉取版：轮询更新实时价格 & 指标，OCO 挂单
  • 不使用 WebSocket，定时拉取 K 线 & 最新价格
  • 3m/15m/1h K 线历史 + 实时价更新未完周期
  • 实时计算 SuperTrend、%B、MACD 等指标
  • 收盘新条后自动滚动 OHLC 窗口
  • 一次性 OTOCO 挂单（LIMIT entry + STOP_LIMIT SL + TAKE_PROFIT_LIMIT TP）
  • 多空交替，防止重复下单
  • 强制加载 /root/zhibai/.env，适配 Ubuntu22.04
"""

import os, time, asyncio, logging
from math import floor
from collections import deque
from dotenv import load_dotenv

import pandas as pd
import pandas_ta as pta
from ta.trend import MACD
from ta.volatility import BollingerBands
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ─── 配置 ──────────────────────────────────────────────────────────────
load_dotenv('/root/zhibai/.env')
API_KEY = os.getenv('BINANCE_API_KEY')
API_SEC = os.getenv('BINANCE_SECRET_KEY')
SYMBOL     = 'ETHUSDC'
PERIODS    = {'3m':180, '15m':900, '1h':3600}  # seconds
MAX_BARS   = 500

# 策略参数
BB_PERIOD    = 20; BB_STD     = 2
MACD_FAST    = 12; MACD_SLOW   = 26; MACD_SIGNAL = 9

# OTOCO endpoint (一次性提交 OCO bracket 单) 币安更新文档.txt](file-service://file-JHjASW8P4vZphWZjZwbZSR)
# python-binance 方法: client.futures_order_otoco(**params)

client = Client(API_KEY, API_SEC)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── 全局 OHLC 缓存 ─────────────────────────────────────────────────────
ohlc = {tf: deque(maxlen=MAX_BARS) for tf in PERIODS}  # each entry: dict with open_time, open, high, low, close

last_main = last_macd = last_st3 = None

# ─── 初始化历史 K 线 ───────────────────────────────────────────────────
def init_ohlc():
    for tf, sec in PERIODS.items():
        bars = client.futures_klines(symbol=SYMBOL, interval=tf, limit=MAX_BARS)
        for b in bars:
            ohlc[tf].append({
                'open_time': int(b[0]),
                'open':  float(b[1]),
                'high':  float(b[2]),
                'low':   float(b[3]),
                'close': float(b[4]),
            })

# ─── 实时 OTOCO 挂单 ───────────────────────────────────────────────────
async def place_otoco(side, qty, entry, off_sl, off_tp):
    side_s = 'BUY' if side=='long' else 'SELL'
    sl_price = round(entry * off_sl, 2)
    sl_limit = round(sl_price * (0.995 if side=='long' else 1.005), 2)
    tp_price = round(entry * off_tp, 2)
    tp_limit = round(tp_price * (0.995 if side=='short' else 1.005), 2)
    params = {
        'symbol': SYMBOL,
        'side': side_s,
        'quantity': round(qty,6),
        'price': str(round(entry,2)),
        'stopPrice': str(sl_price),
        'stopLimitPrice': str(sl_limit),
        'stopLimitTimeInForce':'GTC',
        'takeProfitPrice': str(tp_price),
        'takeProfitLimitPrice': str(tp_limit),
        'takeProfitTimeInForce':'GTC',
        'newOrderRespType':'RESULT',
    }
    try:
        res = client.futures_order_otoco(**params)
        logging.info(f"OTOCO 下单[{side}]: entry={entry}@{qty}, SL@{sl_price}, TP@{tp_price}")
        return res
    except BinanceAPIException as e:
        logging.error(f"OTOCO 下单失败: {e.code} {e.message}")
        return None

# ─── 指标计算 & 信号 ───────────────────────────────────────────────────
def main_signal(df3, df1h, df15):
    # SuperTrend 15m
    st15 = pta.supertrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=3.0)
    sup = st15["SUPERT_10_3.0"].iat[-1]
    trend = 1 if df15['close'].iat[-1] > sup else -1

    # 1h %B
    bb1 = BollingerBands(df1h['close'], window=BB_PERIOD, window_dev=BB_STD)
    up1, lo1 = bb1.bollinger_hband().iat[-1], bb1.bollinger_lband().iat[-1]
    p1 = (df1h['close'].iat[-1]-lo1)/(up1-lo1) if up1>lo1 else 0.5

    # 3m %B
    bb3 = BollingerBands(df3['close'], window=BB_PERIOD, window_dev=BB_STD)
    up3, lo3 = bb3.bollinger_hband().iat[-1], bb3.bollinger_lband().iat[-1]
    p3 = (df3['close'].iat[-1]-lo3)/(up3-lo3) if up3>lo3 else 0.5

    if p3 <= 0:
        strength = 'strong' if p3 < 0.2 else 'weak'
        qty = 0.12 if (strength=='strong' and trend==1) else 0.03
        return 'long', strength, qty
    if p3 >= 1:
        strength = 'strong' if p3 > 0.8 else 'weak'
        qty = 0.12 if (strength=='strong' and trend==-1) else 0.03
        return 'short', strength, qty
    return None, None, None

def macd_signal(df15):
    m = MACD(df15['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    h, p = m.macd_diff().iat[-1], m.macd_diff().iat[-2]
    if p>0 and h<0: return 'short', 0.15
    if p<0 and h>0: return 'long',  0.15
    return None, None

def st3_signal(df15):
    s1 = df15['close'].iat[-1] > pta.supertrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=1.0)["SUPERT_10_1.0"].iat[-1]
    s2 = df15['close'].iat[-1] > pta.supertrend(df15['high'], df15['low'], df15['close'], length=11, multiplier=2.0)["SUPERT_11_2.0"].iat[-1]
    s3 = df15['close'].iat[-1] > pta.supertrend(df15['high'], df15['low'], df15['close'], length=12, multiplier=3.0)["SUPERT_12_3.0"].iat[-1]
    if s1 and s2 and s3: return 'long',  0.15
    if not s1 and not s2 and not s3: return 'short', 0.15
    return None, None

# ─── 周期轮询 & 实时更新 ─────────────────────────────────────────────────
async def poll_loop():
    global last_main, last_macd, last_st3
    init_ohlc()
    logging.info("轮询启动：每秒更新最新价，每分钟拉取 K 线")
    last_kline_update = time.time()
    while True:
        now = time.time()
        # 每分钟更新历史 K 线（闭盘后新条）
        if now - last_kline_update >= 60:
            init_ohlc()
            last_kline_update = now

        # 获取最新价格
        price = float(client.futures_symbol_ticker(symbol=SYMBOL)['price'])
        ts_ms = int(now*1000)
        # 更新各周期未完当前条
        for tf, sec in PERIODS.items():
            dq = ohlc[tf]
            if not dq: continue
            bar = dq[-1]
            # 判断是否跨周期，需新建条
            if ts_ms >= bar['open_time'] + sec*1000:
                # 新条 open 用前 close
                new_open = bar['close']
                new_ot = floor(now / sec)*sec*1000
                dq.append({
                    'open_time': new_ot,
                    'open': new_open,
                    'high': new_open,
                    'low': new_open,
                    'close': new_open,
                })
                bar = dq[-1]
            # 更新高低收
            bar['close'] = price
            bar['high']  = max(bar['high'], price)
            bar['low']   = min(bar['low'], price)

        # 构建 DataFrame
        df3  = pd.DataFrame(ohlc['3m'])
        df15 = pd.DataFrame(ohlc['15m'])
        df1h = pd.DataFrame(ohlc['1h'])

        # 主策略
        side, strength, qty = main_signal(df3, df1h, df15)
        if side and side != last_main:
            entry = price
            off_sl = 0.98 if side=='long' else 1.02
            off_tp = 1.02 if side=='long' else 0.98
            logging.info(f"[主] 信号 {side} 强度={strength} qty={qty}")
            if await place_otoco(side, qty, entry, off_sl, off_tp):
                last_main = side

        # MACD 子策略
        m_side, m_qty = macd_signal(df15)
        if m_side and m_side != last_macd:
            entry = price
            off_sl = 0.97 if m_side=='long' else 1.03
            off_tp = 1.00
            logging.info(f"[MACD] 信号 {m_side} qty={m_qty}")
            if await place_otoco(m_side, m_qty, entry, off_sl, off_tp):
                last_macd = m_side

        # 三因子 SuperTrend
        st_side, st_qty = st3_signal(df15)
        if st_side and st_side != last_st3:
            entry = price
            off_sl = 0.97 if st_side=='long' else 1.03
            off_tp = 1.01 if st_side=='long' else 0.99
            logging.info(f"[ST3] 信号 {st_side} qty={st_qty}")
            if await place_otoco(st_side, st_qty, entry, off_sl, off_tp):
                last_st3 = st_side

        await asyncio.sleep(1)

if __name__=='__main__':
    try:
        asyncio.run(poll_loop())
    except KeyboardInterrupt:
        logging.info("策略终止")