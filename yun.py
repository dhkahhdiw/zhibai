#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yun.py — 实时跟踪版：WebSocket + REST OTOCO
  • 实时订阅 ETHUSDC 3m/15m/1h K 线
  • 收盘事件触发指标更新 & 策略信号
  • 一次性 OTOCO 挂单框架（入场限价 + 止损限价 + 止盈限价）
  • 多空交替执行，防止重复下单
  • 加载 /root/zhibai/.env，适配 Vultr VHF‑1C‑1GB Ubuntu22.04
"""

import os, time, asyncio, logging
from collections import deque
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as pta
from ta.trend import MACD
from ta.volatility import BollingerBands
from binance.client import Client
from binance.exceptions import BinanceAPIException
import websockets

# ─── 配置 ──────────────────────────────────────────────────────────────
load_dotenv('/root/zhibai/.env')
API_KEY = os.getenv('BINANCE_API_KEY')
API_SEC = os.getenv('BINANCE_SECRET_KEY')
SYMBOL      = 'ETHUSDC'
STREAM_URL  = f"wss://fstream.binance.com/stream?streams={SYMBOL.lower()}@kline_3m/{SYMBOL.lower()}@kline_15m/{SYMBOL.lower()}@kline_1h"
MAX_BARS    = 500

# OTOCO 参数
# 示例： off_sl=0.98 表示止损触发价 = entry*0.98； off_tp=1.02 表示止盈触发价 = entry*1.02
# OTOCO 下单路径参考 币安更新日志：POST /api/v3/orderList/otoco 币安更新文档.txt](file-service://file-JHjASW8P4vZphWZjZwbZSR)

# 策略参数
BB_PERIOD   = 20; BB_STD    = 2
MACD_FAST   = 12; MACD_SLOW  = 26; MACD_SIGNAL = 9

# 初始化 Binance 客户端（支持 futures_* 方法）
client = Client(API_KEY, API_SEC)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ─── 全局状态 ───────────────────────────────────────────────────────────
ohlc = {
    '3m':  deque(maxlen=MAX_BARS),
    '15m': deque(maxlen=MAX_BARS),
    '1h':  deque(maxlen=MAX_BARS),
}
last_main = last_macd = last_st3 = None

# ─── 辅助：REST 拉取历史 K 线 ───────────────────────────────────────────
def fetch_klines(interval):
    data = client.futures_klines(symbol=SYMBOL, interval=interval, limit=MAX_BARS)
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','tbav','tqav','ignore'
    ])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

def init_ohlc():
    for tf in ('3m','15m','1h'):
        interval = {'3m':'3m','15m':'15m','1h':'1h'}[tf]
        df = fetch_klines(interval)
        for _, row in df.iterrows():
            ohlc[tf].append({
                'open': row.open, 'high': row.high,
                'low': row.low,   'close': row.close
            })

# ─── OTOCO 下单 ─────────────────────────────────────────────────────────
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
        'stopLimitTimeInForce': 'GTC',
        'takeProfitPrice': str(tp_price),
        'takeProfitLimitPrice': str(tp_limit),
        'takeProfitTimeInForce': 'GTC',
        'newOrderRespType': 'RESULT',
    }
    try:
        res = client.futures_post('orderList/otoco', params)
        logging.info(f"OTOCO下单[{side}] entry={entry}@{qty}, SL@{sl_price}, TP@{tp_price}")
        return res
    except BinanceAPIException as e:
        logging.error(f"OTOCO下单失败: {e.code} {e.message}")
        return None

# ─── 策略信号 ───────────────────────────────────────────────────────────
def main_signal(df3m, df1h, df15):
    # 15m SuperTrend via pandas_ta
    st15 = pta.supertrend(df15['high'], df15['low'], df15['close'], length=10, multiplier=3.0)
    sup = st15["SUPERT_10_3.0"].iat[-1]
    trend = 1 if df15['close'].iat[-1] > sup else -1

    # 1h %B
    bb1 = BollingerBands(df1h['close'], window=BB_PERIOD, window_dev=BB_STD)
    up1, lo1 = bb1.bollinger_hband().iat[-1], bb1.bollinger_lband().iat[-1]
    p1 = (df1h['close'].iat[-1] - lo1) / (up1 - lo1) if up1>lo1 else 0.5

    # 3m %B
    bb3 = BollingerBands(df3m['close'], window=BB_PERIOD, window_dev=BB_STD)
    up3, lo3 = bb3.bollinger_hband().iat[-1], bb3.bollinger_lband().iat[-1]
    p3 = (df3m['close'].iat[-1] - lo3) / (up3 - lo3) if up3>lo3 else 0.5

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

# ─── WebSocket 处理 ─────────────────────────────────────────────────────
async def ws_loop():
    global last_main, last_macd, last_st3
    # 初始化历史
    init_ohlc()
    logging.info("WebSocket 连接中…")
    async with websockets.connect(STREAM_URL) as ws:
        async for msg in ws:
            data = await asyncio.wrap_future(asyncio.get_event_loop().run_in_executor(None, lambda: None))  # no-op to yield
            msg = await ws.recv()
            stream = await asyncio.get_event_loop().run_in_executor(None, lambda: msg)
            import json
            payload = json.loads(stream)['data']['k']
            tf = payload['i']  # "3m"/"15m"/"1h"
            if payload['x']:  # 收盘
                bar = {
                    'open':  float(payload['o']),
                    'high':  float(payload['h']),
                    'low':   float(payload['l']),
                    'close': float(payload['c']),
                }
                ohlc[tf].append(bar)

                # 构建 DataFrame
                df3  = pd.DataFrame(ohlc['3m'])
                df15 = pd.DataFrame(ohlc['15m'])
                df1h = pd.DataFrame(ohlc['1h'])

                # 主策略
                side, strength, qty = main_signal(df3, df1h, df15)
                if side and side != last_main:
                    entry = df3['close'].iat[-1]
                    off_sl = 0.98 if side=='long' else 1.02
                    off_tp = 1.02 if side=='long' else 0.98
                    logging.info(f"[主] 信号 {side} 强度={strength} qty={qty}")
                    if await place_otoco(side, qty, entry, off_sl, off_tp):
                        last_main = side

                # MACD 子策略
                m_side, m_qty = macd_signal(df15)
                if m_side and m_side != last_macd:
                    entry = df15['close'].iat[-1]
                    off_sl = 0.97 if m_side=='long' else 1.03
                    off_tp = 1.00
                    logging.info(f"[MACD] 信号 {m_side} qty={m_qty}")
                    if await place_otoco(m_side, m_qty, entry, off_sl, off_tp):
                        last_macd = m_side

                # 三因子 SuperTrend 子策略
                st_side, st_qty = st3_signal(df15)
                if st_side and st_side != last_st3:
                    entry = df15['close'].iat[-1]
                    off_sl = 0.97 if st_side=='long' else 1.03
                    off_tp = 1.01 if st_side=='long' else 0.99
                    logging.info(f"[ST3] 信号 {st_side} qty={st_qty}")
                    if await place_otoco(st_side, st_qty, entry, off_sl, off_tp):
                        last_st3 = st_side

async def main():
    try:
        await ws_loop()
    except KeyboardInterrupt:
        logging.info("策略已终止")

if __name__ == '__main__':
    asyncio.run(main())