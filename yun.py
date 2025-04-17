#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版交易策略 — 主要修复：
1. 超级趋势指标自定义实现
2. 修复Binance客户端初始化
3. 增强异步处理性能
4. 兼容TA 0.11.0库
"""

import os
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.volatility import average_true_range
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# ─── 配置优化 ──────────────────────────────────────────────────────────────
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL = 'ETHUSDT'
LEVERAGE = 50
INTERVALS = {
    '3m': '3m',
    '15m': '15m',
    '1h': '1h'
}

# 异步客户端初始化
client = None
executor = ThreadPoolExecutor(max_workers=8)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)s] %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/root/zhibai/trading.log'),
        logging.StreamHandler()
    ]
)


# ─── 自定义指标实现 ────────────────────────────────────────────────────────
def calculate_super_trend(df, period=10, factor=3):
    """
    自定义超级趋势指标
    返回添加 super_trend 和 trend_direction 列的DataFrame
    """
    hl2 = (df['high'] + df['low']) / 2
    df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=period)

    df['upper_band'] = hl2 + (factor * df['atr'])
    df['lower_band'] = hl2 - (factor * df['atr'])

    st = np.zeros(len(df))
    direction = np.zeros(len(df))

    for i in range(1, len(df)):
        close = df['close'].iloc[i]
        prev_upper = df['upper_band'].iloc[i - 1]
        prev_lower = df['lower_band'].iloc[i - 1]
        prev_st = st[i - 1]

        if close > prev_upper:
            st[i] = df['lower_band'].iloc[i]
            direction[i] = 1
        elif close < prev_lower:
            st[i] = df['upper_band'].iloc[i]
            direction[i] = -1
        else:
            st[i] = prev_st
            direction[i] = direction[i - 1]

    df['super_trend'] = st
    df['trend_direction'] = direction
    return df


# ─── 异步工具函数优化 ──────────────────────────────────────────────────────
async def get_klines(symbol, interval, limit=50):
    """异步获取K线数据并计算指标"""
    try:
        client = await AsyncClient.create(API_KEY, SECRET_KEY)
        klines = await client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        await client.close_connection()

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'tbav', 'tqav', 'ignore'
        ]).astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })

        # 计算必要指标
        if interval == '15m':
            df = calculate_super_trend(df)

        return df
    except Exception as e:
        logging.error(f"数据获取失败: {str(e)}")
        return None


async def place_order(side, qty, price=None, reduce_only=False):
    """增强版异步下单函数"""
    try:
        client = await AsyncClient.create(API_KEY, SECRET_KEY)

        order_params = {
            'symbol': SYMBOL,
            'side': 'BUY' if side == 'long' else 'SELL',
            'type': 'LIMIT' if price else 'MARKET',
            'quantity': round(qty, 3),
            'positionSide': 'LONG' if side == 'long' else 'SHORT',
            'reduceOnly': reduce_only,
            'price': round(price, 2) if price else None,
            'timeInForce': 'GTC' if price else 'IOC'
        }

        # 过滤空值参数
        order_params = {k: v for k, v in order_params.items() if v is not None}

        response = await client.futures_create_order(**order_params)
        logging.info(f"订单成功 {side}@{qty}: {response}")
        return response
    except BinanceAPIException as e:
        logging.error(f"下单错误 {e.code}: {e.message}")
        return None
    finally:
        await client.close_connection()


# ─── 策略信号生成优化 ─────────────────────────────────────────────────────
def main_signal(df_3m, df_1h, df_15m):
    """主策略信号生成"""
    try:
        # 15分钟趋势判断
        trend = df_15m['trend_direction'].iloc[-1]

        # 1小时布林带强度
        bb_1h = BollingerBands(df_1h['close'], 20, 2)
        pb_1h = (df_1h['close'].iloc[-1] - bb_1h.bollinger_lband().iloc[-1]) / \
                (bb_1h.bollinger_hband().iloc[-1] - bb_1h.bollinger_lband().iloc[-1])

        # 3分钟布林带信号
        bb_3m = BollingerBands(df_3m['close'], 20, 2)
        pb_3m = (df_3m['close'].iloc[-1] - bb_3m.bollinger_lband().iloc[-1]) / \
                (bb_3m.bollinger_hband().iloc[-1] - bb_3m.bollinger_lband().iloc[-1])

        signal = None
        strength = None
        qty = 0

        if pb_3m <= 0:
            strength = 'strong' if pb_1h < 0.2 else 'weak'
            qty = 0.12 if (trend == 1 and strength == 'strong') else 0.03
            signal = 'long'
        elif pb_3m >= 1:
            strength = 'strong' if pb_1h > 0.8 else 'weak'
            qty = 0.12 if (trend == -1 and strength == 'strong') else 0.03
            signal = 'short'

        return signal, strength, qty
    except Exception as e:
        logging.error(f"信号生成错误: {str(e)}")
        return None, None, None


# ─── 主循环优化 ───────────────────────────────────────────────────────────
async def strategy_engine():
    """优化后的策略引擎"""
    state = {
        'last_main': None,
        'last_macd': None,
        'last_st': None
    }

    while True:
        try:
            # 并行获取数据
            df_3m, df_1h, df_15m = await asyncio.gather(
                get_klines(SYMBOL, INTERVALS['3m']),
                get_klines(SYMBOL, INTERVALS['1h']),
                get_klines(SYMBOL, INTERVALS['15m'])
            )

            # 主策略执行
            signal, strength, qty = main_signal(df_3m, df_1h, df_15m)
            if signal and signal != state['last_main']:
                price = df_3m['close'].iloc[-1]
                if await place_order(signal, qty, price=price):
                    state['last_main'] = signal

            await asyncio.sleep(0.5)  # 高频控制间隔

        except Exception as e:
            logging.critical(f"引擎异常: {str(e)}")
            await asyncio.sleep(5)


if __name__ == '__main__':
    # 性能优化配置
    import uvloop

    uvloop.install()

    # 启动策略
    try:
        asyncio.run(strategy_engine())
    except KeyboardInterrupt:
        logging.info("策略正常退出")