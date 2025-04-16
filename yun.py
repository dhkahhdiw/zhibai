#!/usr/bin/env python3
import os, time, asyncio, logging
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# --------------------
# 环境加载与基本配置
# --------------------
ENV_PATH = '/root/zhibai/.env'
load_dotenv(ENV_PATH)
API_KEY    = os.getenv('BINANCE_API_KEY', '').strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '').strip()
SYMBOL     = 'ETHUSDC'  # 合约交易对

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
client = Client(API_KEY, SECRET_KEY)
# 使用 futures 模块接口时，python-binance 的 Client 已支持 futures_create_order 等方法
executor = ThreadPoolExecutor(max_workers=4)

# --------------------
# 全局控制变量
# --------------------
order_lock = asyncio.Lock()    # 防重入锁
position_tracker = {'long': 0.0, 'short': 0.0}
last_order_side = None  # 记录上一次下单方向（'long'或'short'）
last_main_trend = 1     # 默认主趋势为上升

# --------------------
# 余额检查（使用合约账户余额查询）
# --------------------
def get_future_balance(asset='USDC'):
    try:
        bal_list = client.futures_account_balance()
        for b in bal_list:
            if b['asset'] == asset:
                return float(b['balance'])
        return 0.0
    except Exception as e:
        logging.error(f"获取合约{asset}余额异常: {e}")
        return 0.0

def sufficient_balance(order_side, quantity, price):
    """
    对于 BUY（多单），检查 USDC 余额是否足够支付成本（quantity × price）；
    对于 SELL（空单），检查 ETH 持仓（未平仓）是否足够。
    这里只做简单检查，实际合约中需结合仓位和杠杆管理。
    """
    if order_side.upper() == 'BUY':
        usdc = get_future_balance('USDC')
        cost = quantity * price
        if usdc >= cost:
            return True
        else:
            logging.error(f"USDC余额不足: {usdc}, 需要成本: {cost}")
            return False
    elif order_side.upper() == 'SELL':
        # futures持仓查询，此处简单取本地记录
        if position_tracker['long'] >= quantity:
            return True
        else:
            logging.error(f"持仓不足（多单平仓时）：当前多仓 {position_tracker['long']}, 需要: {quantity}")
            return False
    return False

# --------------------
# 数据获取函数
# --------------------
def get_klines(symbol, interval, lookback_minutes):
    end_time = int(time.time() * 1000)
    start_time = end_time - lookback_minutes * 60 * 1000
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_time, endTime=end_time)
    except BinanceAPIException as e:
        logging.error(f"Kline获取错误: {e}")
        return None
    df = pd.DataFrame(klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_asset_volume','number_of_trades',
        'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

async def async_get_klines(symbol, interval, lookback_minutes):
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(executor, get_klines, symbol, interval, lookback_minutes)
        return df
    except Exception as e:
        logging.error(f"Async Kline错误: {e}")
        return None

# --------------------
# 指标计算函数
# --------------------
def compute_supertrend(df, period=20, multiplier=3):
    df = df.copy()
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period)
    df['ATR'] = atr.average_true_range()
    df['ma'] = df['close'].rolling(window=period).mean()
    df['upper_band'] = df['ma'] + multiplier * df['ATR']
    df['lower_band'] = df['ma'] - multiplier * df['ATR']
    supertrend = [np.nan] * len(df)
    trend = [0] * len(df)
    for i in range(period, len(df)):
        if i == period:
            if df.loc[df.index[i-1],'close'] > df.loc[df.index[i-1],'ma']:
                trend[i-1] = 1
                supertrend[i-1] = df.loc[df.index[i-1],'lower_band']
            else:
                trend[i-1] = -1
                supertrend[i-1] = df.loc[df.index[i-1],'upper_band']
        curr_close = df.loc[df.index[i],'close']
        prev_supertrend = supertrend[i-1]
        prev_trend = trend[i-1]
        curr_lower = df.loc[df.index[i],'lower_band']
        curr_upper = df.loc[df.index[i],'upper_band']
        if prev_trend == 1:
            curr_lower = max(curr_lower, prev_supertrend)
        elif prev_trend == -1:
            curr_upper = min(curr_upper, prev_supertrend)
        if curr_close > curr_lower:
            trend[i] = 1
            supertrend[i] = curr_lower
        elif curr_close < curr_upper:
            trend[i] = -1
            supertrend[i] = curr_upper
        else:
            trend[i] = prev_trend
            supertrend[i] = prev_supertrend
    df['supertrend'] = supertrend
    df['trend'] = trend
    return df

def compute_macd(df, fast=12, slow=26, signal=9):
    macd = MACD(close=df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df

def compute_bollinger_percent_b(df, window=20, std_multiplier=2):
    bb = BollingerBands(close=df['close'], window=window, window_dev=std_multiplier)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df

# --------------------
# 信号生成函数
# --------------------
def generate_main_trend_signal(df_15m):
    df = compute_supertrend(df_15m, period=20, multiplier=3)
    df = compute_macd(df, fast=12, slow=26, signal=9)
    latest = df.iloc[-1]
    if latest['trend'] == 1 and latest['macd_diff'] > 0:
        result = 1
    elif latest['trend'] == -1 and latest['macd_diff'] < 0:
        result = -1
    else:
        result = 0
    global last_main_trend
    if result == 0:
        result = last_main_trend if last_main_trend is not None else 1
    last_main_trend = result
    return result

def generate_strength_signal(df_1h):
    df = compute_bollinger_percent_b(df_1h, window=20, std_multiplier=2)
    pb = df.iloc[-1]['percent_b']
    return {'long_strength': 'strong' if pb < 0.2 else 'weak',
            'short_strength': 'strong' if pb > 0.8 else 'weak'}

def generate_3m_trade_orders(df_3m, current_price, trend, strength, side):
    orders = []
    grid_levels = [0.25, 0.4, 0.6, 0.8, 1.6]
    if strength == 'strong':
        base_qty = 0.12 if side=='long' and trend==1 else 0.07
    else:
        base_qty = 0.03 if side=='long' else 0.015
    for level in grid_levels:
        if side == 'long':
            order_price = current_price * (1 - level/100)
        else:
            order_price = current_price * (1 + level/100)
        orders.append({
            'price_offset_pct': level,
            'order_price': order_price,
            'quantity': base_qty,
            'side': side
        })
    return {'orders': orders}

# --------------------
# 下单与仓位管理（异步封装，采用 futures 接口和余额检查）
# --------------------
async def async_place_order(order_side, order_type, quantity, price=None):
    async with order_lock:
        loop = asyncio.get_running_loop()
        try:
            # 使用余额检查避免触发余额不足错误
            if order_side.upper() == 'BUY':
                if not sufficient_balance('BUY', quantity, price):
                    logging.error("买单余额不足，跳过下单")
                    return None
            elif order_side.upper() == 'SELL':
                if not sufficient_balance('SELL', quantity, price):
                    logging.error("卖单（平仓）余额不足，跳过下单")
                    return None
            res = await loop.run_in_executor(executor, place_order, order_side, order_type, quantity, price)
            if res is None:
                logging.error("下单返回空结果")
            else:
                logging.info(f"下单成功：orderId={res.get('orderId','N/A')}")
            return res
        except Exception as e:
            logging.exception(f"async_place_order异常: {e}")
            return None

def place_order(order_side, order_type, quantity, price=None):
    try:
        # 对于合约交易，调用 futures_create_order
        if order_type == 'LIMIT' and price is not None:
            order = client.futures_create_order(
                symbol=SYMBOL,
                side=order_side,
                type=order_type,
                timeInForce='GTC',
                quantity=quantity,
                price=str(round(price, 2))
            )
        else:
            order = client.futures_create_order(
                symbol=SYMBOL,
                side=order_side,
                type=order_type,
                quantity=quantity
            )
        return order
    except BinanceAPIException as e:
        logging.error(f"下单失败: {e}")
        return None

def update_position(side, quantity, action='open'):
    if side == 'long':
        if action == 'open':
            position_tracker['long'] += quantity
        else:
            position_tracker['long'] = max(position_tracker['long'] - quantity, 0)
    else:
        if action == 'open':
            position_tracker['short'] += quantity
        else:
            position_tracker['short'] = max(position_tracker['short'] - quantity, 0)
    logging.info(f"当前仓位: {position_tracker}")

# --------------------
# 止盈挂单函数（必须提前量挂单）
# --------------------
async def async_place_take_profit_orders(entry_price, side, signal_strength, base_quantity):
    loop = asyncio.get_running_loop()
    orders = []
    if signal_strength == 'strong':
        offsets = [1.02, 1.23, 1.5, 1.8, 2.2]
        order_prop = 0.20
    else:
        offsets = [1.23, 1.8]
        order_prop = 0.50
    for offset in offsets:
        if side == 'long':
            tp_price = entry_price * (1 + offset/100.0)
            tp_side = 'SELL'
        else:
            tp_price = entry_price * (1 - offset/100.0)
            tp_side = 'BUY'
        qty = base_quantity * order_prop
        orders.append({'order_side': tp_side, 'price': tp_price, 'quantity': qty, 'offset': offset})
    for order in orders:
        res = await async_place_order(order['order_side'], 'LIMIT', order['quantity'], order['price'])
        if res:
            logging.info(f"止盈订单成功：side={order['order_side']}, price={order['price']:.4f}, quantity={order['quantity']}, offset={order['offset']}%")
        else:
            logging.error(f"止盈订单失败：offset={order['offset']}%")
    return orders

# --------------------
# 止损及动态跟踪函数
# --------------------
def calculate_stop_loss(entry_price, side):
    return entry_price * (0.98 if side == 'long' else 1.02)

def trailing_stop_update(current_price, bb_width):
    return current_price - bb_width if bb_width else current_price

# --------------------
# 并行子策略（独立调度）
# --------------------
async def macd_strategy(df_15m):
    try:
        df_macd = compute_macd(df_15m.copy(), fast=12, slow=26, signal=9)
        macd_diff = df_macd.iloc[-1]['macd_diff']
        if 11 <= abs(macd_diff) < 20:
            logging.info("15m MACD策略：离轴11～20信号，触发约0.1ETH订单")
        elif abs(macd_diff) >= 20:
            logging.info("15m MACD策略：离轴≥20信号，触发约0.15ETH订单")
    except Exception as e:
        logging.exception(f"MACD子策略异常: {e}")

async def supertrend_strategy(df_15m):
    try:
        df_factor1 = compute_supertrend(df_15m.copy(), period=10, multiplier=1)
        df_factor2 = compute_supertrend(df_15m.copy(), period=11, multiplier=2)
        df_factor3 = compute_supertrend(df_15m.copy(), period=12, multiplier=3)
        t1, t2, t3 = df_factor1.iloc[-1]['trend'], df_factor2.iloc[-1]['trend'], df_factor3.iloc[-1]['trend']
        if t1 == t2 == t3 == 1:
            logging.info("超级趋势子策略：三指标均转绿，触发市价多单0.15ETH")
            res = await async_place_order('BUY', 'MARKET', 0.15)
            if res:
                update_position('long', 0.15, action='open')
        elif t1 == t2 == t3 == -1:
            logging.info("超级趋势子策略：三指标均转红，触发市价空单0.15ETH")
            res = await async_place_order('SELL', 'MARKET', 0.15)
            if res:
                update_position('short', 0.15, action='open')
    except Exception as e:
        logging.exception(f"超级趋势子策略异常: {e}")

# --------------------
# 主策略循环（异步调度，60秒刷新）
# --------------------
async def main_strategy_loop():
    global last_order_side
    while True:
        try:
            df_15m, df_1h, df_3m = await asyncio.gather(
                async_get_klines(SYMBOL, Client.KLINE_INTERVAL_15MINUTE, 300),
                async_get_klines(SYMBOL, Client.KLINE_INTERVAL_1HOUR, 1440),
                async_get_klines(SYMBOL, Client.KLINE_INTERVAL_3MINUTE, 600)
            )
            if df_15m is None or df_1h is None or df_3m is None:
                logging.warning("数据获取失败，等待重试")
                await asyncio.sleep(30)
                continue

            # 主趋势判断：15m超级趋势+MACD（强制返回1或-1）
            main_trend = generate_main_trend_signal(df_15m)
            logging.info(f"主趋势：{'上升' if main_trend==1 else '下降'}")

            # 强弱信号：1h Bollinger %B
            strength_info = generate_strength_signal(df_1h)
            logging.info(f"1小时信号: {strength_info}")

            # 最新3m数据：价格及 Bollinger %B
            current_price = df_3m.iloc[-1]['close']
            df_3m_bb = compute_bollinger_percent_b(df_3m.copy(), window=20, std_multiplier=2)
            latest_percent_b = df_3m_bb.iloc[-1]['percent_b']
            logging.info(f"最新价格: {current_price}, 3m %B: {latest_percent_b}")

            # 下单信号：当3m %B <= 0则触发多单，下单价格基于挂单网格计算；当 %B >= 1 触发空单
            order_side = None
            if latest_percent_b <= 0:
                order_side = 'long'
            elif latest_percent_b >= 1:
                order_side = 'short'
            else:
                logging.info("3m信号不满足下单条件")

            if order_side:
                if last_order_side == order_side:
                    logging.info("本轮已触发相同方向下单，跳过")
                else:
                    if order_side == 'long':
                        signal_strength = strength_info['long_strength']
                        qty = 0.12 if main_trend==1 and signal_strength=='strong' else 0.03
                    else:
                        signal_strength = strength_info['short_strength']
                        qty = 0.07 if main_trend==-1 and signal_strength=='strong' else 0.015

                    orders_dict = generate_3m_trade_orders(df_3m, current_price, trend=main_trend,
                                                           strength=signal_strength, side=order_side)
                    logging.info(f"生成挂单信号: {orders_dict}")
                    best_order = orders_dict['orders'][0]
                    side_str = 'BUY' if order_side=='long' else 'SELL'

                    # 仓位控制：对于上升趋势，多仓上限1.0，空仓上限0.75；下降趋势反之
                    if main_trend == 1:
                        if order_side=='long' and position_tracker['long'] >= 1.0:
                            logging.warning("多仓超限，暂停下单")
                        elif order_side=='short' and position_tracker['short'] >= 0.75:
                            logging.warning("空仓超限，暂停下单")
                        else:
                            order_res = await async_place_order(side_str, 'LIMIT', qty, best_order['order_price'])
                            if order_res:
                                update_position(order_side, qty, action='open')
                                last_order_side = order_side
                                entry_price = best_order['order_price']
                                sl_price = calculate_stop_loss(entry_price, order_side)
                                logging.info(f"入场价: {entry_price}, 初始止损价: {sl_price}")
                                await async_place_take_profit_orders(entry_price, order_side, signal_strength, qty)
                    else:
                        if order_side=='long' and position_tracker['long'] >= 0.75:
                            logging.warning("多仓超限，暂停下单")
                        elif order_side=='short' and position_tracker['short'] >= 1.0:
                            logging.warning("空仓超限，暂停下单")
                        else:
                            order_res = await async_place_order(side_str, 'LIMIT', qty, best_order['order_price'])
                            if order_res:
                                update_position(order_side, qty, action='open')
                                last_order_side = order_side
                                entry_price = best_order['order_price']
                                sl_price = calculate_stop_loss(entry_price, order_side)
                                logging.info(f"入场价: {entry_price}, 初始止损价: {sl_price}")
                                await async_place_take_profit_orders(entry_price, order_side, signal_strength, qty)
            # 当15m趋势变化时重置轮流记录
            if len(df_15m) >= 2 and df_15m.iloc[-2]['close'] != df_15m.iloc[-1]['close']:
                last_order_side = None

            # 并行调度子策略任务
            asyncio.create_task(macd_strategy(df_15m))
            asyncio.create_task(supertrend_strategy(df_15m))

            # 仓位控制提醒
            if main_trend == 1:
                if position_tracker['long'] > 1.0:
                    logging.warning("多仓超出上限")
                if position_tracker['short'] > 0.75:
                    logging.warning("空仓不足，请及时调仓")
            else:
                if position_tracker['long'] > 0.75:
                    logging.warning("多仓不足，请及时调仓")
                if position_tracker['short'] > 1.0:
                    logging.warning("空仓超出上限")
            # 止损动态跟踪（示意）：基于3m Bollinger带宽更新
            bb_width = df_3m_bb.iloc[-1]['bb_upper'] - df_3m_bb.iloc[-1]['bb_lower']
            trailing_sl = trailing_stop_update(current_price, bb_width)
            logging.info(f"动态跟踪止损价更新为: {trailing_sl}")

        except Exception as e:
            logging.exception(f"主策略异常: {e}")
        await asyncio.sleep(60)

# --------------------
# 主入口与调度
# --------------------
async def main():
    logging.info("启动合约ETH USDC固定盈利交易策略（高频REST版）...")
    await main_strategy_loop()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logging.exception(f"策略主程序异常: {e}")