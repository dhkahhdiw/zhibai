#!/usr/bin/env python3
# coding: utf-8

import os, time, json, hmac, hashlib, asyncio, logging, uuid, urllib.parse, base64
import uvloop, aiohttp, pandas as pd, websockets
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# —— 高性能事件循环 ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv('/root/zhibai/.env')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    API_KEY          = os.getenv('BINANCE_API_KEY')
    SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode()
    ED25519_API      = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET_URL    = (
        'wss://fstream.binance.com/stream?streams='
        f'{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice'
    )
    WS_USER_URL      = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE        = 'https://fapi.binance.com'
    RECV_WINDOW      = 5000
    MIN_NOTIONAL_USD = 20.0

# —— 全局状态 ——
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None
klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
lock = asyncio.Lock()

# —— 轮换标记 ——
last_trend   = None     # 'UP' / 'DOWN'
last_signal  = None     # 最后一次下单方向
macd_cycle   = None     # 'UP'/'DOWN'
rvgi_cycle   = None     # 'UP'/'DOWN'
triple_cycle = None     # 'UP'/'DOWN'

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— 时间 & 模式 ——
async def sync_time():
    global time_offset
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    time_offset = srv - int(time.time() * 1000)
    logging.info("Time offset: %d ms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    j = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    is_hedge_mode = j.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# —— 签名 & 下单 ——
def sign_params(p: dict) -> str:
    qs = urllib.parse.urlencode(sorted(p.items()))
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

def sign_ws(params: dict) -> str:
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = ed_priv.sign(payload.encode())
    return base64.b64encode(sig).decode()

async def order(side, otype, qty=None, price=None, stopPrice=None, reduceOnly=False):
    ts = int(time.time() * 1000 + time_offset)
    p = {'symbol': Config.SYMBOL, 'side': side, 'type': otype,
         'timestamp': ts, 'recvWindow': Config.RECV_WINDOW}
    if is_hedge_mode and otype in ('LIMIT', 'MARKET'):
        p['positionSide'] = 'LONG' if side == 'BUY' else 'SHORT'
    if otype == 'LIMIT':
        p.update({'timeInForce': 'GTC',
                  'quantity': f"{qty:.6f}",
                  'price': f"{price:.2f}"})
        if reduceOnly:
            p['reduceOnly'] = 'true'
    elif otype in ('STOP_MARKET', 'TAKE_PROFIT_MARKET'):
        p.update({'closePosition': 'true',
                  'stopPrice': f"{stopPrice:.2f}"})
    else:
        p['quantity'] = f"{qty:.6f}"
    qs = sign_params(p)
    res = await (await session.post(f"{Config.REST_BASE}/fapi/v1/order?{qs}",
                headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    if res.get('code'):
        logging.error("Order ERR %s %s: %s", otype, side, res)
        return False
    # 校验名义
    if otype == 'LIMIT':
        notional = float(res['origQty']) * float(res['price'])
        if notional < Config.MIN_NOTIONAL_USD:
            logging.warning("Notional %.2f < %.2f, skipped",
                            notional, Config.MIN_NOTIONAL_USD)
            return False
    logging.info("Order OK %s %s qty=%s", otype, side, qty or '')
    return True

# —— 指标更新 ——
def update_indicators():
    for tf, df in klines.items():
        if len(df) < 20:
            continue
        bb = BollingerBands(df['close'], 20, 2)
        df['bb_up'] = bb.bollinger_hband()
        df['bb_dn'] = bb.bollinger_lband()
        df['bb_pct'] = (df['close'] - df['bb_dn']) / (df['bb_up'] - df['bb_dn'])
        if tf == '15m':
            hl2 = (df['high'] + df['low']) / 2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3 * atr
            df['macd'] = MACD(df['close'], 12, 26, 9).macd_diff()
        if tf == '3m':
            num = (df['close'] - df['open']).ewm(span=10).mean()
            den = (df['high'] - df['low']).ewm(span=10).mean()
            df['rvgi'] = num / den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()

# —— SuperTrend ——
def supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    hl2 = (df['high'] + df['low']) / 2
    atr = df['high'].rolling(period).max() - df['low'].rolling(period).min()
    up = hl2 + multiplier * atr
    dn = hl2 - multiplier * atr
    st = pd.Series(index=df.index)
    dirc = pd.Series(True, index=df.index)
    for i in range(len(df)):
        if i == 0:
            st.iat[0] = up.iat[0]
        else:
            prev = st.iat[i-1]; price = df['close'].iat[i]
            if price > prev:
                st.iat[i] = max(dn.iat[i], prev); dirc.iat[i] = True
            else:
                st.iat[i] = min(up.iat[i], prev); dirc.iat[i] = False
    return st, dirc

# —— Market WS ——
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o = json.loads(msg)
                    stream, data = o['stream'], o['data']
                    if stream.endswith('@markPrice'):
                        latest_price = float(data['p'])
                        price_ts = time.time()
                    if 'kline' in stream:
                        tf = stream.split('@')[1].split('_')[1]
                        k = data['k']
                        rec = {'open': float(k['o']),
                               'high': float(k['h']),
                               'low':  float(k['l']),
                               'close':float(k['c'])}
                        async with lock:
                            df = klines[tf]
                            if df.empty or int(k['t']) > df.index[-1]:
                                klines[tf] = pd.concat([df, pd.DataFrame([rec])],
                                                       ignore_index=True)
                            else:
                                df.iloc[-1] = list(rec.values())
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(2)

# —— User WS ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params = {'apiKey': Config.ED25519_API,
                          'timestamp': int(time.time() * 1000)}
                params['signature'] = sign_ws(params)
                await ws.send(json.dumps({
                    'id': str(uuid.uuid4()),
                    'method': 'session.logon',
                    'params': params
                }))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({
                            'id': str(uuid.uuid4()),
                            'method': 'session.status'
                        }))
                asyncio.create_task(hb())
                async for _ in ws:
                    pass
        except Exception as e:
            logging.error("User WS error: %s", e)
            await asyncio.sleep(5)

# —— 趋势监控 & 信号复位 ——
async def trend_watcher():
    global last_trend, last_signal
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            df = klines['15m']
            if 'st' not in df.columns or latest_price is None:
                continue
            stv = df['st'].iat[-1]
            trend = 'UP' if latest_price > stv else 'DOWN'
            if trend != last_trend:
                last_trend, last_signal = trend, None

# —— 主策略（3m BB%B） ——
async def main_strategy():
    global last_signal
    levels = [0.0025, 0.0040, 0.0060, 0.0080, 0.0160]
    tp_offs = [0.0102, 0.0123, 0.0150, 0.0180, 0.0220]
    sl_up, sl_dn = 0.98, 1.02

    while price_ts is None:
        await asyncio.sleep(0.1)
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if any(len(klines[tf]) < 20 for tf in ('3m','15m','1h')):
                continue
            p    = latest_price
            df3m = klines['3m']
            df1h = klines['1h']
            df15 = klines['15m']
            if 'bb_pct' not in df3m.columns or 'bb_pct' not in df1h.columns:
                continue
            bb1  = df1h['bb_pct'].iat[-1]
            bb3  = df3m['bb_pct'].iat[-1]
            stv  = df15['st'].iat[-1]
            trend = 'UP' if p > stv else 'DOWN'
            # 本轮仅首次触发
            if last_signal is None and (bb3 <= 0 or bb3 >= 1):
                strong = (trend=='UP' and bb1<0.2) or (trend=='DOWN' and bb1>0.8)
                qty = 0.12 if strong else 0.03
                if trend == 'DOWN': qty = 0.07 if strong else 0.015
                side, rev = ('BUY','SELL') if trend=='UP' else ('SELL','BUY')
                # 挂单
                for off in levels:
                    price_off = p * (1+off if side=='BUY' else 1-off)
                    await order(side,'LIMIT',qty=qty,price=price_off)
                # 止盈
                for off in tp_offs:
                    price_tp = p * (1+off if rev=='BUY' else 1-off)
                    await order(rev,'LIMIT',qty=qty*0.2,price=price_tp,reduceOnly=True)
                # 初始止损
                slp = p * (sl_up if trend=='UP' else sl_dn)
                tp_type = 'STOP_MARKET' if trend=='UP' else 'TAKE_PROFIT_MARKET'
                await order(rev,tp_type,stopPrice=slp)
                last_signal = trend

# —— MACD 子策略 ——
async def macd_strategy():
    global macd_cycle, last_signal
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if len(df) < 26 or 'macd' not in df.columns:
                continue
            prev, cur = float(df['macd'].iat[-2]), float(df['macd'].iat[-1])
            osc = abs(cur)
            if prev > 0 and cur < prev and osc >= 11 and macd_cycle != 'DOWN':
                if last_signal != 'DOWN':
                    await order('SELL','MARKET',qty=0.017)
                    last_signal = 'DOWN'
                macd_cycle = 'DOWN'
            if prev < 0 and cur > prev and osc >= 11 and macd_cycle != 'UP':
                if last_signal != 'UP':
                    await order('BUY','MARKET',qty=0.017)
                    last_signal = 'UP'
                macd_cycle = 'UP'

# —— RVGI 子策略 ——
async def rvgi_strategy():
    global rvgi_cycle, last_signal
    while True:
        await asyncio.sleep(10)
        async with lock:
            df = klines['15m']
            if len(df) < 10 or 'rvgi' not in df.columns or 'rvsig' not in df.columns:
                continue
            rv, sg = float(df['rvgi'].iat[-1]), float(df['rvsig'].iat[-1])
            if rv > sg and rvgi_cycle != 'UP':
                if last_signal != 'UP':
                    await order('BUY','MARKET',qty=0.016)
                    last_signal = 'UP'
                await order('SELL','LIMIT',qty=0.016,price=latest_price*1.06,reduceOnly=True)
                await order('SELL','STOP_MARKET',stopPrice=latest_price*0.98)
                rvgi_cycle = 'UP'
            if rv < sg and rvgi_cycle != 'DOWN':
                if last_signal != 'DOWN':
                    await order('SELL','MARKET',qty=0.016)
                    last_signal = 'DOWN'
                await order('BUY','LIMIT',qty=0.016,price=latest_price*0.94,reduceOnly=True)
                await order('BUY','TAKE_PROFIT_MARKET',stopPrice=latest_price*1.02)
                rvgi_cycle = 'DOWN'

# —— 三重 SuperTrend 子策略 ——
async def triple_st_strategy():
    global triple_cycle, last_signal
    while True:
        await asyncio.sleep(30)
        async with lock:
            df = klines['15m']
            if len(df) < 12:
                continue
            s1,d1 = supertrend(df,10,1)
            s2,d2 = supertrend(df,11,2)
            s3,d3 = supertrend(df,12,3)
            up = d1.iat[-1] and d2.iat[-1] and d3.iat[-1]
            dn = not (d1.iat[-1] or d2.iat[-1] or d3.iat[-1])
            if up and triple_cycle != 'UP':
                if last_signal != 'UP':
                    await order('BUY','MARKET',qty=0.015)
                    last_signal = 'UP'
                triple_cycle = 'UP'
            if dn and triple_cycle != 'DOWN':
                if last_signal != 'DOWN':
                    await order('SELL','MARKET',qty=0.015)
                    last_signal = 'DOWN'
                triple_cycle = 'DOWN'
            # 止盈，不改变 last_signal
            if triple_cycle == 'UP' and not up:
                await order('SELL','MARKET',qty=0.015)
            if triple_cycle == 'DOWN' and not dn:
                await order('BUY','MARKET',qty=0.015)

# —— 启动 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    await detect_mode()
    try:
        await asyncio.gather(
            market_ws(), user_ws(), trend_watcher(),
            main_strategy(), macd_strategy(),
            rvgi_strategy(), triple_st_strategy()
        )
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())