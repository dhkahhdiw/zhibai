#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import hmac
import hashlib
import asyncio
import logging
import urllib.parse
import base64
import uuid

import uvloop
import aiohttp
import pandas as pd
import websockets

from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# ———— 高性能事件循环 ————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ———— 加载环境变量 ————
load_dotenv('/root/zhibai/.env')

# ———— 全局配置 ————
class Config:
    SYMBOL           = 'ETHUSDC'
    PAIR             = SYMBOL.lower()
    API_KEY          = os.getenv('BINANCE_API_KEY')
    SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode('ascii')
    ED25519_API      = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH = os.getenv('ED25519_KEY_PATH')
    WS_MARKET_URL    = (
        'wss://fstream.binance.com/stream?streams='
        f'{PAIR}@kline_3m/'
        f'{PAIR}@kline_15m/'
        f'{PAIR}@kline_1h/'
        f'{PAIR}@markPrice'
    )
    WS_USER_URL      = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE        = 'https://fapi.binance.com'
    RECV_WINDOW      = 5000

# ———— 日志配置 ————
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ———— 全局状态 ————
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None
klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
lock = asyncio.Lock()
last_side = None        # 上一次下单方向：'BUY' or 'SELL'
last_trend = None       # 上一次趋势：'UP' or 'DOWN'

# ———— 加载 Ed25519 私钥 ————
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# ———— 时间同步 ————
async def sync_time():
    global time_offset
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    loc = int(time.time() * 1000)
    time_offset = srv - loc
    logging.info("Time offset: %d ms", time_offset)

# ———— 检测持仓模式 ————
async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000 + time_offset)
    params = {'timestamp': ts, 'recvWindow': Config.RECV_WINDOW}
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    async with session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY}) as r:
        data = await r.json()
    is_hedge_mode = data.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# ———— HMAC 签名 ————
def build_signed_query(params: dict) -> str:
    qs = urllib.parse.urlencode(sorted(params.items()), safe='')
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

# ———— Ed25519 WebSocket 签名 ————
def sign_ws(params: dict) -> str:
    payload = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = ed_priv.sign(payload.encode('ascii'))
    return base64.b64encode(sig).decode('ascii')

# ———— 下单函数 ————
async def rest_order(side: str, otype: str, qty: float=None, price: float=None, stopPrice: float=None):
    ts = int(time.time() * 1000 + time_offset)
    params = {
        'symbol':     Config.SYMBOL,
        'side':       side,
        'type':       otype,
        'timestamp':  ts,
        'recvWindow': Config.RECV_WINDOW
    }
    if is_hedge_mode and otype in ('LIMIT','MARKET'):
        params['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'
    if otype == 'LIMIT':
        params.update({
            'timeInForce':'GTC',
            'quantity':   f"{qty:.6f}",
            'price':      f"{price:.2f}"
        })
    elif otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
        params.update({
            'closePosition':'true',
            'stopPrice':    f"{stopPrice:.2f}"
        })
    else:  # MARKET
        params.update({'quantity': f"{qty:.6f}"})
    query = build_signed_query(params)
    url   = f"{Config.REST_BASE}/fapi/v1/order?{query}"
    async with session.post(url, headers={'X-MBX-APIKEY': Config.API_KEY}) as r:
        res = await r.json()
    if res.get('code'):
        logging.error("Order ERR %s %s: %s", otype, side, res)
        return False
    logging.info("Order OK %s %s qty=%.4f", otype, side, qty or 0)
    return True

# ———— 分批挂单＋止盈/止损组合 ————
async def place_scaled_orders(side: str, base_qty: float, entry: float, bb3: float, trend: str, strength: str):
    """
    1. 按强信号/弱信号、顺势/逆势决定 qty；
    2. 在 3m BB%<=0 或 >=1 的五个价位(0.25%,0.4%,0.6%,0.8%,1.6%)挂单，每档占 base_qty*20%；
    3. 挂单时同时在买入价基础上分批挂 5 档止盈(1.02%,1.23%,1.5%,1.8%,2.2%)，每档占20%;
       对弱信号止盈只挂两档(1.23%,1.8%)，各 50%。
    4. 初始止损价 entry*0.98(多) / entry*1.02(空)。
    """
    # 1. 计算单笔基础仓位
    if strength=='strong':
        qty = 0.12 if ((side=='BUY' and trend=='UP') or (side=='SELL' and trend=='DOWN')) else 0.07
    else:
        qty = 0.03 if ((side=='BUY' and trend=='UP') or (side=='SELL' and trend=='DOWN')) else 0.015

    # 2. 判断 bb3 是否触发主价位
    if not ((side=='BUY' and bb3<=0) or (side=='SELL' and bb3>=1)):
        return

    # 3. 五档挂单
    offsets = [0.0025,0.004,0.006,0.008,0.016]
    take_offsets = [0.0102,0.0123,0.015,0.018,0.022]
    if strength=='weak':
        take_offsets = [0.0123,0.018]

    for i,off in enumerate(offsets):
        px = entry*(1-off if side=='SELL' else 1+off)
        part_qty = qty*0.2
        await rest_order(side, 'LIMIT', qty=part_qty, price=px)
        # 同时挂止盈
        take_off = take_offsets[i if i<len(take_offsets) else -1]
        tp_px = entry*(1+take_off if side=='BUY' else 1-take_off)
        sl_px = entry*(0.98 if side=='BUY' else 1.02)
        opp = 'SELL' if side=='BUY' else 'BUY'
        # 止盈限价单
        await rest_order(opp, 'LIMIT', qty=part_qty, price=tp_px)
        # 止损市价单
        await rest_order(opp, 'STOP_MARKET', stopPrice=sl_px)

# ———— 指标计算 ————
def update_indicators():
    for tf, df in klines.items():
        if df.shape[0]<20:
            continue
        bb = BollingerBands(df['close'],20,2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct'] = (df['close']-df['bb_dn'])/(df['bb_up']-df['bb_dn'])
        if tf=='15m':
            hl2 = (df['high']+df['low'])/2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3*atr
            df['macd'] = MACD(df['close'],12,26,9).macd_diff()
        if tf=='3m':
            num = (df['close']-df['open']).ewm(span=10).mean()
            den = (df['high']-df['low']).ewm(span=10).mean()
            df['rvgi']  = num/den
            df['rvsig'] = df['rvgi'].ewm(span=4).mean()
        klines[tf] = df

# ———— 市场数据 WS ————
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                async for msg in ws:
                    o = json.loads(msg); s=o['stream']; d=o['data']
                    if s.endswith('@markPrice'):
                        latest_price = float(d['p']); price_ts = time.time()
                    if 'kline' in s:
                        tf = s.split('@')[1].split('_')[1]; k=d['k']
                        rec = dict(open=float(k['o']), high=float(k['h']),
                                   low=float(k['l']), close=float(k['c']))
                        async with lock:
                            df = klines[tf]
                            if df.empty or int(k['t'])>df.index[-1]:
                                klines[tf] = pd.concat([df,pd.DataFrame([rec])],ignore_index=True)
                            else:
                                df.iloc[-1] = list(rec.values()); klines[tf]=df
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(5)

# ———— 用户流 WS ————
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                params = {'apiKey':Config.ED25519_API,'timestamp':int(time.time()*1000)}
                params['signature'] = sign_ws(params)
                await ws.send(json.dumps({'id':str(uuid.uuid4()),
                                          'method':'session.logon','params':params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({'id':str(uuid.uuid4()),
                                                  'method':'session.status'}))
                asyncio.create_task(hb())
                async for msg in ws:
                    logging.debug("User WS <- %s", msg)
        except Exception as e:
            logging.error("User WS error: %s", e)
            await asyncio.sleep(5)

# ———— 趋势监控 ————
async def trend_watcher():
    global last_trend, last_side
    while True:
        await asyncio.sleep(0.1)
        async with lock:
            if 'st' not in klines['15m'] or latest_price is None:
                continue
            trend = 'UP' if latest_price>klines['15m']['st'].iloc[-1] else 'DOWN'
            if trend!=last_trend:
                last_trend, last_side = trend, None

# ———— 主策略 ————
async def main_strategy():
    while price_ts is None:
        await asyncio.sleep(0.2)
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if any(klines[tf].shape[0]<20 for tf in ('3m','15m','1h')):
                continue
            p   = latest_price
            bb1 = klines['1h']['bb_pct'].iloc[-1]
            bb3 = klines['3m']['bb_pct'].iloc[-1]
            trend = 'UP' if p>klines['15m']['st'].iloc[-1] else 'DOWN'
            # 强弱信号判定
            strength = 'strong' if ((trend=='UP' and bb1<0.2) or (trend=='DOWN' and bb1>0.8)) else 'weak'
            side = 'BUY' if trend=='UP' else 'SELL'
            await place_scaled_orders(side, 1.0, p, bb3, trend, strength)

# ———— 子策略们 ————
async def macd_strategy():
    fired=False
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if df.shape[0]<26: continue
            prev, cur = df['macd'].iloc[-2], df['macd'].iloc[-1]
            if not fired and prev>0 and cur<prev:
                await rest_order('SELL','MARKET',qty=0.15); fired=True
            elif fired and prev<0 and cur>prev:
                await rest_order('BUY','MARKET',qty=0.15);  fired=False

async def rvgi_strategy():
    long_cnt=short_cnt=0
    while True:
        await asyncio.sleep(10)
        async with lock:
            df = klines['15m']  # 用15m作为RVGI-10
            if df.shape[0]<10: continue
            rv = df['macd'].ewm(span=10).mean().iloc[-1]
            sig= df['macd'].ewm(span=9).mean().iloc[-1]
            # 省略过滤条件
            if rv>sig and long_cnt<4:
                await rest_order('BUY','MARKET',qty=0.05); long_cnt+=1
            if rv<sig and short_cnt<4:
                await rest_order('SELL','MARKET',qty=0.05); short_cnt+=1

async def triple_st_strategy():
    active=False
    while True:
        await asyncio.sleep(30)
        async with lock:
            df=klines['15m']
            if df.shape[0]<3: continue
            stv=df['st']; p=latest_price
            rise=stv.iloc[-3]<stv.iloc[-2]<stv.iloc[-1]<p
            fall=stv.iloc[-3]>stv.iloc[-2]>stv.iloc[-1]>p
            if rise and not active:
                await rest_order('BUY','MARKET',qty=0.15); active=True
            elif fall and not active:
                await rest_order('SELL','MARKET',qty=0.15); active=True
            prev=stv.iloc[-2]
            if active and ((rise and p<prev) or (fall and p>prev)):
                side='SELL' if rise else 'BUY'
                await rest_order(side,'MARKET',qty=0.15); active=False

# ———— 启动 ————
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    await detect_mode()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            trend_watcher(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy(),
        )
    finally:
        await session.close()

if __name__=='__main__':
    asyncio.run(main())