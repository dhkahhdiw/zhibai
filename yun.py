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
import uuid
import base64

import uvloop
import aiohttp
import pandas as pd
import websockets

from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# ———— uvloop for high-performance asyncio ————
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# ———— Load env ————
load_dotenv('/root/zhibai/.env')

# ———— Config ————
class Config:
    API_KEY         = os.getenv('BINANCE_API_KEY')
    SECRET_KEY      = os.getenv('BINANCE_SECRET_KEY').encode('ascii')
    ED25519_API_KEY = os.getenv('ED25519_API_KEY')
    ED25519_KEY_PATH= os.getenv('ED25519_KEY_PATH')
    SYMBOL          = 'ETHUSDC'
    PAIR_LOWER      = SYMBOL.lower()
    WS_MARKET_URL   = (
        'wss://fstream.binance.com/stream?streams='
        f'{PAIR_LOWER}@kline_3m/'
        f'{PAIR_LOWER}@kline_15m/'
        f'{PAIR_LOWER}@kline_1h/'
        f'{PAIR_LOWER}@markPrice'
    )
    WS_USER_URL     = 'wss://ws-fapi.binance.com/ws-fapi/v1'
    REST_BASE       = 'https://fapi.binance.com'
    RECV_WINDOW     = 5000
    MAX_POS         = 0.49  # ETH
    TIME_SYNC_INTERVAL = 3600

# ———— Globals ————
klines       = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
latest_price = None
price_ts     = None
lock         = asyncio.Lock()
last_side    = None
session      = None
TIME_OFFSET  = 0
position     = {'long':0.0, 'short':0.0}

# ———— Load Ed25519 key ————
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# ———— Logging ————
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)

# ———— Time sync ————
async def sync_time():
    global TIME_OFFSET
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    local = int(time.time()*1000)
    TIME_OFFSET = srv - local
    logging.info("Time offset: %d ms", TIME_OFFSET)

# ———— Build & sign REST query ————
def build_signed_query(params: dict) -> str:
    qs = urllib.parse.urlencode(sorted(params.items()), safe='')
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig

# ———— REST order ————
async def rest_order(side, otype, qty, price=None, stopPrice=None):
    global position
    ts = int(time.time()*1000 + TIME_OFFSET)
    params = {
        'symbol':Config.SYMBOL, 'side':side, 'type':otype,
        'timestamp':ts, 'recvWindow':Config.RECV_WINDOW
    }
    if otype=='LIMIT':
        params.update({'timeInForce':'GTC',
                       'quantity':f"{qty:.6f}",
                       'price':f"{price:.2f}"})
    else:
        params.update({'closePosition':'true',
                       'stopPrice':f"{stopPrice:.2f}"})
    query = build_signed_query(params)
    url = f"{Config.REST_BASE}/fapi/v1/order?{query}"
    async with session.post(url, headers={'X-MBX-APIKEY':Config.API_KEY}) as r:
        res = await r.json()
    if 'code' in res:
        logging.error("REST %s %s ERROR %s", otype, side, res)
        return False
    logging.info("REST %s %s OK %s", otype, side, res.get('orderId'))
    # update pos
    if side=='BUY': position['long'] += qty
    else:           position['short'] += qty
    return True

# ———— Bracket ————
async def bracket(qty, entry, side):
    # check max pos
    tot = position['long'] + position['short'] + qty
    if tot > Config.MAX_POS:
        logging.warning("exceed max pos %.2f", tot)
        return
    if not await rest_order(side, 'LIMIT', qty, price=entry): return
    tp = entry * (1.02 if side=='BUY' else 0.98)
    sl = entry * (0.98 if side=='BUY' else 1.02)
    tp_side = 'SELL' if side=='BUY' else 'BUY'
    await asyncio.gather(
        rest_order(tp_side, 'TAKE_PROFIT_MARKET', qty, stopPrice=tp),
        rest_order(tp_side, 'STOP_MARKET', qty, stopPrice=sl)
    )

# ———— Update indicators ————
def update_indicators():
    for tf, df in klines.items():
        if df.empty: continue
        bb = BollingerBands(df['close'],20,2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct'] = (df['close']-df['bb_dn'])/(df['bb_up']-df['bb_dn'])
        if tf=='15m':
            hl2 = (df['high']+df['low'])/2
            atr= df['high'].rolling(10).max()-df['low'].rolling(10).min()
            df['st']= hl2-3*atr
            df['macd']= MACD(df['close'],12,26,9).macd_diff()
        if tf=='3m':
            num= (df['close']-df['open']).ewm(span=10).mean()
            den= (df['high']-df['low']).ewm(span=10).mean()
            df['rvgi']= num/den
            df['rvsig']= df['rvgi'].ewm(span=4).mean()
        klines[tf]=df

# ———— Market WS ————
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o=json.loads(msg); s=o['stream']; d=o['data']
                    if s.endswith('@markPrice'):
                        latest_price=float(d['p'])
                        price_ts=int(time.time()*1000)
                    if 'kline' in s:
                        tf=s.split('@')[1].split('_')[1]; k=d['k']
                        rec={ 'open':float(k['o']),'high':float(k['h']),
                              'low':float(k['l']),'close':float(k['c']) }
                        async with lock:
                            df=klines[tf]
                            if df.empty or int(k['t'])>df.index[-1]:
                                df=pd.concat([df,pd.DataFrame([rec])])
                            else:
                                df.iloc[-1]=list(rec.values())
                            klines[tf]=df
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error %s; reconnect in 5s", e)
            await asyncio.sleep(5)

# ———— User WS (Ed25519 JSON-RPC) ————
def sign_ws_ed25519(params:dict)->str:
    payload='&'.join(f"{k}={v}" for k,v in sorted(params.items()))
    sig=ed_priv.sign(payload.encode('ascii'))
    return base64.b64encode(sig).decode()

async def user_ws():
    async with websockets.connect(Config.WS_USER_URL) as ws:
        logging.info("User WS connected")
        while True:
            # 发送心跳或下单RPC示例
            req_id=str(uuid.uuid4())
            params={
                "apiKey": Config.ED25519_API_KEY,
                "timestamp": int(time.time()*1000),
                "recvWindow": Config.RECV_WINDOW
            }
            sig=sign_ws_ed25519(params)
            rpc={"id":req_id,"method":"order.place","params":{**params,"signature":sig}}
            await ws.send(json.dumps(rpc))
            await asyncio.sleep(30)

# ———— Main strategy ————
async def main_strategy():
    global last_side
    while price_ts is None:
        await asyncio.sleep(0.5)
    while True:
        async with lock:
            if any(klines[tf].empty for tf in klines):
                await asyncio.sleep(1); continue
            p=latest_price
            st=klines['15m']['st'].iloc[-1]
            bb1h=klines['1h']['bb_pct'].iloc[-1]
            bb3m=klines['3m']['bb_pct'].iloc[-1]
            up,down=p>st,p<st
            sl= up and bb1h<0.2
            ss= down and bb1h>0.8
            if sl and bb3m<=0 and last_side!='LONG':
                await bracket(0.12,p,'BUY'); last_side='LONG'
            elif ss and bb3m>=1 and last_side!='SHORT':
                await bracket(0.12,p,'SELL');last_side='SHORT'
            elif up and not sl and 0<bb3m<=0.5 and last_side!='LONG':
                await bracket(0.03,p,'BUY'); last_side='LONG'
            elif down and not ss and 0.5<=bb3m<1 and last_side!='SHORT':
                await bracket(0.03,p,'SELL');last_side='SHORT'
            elif sl and bb3m>=1 and last_side!='SHORT':
                await bracket(0.07,p,'SELL');last_side='SHORT'
            elif ss and bb3m<=0 and last_side!='LONG':
                await bracket(0.07,p,'BUY'); last_side='LONG'
        await asyncio.sleep(0.5)

# ———— 15m MACD 子策略 ————
async def macd_strategy():
    fired=False
    while True:
        async with lock:
            df=klines['15m']
            if len(df)<27: continue
            cur,pre=df['macd'].iloc[-1],df['macd'].iloc[-2]
            if not fired and pre>0 and cur<pre:
                await bracket(0.15,latest_price,'SELL');fired=True
            if fired and pre<0 and cur>pre:
                await bracket(0.15,latest_price,'BUY'); fired=False
        await asyncio.sleep(15)

# ———— 15m RVGI 子策略 ————
async def rvgi_strategy():
    cnt_long=cnt_short=0
    while True:
        async with lock:
            df=klines['15m']
            if 'rvgi' not in df: continue
            rv,sg=df['rvgi'].iat[-1],df['rvsig'].iat[-1]
            if rv>sg and cnt_long*0.05<0.2:
                await bracket(0.05,latest_price,'BUY');cnt_long+=1
            if rv<sg and cnt_short*0.05<0.2:
                await bracket(0.05,latest_price,'SELL');cnt_short+=1
        await asyncio.sleep(20)

# ———— Triple ST 子策略 ————
async def triple_st_strategy():
    active=False
    while True:
        async with lock:
            df=klines['15m']
            if 'st' not in df or len(df['st'])<3: continue
            stv=df['st']; p=latest_price
            rise=stv.iat[-3]<stv.iat[-2]<stv.iat[-1]<p
            fall=stv.iat[-3]>stv.iat[-2]>stv.iat[-1]>p
            if rise and not active:
                await bracket(0.15,p,'BUY');active=True
            if fall and not active:
                await bracket(0.15,p,'SELL');active=True
            prev=stv.iat[-2]
            if active and ((rise and p<prev) or (fall and p>prev)):
                side='SELL' if rise else 'BUY'
                await bracket(0.15,p,side);active=False
        await asyncio.sleep(30)

# ———— Time sync task ————
async def time_sync_task():
    while True:
        await sync_time()
        await asyncio.sleep(Config.TIME_SYNC_INTERVAL)

# ———— Entry point ————
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy(),
            time_sync_task()
        )
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())