#!/usr/bin/env python3
# coding: utf-8

import os, time, json, hmac, hashlib, asyncio, logging, urllib.parse, base64, uuid
import uvloop, aiohttp, pandas as pd, websockets
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD

# —— uvloop for high performance ——
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# —— load env ——
load_dotenv('/root/zhibai/.env')
API_KEY         = os.getenv('BINANCE_API_KEY')
SECRET_KEY      = os.getenv('BINANCE_SECRET_KEY').encode()
ED25519_API_KEY = os.getenv('ED25519_API_KEY')
ED25519_KEY_PATH= os.getenv('ED25519_KEY_PATH')

SYMBOL      = 'ETHUSDC'
PAIR        = SYMBOL.lower()
WS_MARKETS  = (
    f"wss://fstream.binance.com/stream?streams="
    f"{PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
)
WS_USER     = 'wss://ws-fapi.binance.com/ws-fapi/v1'
REST_BASE   = 'https://fapi.binance.com'
RECV_WINDOW = 5000
MAX_POS     = 0.49  # ETH

# —— logging ——
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('trading_bot.log')]
)

# —— globals ——
session: aiohttp.ClientSession
TIME_OFFSET  = 0
latest_price = None
price_ts     = None
lock         = asyncio.Lock()
last_side    = None

# track simple positions
position = {'LONG':0.0, 'SHORT':0.0}

# kline dataframes
klines = {'3m':pd.DataFrame(), '15m':pd.DataFrame(), '1h':pd.DataFrame()}

# —— load Ed25519 key ——
with open(ED25519_KEY_PATH,'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— sign HMAC ——
def sign_hmac(params):
    qs = urllib.parse.urlencode(sorted(params.items()), safe='')
    return hmac.new(SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

# —— update indicators ——
def update_indicators():
    for tf, df in klines.items():
        if df.shape[0] < 20:
            continue
        bb = BollingerBands(df['close'],20,2)
        df['bb_up']  = bb.bollinger_hband()
        df['bb_dn']  = bb.bollinger_lband()
        df['bb_pct']= (df['close']-df['bb_dn'])/(df['bb_up']-df['bb_dn'])
        if tf=='15m':
            hl2=(df['high']+df['low'])/2
            atr=df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st']   = hl2 - 3*atr
            df['macd']= MACD(df['close'],12,26,9).macd_diff()
        if tf=='3m':
            num=(df['close']-df['open']).ewm(span=10).mean()
            den=(df['high']-df['low']).ewm(span=10).mean()
            df['rvgi'] = num/den
            df['rvsig']= df['rvgi'].ewm(span=4).mean()
        klines[tf]=df

# —— market data WS (no client ping) ——
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(
                WS_MARKETS,
                ping_interval=None,
                ping_timeout=None
            ) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o=json.loads(msg); s=o['stream']; d=o['data']
                    if s.endswith('@markPrice'):
                        latest_price=float(d['p'])
                        price_ts=int(time.time()*1000)
                    if 'kline' in s:
                        tf=s.split('@')[1].split('_')[1]
                        k=d['k']
                        rec={'open':float(k['o']),'high':float(k['h']),
                             'low':float(k['l']),'close':float(k['c'])}
                        async with lock:
                            df=klines[tf]
                            if df.empty or int(k['t'])>df.index[-1]:
                                df=pd.concat([df,pd.DataFrame([rec])],ignore_index=True)
                            else:
                                df.iloc[-1]=list(rec.values())
                            klines[tf]=df
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error, reconnect in 5s: %s", e)
            await asyncio.sleep(5)

# —— REST order ——
async def rest_order(side, otype, qty=None, price=None, stopPrice=None):
    ts=int(time.time()*1000+TIME_OFFSET)
    params={'symbol':SYMBOL,'side':side,'type':otype,
            'timestamp':ts,'recvWindow':RECV_WINDOW,
            'positionSide':('LONG' if side=='BUY' else 'SHORT')}
    if otype=='LIMIT':
        params.update({'timeInForce':'GTC',
                       'quantity':f"{qty:.6f}",'price':f"{price:.2f}"})
    else:
        params.update({'closePosition':'true','stopPrice':f"{stopPrice:.2f}"})
    params['signature']=sign_hmac(params)
    url=f"{REST_BASE}/fapi/v1/order?"+urllib.parse.urlencode(params)
    async with session.post(url,headers={'X-MBX-APIKEY':API_KEY}) as r:
        ret=await r.json()
    if 'code' in ret:
        logging.error("Order ERR %s %s: %s", otype, side, ret)
        return False
    logging.info("Order OK %s %s qty=%.4f", otype, side, qty or 0)
    position['LONG' if side=='BUY' else 'SHORT']+=qty or 0
    return True

# —— bracket orders ——
async def bracket(qty, entry, side):
    total = position['LONG']+position['SHORT']+qty
    if total>MAX_POS:
        logging.warning("Exceed max pos:%.4f", total)
        return
    if not await rest_order(side,'LIMIT',qty,price=entry): return
    tp=entry*(1.02 if side=='BUY' else 0.98)
    sl=entry*(0.98 if side=='BUY' else 1.02)
    ps='SELL' if side=='BUY' else 'BUY'
    await asyncio.gather(
        rest_order(ps,'TAKE_PROFIT_MARKET',stopPrice=tp,side=ps),
        rest_order(ps,'STOP_MARKET',stopPrice=sl,side=ps)
    )

# —— main strategy ——
async def main_strategy():
    global last_side
    while price_ts is None:
        await asyncio.sleep(0.2)
    while True:
        async with lock:
            if any(len(klines[tf])<20 for tf in ('3m','15m','1h')):
                continue
            p=latest_price
            st=klines['15m']['st'].iloc[-1]
            b1=klines['1h']['bb_pct'].iloc[-1]
            b3=klines['3m']['bb_pct'].iloc[-1]
            up,down=(p>st),(p<st)
            sl=up and b1<0.2; ss=down and b1>0.8
            if sl and b3<=0 and last_side!='LONG':
                await bracket(0.12,p,'BUY'); last_side='LONG'
            elif ss and b3>=1 and last_side!='SHORT':
                await bracket(0.12,p,'SELL'); last_side='SHORT'
            elif up and not sl and 0< b3<=0.5 and last_side!='LONG':
                await bracket(0.03,p,'BUY'); last_side='LONG'
            elif down and not ss and 0.5<=b3<1 and last_side!='SHORT':
                await bracket(0.03,p,'SELL'); last_side='SHORT'
        await asyncio.sleep(0.5)

# —— 15m MACD sub-strategy ——
async def macd_strategy():
    fired=False
    while True:
        async with lock:
            df=klines['15m']
            if len(df)<26:
                pass
            else:
                cur,pre=df['macd'].iloc[-1],df['macd'].iloc[-2]
                std=df['macd'].std()
                if not fired and pre>0 and cur<pre and abs(cur)>1.5*std:
                    await bracket(0.15,latest_price,'SELL'); fired=True
                elif fired and pre<0 and cur>pre and abs(cur)>1.5*std:
                    await bracket(0.15,latest_price,'BUY'); fired=False
        await asyncio.sleep(15)

# —— 3m RVGI sub-strategy ——
async def rvgi_strategy():
    buy_fired=sell_fired=False
    while True:
        async with lock:
            df=klines['3m']
            if 'rvgi' in df:
                rv,sg=df['rvgi'].iloc[-1],df['rvsig'].iloc[-1]
                if not buy_fired and rv>sg:
                    await bracket(0.05,latest_price,'BUY'); buy_fired=True; sell_fired=False
                if not sell_fired and rv<sg:
                    await bracket(0.05,latest_price,'SELL'); sell_fired=True; buy_fired=False
        await asyncio.sleep(5)

# —— SuperTrend sub-strategy ——
async def st_strategy():
    last=None
    while True:
        async with lock:
            df=klines['15m']
            if 'st' in df:
                cur,pre=df['st'].iloc[-1],df['st'].iloc[-2]
                price=latest_price
                if pre>price and cur<price and last!='LONG':
                    await bracket(0.15,price,'BUY'); last='LONG'
                if pre<price and cur>price and last!='SHORT':
                    await bracket(0.15,price,'SELL'); last='SHORT'
        await asyncio.sleep(10)

# —— time sync ——
async def sync_time():
    global TIME_OFFSET
    r=await session.get(f"{REST_BASE}/fapi/v1/time")
    srv=(await r.json())['serverTime']
    TIME_OFFSET=srv-int(time.time()*1000)
    logging.info("Time offset:%dms", TIME_OFFSET)

# —— main entry ——
async def main():
    global session
    session=aiohttp.ClientSession()
    await sync_time()
    try:
        await asyncio.gather(
            market_ws(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            st_strategy(),
        )
    finally:
        await session.close()

if __name__=='__main__':
    asyncio.run(main())