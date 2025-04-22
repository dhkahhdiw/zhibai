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
WS_MARKETS  = f"wss://fstream.binance.com/stream?streams={PAIR}@kline_3m/{PAIR}@kline_15m/{PAIR}@kline_1h/{PAIR}@markPrice"
WS_USER     = 'wss://ws-fapi.binance.com/ws-fapi/v1'
REST_BASE   = 'https://fapi.binance.com'
RECV_WINDOW = 5000

# —— global state ——
klines       = {'3m':pd.DataFrame(), '15m':pd.DataFrame(), '1h':pd.DataFrame()}
latest_price = None
price_ts     = None
lock         = asyncio.Lock()
last_side    = None
session      = None
TIME_OFFSET  = 0

# —— load Ed25519 private key ——
with open(ED25519_KEY_PATH,'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— sign REST HMAC SHA256 ——
def sign_hmac(params:dict)->str:
    qs = urllib.parse.urlencode(sorted(params.items()), safe='')
    return hmac.new(SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()

# —— update indicators ——
def update_indicators():
    for tf, df in klines.items():
        if len(df)<20: continue
        bb = BollingerBands(df['close'],20,2)
        df['bb_up']=bb.bollinger_hband(); df['bb_dn']=bb.bollinger_lband()
        df['bb_pct']=(df['close']-df['bb_dn'])/(df['bb_up']-df['bb_dn'])
        if tf=='15m':
            hl2=(df['high']+df['low'])/2
            atr=df['high'].rolling(10).max()-df['low'].rolling(10).min()
            df['st']=hl2-3*atr
            df['macd']=MACD(df['close'],12,26,9).macd_diff()
        if tf=='3m':
            num=(df['close']-df['open']).ewm(span=10).mean()
            den=(df['high']-df['low']).ewm(span=10).mean()
            df['rvgi']=num/den; df['rvsig']=df['rvgi'].ewm(span=4).mean()
        klines[tf]=df

# —— market data WS ——
async def market_ws():
    global latest_price, price_ts
    async with websockets.connect(WS_MARKETS) as ws:
        logging.info("Market WS connected")
        async for msg in ws:
            o=json.loads(msg); s=o['stream']; d=o['data']
            if s.endswith('@markPrice'):
                latest_price=float(d['p']); price_ts=int(time.time()*1000)
            if 'kline' in s:
                tf=s.split('@')[1].split('_')[1]; k=d['k']
                rec={'open':float(k['o']),'high':float(k['h']),
                     'low':float(k['l']),'close':float(k['c'])}
                async with lock:
                    df=klines[tf]
                    if df.empty or int(k['t'])>df.index[-1]:
                        df=pd.concat([df,pd.DataFrame([rec])], ignore_index=True)
                    else:
                        df.iloc[-1]=list(rec.values())
                    klines[tf]=df
                    update_indicators()

# —— REST order ——
async def rest_order(side, otype, qty=None, price=None, stopPrice=None, positionSide=None):
    """Place HMAC-signed futures orders, with correct params per type."""
    ts=int(time.time()*1000+TIME_OFFSET)
    params={'symbol':SYMBOL,'side':side,'type':otype,
            'timestamp':ts,'recvWindow':RECV_WINDOW}
    # include positionSide in every order (dual-mode required) binanec合约.txt](file-service://file-LMnfGcqyxu7gkTs7aSjyZ3)
    params['positionSide']=positionSide or ('LONG' if side=='BUY' else 'SHORT')
    if otype=='LIMIT':
        params.update({'timeInForce':'GTC','quantity':f"{qty:.6f}",'price':f"{price:.2f}"})
    else:  # STOP_MARKET or TAKE_PROFIT_MARKET
        # use closePosition=true to close full pos; do NOT send quantity binanec合约.txt](file-service://file-LMnfGcqyxu7gkTs7aSjyZ3)
        params.update({'closePosition':'true','stopPrice':f"{stopPrice:.2f}"})
    # sign & send as query string
    params['signature']=sign_hmac(params)
    url=f"{REST_BASE}/fapi/v1/order?"+urllib.parse.urlencode(params)
    async with session.post(url, headers={'X-MBX-APIKEY':API_KEY}) as r:
        res=await r.json()
        prefix=f"{otype} {side}"
        if 'code' in res:
            logging.error("Order ERR %s: %s", prefix, res)
            return False
        logging.info("Order OK %s qty=%s", prefix, qty or '',)
        return True

# —— bracket orders (entry+TP+SL) ——
async def bracket(qty, entry, side):
    ps=('LONG' if side=='BUY' else 'SHORT')
    # entry
    if not await rest_order(side,'LIMIT',qty,price=entry,positionSide=ps): return
    # take profit
    tp=entry*(1.02 if side=='BUY' else 0.98)
    await rest_order('SELL' if side=='BUY' else 'BUY','TAKE_PROFIT_MARKET',
                     stopPrice=tp, positionSide=ps)
    # stop loss
    sl=entry*(0.98 if side=='BUY' else 1.02)
    await rest_order('SELL' if side=='BUY' else 'BUY','STOP_MARKET',
                     stopPrice=sl, positionSide=ps)

# —— main strategy ——
async def main_strategy():
    global last_side
    while price_ts is None: await asyncio.sleep(0.2)
    while True:
        async with lock:
            if any(len(klines[tf])<20 for tf in ('3m','15m','1h')):
                continue
            p=latest_price
            st=klines['15m']['st'].iloc[-1]
            bb1=klines['1h']['bb_pct'].iloc[-1]
            bb3=klines['3m']['bb_pct'].iloc[-1]
            up,down=(p>st),(p<st)
            strong_long=up and bb1<0.2
            strong_short=down and bb1>0.8
            # 顺势强、弱、逆势示例
            if strong_long and bb3<=0 and last_side!='LONG':
                await bracket(0.12,p,'BUY'); last_side='LONG'
            elif strong_short and bb3>=1 and last_side!='SHORT':
                await bracket(0.12,p,'SELL'); last_side='SHORT'
            elif up and not strong_long and 0<bb3<=0.5 and last_side!='LONG':
                await bracket(0.03,p,'BUY'); last_side='LONG'
            elif down and not strong_short and 0.5<=bb3<1 and last_side!='SHORT':
                await bracket(0.03,p,'SELL'); last_side='SHORT'
        await asyncio.sleep(0.5)

# —— MACD 子策略 ——
async def macd_strategy():
    fired=False
    while True:
        async with lock:
            df=klines['15m']
            if len(df)<26:
                await asyncio.sleep(1); continue
            macd_now,macd_prev=df['macd'].iloc[-1],df['macd'].iloc[-2]
            std=df['macd'].std()
            if not fired and macd_prev>0 and macd_now<macd_prev and abs(macd_now)>1.5*std:
                await bracket(0.15,latest_price,'SELL'); fired=True
            elif fired and macd_prev<0 and macd_now>macd_prev and abs(macd_now)>1.5*std:
                await bracket(0.15,latest_price,'BUY'); fired=False
        await asyncio.sleep(15)

# —— RVGI 子策略 ——
async def rvgi_strategy():
    fired_buy=fired_sell=False
    while True:
        async with lock:
            df=klines['3m']
            if 'rvgi' not in df:
                await asyncio.sleep(1); continue
            rv,sg=df['rvgi'].iloc[-1],df['rvsig'].iloc[-1]
            if not fired_buy and rv>sg:
                await bracket(0.05,latest_price,'BUY'); fired_buy=True; fired_sell=False
            if not fired_sell and rv<sg:
                await bracket(0.05,latest_price,'SELL'); fired_sell=True; fired_buy=False
        await asyncio.sleep(5)

# —— SuperTrend 子策略 ——
async def st_strategy():
    last_dir=None
    while True:
        async with lock:
            df=klines['15m']
            if 'st' not in df:
                await asyncio.sleep(1); continue
            cur_st=df['st'].iloc[-1]; prev_st=df['st'].iloc[-2]
            price=latest_price
            # 由红转绿 或 绿转红 判定
            if prev_st>price and cur_st<price and last_dir!='LONG':
                await bracket(0.15,price,'BUY'); last_dir='LONG'
            if prev_st<price and cur_st>price and last_dir!='SHORT':
                await bracket(0.15,price,'SELL'); last_dir='SHORT'
        await asyncio.sleep(10)

# —— time sync ——
async def sync_time():
    global TIME_OFFSET
    async with session.get(f"{REST_BASE}/fapi/v1/time") as r:
        srv=await r.json(); srv_ts=srv['serverTime']
        loc=int(time.time()*1000)
        TIME_OFFSET=srv_ts-loc
        logging.info("Time offset: %d ms", TIME_OFFSET)

# —— main entry ——
async def main():
    global session
    logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
    session = aiohttp.ClientSession()
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