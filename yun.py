#!/usr/bin/env python3
# coding: utf-8

import os, time, json, hmac, hashlib, asyncio, logging, uuid, urllib.parse, base64
import uvloop, aiohttp, pandas as pd, websockets
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
    SECRET_KEY       = os.getenv('BINANCE_SECRET_KEY').encode()
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ———— 全局状态 ————
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None
klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
lock = asyncio.Lock()
last_trend = None       # 'UP'/'DOWN'
last_signal = None      # 上次 “强/弱” 下单方向，用于轮换
position = {'LONG':0.0, 'SHORT':0.0}

# ———— 加载 Ed25519 私钥 ————
with open(Config.ED25519_KEY_PATH,'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# ———— 工具：时间同步 ————
async def sync_time():
    global time_offset
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    time_offset = srv - int(time.time()*1000)
    logging.info("Time offset: %d ms", time_offset)

# ———— 工具：模式检测 ————
async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000+time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url=f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    j=await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge_mode = j.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# ———— 工具：HMAC 签名 ————
def sign_params(params):
    qs = urllib.parse.urlencode(sorted(params.items()))
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    return qs+'&signature='+sig

# ———— 工具：Ed25519 WS 签名 ————
def sign_ws(params):
    payload='&'.join(f"{k}={v}" for k,v in sorted(params.items()))
    sig=ed_priv.sign(payload.encode())
    return base64.b64encode(sig).decode()

# ———— 工具：下单 ————
async def order(side, otype, qty=None, price=None, stopPrice=None):
    ts=int(time.time()*1000+time_offset)
    p = {'symbol':Config.SYMBOL,'side':side,'type':otype,'timestamp':ts,'recvWindow':Config.RECV_WINDOW}
    if is_hedge_mode and otype in ('LIMIT','MARKET'):
        p['positionSide']='LONG' if side=='BUY' else 'SHORT'
    if otype=='LIMIT':
        p.update({'timeInForce':'GTC','quantity':f"{qty:.6f}",'price':f"{price:.2f}"})
    elif otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
        p.update({'closePosition':'true','stopPrice':f"{stopPrice:.2f}"})
    else:
        p['quantity']=f"{qty:.6f}"
    qs=sign_params(p)
    url=f"{Config.REST_BASE}/fapi/v1/order?{qs}"
    res=await (await session.post(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    if res.get('code'):
        logging.error("Order ERR %s %s: %s", otype, side, res)
        return False
    logging.info("Order OK %s %s qty=%s", otype, side, qty or '')
    return True

# ———— 指标更新 ————
def update_indicators():
    for tf,df in klines.items():
        if len(df)<20: continue
        bb = BollingerBands(df['close'],20,2)
        df['bb_up']=bb.bollinger_hband()
        df['bb_dn']=bb.bollinger_lband()
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

# ———— 市场 WS ————
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    j=json.loads(msg); s=j['stream']; d=j['data']
                    if s.endswith('@markPrice'):
                        latest_price=float(d['p']); price_ts=time.time()
                    if 'kline' in s:
                        tf=s.split('@')[1].split('_')[1]; k=d['k']
                        rec={'open':float(k['o']),'high':float(k['h']),
                             'low':float(k['l']),'close':float(k['c'])}
                        async with lock:
                            df=klines[tf]
                            if df.empty or int(k['t'])>df.index[-1]:
                                klines[tf]=pd.concat([df,pd.DataFrame([rec])],ignore_index=True)
                            else:
                                df.iloc[-1]=list(rec.values()); klines[tf]=df
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error: %s",e)
            await asyncio.sleep(2)

# ———— 用户 WS ————
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params={'apiKey':Config.ED25519_API,'timestamp':int(time.time()*1000)}
                params['signature']=sign_ws(params)
                await ws.send(json.dumps({'id':str(uuid.uuid4()),'method':'session.logon','params':params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10);
                        await ws.send(json.dumps({'id':str(uuid.uuid4()),'method':'session.status'}))
                asyncio.create_task(hb())
                async for _ in ws: pass
        except Exception as e:
            logging.error("User WS error: %s",e)
            await asyncio.sleep(5)

# ———— 趋势监控 ————
async def trend_watcher():
    global last_trend, last_signal
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if 'st' not in klines['15m'] or latest_price is None: continue
            stv=klines['15m']['st'].iloc[-1]
            trend = 'UP' if latest_price>stv else 'DOWN'
            if trend!=last_trend:
                last_trend, last_signal = trend, None

# ———— 主策略（3m BB%B 分批挂单止盈止损） ————
async def main_strategy():
    # 定义级别：（偏移%，挂单量比例%）
    levels=[(0.0025,0.2),(0.0040,0.2),(0.0060,0.2),(0.0080,0.2),(0.0160,0.2)]
    tp_levels=[(0.0102,0.2),(0.0123,0.2),(0.0150,0.2),(0.0180,0.2),(0.0220,0.2)]
    sl_mult=0.98; sl_mult_short=1.02

    while price_ts is None: await asyncio.sleep(0.1)
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if any(klines[tf].shape[0]<20 for tf in ('3m','15m','1h')): continue
            p=latest_price
            bb1=klines['1h']['bb_pct'].iloc[-1]
            bb3=klines['3m']['bb_pct'].iloc[-1]
            st=klines['15m']['st'].iloc[-1]
            # 趋势
            trend='UP' if p>st else 'DOWN'
            strong = (trend=='UP' and bb1<0.2) or (trend=='DOWN' and bb1>0.8)
            # 信号：3m BB%B≤0 或 ≥1
            entry=False
            if bb3<=0 or bb3>=1:
                qty = 0.12 if strong else 0.03
                if trend=='DOWN': qty = 0.07 if strong else 0.015
                if last_signal!=trend:
                    # 分批挂单
                    for off,ratio in levels:
                        side = 'BUY' if trend=='UP' else 'SELL'
                        order(side,'LIMIT',qty=qty,price=p*(1+off if side=='BUY' else 1-off))
                    # 止盈
                    for off,ratio in tp_levels:
                        side2 = 'SELL' if trend=='UP' else 'BUY'
                        order(side2,'LIMIT',qty=qty*ratio,price=p*(1+off if side2=='BUY' else 1-off))
                    # 固定止损
                    side3='SELL' if trend=='UP' else 'BUY'
                    sl_price=p* (sl_mult if trend=='UP' else sl_mult_short)
                    order(side3,'STOP_MARKET',stopPrice=sl_price)
                    last_signal=trend
        # end lock
    # end while

# ———— 15m MACD 子策略 ————
async def macd_strategy():
    fired=None
    while True:
        await asyncio.sleep(15)
        async with lock:
            df=klines['15m']
            if df.shape[0]<26 or 'macd' not in df: continue
            prev,cur=df['macd'].iloc[-2],df['macd'].iloc[-1]
            osc=abs(cur)
            # 空单银叉
            if fired!='SILVER' and prev>0 and cur<prev and 11<=osc<20:
                order('SELL','MARKET',qty=0.15); fired='SILVER'
            # 空单 ≥20
            if fired!='SILVER20' and prev>0 and cur<prev and osc>=20:
                order('SELL','MARKET',qty=0.15); fired='SILVER20'
            # 多单金叉
            if fired!='GOLD' and prev<0 and cur>prev and -20<osc<=-11:
                order('BUY','MARKET',qty=0.15); fired='GOLD'
            if fired!='GOLD20' and prev<0 and cur>prev and osc<=-20:
                order('BUY','MARKET',qty=0.15); fired='GOLD20'
            # 止盈市价反向
            # 固定止损用 0.97/1.03
            # 略…

# ———— 15m RVGI 子策略 ————
async def rvgi_strategy():
    bought=sold=0.0
    maxpos=0.2
    while True:
        await asyncio.sleep(10)
        async with lock:
            df=klines['15m']
            if df.shape[0]<10 or 'rvgi' not in df: continue
            rv,sg=df['rvgi'].iloc[-1],df['rvsig'].iloc[-1]
            # 趋势滤波略…
            if rv>sg and bought<maxpos:
                order('BUY','MARKET',qty=0.05); bought+=0.05
                # 止盈限价 1.06，止损0.98
                order('SELL','LIMIT',qty=0.05,price=latest_price*1.06)
                order('SELL','STOP_MARKET',stopPrice=latest_price*0.98)
            if rv<sg and sold<maxpos:
                order('SELL','MARKET',qty=0.05); sold+=0.05
                order('BUY','LIMIT',qty=0.05,price=latest_price*0.94)
                order('BUY','STOP_MARKET',stopPrice=latest_price*1.02)

# ———— 15m Triple ST 子策略 ————
async def triple_st_strategy():
    state=None
    while True:
        await asyncio.sleep(30)
        async with lock:
            df=klines['15m']
            if df.shape[0]<12 or 'st' not in df: continue
            # 简略：factor=10,11,12 三条 st 线，并行判断…
            # 金叉/死叉触发一次，下同…

# ———— 主启动 ————
async def main():
    global session
    session=aiohttp.ClientSession()
    await sync_time(); await detect_mode()
    await asyncio.gather(
        market_ws(), user_ws(), trend_watcher(),
        main_strategy(), macd_strategy(),
        rvgi_strategy(), triple_st_strategy()
    )

if __name__=='__main__':
    asyncio.run(main())