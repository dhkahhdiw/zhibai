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

session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None
klines = {'3m': pd.DataFrame(), '15m': pd.DataFrame(), '1h': pd.DataFrame()}
lock = asyncio.Lock()

# ———— 全局标记 ————
last_trend = None        # 'UP'/'DOWN'
last_signal = None       # 主策略上次落单趋势
macd_state = None        # 'SILVER','SILVER20','GOLD','GOLD20'
rvgi_state = {'long':0.0,'short':0.0}
triple_st_state = None   # 'UP'/'DN'

# ———— 加载 Ed25519 私钥 ————
with open(Config.ED25519_KEY_PATH,'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# ———— 时间同步 & 模式检测 ————
async def sync_time():
    global time_offset
    r = await session.get(f"{Config.REST_BASE}/fapi/v1/time")
    srv = (await r.json())['serverTime']
    time_offset = srv - int(time.time()*1000)
    logging.info("Time offset: %d ms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time()*1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp':ts,'recvWindow':Config.RECV_WINDOW})
    sig= hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url=f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    j = await (await session.get(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    is_hedge_mode = j.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# ———— 签名与下单 ————
def sign_params(p):
    qs = urllib.parse.urlencode(sorted(p.items()))
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    return f"{qs}&signature={sig}"

def sign_ws(params):
    payload = '&'.join(f"{k}={v}" for k,v in sorted(params.items()))
    return base64.b64encode(ed_priv.sign(payload.encode())).decode()

async def order(side, otype, qty=None, price=None, stopPrice=None, reduceOnly=False):
    ts = int(time.time()*1000 + time_offset)
    p = {'symbol':Config.SYMBOL,'side':side,'type':otype,'timestamp':ts,'recvWindow':Config.RECV_WINDOW}
    if is_hedge_mode and otype in ('LIMIT','MARKET'):
        p['positionSide'] = 'LONG' if side=='BUY' else 'SHORT'
    if otype=='LIMIT':
        p.update({'timeInForce':'GTC','quantity':f"{qty:.6f}",'price':f"{price:.2f}"})
        if reduceOnly: p['reduceOnly']='true'
    elif otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
        p.update({'closePosition':'true','stopPrice':f"{stopPrice:.2f}"})
    else:
        p['quantity'] = f"{qty:.6f}"
    qs = sign_params(p)
    url = f"{Config.REST_BASE}/fapi/v1/order?{qs}"
    res = await (await session.post(url, headers={'X-MBX-APIKEY':Config.API_KEY})).json()
    if res.get('code'):
        logging.error("Order ERR %s %s: %s", otype, side, res)
        return False
    logging.info("Order OK %s %s qty=%s", otype, side, qty or '')
    return True

# ———— 指标计算 ————
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

# ———— WS 订阅 ————
async def market_ws():
    global latest_price, price_ts
    while True:
        try:
            async with websockets.connect(Config.WS_MARKET_URL) as ws:
                logging.info("Market WS connected")
                async for msg in ws:
                    o = json.loads(msg); s=o['stream']; d=o['data']
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
                                df.iloc[-1]=list(rec.values())
                            update_indicators()
        except Exception as e:
            logging.error("Market WS error: %s", e)
            await asyncio.sleep(2)

async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params={'apiKey':Config.ED25519_API,'timestamp':int(time.time()*1000)}
                params['signature']=sign_ws(params)
                await ws.send(json.dumps({'id':str(uuid.uuid4()),
                                          'method':'session.logon','params':params}))
                async def hb():
                    while True:
                        await asyncio.sleep(10)
                        await ws.send(json.dumps({'id':str(uuid.uuid4()),
                                                  'method':'session.status'}))
                asyncio.create_task(hb())
                async for _ in ws: pass
        except Exception as e:
            logging.error("User WS error: %s", e)
            await asyncio.sleep(5)

# ———— 趋势监控 ————
async def trend_watcher():
    global last_trend, last_signal
    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if 'st' not in klines['15m'] or latest_price is None: continue
            stv = klines['15m']['st'].iloc[-1]
            trend = 'UP' if latest_price > stv else 'DOWN'
            if trend != last_trend:
                last_trend, last_signal = trend, None

# ———— 主策略 ————
async def main_strategy():
    global last_signal
    levels    = [0.0025,0.0040,0.0060,0.0080,0.0160]
    tp_levels = [0.0102,0.0123,0.0150,0.0180,0.0220]
    sl_up, sl_dn = 0.98, 1.02

    # 等待首个价
    while price_ts is None:
        await asyncio.sleep(0.1)

    while True:
        await asyncio.sleep(0.2)
        async with lock:
            if any(len(klines[tf])<20 for tf in ('3m','15m','1h')):
                continue
            p   = latest_price
            bb1 = klines['1h']['bb_pct'].iloc[-1]
            bb3 = klines['3m']['bb_pct'].iloc[-1]
            stv = klines['15m']['st'].iloc[-1]
            trend  = 'UP' if p>stv else 'DOWN'
            strong = (trend=='UP' and bb1<0.2) or (trend=='DOWN' and bb1>0.8)

            # 只在 首次 bb3 突破 且 与上次趋势不同时 下单
            if (bb3<=0 or bb3>=1) and last_signal != trend:
                # 按强弱 & 方向 决定 qty
                qty = 0.12 if strong else 0.03
                if trend=='DOWN': qty = 0.07 if strong else 0.015

                side, rev = ('BUY','SELL') if trend=='UP' else ('SELL','BUY')

                # 第一次挂单：分批限价
                for off in levels:
                    price_off = p * (1+off if side=='BUY' else 1-off)
                    await order(side, 'LIMIT', qty=qty, price=price_off)

                # 止盈：分批限价 + reduceOnly
                for off, ratio in zip(tp_levels, [0.2]*5):
                    price_tp = p * (1+off if rev=='BUY' else 1-off)
                    await order(rev, 'LIMIT', qty=qty*ratio,
                                price=price_tp, reduceOnly=True)

                # 初始止损
                slp = p * (sl_up if trend=='UP' else sl_dn)
                await order(rev, 'STOP_MARKET', stopPrice=slp)

                last_signal = trend

# ———— 15m MACD 子策略 ————
async def macd_strategy():
    global macd_state
    while True:
        await asyncio.sleep(15)
        async with lock:
            df = klines['15m']
            if len(df)<26 or 'macd' not in df: continue
            prev = float(df['macd'].iloc[-2])
            cur  = float(df['macd'].iloc[-1])
            osc  = abs(cur)

            # 银叉 空
            if macd_state!='SILVER' and prev>0 and cur<prev and 11<=osc<20:
                await order('SELL','MARKET', qty=0.15)
                macd_state='SILVER'
            if macd_state!='SILVER20' and prev>0 and cur<prev and osc>=20:
                await order('SELL','MARKET', qty=0.15)
                macd_state='SILVER20'
            # 金叉 多
            if macd_state!='GOLD' and prev<0 and cur>prev and 11<=osc<20:
                await order('BUY','MARKET', qty=0.15)
                macd_state='GOLD'
            if macd_state!='GOLD20' and prev<0 and cur>prev and osc>=20:
                await order('BUY','MARKET', qty=0.15)
                macd_state='GOLD20'

# ———— 15m RVGI 子策略 ————
async def rvgi_strategy():
    while True:
        await asyncio.sleep(10)
        async with lock:
            df=klines['15m']
            if len(df)<10 or 'rvgi' not in df: continue
            rv, sg = float(df['rvgi'].iloc[-1]), float(df['rvsig'].iloc[-1])
            # 多头开仓
            if rv>sg and rvgi_state['long']<0.2:
                await order('BUY','MARKET', qty=0.05)
                rvgi_state['long'] += 0.05
                # 止盈 / 止损
                await order('SELL','LIMIT', qty=0.05, price=latest_price*1.06, reduceOnly=True)
                await order('SELL','STOP_MARKET', stopPrice=latest_price*0.98)
            # 空头开仓
            if rv<sg and rvgi_state['short']<0.2:
                await order('SELL','MARKET', qty=0.05)
                rvgi_state['short'] += 0.05
                await order('BUY','LIMIT', qty=0.05, price=latest_price*0.94, reduceOnly=True)
                await order('BUY','STOP_MARKET', stopPrice=latest_price*1.02)

# ———— SuperTrend 计算 ————
def supertrend(df, period=10, multiplier=3.0):
    hl2=(df['high']+df['low'])/2
    atr=df['high'].rolling(period).max()-df['low'].rolling(period).min()
    up=hl2+multiplier*atr; dn=hl2-multiplier*atr
    st=pd.Series(index=df.index); dirc=pd.Series(True,index=df.index)
    for i in range(len(df)):
        if i==0:
            st.iloc[i]=up.iloc[i]
        else:
            prev=st.iloc[i-1]; price=df['close'].iloc[i]
            if price>prev:
                st.iloc[i]=max(dn.iloc[i],prev); dirc.iloc[i]=True
            else:
                st.iloc[i]=min(up.iloc[i],prev); dirc.iloc[i]=False
    return st, dirc

# ———— 15m 三重 SuperTrend 子策略 ————
async def triple_st_strategy():
    global triple_st_state
    while True:
        await asyncio.sleep(30)
        async with lock:
            df=klines['15m']
            if len(df)<12: continue
            st1, d1 = supertrend(df,10,3)
            st2, d2 = supertrend(df,11,3)
            st3, d3 = supertrend(df,12,3)
            up = d1.iloc[-1] and d2.iloc[-1] and d3.iloc[-1]
            dn = not d1.iloc[-1] and not d2.iloc[-1] and not d3.iloc[-1]
            # 开仓
            if up and triple_st_state!='UP':
                await order('BUY','MARKET', qty=0.15)
                triple_st_state='UP'
            if dn and triple_st_state!='DN':
                await order('SELL','MARKET', qty=0.15)
                triple_st_state='DN'
            # 止盈
            if triple_st_state=='UP' and not(up):
                await order('SELL','MARKET', qty=0.15); triple_st_state=None
            if triple_st_state=='DN' and not(dn):
                await order('BUY','MARKET', qty=0.15); triple_st_state=None

# ———— 启动 ————
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time(); await detect_mode()
    try:
        await asyncio.gather(
            market_ws(), user_ws(), trend_watcher(),
            main_strategy(), macd_strategy(),
            rvgi_strategy(), triple_st_strategy()
        )
    finally:
        await session.close()

if __name__=='__main__':
    asyncio.run(main())