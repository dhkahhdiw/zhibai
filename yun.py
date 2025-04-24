#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import hmac
import hashlib
import asyncio
import logging
import uuid
import urllib.parse
import base64

import uvloop
import aiohttp
import pandas as pd
import websockets
import watchfiles
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from ta.volatility import BollingerBands
from ta.trend import MACD
from numba import jit

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
    SYNC_INTERVAL    = 300   # 5分钟
    MAX_POSITION     = 2.0   # 最多持仓量

# 全局状态
session: aiohttp.ClientSession
time_offset = 0
is_hedge_mode = False
latest_price = None
price_ts = None

# —— 加载 Ed25519 私钥 ——
with open(Config.ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)

# —— 时间同步 & 模式检测 ——
async def sync_time():
    global time_offset
    js = await (await session.get(f"{Config.REST_BASE}/fapi/v1/time")).json()
    srv_ms = js['serverTime']
    local_ms = int(time.time() * 1000)
    time_offset = srv_ms - local_ms
    logging.info("Time offset: %dms", time_offset)

async def detect_mode():
    global is_hedge_mode
    ts = int(time.time() * 1000 + time_offset)
    qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
    sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
    url = f"{Config.REST_BASE}/fapi/v1/positionSide/dual?{qs}&signature={sig}"
    res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
    is_hedge_mode = res.get('dualSidePosition', False)
    logging.info("Hedge mode: %s", is_hedge_mode)

# —— PositionTracker ——
class PositionTracker:
    def __init__(self):
        self.long = self.short = 0.0
        self.lock = asyncio.Lock()

    async def sync(self):
        ts = int(time.time() * 1000 + time_offset)
        qs = urllib.parse.urlencode({'timestamp': ts, 'recvWindow': Config.RECV_WINDOW})
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        url = f"{Config.REST_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
        res = await (await session.get(url, headers={'X-MBX-APIKEY': Config.API_KEY})).json()
        async with self.lock:
            for pos in res:
                if pos['symbol'] == Config.SYMBOL:
                    amt = abs(float(pos['positionAmt']))
                    if pos['positionSide']=='LONG': self.long = amt
                    else: self.short = amt

    async def get_available(self, side: str) -> float:
        async with self.lock:
            if side=='BUY':
                return max(0.0, Config.MAX_POSITION - (self.long - self.short))
            return max(0.0, Config.MAX_POSITION - (self.short - self.long))

pos_tracker = PositionTracker()

# —— 批量下单器 & 单笔下单器 ——
class BatchOrderManager:
    BATCH_SIZE = 5
    def __init__(self):
        self.batch = []

    async def add(self, order):
        self.batch.append(order)
        if len(self.batch)>=self.BATCH_SIZE:
            await self.flush()

    async def flush(self):
        if not self.batch: return
        payload = {'batchOrders': json.dumps(self.batch)}
        await session.post(f"{Config.REST_BASE}/fapi/v1/batchOrders",
                           headers={'X-MBX-APIKEY':Config.API_KEY},
                           data=payload)
        self.batch.clear()

batch_mgr = BatchOrderManager()

class OrderManager:
    def __init__(self):
        self.lock = asyncio.Lock()

    def _sign(self, params):
        qs = urllib.parse.urlencode(sorted(params.items()))
        sig = hmac.new(Config.SECRET_KEY, qs.encode(), hashlib.sha256).hexdigest()
        return qs + "&signature=" + sig

    async def place(self, side, otype, qty=None, price=None, stopPrice=None, reduceOnly=False):
        # 仓位检查
        avail = await pos_tracker.get_available(side)
        if qty and qty>avail:
            logging.warning(f"仓位不足: 需 {qty:.6f}, 可用 {avail:.6f}")
            return
        ts = int(time.time()*1000 + time_offset)
        params = {'symbol':Config.SYMBOL,'side':side,'type':otype,
                  'timestamp':ts,'recvWindow':Config.RECV_WINDOW}
        if is_hedge_mode and otype in ('LIMIT','MARKET'):
            params['positionSide']='LONG' if side=='BUY' else 'SHORT'
        if otype=='LIMIT':
            params.update({'timeInForce':'GTC','quantity':f"{qty:.6f}",'price':f"{price:.2f}"})
            if reduceOnly: params['reduceOnly']='true'
        elif otype in ('STOP_MARKET','TAKE_PROFIT_MARKET'):
            params.update({'closePosition':'true','stopPrice':f"{stopPrice:.2f}"})
        else:
            params['quantity']=f"{qty:.6f}"
        # 转成批量下单格式
        async with self.lock:
            batch_mgr.add({'method':'POST','params':params})

order_mgr = OrderManager()

# —— 信号熔断与仲裁 ——
class SignalArbiter:
    PRIOR = {'main':3,'triple':2,'macd':1,'rvgi':1}
    def __init__(self):
        self.signals = {}  # strat -> (side, ts)
        self.lock = asyncio.Lock()

    async def register(self, strat, side):
        async with self.lock:
            self.signals[strat] = (side, time.time())

    async def decide(self):
        async with self.lock:
            score={'BUY':0.0,'SELL':0.0}
            for s,(side,ts) in self.signals.items():
                w=self.PRIOR.get(s,1)
                decay=max(0.0,1-(time.time()-ts)/60)
                score[side]+=w*decay
            if abs(score['BUY']-score['SELL'])<1.0:
                return None
            return 'BUY' if score['BUY']>score['SELL'] else 'SELL'

arbiter = SignalArbiter()

# —— 动态参数 ——
class DynamicParameters:
    def __init__(self):
        self.base=[0.0025,0.0040,0.0060,0.0080,0.0160]
    def update(self, atr):
        vol=atr/50
        return [x*vol for x in self.base]

dyn_params = DynamicParameters()

# —— DataManager ——
class DataManager:
    def __init__(self):
        # 初始只包含 OHLC
        self.klines={tf:pd.DataFrame(columns=['open','high','low','close'])
                     for tf in ('3m','15m','1h')}
        self.last_ts={tf:0 for tf in ('3m','15m','1h')}
        self.lock=asyncio.Lock()
        self.bb3=BollingerBands(close=pd.Series([0]*20),window=20,window_dev=2)  # 用于 3m

    async def update_kline(self, tf, rec):
        """按字典追加新行，避免列数不匹配"""
        async with self.lock:
            df = self.klines[tf]
            new_row = {
                'open': rec['o'],
                'high': rec['h'],
                'low':  rec['l'],
                'close':rec['c']
            }
            # 新周期
            if df.empty or rec['t'] > self.last_ts[tf]:
                df2 = pd.DataFrame([new_row])
                self.klines[tf] = pd.concat([df, df2], ignore_index=True)
            else:
                # 更新末行
                for k,v in new_row.items():
                    df.at[len(df)-1, k] = v
            self.last_ts[tf] = rec['t']
            self._update(tf)

    def _update(self, tf):
        df = self.klines[tf]
        if len(df) < 20:
            return
        # 根据不同周期计算指标
        if tf=='3m':
            # 用自定义 Bollinger
            series = df['close']
            sma = series.rolling(20).mean().iat[-1]
            std = series.rolling(20).std().iat[-1]
            up = sma + 2*std
            dn = sma - 2*std
            df.at[len(df)-1,'bb_up'] = up
            df.at[len(df)-1,'bb_dn'] = dn
            df['bb_pct'] = (df['close']-df['bb_dn'])/(df['bb_up']-df['bb_dn'])
        else:
            bb = BollingerBands(df['close'],20,2)
            df['bb_up'] = bb.bollinger_hband()
            df['bb_dn'] = bb.bollinger_lband()
            df['bb_pct'] = (df['close']-df['bb_dn'])/(df['bb_up']-df['bb_dn'])
        if tf=='15m':
            hl2 = (df['high']+df['low'])/2
            atr = df['high'].rolling(10).max() - df['low'].rolling(10).min()
            df['st'] = hl2 - 3*atr
            df['macd'] = MACD(df['close'],12,26,9).macd_diff()

    async def get(self, tf, col):
        async with self.lock:
            df = self.klines[tf]
            if col in df.columns and len(df)>0:
                return df[col].iat[-1]
        return None

data_mgr = DataManager()

# —— Numba 加速 SuperTrend ——
@jit(nopython=True)
def numba_supertrend(high, low, close, period, mult):
    n = len(close)
    st = [0.0]*n
    dirc = [True]*n
    hl2 = [(high[i]+low[i])/2 for i in range(n)]
    atr = [max(high[i-period+1:i+1]) - min(low[i-period+1:i+1]) for i in range(n)]
    up = [hl2[i] + mult*atr[i] for i in range(n)]
    dn = [hl2[i] - mult*atr[i] for i in range(n)]
    st[0] = up[0]
    for i in range(1,n):
        if close[i] > st[i-1]:
            st[i] = max(dn[i], st[i-1]); dirc[i]=True
        else:
            st[i] = min(up[i], st[i-1]);   dirc[i]=False
    return st, dirc

# —— 市场数据 WS ——
async def market_ws():
    global latest_price, price_ts
    retry = 0
    while True:
        try:
            async with websockets.connect(
                Config.WS_MARKET_URL,
                ping_interval=15, ping_timeout=10, close_timeout=5
            ) as ws:
                logging.info("Market WS connected")
                retry = 0
                async for msg in ws:
                    o = json.loads(msg)
                    s, d = o['stream'], o['data']
                    if s.endswith('@markPrice'):
                        latest_price, price_ts = float(d['p']), time.time()
                    if 'kline' in s:
                        tf = s.split('@')[1].split('_')[1]
                        k = d['k']
                        rec = {
                            't': k['t'],
                            'o': float(k['o']),
                            'h': float(k['h']),
                            'l': float(k['l']),
                            'c': float(k['c'])
                        }
                        await data_mgr.update_kline(tf, rec)
        except Exception as e:
            delay = min(2**retry, 30)
            logging.error("Market WS err %s, retry in %ds", e, delay)
            await asyncio.sleep(delay)
            retry += 1

# —— 用户流 WS ——
async def user_ws():
    while True:
        try:
            async with websockets.connect(Config.WS_USER_URL) as ws:
                logging.info("User WS connected")
                params = {
                    'apiKey': Config.ED25519_API,
                    'timestamp': int(time.time()*1000 + time_offset)
                }
                payload = '&'.join(f"{k}={v}" for k,v in sorted(params.items()))
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                params['signature'] = sig
                await ws.send(json.dumps({
                    'id': str(uuid.uuid4()),
                    'method': 'session.logon',
                    'params': params
                }))
                # 心跳
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
            logging.error("User WS err %s", e)
            await asyncio.sleep(5)

# —— 配置热加载 & 定时维护 ——
async def config_watcher():
    async for _ in watchfiles.awatch('/root/zhibai/'):
        logging.info("Config changed, reloading…")
        load_dotenv('/root/zhibai/.env')
        await pos_tracker.sync()

async def maintenance():
    asyncio.create_task(config_watcher())
    while True:
        await asyncio.sleep(Config.SYNC_INTERVAL)
        await sync_time()
        await detect_mode()
        await pos_tracker.sync()
        await batch_mgr.flush()

# —— 主策略 ——
async def main_strategy():
    tp_offs = [0.0102,0.0123,0.0150,0.0180,0.0220]
    while price_ts is None:
        await asyncio.sleep(0.1)
    while True:
        await asyncio.sleep(0.5)
        p = latest_price
        bb1 = await data_mgr.get('1h','bb_pct')
        bb3 = await data_mgr.get('3m','bb_pct')
        st  = await data_mgr.get('15m','st')
        if None in (bb1,bb3,st):
            continue
        if bb3<=0 or bb3>=1:
            side = 'BUY' if p>st else 'SELL'
            await arbiter.register('main', side)
            if await arbiter.decide() == side:
                strong = (p>st and bb1<0.2) or (p<st and bb1>0.8)
                levels = dyn_params.update(st)
                qty    = 0.12 if (p>st and strong) else 0.03
                if p<st:
                    qty = 0.07 if strong else 0.015
                rev = 'SELL' if side=='BUY' else 'BUY'
                # 逐级挂单
                for off in levels:
                    price_off = p*(1+off if side=='BUY' else 1-off)
                    await order_mgr.place(side,'LIMIT',qty=qty,price=price_off)
                # 止盈
                for off in tp_offs:
                    pt = p*(1+off if rev=='BUY' else 1-off)
                    await order_mgr.place(rev,'LIMIT',qty=qty*0.2,price=pt,reduceOnly=True)
                # 止损/止盈市价单
                slp = p*(0.98 if side=='BUY' else 1.02)
                ttype = 'STOP_MARKET' if side=='BUY' else 'TAKE_PROFIT_MARKET'
                await order_mgr.place(rev, ttype, stopPrice=slp)

# —— 子策略：MACD ——
macd_cycle = None
async def macd_strategy():
    global macd_cycle
    while True:
        await asyncio.sleep(15)
        df = data_mgr.klines['15m']
        if len(df)<27 or 'macd' not in df:
            continue
        prev, cur = df['macd'].iat[-2], df['macd'].iat[-1]
        osc = abs(cur)
        if prev>0 and cur<prev and osc>=11 and macd_cycle!='DOWN':
            macd_cycle='DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.017)
            await arbiter.register('macd','SELL')
        if prev<0 and cur>prev and osc>=11 and macd_cycle!='UP':
            macd_cycle='UP'
            await order_mgr.place('BUY','MARKET',qty=0.017)
            await arbiter.register('macd','BUY')

# —— 子策略：RVGI ——
rvgi_cycle = None
async def rvgi_strategy():
    global rvgi_cycle
    while True:
        await asyncio.sleep(10)
        df = data_mgr.klines['15m']
        if len(df)<11 or not all(c in df.columns for c in ('rvgi','rvsig')):
            continue
        rv, sg = df['rvgi'].iat[-1], df['rvsig'].iat[-1]
        if rv>sg and rvgi_cycle!='UP':
            rvgi_cycle='UP'
            await order_mgr.place('BUY','MARKET',qty=0.016)
            await order_mgr.place('SELL','LIMIT',qty=0.016,price=latest_price*1.06,reduceOnly=True)
            await order_mgr.place('SELL','STOP_MARKET',stopPrice=latest_price*0.98)
            await arbiter.register('rvgi','BUY')
        if rv<sg and rvgi_cycle!='DOWN':
            rvgi_cycle='DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.016)
            await order_mgr.place('BUY','LIMIT',qty=0.016,price=latest_price*0.94,reduceOnly=True)
            await order_mgr.place('BUY','TAKE_PROFIT_MARKET',stopPrice=latest_price*1.02)
            await arbiter.register('rvgi','SELL')

# —— 子策略：三重 SuperTrend ——
triple_cycle = None
async def triple_st_strategy():
    global triple_cycle
    while True:
        await asyncio.sleep(30)
        df = data_mgr.klines['15m']
        if len(df)<12:
            continue
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        s1, d1 = numba_supertrend(high,low,close,10,1)
        s2, d2 = numba_supertrend(high,low,close,11,2)
        s3, d3 = numba_supertrend(high,low,close,12,3)
        up = d1[-1] and d2[-1] and d3[-1]
        dn = not (d1[-1] or d2[-1] or d3[-1])
        if up and triple_cycle!='UP':
            triple_cycle='UP'
            await order_mgr.place('BUY','MARKET',qty=0.015)
            await arbiter.register('triple','BUY')
        if dn and triple_cycle!='DOWN':
            triple_cycle='DOWN'
            await order_mgr.place('SELL','MARKET',qty=0.015)
            await arbiter.register('triple','SELL')
        # 反向平仓
        if triple_cycle=='UP' and not up:
            await order_mgr.place('SELL','MARKET',qty=0.015)
        if triple_cycle=='DOWN' and not dn:
            await order_mgr.place('BUY','MARKET',qty=0.015)

# —— 启动 ——
async def main():
    global session
    session = aiohttp.ClientSession()
    await sync_time()
    await detect_mode()
    try:
        await asyncio.gather(
            market_ws(),
            user_ws(),
            maintenance(),
            main_strategy(),
            macd_strategy(),
            rvgi_strategy(),
            triple_st_strategy()
        )
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())