#!/usr/bin/env python3
# coding: utf-8

import os
import time
import json
import math
import asyncio
import logging
import signal

import ccxt.async_support as ccxt
import aiohttp
import websockets
import pandas as pd
from numba import jit
from ta.trend import MACD, ADXIndicator
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from collections import defaultdict

# —— 全局日志配置 ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— 环境与密钥加载 ——
load_dotenv('/root/zhibai/.env')
ED25519_KEY_PATH = os.getenv('YZ_ED25519_KEY_PATH')
with open(ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 private key loaded")

# —— 交易对与时间框架 ——
SYMBOL       = 'ETH/USDC'
TF_CONFIG    = {'3m':'3m','15m':'15m','1h':'1h'}
TIMEFRAMES   = list(TF_CONFIG.values())

# —— CCXT 交易所实例 ——
exchange = ccxt.binance({
    'apiKey': os.getenv('YZ_BINANCE_API_KEY'),
    'secret': os.getenv('YZ_BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'hedgeMode': True},
})

# —— 工具函数 ——
def quantize(val: float, step: float) -> float:
    return math.floor(val/step) * step

@jit(nopython=True)
def numba_supertrend(h, l, c, per, mult):
    n = len(c)
    st = [0.0]*n; dirc=[False]*n
    hl2= [(h[i]+l[i])/2 for i in range(n)]
    atr= [max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
    up = [hl2[i] + mult*atr[i] for i in range(n)]
    dn = [hl2[i] - mult*atr[i] for i in range(n)]
    st[0],dirc[0]=up[0],True
    for i in range(1,n):
        if c[i]>st[i-1]:
            st[i],dirc[i]=max(dn[i],st[i-1]),True
        else:
            st[i],dirc[i]=min(up[i],st[i-1]),False
    return st,dirc

# —— 数据管理 ——
class DataManager:
    def __init__(self):
        self.klines = {tf: pd.DataFrame() for tf in TIMEFRAMES}
        self.price  = None
        self.lock   = asyncio.Lock()
        self._evt   = asyncio.Event()

    async def load_history(self):
        LOG.info("Loading history candles")
        async with self.lock:
            for tf in TIMEFRAMES:
                LOG.debug(f" Fetching {tf}")
                ohlcv = await exchange.fetch_ohlcv(SYMBOL, TF_CONFIG[tf], limit=1000)
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
                df.set_index('ts', inplace=True)
                self.klines[tf] = df
        LOG.info("History load complete")

    async def update_kline(self, tf, ohlc):
        ts, o, h, l, c, v = ohlc
        async with self.lock:
            df = self.klines[tf]
            df.loc[ts] = [o,h,l,c,v]
            if len(df)>1000:
                df.drop(df.index[0], inplace=True)
            self.klines[tf] = df
            LOG.debug(f"{tf} kline updated @ {ts}: o={o:.4f},c={c:.4f}")
            self._evt.set()

    async def set_price(self, price):
        async with self.lock:
            self.price = price
            LOG.debug(f"Mark price update: {price:.4f}")
            self._evt.set()

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

data_mgr = DataManager()

# —— 持仓与 OCO 管理 ——
class PositionTracker:
    def __init__(self):
        self.positions = {}
        self.lock      = asyncio.Lock()
        self.next_id   = 1

    async def on_fill(self, side, amount, price, sl, tp):
        async with self.lock:
            cid = self.next_id; self.next_id += 1
            pos = {
                'side': side, 'qty': amount,
                'sl': sl, 'tp': tp,
                'active': True,
                'sl_oid': None, 'tp_oid': None
            }
            self.positions[cid] = pos
            LOG.info(f"[POS#{cid}] Open {side}@{price:.4f} SL={sl:.4f} TP={tp:.4f}")

            # 下 OCO：STOP_MARKET + TAKE_PROFIT_MARKET
            sl_ord = await exchange.fapiPrivatePostOrder({
                'symbol': exchange.market_id(SYMBOL),
                'side': 'SELL' if side=='buy' else 'BUY',
                'type': 'STOP_MARKET',
                'stopPrice': sl,
                'closePosition': True,
                'workingType':'MARK_PRICE',
                'newClientOrderId':f"sl_{cid}"
            })
            tp_ord = await exchange.fapiPrivatePostOrder({
                'symbol': exchange.market_id(SYMBOL),
                'side': 'SELL' if side=='buy' else 'BUY',
                'type': 'TAKE_PROFIT_MARKET',
                'stopPrice': tp,
                'closePosition': True,
                'workingType':'MARK_PRICE',
                'newClientOrderId':f"tp_{cid}"
            })
            pos['sl_oid'] = sl_ord['orderId']
            pos['tp_oid'] = tp_ord['orderId']

    async def on_order_update(self, msg):
        oid    = msg['o']['i']
        status = msg['o']['X']
        async with self.lock:
            for cid, p in self.positions.items():
                if not p['active']: continue
                if oid in (p['sl_oid'], p['tp_oid']) and status=='FILLED':
                    p['active'] = False
                    other = p['tp_oid'] if oid==p['sl_oid'] else p['sl_oid']
                    LOG.info(f"[POS#{cid}] OCO trigger {oid} → cancel {other}")
                    await exchange.cancel_order(other, SYMBOL)
                    return

pos_tracker = PositionTracker()

# —— 策略实现 ——
class MainStrategy:
    def __init__(self): self.last=0
    async def check(self, price):
        now = time.time()
        if now-self.last<1: return
        df15 = data_mgr.klines['15m']
        if len(df15)<100: return
        df = data_mgr.klines["15m"]
        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99): return
        adx = ADXIndicator(df15['high'],df15['low'],df15['close'],14).adx().iat[-1]
        if adx<=25:
            LOG.debug("Main: ADX too low"); return
        bb = (df15['close']-df15['close'].rolling(20).mean()+2*df15['close'].rolling(20).std())/(4*df15['close'].rolling(20).std())
        sig = bb.iat[-1]
        if not (sig<=0 or sig>=1):
            LOG.debug("Main: BB no signal"); return
        side = 'buy' if sig<=0 else 'sell'
        level = 0.005 if side=='buy' else -0.005
        qty   = 0.016
        price0= price*(1+level)
        sl    = price*(0.98 if side=='buy' else 1.02)
        tp    = price*(1.02 if side=='buy' else 0.98)
        LOG.info(f"Main: placing {side} {qty}@{price0:.4f}")
        order = await exchange.create_limit_order(SYMBOL, side, qty, price0, {'timeInForce':'GTC'})
        if order['status']=='closed':
            await pos_tracker.on_fill(side, qty, price0, sl, tp)
        self.last = now

class MACDStrategy:
    def __init__(self): self.in_pos=False
    async def check(self, price):
        df = data_mgr.klines['15m']
        if len(df)<30: return
        df = data_mgr.klines["15m"]
        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99): return
        macd = MACD(df['close'],12,26,9).macd_diff()
        prev, curr = macd.iat[-2], macd.iat[-1]
        if prev>0>curr and not self.in_pos:
            qty=0.015; price0=price*1.005
            LOG.info(f"MACD: SELL {qty}@{price0:.4f}")
            o=await exchange.create_limit_order(SYMBOL,'sell',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('sell',qty,price0,price*1.03,price*0.97)
                self.in_pos=True
        if prev<0<curr and self.in_pos:
            qty=0.015; price0=price*0.995
            LOG.info(f"MACD: BUY {qty}@{price0:.4f}")
            o=await exchange.create_limit_order(SYMBOL,'buy',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('buy',qty,price0,price*0.97,price*1.03)
                self.in_pos=False

class TripleTrendStrategy:
    def __init__(self): self.last=0; self.active=False
    async def check(self, price):
        now=time.time()
        if now-self.last<1: return
        df15=data_mgr.klines['15m']
        if len(df15)<30: return
        df = data_mgr.klines["15m"]
        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99): return
        h,l,c = df15['high'].values, df15['low'].values, df15['close'].values
        st,dirc = numba_supertrend(h,l,c,10,3)
        up = dirc[-1]; dn = not dirc[-1]
        if up and not self.active:
            qty=0.017; price0=price*0.996
            LOG.info(f"Triple: BUY {qty}@{price0:.4f}")
            o=await exchange.create_limit_order(SYMBOL,'buy',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('buy',qty,price0,price*0.97,price*1.02)
            self.active=True
        if dn and not self.active:
            qty=0.017; price0=price*1.004
            LOG.info(f"Triple: SELL {qty}@{price0:.4f}")
            o=await exchange.create_limit_order(SYMBOL,'sell',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('sell',qty,price0,price*1.03,price*0.98)
            self.active=True
        self.last=now

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket 连接 ——
async def market_ws():
    streams = []
    mid = exchange.market_id(SYMBOL)
    for tf in TF_CONFIG:
        streams.append(f"{mid}@kline_{tf}")
    streams.append(f"{mid}@markPrice")
    uri = f"wss://fstream.binance.com/stream?streams=" + '/'.join(streams)
    retry = 0
    while True:
        try:
            LOG.info("Connecting market WS")
            async with websockets.connect(uri, ping_interval=None) as ws:
                retry=0
                async for msg in ws:
                    o = json.loads(msg)
                    st= o['stream']; d=o['data']
                    if st.endswith('@markPrice'):
                        await data_mgr.set_price(float(d['p']))
                    else:
                        tf = st.split('@')[1].split('_')[1]
                        k  = d['k']
                        await data_mgr.update_kline(tf, [
                            k['t'],float(k['o']),float(k['h']),
                            float(k['l']),float(k['c']),float(k['v'])
                        ])
        except Exception as e:
            LOG.error(f"Market WS error: {e}")
            await asyncio.sleep(min(2**retry,30))
            retry+=1

async def user_ws():
    retry=0
    while True:
        try:
            LOG.info("Getting listen key")
            lk = (await exchange.fapiPrivatePostListenKey())['listenKey']
            uri = f"wss://fstream.binance.com/ws/{lk}"
            async with websockets.connect(uri, ping_interval=None) as ws:
                LOG.info("Connected user WS")
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get('e')=='ORDER_TRADE_UPDATE':
                        await pos_tracker.on_order_update(data)
        except Exception as e:
            LOG.error(f"User WS error: {e}")
            await asyncio.sleep(5); retry+=1

# —— 后台任务 ——
async def maintenance():
    while True:
        await asyncio.sleep(60)
        try:
            await exchange.load_markets()
            LOG.debug("Markets refreshed")
        except Exception:
            LOG.exception("Maintenance error")

async def engine():
    while True:
        await data_mgr.wait_update()
        price = data_mgr.price
        for strat in strategies:
            try:
                await strat.check(price)
            except Exception:
                LOG.exception("Strategy error")

# —— 启动 ——
async def main():
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, lambda: asyncio.create_task(exchange.close()))
    await data_mgr.load_history()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())