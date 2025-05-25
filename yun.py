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

# —— Logging ——
LOG = logging.getLogger('bot')
LOG.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— Load env ——
load_dotenv('/root/zhibai/.env')
ED25519_KEY = os.getenv('YZ_ED25519_KEY_PATH')
with open(ED25519_KEY, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
LOG.info("Ed25519 key loaded")

# —— Config ——
SYMBOL = 'ETH/USDC'
TIMEFRAMES = {'3m':'3m','15m':'15m','1h':'1h'}
TFS = list(TIMEFRAMES.values())
CCXT_PARAMS = {
    'apiKey': os.getenv('YZ_BINANCE_API_KEY'),
    'secret': os.getenv('YZ_BINANCE_SECRET_KEY'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'hedgeMode': True},
}

# —— Initialize exchange client ——
exchange = ccxt.binance(CCXT_PARAMS)

# —— Data Manager ——
class DataManager:
    def __init__(self):
        self.klines = {tf:pd.DataFrame() for tf in TFS}
        self.lock = asyncio.Lock()
        self.price = None
        self._evt = asyncio.Event()

    async def load_history(self):
        async with self.lock:
            for tf in TFS:
                ohlc = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAMES[tf], limit=1000)
                df = pd.DataFrame(ohlc, columns=['ts','open','high','low','close','vol'])
                df.set_index('ts', inplace=True)
                self.klines[tf] = df
        LOG.debug("Loaded history")

    async def update_kline(self, tf, ohlc):
        async with self.lock:
            df = self.klines[tf]
            ts = ohlc[0]
            if ts in df.index:
                df.loc[ts] = ohlc[1:5]
            else:
                df.loc[ts] = ohlc[1:5]
                if len(df) > 1000:
                    df.drop(df.index[0], inplace=True)
            self.klines[tf] = df
            self._evt.set()

    async def set_price(self, price):
        async with self.lock:
            self.price = price
            self._evt.set()

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

data_mgr = DataManager()

# —— Supertrend JIT ——
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

# —— Position & OCO Manager ——
class PositionTracker:
    def __init__(self):
        self.positions = {}     # cloid -> pos
        self.lock = asyncio.Lock()
        self.next_id = 1

    async def on_fill(self, side, amount, price, sl, tp):
        async with self.lock:
            cid = self.next_id; self.next_id+=1
            pos = {
                'side': side,
                'qty': amount,
                'sl': sl,
                'tp': tp,
                'active': True,
                'sl_oid': None,
                'tp_oid': None
            }
            self.positions[cid] = pos
            LOG.info(f"Opened #{cid} {side}@{price:.4f} SL={sl:.4f} TP={tp:.4f}")
            # place OCO
            oco = await exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell' if side=='buy' else 'buy', amount, None, {
                'stopPrice': sl,
                'closePosition': True,
                'workingType': 'MARK_PRICE',
                'newClientOrderId':f"sl_{cid}"
            })
            pos['sl_oid'] = oco['id']
            tp_ord = await exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell' if side=='buy' else 'buy', amount, None, {
                'stopPrice': tp,
                'closePosition': True,
                'workingType': 'MARK_PRICE',
                'newClientOrderId':f"tp_{cid}"
            })
            pos['tp_oid'] = tp_ord['id']

    async def on_order_update(self, msg):
        oid = msg['o']['i']
        status = msg['o']['X']
        async with self.lock:
            # check each pos
            for cid, p in self.positions.items():
                if not p['active']: continue
                if oid in (p['sl_oid'], p['tp_oid']) and status=='FILLED':
                    p['active']=False
                    other = p['tp_oid'] if oid==p['sl_oid'] else p['sl_oid']
                    await exchange.cancel_order(other, SYMBOL)
                    LOG.info(f"OCO #{cid}: triggered {oid}, canceled {other}")
                    return

pos_tracker = PositionTracker()

# —— Strategy Base ——
class Strategy:
    async def check(self, price): ...

class MainStrategy(Strategy):
    def __init__(self):
        self.last = 0
    async def check(self, price):
        now=time.time()
        if now-self.last<1: return
        df15 = data_mgr.klines['15m']
        df = data_mgr.klines["15m"]
        if len(df15)<99: return
        # indicators
        adx = ADXIndicator(df15['high'],df15['low'],df15['close'],14)
        if adx.adx().iat[-1]<=25: return
        ma7, ma25, ma99 = df.ma7.iat[-1], df.ma25.iat[-1], df.ma99.iat[-1]
        if not (price < ma7 < ma25 < ma99 or price > ma7 > ma25 > ma99): return
        bb3 = (df15['close']-df15['close'].rolling(20).mean()+2*df15['close'].rolling(20).std())/(4*df15['close'].rolling(20).std())
        if not (bb3.iat[-1]<=0 or bb3.iat[-1]>=1): return
        side = 'buy' if bb3.iat[-1]<=0 else 'sell'
        # compute levels
        level = 0.005 if side=='buy' else -0.005
        qty = 0.016
        price0 = price*(1+level)
        sl = price*(0.98 if side=='buy' else 1.02)
        tp = price*(1.02 if side=='buy' else 0.98)
        order = await exchange.create_limit_order(SYMBOL, side, qty, price0, {'timeInForce':'GTC'})
        if order['status']=='closed':
            await pos_tracker.on_fill(side, qty, price0, sl, tp)
        self.last = now

class MACDStrategy(Strategy):
    def __init__(self): self.in_pos=False
    async def check(self, price):
        df = data_mgr.klines['15m']
        if len(df)<30: return
        macd = MACD(df['close'],12,26,9).macd_diff()
        prev, curr = macd.iat[-2], macd.iat[-1]
        if prev>0>curr and not self.in_pos:
            qty=0.015; price0=price*1.005
            o = await exchange.create_limit_order(SYMBOL,'sell',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('sell',qty,price0,price*1.03,price*0.97)
                self.in_pos=True
        if prev<0<curr and self.in_pos:
            qty=0.015; price0=price*0.995
            o = await exchange.create_limit_order(SYMBOL,'buy',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('buy',qty,price0,price*0.97,price*1.03)
                self.in_pos=False

class TripleTrendStrategy(Strategy):
    def __init__(self): self.last=0; self.active=False
    async def check(self, price):
        now=time.time()
        if now-self.last<1: return
        df15=data_mgr.klines['15m']
        if len(df15)<30: return
        h,l,c = df15['high'].values, df15['low'].values, df15['close'].values
        st,dirc = numba_supertrend(h,l,c,10,3)
        up, dn = dirc[-1], not dirc[-1]
        if up and not self.active:
            qty=0.017; price0=price*0.996
            o = await exchange.create_limit_order(SYMBOL,'buy',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('buy',qty,price0,price*0.97,price*1.02)
            self.active=True
        if dn and not self.active:
            qty=0.017; price0=price*1.004
            o = await exchange.create_limit_order(SYMBOL,'sell',qty,price0,{'timeInForce':'GTC'})
            if o['status']=='closed':
                await pos_tracker.on_fill('sell',qty,price0,price*1.03,price*0.98)
            self.active=True
        self.last=now

strategies = [MainStrategy(), MACDStrategy(), TripleTrendStrategy()]

# —— WebSocket Market & User Streams ——
async def market_ws():
    uri = f"wss://fstream.binance.com/ws/{exchange.market_id(SYMBOL)}@kline_3m/{exchange.market_id(SYMBOL)}@kline_15m/{exchange.market_id(SYMBOL)}@kline_1h/{exchange.market_id(SYMBOL)}@markPrice"
    async for ws in websockets.connect(uri, ping_interval=None):
        try:
            async for msg in ws:
                o = json.loads(msg)
                stream, data = o['stream'], o['data']
                if stream.endswith('@markPrice'):
                    price = float(data['p']); await data_mgr.set_price(price)
                else:
                    tf = stream.split('@')[1].split('_')[1]
                    k = data['k']
                    await data_mgr.update_kline(tf, [
                        k['t'], float(k['o']), float(k['h']), float(k['l']), float(k['c']), float(k['v'])
                    ])
        except Exception as e:
            LOG.error("Market WS error %s", e)
            await asyncio.sleep(5)

async def user_ws():
    # use ED25519_API key to sign user stream (ccxt does not support ed25519 directly)
    # fallback: use REST to get listenKey then websockets
    while True:
        try:
            lk = (await exchange.fapiPrivatePostListenKey())['listenKey']
            uri = f"wss://fstream.binance.com/ws/{lk}"
            async with websockets.connect(uri, ping_interval=None) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get('e')=='ORDER_TRADE_UPDATE':
                        await pos_tracker.on_order_update(data)
        except Exception as e:
            LOG.error("User WS error %s", e)
            await asyncio.sleep(5)

# —— Maintenance & Engine ——
async def maintenance():
    while True:
        await asyncio.sleep(60)
        try:
            await exchange.load_markets()
        except Exception:
            LOG.exception("maintenance")

async def engine():
    while True:
        await data_mgr.wait_update()
        price = data_mgr.price
        if price is None: continue
        for strat in strategies:
            try:
                await strat.check(price)
            except Exception:
                LOG.exception("strat")

# —— Main ——
async def main():
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, lambda: asyncio.create_task(exchange.close()))
    await data_mgr.load_history()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())