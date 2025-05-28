#!/usr/bin/env python3
# coding: utf-8

import os, time, math, json, signal, asyncio, logging, base64
import ccxt.async_support as ccxt
import websockets
import pandas as pd
from ta.trend import MACD, ADXIndicator
from numba import jit
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# —— 日志 ——
LOG = logging.getLogger('vhf_bot')
LOG.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LOG.addHandler(sh)

# —— 环境 & 密钥 ——
load_dotenv('/root/zhibai/.env')
ED25519_API_KEY  = os.getenv('YZ_ED25519_API_KEY')
ED25519_KEY_PATH = os.getenv('YZ_ED25519_KEY_PATH')
REST_API_KEY     = os.getenv('YZ_BINANCE_API_KEY')
REST_SECRET_KEY  = os.getenv('YZ_BINANCE_SECRET_KEY')

with open(ED25519_KEY_PATH, 'rb') as f:
    ed_priv = load_pem_private_key(f.read(), password=None)
    assert isinstance(ed_priv, Ed25519PrivateKey)
    LOG.info("✅ Ed25519 private key loaded")

# —— 基础配置 ——
SYMBOL     = 'ETH/USDC'
TF_CONFIG  = {'3m':'3m','15m':'15m','1h':'1h'}
TIMEFRAMES = list(TF_CONFIG.values())

exchange = ccxt.binance({
    'apiKey':    REST_API_KEY,
    'secret':    REST_SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType':    'future',
        'hedgeMode':      True,
        'recvWindow':     3000,
        'timeDifference': 0,
    },
})

# —— 超趋势加速 ——
@jit(nopython=True)
def numba_supertrend(h, l, c, per, mult):
    n = len(c)
    st = [0.0]*n; dirc=[False]*n
    hl2 = [(h[i]+l[i])*0.5 for i in range(n)]
    atr = [max(h[max(0,i-per+1):i+1]) - min(l[max(0,i-per+1):i+1]) for i in range(n)]
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
        # 初始化空 DF 并预置 ma 列
        cols = ['open','high','low','close','vol','ma7','ma25','ma99']
        self.klines = {tf: pd.DataFrame(columns=cols) for tf in TIMEFRAMES}
        self.price  = None
        self.lock   = asyncio.Lock()
        self._evt   = asyncio.Event()

    async def load_history(self):
        LOG.info("▶️ Loading history")
        async with self.lock:
            for alias,tf in TF_CONFIG.items():
                ohlcv = await exchange.fetch_ohlcv(SYMBOL, tf, limit=1000)
                df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
                df.set_index('ts',inplace=True)
                # 批量计算均线
                df['ma7']  = df['close'].rolling(7).mean()
                df['ma25'] = df['close'].rolling(25).mean()
                df['ma99'] = df['close'].rolling(99).mean()
                self.klines[tf] = df
        LOG.info("✔️ History loaded")

    async def update_kline(self, alias, ohlc):
        ts,o,h,l,c,v = ohlc
        tf = TF_CONFIG[alias]
        async with self.lock:
            df = self.klines[tf]
            # 增量计算：取最近 n-1 个 close 与新 c 拼接
            closes7  = list(df['close'].iloc[-6:])  + [c]
            closes25 = list(df['close'].iloc[-24:]) + [c]
            closes99 = list(df['close'].iloc[-98:]) + [c]
            ma7  = sum(closes7)/7
            ma25 = sum(closes25)/25
            ma99 = sum(closes99)/99
            new = {'open':o,'high':h,'low':l,'close':c,'vol':v,'ma7':ma7,'ma25':ma25,'ma99':ma99}
            df.loc[ts] = new
            if len(df)>1000: df.drop(df.index[0], inplace=True)
            self.klines[tf] = df
            self.price = self.price  # no-op but mark that price unchanged
            self._evt.set()
        LOG.debug(f"{tf}@{ts} updated")

    async def set_price(self, price):
        async with self.lock:
            self.price = price
            self._evt.set()

    async def wait_update(self):
        await self._evt.wait()
        self._evt.clear()

data_mgr = DataManager()

# —— 持仓 & OCO ——
class PositionTracker:
    def __init__(self):
        self.positions = {}
        self.lock      = asyncio.Lock()
        self._cid      = 1

    async def on_fill(self, side, qty, entry, sl, tp):
        async with self.lock:
            cid = self._cid; self._cid+=1
            self.positions[cid] = dict(side=side,qty=qty,sl=sl,tp=tp,active=True)
            LOG.info(f"[POS#{cid}] {side}@{entry:.4f} SL={sl:.4f} TP={tp:.4f}")
            opp = 'sell' if side=='buy' else 'buy'
            base = exchange.market_id(SYMBOL)
            # 下 STOP_MARKET
            slp = sl if side=='buy' else tp
            sl_ord = await exchange.fapiPrivatePostOrder({
                'symbol': base,
                'side':   opp.upper(),
                'type':   'STOP_MARKET',
                'stopPrice': slp,
                'closePosition': True,
                'workingType':'MARK_PRICE',
                'newClientOrderId':f"sl_{cid}"
            })
            # 下 TAKE_PROFIT_MARKET
            tpp = tp if side=='buy' else sl
            tp_ord = await exchange.fapiPrivatePostOrder({
                'symbol': base,
                'side':   opp.upper(),
                'type':   'TAKE_PROFIT_MARKET',
                'stopPrice': tpp,
                'closePosition': True,
                'workingType':'MARK_PRICE',
                'newClientOrderId':f"tp_{cid}"
            })
            self.positions[cid].update(sl_oid=sl_ord['orderId'], tp_oid=tp_ord['orderId'])

    async def on_order_update(self, msg):
        oid = msg['o']['i']; status = msg['o']['X']
        async with self.lock:
            for cid,p in self.positions.items():
                if not p['active']: continue
                if oid in (p['sl_oid'], p['tp_oid']) and status=='FILLED':
                    p['active'] = False
                    other = p['tp_oid'] if oid==p['sl_oid'] else p['sl_oid']
                    LOG.info(f"[POS#{cid}] OCO fill {oid}, cancel {other}")
                    await exchange.cancel_order(other, SYMBOL)

pos_tracker = PositionTracker()

# —— 策略基类 & 实例 ——
class BaseStrategy:
    priority = 1
    async def check(self, price): ...

class MainStrategy(BaseStrategy):
    priority = 1
    def __init__(self): self._ts=0
    async def check(self, price):
        if time.time()-self._ts<1: return
        df = data_mgr.klines['15m'];
        if len(df)<100: return
        ma7,ma25,ma99 = df[['ma7','ma25','ma99']].iloc[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        adx = ADXIndicator(df['high'],df['low'],df['close'],14).adx().iat[-1]
        if adx<=25: return
        bb = (df['close'].iat[-1]-df['close'].rolling(20).mean().iat[-1]+2*df['close'].rolling(20).std().iat[-1])\
             /(4*df['close'].rolling(20).std().iat[-1])
        if not (bb<=0 or bb>=1): return
        side = 'buy' if bb<=0 else 'sell'
        level = 0.005 if side=='buy' else -0.005
        qty, price0 = 0.016, price*(1+level)
        sl, tp = price*0.98 if side=='buy' else price*1.02, price*1.02 if side=='buy' else price*0.98
        LOG.info(f"▶️ Main → {side}@{price0:.4f}")
        o = await exchange.create_order(SYMBOL, 'limit', side, qty, price0, {'timeInForce':'GTC'})
        if o.get('status')=='closed': await pos_tracker.on_fill(side, qty, price0, sl, tp)
        self._ts=time.time()

class MACDStrategy(BaseStrategy):
    priority = 2
    def __init__(self): self.in_pos=False
    async def check(self, price):
        df = data_mgr.klines['15m']
        if len(df)<50: return
        ma7,ma25,ma99 = df[['ma7','ma25','ma99']].iloc[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        diff = MACD(df['close'],12,26,9).macd_diff()
        prev,curr = diff.iat[-2], diff.iat[-1]
        if prev>0>curr and not self.in_pos:
            qty, price0 = 0.015, price*1.005
            LOG.info(f"▶️ MACD SELL@{price0:.4f}")
            o = await exchange.create_order(SYMBOL,'limit','sell',qty,price0,{'timeInForce':'GTC'})
            if o.get('status')=='closed':
                await pos_tracker.on_fill('sell',qty,price0,price*1.03,price*0.97)
                self.in_pos=True
        if prev<0<curr and self.in_pos:
            qty, price0 = 0.015, price*0.995
            LOG.info(f"▶️ MACD BUY@{price0:.4f}")
            o = await exchange.create_order(SYMBOL,'limit','buy',qty,price0,{'timeInForce':'GTC'})
            if o.get('status')=='closed':
                await pos_tracker.on_fill('buy',qty,price0,price*0.97,price*1.03)
                self.in_pos=False

class TripleTrendStrategy(BaseStrategy):
    priority = 3
    def __init__(self): self._ts=0; self.active=False
    async def check(self, price):
        if time.time()-self._ts<1: return
        df = data_mgr.klines['15m']
        if len(df)<30: return
        ma7,ma25,ma99 = df[['ma7','ma25','ma99']].iloc[-1]
        if not (price<ma7<ma25<ma99 or price>ma7>ma25>ma99): return
        h,l,c = df['high'].values, df['low'].values, df['close'].values
        st,dirc = numba_supertrend(h,l,c,10,3)
        if dirc[-1] and not self.active:
            qty, price0 = 0.017, price*0.996
            LOG.info(f"▶️ Trend BUY@{price0:.4f}")
            o = await exchange.create_order(SYMBOL,'limit','buy',qty,price0,{'timeInForce':'GTC'})
            if o.get('status')=='closed': await pos_tracker.on_fill('buy',qty,price0,price*0.97,price*1.02)
            self.active=True
        if not dirc[-1] and not self.active:
            qty, price0 = 0.017, price*1.004
            LOG.info(f"▶️ Trend SELL@{price0:.4f}")
            o = await exchange.create_order(SYMBOL,'limit','sell',qty,price0,{'timeInForce':'GTC'})
            if o.get('status')=='closed': await pos_tracker.on_fill('sell',qty,price0,price*1.03,price*0.98)
            self.active=True
        self._ts=time.time()

strategies = sorted([MainStrategy(), MACDStrategy(), TripleTrendStrategy()],
                    key=lambda s: s.priority)

# —— 市场 WS ——
async def market_ws():
    streams = [f"{exchange.market_id(SYMBOL).lower()}@kline_{tf}" for tf in TF_CONFIG] + \
              [f"{exchange.market_id(SYMBOL).lower()}@markPrice"]
    uri = "wss://fstream.binance.com/stream?streams=" + "/".join(streams)
    retry = 0
    while True:
        try:
            LOG.info("🔗 market WS connecting")
            async with websockets.connect(uri, ping_interval=20) as ws:
                retry = 0
                async for msg in ws:
                    o = json.loads(msg)
                    st,d = o['stream'], o['data']
                    if st.endswith('markPrice'):
                        await data_mgr.set_price(float(d['p']))
                    else:
                        alias = st.split('@')[1].split('_')[1]
                        k = d['k']
                        await data_mgr.update_kline(alias, [
                            k['t'], float(k['o']), float(k['h']),
                            float(k['l']), float(k['c']), float(k['v'])
                        ])
        except Exception as e:
            LOG.warning(f"market_ws ▶️ {e}")
            await asyncio.sleep(min(2**retry,30)); retry+=1

# —— 用户 WS (Ed25519 登录) ——
async def user_ws():
    uri = "wss://ws-fapi.binance.com/ws-fapi/v1"  # 正式环境  [oai_citation:1‡binance-Rest.txt](file-service://file-HAcrt27JLMgJymPcB5GQtG)
    retry = 0
    while True:
        try:
            LOG.info("🔑 user WS connecting")
            async with websockets.connect(uri, ping_interval=20) as ws:
                # 构造 session.logon 请求
                ts = int(time.time()*1000)
                params = {
                    'apiKey':    ED25519_API_KEY,
                    'timestamp': ts
                }
                # 签名：对按字母序排的 param 字串进行 base64(Ed25519) 签名
                payload = '&'.join(f'{k}={params[k]}' for k in sorted(params))
                sig = base64.b64encode(ed_priv.sign(payload.encode())).decode()
                req = {
                    'id':       ts,
                    'method':   'session.logon',
                    'params':   {**params, 'signature': sig}
                }
                await ws.send(json.dumps(req))
                LOG.info("✅ session.logon sent")
                retry = 0
                async for msg in ws:
                    data = json.loads(msg)
                    # 仅处理 ORDER_TRADE_UPDATE 类型
                    if data.get('method')=='account.update':
                        for evt in data['params']['events']:
                            if evt['e']=='ORDER_TRADE_UPDATE':
                                await pos_tracker.on_order_update(evt)
        except Exception as e:
            LOG.warning(f"user_ws ▶️ {e}")
            await asyncio.sleep(5); retry+=1

# —— 维护 & 引擎 ——
async def maintenance():
    while True:
        await asyncio.sleep(60)
        try:
            await exchange.load_markets()
            LOG.debug("markets refreshed")
        except Exception as e:
            LOG.warning(f"maintenance ▶️ {e}")

async def engine():
    while True:
        await data_mgr.wait_update()
        price = data_mgr.price
        for strat in strategies:
            try: await strat.check(price)
            except Exception as e: LOG.error(f"{type(strat).__name__} ▶️ {e}")

# —— 启动 ——
async def main():
    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, lambda: asyncio.create_task(exchange.close()))
    await data_mgr.load_history()
    await asyncio.gather(market_ws(), user_ws(), maintenance(), engine())

if __name__=='__main__':
    asyncio.run(main())