import time
from random import random

from stgs.constants import *


# 市场信息
class MarketData(object):
    chain = ""
    exchange = "xxxx"
    market = "SPOT"

    def __init__(self, config=None):
        self.set_config(config)

    def set_config(self, config=None):
        if config is None:
            return
        for key, value in config.items():
            # assert getattr(self, key) is not None
            setattr(self, key, value)

    def get_config(self):
        return self.__dict__


# 币种信息
class TokenInfo(object):
    token_name: str
    inv: float
    value: float

    def __init__(self, token_name="", inv=0.0):
        self.token_name = token_name.upper()
        self.inv = inv
        self.value = inv

    def get_config(self):
        return self.__dict__


class OrderInfo(object):
    id = ""
    chain = ""
    exchange = "binance"
    market = "SPOT"
    base_c = "ETH"
    quote_c = "USDT"

    trade_type = ""
    contract = ""
    trade_act = TA_EMPTY
    is_taker = False
    amount = 0.0
    price = 0.0
    filled = 0.0
    filled_price = 0.0
    fee_amount = 0.0
    fee = 0.0

    last_filled = 0.0
    last_price = 0.0
    price_ref_bid = price_ref_ask = 0.0

    life_time = 0  # 毫秒计

    status = OT_NORMAL
    upd_ts = 0

    def __init__(self, order_id="", chain="", exchange="", market="", base_c="", quote_c="", symbol_type="", contract="", trade_action="", taker=False,
                 amount=0.0, price=0.0, life_time=0, ts=0):
        if order_id != "":
            self.id = order_id
        else:
            _ts = time.time_ns() + int(random() * 1000)
            self.id = str(_ts)

        if chain != "":
            self.chain = chain
        if exchange != "":
            self.exchange = exchange
        if market != "":
            self.market = market
        if base_c != "":
            self.base_c = base_c
        if quote_c != "":
            self.quote_c = quote_c
        if symbol_type != "":
            # 采用symbol_type的格式
            self.trade_type = symbol_type
        self.contract = contract
        if trade_action != "":
            self.trade_act = trade_action
        self.is_taker = taker
        if amount > 0:
            self.amount = amount
        if price > 0:
            self.price = price
        if life_time > 0:
            # 订单的生命周期
            self.life_time = life_time
        if ts > 0:
            self.upd_ts = ts
        pass


class PriceInfo(object):
    model_name = PRM_NORMAL
    ap = 0.0
    bp = 0.0
    p = 0.0
    s = 0.0

    def __init__(self, ap=0.0, bp=0.0, p=0.0, s=0.0, name=PRM_NORMAL):
        self.ap = ap
        self.bp = bp
        self.p = p
        self.s = s
        self.model_name = name

    def value(self):
        return self.ap, self.bp, self.p, self.s

    def __repr__(self):
        return f'{self.__dict__}'

