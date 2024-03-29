import json
import tomli
import numpy as np
import os
import sys

# from Strategy import StrategyInterface
# import AccountManager as AMr
# import DataManager as DMr
# from utils.trading_date import get_delivery_type
from stgs.logger import bt_logger

from stgs import data_struct as DataType
from stgs.constants import *
from stgs.asmaker_params import *

sys.path.append("../")
from importlib.metadata import version

if version('protobuf') > '4.':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".", "proto/pypb4"))
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".", "proto/py"))

from order.order_pb2 import TradeSide

backtest_conf_path = "conf/backtest.toml"
asmaker_conf_path = "stgs/asm_conf.toml"

'''
class AsMakerStrategy(object):
    pub_account = None
    schedule_config = None
    token_keys = []
    pair_keys = []
    clock = 0

    def __init__(self, account_data: AMr.AccountData):
        super(AsMakerStrategy, self).__init__()
        self.set_account(account_data)

        self.maker = AsMaker()
        self.maker.set_account(account_data)

        self.load_config()
        # self.get_trade_config()

    def load_config(self):
        fp = open(backtest_conf_path, 'rb')
        self.schedule_config = tomli.load(fp, parse_float=float)
        fp.close()

    def set_account(self, account_data: AMr.AccountData):
        if account_data is None:
            return
        self.pub_account = account_data
        self.token_keys = account_data.token_stat_info.keys()
        self.pair_keys = account_data.stat_info.keys()

    def processing(self, data_records: {} = None, order_list: {} = None, ts=0):
        self.clock = ts
        # order_list = []

        for drk, _record in data_records.items():
            # 获取交易品种（特定交易所的交易对的特定交易品种，如现货、期货。。。）相关数据
            if _record is None or len(_record.data_record_list) == 0:
                continue
            # 使用数据
            _data = _record.get_data(DS_MARKET_PRICE)
            if _data is not None and len(_data) > 0:
                self.update_market_price(_data)
            if 'ap' not in _data or _data['ap'] == 0 or 'bp' not in _data or _data['bp'] == 0:
                _mid_price = 0
            else:
                _mid_price = (_data['ap'] + _data['bp']) / 2
            for ds, _data in _record.data_record_list.items():
                # 针对所需数据源（如tickers，depth。。。）
                if _data is None or len(_data) == 0 or ds == DS_MARKET_PRICE:
                    continue
                # self.logger.info(f'=== STG processing ==> {drk}|{ds}|{_data}')
                if ds == DS_DEPTH:
                    self.update_depth(_data)
                elif ds == DS_TRADE:
                    self.update_trade(_data, _mid_price)
                pass
            pass

        self.receiving_result(order_list, ts)
        new_order_list = self.maker.generate_actions(ts)

        if new_order_list is None or len(new_order_list) == 0:
            new_order_list = None
        return new_order_list

    def receiving_result(self, result, ts=0):
        if result is None or len(result) == 0:
            return

        self.maker.receiving_results(result["taker"], ts)
        self.maker.receiving_results(result["maker_buy"], ts)
        self.maker.receiving_results(result["maker_sell"], ts)
        pass

    def finished(self, ts: int, history_path: str):
        self.maker.finished(ts, history_path)
        pass

    def update_depth(self, depth_data):
        # todo: 滚动更新保存的depth数据，用于计算K；滚动窗口宽度为超参数
        self.maker.update_depth(depth_data)
        pass

    def update_trade(self, trade_data, mid_price):
        # todo：滚动更新保存的trade数据，用于估算alpha；滚动窗口宽度为超参数
        self.maker.update_trade(trade_data, mid_price)
        pass

    def update_market_price(self, m_price_data):
        # todo: 滚动更新保存的市场中间价数据，用于统计中间价的波动率；滚动窗口宽度为超参数
        self.maker.update_market_price(m_price_data)
        pass
'''


###########################################################################################################
class AsMakerConfig(object):
    actionWnd = 10 * 1000  # 交易窗口宽度，毫秒
    lifetime = 1000  # 订单生存周期，毫秒
    portion = 0.01  # 每笔订单交易量
    holding_limit = 1.0  # 持仓上限

    tradeWnd = actionWnd * 10 // 100  # trade数据观察量
    priceWnd = actionWnd * 2 // 100  # 盘口价格波动观察窗口宽度，每百毫秒1个采样点
    depthWnd = actionWnd // 100  # 成交影响力观察窗口宽度，每百毫秒1个采样点
    trade_ratio = 0.6  # 参考成交总量
    trade_grid = 10  # 参考成交档位
    data_groups = 10  # 数据分组数，用于估算alpha时的数据分组

    def __init__(self, aw=10 * 1000, life=100, tw=200, pw=200, dw=100, tr=0.6, tg=10, dg=30):
        self.set(aw, life, tw, pw, dw, tr, tg, dg)

    def set(self, aw=10 * 1000, life=100, tw=200, pw=200, dw=100, tr=0.6, tg=10, dg=30):
        self.actionWnd = aw
        self.lifetime = life
        self.tradeWnd = tw
        self.priceWnd = pw
        self.depthWnd = dw
        self.trade_ratio = tr
        self.trade_grid = tg
        self.data_groups = dg


class AsMakerParams(object):
    tick = 0.1
    gamma = 0.1
    alpha = 0.0
    sigma = 0.0
    k = 0.0

    def __str__(self):
        return f'AsMaker: tick={self.tick}, gamma={self.gamma}, alpha={self.alpha}, sigma={self.sigma}, k={self.k}'


class MakerOptions(object):
    normal = True  # true：简单盘口挂单；false：maker模型挂单
    maker_mode = 1  # 1: AS; 2: depth档位
    depth = 3    # depth参考档位：应从统计数据中得到，作为配置值，考虑比统计结果调低一档
    is_test = True  # true：测试模式，不实际发单；false：实际运行模式
    only_once = False  # only_once：只发1单（通常用于线上功能实测）
    trader_limit = 2  # 同时发单上限（单侧：买/卖）

    def __str__(self):
        return f'MakerOptions: normal? {self.normal}（{self.maker_mode} with {self.depth}）, is_test? {self.is_test}, only_once? {self.only_once}, trader limit: {self.trader_limit}(buy/sell each))'


# AsMaker使用方法：
# 1 初始化和setup；2 更新数据：update_depth/trade/mid_price；3 （衡量时机）产生交易信号generate_actions
class AsMaker(object):
    name = "AsMaker"
    normal_name = "ticker_base"
    model_name = "AS_base"

    clock = 0
    maker_config: AsMakerConfig = None
    maker_params: AsMakerParams = None
    maker_options: MakerOptions = None

    # trade_config: AMr.TradePairInfo = None

    def __init__(self, config_path="", account_id="", co_pairs=1, logger=None):
        self.account_id = account_id
        self.maker_config = AsMakerConfig()
        self.maker_params = AsMakerParams()
        self.maker_options = MakerOptions()
        # self.trade_config = AMr.TradePairInfo()
        self.co_pairs = co_pairs

        self.depth_data = []
        self.depth_counter = 0

        self.trade_data = []
        self.price_delta = []
        self.hit_counter = OrderHitCounter()
        self.hit_records = []
        self.trade_counter = 0

        self.market_price = []
        self.mid_price = []
        self.price_counter = 0

        self.ts_list = []
        self.order_list = {"buy": [], "sell": []}

        # todo: 有待配置化，通过配置文件设置
        self.base_token = "BTC"
        self.quote_token = "USDT"
        self.symbol = self.base_token + "_" + self.quote_token
        self.base_amount = self.quote_amount = self.base_origin = self.quote_origin = 0.0
        self.base_limit = self.quote_limit = 0.0
        self.holding_rate = 0.1
        self.lever_rate = 1
        self.self_balance = False
        self.uni_account = False
        self.symbol_type = "SPOT_NORMAL"
        self.market_info = DataType.MarketData({"chain": "", "exchange": "binance", "market": "spot"})

        self.account = None
        # self.normal = True
        # self.is_test = True
        # self.only_once = True
        self.period_start = 0.0

        self.setup(config_path)
        if logger is not None:
            self.logger = logger
        else:
            self.logger = bt_logger

    def get_name(self, with_model=False):
        _name = self.name
        if with_model:
            if self.maker_options.normal:
                _name += "_" + self.normal_name
            else:
                _name += "_" + self.model_name
        return _name

    def setup(self, config_path=""):
        # todo: 设置重要配置及超参数，数据来源待定，可能是配置文件；如无则使用默认值
        _config_path = config_path
        if _config_path == "":
            _config_path = asmaker_conf_path
        with open(_config_path, 'rb') as fp:
            _config = tomli.load(fp, parse_float=float)
            if "script" in _config.keys():
                script_config = _config["script"]
                if "name" in script_config.keys():
                    self.name = script_config["name"]
                if "normal_name" in script_config.keys():
                    self.normal_name = script_config["normal_name"]
                if "model_name" in script_config.keys():
                    self.model_name = script_config["model_name"]
            if "config" in _config.keys():
                for key, value in _config["config"].items():
                    setattr(self.maker_config, key, value)
            if "params" in _config.keys():
                for key, value in _config["params"].items():
                    setattr(self.maker_params, key, value)
            if "options" in _config.keys():
                for key, value in _config["options"].items():
                    setattr(self.maker_options, key, value)
            if "pair_info" in _config.keys():
                pair_config = _config["pair_info"]
                if "symbol_type" in pair_config.keys():
                    self.symbol_type = pair_config["symbol_type"].upper()
                    if SPOT_STR in self.symbol_type:
                        self.symbol_type = "SPOT_NORMAL"
                if "base_c" in pair_config.keys():
                    self.base_token = pair_config["base_c"].upper()
                if "base_q" in pair_config.keys():
                    self.base_amount = self.base_origin = pair_config["base_q"]
                if "base_limit" in pair_config.keys():
                    self.base_limit = pair_config["base_limit"]
                if "quote_c" in pair_config.keys():
                    self.quote_token = pair_config["quote_c"].upper()
                if "quote_q" in pair_config.keys():
                    self.quote_amount = self.quote_origin = pair_config["quote_q"]
                if "quote_limit" in pair_config.keys():
                    self.quote_limit = pair_config["quote_limit"]
                else:
                    self.quote_limit = self.quote_amount
                if "holding_rate" in pair_config.keys():
                    self.holding_rate = pair_config["holding_rate"]
                if "lever_rate" in pair_config.keys():
                    self.lever_rate = pair_config["lever_rate"]
                self.symbol = self.base_token + "_" + self.quote_token
            if "market_info" in _config.keys():
                for key, value in _config["market_info"].items():
                    setattr(self.market_info, key, value.upper())

        self.hit_counter.reset(tick=self.maker_params.tick, grids=self.maker_config.trade_grid)

    def set_logger(self, logger=None):
        if logger is not None:
            self.logger = logger

    def update_depth(self, depth):
        # todo: 滚动更新保存的depth数据，用于计算K；滚动窗口宽度为超参数
        self.depth_data.append(depth)
        self.depth_counter += 1
        if len(self.depth_data) > self.maker_config.depthWnd:
            del self.depth_data[0]
            if self.depth_counter >= len(self.depth_data):
                # todo: 更新计算K
                # estimate_K(self.depth_data, self.maker_config.trade_ratio, self.maker_config.trade_grid)
                self.depth_counter = 0

        # self.logger.info(f'received new depth: {depth}')
        pass

    def update_trade(self, trade_list, mid_price=0.0):
        # todo：滚动更新保存的trade数据（只关心各单成交量），用于估算k；滚动窗口宽度为超参数
        update_size = len(trade_list)
        if mid_price == 0:
            mid_price = self.mid_price[-1]

        # self.logger.info(f'new trade data arrived: {len(trade_list)}/{mid_price}')
        for index, trade in enumerate(trade_list):
            _mid_price = trade.mark_price
            # self.logger.info(f'\t\ttrade data: {_mid_price} {trade.amount}/{index} {self.hit_counter.counter}')
            if _mid_price <= 0:
                _mid_price = mid_price
            self.hit_counter.set_price(_mid_price)
            self.price_delta.append(abs(trade.price - _mid_price))
            self.trade_data.append(trade.amount)
            self.hit_counter.count(trade.price, TA_BUY in TradeSide.Name(trade.taker_side))
        self.trade_counter += update_size
        # self.logger.info(f'ready to calc k: {update_size} {len(self.trade_data)}/{self.maker_config.tradeWnd}')
        if len(self.trade_data) > self.maker_config.tradeWnd:
            del self.trade_data[0:update_size]
            del self.price_delta[0:update_size]
            # self.logger.info(f'before calc k: {self.hit_counter.counter}')
            if self.trade_counter >= len(self.trade_data):
                # todo: 更新计算alpha
                # self.maker_params.alpha = AlphaEstimator(self.trade_data, self.maker_config.data_groups).fit()
                # todo: 更新计算k
                # self.maker_params.K = estimate_K(self.trade_data, self.price_delta)
                self.maker_params.k = self.hit_counter.calc_k()
                self.hit_records.append(self.hit_counter.counter.copy())
                self.hit_records[-1].append(self.maker_params.k)

                self.trade_counter = 0
                self.hit_counter.reset()
        self.logger.info(f'received new trade: {len(trade_list)}/{self.trade_counter}/{self.maker_config.tradeWnd}, at {mid_price}; k={self.maker_params.k}')

    def update_market_price(self, m_price_data):
        # todo: 滚动更新保存的市场中间价数据，用于统计中间价的波动率；滚动窗口宽度为超参数
        if len(self.ts_list) > 0 and abs(self.ts_list[-1] - m_price_data['time'] / 1000) <= (ZERO_TH * 10000):
            self.ts_list[-1] = m_price_data['time'] / 1000  # 以秒计？
            self.market_price[-1] = m_price_data
            self.mid_price[-1] = (m_price_data['ap'] + m_price_data['bp']) / 2
        else:
            self.market_price.append(m_price_data)
            self.ts_list.append(m_price_data['time'] * 1.0 / 1000)  # 以秒计？
            self.mid_price.append((m_price_data['ap'] + m_price_data['bp']) / 2)
        self.price_counter += 1
        if len(self.mid_price) > self.maker_config.priceWnd:
            del self.mid_price[0]
            del self.ts_list[0]
            # self.logger.info(
            #    f'ready to estimate sigma: {self.ts_list[0]} {self.price_counter}/{len(self.mid_price)}/{len(self.market_price)}')
            if self.price_counter >= len(self.mid_price):
                # todo: 更新计算sigma
                self.maker_params.sigma = estimate_sigma(self.mid_price, self.ts_list)
                # for test
                if self.maker_params.sigma == 0:
                    self.maker_params.sigma = 1.0

                self.price_counter = 0
        # self.logger.info(f'received new market price: {m_price_data}; sigma={self.maker_params.sigma}')
        pass

    def get_ref_price(self, token_amount=0.0, mode=PRM_AS):
        # print(f'get ref price: {len(self.market_price)}')
        if len(self.market_price) == 0:
            return DataType.PriceInfo(name=mode)

        # todo: 计算参考交易价格
        ap = bp = 0.0

        # todo: account数据，考虑通过参数传入
        # q = self.account.account.capital_list[self.base_token].inv - self.account.account_origin.capital_list[self.base_token].inv
        self.base_amount = token_amount
        q = np.max([0, token_amount - self.base_origin])

        sigma = self.maker_params.sigma
        gamma = self.maker_params.gamma

        # print(self.maker_params.alpha, type(self.maker_params.alpha))

        if np.isnan(self.maker_params.alpha) or self.maker_params.alpha == 0:
            self.maker_params.alpha = 1.0

        # k = self.maker_params.alpha * self.maker_params.K
        k = self.maker_params.k
        # print(self.maker_params)
        if k == 0:
            k = 1.5

        _T = self.maker_config.actionWnd
        if self.period_start == 0:
            self.period_start = self.clock
        t = self.clock - self.period_start
        if t >= _T:
            t = _T
            self.period_start = 0.0

        if mode == PRM_AS and k == 0:
            return DataType.PriceInfo(name=mode)

        # todo: 修正价格，当 r 偏离盘口过大，可能导致报价不能形成有效挂单时，考虑调整报价
        if mode == PRM_AS:
            # AS 模型
            r = self.mid_price[-1] - q * gamma * sigma ** 2 * ((_T - t) / _T)
            # r = self.mid_price[-1] * (1 - q * gamma * sigma ** 2)
            r_spread = 2 / gamma * np.log(1 + gamma / k)
            ap = r + r_spread / 2
            bp = r - r_spread / 2
        elif mode == PRM_DEPTH:
            # 深度档位
            _depth_list = self.depth_data[-1]["asks"]
            _depth_pos = max(int(self.maker_options.depth) - 1, 0)
            if _depth_pos >= len(_depth_list):
                _depth = len(_depth_list) - 1
            ap = _depth_list[_depth_pos]["price"]
            _depth_list = self.depth_data[-1]["bids"]
            _depth_pos = max(int(self.maker_options.depth) - 1, 0)
            if _depth_pos >= len(_depth_list):
                _depth = len(_depth_list) - 1
            bp = _depth_list[_depth_pos]["price"]
            r = (ap + bp) / 2
            r_spread = (ap - bp) / 2
            pass
        else:
            # 盘口价格
            _m_price = self.market_price[-1]
            r = self.mid_price[-1]
            r_spread = (_m_price["ap"] - _m_price["bp"]) / 2
            ap = _m_price["ap"]
            bp = _m_price["bp"]
            pass
        return DataType.PriceInfo(ap, bp, r, r_spread, mode)


if __name__ == "__main__":
    maker_config_path = "./trade_conf/okex/asm_conf_eth_usdt_usf.toml"
    account_id = "test"
    as_maker = AMr.AsMaker(maker_config_path, account_id)


