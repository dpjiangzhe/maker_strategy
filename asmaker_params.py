import numpy as np
import matplotlib.pyplot as plt
from stgs.logger import bt_logger


def estimate_sigma_from_delta(p_deltas, t_deltas):
    try:
        t_sqrt_deltas = np.sqrt(t_deltas)
        # zeros = t_sqrt_deltas[t_sqrt_deltas == 0]
        # t_sqrt_deltas = np.delete(t_sqrt_deltas, zeros)
        # p_deltas = np.delete(p_deltas, zeros)
        p_deltas_m = p_deltas/t_sqrt_deltas
        # nans = np.isnan(p_deltas_m)
        # p_deltas_m = np.delete(p_deltas_m, np.where(nans))
        return np.std(p_deltas_m)
    except Exception as e:
        bt_logger.info(f'###### ERROR: {e}')
    return 0.0


def estimate_sigma(ps, ts):
    ps = np.array(ps)
    ts = np.array(ts)
    ps_l = ps[:-1]
    ps_c = ps[1:]
    p_deltas = ps_c-ps_l
    ts_l = ts[:-1]
    ts_c = ts[1:]
    t_deltas = ts_c-ts_l
    return estimate_sigma_from_delta(p_deltas, t_deltas)


############################################################################################
class AlphaEstimator:
    def __init__(self, quantities, group_num):
        self.quantities = np.array(quantities)
        self.group_num = group_num

    def outlier_recognition(self):
        Q1 = np.percentile(self.quantities, 25)
        Q3 = np.percentile(self.quantities, 75)
        IQR = Q3-Q1
        return np.logical_or(self.quantities <= Q1-1.5*IQR, self.quantities >= Q3+1.5*IQR)

    def outlier_cleaning(self):
        idx = np.logical_not(self.outlier_recognition())
        data_cleaned = self.quantities[idx]
        return data_cleaned

    def data_grouping(self):
        data_cleaned = self.outlier_cleaning()
        data_info = plt.hist(data_cleaned, bins=self.group_num)
        freq = data_info[0]
        xs = [(data_info[1][i]+data_info[1][i+1])/2 for i in range(data_info[1].shape[0]-1)]
        return xs, freq

    def fit(self):
        xs, freq = self.data_grouping()
        alpha = estimate_alpha(xs, freq)[0]
        return alpha


def estimate_alpha(xs, freq):
    zs = np.log(xs).reshape(1, -1)
    ys = np.log(freq).reshape(1, -1)
    zs0 = np.ones_like(zs).reshape(1, -1)
    zs_ext = np.vstack([zs, zs0])
    b = np.linalg.lstsq(zs_ext.T, ys.T)[0][0]
    alpha = -1-b
    return alpha


############################################################################################
class OrderHitCounter(object):

    def __init__(self, price=0.0, tick=0.01, grids=10):
        # 0位无用
        self.base_price = price
        self.tick = tick
        self.grids = grids + 1
        self.counter = self.reset(self.base_price, self.tick, self.grids)

    def count(self, price, is_buy):
        spread = abs(price - self.base_price)
        index = 0
        if (is_buy and price > self.base_price) or (not is_buy and price < self.base_price):
            index = int(spread / self.tick)

        if index >= self.grids:
            index = self.grids - 1

        # index=0: 所有与中间价的价差小于1 tick的订单统一计数，相当于未命中
        # index>=grids：所有与中间价的价差过大的订单统一计数
        for i in range(1, index+1):
            self.counter[i] += 1

    def calc_k(self):
        total_value = 0.0
        counter = 0
        k = 0.0
        for i in range(1, self.grids):
            for j in range(i+1, self.grids+1):
                if self.counter[i] > 0 and self.counter[j] > 0:
                    total_value += (np.log(self.counter[j] * 1.0 / self.counter[i])) / ((i - j) * self.tick)
                    counter += 1
        if counter > 0:
            # todo: 确定系数调整方案
            k = total_value / counter
        return k

    def set_price(self, price):
        if price > 0:
            self.base_price = price

    def reset(self, price=0.0, tick=0.0, grids=0):
        if price > 0:
            self.base_price = price
        if tick > 0:
            self.tick = tick
        if grids > 0:
            self.grids = grids
        self.counter = [0] * (self.grids + 1)
        return self.counter


def generate_delta_p(depth, mid_price, trade_ratio, grids):
    # 对于一条depth数据（无论ask还是bid），计算其可能的吃单的量、价分布
    total = 0.0
    for item in depth:
        total += item['s']
    max_q = total * trade_ratio
    Qs = max_q * (np.arange(1, grids+1) / grids)
    pQs = [get_pQ_from_depth(depth, Q) for Q in Qs]
    for i, item in enumerate(pQs):
        if item > 0:
            pQs[i] = abs(item - mid_price)
    # pQs = np.array(pQs)
    return Qs, pQs


def get_pQ_from_depth(depth, Q):
    q = 0
    target = 0
    for i, item in enumerate(depth):
        q = q+item['s']
        if q >= Q:
            target = i
            break
    if q < Q:
        return -1
    else:
        return depth[target]['p']


def calc_K(Qs, delta_ps):
    Qs = np.array(Qs)
    delta_ps = np.array(delta_ps)
    lnQs = np.log(Qs).reshape(-1, 1)
    delta_ps = delta_ps.reshape(-1, 1)
    return np.linalg.lstsq(delta_ps, lnQs, rcond=None)[0][0][0]


def generate_delta_p_from_trade(trade, mid_price, grids):
    pass


"""
def estimate_K(depths, trade_ratio, grids):
    Qs = []
    delta_ps = []
    for i, depth_data in enumerate(depths):
        depth_ask = depth_data.get('asks')
        depth_bid = depth_data.get('bids')
        if depth_ask is None or len(depth_ask) == 0 or depth_bid is None or len(depth_bid) == 0:
            continue

        mid_price = (depth_ask[0]['p'] + depth_bid[0]['p']) / 2

        _new_Qs, _new_pQs = generate_delta_p(depth_ask, mid_price, trade_ratio, grids)
        Qs += list(_new_Qs)
        delta_ps += list(_new_pQs)

        _new_Qs, _new_pQs = generate_delta_p(depth_bid, mid_price, trade_ratio, grids)
        Qs += list(_new_Qs)
        delta_ps += list(_new_pQs)
    return calc_K(Qs, delta_ps)
"""


def estimate_K(trade_q, price_delta):
    return calc_K(trade_q, price_delta)
