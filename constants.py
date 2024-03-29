# file source
UNKNOWN_FILE = ""
LOCAL_FILE = "LOCAL"
S3_FILE = "S3"

# file type
CSV_FILE = "csv"
LOG_FILE = "log"
PKL_FILE = "pkl"
FT_FILE = "ft"
PQ_FILE = "pq"
CACHE_FILE = PQ_FILE

# data path struct type
PST_NORMAL = "normal"
PST_D3QN = "d3qn"

# bucket name
BN_DEPTH = "depths"
BN_TICKER = "dpticker"
BN_TRADE = "dp-trade"
BN_INTERN = "dp4intern"

# data type scribe
DS_EMPTY = ""
DS_DEPTH = "depth"
DS_TICKERS = "tickers"
DS_TRADE = "trade"
DS_FEATURES = "features"
DS_MARKET_PRICE = "m_price"

# data record type
DRT_DATA = "data"
DRT_INDEX = "index"

# trade type scribe
SPOT_STR = "SPOT"
FUTURE_STR = "FUTURE"
SWAP_STR = "SWAP"
OPTION_STR = "OPTION"
U_BASED_STR = "U"
C_BASED_STR = "C"
COIN_BASED_STR = "COIN"

# trading date related
THIS_STR = "THIS"
NEXT_STR = "NEXT"
WEEK_STR = "WEEK"
MONTH_STR = "MONTH"
QUARTER_STR = "QUARTER"
FUTURE_NEXT_WEEK = "FUTURE_NEXT_WEEK"
FUTURE_THIS_WEEK = "FUTURE_THIS_WEEK"
FUTURE_NEXT_QUARTER = "FUTURE_NEXT_QUARTER"
FUTURE_THIS_QUARTER = "FUTURE_THIS_QUARTER"
DATE_SETTLEMENT = "settlement"
DATE_DELIVERY = "delivery"
DLV_THIS_WEEK = "THIS_WEEK"
DLV_NEXT_WEEK = "NEXT_WEEK"
DLV_THIS_MONTH = "THIS_MONTH"
DLV_NEXT_MONTH = "NEXT_MONTH"
DLV_THIS_QUARTER = "THIS_QUARTER"
DLV_NEXT_QUARTER = "NEXT_QUARTER"

# trade action str
TA_EMPTY = ""
TA_BUY = "BUY"
TA_SELL = "SELL"
TA_CANCEL = "CANCEL"
TA_CANCEL_ALL = "CANCEL_ALL"

# pre-processing mode
PPM_TRANSFER = "trans_save"
PPM_FULL_DEPTH = "recover_depth"
PPM_FEATURING = "extract_features"

# config names
ENV_CONFIG = "env_config"
ACC_CONFIG = "acc_config"
DATA_CONFIG = "data_config"

# schedule mode
SM_SIMULATING = "simulating"
SM_PRE_PROCESSING = "pre_processing"

# order status
OT_NORMAL = "normal"
OT_FILLED = "filled"
OT_PARTLY_FILLED = "partly_filled"
OT_EXPIRED = "expired"
OT_CANCELED = "canceled"
OT_OPENED = "opened"
OT_TERMINATED = [OT_FILLED, OT_PARTLY_FILLED, OT_EXPIRED, OT_CANCELED]
OT_LIVING = [OT_OPENED, OT_NORMAL]

# time statics
RTA_EMPTY = ""
RTA_INIT = "init"
RTA_SHOW = "show"
RTA_load = "load"
RTA_loc = "loc"
RTA_loc1 = "loc1"
RTA_loc2 = "loc2"
RTA_get = "get"
RTA_get1 = "get1"
RTA_get2 = "get2"
RTA_recover = "recover"
RTA_mat = "mat"
RTA_others = "others"
RTA_register = "register"
RTA_merge_append = "append"
RTA_merge_concat = "concat"
RTA_strategy = "strategy"

TYPE_DICT = type({})

DEFAULT_SPAN = 100          # 默认窗口宽度，100毫秒
DEFAULT_SIZE = 10 * 60 * 60     # 默认数据集大小，为通常的一小时depth数据量（约每100毫秒一条）
DEFAULT_BLOCK_SIZE = 5   # 默认考虑启用二分查找的数据量
DEFAULT_RATIO = DEFAULT_SIZE / DEFAULT_SPAN   # 默认数据量/窗口宽度比
DEFAULT_LEVEL = 20
DEFAULT_PACKS = 3
DEFAULT_CLOCK = 0
ZERO_TH = 1e-10
ERROR_LEVEL = 30
MAX_PORTION = 0.3
MIN_TICKS = 5
RISK_WND = 600

# pair key options
FOR_DATA = "data"
FOR_STAT = "stat"
FOR_INV = "inv"

# operation signal
TOS_WAIT = ""
TOS_CANCEL = "cancel"
TOS_DEFENCE = "away"
TOS_ATTACK = "closer"
SOS_ERROR = 0
SOS_NORMAL = 1
SOS_HALT = 2
SOS_STOP = 999

# capital name
DEFAULT_COIN = "USDT"

# pricing mode
PRM_NORMAL = "normal"
PRM_AS = "AS"
PRM_DEPTH = "depth"
