import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler


def NewLogger(log_path: str, log_name: str) -> logging.Logger:
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(lineno)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    formatter.converter = beijing
    handler = TimedRotatingFileHandler(os.path.join(log_path, log_name + ".log"), when="D", interval=1, backupCount=7)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def beijing(sec):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


# 全局变量
# stra_logger = NewLogger("./logs", f"stra")
bt_logger = NewLogger("./logs", f"bt")
# sc_logger = NewLogger("./logs", f"sc")
