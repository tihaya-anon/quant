"""全局配置：路径、样本区间、股票池、交易成本。"""

import os
from pathlib import Path

# 项目根目录（src/quant/config.py -> 上三级）
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # baostock 原始数据缓存（每只证券一个文件）
PROCESSED_DIR = DATA_DIR / "processed"  # 宽表面板数据
RESULTS_DIR = DATA_DIR / "results"  # 策略回测结果（按股票池分子目录，见文件末尾），供 notebook 读取

START_DATE = "2016-01-01"
END_DATE = os.environ.get("QUANT_END", "2026-11-12")  # 本地数据包缺 2024-2025 时可设 QUANT_END=2023-12-31
# 回测起点：留出一年做指标预热（如 12 个月动量、250 日波动率）
BACKTEST_START = "2017-01-01"

# 股票池：
#   "sz50"       上证50 **当前**成分股（旧版默认，存在幸存者偏差，仅保留作对照）
#   "sz50_pit"   上证50 历史成分股（point-in-time）
#   "hs300_pit"  沪深300 历史成分股（默认）
#   "zz800_pit"  沪深300 + 中证500 历史成分股
#   "cap300"     本地数据包：每月末按当时流通市值取前 300（近似沪深300，默认，不依赖 baostock）
#   "cap800"     本地数据包：每月末按当时流通市值取前 800（近似中证800）
# cap*：由 quant.local_data 从购买的日K数据包构建，见该模块说明。
# *_pit：每月末查询一次当时的指数成分，下载所有曾经入选的股票（含已退市），
# 回测时某只股票只有在"当时是成分股"的日子才可被选入，消除幸存者偏差。
UNIVERSE = os.environ.get("QUANT_UNIVERSE", "cap300")  # 也可用环境变量切换
INDICES = {"sh.000016": "上证50", "sh.000300": "沪深300", "sh.000905": "中证500"}
BENCHMARK = {"sz50": "sh.000016", "sz50_pit": "sh.000016", "hs300_pit": "sh.000300", "zz800_pit": "sh.000300",
             "cap300": "sh.000300", "cap800": "sh.000300"}[UNIVERSE]

# 交易成本（单边）：佣金+印花税近似 0.1%，滑点 0.1%
FEES = 0.001
SLIPPAGE = 0.001
INIT_CASH = 1_000_000

# 并行下载进程数（baostock 单连接很慢，每个进程独立登录）
DOWNLOAD_WORKERS = 1  # baostock 对高频/高并发请求会封禁 IP（"黑名单用户"），宁慢勿快
REQUEST_INTERVAL = 0.5  # 每次请求后的休眠秒数
BENCHMARK_NAME = INDICES[BENCHMARK] + "指数"
RESULTS_DIR = RESULTS_DIR / UNIVERSE
