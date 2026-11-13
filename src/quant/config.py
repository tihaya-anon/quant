"""全局配置：路径、样本区间、股票池、交易成本。"""

from pathlib import Path

# 项目根目录（src/quant/config.py -> 上三级）
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # baostock 原始数据缓存（每只证券一个文件）
PROCESSED_DIR = DATA_DIR / "processed"  # 宽表面板数据
RESULTS_DIR = DATA_DIR / "results"  # 策略回测结果，供 notebook 读取

START_DATE = "2016-01-01"
END_DATE = "2026-11-12"
# 回测起点：留出一年做指标预热（如 12 个月动量、250 日波动率）
BACKTEST_START = "2017-01-01"

# 股票池：上证50 当前成分股（注意：用"当前"成分股回测历史存在幸存者偏差）
UNIVERSE = "sz50"  # 可选 "sz50" / "hs300"
BENCHMARK = "sh.000016"  # 上证50指数
INDICES = {"sh.000016": "上证50", "sh.000300": "沪深300"}

# 交易成本（单边）：佣金+印花税近似 0.1%，滑点 0.1%
FEES = 0.001
SLIPPAGE = 0.001
INIT_CASH = 1_000_000

# 并行下载进程数（baostock 单连接很慢，每个进程独立登录）
DOWNLOAD_WORKERS = 8
