# 各页面第一个代码块执行：%run _setup.py
import warnings

import pandas as pd
from IPython.display import display

from quant import viz
from quant.viz import load, show_stats, plot_equity, plot_grid, plot_yearly, plot_lines, show, compare

warnings.filterwarnings("ignore")
COLS = ["年化收益", "年化波动", "夏普比率", "最大回撤", "月均单边换手"]
import plotly.graph_objects as go


def P(x: float, d: int = 1) -> str:
    """百分比格式，供正文内联 `{python} P(x)` 使用。"""
    return f"{x * 100:.{d}f}%"


def F(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"
