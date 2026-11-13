"""均值回归策略（布林带 / Z-score 择时）。

思路：价格短期偏离均值过多后倾向于回归。
- Z = (收盘价 - N 日均线) / N 日标准差
- Z < -k（跌破布林下轨）时买入；Z 回到 0 以上（回到均线）时卖出；另设 10% 止损防止"接飞刀"
- 每只股票独立回测、各分配等额资金，再汇总成组合；并扫描 (N, k) 参数网格
"""

from __future__ import annotations

import pandas as pd

from quant import backtest as bt
from quant import config

STOP_LOSS = 0.10


def zscore(close: pd.DataFrame, window: int) -> pd.DataFrame:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    return (close - ma) / sd


def signals(close: pd.DataFrame, window: int, k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore(close, window)
    entries = (z < -k) & (z.shift(1) >= -k)  # 下穿下轨当日
    exits = z > 0
    return entries, exits


def _run_one(panel: dict, close, open_, window: int, k: float):
    entries, exits = signals(panel["close"], window, k)
    return bt.backtest_signals(close, open_, entries, exits, panel["tradable"], sl_stop=STOP_LOSS)


def _combined_value(pf) -> pd.Series:
    """各股票独立资金 -> 汇总净值，归一化到初始资金。"""
    v = pf.value().sum(axis=1)
    return v / v.iloc[0] * config.INIT_CASH


def run(panel: dict, window: int = 20, k: float = 2.0, example: str = "sh.600519") -> dict:
    close, open_ = bt._exec_prices(panel)
    # 只回测在回测起点前已上市的股票，避免上市前资金闲置拉低组合收益
    listed = panel["close"].loc[:config.BACKTEST_START].iloc[-1].notna()
    cols = listed[listed].index
    sub = {**panel, "close": panel["close"][cols], "tradable": panel["tradable"][cols]}
    close, open_ = close[cols], open_[cols]

    pf = _run_one(sub, close, open_, window, k)
    bh = close.loc[config.BACKTEST_START:]
    equity = pd.DataFrame(
        {
            f"均值回归 BB({window},{k})": _combined_value(pf),
            "买入持有(等额)": (bh / bh.iloc[0]).mean(axis=1) * config.INIT_CASH,
            "上证50指数": bt.benchmark_value(panel),
        }
    )

    # 每只股票的结果
    names = panel["names"]
    per_stock = pd.DataFrame(
        {
            "名称": [names.get(c, "") for c in cols],
            "策略收益": pf.total_return().values,
            "买入持有收益": (bh.iloc[-1] / bh.iloc[0] - 1).values,
            "交易次数": pf.trades.count().values,
            "胜率": pf.trades.win_rate().values,
            "持仓时间占比": (pf.position_mask().mean()).values,
        },
        index=cols,
    ).sort_values("策略收益", ascending=False)

    # 参数网格：窗口 x 阈值 -> 组合夏普
    grid = {}
    for w in [10, 20, 40, 60]:
        for kk in [1.5, 2.0, 2.5]:
            grid[(w, kk)] = bt.perf_stats(_combined_value(_run_one(sub, close, open_, w, kk)))
    grid = pd.DataFrame(grid).T
    grid.index.names = ["窗口", "阈值k"]

    # 单只股票示例：价格、布林带、买卖点
    px = panel["close"][example]
    ma, sd = px.rolling(window).mean(), px.rolling(window).std()
    ex_trades = pf.trades.records_readable
    ex_trades = ex_trades[ex_trades["Column"] == example]
    example_data = {
        "code": example,
        "name": names.get(example, ""),
        "bands": pd.DataFrame({"close": px, "ma": ma, "upper": ma + k * sd, "lower": ma - k * sd}).loc[config.BACKTEST_START:],
        "trades": ex_trades,
        "value": pf.value()[example],
    }

    return {
        "description": __doc__,
        "params": {"window": window, "k": k, "stop_loss": STOP_LOSS},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": equity.apply(bt.perf_stats).T,
        "per_stock": per_stock,
        "grid": grid,
        "example": example_data,
    }
