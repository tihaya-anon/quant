"""动量策略（横截面动量）。

思路：过去表现好的股票未来一段时间倾向于继续跑赢。
- 信号：过去 L 个交易日收益，剔除最近 21 日（经典 "12-1 动量"，规避短期反转）
- 每月末按动量排序，等权持有排名前 N 的股票，下一交易日开盘调仓
- 同时测试多组 (回看期 L, 持仓数 N)，观察参数敏感性
"""

from __future__ import annotations

import pandas as pd

from quant import backtest as bt
from quant import config

SKIP = 21  # 剔除最近一个月


def momentum_score(close: pd.DataFrame, lookback: int, skip: int = SKIP) -> pd.DataFrame:
    return close.shift(skip) / close.shift(lookback) - 1


def top_n_weights(score: pd.DataFrame, tradable: pd.DataFrame, dates: pd.DatetimeIndex, n: int,
                  ascending: bool = False) -> pd.DataFrame:
    """在调仓日选出得分最高（ascending=True 则最低）的 n 只可交易股票，等权。"""
    s = score.loc[dates].where(tradable.loc[dates])
    rank = s.rank(axis=1, ascending=ascending, method="first")
    picked = rank <= n
    return picked.div(picked.sum(axis=1), axis=0).fillna(0.0)


def run(panel: dict, lookback: int = 252, top_n: int = 10) -> dict:
    close, tradable = panel["close"], panel["tradable"]
    dates = bt.month_ends(close.index)
    dates = dates[dates >= pd.Timestamp(config.BACKTEST_START) - pd.Timedelta(days=40)]

    score = momentum_score(close, lookback)
    w_top = top_n_weights(score, tradable, dates, top_n)
    w_bottom = top_n_weights(score, tradable, dates, top_n, ascending=True)
    pf_top = bt.backtest_weights(panel, w_top)
    pf_bottom = bt.backtest_weights(panel, w_bottom)

    # 等权持有全部股票池作为"市场"对照
    w_all = tradable.loc[dates].astype(float)
    w_all = w_all.div(w_all.sum(axis=1), axis=0)
    pf_all = bt.backtest_weights(panel, w_all)

    equity = pd.DataFrame(
        {
            f"动量 Top{top_n}": pf_top.value(),
            f"动量 Bottom{top_n}": pf_bottom.value(),
            "股票池等权": pf_all.value(),
            "上证50指数": bt.benchmark_value(panel),
        }
    )

    # 参数敏感性：回看期 x 持仓数
    grid = {}
    for lb in [63, 126, 252]:
        sc = momentum_score(close, lb)
        for n in [5, 10, 15]:
            pf = bt.backtest_weights(panel, top_n_weights(sc, tradable, dates, n))
            grid[(lb, n)] = bt.perf_stats(pf.value())
    grid = pd.DataFrame(grid).T
    grid.index.names = ["回看期", "持仓数"]

    names = panel["names"]
    holdings = w_top.loc[w_top.index >= config.BACKTEST_START]
    latest = holdings.iloc[-1]
    latest = latest[latest > 0].rename(index=lambda c: f"{c} {names.get(c, '')}")

    return {
        "description": __doc__,
        "params": {"lookback": lookback, "skip": SKIP, "top_n": top_n},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": equity.apply(bt.perf_stats).T,
        "grid": grid,
        "holdings": holdings,  # 每月持仓权重
        "latest_holdings": latest,
        "trades": pf_top.orders.records_readable,
        "turnover": (holdings.diff().abs().sum(axis=1) / 2).iloc[1:],  # 单边换手率
    }
