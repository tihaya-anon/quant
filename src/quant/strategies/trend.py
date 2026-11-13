"""趋势跟随策略（指数择时）。

思路：顺势而为，趋势形成后跟进，趋势反转时离场，靠少数大趋势盈利、用小亏损过滤震荡。
标的：沪深300 指数（实际可用对应 ETF 交易）。两种经典规则：
1. 双均线：快线上穿慢线买入，下穿卖出；扫描 (快, 慢) 参数网格看稳健性
2. 唐奇安通道突破（海龟法则简化版）：收盘创 N 日新高买入，跌破 M 日新低卖出
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from quant import backtest as bt

TARGET = "sh.000300"


def ma_cross_signals(close: pd.Series, fast: int, slow: int):
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    above = f > s
    return above & ~above.shift(1, fill_value=False), ~above & above.shift(1, fill_value=False)


def donchian_signals(close: pd.Series, entry_n: int, exit_n: int):
    upper = close.rolling(entry_n).max().shift(1)
    lower = close.rolling(exit_n).min().shift(1)
    return close > upper, close < lower, upper, lower


def run(panel: dict, fast: int = 20, slow: int = 60, entry_n: int = 55, exit_n: int = 20) -> dict:
    close = panel["index_close"][TARGET]
    open_ = panel["index_open"][TARGET].fillna(close)

    def one(entries, exits) -> vbt.Portfolio:
        return bt.backtest_signals(close.to_frame(), open_.to_frame(), entries.to_frame(), exits.to_frame())

    ma_en, ma_ex = ma_cross_signals(close, fast, slow)
    dc_en, dc_ex, dc_up, dc_lo = donchian_signals(close, entry_n, exit_n)
    pf_ma, pf_dc = one(ma_en, ma_ex), one(dc_en, dc_ex)

    equity = pd.DataFrame(
        {
            f"双均线 MA({fast},{slow})": pf_ma.value().iloc[:, 0],
            f"唐奇安 ({entry_n},{exit_n})": pf_dc.value().iloc[:, 0],
            "沪深300 买入持有": bt.benchmark_value(panel, TARGET),
        }
    )

    # 双均线参数网格：一次性构造所有组合的信号矩阵，vectorbt 并行回测
    fasts, slows = [5, 10, 20, 30, 40, 60], [20, 40, 60, 90, 120, 180, 250]
    combos = [(f, s) for f in fasts for s in slows if f < s]
    en = pd.concat({c: ma_cross_signals(close, *c)[0] for c in combos}, axis=1)
    ex = pd.concat({c: ma_cross_signals(close, *c)[1] for c in combos}, axis=1)
    n = len(combos)
    pf_grid = bt.backtest_signals(
        pd.DataFrame(np.tile(close.values[:, None], n), index=close.index, columns=en.columns),
        pd.DataFrame(np.tile(open_.values[:, None], n), index=close.index, columns=en.columns),
        en, ex,
    )
    grid_stats = pf_grid.value().apply(bt.perf_stats).T
    grid_stats.index = pd.MultiIndex.from_tuples(combos, names=["快线", "慢线"])

    sl = slice(equity.index[0], None)
    return {
        "description": __doc__,
        "params": {"target": TARGET, "fast": fast, "slow": slow, "entry_n": entry_n, "exit_n": exit_n},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": equity.apply(bt.perf_stats).T,
        "grid": grid_stats,
        "price": pd.DataFrame(
            {
                "close": close,
                "ma_fast": close.rolling(fast).mean(),
                "ma_slow": close.rolling(slow).mean(),
                "dc_upper": dc_up,
                "dc_lower": dc_lo,
            }
        ).loc[sl],
        "trades_ma": pf_ma.trades.records_readable,
        "trades_dc": pf_dc.trades.records_readable,
        "position_ma": pf_ma.position_mask().iloc[:, 0],
    }
