"""回测公共工具：基于 vectorbt 的权重回测 / 信号回测、绩效统计、结果读写。

执行约定（避免未来函数）：t 日收盘后产生信号，t+1 日开盘价成交。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from quant import config

TRADING_DAYS = 252


# ---------------------------------------------------------------- 日期工具


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个月最后一个交易日。"""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


def _exec_prices(panel: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """估值用收盘价、成交用开盘价；上市前的 NaN 用首个有效值回填（此时持仓必为 0，不影响结果）。"""
    close = panel["close"].ffill().bfill()
    open_ = panel["open"].fillna(panel["close"]).ffill().bfill()
    return close, open_


# ---------------------------------------------------------------- 回测


def backtest_weights(panel: dict, weights: pd.DataFrame, start: str = config.BACKTEST_START) -> vbt.Portfolio:
    """按目标权重调仓（组合层面共享资金）。

    weights: index 为调仓信号日（收盘后决定），columns 为股票代码，行内权重和 <= 1。
    在下一交易日开盘按目标权重调仓；停牌/不可交易的股票当天不下单（保持原仓位）。
    """
    close, open_ = _exec_prices(panel)
    target = weights.reindex(close.index).reindex(columns=close.columns)
    target.loc[weights.index] = target.loc[weights.index].fillna(0.0)
    target = target.shift(1)  # t 日信号 -> t+1 日执行
    target = target.where(panel["tradable"].reindex_like(target).fillna(False).astype(bool))

    sl = slice(start, None)
    return vbt.Portfolio.from_orders(
        close=close.loc[sl],
        size=target.loc[sl],
        size_type="targetpercent",
        price=open_.loc[sl],
        fees=config.FEES,
        slippage=config.SLIPPAGE,
        init_cash=config.INIT_CASH,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",  # 先卖后买，保证资金充足
        freq="1D",
    )


def backtest_signals(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    tradable: pd.DataFrame | None = None,
    start: str = config.BACKTEST_START,
    **kwargs,
) -> vbt.Portfolio:
    """择时信号回测：每列独立（每只证券/每组参数一份资金），信号次日开盘成交。"""
    entries, exits = entries.shift(1, fill_value=False), exits.shift(1, fill_value=False)
    if tradable is not None:
        ok = tradable.reindex_like(entries).fillna(False).astype(bool)
        entries, exits = entries & ok, exits & ok
    sl = slice(start, None)
    return vbt.Portfolio.from_signals(
        close=close.loc[sl],
        entries=entries.loc[sl],
        exits=exits.loc[sl],
        price=open_.loc[sl],
        fees=config.FEES,
        slippage=config.SLIPPAGE,
        init_cash=config.INIT_CASH,
        freq="1D",
        **kwargs,
    )


# ---------------------------------------------------------------- 绩效


def perf_stats(value: pd.Series) -> pd.Series:
    """由净值序列计算常用绩效指标。"""
    value = value.dropna()
    ret = value.pct_change().dropna()
    years = len(ret) / TRADING_DAYS
    total = value.iloc[-1] / value.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    dd = value / value.cummax() - 1
    return pd.Series(
        {
            "累计收益": total,
            "年化收益": cagr,
            "年化波动": vol,
            "夏普比率": ret.mean() / ret.std() * np.sqrt(TRADING_DAYS) if ret.std() > 0 else np.nan,
            "最大回撤": dd.min(),
            "卡玛比率": cagr / -dd.min() if dd.min() < 0 else np.nan,
            "日胜率": (ret > 0).mean(),
        }
    )


def benchmark_value(panel: dict, code: str = config.BENCHMARK, start: str = config.BACKTEST_START) -> pd.Series:
    """基准指数净值（归一化到初始资金）。"""
    px = panel["index_close"][code].loc[start:]
    return px / px.iloc[0] * config.INIT_CASH


def drawdown(value: pd.Series) -> pd.Series:
    return value / value.cummax() - 1


# ---------------------------------------------------------------- 结果读写


def save_results(name: str, results: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, config.RESULTS_DIR / f"{name}.pkl")
    print(f"结果已保存: {config.RESULTS_DIR / f'{name}.pkl'}")


def load_results(name: str) -> dict:
    path = config.RESULTS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}，请先运行: uv run python -m quant.run")
    return pd.read_pickle(path)
