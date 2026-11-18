"""增强版策略共用的组合构建工具：缓冲区选股、逆波动率加权、波动率目标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest as bt
from quant import config


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = bt.month_ends(index)
    return dates[dates >= pd.Timestamp(config.BACKTEST_START) - pd.Timedelta(days=40)]


def buffered_selection(score: pd.DataFrame, tradable: pd.DataFrame, dates: pd.DatetimeIndex,
                       n: int, buffer: int, can_trade: pd.DataFrame | None = None) -> pd.DataFrame:
    """带缓冲区（迟滞）的 TopN 选股，用来降低换手。

    已持有的股票只要排名仍在前 n+buffer 就继续持有；空出的名额按排名补齐。
    返回 bool 矩阵（index=dates）。
    can_trade：能否成交（停牌为 False）；停牌的持仓卖不掉，只能继续持有。
    被剔出成分股但仍可交易的持仓会被卖出。
    """
    can_trade = tradable if can_trade is None else can_trade
    held: set[str] = set()
    rows = {}
    for d in dates:
        s = score.loc[d].where(tradable.loc[d]).dropna().sort_values(ascending=False)
        rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
        keep = {c for c in held if c in rank.index and rank[c] <= n + buffer}
        # 停牌的持仓无法卖出，只能继续保留
        keep |= {c for c in held if c not in rank.index and not can_trade.loc[d].get(c, False)}
        for c in rank.index:
            if len(keep) >= n:
                break
            keep.add(c)
        held = keep
        rows[d] = pd.Series(True, index=sorted(held), dtype=bool)
    return pd.DataFrame(rows).T.reindex(columns=score.columns).fillna(False).astype(bool)


def inverse_vol_weights(picked: pd.DataFrame, vol: pd.DataFrame, cap: float = 0.2) -> pd.DataFrame:
    """在选中的股票上按 1/波动率 加权，单票上限 cap。"""
    inv = (1 / vol.loc[picked.index]).where(picked).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    for _ in range(10):  # 迭代截断并重新分配超出上限的部分
        over = w > cap
        if not over.any().any():
            break
        excess = (w - cap).clip(lower=0).sum(axis=1)
        w = w.clip(upper=cap)
        free = w.where(~over & w.notna())
        w = w + free.div(free.sum(axis=1), axis=0).mul(excess, axis=0).fillna(0)
    return w.fillna(0.0)


def vol_target_scale(daily_ret: pd.Series, target: float, window: int = 60, cap: float = 1.0) -> pd.Series:
    """波动率目标：仓位 = 目标年化波动 / 近期实现波动，最高 cap（不加杠杆）。"""
    realized = daily_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(bt.TRADING_DAYS)
    return (target / realized).clip(upper=cap).fillna(cap)
