"""趋势跟随·增强版（沪深300 择时）。

基础版的问题：单一均线对/通道参数，信号要么满仓要么空仓，在震荡市里反复打脸；
仓位与市场波动无关，高波动下跌段损失大。改进：
1. 信号集成：多组均线交叉 + 多组唐奇安通道 + 时间序列动量，取"看多比例"作为 0~1 的连续仓位，
   降低对单组参数的依赖（参数网格本身已经说明单组参数很不稳定）。
2. 波动率目标：仓位再乘以 目标波动/近 20 日实现波动（上限 1），下跌往往伴随波动放大，自动降仓。
3. 调仓阈值：目标仓位变化超过 threshold 才调整，避免每天小额交易。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest as bt
from quant.enhanced.common import vol_target_scale
from quant.strategies.trend import TARGET, ma_cross_signals

MA_PAIRS = [(10, 60), (20, 120), (40, 250)]
DONCHIAN = [(20, 10), (55, 20), (120, 60)]
TSMOM = [60, 120, 250]


def ensemble_signal(close: pd.Series) -> pd.DataFrame:
    """每个子信号输出 1=看多 / 0=看空，列为子信号名。"""
    sig = {}
    for f, s in MA_PAIRS:
        sig[f"MA{f}/{s}"] = (close.rolling(f).mean() > close.rolling(s).mean()).astype(float)
    for en, ex in DONCHIAN:
        up, lo = close.rolling(en).max().shift(1), close.rolling(ex).min().shift(1)
        state = pd.Series(np.where(close > up, 1.0, np.where(close < lo, 0.0, np.nan)), index=close.index)
        sig[f"DC{en}/{ex}"] = state.ffill().fillna(0.0)
    for n in TSMOM:
        sig[f"TSMOM{n}"] = (close / close.shift(n) > 1).astype(float)
    return pd.DataFrame(sig)


def apply_threshold(target: pd.Series, threshold: float) -> pd.Series:
    """只有当目标仓位与当前仓位偏离超过阈值（或目标为 0）时才调整。返回调仓日的目标。"""
    cur, out = 0.0, {}
    for d, x in target.items():
        if abs(x - cur) > threshold or (x == 0 and cur > 0):
            cur = x
            out[d] = x
    return pd.Series(out, dtype=float)


def run(panel: dict, vol_target: float = 0.15, threshold: float = 0.1) -> dict:
    close = panel["index_close"][TARGET]
    open_ = panel["index_open"][TARGET].fillna(close)
    c, o = close.to_frame(), open_.to_frame()

    sig = ensemble_signal(close)
    raw = sig.mean(axis=1)
    scale = vol_target_scale(close.pct_change(), vol_target, window=20)
    target = (raw * scale).round(2)

    def run_w(x: pd.Series):
        w = apply_threshold(x, threshold).to_frame(TARGET)
        return bt.backtest_target(c, o, w), w

    pf, w = run_w(target)
    pf_raw, _ = run_w(raw.round(2))
    ma_en, ma_ex = ma_cross_signals(close, 20, 60)
    pf_base = bt.backtest_signals(c, o, ma_en.to_frame(), ma_ex.to_frame())

    equity = pd.DataFrame({
        "趋势+ (集成信号/目标波动)": pf.value(),
        "趋势+ 仅集成信号": pf_raw.value(),
        "基础版 双均线 MA(20,60)": pf_base.value().iloc[:, 0],
        "沪深300 买入持有": bt.benchmark_value(panel, TARGET),
    })
    stats = equity.apply(bt.perf_stats).T
    exposure = w[TARGET].reindex(close.index).ffill().fillna(0).loc[equity.index[0]:]
    stats["平均仓位"] = [exposure.mean(), np.nan, pf_base.position_mask().iloc[:, 0].mean(), 1.0]

    # 各子信号单独回测：说明单一信号的不稳定性
    singles = {k: bt.perf_stats(bt.backtest_target(c, o, v.to_frame(TARGET).loc[v.diff() != 0]).value())
               for k, v in sig.items()}

    return {
        "description": __doc__,
        "params": {"ma_pairs": MA_PAIRS, "donchian": DONCHIAN, "tsmom": TSMOM,
                   "vol_target": vol_target, "threshold": threshold},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": stats,
        "singles": pd.DataFrame(singles).T,
        "exposure": exposure,
        "price": close.loc[equity.index[0]:],
        "weights": w,
        "weights_mode": "full",
        "asset": "index",
    }
