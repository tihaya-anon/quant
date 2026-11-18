"""动量策略·增强版。

相对基础版（12-1 原始收益 Top10 等权）的改进：
1. 风险调整动量：收益 / 波动率，并合成 12-1 与 6-1 两个期限，降低对单一参数的依赖、
   避免高波动股票因"运气"排在前面（A股高波动股票后续收益往往更差）。
2. 缓冲区选股：持仓只要还在前 N+buffer 名就不换，显著降低换手和交易成本。
3. 逆波动率加权 + 单票上限，替代等权。
4. 组合波动率目标：用策略自身过去 60 日的实现波动缩放总仓位（两遍法，无未来函数），
   在动量崩溃常见的高波动阶段自动降仓。
"""

from __future__ import annotations

import pandas as pd

from quant import backtest as bt
from quant.enhanced.common import (buffered_selection, inverse_vol_weights, rebalance_dates,
                                   vol_target_scale)
from quant.strategies.factor import standardize
from quant.strategies.momentum import momentum_score, top_n_weights


def risk_adjusted_score(close: pd.DataFrame) -> pd.DataFrame:
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(252, min_periods=120).std()
    s12 = momentum_score(close, 252) / vol
    s6 = momentum_score(close, 126) / ret.rolling(126, min_periods=60).std()
    return pd.concat([standardize(s12), standardize(s6)]).groupby(level=0).mean().reindex(close.index)


def run(panel: dict, top_n: int = 10, buffer: int = 5, vol_target: float = 0.20) -> dict:
    close, tradable = panel["close"], panel["tradable"]
    dates = rebalance_dates(close.index)
    vol60 = close.pct_change(fill_method=None).rolling(60).std()

    score = risk_adjusted_score(close)
    picked = buffered_selection(score, tradable, dates, top_n, buffer, bt.exec_mask(panel))
    w_raw = inverse_vol_weights(picked, vol60)

    # 第一遍：满仓回测得到策略日收益；第二遍：按调仓日之前的实现波动缩放仓位
    pf_raw = bt.backtest_weights(panel, w_raw)
    scale = vol_target_scale(pf_raw.value().pct_change(), vol_target).reindex(dates).fillna(1.0)
    w = w_raw.mul(scale, axis=0)
    pf = bt.backtest_weights(panel, w)

    w_base = top_n_weights(momentum_score(close, 252), tradable, dates, top_n)
    pf_base = bt.backtest_weights(panel, w_base)
    turnover = lambda ww: (ww.diff().abs().sum(axis=1) / 2).iloc[1:]  # noqa: E731

    equity = pd.DataFrame({
        "动量+ (风险调整/缓冲/逆波/目标波动)": pf.value(),
        "动量+ 不做波动率目标": pf_raw.value(),
        f"基础版 动量 Top{top_n}": pf_base.value(),
        bt.config.BENCHMARK_NAME: bt.benchmark_value(panel),
    })
    stats = equity.apply(bt.perf_stats).T
    stats["月均单边换手"] = [turnover(w).mean(), turnover(w_raw).mean(), turnover(w_base).mean(), float("nan")]
    return {
        "description": __doc__,
        "params": {"top_n": top_n, "buffer": buffer, "vol_target": vol_target},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": stats,
        "exposure": scale,
        "weights": w,
        "weights_base": w_base,
        "weights_mode": "full",
        "asset": "stock",
    }
