"""多因子选股·增强版。

基础版：5 个因子等权合成、Top10 等权、每月全换。改进：
1. 滚动 IC 加权：每个因子的权重 = 过去 window 个月 Rank IC 均值（只用截至上月已实现的 IC，
   无未来函数），负值截为 0；全部为负时退化为等权。动量也作为候选因子，交给 IC 决定是否使用。
2. 持仓从 10 只扩到 30 只 + 缓冲区（前 45 名内不换）+ 逆波动率加权：降低个股噪声、换手和集中度。
   在 cap300 上这是改进的主要来源（夏普 0.47 -> 0.52，最大回撤 -40% -> -31%）。
3. 市值中性化（可选，默认关）：复合得分对 log(流通市值) 回归取残差。在市值前 300 这种
   市值区间本就很窄的股票池里，中性化没有帮助（持 10 只时还明显拖累），因此默认关闭、保留为消融对照。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest as bt
from quant.enhanced.common import buffered_selection, inverse_vol_weights, rebalance_dates
from quant.strategies.factor import COMPOSITE, compute_factors, rank_ic, standardize
from quant.strategies.momentum import top_n_weights

CANDIDATES = [*COMPOSITE, "动量12-1"]


def neutralize(score: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """逐期横截面回归 score ~ a + b*exposure，返回残差。"""
    out = score.copy() * np.nan
    for d in score.index:
        y, x = score.loc[d], exposure.loc[d]
        m = y.notna() & x.notna()
        if m.sum() < 10:
            continue
        b = np.polyfit(x[m], y[m], 1)
        out.loc[d, m] = y[m] - np.polyval(b, x[m])
    return out


def ic_weights(ic: pd.DataFrame, window: int) -> pd.DataFrame:
    """调仓日 d 可用的因子权重。ic.loc[d] 依赖 d 之后一个月的收益，所以必须 shift(1)。"""
    w = ic.shift(1).rolling(window, min_periods=12).mean().clip(lower=0)
    w = w.div(w.sum(axis=1), axis=0)
    eq = pd.DataFrame(1 / ic.shape[1], index=ic.index, columns=ic.columns)
    return w.where(w.sum(axis=1) > 0, eq)


def run(panel: dict, top_n: int = 30, buffer: int = 15, ic_window: int = 24, neutral: bool = False,
        base_n: int = 10) -> dict:
    close, tradable = panel["close"], panel["tradable"]
    dates = rebalance_dates(close.index)
    mask = tradable.loc[dates]

    raw = compute_factors(panel)
    z = {k: standardize(raw[k].loc[dates].where(mask)) for k in CANDIDATES}
    fwd = close.loc[dates].pct_change(fill_method=None).shift(-1)
    ic = pd.DataFrame({k: rank_ic(z[k], fwd) for k in CANDIDATES}).reindex(dates)
    fw = ic_weights(ic, ic_window)

    stacked = pd.concat({k: z[k].mul(fw[k], axis=0) for k in CANDIDATES})
    comp = stacked.groupby(level=1, sort=False).sum(min_count=3).reindex(dates)
    log_cap = -raw["小市值"].loc[dates]
    comp_n = standardize(neutralize(comp, log_cap))

    vol60 = close.pct_change(fill_method=None).rolling(60).std()

    def construct(score: pd.DataFrame) -> pd.DataFrame:
        picked = buffered_selection(score, tradable, dates, top_n, buffer, bt.exec_mask(panel))
        return inverse_vol_weights(picked, vol60)

    w = construct(comp_n if neutral else comp)
    w_alt = construct(comp if neutral else comp_n)  # 消融：中性化开/关对调
    pf = bt.backtest_weights(panel, w)

    # 消融：IC 加权但等权 TopN、不缓冲；基础版：等权合成 Top base_n
    w_ic = top_n_weights(comp, tradable, dates, top_n)
    base_comp = pd.concat({k: z[k] for k in COMPOSITE}).groupby(level=1, sort=False).mean().reindex(dates)
    w_base = top_n_weights(base_comp, tradable, dates, base_n)
    w_all = mask.astype(float).div(mask.sum(axis=1), axis=0)

    tag = "中性化/" if neutral else ""
    equity = pd.DataFrame({
        f"因子+ (IC加权/{tag}缓冲/逆波 Top{top_n})": pf.value(),
        f"因子+ {'去掉' if neutral else '加上'}市值中性化": bt.backtest_weights(panel, w_alt).value(),
        f"因子+ 仅IC加权 等权Top{top_n}": bt.backtest_weights(panel, w_ic).value(),
        f"基础版 等权复合 Top{base_n}": bt.backtest_weights(panel, w_base).value(),
        "股票池等权": bt.backtest_weights(panel, w_all).value(),
        bt.config.BENCHMARK_NAME: bt.benchmark_value(panel),
    })
    stats = equity.apply(bt.perf_stats).T
    to = lambda ww: (ww.diff().abs().sum(axis=1) / 2).iloc[1:].mean()  # noqa: E731
    stats["月均单边换手"] = [to(w), to(w_alt), to(w_ic), to(w_base), np.nan, np.nan]

    ic_comp = pd.Series({"基础版复合": rank_ic(standardize(base_comp), fwd).mean(),
                         "IC加权复合": rank_ic(standardize(comp), fwd).mean(),
                         "IC加权+中性化": rank_ic(comp_n, fwd).mean()})
    return {
        "description": __doc__,
        "params": {"candidates": CANDIDATES, "top_n": top_n, "buffer": buffer, "ic_window": ic_window,
                   "neutral": neutral},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": stats,
        "factor_weights": fw.loc[pd.Timestamp(bt.config.BACKTEST_START):],
        "composite_ic": ic_comp,
        "weights": w,
        "weights_mode": "full",
        "asset": "stock",
    }
