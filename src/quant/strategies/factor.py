"""多因子选股（因子投资）。

思路：用若干有经济学解释的"因子"给股票打分，长期持有高分股票以获取因子溢价。
因子（均在月末计算，方向统一为"越大越好"）：
- 价值 EP  = 1 / 市盈率TTM
- 价值 BP  = 1 / 市净率MRQ
- 低波动   = -过去 60 日日收益标准差
- 短期反转 = -过去 21 日收益
- 小市值   = -log(流通市值)，流通市值 ≈ 成交额 / 换手率
- 动量     = 12-1 月收益（仅作 IC 对比，不进复合因子）
流程：去极值(5%/95%) -> 横截面标准化 -> 等权合成复合因子
检验：Rank IC（因子与下月收益的秩相关）、五分组分层回测、Top 组合含成本回测
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest as bt
from quant import config
from quant.strategies.momentum import momentum_score, top_n_weights

COMPOSITE = ["价值EP", "价值BP", "低波动", "短期反转", "小市值"]
N_GROUPS = 5


def compute_factors(panel: dict) -> dict[str, pd.DataFrame]:
    close = panel["close"]
    ret = close.pct_change()
    pe, pb = panel["peTTM"], panel["pbMRQ"]
    turn = panel["turn"].where(panel["turn"] > 0)
    float_cap = (panel["amount"] / (turn / 100)).rolling(20, min_periods=10).median()
    return {
        "价值EP": 1 / pe.where(pe != 0),
        "价值BP": 1 / pb.where(pb > 0),
        "低波动": -ret.rolling(60).std(),
        "短期反转": -(close / close.shift(21) - 1),
        "小市值": -np.log(float_cap),
        "动量12-1": momentum_score(close, 252),
    }


def standardize(f: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
    """横截面去极值 + Z-score。"""
    lo, hi = f.quantile(lower, axis=1), f.quantile(upper, axis=1)
    f = f.clip(lo, hi, axis=0)
    return f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0)


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """每期因子值与下期收益的 Spearman 秩相关。"""
    fr, rr = factor.rank(axis=1), fwd.where(factor.notna()).rank(axis=1)
    return fr.corrwith(rr, axis=1, method="pearson").dropna()


def run(panel: dict, top_n: int = 10) -> dict:
    close, tradable = panel["close"], panel["tradable"]
    dates = bt.month_ends(close.index)
    dates = dates[dates >= pd.Timestamp(config.BACKTEST_START) - pd.Timedelta(days=40)]
    universe_mask = tradable.loc[dates]

    raw = compute_factors(panel)
    z = {k: standardize(v.loc[dates].where(universe_mask)) for k, v in raw.items()}
    z_comp = pd.concat({k: z[k] for k in COMPOSITE}).groupby(level=1, sort=False).mean()
    n_valid = pd.concat({k: z[k].notna() for k in COMPOSITE}).groupby(level=1, sort=False).sum()
    composite = z_comp.reindex(dates).where(n_valid.reindex(dates) >= 3)
    z["复合因子"] = composite

    # 下月收益（月末收盘 -> 下月末收盘），用于 IC 和分层分析
    fwd = close.loc[dates].pct_change(fill_method=None).shift(-1)

    ic = pd.DataFrame({k: rank_ic(v, fwd) for k, v in z.items()})
    ic_summary = pd.DataFrame(
        {
            "IC均值": ic.mean(),
            "IC标准差": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            "t值": ic.mean() / ic.std() * np.sqrt(ic.count()),
            "IC>0占比": (ic > 0).mean(),
        }
    )

    # 五分组分层：每月按复合因子分成 5 组，组内等权，看下月收益（未扣费）
    groups = composite.rank(axis=1, pct=True).apply(lambda r: np.ceil(r * N_GROUPS))
    layer = pd.DataFrame(
        {f"Q{g}": fwd.where(groups == g).mean(axis=1) for g in range(1, N_GROUPS + 1)}
    ).dropna(how="all")
    layer["多空(Q5-Q1)"] = layer[f"Q{N_GROUPS}"] - layer["Q1"]
    layer_nav = (1 + layer.fillna(0)).cumprod()

    # 可交易回测：复合因子 Top N，含交易成本，次日开盘调仓
    w_top = top_n_weights(composite, tradable, dates, top_n)
    pf = bt.backtest_weights(panel, w_top)
    w_all = universe_mask.astype(float)
    w_all = w_all.div(w_all.sum(axis=1), axis=0)
    equity = pd.DataFrame(
        {
            f"复合因子 Top{top_n}": pf.value(),
            "股票池等权": bt.backtest_weights(panel, w_all).value(),
            "上证50指数": bt.benchmark_value(panel),
        }
    )

    # 因子间相关性（各期横截面相关系数的均值）
    corr = sum(
        pd.DataFrame({k: z[k].loc[d] for k in raw}).corr(method="spearman").fillna(0) for d in dates
    ) / len(dates)

    names = panel["names"]
    last = composite.dropna(how="all").index[-1]
    latest_scores = pd.DataFrame({k: z[k].loc[last] for k in [*COMPOSITE, "复合因子"]})
    latest_scores.insert(0, "名称", [names.get(c, "") for c in latest_scores.index])

    return {
        "description": __doc__,
        "params": {"composite": COMPOSITE, "top_n": top_n, "groups": N_GROUPS},
        "ic": ic,
        "ic_summary": ic_summary,
        "layer_returns": layer,
        "layer_nav": layer_nav,
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": equity.apply(bt.perf_stats).T,
        "factor_corr": corr,
        "latest_date": last,
        "latest_scores": latest_scores.sort_values("复合因子", ascending=False),
    }
