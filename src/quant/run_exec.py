"""执行仿真实验：把增强版策略的目标权重交给 backtrader（quant.execution）重放。

前置：uv run python -m quant.run   （生成各策略的 weights）
用法：uv run python -m quant.run_exec [--quick]

实验：
1. 对账：零摩擦 backtrader vs 零成本 vectorbt，确认两套框架对同一组权重给出一致结果
2. 现实执行：四个增强版策略在 1000 万资金下的净值、成本分项、撮合受阻情况，对比 vectorbt 理想化结果
3. 资金容量：不同资金规模下收益如何被冲击成本侵蚀
4. 敏感性：参与率上限 × 冲击系数
5. 换手控制的价值：基础版动量（无缓冲）vs 增强版动量，在大资金下的成本差异
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import replace

import pandas as pd

from quant import backtest as vb
from quant import config
from quant import execution as ex
from quant.data import load_panel

STRATS = {
    "momentum_plus": "动量+",
    "mean_reversion_plus": "均值回归+",
    "trend_plus": "趋势+",
    "factor_plus": "因子+",
}
BASE_CASH = 10_000_000
CAPITALS = [1e6, 1e7, 1e8, 1e9, 5e9]


# ---------------------------------------------------------------- 并行执行
# 每次 backtrader 回测彼此独立；股票池扩大后单次回测要几十秒，用多进程并行。
# 每个工作进程只加载一次面板和策略结果（initializer），任务只传轻量的参数。

_PANEL: dict = {}
_RES: dict = {}


def _init_worker() -> None:
    warnings.filterwarnings("ignore")
    _PANEL.update(load_panel())
    for k in [*STRATS, "momentum"]:
        _RES[k] = vb.load_results(k)


def _job(key, strat: str, cfg: ex.ExecConfig, use_base_weights: bool = False):
    r = _RES[strat]
    w = r["weights_base"] if use_base_weights else r["weights"]
    t = time.time()
    e = ex.run(_PANEL, w, r["weights_mode"], r["asset"], cfg)
    return key, e, time.time() - t


def _norm(v: pd.Series) -> pd.Series:
    return v / v.iloc[0]


def main(quick: bool = False, workers: int | None = None) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    warnings.filterwarnings("ignore")
    t0 = time.time()
    panel = load_panel()
    res = {k: vb.load_results(k) for k in STRATS}
    out: dict = {"universe": config.UNIVERSE, "base_cash": BASE_CASH}
    caps = CAPITALS[:3] if quick else CAPITALS
    big = 1e8 if quick else 1e9

    # ---- 任务清单
    jobs = [(("reconcile",), "momentum_plus", ex.ExecConfig.frictionless(BASE_CASH), False)]
    jobs += [(("real", k), k, ex.ExecConfig(cash=BASE_CASH), False) for k in STRATS]
    jobs += [(("cap", k, c), k, ex.ExecConfig(cash=c), False)
             for k in ["momentum_plus", "factor_plus", "mean_reversion_plus"] for c in caps]
    jobs += [(("sens", part, eta), "momentum_plus", replace(ex.ExecConfig(cash=big), participation=part, impact_eta=eta),
              False) for part in [0.02, 0.05, 0.10, 0.20] for eta in [0.35, 0.7, 1.4]]
    jobs += [(("turn", label, c), "momentum_plus", ex.ExecConfig(cash=c), base)
             for c in caps for label, base in [("基础版 动量Top10", True), ("动量+", False)]]

    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 2))
    print(f"共 {len(jobs)} 次回测，{workers} 个进程并行", flush=True)
    done: dict = {}
    with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
        futs = [pool.submit(_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            key, e, dt = f.result()
            done[key] = e
            st = e.stats()
            print(f"[{i}/{len(jobs)}] {key}: 年化 {st['年化收益']:.2%}  成本 {st['总成本(年化bp)']:.0f}bp  "
                  f"单次 {dt:.0f}s  累计 {time.time() - t0:.0f}s", flush=True)

    # ---- 1. 对账
    fees, slip = config.FEES, config.SLIPPAGE
    config.FEES = config.SLIPPAGE = 0
    v0 = vb.backtest_weights(panel, res["momentum_plus"]["weights"]).value()
    config.FEES, config.SLIPPAGE = fees, slip
    rec = pd.DataFrame({"backtrader 零摩擦": _norm(done[("reconcile",)].value), "vectorbt 零成本": _norm(v0)}).dropna()
    out["reconcile"] = rec
    print("\n== 1. 对账 ==")
    print(rec.apply(vb.perf_stats).T[["年化收益", "夏普比率", "最大回撤"]].round(4))

    # ---- 2. 现实执行
    real, equity = {}, {}
    for k, name in STRATS.items():
        e = done[("real", k)]
        real[name] = e
        equity[f"{name} 事件驱动执行"] = _norm(e.value)
        equity[f"{name} 向量化(固定0.2%)"] = _norm(res[k]["equity"].iloc[:, 0])
    out["real_stats"] = pd.DataFrame({k: v.stats() for k, v in real.items()}).T
    out["real_equity"] = pd.DataFrame(equity)
    out["vector_stats"] = pd.DataFrame({STRATS[k]: res[k]["stats"].iloc[0] for k in STRATS}).T
    out["blocked"] = pd.DataFrame({k: v.blocked for k, v in real.items()}).T.fillna(0)
    out["costs_log"] = {k: v.costs for k, v in real.items()}
    print(f"\n== 2. 现实执行（资金 {BASE_CASH / 1e4:.0f} 万）==")
    print(out["real_stats"][["年化收益", "夏普比率", "最大回撤", "总成本(年化bp)", "年化双边换手"]].round(3))

    # ---- 3. 资金容量
    out["capacity"] = pd.DataFrame({(STRATS[k], c): done[("cap", k, c)].stats()
                                    for k in ["momentum_plus", "factor_plus", "mean_reversion_plus"] for c in caps}
                                   ).T.rename_axis(["策略", "资金"])
    out["capacity_equity"] = pd.DataFrame({(STRATS[k], c): _norm(done[("cap", k, c)].value)
                                           for k in ["momentum_plus", "factor_plus", "mean_reversion_plus"] for c in caps})
    print("\n== 3. 资金容量 ==")
    print(out["capacity"][["年化收益", "冲击(年化bp)", "总成本(年化bp)"]].round(3))

    # ---- 4. 敏感性
    out["sensitivity"] = pd.DataFrame({(p_, e_): done[("sens", p_, e_)].stats()
                                       for p_ in [0.02, 0.05, 0.10, 0.20] for e_ in [0.35, 0.7, 1.4]}
                                      ).T.rename_axis(["参与率上限", "冲击系数"])
    out["sensitivity_cash"] = big
    print(f"\n== 4. 敏感性（动量+，{big / 1e8:.0f} 亿）==")
    print(out["sensitivity"]["年化收益"].unstack().round(4))

    # ---- 5. 换手控制的价值
    out["turnover_value"] = pd.DataFrame({(label, c): done[("turn", label, c)].stats()
                                          for c in caps for label in ["基础版 动量Top10", "动量+"]}
                                         ).T.rename_axis(["策略", "资金"])
    out["base_momentum_vector"] = vb.load_results("momentum")["stats"].iloc[0]
    print("\n== 5. 基础版动量 vs 动量+ ==")
    print(out["turnover_value"][["年化收益", "夏普比率", "总成本(年化bp)", "年化双边换手"]].round(3))

    vb.save_results("execution", out)
    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    w = next((int(a.split("=")[1]) for a in args if a.startswith("--workers=")), None)
    main(quick="--quick" in args, workers=w)
