"""运行全部策略并把结果保存到 data/results/，供 notebook 可视化。

用法：
    uv run python -m quant.run                    # 基础版 + 增强版全部
    uv run python -m quant.run momentum factor_plus   # 指定若干个
"""

from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

from quant import backtest as bt
from quant import config
from quant.data import load_panel
from quant.enhanced import factor as factor_plus
from quant.enhanced import mean_reversion as mean_reversion_plus
from quant.enhanced import momentum as momentum_plus
from quant.enhanced import trend as trend_plus
from quant.strategies import factor, mean_reversion, momentum, trend

BASE = {
    "momentum": momentum.run,
    "mean_reversion": mean_reversion.run,
    "trend": trend.run,
    "factor": factor.run,
}
ENHANCED = {
    "momentum_plus": momentum_plus.run,
    "mean_reversion_plus": mean_reversion_plus.run,
    "trend_plus": trend_plus.run,
    "factor_plus": factor_plus.run,
}
STRATEGIES = {**BASE, **ENHANCED}


def main(names: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    panel = load_panel()
    print(f"股票池: {config.UNIVERSE}  ({panel['close'].shape[1]} 只)\n")
    names = names or list(STRATEGIES)
    summary = {}
    for name in names:
        t = time.time()
        print(f"== {name} ==")
        res = STRATEGIES[name](panel)
        res["universe"] = config.UNIVERSE
        bt.save_results(name, res)
        summary[name] = res["stats"]
        print(res["stats"].to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"耗时 {time.time() - t:.1f}s\n")
    for group, key in [(BASE, "summary"), (ENHANCED, "summary_plus")]:
        if set(group) <= set(names):
            bt.save_results(key, {"stats": pd.concat({k: summary[k] for k in group})})


if __name__ == "__main__":
    main(sys.argv[1:] or None)
