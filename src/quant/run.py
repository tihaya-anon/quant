"""运行全部策略并把结果保存到 data/results/，供 notebook 可视化。

用法：uv run python -m quant.run [momentum mean_reversion trend factor]
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from quant import backtest as bt
from quant.data import load_panel
from quant.strategies import factor, mean_reversion, momentum, trend

STRATEGIES = {
    "momentum": momentum.run,
    "mean_reversion": mean_reversion.run,
    "trend": trend.run,
    "factor": factor.run,
}


def main(names: list[str] | None = None) -> None:
    panel = load_panel()
    names = names or list(STRATEGIES)
    summary = []
    for name in names:
        t = time.time()
        print(f"== {name} ==")
        res = STRATEGIES[name](panel)
        bt.save_results(name, res)
        stats = res["stats"].copy()
        stats.index = pd.MultiIndex.from_product([[name], stats.index])
        summary.append(stats)
        print(res["stats"].to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"耗时 {time.time() - t:.1f}s\n")
    if set(names) == set(STRATEGIES):
        bt.save_results("summary", {"stats": pd.concat(summary)})


if __name__ == "__main__":
    main(sys.argv[1:] or None)
