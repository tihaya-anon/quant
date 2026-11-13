# quant

用 baostock 真实 A 股数据 + vectorbt 演示四种主流量化策略：动量、均值回归、趋势跟随、因子投资。

## 结构

```
src/quant/
  config.py               # 路径、区间、股票池、交易成本
  data.py                 # baostock 下载 + 缓存 + 构建宽表面板
  backtest.py             # vectorbt 回测封装、绩效统计、结果读写
  run.py                  # 运行全部策略，结果写入 data/results/
  strategies/
    momentum.py           # 横截面 12-1 动量，月度调仓 Top10
    mean_reversion.py     # 布林带 Z-score 择时
    trend.py              # 沪深300 双均线 / 唐奇安通道 + 参数网格
    factor.py             # 多因子：IC、五分组、Top10 回测
notebooks/strategies.ipynb  # 只读取 data/results/ 做可视化
```

## 使用

```bash
uv sync
uv run python -m quant.data   # 下载数据（有缓存，可断点续跑）
uv run python -m quant.run    # 回测，可指定单个：uv run python -m quant.run factor
```

然后打开 `notebooks/strategies.ipynb`（内核选 `.venv`）。
