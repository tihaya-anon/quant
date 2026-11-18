# quant

用 baostock 真实 A 股数据演示四种主流量化策略（动量、均值回归、趋势跟随、因子投资）：
基础版 → 增强版 → 事件驱动执行仿真。

## 结构

```
src/quant/
  config.py               # 路径、区间、股票池、交易成本
  local_data.py           # 本地数据包（全A日K+复权因子）-> parquet -> 面板，市值股票池
  data.py                 # baostock 下载 + 历史成分股（point-in-time）+ 宽表面板
  backtest.py             # vectorbt 回测封装、绩效统计、结果读写
  run.py                  # 运行基础版 + 增强版，结果写入 data/results/<股票池>/
  strategies/             # 基础版
    momentum.py           # 横截面 12-1 动量，月度调仓 Top10
    mean_reversion.py     # 布林带 Z-score 择时
    trend.py              # 沪深300 双均线 / 唐奇安通道 + 参数网格
    factor.py             # 多因子：IC、五分组、Top10 回测
  enhanced/               # 增强版（均含与基础版的对比和消融）
    momentum.py           # 风险调整动量、缓冲区、逆波动率、波动率目标
    mean_reversion.py     # MA250 趋势过滤、共享资金名额制、时间止损
    trend.py              # 多信号集成 + 波动率目标 + 调仓阈值
    factor.py             # 滚动 IC 加权、市值中性化、缓冲区
  execution.py            # backtrader 事件驱动执行：费用/印花税/价差/平方根冲击/参与率/涨跌停/停牌/整手
  run_exec.py             # 执行仿真实验：对账、成本分解、资金容量、敏感性
  viz.py                  # plotly 绘图样式
  report.py               # 汇总结果 → reports/<股票池>.html
```

## 股票池

`config.UNIVERSE`（或环境变量 `QUANT_UNIVERSE`）：

| 值 | 含义 |
|---|---|
| `cap300`（默认） | 本地数据包：每月末按**当时**流通市值取前 300（近似沪深300） |
| `cap800` | 本地数据包：流通市值前 800（近似中证800） |
| `hs300_pit` | 沪深300 **历史**成分股：每月末查询当时成分，含已退市股票 |
| `zz800_pit` | 沪深300 + 中证500 历史成分股 |
| `sz50_pit` | 上证50 历史成分股 |
| `sz50` | 上证50 **当前**成分股（旧版，有幸存者偏差，仅作对照） |

## 使用

```bash
uv sync
# 数据（二选一）
uv run python -m quant.local_data all   # cap300/cap800：从购买的日K数据包构建（路径见 local_data.py / QUANT_LOCAL_SRC）
uv run python -m quant.data             # *_pit / sz50：从 baostock 下载（注意限速，过快会被封 IP）

uv run python -m quant.run        # 基础版 + 增强版
uv run python -m quant.run_exec   # 执行仿真（较慢，--quick 只跑小资金）
uv run python -m quant.report     # 生成 reports/<股票池>.html，浏览器打开即可

# 对比有偏股票池：
QUANT_UNIVERSE=sz50 uv run python -m quant.run
```
