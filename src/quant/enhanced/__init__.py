"""四个基础策略的增强版。

每个模块的 run(panel) 返回与基础版相同结构的结果字典，并额外包含：
- "weights":  信号日目标权重（t 日收盘决定），供事件驱动执行引擎 quant.engine 复用
- "weights_mode": "full"（行内未列出的=0）或 "hold"（行内 NaN=不动）
- "asset": "stock"（上证50 成分股）或 "index"（指数，用 ETF 近似交易）
"""
