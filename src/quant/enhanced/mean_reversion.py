"""均值回归策略·增强版。

基础版的问题：每只股票独立分配 1/N 资金、绝大部分时间空仓（资金利用率约 10%），
且在趋势性下跌中反复"接飞刀"。改进：
1. 个股趋势过滤：只在长期上升趋势中买回调（收盘 > MA250"牛熊线"）。
2. 大盘过滤：沪深300 在 MA250 之上才开新仓。熊市里个股超跌信号成片出现、几乎都会继续下跌，
   这是从上证50 扩到 300 只股票后最关键的修正（无大盘过滤时最大回撤 -56%，加上后约 -17%）。
3. 更严格的入场阈值 k=2.5：股票池越大，k=2 的信号越多、质量越差（网格中 k=2.5 全面优于 2.0）。
4. 共享资金池 + 20 个名额（每只 5%）；信号多于空余名额时优先买 Z 最低的。
5. 出场：被剔出股票池时卖出；回到均线（Z>0）、持有满 max_hold 天、或较信号价下跌 stop 止损。

教学提示：参数最初在上证50 上调出（k=2、10 名额、无大盘过滤），换到更大的股票池后完全失效——
这是"参数对样本过拟合"的典型例子。即便修正后，大盘股上的抄底型均值回归收益仍然很低。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import backtest as bt
from quant.strategies.mean_reversion import _combined_value, _run_one, zscore


def generate_weights(close: pd.DataFrame, tradable: pd.DataFrame, can_trade: pd.DataFrame | None = None, window: int = 20, k: float = 2.0,
                     trend: int = 250, slots: int = 10, max_hold: int = 30, stop: float = 0.10,
                     regime: pd.Series | None = None) -> pd.DataFrame:
    """逐日状态机生成目标权重（hold 模式：NaN = 不动，0 = 卖出，1/slots = 买入）。

    regime: 大盘环境（bool，按日期），为 False 的日子不开新仓（已有持仓照常按规则退出）。
    """
    allow = regime.reindex(close.index).fillna(False).values if regime is not None else np.ones(len(close), bool)
    z = zscore(close, window).values
    up = (close > close.rolling(trend).mean()).values if trend else np.ones(close.shape, bool)
    px, ok = close.values, tradable.reindex_like(close).fillna(False).values
    sell_ok = (can_trade if can_trade is not None else tradable).reindex_like(close).fillna(False).values
    held: dict[int, tuple[int, float]] = {}  # 列号 -> (入场信号日序号, 信号价)
    rows = {}
    for t in range(len(close)):
        row = np.full(close.shape[1], np.nan)
        # 出场（只有可交易时才发卖单，否则留到之后）
        for j, (t0, p0) in list(held.items()):
            if sell_ok[t, j] and (not ok[t, j] or z[t, j] > 0 or t - t0 >= max_hold or px[t, j] < p0 * (1 - stop)):
                row[j] = 0.0
                del held[j]
        # 入场：下穿 -k、处于上升趋势、可交易，按 Z 从低到高填满空余名额
        free = slots - len(held)
        if free > 0 and t > 0 and allow[t]:
            cand = np.where((z[t] < -k) & (z[t - 1] >= -k) & up[t] & ok[t])[0]
            cand = [j for j in cand[np.argsort(z[t, cand])] if j not in held][:free]
            for j in cand:
                row[j] = 1.0 / slots
                held[j] = (t, px[t, j])
        if not np.isnan(row).all():
            rows[close.index[t]] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=close.columns)


MARKET = "sh.000300"


def market_regime(panel: dict, window: int | None) -> pd.Series | None:
    """大盘环境：沪深300 收盘 > MA(window)。window=None 表示不过滤。"""
    if not window:
        return None
    idx = panel["index_close"][MARKET]
    return idx > idx.rolling(window).mean()


def run(panel: dict, window: int = 20, k: float = 2.5, trend: int = 250, slots: int = 20,
        max_hold: int = 30, stop: float = 0.10, market: int | None = 250) -> dict:
    close, tradable, ct = panel["close"], panel["tradable"], bt.exec_mask(panel)
    w = generate_weights(close, tradable, ct, window, k, trend, slots, max_hold, stop, market_regime(panel, market))
    pf = bt.backtest_weights(panel, w, fill_missing=False)

    # 消融：去掉大盘过滤
    w_nt = generate_weights(close, tradable, ct, window, k, trend, slots, max_hold, stop, None)
    pf_nt = bt.backtest_weights(panel, w_nt, fill_missing=False)

    # 基础版（复用原实现）
    c, o = bt._exec_prices(panel)
    listed = close.loc[:bt.config.BACKTEST_START].iloc[-1].notna()
    cols = listed[listed].index
    sub = {**panel, "close": close[cols], "tradable": tradable[cols], "can_trade": ct[cols]}
    base = _combined_value(_run_one(sub, c[cols], o[cols], 20, 2.0))

    equity = pd.DataFrame({
        f"均值回归+ (大盘+个股趋势过滤/k={k}/{slots}名额)": pf.value(),
        "均值回归+ 无大盘过滤": pf_nt.value(),
        "基础版 BB(20,2.0)": base,
        bt.config.BENCHMARK_NAME: bt.benchmark_value(panel),
    })
    stats = equity.apply(bt.perf_stats).T
    stats["平均仓位"] = [
        1 - (pf.cash() / pf.value()).mean(), 1 - (pf_nt.cash() / pf_nt.value()).mean(), np.nan, 1.0]
    trades = pf.trades.records_readable

    # 参数网格：大盘过滤（0=不过滤） × 入场阈值 k
    grid = {}
    for mk in [0, 120, 250]:
        for kk in [2.0, 2.5, 3.0]:
            ww = generate_weights(close, tradable, ct, window, kk, trend, slots, max_hold, stop,
                                  market_regime(panel, mk))
            grid[(mk, kk)] = bt.perf_stats(bt.backtest_weights(panel, ww, fill_missing=False).value())
    grid = pd.DataFrame(grid).T
    grid.index.names = ["大盘MA窗口", "阈值k"]

    return {
        "description": __doc__,
        "params": {"window": window, "k": k, "trend": trend, "slots": slots, "max_hold": max_hold, "stop": stop,
                   "market": market},
        "equity": equity,
        "drawdown": equity.apply(bt.drawdown),
        "stats": stats,
        "grid": grid,
        "trades": trades,
        "weights": w,
        "weights_mode": "hold",
        "asset": "stock",
    }
