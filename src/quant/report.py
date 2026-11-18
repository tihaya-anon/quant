"""把 data/results/<股票池>/ 下的回测结果汇总成一个离线 HTML 报表（plotly，悬停查看数值）。

用法：uv run python -m quant.report        → reports/<股票池>.html
"""

from __future__ import annotations

import html
import warnings

import pandas as pd
import plotly.graph_objects as go

from quant import backtest as vb
from quant import config
from quant import viz

STRATS = [("momentum", "momentum_plus", "动量"), ("mean_reversion", "mean_reversion_plus", "均值回归"),
          ("trend", "trend_plus", "趋势"), ("factor", "factor_plus", "因子")]


def _load(name: str) -> dict | None:
    try:
        return vb.load_results(name)
    except FileNotFoundError:
        return None


def _line(df: pd.DataFrame, title: str, log: bool = False, fmt: str = ".2f", **kw) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        name = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
        fig.add_scatter(x=df.index, y=df[c], name=name, line=dict(color=viz.color_of(name, i)),
                        hovertemplate=f"%{{y:{fmt}}}")
    fig.update_layout(title=title, **kw)
    if log:
        fig.update_yaxes(type="log")
    return fig


def execution_figs(o: dict) -> list:
    out: list = [f"<h2>执行仿真（backtrader，资金 {o['base_cash'] / 1e4:.0f} 万）</h2>",
                 "<p>对账差异主要来自停牌/涨跌停/整手约束（vectorbt 不建模）。"
                 "大资金档位里参与率上限会让部分订单放弃成交，成本被封顶，收益可能反而略好——这是模型假象。</p>",
                 _line(o["reconcile"], "对账：零摩擦 backtrader vs 零成本 vectorbt", log=True),
                 viz.stats_html(o["real_stats"]),
                 _line(o["real_equity"], "事件驱动执行 vs 向量化（固定 0.2%）", log=True, height=520)]

    bp = [c for c in o["real_stats"].columns if c.endswith("(年化bp)") and not c.startswith("总")]
    fig = go.Figure()
    for c in bp:
        k = c.replace("(年化bp)", "")
        fig.add_bar(x=o["real_stats"].index, y=o["real_stats"][c], name=k, marker_color=viz.COST_COLORS.get(k),
                    hovertemplate="%{y:.0f} bp")
    fig.update_layout(title="成本拆分（年化 bp）", barmode="stack", hovermode="x unified")
    out.append(fig)

    cap = o["capacity"]["年化收益"].unstack(0)
    fig = _line(cap, "资金容量：年化收益 vs 资金规模", fmt=".2%")
    fig.update_traces(mode="lines+markers")
    fig.update_xaxes(type="log", title="资金（元）")
    fig.update_yaxes(tickformat=".0%")
    out.append(fig)

    out.append(viz.plot_grid(o["sensitivity"], "年化收益",
                             f"敏感性：参与率上限 × 冲击系数（动量+，{o['sensitivity_cash'] / 1e8:.0f} 亿）", fmt=".2%"))

    tv = o["turnover_value"][["年化收益", "总成本(年化bp)"]].unstack(0)
    fig = go.Figure()
    for i, s in enumerate(tv["年化收益"].columns):
        fig.add_scatter(x=tv.index, y=tv["年化收益"][s], name=s, mode="lines+markers",
                        customdata=tv["总成本(年化bp)"][s], line=dict(color=viz.SERIES[i]),
                        hovertemplate="年化 %{y:.2%}<br>成本 %{customdata:.0f} bp")
    fig.update_layout(title="换手控制的价值：基础版动量 vs 动量+")
    fig.update_xaxes(type="log", title="资金（元）")
    fig.update_yaxes(tickformat=".0%")
    out.append(fig)
    return out


def build() -> list:
    parts: list = []
    for base, plus, label in STRATS:
        rb, rp = _load(base), _load(plus)
        if rb is None and rp is None:
            continue
        parts.append(f"<h2>{label}</h2>")
        for r in (rp, rb):
            if r is None:
                continue
            parts.append(f"<p>{html.escape(r.get('description', ''))}</p>")
            parts.append(viz.stats_html(r["stats"]))
            parts.append(viz.plot_equity(r, r["stats"].index[0]))
            g = r.get("grid")
            if isinstance(g, pd.DataFrame) and g.index.nlevels == 2 and "夏普比率" in g:
                parts.append(viz.plot_grid(g, "夏普比率", f"{r['stats'].index[0]} 参数网格（夏普）"))
        if rp is not None and "factor_weights" in rp:
            parts.append(_line(rp["factor_weights"], "因子+ 滚动 IC 权重", fmt=".2f"))
    ex = _load("execution")
    if ex is not None:
        parts += execution_figs(ex)
    return parts


CSS = """body{font-family:system-ui,-apple-system,'Microsoft YaHei',sans-serif;max-width:1200px;margin:24px auto;
padding:0 16px;color:#0b0b0b;background:#fcfcfb}h2{margin-top:48px;border-bottom:1px solid #e1e0d9}
table{border-collapse:collapse;font-size:13px;margin:12px 0}td,th{padding:4px 10px;border-bottom:1px solid #e1e0d9;
text-align:right}th:first-child,td:first-child{text-align:left}p{color:#52514e}"""


def main() -> None:
    warnings.filterwarnings("ignore")
    parts = build()
    body, first = [], True
    for p in parts:
        if isinstance(p, go.Figure):
            body.append(p.to_html(full_html=False, include_plotlyjs=first))
            first = False
        else:
            body.append(p)
    out = config.ROOT / "reports" / f"{config.UNIVERSE}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>回测报表 {config.UNIVERSE}</title>"
                   f"<style>{CSS}</style></head><body><h1>回测报表 · {config.UNIVERSE}</h1>"
                   f"{''.join(body)}</body></html>", encoding="utf-8")
    print(f"报表已生成: {out}")


if __name__ == "__main__":
    main()
