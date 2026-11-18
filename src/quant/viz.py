"""plotly 样式与绘图函数，供 quant.report 生成 HTML 报表。各函数返回 Figure。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCH = "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
COST_COLORS = {"佣金": "#86b6ef", "过户费": "#c3c2b7", "印花税": "#2a78d6", "价差": "#eda100", "冲击": "#e34948"}

pio.templates["quant"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif", color=INK2, size=12),
    paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", colorway=SERIES,
    xaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zerolinecolor="#c3c2b7"),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
    hovermode="x unified", margin=dict(l=60, r=20, t=70, b=40), height=420,
))
pio.templates.default = "quant"

PCT_COLS = ["累计收益", "年化收益", "年化波动", "最大回撤", "日胜率", "平均仓位", "月均单边换手", "平均参与率"]
BENCH_WORDS = ("指数", "买入持有", "股票池等权")


def stats_html(stats: pd.DataFrame) -> str:
    fmt = {c: "{:.1%}" for c in stats.columns if c in PCT_COLS}
    fmt.update({c: "{:.2f}" for c in stats.columns if c not in PCT_COLS})
    fmt.update({c: "{:.0f}" for c in stats.columns if "bp" in c or "次数" in c or "订单" in c})
    return stats.style.format(fmt, na_rep="–").to_html()  # type: ignore


def color_of(name: str, i: int) -> str:
    return BENCH if any(w in name for w in BENCH_WORDS) else SERIES[i % len(SERIES)]


def plot_equity(res: dict, title: str, cols: list[str] | None = None):
    """净值（对数）+ 回撤。第一列为主策略加粗，其它策略细线，基准灰色虚线。"""
    eq = res["equity"] if cols is None else res["equity"][cols]
    eq = eq / eq.iloc[0]
    dd = eq / eq.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    for i, col in enumerate(eq.columns):
        c = color_of(col, i)
        dash = "dot" if c == BENCH else None
        w = 2.5 if i == 0 else 1.5
        fig.add_scatter(x=eq.index, y=eq[col], name=col, line=dict(color=c, width=w, dash=dash), row=1, col=1,
                        legendgroup=col, hovertemplate="%{y:.2f}")
        fig.add_scatter(x=dd.index, y=dd[col], name=col, line=dict(color=c, width=1.2, dash=dash), row=2, col=1,
                        legendgroup=col, showlegend=False, hovertemplate="%{y:.1%}")
    fig.update_yaxes(type="log", title_text="净值（对数）", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="回撤", row=2, col=1)
    fig.update_layout(title=title, height=600)
    return fig


def plot_grid(grid: pd.DataFrame, metric: str, title: str, fmt: str = ".2f"):
    m = grid[metric].unstack()
    lim = max(abs(m.min().min()), abs(m.max().max()))
    fig = go.Figure(go.Heatmap(
        z=m.values, x=[str(c) for c in m.columns], y=[str(i) for i in m.index], colorscale=DIVERGING,
        zmin=-lim, zmax=lim, text=m.map(lambda v: "" if pd.isna(v) else format(v, fmt)).values,
        texttemplate="%{text}", xgap=2, ygap=2, colorbar=dict(title=metric),
        hovertemplate=f"{m.index.name}=%{{y}}<br>{m.columns.name}=%{{x}}<br>{metric}=%{{z:{fmt}}}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=m.columns.name, yaxis_title=m.index.name, hovermode="closest",
                      height=380, xaxis=dict(type="category", showgrid=False), yaxis=dict(type="category", showgrid=False))
    return fig
