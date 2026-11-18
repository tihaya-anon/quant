"""plotly 样式、结果读取、绘图与 HTML 排版，供 report/*.qmd（Quarto）使用。

- 绘图函数返回 plotly Figure；输出时统一转成 HTML 片段（plotly.js 由页面头部加载一次，
  不走 notebook 渲染器——它会注入 require.js，导致 MathJax 加载失败）。
- 颜色按序列名登记：同一序列在同一页的所有图里颜色一致（左右对比时尤其重要）。
- compare() 自己排双栏：标题、图、表按行对齐，不依赖 Quarto 的响应式布局。
"""

from __future__ import annotations

import html as _html

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML
from plotly.subplots import make_subplots

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCH = "#898781"
BENCH2 = "#c3c2b7"  # 股票池等权，与指数区分
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
DIVERGING = [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]]
COST_COLORS = {"佣金": "#86b6ef", "过户费": "#c3c2b7", "印花税": "#2a78d6", "价差": "#eda100", "冲击": "#e34948"}

pio.templates["quant"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif", color=INK2, size=12),
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", colorway=SERIES,
    xaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zerolinecolor="#c3c2b7"),
    # 图例放在图下方，避免和标题挤在一起
    legend=dict(orientation="h", y=-0.12, yanchor="top", x=0, xanchor="left"),
    title=dict(x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=14, color=INK)),
    hovermode="x unified", margin=dict(l=55, r=15, t=40, b=20), height=420,
))
pio.templates.default = "quant"

PCT_COLS = ["累计收益", "年化收益", "年化波动", "最大回撤", "日胜率", "平均仓位", "月均单边换手", "平均参与率"]
BENCH_WORDS = ("指数", "买入持有", "股票池等权")


# ---------------------------------------------------------------- 数据

def load(universe: str, name: str) -> dict:
    """读取 data/results/<universe>/<name>.pkl，不依赖 QUANT_UNIVERSE，方便同页对比多个股票池。"""
    from quant import config
    return pd.read_pickle(config.DATA_DIR / "results" / universe / f"{name}.pkl")


# ---------------------------------------------------------------- 颜色

_REGISTRY: dict[str, str] = {}


def _key(name: str) -> str:
    """'基础版 动量 Top10' 与 '动量 Top10' 视为同一序列。"""
    return name.replace("基础版 ", "").strip()


def is_bench(name: str) -> bool:
    return any(w in name for w in BENCH_WORDS)


def color_of(name: str, i: int = 0) -> str:
    if is_bench(name):
        return BENCH2 if "股票池等权" in name else BENCH
    k = _key(name)
    if k not in _REGISTRY:
        _REGISTRY[k] = SERIES[len(_REGISTRY) % len(SERIES)]
    return _REGISTRY[k]


def _dash(name: str):
    if "股票池等权" in name:
        return "dash"
    return "dot" if is_bench(name) else None


# ---------------------------------------------------------------- 表格

def show_stats(stats: pd.DataFrame, cols: list[str] | None = None):
    stats = stats if cols is None else stats[[c for c in cols if c in stats.columns]]
    fmt = {c: "{:.1%}" for c in stats.columns if c in PCT_COLS}
    fmt.update({c: "{:.2f}" for c in stats.columns if c not in PCT_COLS})
    fmt.update({c: "{:.0f}" for c in stats.columns if "bp" in c or "次数" in c or "订单" in c})
    return stats.style.format(fmt, na_rep="–")  # type: ignore


# ---------------------------------------------------------------- 图

def plot_equity(res: dict, title: str = "", cols: list[str] | None = None, height: int = 520):
    """净值（对数）+ 回撤。第一列加粗，基准灰色虚线。"""
    eq = res["equity"] if cols is None else res["equity"][cols]
    eq = eq / eq.bfill().iloc[0]
    dd = eq / eq.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    for i, col in enumerate(eq.columns):
        c, dash = color_of(col), _dash(col)
        w = 2.4 if i == 0 else 1.4
        fig.add_scatter(x=eq.index, y=eq[col], name=col, line=dict(color=c, width=w, dash=dash), row=1, col=1,
                        legendgroup=col, hovertemplate="%{y:.2f}")
        fig.add_scatter(x=dd.index, y=dd[col], name=col, line=dict(color=c, width=1.1, dash=dash), row=2, col=1,
                        legendgroup=col, showlegend=False, hovertemplate="%{y:.1%}")
    fig.update_yaxes(type="log", title_text="净值（对数）", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="回撤", row=2, col=1)
    fig.update_layout(title=title, height=height, legend_y=-0.06)
    return fig


def plot_grid(grid: pd.DataFrame, metric: str, title: str = "", fmt: str = ".2f", lim: float | None = None):
    """参数网格热力图。lim 指定色标范围，左右对比时传同一个值，颜色才可比。"""
    m = grid[metric].unstack()
    lim = lim or max(abs(m.min().min()), abs(m.max().max()))
    fig = go.Figure(go.Heatmap(
        z=m.values, x=[str(c) for c in m.columns], y=[str(i) for i in m.index], colorscale=DIVERGING,
        zmin=-lim, zmax=lim, text=m.map(lambda v: "" if pd.isna(v) else format(v, fmt)).values,
        texttemplate="%{text}", xgap=2, ygap=2, colorbar=dict(title=metric, thickness=12),
        hovertemplate=f"{m.index.name}=%{{y}}<br>{m.columns.name}=%{{x}}<br>{metric}=%{{z:{fmt}}}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=m.columns.name, yaxis_title=m.index.name, hovermode="closest",
                      height=320, margin_b=50,
                      xaxis=dict(type="category", showgrid=False), yaxis=dict(type="category", showgrid=False))
    return fig


def plot_yearly(equity: pd.DataFrame, title: str = ""):
    """各年收益分组柱状图。"""
    y = equity.resample("YE").last()
    ret = (y / y.shift(1) - 1).iloc[1:]
    ret.index = ret.index.year.astype(str)
    fig = go.Figure()
    for c in ret.columns:
        fig.add_bar(x=ret.index, y=ret[c], name=c, marker_color=color_of(c), hovertemplate="%{y:.1%}")
    fig.update_layout(title=title, barmode="group", yaxis_tickformat=".0%", height=380)
    return fig


def plot_lines(df: pd.DataFrame, title: str = "", fmt: str = ".2f", height: int = 360, **layout):
    """通用多序列折线（列 = 序列）。"""
    fig = go.Figure()
    for c in df.columns:
        name = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
        fig.add_scatter(x=df.index, y=df[c], name=name, line=dict(color=color_of(name), dash=_dash(name)),
                        hovertemplate=f"%{{y:{fmt}}}")
    fig.update_layout(title=title, height=height, **layout)
    return fig


# ---------------------------------------------------------------- HTML 输出

def _frag(item) -> str:
    if item is None:
        return ""
    if isinstance(item, go.Figure):
        return item.to_html(full_html=False, include_plotlyjs=False,
                            config={"displaylogo": False, "responsive": True})
    if isinstance(item, pd.io.formats.style.Styler):
        return item.to_html()
    if isinstance(item, pd.DataFrame):
        return item.to_html()
    return str(item)  # 已是 HTML 字符串


def show(item) -> HTML:
    """单个图/表输出。"""
    return HTML(f'<div class="q-single">{_frag(item)}</div>')


def compare(left: tuple[str, list], right: tuple[str, list]) -> HTML:
    """双栏对比：(标题, [图或表, ...])。两侧第 k 个元素放在同一行，保证对齐。"""
    (lt, li), (rt, ri) = left, right
    n = max(len(li), len(ri))
    li, ri = li + [None] * (n - len(li)), ri + [None] * (n - len(ri))
    cells = [f'<div class="q-head">{_html.escape(lt)}</div>', f'<div class="q-head">{_html.escape(rt)}</div>']
    for a, b in zip(li, ri):
        cells += [f'<div class="q-cell">{_frag(a)}</div>', f'<div class="q-cell">{_frag(b)}</div>']
    return HTML(f'<div class="q-compare">{"".join(cells)}</div>')
