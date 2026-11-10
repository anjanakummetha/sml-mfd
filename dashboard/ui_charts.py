"""Plotly figure builders for the backtester UI. All colors come from TOKENS (light/dark)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Validated categorical slots (dataviz reference palette) + chart chrome, per mode.
TOKENS = {
    "light": {
        "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "pos": "#2a78d6", "neg": "#e34948",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    },
    "dark": {
        "cat": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "pos": "#3987e5", "neg": "#e66767",
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    },
}
SEQ_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
            "#104281", "#0d366b"]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
LEG = {"ls": 0, "long": 1, "short": 2}          # fixed categorical slots, never re-ordered


def _base(fig, t, height=300, yfmt=".0%", title=None, legend=True, hover="x unified"):
    top = (34 if title else 8) + (26 if legend else 0)
    fig.update_layout(
        height=height, margin=dict(l=8, r=24, t=top, b=8),
        title=dict(text=title, x=0, xanchor="left", xref="paper", y=1, yref="container", yanchor="top",
                   pad=dict(t=8), font=dict(size=13, color=t["ink"])) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=t["ink2"]),
        showlegend=legend, legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
                                       font=dict(color=t["ink2"]), bgcolor="rgba(0,0,0,0)"),
        hovermode=hover, hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], tickfont=dict(color=t["muted"]), ticks="", automargin=True)
    fig.update_yaxes(gridcolor=t["grid"], gridwidth=1, zeroline=True, zerolinecolor=t["axis"], zerolinewidth=1,
                     tickformat=yfmt, tickfont=dict(color=t["muted"]), linecolor=t["axis"], automargin=True)
    return fig


def _end_label(fig, x, y, text, t):
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=6,
                       font=dict(size=11, color=t["ink2"]))


def cumulative(monthly: pd.DataFrame, t, log=False):
    fig = go.Figure()
    for key, col, name in [("ls", "strategy_ret", "Long − short"), ("long", "long_ret", "Long leg"),
                           ("short", "short_ret", "Short leg (return of stocks sold)")]:
        if col not in monthly:
            continue
        w = (1 + monthly[col].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=w.index, y=w, name=name, mode="lines",
                                 line=dict(color=t["cat"][LEG[key]], width=2), hovertemplate="%{y:.2f}"))
        _end_label(fig, w.index[-1], w.iloc[-1], f"{w.iloc[-1]:.2f}", t)
    fig.add_hline(y=1, line=dict(color=t["axis"], width=1))
    _base(fig, t, height=340, yfmt=".1f", title="Cumulative value of $1 (excess returns, compounded)")
    if log:
        fig.update_yaxes(type="log")
    return fig


def drawdown(monthly: pd.DataFrame, t):
    w = (1 + monthly["strategy_ret"].fillna(0)).cumprod()
    dd = w / w.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines", fill="tozeroy", name="Drawdown",
                               line=dict(color=t["neg"], width=2), fillcolor=_alpha(t["neg"], 0.10),
                               hovertemplate="%{y:.1%}"))
    return _base(fig, t, height=240, yfmt=".0%", title="Drawdown, long − short", legend=False)


def monthly_bars(monthly: pd.DataFrame, t, window=12):
    r = monthly["strategy_ret"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r.index, y=r, name="Monthly return",
                         marker=dict(color=np.where(r >= 0, t["pos"], t["neg"]), line=dict(width=0)),
                         hovertemplate="%{y:.2%}"))
    roll = r.rolling(window).mean()
    fig.add_trace(go.Scatter(x=roll.index, y=roll, name=f"{window}-month moving average", mode="lines",
                             line=dict(color=t["ink"], width=2), hovertemplate="%{y:.2%}"))
    _base(fig, t, height=240, yfmt=".0%", title="Monthly return, long − short")
    fig.update_layout(bargap=0.15)
    return fig


def bucket_means(table: pd.DataFrame, t):
    """table: index = bucket, columns mean, se (decimals)."""
    mu, se = table["mean"], table["se"]
    fig = go.Figure(go.Bar(
        x=mu.index, y=mu, error_y=dict(type="data", array=1.96 * se, color=t["muted"], thickness=1, width=4),
        marker=dict(color=np.where(mu >= 0, t["pos"], t["neg"]), line=dict(width=0)),
        customdata=mu / se, hovertemplate="Bucket %{x}: %{y:.2%} (t = %{customdata:.2f})<extra></extra>"))
    _base(fig, t, height=300, yfmt=".1%", title="Mean monthly excess return by bucket (95% CI)", legend=False,
          hover="closest")
    fig.update_xaxes(title=dict(text="Score bucket (1 = lowest)", font=dict(color=t["muted"])), dtick=1)
    fig.update_layout(bargap=0.35)
    return fig


def bucket_cumulative(buckets: pd.DataFrame, t):
    p = buckets.pivot(index="month", columns="bucket", values="ret").fillna(0)
    n = p.shape[1]
    colors = sample_colorscale([[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
                               [i / max(n - 1, 1) for i in range(n)])
    fig = go.Figure()
    for b, c in zip(p.columns, colors):
        w = (1 + p[b]).cumprod()
        fig.add_trace(go.Scatter(x=w.index, y=w, name=f"{b}", mode="lines", line=dict(color=c, width=2),
                                 hovertemplate=f"Bucket {b}: " + "%{y:.2f}<extra></extra>"))
        if b in (p.columns[0], p.columns[-1]):
            _end_label(fig, w.index[-1], w.iloc[-1], f"{b}", t)
    return _base(fig, t, height=300, yfmt=".1f", title="Cumulative value of $1 by bucket (darker = higher score)",
                 hover="closest")


def annual_bars(r: pd.Series, t):
    y = (1 + r.dropna()).groupby(r.dropna().index.year).prod() - 1
    fig = go.Figure(go.Bar(x=y.index, y=y, marker=dict(color=np.where(y >= 0, t["pos"], t["neg"]), line=dict(width=0)),
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    _base(fig, t, height=260, yfmt=".0%", title="Calendar-year return, long − short", legend=False, hover="closest")
    fig.update_xaxes(dtick=1)
    fig.update_layout(bargap=0.35)
    return fig


def rolling_sharpe(monthly: pd.DataFrame, t, window=24):
    r = monthly["strategy_ret"]
    rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(12)
    fig = go.Figure(go.Scatter(x=rs.index, y=rs, mode="lines", name="Rolling Sharpe",
                               line=dict(color=t["cat"][0], width=2), hovertemplate="%{y:.2f}"))
    return _base(fig, t, height=260, yfmt=".1f", title=f"Rolling {window}-month Sharpe ratio (annualized)",
                 legend=False)


def rank_ic(ic: pd.Series, t):
    ic = ic.dropna()
    fig = go.Figure(go.Bar(x=ic.index, y=ic, name="Rank IC",
                           marker=dict(color=np.where(ic >= 0, t["pos"], t["neg"]), line=dict(width=0)),
                           hovertemplate="%{y:.3f}<extra>Rank IC</extra>"))
    fig.add_hline(y=ic.mean(), line=dict(color=t["ink"], width=1))
    _base(fig, t, height=260, yfmt=".2f", title="Monthly rank IC (Spearman, score vs. next-month return)",
          legend=False)
    fig.update_layout(bargap=0.15)
    return fig


def lines(series: list[tuple[pd.Series, str, int]], title, t, yfmt=",d", legend=True):
    """series: (values, name, categorical slot)."""
    fig = go.Figure()
    for s, name, slot in series:
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines", line=dict(color=t["cat"][slot], width=2)))
    return _base(fig, t, height=260, yfmt=yfmt, title=title, legend=legend)


def compare(runs: list[dict], t, log=False):
    fig = go.Figure()
    for r in runs:
        w = (1 + r["returns"].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=w.index, y=w, name=r["label"], mode="lines",
                                 line=dict(color=t["cat"][r["slot"]], width=2), hovertemplate="%{y:.2f}"))
    fig.add_hline(y=1, line=dict(color=t["axis"], width=1))
    _base(fig, t, height=360, yfmt=".1f", title="Cumulative value of $1, long − short")
    if log:
        fig.update_yaxes(type="log")
    return fig


def _alpha(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"
