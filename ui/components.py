"""
ui/components.py — Reusable Streamlit widgets used across all pages.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from ui.style import (
    SEVERITY_COLORS, STATUS_KPI_COLORS, PLOTLY_LAYOUT
)


# ── Page chrome ──────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-head">{text}</div>', unsafe_allow_html=True)


# ── KPI metric cards ──────────────────────────────────────────────────────────

def kpi_card(label: str, value, sub: str = "", color: str = "#00d4aa") -> str:
    """Return HTML for a single KPI card."""
    return f"""
    <div class="kpi-card" style="--kpi-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def kpi_row(cards: list):
    """
    Render a responsive row of KPI cards.
    cards = list of dicts with keys: label, value, sub (opt), color (opt)
    """
    html = '<div class="kpi-grid">'
    for c in cards:
        html += kpi_card(
            label=c.get("label", ""),
            value=c.get("value", "—"),
            sub=c.get("sub", ""),
            color=c.get("color", "#00d4aa"),
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Insight cards ─────────────────────────────────────────────────────────────

def insight_cards(insights: list):
    """Render a list of insight dicts as styled cards."""
    if not insights:
        st.markdown(
            '<div class="insight-card" style="--ic-color:#10b981">'
            '<div class="ic-title">✓ No anomalies detected</div>'
            '<div class="ic-detail">All KPIs are within normal ranges.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    icons = {"critical": "⬤", "warning": "◆", "info": "◉", "success": "✓"}
    for ins in insights:
        sev   = ins.get("severity", "info")
        color = SEVERITY_COLORS.get(sev, "#3b82f6")
        icon  = icons.get(sev, "•")
        st.markdown(
            f'<div class="insight-card" style="--ic-color:{color}">'
            f'<div class="ic-title">{icon} {ins["title"]}</div>'
            f'<div class="ic-detail">{ins.get("detail","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Charts ────────────────────────────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    title: str = "",
    color_map: dict = None,
    horizontal: bool = False,
) -> go.Figure:
    kwargs = dict(
        data_frame=df, x=x, y=y, title=title,
        color_discrete_map=color_map or STATUS_KPI_COLORS,
        template="plotly_dark",
    )
    if color:
        kwargs["color"] = color
    if horizontal:
        kwargs["orientation"] = "h"
        kwargs["x"], kwargs["y"] = kwargs["y"], kwargs["x"]

    fig = px.bar(**kwargs)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return fig


def donut_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
    color_map: dict = None,
) -> go.Figure:
    fig = px.pie(
        df, names=names, values=values,
        hole=0.6, title=title,
        color=names,
        color_discrete_map=color_map or STATUS_KPI_COLORS,
        template="plotly_dark",
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=10,
        marker=dict(line=dict(color="#0d1117", width=2)),
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    title: str = "",
    color_map: dict = None,
) -> go.Figure:
    kwargs = dict(
        data_frame=df, x=x, y=y, title=title,
        color_discrete_map=color_map or {},
        markers=True,
        template="plotly_dark",
    )
    if color:
        kwargs["color"] = color

    fig = px.line(**kwargs)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(line_width=2, marker_size=5)
    return fig


def status_badge(status: str) -> str:
    cls_map = {
        "Critical Idle":    "badge-critical",
        "Idle":             "badge-warning",
        "Soft Idle":        "badge-info",
        "Active":           "badge-success",
        "Expired":          "badge-critical",
        "Critical":         "badge-critical",
        "Warning":          "badge-warning",
        "Healthy":          "badge-success",
        "Critical (>48h)":  "badge-critical",
        "Warning (24–48h)": "badge-warning",
        "Online":           "badge-success",
        "open":             "badge-warning",
        "closed":           "badge-success",
    }
    cls = cls_map.get(status, "badge-neutral")
    return f'<span class="badge {cls}">{status}</span>'


def data_table(df: pd.DataFrame, max_rows: int = 100):
    """Render a styled dataframe. Limits rows for performance."""
    if df.empty:
        st.info("No data to display.")
        return
    show = df.head(max_rows)
    st.dataframe(show, use_container_width=True, hide_index=True)


def no_data_placeholder(message: str = "No data available for the selected period."):
    st.markdown(
        f"""
        <div style="text-align:center;padding:48px 0;color:#3d5a73;
                    border:1px dashed #1e2d40;border-radius:10px;margin:12px 0">
            <div style="font-size:32px;margin-bottom:8px">◻</div>
            <div style="font-size:13px">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )