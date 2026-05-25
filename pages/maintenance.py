"""
ui/pages/maintenance.py — Trolley Maintenance KPI page.
Resolution time = time between consecutive createdOn entries per trolley.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from ui.components import (
    page_header, section_header, kpi_row, insight_cards,
    bar_chart, line_chart, donut_chart, data_table, no_data_placeholder,
)
from ui.style import PLOTLY_LAYOUT
from kpis.maintenance import (
    get_maintenance_data, get_maintenance_summary,
    get_reason_breakdown, get_zone_maintenance_breakdown, get_resolution_trend,
)
from utils.anomaly import generate_maintenance_insights


@st.cache_data(ttl=180, show_spinner=False)
def _load():
    return get_maintenance_data()


TYPE_COLORS = {"minor": "#3b82f6", "major": "#ef4444"}
STATUS_COLORS = {"open": "#f59e0b", "closed": "#10b981"}


def render():
    page_header(
        "Trolley Maintenance KPI",
        "Job tracking, SLA monitoring, and repeat-failure detection"
    )

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("⚙  Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_type = st.multiselect(
                "Job type", ["minor", "major"], default=["minor", "major"]
            )
        with fc2:
            filter_status = st.radio(
                "Job status", ["All", "Open Only", "Closed Only"], horizontal=True
            )
        with fc3:
            minor_sla = st.number_input("Minor SLA (hrs)", min_value=1, value=24)
            major_sla = st.number_input("Major SLA (hrs)", min_value=1, value=72)

    with st.spinner("Fetching maintenance data…"):
        df = _load()

    if df.empty:
        no_data_placeholder("No maintenance records found.")
        return

    # Apply filters
    if filter_type:
        df = df[df["type"].isin(filter_type)]
    if filter_status == "Open Only":
        df = df[df["is_open"]]
    elif filter_status == "Closed Only":
        df = df[~df["is_open"]]

    # Recompute SLA with slider values
    sla_map = {"minor": minor_sla, "major": major_sla}
    df["sla_hours"] = df["type"].map(sla_map).fillna(24)
    df["sla_breached"] = (
        ~df["is_open"] & (df["job_duration_hours"] > df["sla_hours"])
    )

    summary = get_maintenance_summary(df)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi_row([
        {"label": "Total Jobs",          "value": summary.get("total_jobs","—"),       "color":"#00d4aa"},
        {"label": "Open Jobs",           "value": summary.get("open_jobs","—"),
         "sub": f'<span class="warn">{summary.get("major_open","—")} major</span>',  "color":"#f59e0b"},
        {"label": "Closed Jobs",         "value": summary.get("closed_jobs","—"),
         "sub": '<span class="up">resolved</span>',                                   "color":"#10b981"},
        {"label": "SLA Breaches",        "value": summary.get("sla_breached","—"),
         "sub": '<span class="down">exceeded time limit</span>',                      "color":"#ef4444"},
        {"label": "Avg Resolution (hrs)","value": summary.get("avg_resolution_hrs","—"),
         "color":"#3b82f6"},
        {"label": "Repeat Trolleys (3+)","value": summary.get("repeat_trolleys","—"),
         "sub": '<span class="down">needs overhaul</span>',                           "color":"#a855f7"},
    ])

    # ── Charts ────────────────────────────────────────────────────────────────
    section_header("Maintenance Breakdown")
    c1, c2 = st.columns([3, 2])

    with c1:
        reason_df = get_reason_breakdown(df)
        if not reason_df.empty:
            fig = bar_chart(
                reason_df, x="reason_category", y="count", color="type",
                title="Jobs by Failure Category & Type",
                color_map=TYPE_COLORS,
                horizontal=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("Reason breakdown unavailable")

    with c2:
        if "type" in df.columns:
            type_counts = df["type"].value_counts().reset_index()
            type_counts.columns = ["type", "count"]
            fig = donut_chart(type_counts, "type", "count", "Minor vs Major", TYPE_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    # ── Resolution time distribution ──────────────────────────────────────────
    section_header("Resolution Time (Consecutive Entry Method)")
    st.caption(
        "Duration = time between consecutive maintenance records per trolley. "
        "Last record per trolley = open job."
    )

    closed_df = df.dropna(subset=["job_duration_hours"])
    if not closed_df.empty:
        fig = px.histogram(
            closed_df, x="job_duration_hours", color="type",
            color_discrete_map=TYPE_COLORS,
            nbins=20,
            title="Resolution Time Distribution (hrs)",
            labels={"job_duration_hours": "Hours to next record"},
            template="plotly_dark",
        )
        fig.add_vline(x=minor_sla, line_dash="dash", line_color="#3b82f6",
                      annotation_text=f"Minor SLA ({minor_sla}h)")
        fig.add_vline(x=major_sla, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Major SLA ({major_sla}h)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        no_data_placeholder("No closed job resolution times available")

    # ── Zone breakdown ─────────────────────────────────────────────────────────
    zone_df = get_zone_maintenance_breakdown(df)
    if not zone_df.empty:
        section_header("Maintenance by Zone")
        st.dataframe(
            zone_df.style.format({
                "avg_hours": "{:.1f}",
                "total_jobs": "{:.0f}",
                "open_jobs": "{:.0f}",
                "sla_breached": "{:.0f}",
            }),
            use_container_width=True, hide_index=True
        )

    # ── Trend ─────────────────────────────────────────────────────────────────
    trend_df = get_resolution_trend(df)
    if not trend_df.empty:
        section_header("Job Creation Trend")
        fig = line_chart(
            trend_df, x="date", y="count", color="type",
            title="Daily Maintenance Jobs", color_map=TYPE_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    section_header("AI-Generated Insights")
    insight_cards(generate_maintenance_insights(df))

    # ── Detail table ──────────────────────────────────────────────────────────
    section_header("Maintenance Records")
    display_cols = [c for c in [
        "maintenanceId", "trolleyId", "type", "reason_category", "reason",
        "is_open", "job_duration_hours", "sla_hours", "sla_breached",
        "createdOn", "geozoneName", "doneBy"
    ] if c in df.columns]

    tab1, tab2 = st.tabs(["All Records", "SLA Breaches"])
    with tab1:
        data_table(df[display_cols].sort_values("createdOn", ascending=False))
    with tab2:
        breaches = df[df["sla_breached"]][display_cols] if "sla_breached" in df.columns else pd.DataFrame()
        data_table(breaches)