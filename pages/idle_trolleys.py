"""
ui/pages/idle_trolleys.py — Full KPI page for idle trolley analysis.
"""

import streamlit as st
import pandas as pd
from ui.components import (
    page_header, section_header, kpi_row, insight_cards,
    bar_chart, donut_chart, data_table, no_data_placeholder,
)
from ui.style import STATUS_KPI_COLORS
from kpis.idle_trolleys import (
    get_idle_trolley_data, get_idle_summary, get_zone_idle_breakdown,
)
from utils.anomaly import generate_idle_insights


IDLE_COLOR_MAP = {
    "Active":       "#10b981",
    "Soft Idle":    "#3b82f6",
    "Idle":         "#f59e0b",
    "Critical Idle":"#ef4444",
}


@st.cache_data(ttl=180, show_spinner=False)
def _load():
    return get_idle_trolley_data()


def render():
    page_header(
        "Idle Trolley Analysis",
        "Detects trolleys that have not moved within configurable thresholds"
    )

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("⚙  Filters", expanded=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            threshold = st.slider(
                "Idle threshold (minutes)", 15, 240, 60, step=15,
                help="Trolleys idle longer than this are flagged"
            )
        with fcol2:
            selected_statuses = st.multiselect(
                "Show statuses",
                ["Active", "Soft Idle", "Idle", "Critical Idle"],
                default=["Soft Idle", "Idle", "Critical Idle"],
            )

    with st.spinner("Fetching trolley data…"):
        df = _load()

    if df.empty:
        no_data_placeholder("No device data found. Check DB connection or table contents.")
        return

    # Re-apply threshold from slider
    df["idle_status"] = df["idle_minutes"].apply(
        lambda m: (
            "Active"       if m is None or m < 30 else
            "Soft Idle"    if m < threshold else
            "Idle"         if m < threshold * 2 else
            "Critical Idle"
        )
    )

    filtered = df[df["idle_status"].isin(selected_statuses)] if selected_statuses else df
    summary  = get_idle_summary(filtered)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi_row([
        {"label": "Total Trolleys",   "value": summary.get("total_trolleys","—"),   "color":"#00d4aa"},
        {"label": "Critical Idle",    "value": summary.get("critical_idle","—"),
         "sub": '<span class="down">≥120 min</span>',                              "color":"#ef4444"},
        {"label": "Idle",             "value": summary.get("idle","—"),
         "sub": '<span class="warn">60–120 min</span>',                            "color":"#f59e0b"},
        {"label": "Active Trolleys",  "value": summary.get("active","—"),
         "sub": '<span class="up">moved recently</span>',                          "color":"#10b981"},
        {"label": "Avg Idle (min)",   "value": summary.get("avg_idle_minutes","—"),"color":"#3b82f6"},
        {"label": "Low Battery",      "value": summary.get("low_battery","—"),
         "sub": '<span class="down">&lt;20% charge</span>',                        "color":"#a855f7"},
    ])

    # ── Charts row ────────────────────────────────────────────────────────────
    section_header("Idle Distribution")
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        zone_df = get_zone_idle_breakdown(filtered)
        if not zone_df.empty:
            fig = bar_chart(
                zone_df, x="geozoneName", y="count", color="idle_status",
                title="Idle Trolleys by Zone & Status",
                color_map=IDLE_COLOR_MAP,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("Zone breakdown unavailable")

    with ch2:
        if "idle_status" in filtered.columns:
            status_counts = filtered["idle_status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig = donut_chart(
                status_counts, "status", "count",
                "Status Breakdown", IDLE_COLOR_MAP
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder()

    # ── Battery vs idle scatter ───────────────────────────────────────────────
    if "batteryLevel" in filtered.columns and "idle_minutes" in filtered.columns:
        section_header("Battery Level vs Idle Duration")
        import plotly.express as px
        from ui.style import PLOTLY_LAYOUT
        fig = px.scatter(
            filtered.dropna(subset=["idle_minutes", "batteryLevel"]),
            x="idle_minutes", y="batteryLevel",
            color="idle_status",
            color_discrete_map=IDLE_COLOR_MAP,
            hover_data=["deviceEndpoint", "geozoneName"] if "geozoneName" in filtered.columns else ["deviceEndpoint"],
            title="Battery Level vs Idle Duration",
            labels={"idle_minutes": "Idle Duration (min)", "batteryLevel": "Battery Level (%)"},
            template="plotly_dark",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    section_header("AI-Generated Insights")
    insight_cards(generate_idle_insights(filtered))

    # ── Detail table ──────────────────────────────────────────────────────────
    section_header("Trolley Detail Table")
    display_cols = [c for c in [
        "deviceEndpoint", "trolleyId", "idle_status", "idle_minutes",
        "batteryLevel", "has_fault", "geozoneName", "locationSource", "last_seen"
    ] if c in filtered.columns]

    tab1, tab2 = st.tabs(["All Trolleys", "Critical & Idle Only"])
    with tab1:
        data_table(filtered[display_cols].sort_values(
            "idle_minutes", ascending=False, na_position="last"
        ) if "idle_minutes" in filtered.columns else filtered[display_cols])

    with tab2:
        critical = filtered[filtered["idle_status"].isin(["Critical Idle", "Idle"])]
        data_table(critical[display_cols].sort_values(
            "idle_minutes", ascending=False, na_position="last"
        ) if not critical.empty else critical)