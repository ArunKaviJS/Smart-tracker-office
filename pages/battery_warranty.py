"""
ui/pages/battery_warranty.py — Battery Warranty KPI page.
"""

import streamlit as st
import plotly.express as px
from ui.components import (
    page_header, section_header, kpi_row, insight_cards,
    bar_chart, donut_chart, line_chart, data_table, no_data_placeholder,
)
from ui.style import PLOTLY_LAYOUT, STATUS_KPI_COLORS
from kpis.battery_warranty import (
    get_battery_warranty_data, get_device_fault_data,
    get_battery_summary, get_warranty_status_breakdown, get_expiry_timeline,
)
from utils.anomaly import generate_battery_insights


WARRANTY_COLORS = {
    "Expired":  "#ef4444",
    "Critical": "#f97316",
    "Warning":  "#f59e0b",
    "Healthy":  "#10b981",
    "Unknown":  "#566779",
}


@st.cache_data(ttl=300, show_spinner=False)
def _load():
    batt = get_battery_warranty_data()
    faults = get_device_fault_data()
    return batt, faults


def render():
    page_header(
        "Battery Warranty KPI",
        "Warranty expiry tracking, fault detection, and claim prioritisation"
    )

    with st.spinner("Loading battery data…"):
        df, fault_df = _load()

    if df.empty:
        no_data_placeholder("No battery records found in tbl_battery.")
        return

    summary = get_battery_summary(df)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi_row([
        {"label": "Total Batteries",  "value": summary.get("total_batteries","—"), "color":"#00d4aa"},
        {"label": "Expired",          "value": summary.get("expired","—"),
         "sub": '<span class="down">warranty lapsed</span>',                      "color":"#ef4444"},
        {"label": "Expiring ≤30 days","value": summary.get("critical","—"),
         "sub": '<span class="down">file claims now</span>',                      "color":"#f97316"},
        {"label": "Expiring ≤90 days","value": summary.get("warning","—"),
         "sub": '<span class="warn">schedule review</span>',                      "color":"#f59e0b"},
        {"label": "Healthy",          "value": summary.get("healthy","—"),
         "sub": '<span class="up">&gt;90 days left</span>',                       "color":"#10b981"},
        {"label": "Avg Days Left",    "value": int(summary.get("avg_days_left",0)),"color":"#3b82f6"},
    ])

    # ── Charts row ────────────────────────────────────────────────────────────
    section_header("Warranty Status Overview")
    c1, c2 = st.columns([2, 3])

    with c1:
        status_df = get_warranty_status_breakdown(df)
        if not status_df.empty:
            fig = donut_chart(
                status_df, "warranty_status", "count",
                "Warranty Status Split", WARRANTY_COLORS
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder()

    with c2:
        timeline_df = get_expiry_timeline(df)
        if not timeline_df.empty:
            fig = bar_chart(
                timeline_df, x="expiry_month", y="count",
                title="Batteries Expiring by Month",
            )
            fig.update_traces(marker_color="#f97316", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("Expiry timeline unavailable")

    # ── Days remaining histogram ───────────────────────────────────────────────
    section_header("Days Remaining Distribution")
    fig = px.histogram(
        df, x="days_remaining",
        color="warranty_status",
        color_discrete_map=WARRANTY_COLORS,
        nbins=24,
        title="Distribution of Warranty Days Remaining",
        labels={"days_remaining": "Days until expiry"},
        template="plotly_dark",
    )
    fig.add_vline(x=0,  line_dash="dash", line_color="#ef4444", annotation_text="Expiry")
    fig.add_vline(x=30, line_dash="dot",  line_color="#f97316", annotation_text="30-day mark")
    fig.add_vline(x=90, line_dash="dot",  line_color="#f59e0b", annotation_text="90-day mark")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # ── Device fault summary (from device_data) ────────────────────────────────
    if not fault_df.empty:
        section_header("Device Battery Fault Summary")
        st.caption("Fault registers parsed from tbl_device_data batteryFault column")
        fault_active = fault_df[fault_df["has_fault"]]
        if not fault_active.empty:
            st.error(f"⚠ {len(fault_active)} devices reporting active battery faults")
            data_table(fault_active[[c for c in [
                "deviceEndpoint", "avg_battery_level", "min_battery_level",
                "has_fault", "last_fault", "last_seen"
            ] if c in fault_active.columns]])
        else:
            st.success("✓ No active battery fault codes detected across all devices")

    # ── AI Insights ───────────────────────────────────────────────────────────
    section_header("AI-Generated Insights")
    insight_cards(generate_battery_insights(df))

    # ── Detail table ──────────────────────────────────────────────────────────
    section_header("Battery Records")
    display_cols = [c for c in [
        "batteryId", "serialNo", "warrantyStartDate", "warrantyExpiryDate",
        "warrantyDuration", "days_remaining", "warranty_status", "duration_mismatch"
    ] if c in df.columns]

    tab1, tab2, tab3 = st.tabs(["All Batteries", "Expired & Critical", "Date Mismatches"])
    with tab1:
        data_table(df[display_cols].sort_values("days_remaining"))
    with tab2:
        urgent = df[df["warranty_status"].isin(["Expired", "Critical"])]
        data_table(urgent[display_cols].sort_values("days_remaining") if not urgent.empty else urgent)
    with tab3:
        if "duration_mismatch" in df.columns:
            mismatched = df[df["duration_mismatch"]]
            data_table(mismatched[display_cols] if not mismatched.empty else mismatched)
        else:
            st.info("Duration mismatch detection not available.")