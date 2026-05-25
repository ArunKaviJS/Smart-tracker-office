"""
ui/pages/overview.py — Executive summary page showing all 4 KPIs at a glance.
"""

import streamlit as st
import plotly.graph_objects as go
from ui.components import (
    page_header, section_header, kpi_row, insight_cards,
    donut_chart, no_data_placeholder,
)
from ui.style import STATUS_KPI_COLORS, PLOTLY_LAYOUT
from kpis.idle_trolleys   import get_idle_trolley_data, get_idle_summary
from kpis.maintenance     import get_maintenance_data, get_maintenance_summary
from kpis.battery_warranty  import get_battery_warranty_data, get_battery_summary
from kpis.tracker_warranty  import get_tracker_data, get_tracker_summary
from utils.anomaly import (
    generate_idle_insights, generate_maintenance_insights,
    generate_battery_insights, generate_tracker_insights,
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_all():
    idle_df  = get_idle_trolley_data()
    maint_df = get_maintenance_data()
    batt_df  = get_battery_warranty_data()
    track_df = get_tracker_data()
    return idle_df, maint_df, batt_df, track_df


def render():
    page_header(
        "KPI Command Centre",
        "Real-time operational insights across trolleys, maintenance, batteries, and trackers"
    )

    with st.spinner("Loading all KPIs…"):
        idle_df, maint_df, batt_df, track_df = _load_all()

    i_sum = get_idle_summary(idle_df)
    m_sum = get_maintenance_summary(maint_df)
    b_sum = get_battery_summary(batt_df)
    t_sum = get_tracker_summary(track_df)

    # ── Top-level KPI strip ──────────────────────────────────────────────────
    section_header("Fleet Overview")
    kpi_row([
        {
            "label": "Total Trolleys",
            "value": i_sum.get("total_trolleys", "—"),
            "sub":   f'<span class="up">{i_sum.get("active","—")} active</span>',
            "color": "#00d4aa",
        },
        {
            "label": "Critically Idle",
            "value": i_sum.get("critical_idle", "—"),
            "sub":   f'<span class="down">&gt;120 min stationary</span>',
            "color": "#ef4444",
        },
        {
            "label": "Open Maintenance",
            "value": m_sum.get("open_jobs", "—"),
            "sub":   f'<span class="warn">{m_sum.get("major_open","—")} major jobs</span>',
            "color": "#f59e0b",
        },
        {
            "label": "Warranties Expiring",
            "value": b_sum.get("critical", 0) + b_sum.get("expired", 0),
            "sub":   f'<span class="down">{b_sum.get("expired","—")} already expired</span>',
            "color": "#f97316",
        },
        {
            "label": "Trackers Offline",
            "value": t_sum.get("critical_offline", "—"),
            "sub":   f'<span class="warn">{t_sum.get("claim_candidates","—")} claim candidates</span>',
            "color": "#a855f7",
        },
    ])

    # ── Four donut mini-charts ────────────────────────────────────────────────
    section_header("Status Distribution")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not idle_df.empty and "idle_status" in idle_df.columns:
            counts = idle_df["idle_status"].value_counts().reset_index()
            counts.columns = ["status", "count"]
            fig = donut_chart(counts, "status", "count", "Trolley Idle Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("No trolley data")

    with col2:
        if not maint_df.empty:
            import pandas as pd
            job_counts = pd.DataFrame({
                "status": ["Open", "Closed"],
                "count":  [m_sum.get("open_jobs", 0), m_sum.get("closed_jobs", 0)]
            })
            fig = donut_chart(
                job_counts, "status", "count", "Maintenance Jobs",
                color_map={"Open": "#f59e0b", "Closed": "#10b981"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("No maintenance data")

    with col3:
        if not batt_df.empty and "warranty_status" in batt_df.columns:
            counts = batt_df["warranty_status"].value_counts().reset_index()
            counts.columns = ["status", "count"]
            fig = donut_chart(counts, "status", "count", "Battery Warranty Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("No battery data")

    with col4:
        if not track_df.empty and "offline_status" in track_df.columns:
            counts = track_df["offline_status"].value_counts().reset_index()
            counts.columns = ["status", "count"]
            fig = donut_chart(
                counts, "status", "count", "Tracker Online Status",
                color_map={
                    "Online":           "#10b981",
                    "Warning (24–48h)": "#f59e0b",
                    "Critical (>48h)":  "#ef4444",
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_placeholder("No tracker data")

    # ── Combined insights panel ───────────────────────────────────────────────
    section_header("AI-Generated Insights — All KPIs")
    all_insights = (
        generate_idle_insights(idle_df) +
        generate_maintenance_insights(maint_df) +
        generate_battery_insights(batt_df) +
        generate_tracker_insights(track_df)
    )
    # Sort by severity priority
    severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
    all_insights.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 99))
    insight_cards(all_insights)