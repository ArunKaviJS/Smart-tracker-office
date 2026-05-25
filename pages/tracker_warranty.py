"""
ui/pages/tracker_warranty.py — Tracker / Device Warranty KPI page.
"""

import streamlit as st
import plotly.express as px
from ui.components import (
    page_header, section_header, kpi_row, insight_cards,
    bar_chart, donut_chart, data_table, no_data_placeholder,
)
from ui.style import PLOTLY_LAYOUT
from kpis.tracker_warranty import (
    get_tracker_data, get_tracker_summary,
    get_firmware_breakdown, get_offline_tracker_list,
)
from utils.anomaly import generate_tracker_insights


OFFLINE_COLORS = {
    "Online":           "#10b981",
    "Warning (24–48h)": "#f59e0b",
    "Critical (>48h)":  "#ef4444",
}


@st.cache_data(ttl=180, show_spinner=False)
def _load():
    return get_tracker_data()


def render():
    page_header(
        "Tracker Warranty KPI",
        "Offline detection, location accuracy, firmware audit, and claim candidates"
    )

    with st.spinner("Loading tracker data…"):
        df = _load()

    if df.empty:
        no_data_placeholder("No tracker data found in tbl_device_data.")
        return

    summary = get_tracker_summary(df)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi_row([
        {"label": "Total Trackers",    "value": summary.get("total_trackers","—"),    "color":"#00d4aa"},
        {"label": "Critical Offline",  "value": summary.get("critical_offline","—"),
         "sub": '<span class="down">&gt;48 hours silent</span>',                     "color":"#ef4444"},
        {"label": "Warning Offline",   "value": summary.get("warning_offline","—"),
         "sub": '<span class="warn">24–48 hours</span>',                             "color":"#f59e0b"},
        {"label": "Online",            "value": summary.get("online","—"),
         "sub": '<span class="up">reporting normally</span>',                        "color":"#10b981"},
        {"label": "Claim Candidates",  "value": summary.get("claim_candidates","—"),
         "sub": '<span class="down">fault / offline / location</span>',              "color":"#a855f7"},
        {"label": "Firmware Versions", "value": summary.get("firmware_versions","—"),
         "sub": "unique versions in fleet",                                          "color":"#3b82f6"},
    ])

    # ── Charts ────────────────────────────────────────────────────────────────
    section_header("Tracker Status")
    c1, c2 = st.columns([2, 3])

    with c1:
        if "offline_status" in df.columns:
            counts = df["offline_status"].value_counts().reset_index()
            counts.columns = ["status", "count"]
            fig = donut_chart(counts, "status", "count", "Online / Offline Status", OFFLINE_COLORS)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Scatter: hours offline vs location fail rate
        if "location_fail_rate" in df.columns:
            fig = px.scatter(
                df,
                x="hours_offline",
                y="location_fail_rate",
                color="offline_status",
                color_discrete_map=OFFLINE_COLORS,
                hover_data=["deviceEndpoint", "firmwareVersion"],
                title="Offline Duration vs Location Failure Rate",
                labels={
                    "hours_offline":       "Hours since last report",
                    "location_fail_rate":  "Failed location readings (%)",
                },
                template="plotly_dark",
            )
            fig.add_hline(y=0.3, line_dash="dash", line_color="#f59e0b",
                          annotation_text="30% failure threshold")
            fig.add_vline(x=48, line_dash="dash", line_color="#ef4444",
                          annotation_text="48-hr mark")
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ── Firmware breakdown ─────────────────────────────────────────────────────
    section_header("Firmware Audit")
    fw_df = get_firmware_breakdown(df)
    if not fw_df.empty:
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = bar_chart(
                fw_df, x="firmwareVersion", y="count",
                title="Trackers per Firmware Version",
            )
            fig.update_traces(marker_color="#3b82f6", marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(
                fw_df.rename(columns={
                    "firmwareVersion": "Firmware",
                    "count": "Trackers",
                    "avg_offline": "Avg Hrs Offline",
                }).style.format({"Avg Hrs Offline": "{:.1f}"}),
                use_container_width=True, hide_index=True
            )

    # ── Battery level distribution ─────────────────────────────────────────────
    if "avg_battery" in df.columns:
        section_header("Tracker Battery Levels")
        fig = px.histogram(
            df, x="avg_battery",
            color="offline_status",
            color_discrete_map=OFFLINE_COLORS,
            nbins=20,
            title="Average Tracker Battery Level Distribution",
            labels={"avg_battery": "Avg Battery Level (%)"},
            template="plotly_dark",
        )
        fig.add_vline(x=20, line_dash="dash", line_color="#ef4444",
                      annotation_text="Critical (<20%)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    section_header("AI-Generated Insights")
    insight_cards(generate_tracker_insights(df))

    # ── Tables ────────────────────────────────────────────────────────────────
    section_header("Tracker Records")
    offline_list = get_offline_tracker_list(df)

    tab1, tab2 = st.tabs(["All Trackers", "Offline & Claim Candidates"])
    with tab1:
        display_cols = [c for c in [
            "deviceEndpoint", "firmwareVersion", "last_seen",
            "hours_offline", "offline_status", "avg_battery",
            "has_fault", "location_fail_rate", "claim_candidate"
        ] if c in df.columns]
        data_table(df[display_cols].sort_values("hours_offline", ascending=False))

    with tab2:
        if not offline_list.empty:
            data_table(offline_list)
        else:
            st.success("✓ All trackers are online and reporting normally.")