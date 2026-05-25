"""
main.py — KPI Dashboard Entry Point
Run with: streamlit run main.py
"""

import streamlit as st

# ── Must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="KPI Insights Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
from ui.style import DASHBOARD_CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# ── Page imports ──────────────────────────────────────────────────────────────
from pages import overview, idle_trolleys, maintenance, battery_warranty, tracker_warranty
from pages import schema_inspector

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = {
    "◈  Overview":              overview,
    "◎  Idle Trolleys":         idle_trolleys,
    "⚙  Maintenance":           maintenance,
    "⬡  Battery Warranty":      battery_warranty,
    "◉  Tracker Warranty":      tracker_warranty,
    "⬚  Schema Inspector":      schema_inspector,   # debug
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding:16px 0 24px 0">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;
                        font-weight:600;color:#e8edf2;letter-spacing:1px">
                KPI Insights
            </div>
            <div style="font-size:10px;color:#3d5a73;letter-spacing:2px;
                        text-transform:uppercase;margin-top:2px">
                Automated Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = st.radio(
        label="Navigation",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2d40;margin:16px 0'>", unsafe_allow_html=True)

    # ── Cache controls ────────────────────────────────────────────────────────
    if st.button("↺  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """
        <div style="padding:12px 0;font-size:10px;color:#3d5a73">
            <div style="margin-bottom:4px">Cache TTL: 3–5 minutes</div>
            <div>DB: automated_kpi_insights</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render selected page ──────────────────────────────────────────────────────
PAGES[selected].render()