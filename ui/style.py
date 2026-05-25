"""
ui/style.py — Global Streamlit CSS + Plotly theme
Industrial dark dashboard aesthetic: #0d1117 base, teal accents, mono typography.
"""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stApp {
    background-color: #0d1117;
}
.block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #10161f !important;
    border-right: 1px solid #1e2a38 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #8b98a9 !important;
    font-size: 13px !important;
    padding: 6px 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #00d4aa !important;
}

/* ── KPI metric cards ────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px;
    margin: 8px 0 24px 0;
}
.kpi-card {
    background: #131c28;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 18px 20px 14px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #2d4a66; }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--kpi-color, #00d4aa);
    opacity: 0.8;
}
.kpi-card .kpi-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #566779;
    margin-bottom: 10px;
}
.kpi-card .kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px;
    font-weight: 600;
    color: #e8edf2;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-card .kpi-sub {
    font-size: 11px;
    color: #566779;
}
.kpi-card .kpi-sub .up   { color: #10b981; }
.kpi-card .kpi-sub .down { color: #ef4444; }
.kpi-card .kpi-sub .warn { color: #f59e0b; }

/* ── Section header ──────────────────────────────── */
.section-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #3d5a73;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 8px;
    margin: 24px 0 14px 0;
}

/* ── Insight cards ───────────────────────────────── */
.insight-card {
    background: #131c28;
    border: 1px solid #1e2d40;
    border-left: 3px solid var(--ic-color, #3b82f6);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
}
.insight-card .ic-title {
    font-size: 13px;
    font-weight: 500;
    color: #c9d4e0;
    margin-bottom: 4px;
}
.insight-card .ic-detail {
    font-size: 11px;
    color: #566779;
    line-height: 1.5;
}

/* ── Badge ───────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.badge-critical { background:#2d1010; color:#ef4444; border:1px solid #5c1d1d; }
.badge-warning  { background:#2d2010; color:#f59e0b; border:1px solid #5c4020; }
.badge-success  { background:#0d2d1e; color:#10b981; border:1px solid #1a5c3e; }
.badge-info     { background:#0d1e2d; color:#3b82f6; border:1px solid #1a3e5c; }
.badge-neutral  { background:#1a2435; color:#8b98a9; border:1px solid #2d3e50; }

/* ── Page title ──────────────────────────────────── */
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    color: #e8edf2;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
.page-subtitle {
    font-size: 12px;
    color: #566779;
    margin-bottom: 24px;
}

/* ── Streamlit overrides ─────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stMetric { display: none; }
[data-testid="stDataFrame"] { border: 1px solid #1e2d40; border-radius: 8px; overflow: hidden; }
.stSpinner > div { border-top-color: #00d4aa !important; }
div[data-testid="stMarkdownContainer"] p { color: #8b98a9; font-size: 13px; }

/* ── Tab style ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e2d40;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #566779 !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
}
</style>
"""

SEVERITY_COLORS = {
    "critical": "#ef4444",
    "warning":  "#f59e0b",
    "info":     "#3b82f6",
    "success":  "#10b981",
    "neutral":  "#566779",
}

STATUS_KPI_COLORS = {
    "Critical Idle":   "#ef4444",
    "Idle":            "#f59e0b",
    "Soft Idle":       "#3b82f6",
    "Active":          "#10b981",
    "Expired":         "#ef4444",
    "Critical":        "#f97316",
    "Warning":         "#f59e0b",
    "Healthy":         "#10b981",
    "Critical (>48h)": "#ef4444",
    "Warning (24–48h)":"#f59e0b",
    "Online":          "#10b981",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#131c28",
    font=dict(family="IBM Plex Sans", color="#8b98a9", size=11),
    title_font=dict(family="IBM Plex Mono", color="#c9d4e0", size=13),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b98a9", size=10),
    ),
    xaxis=dict(
        gridcolor="#1e2d40",
        linecolor="#1e2d40",
        tickfont=dict(color="#566779", size=10),
        title_font=dict(color="#566779"),
    ),
    yaxis=dict(
        gridcolor="#1e2d40",
        linecolor="#1e2d40",
        tickfont=dict(color="#566779", size=10),
        title_font=dict(color="#566779"),
    ),
    margin=dict(l=40, r=20, t=40, b=40),
)