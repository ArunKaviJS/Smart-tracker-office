"""
Trolley Demand Forecasting — Full ML Pipeline
Streamlit App: EDA → Preprocessing → XGBRegressor → Live Forecast (Next 5 Days)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb

from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "xgb_trolley_model.pkl"
ENCODER_PATH = "label_encoders.pkl"
FEATURE_COLS = [
    "day_num", "is_weekend", "hour", "geozoneId", "geolayerId", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos",
]
DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trolley Demand Forecast",
    layout="wide",
    page_icon="🛒",
)

st.markdown("""
<style>
    /* General */
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1a237e; }
    h2, h3 { color: #283593; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a237e; }
    [data-testid="stSidebar"] * { color: #fff !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 15px; }

    /* Custom cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .sec-header {
        background: linear-gradient(90deg, #1a237e, #3949ab);
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 17px;
        font-weight: 700;
        margin: 18px 0 10px 0;
    }
    .info-box  { background:#e8eaf6; border-left:4px solid #3949ab; padding:10px 14px; border-radius:5px; margin:6px 0; }
    .ok-box    { background:#e8f5e9; border-left:4px solid #43a047; padding:10px 14px; border-radius:5px; margin:6px 0; }
    .warn-box  { background:#fff8e1; border-left:4px solid #ffa000; padding:10px 14px; border-radius:5px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Utility helpers ────────────────────────────────────────────────────────────
def sec(title: str):
    st.markdown(f'<div class="sec-header">📌 {title}</div>', unsafe_allow_html=True)

def info_md(msg):  st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)
def ok_md(msg):    st.markdown(f'<div class="ok-box">✅ {msg}</div>', unsafe_allow_html=True)
def warn_md(msg):  st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)


# ── DB / Fetch ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    HOST     = os.getenv("DB_HOST")
    USER     = os.getenv("DB_USER")
    PASSWORD = os.getenv("DB_PASSWORD")
    DATABASE = "demand_forcast_for_trolley"
    return create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:3306/{DATABASE}")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_raw() -> pd.DataFrame:
    engine = get_engine()
    query = """
        SELECT historyId, trolleyId, geozoneId, geolayerId, createdOn
        FROM tbl_device_history
        ORDER BY trolleyId ASC,
                 STR_TO_DATE(createdOn, '%%d-%%m-%%Y %%H:%%i') ASC;
    """
    return pd.read_sql(query, engine)


# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_trolley_count(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Parse datetime
    df["createdOn"] = pd.to_datetime(df["createdOn"])
    df["date"] = df["createdOn"].dt.date
    df["time"] = df["createdOn"].dt.time
    df.drop(columns=["createdOn"], inplace=True)
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df = df.sort_values(["trolleyId", "datetime"]).reset_index(drop=True)

    # Stay compression
    df["zone_changed"] = (
        (df["trolleyId"] != df["trolleyId"].shift(1)) |
        (df["geozoneId"] != df["geozoneId"].shift(1))
    )
    df["stay_id"] = df["zone_changed"].cumsum()

    stays_raw = df.groupby("stay_id").agg(
        trolleyId  = ("trolleyId",  "first"),
        geozoneId  = ("geozoneId",  "first"),
        geolayerId = ("geolayerId", "first"),
        enter_time = ("datetime",   "min"),
        exit_time  = ("datetime",   "max"),
        ping_count = ("historyId",  "count"),
    ).reset_index(drop=True)

    stays_raw["duration_mins"] = (
        (stays_raw["exit_time"] - stays_raw["enter_time"])
        .dt.total_seconds() / 60
    ).round(1)
    stays_raw["is_long_stay"] = (stays_raw["duration_mins"] >= 120).astype(int)
    stays = stays_raw[stays_raw["geozoneId"] != 0].copy().reset_index(drop=True)

    # Expand to hourly slots
    min_slot  = stays["enter_time"].min().floor("h")
    max_slot  = stays["exit_time"].max().ceil("h")
    all_slots = pd.date_range(start=min_slot, end=max_slot, freq="1h")

    records = []
    for _, row in stays.iterrows():
        for slot_start in all_slots:
            slot_end = slot_start + pd.Timedelta(hours=1)
            if row["enter_time"] < slot_end and row["exit_time"] > slot_start:
                records.append({
                    "trolleyId"  : row["trolleyId"],
                    "geozoneId"  : row["geozoneId"],
                    "geolayerId" : row["geolayerId"],
                    "slot_start" : slot_start,
                    "date"       : slot_start.date(),
                    "hour"       : slot_start.hour,
                    "day_of_week": slot_start.day_name(),
                    "day_num"    : slot_start.dayofweek,
                    "is_weekend" : int(slot_start.dayofweek >= 5),
                })

    expanded = pd.DataFrame(records)

    tc = (
        expanded
        .groupby(["date","day_of_week","day_num","is_weekend","hour",
                  "slot_start","geozoneId","geolayerId"])
        ["trolleyId"].nunique()
        .reset_index()
        .rename(columns={"trolleyId": "trolley_count"})
        .sort_values(["date","hour","geozoneId"])
        .reset_index(drop=True)
    )
    tc["date"]  = pd.to_datetime(tc["date"])
    tc["month"] = tc["date"].dt.month
    return tc


# ── EDA ────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    sec("Exploratory Data Analysis")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",           f"{len(df):,}")
    c2.metric("Columns",        len(df.columns))
    c3.metric("Unique Geozones",df["geozoneId"].nunique())
    c4.metric("Date Range",
              f"{df['date'].min().strftime('%d %b %y')} → {df['date'].max().strftime('%d %b %y')}")

    st.subheader("Sample Data (top 20)")
    st.dataframe(df.head(20), use_container_width=True)

    # ── Null value table ──────────────────────────────────────────────
    st.subheader("Null Value Analysis")
    null_df = pd.DataFrame({
        "Column"    : df.columns,
        "Null Count": df.isnull().sum().values,
        "Null %"    : (df.isnull().sum().values / len(df) * 100).round(2),
    })
    null_df["Action"] = null_df["Null %"].apply(
        lambda x: "Drop rows (< 30 %)" if 0 < x < 30
        else ("High null – review (≥ 30 %)" if x >= 30 else "✅ No nulls")
    )
    st.dataframe(null_df, use_container_width=True)
    if null_df["Null Count"].sum() == 0:
        ok_md("No null values found in the processed dataset!")
    else:
        warn_md("Null values detected – will be handled in Preprocessing step.")

    # ── Outlier boxplots ──────────────────────────────────────────────
    st.subheader("Outlier Detection — Boxplots")
    num_cols = [c for c in ["trolley_count","hour","day_num","month"] if c in df.columns]
    fig, axes = plt.subplots(1, len(num_cols), figsize=(14, 4))
    for ax, col in zip(axes, num_cols):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#c5cae9"), medianprops=dict(color="#e53935", linewidth=2))
        ax.set_title(col, fontsize=11)
    plt.suptitle("Boxplots for Numeric Columns", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Distribution ──────────────────────────────────────────────────
    st.subheader("Trolley Count Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.hist(df["trolley_count"], bins=30, color="#1a237e", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Trolley Count"); ax2.set_ylabel("Frequency")
    ax2.set_title("Histogram of Trolley Count")
    st.pyplot(fig2); plt.close()

    # ── Heatmap: weekday × hour ───────────────────────────────────────
    st.subheader("Avg Trolley Count: Weekday × Hour")
    pivot = df.pivot_table(values="trolley_count", index="day_num", columns="hour", aggfunc="mean").fillna(0)
    pivot.index = [DAY_NAMES[i][:3] for i in pivot.index]
    fig3, ax3 = plt.subplots(figsize=(16, 4))
    sns.heatmap(pivot, cmap="Blues", ax=ax3, annot=True, fmt=".1f", linewidths=0.3)
    ax3.set_title("Average Trolleys per Hour per Weekday")
    st.pyplot(fig3); plt.close()

    # ── Avg count per geozone ─────────────────────────────────────────
    st.subheader("Average Trolley Count per GeoZone")
    gz_avg = df.groupby("geozoneId")["trolley_count"].mean().sort_values(ascending=False).reset_index()
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    ax4.bar(gz_avg["geozoneId"].astype(str), gz_avg["trolley_count"], color="#3949ab")
    ax4.set_xlabel("GeoZone ID"); ax4.set_ylabel("Avg Trolleys")
    ax4.set_title("Average Trolleys per GeoZone")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig4); plt.close()


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    sec("Data Preprocessing Pipeline")
    df = df.copy()
    orig_rows = len(df)

    # ── Step 1: Null handling ─────────────────────────────────────────
    st.subheader("Step 1 — Null Handling")
    info_md("Rule: if a column has < 30 % nulls → drop those rows. If ≥ 30 % → flag only (no drop).")
    for col in df.columns:
        null_pct = df[col].isnull().mean()
        if null_pct > 0:
            if null_pct < 0.30:
                before = len(df)
                df.dropna(subset=[col], inplace=True)
                info_md(f"<b>{col}</b>: {null_pct*100:.1f}% nulls (<30%) → dropped {before - len(df)} rows")
            else:
                warn_md(f"<b>{col}</b>: {null_pct*100:.1f}% nulls (≥30%) → not dropped (review manually)")
    ok_md(f"Rows after null handling: {len(df):,} &nbsp;(removed {orig_rows - len(df):,} rows)")

    # # ── Step 2: Outlier capping (IQR) ────────────────────────────────
    # st.subheader("Step 2 — Outlier Capping (IQR Method)")
    # cap_cols = ["trolley_count"]
    # for col in cap_cols:
    #     if col in df.columns:
    #         Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    #         IQR    = Q3 - Q1
    #         lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    #         n_out  = ((df[col] < lo) | (df[col] > hi)).sum()
    #         df[col] = df[col].clip(lower=lo, upper=hi)
    #         info_md(f"<b>{col}</b>: capped {n_out} outliers → range [{lo:.1f}, {hi:.1f}]")

    # ── Step 3: Feature Engineering ──────────────────────────────────
    st.subheader("Step 3 — Feature Engineering (Cyclical Encoding)")
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]    / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]   / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]   / 12)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_num"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_num"] / 7)
    ok_md("Added: hour_sin/cos · month_sin/cos · day_sin/cos")

    # # ── Step 4: Categorical Encoding ─────────────────────────────────
    # st.subheader("Step 4 — Categorical Encoding & Decode Reference")
    # cat_cols = ["day_of_week"]
    # encoders: dict[str, LabelEncoder] = {}
    # for col in cat_cols:
    #     if col in df.columns:
    #         le = LabelEncoder()
    #         df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    #         encoders[col] = le
    #         mapping = pd.DataFrame({
    #             "Original": le.classes_,
    #             "Encoded" : le.transform(le.classes_),
    #         })
    #         st.write(f"**{col}** — encode / decode table:")
    #         st.dataframe(mapping, use_container_width=True)

    st.subheader("Preprocessed Data — Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    return df


# ── Model Training ─────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    sec("Model Training — XGBRegressor")

    avail = [f for f in FEATURE_COLS if f in df.columns]
    X, y  = df[avail], df["trolley_count"]

    info_md(f"Features: <b>{avail}</b>")
    info_md(f"Target: <b>trolley_count</b> | X shape: {X.shape}  y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    c1, c2 = st.columns(2)
    c1.metric("Train rows", f"{len(X_train):,}")
    c2.metric("Test rows",  f"{len(X_test):,}")

    Xgb_boost = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0
    )
    with st.spinner("Training… please wait"):
        Xgb_boost.fit(X_train, y_train)

    y_pred = Xgb_boost.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",      f"{mae:.3f}")
    m2.metric("RMSE",     f"{rmse:.3f}")
    m3.metric("R² Score", f"{r2:.3f}")

    # Feature importance
    st.subheader("Feature Importance")
    fi = (
        pd.DataFrame({"Feature": avail, "Importance": Xgb_boost.feature_importances_})
        .sort_values("Importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fi["Feature"], fi["Importance"], color="#1a237e")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Score")
    st.pyplot(fig); plt.close()

    # Actual vs Predicted (first 100)
    st.subheader("Actual vs Predicted (first 100 test samples)")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(y_test.values[:100], label="Actual",    color="#1a237e", alpha=0.85)
    ax2.plot(y_pred[:100],        label="Predicted", color="#e53935", alpha=0.85, linestyle="--")
    ax2.legend(); ax2.set_title("Actual vs Predicted")
    st.pyplot(fig2); plt.close()

    # Save
    joblib.dump(Xgb_boost, MODEL_PATH)
    ok_md(f"Model saved → <b>{MODEL_PATH}</b>")
    return Xgb_boost, avail


# ── Forecast ───────────────────────────────────────────────────────────────────
def forecast_next_5_days(tc_df: pd.DataFrame, model, feature_cols: list):
    sec("Live Forecast — Next 5 Days per GeoZone per Hour")

    # Last entry date → start forecast from next day
    last_date  = tc_df["date"].max()
    start_date = last_date + pd.Timedelta(days=1)
    end_date   = start_date + pd.Timedelta(days=4)

    st.markdown(
        f'<div class="info-box">📅 Last data date: <b>{last_date.date()}</b>'
        f' &nbsp;|&nbsp; Forecast window: <b>{start_date.date()}</b>'
        f' → <b>{end_date.date()}</b></div>',
        unsafe_allow_html=True,
    )

    geozone_layer = tc_df.groupby("geozoneId")["geolayerId"].first().reset_index()
    forecast_dates = [start_date + pd.Timedelta(days=i) for i in range(5)]

    # Build prediction grid
    rows = []
    for fd in forecast_dates:
        for _, gz in geozone_layer.iterrows():
            for hr in range(24):
                rows.append({
                    "date"       : fd,
                    "day_name"   : DAY_NAMES[fd.dayofweek],
                    "day_num"    : fd.dayofweek,
                    "is_weekend" : int(fd.dayofweek >= 5),
                    "hour"       : hr,
                    "geozoneId"  : gz["geozoneId"],
                    "geolayerId" : gz["geolayerId"],
                    "month"      : fd.month,
                    "hour_sin"   : np.sin(2 * np.pi * hr / 24),
                    "hour_cos"   : np.cos(2 * np.pi * hr / 24),
                    "month_sin"  : np.sin(2 * np.pi * fd.month / 12),
                    "month_cos"  : np.cos(2 * np.pi * fd.month / 12),
                    "day_sin"    : np.sin(2 * np.pi * fd.dayofweek / 7),
                    "day_cos"    : np.cos(2 * np.pi * fd.dayofweek / 7),
                })

    pred_df = pd.DataFrame(rows)
    X_pred  = pred_df[[c for c in feature_cols if c in pred_df.columns]]
    pred_df["predicted_trolleys"] = (
        model.predict(X_pred).clip(0).round().astype(int)
    )

    # ── Tabs: one per day ─────────────────────────────────────────────
    tab_labels = [
        f"{DAY_NAMES[fd.dayofweek][:3]} {fd.strftime('%d %b')}"
        for fd in forecast_dates
    ]
    tabs = st.tabs(tab_labels)

    for fd, tab in zip(forecast_dates, tabs):
        with tab:
            day_data = pred_df[pred_df["date"] == fd]
            is_wknd  = fd.dayofweek >= 5
            badge    = "🏖️ Weekend" if is_wknd else "💼 Weekday"
            st.markdown(f"### {DAY_NAMES[fd.dayofweek]}, {fd.strftime('%d %B %Y')}  —  {badge}")

            # KPIs
            total      = day_data["predicted_trolleys"].sum()
            peak_idx   = day_data["predicted_trolleys"].idxmax()
            peak_hour  = day_data.loc[peak_idx, "hour"]
            peak_count = day_data.loc[peak_idx, "predicted_trolleys"]
            peak_zone  = day_data.loc[peak_idx, "geozoneId"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Trolleys (Day)", f"{total:,}")
            k2.metric("Peak Hour",            f"{peak_hour:02d}:00")
            k3.metric("Peak Count",           f"{peak_count}")
            k4.metric("Peak GeoZone",         f"{peak_zone}")

            # Pivot table: hour (rows) × geozone (cols)
            st.markdown("**Predicted Trolleys — Hour × GeoZone**")
            pivot = day_data.pivot_table(
                values="predicted_trolleys",
                index="hour",
                columns="geozoneId",
                aggfunc="sum",
                fill_value=0,
            )
            st.dataframe(
                pivot.style.background_gradient(cmap="Blues", axis=None),
                use_container_width=True,
            )

            # Line chart
            fig = px.line(
                day_data, x="hour", y="predicted_trolleys", color="geozoneId",
                title=f"Hourly Demand by GeoZone — {DAY_NAMES[fd.dayofweek]}",
                labels={"hour":"Hour", "predicted_trolleys":"Trolleys", "geozoneId":"GeoZone"},
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(height=350, xaxis=dict(tickmode="linear", dtick=1))
            st.plotly_chart(fig, use_container_width=True)

    # ── 5-day summary heatmap ─────────────────────────────────────────
    st.subheader("5-Day Demand Heatmap (All GeoZones Combined)")
    summary = (
        pred_df.groupby(["date","hour"])["predicted_trolleys"]
        .sum().reset_index()
    )
    pivot2 = summary.pivot(index="date", columns="hour", values="predicted_trolleys").fillna(0)
    pivot2.index = [
        f"{DAY_NAMES[pd.Timestamp(d).dayofweek][:3]} {pd.Timestamp(d).strftime('%d')}"
        for d in pivot2.index
    ]
    fig5, ax5 = plt.subplots(figsize=(18, 4))
    sns.heatmap(pivot2, cmap="YlOrRd", ax=ax5, annot=True, fmt=".0f",
                linewidths=0.3, linecolor="#eee")
    ax5.set_title("Predicted Trolley Count — Hour × Day")
    ax5.set_xlabel("Hour of Day")
    st.pyplot(fig5); plt.close()

    # ── Per-geozone summary table ─────────────────────────────────────
    st.subheader("Per-GeoZone Daily Summary")
    gz_summary = (
        pred_df.groupby(["date","day_name","geozoneId"])["predicted_trolleys"]
        .agg(total_trolleys="sum", peak_hour_count="max")
        .reset_index()
    )
    gz_summary["date"] = gz_summary["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(gz_summary, use_container_width=True)

    return pred_df


# ── Session state init ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "trolley_count"   : None,
        "preprocessed_df" : None,
        "encoders"        : None,
        "model"           : None,
        "feature_cols"    : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    _init_state()

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🛒 Trolley Forecast")
        st.markdown("---")
        page = st.radio("Navigate", [
            "📥 Fetch & Process Data",
            "📊 EDA",
            "🔧 Preprocessing",
            "🤖 Train Model",
            "🔮 Live Forecast",
        ])
        st.markdown("---")
        st.markdown("**Model Status**")
        if os.path.exists(MODEL_PATH):
            st.success("✅ Model ready on disk")
        else:
            st.warning("⚠️ Model not trained yet")

        if st.session_state.trolley_count is not None:
            tc = st.session_state.trolley_count
            st.markdown("**Data Loaded**")
            st.info(f"{len(tc):,} records\n{tc['date'].min().date()} → {tc['date'].max().date()}")

    # ── Header ────────────────────────────────────────────────────────
    st.title("🛒 Trolley Demand Forecasting")
    st.markdown("**Full ML Pipeline: EDA → Preprocessing → XGBRegressor → Live 5-Day Forecast**")
    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════
    # PAGE: Fetch & Process Data
    # ═════════════════════════════════════════════════════════════════
    if page == "📥 Fetch & Process Data":
        sec("Connect to MySQL & Build Trolley Count Table")
        st.markdown("""
        This step:
        1. Connects to MySQL via SQLAlchemy
        2. Fetches raw device history
        3. Compresses into "stay" records (zone changes)
        4. Expands into hourly slots per geozone
        5. Aggregates unique trolley count per hour
        """)
        if st.button("🔄 Fetch & Process", type="primary"):
            try:
                with st.spinner("Fetching from MySQL…"):
                    raw = fetch_raw()
                ok_md(f"Fetched {len(raw):,} rows from tbl_device_history")
                with st.spinner("Building trolley count table…"):
                    tc = build_trolley_count(raw)
                st.session_state.trolley_count = tc
                ok_md(f"Done → {len(tc):,} hourly geozone records")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                warn_md("Check your <b>.env</b> file for DB_HOST, DB_USER, DB_PASSWORD")

        if st.session_state.trolley_count is not None:
            st.subheader("Preview (top 100)")
            st.dataframe(st.session_state.trolley_count.head(100), use_container_width=True)

    # ═════════════════════════════════════════════════════════════════
    # PAGE: EDA
    # ═════════════════════════════════════════════════════════════════
    elif page == "📊 EDA":
        if st.session_state.trolley_count is None:
            warn_md("Please fetch data first from <b>📥 Fetch & Process Data</b>")
        else:
            run_eda(st.session_state.trolley_count)

    # ═════════════════════════════════════════════════════════════════
    # PAGE: Preprocessing
    # ═════════════════════════════════════════════════════════════════
    elif page == "🔧 Preprocessing":
        if st.session_state.trolley_count is None:
            warn_md("Please fetch data first from <b>📥 Fetch & Process Data</b>")
        else:
            if st.button("▶️ Run Preprocessing", type="primary"):
                df_proc = preprocess(st.session_state.trolley_count)
                st.session_state.preprocessed_df = df_proc
                ok_md("Preprocessing complete! Proceed to <b>🤖 Train Model</b>.")

            if st.session_state.preprocessed_df is not None:
                info_md(f"Preprocessed shape: {st.session_state.preprocessed_df.shape}")
                

    # ═════════════════════════════════════════════════════════════════
    # PAGE: Train Model
    # ═════════════════════════════════════════════════════════════════
    elif page == "🤖 Train Model":
        if st.session_state.preprocessed_df is None:
            warn_md("Please run <b>🔧 Preprocessing</b> first.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🚀 Train XGBRegressor", type="primary"):
                    model, fcols = train_model(st.session_state.preprocessed_df)
                    st.session_state.model        = model
                    st.session_state.feature_cols = fcols

            with col_b:
                if os.path.exists(MODEL_PATH):
                    if st.button("📂 Load Saved Model from Disk"):
                        loaded_xgb_regressor = joblib.load(MODEL_PATH)
                        st.session_state.model        = loaded_xgb_regressor
                        st.session_state.feature_cols = [
                            f for f in FEATURE_COLS
                            if f in st.session_state.preprocessed_df.columns
                        ]
                        ok_md("Loaded <b>loaded_xgb_regressor</b> from disk!")

    # ═════════════════════════════════════════════════════════════════
    # PAGE: Live Forecast
    # ═════════════════════════════════════════════════════════════════
    elif page == "🔮 Live Forecast":
        # ── Model Load Section ─────────────────────────────────────────
        col_load, col_status = st.columns([2, 3])

        with col_load:
            if os.path.exists(MODEL_PATH):
                if st.button("📂 Load Model (xgb_trolley_model.joblib)", type="primary"):
                    loaded_xgb_regressor = joblib.load(MODEL_PATH)
                    st.session_state.model = loaded_xgb_regressor
                    st.session_state.feature_cols = FEATURE_COLS
                    ok_md("Model loaded successfully — <b>loaded_xgb_regressor</b> is ready!")
            else:
                warn_md("No model file found at <b>xgb_trolley_model.joblib</b> — train first.")

        with col_status:
            if st.session_state.model is not None:
                ok_md("✅ Model is loaded and ready for prediction.")
                # Show model file metadata
                import time
                mod_time = os.path.getmtime(MODEL_PATH)
                trained_on = pd.Timestamp(mod_time, unit='s').strftime('%d %b %Y  %H:%M')
                info_md(f"Model file last saved: <b>{trained_on}</b>")
            else:
                warn_md("No model loaded yet — click the button.")

        st.markdown("---")

        if st.session_state.model is None:
            warn_md("No model found. Please train or load a model from <b>🤖 Train Model</b>.")
        elif st.session_state.trolley_count is None:
            warn_md("No data loaded. Please fetch data from <b>📥 Fetch & Process Data</b>.")
        else:
            forecast_next_5_days(
                st.session_state.trolley_count,
                st.session_state.model,
                st.session_state.feature_cols,
            )


if __name__ == "__main__":
    main()