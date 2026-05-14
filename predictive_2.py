import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
PKL_DIR           = "my_prod"
TYPE_ENC_PATH     = os.path.join(PKL_DIR, "type_encoder.pkl")
REASON_ENC_PATH   = os.path.join(PKL_DIR, "reason_encoder.pkl")
MODEL_PATH        = os.path.join(PKL_DIR, "model.pkl")
OTHERS_CACHE_PATH = os.path.join(PKL_DIR, "others_intent_cache.json")
os.makedirs(PKL_DIR, exist_ok=True)

# 11 valid reason categories — "Others" free-text will be mapped to one of these
VALID_REASONS = [
    "Wheel Alignment", "Oiling & Greasing", "Cleaning & Inspection",
    "Parts Replacement", "Battery replaced", "Brake Adjustment",
    "Motor Failure", "Broken & Needs Welding", "Frame Damage",
    "Structural Damage", "Accident",
]



# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🔧 Trolley Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #00e5ff;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: 1px;
        border-left: 6px solid #00e5ff;
    }
    .sub-header {
        color: #aaa;
        font-size: 0.9rem;
        margin-top: -0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        text-align: center;
    }
    .metric-card .val {
        font-size: 2rem;
        font-weight: 800;
        color: #00e5ff;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .lbl {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e6edf3;
        border-bottom: 2px solid #21262d;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Syne', sans-serif;
    }
    .log-box {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #7ee787;
        max-height: 280px;
        overflow-y: auto;
    }
    .alert-warn {
        background: #2d1b00;
        border-left: 4px solid #d29922;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #e3b341;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }
    .alert-ok {
        background: #0d2818;
        border-left: 4px solid #238636;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        color: #3fb950;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stDataFrame"] { border-radius: 8px; }
    div[data-testid="stExpander"] { border: 1px solid #21262d; border-radius: 8px; }
    .stButton > button {
        background: linear-gradient(90deg, #00e5ff22, #00e5ff44);
        border: 1px solid #00e5ff;
        color: #00e5ff;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        letter-spacing: 1px;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #00e5ff;
        color: #0d1117;
    }
    .urgency-critical { color: #ff4b4b; font-weight: 700; }
    .urgency-warning  { color: #ffa500; font-weight: 700; }
    .urgency-ok       { color: #00e5ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOGGING HELPER
# ─────────────────────────────────────────────
def log(msg: str, level: str = "INFO"):
    """Append to session-state log AND print to terminal."""
    tag = {"INFO": "✅", "WARN": "⚠️", "ERR": "❌", "STEP": "🔷"}.get(level, "▶")
    entry = f"{tag} {msg}"
    print(entry)                          # terminal print for debugging
    if "logs" not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(entry)

# ─────────────────────────────────────────────
#  1. DB CONNECTION & DATA LOAD
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_data() -> pd.DataFrame:
    log("STEP 1 ▶ Connecting to database …", "STEP")
    HOST     = os.getenv("DB_HOST",     "localhost")
    USER     = os.getenv("DB_USER",     "root")
    PASSWORD = os.getenv("DB_PASSWORD", "")
    DATABASE = os.getenv("DB_DATABASE", "trolley_db")

    conn_str = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:3306/{DATABASE}"
    print(f"[DB] Engine → {conn_str.replace(PASSWORD, '****')}")

    engine = create_engine(conn_str)
    query  = "SELECT trolleyId, type, reason, createdOn FROM tbl_trolley_maintenance_50k"

    log(f"Running query: {query}", "INFO")
    df = pd.read_sql(query, engine)
    log(f"Loaded {len(df):,} rows, {df.shape[1]} columns", "INFO")
    print("[DB] Sample rows:\n", df.head())
    return df

# ─────────────────────────────────────────────
#  2. EDA  (null check + outlier capping)
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    log("STEP 2 ▶ Starting EDA …", "STEP")

    # ── 2a. NULL ANALYSIS ─────────────────────
    null_counts = df.isnull().sum()
    null_pct    = null_counts / len(df) * 100
    print("[EDA] Null counts:\n", null_counts)
    print("[EDA] Null pct:\n", null_pct)

    rows_before = len(df)
    for col in df.columns:
        pct = null_pct[col]
        if pct == 0:
            continue
        if pct < 30:
            log(f"Column '{col}': {pct:.1f}% nulls → dropping null rows", "WARN")
            df = df[df[col].notna()]
        else:
            log(f"Column '{col}': {pct:.1f}% nulls → filling with mean/mode", "WARN")
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    log(f"After null handling: {rows_before:,} → {len(df):,} rows", "INFO")

    # ── 2b. OUTLIER ANALYSIS (numeric cols) ───
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    capping_info = {}

    fig_cols = min(len(numeric_cols), 4)
    fig, axes = plt.subplots(1, max(fig_cols, 1),
                             figsize=(5 * max(fig_cols, 1), 4),
                             facecolor="#0d1117")

    if fig_cols == 1:
        axes = [axes]

    for i, col in enumerate(numeric_cols[:4]):
        ax = axes[i]
        ax.set_facecolor("#161b22")
        data = df[col].dropna()

        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((data < lo) | (data > hi)).sum()

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        boxprops=dict(facecolor="#1f6feb", color="#00e5ff"),
                        whiskerprops=dict(color="#58a6ff"),
                        capprops=dict(color="#58a6ff"),
                        medianprops=dict(color="#f78166", linewidth=2),
                        flierprops=dict(marker="o", color="#ff4b4b",
                                        markerfacecolor="#ff4b4b", markersize=3))
        ax.set_title(col, color="#c9d1d9", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.spines[:].set_color("#21262d")

        if outliers > 0:
            log(f"Outliers in '{col}': {outliers} → applying IQR capping", "WARN")
            df[col] = df[col].clip(lower=lo, upper=hi)
            capping_info[col] = {"lower": lo, "upper": hi, "count": int(outliers)}
            ax.set_xlabel(f"⚠ {outliers} capped", color="#ffa500", fontsize=7)
        else:
            log(f"No outliers in '{col}'", "INFO")
            ax.set_xlabel("✔ No outliers", color="#3fb950", fontsize=7)

    plt.tight_layout()
    print("[EDA] Boxplot drawn")
    return df, fig, null_pct, capping_info

# ─────────────────────────────────────────────
#  3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log("STEP 3 ▶ Preprocessing …", "STEP")

    print("[PRE] createdOn sample BEFORE parse:\n", df["createdOn"].head())
    df["createdOn"] = pd.to_datetime(df["createdOn"], dayfirst=True).dt.normalize()
    print("[PRE] createdOn sample AFTER parse:\n", df["createdOn"].head())

    df = df.sort_values(["trolleyId", "reason", "createdOn"]).reset_index(drop=True)

    df["day"]         = df["createdOn"].dt.day
    df["month"]       = df["createdOn"].dt.month
    df["day_of_week"] = df["createdOn"].dt.dayofweek

    df["nextIssueOn"] = df.groupby(["trolleyId", "reason"])["createdOn"].shift(-1)
    df["difference"]  = (df["nextIssueOn"] - df["createdOn"]).dt.days

    rows_before = len(df)
    df = df[df["difference"].notna() & (df["difference"] > 0)].copy()
    log(f"Rows after removing NaN/zero difference: {rows_before:,} → {len(df):,}", "INFO")

    print("[PRE] Preprocessed sample:\n", df.head())
    return df


# ─────────────────────────────────────────────
#  KEYWORD → CATEGORY MAPPING
# ─────────────────────────────────────────────
KEYWORD_MAP = {
    "Wheel Alignment"       : ["wheel", "align", "tyre", "tire", "wobble",
                                "steering", "rotation", "rim", "axle"],

    "Oiling & Greasing"     : ["oil", "grease", "lubric", "rust", "squeak",
                                "noise", "friction", "lubricate", "stiff"],

    "Cleaning & Inspection" : ["clean", "inspect", "dirt", "dust", "wash",
                                "check", "service", "hygiene", "debris"],

    "Parts Replacement"     : ["replace", "part", "spare", "worn", "component",
                                "swap", "change", "new part", "fitting"],

    "Battery replaced"      : ["battery", "charge", "power", "dead", "electric",
                                "volt", "discharge", "not starting", "no power"],

    "Brake Adjustment"      : ["brake", "stop", "slow", "skid", "pad",
                                "braking", "handbrake", "pedal"],

    "Motor Failure"         : ["motor", "engine", "start", "run", "speed",
                                "drive", "rpm", "not moving", "overheating"],

    "Broken & Needs Welding": ["weld", "broken", "crack", "snap", "bent",
                                "break", "fracture", "split", "shatter"],

    "Frame Damage"          : ["frame", "body", "dent", "bend", "frame damage",
                                "chassis", "body damage"],

    "Structural Damage"     : ["structural", "collapse", "deform", "warp",
                                "twist", "lean", "tilting", "unstable"],

    "Accident"              : ["accident", "crash", "collide", "collision",
                                "hit", "bump", "impact", "fell", "fall",
                                "topple", "damaged by"],
}

def _call_claude_intent(free_text: str) -> str:
    """
    Keyword-based intent classifier.
    Checks free_text against KEYWORD_MAP and returns
    the category with the most keyword matches.
    Falls back to 'Structural Damage' if nothing matches.
    """
    text_lower = free_text.lower()
    print(f"[INTENT] Classifying: '{free_text[:60]}'")

    scores = {}
    for category, keywords in KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
            print(f"[INTENT]   {category}: {score} match(es)")

    if scores:
        best = max(scores, key=scores.get)
        print(f"[INTENT] Winner → '{best}' (score={scores[best]})")
        return best

    print(f"[INTENT] No keyword match → defaulting to 'Structural Damage'")
    return "Structural Damage"


def resolve_others(df: pd.DataFrame):
    """
    Find all rows where `reason` is NOT a known valid category
    (i.e. free-text 'Others' entries), classify each via Claude,
    replace in-place, and persist cache to JSON.

    Returns: (cleaned_df, report_df)
      report_df — summary table of what was replaced, shown in UI
    """
    log("STEP 3b ▶ Resolving 'Others' free-text reasons …", "STEP")

    # ── Load cache ────────────────────────────────────────────────────
    if os.path.exists(OTHERS_CACHE_PATH):
        with open(OTHERS_CACHE_PATH, "r", encoding="utf-8") as f:
            cache: dict = json.load(f)
        log(f"Loaded intent cache → {len(cache)} existing mappings", "INFO")
        print(f"[INTENT] Cache loaded: {len(cache)} entries")
    else:
        cache = {}
        log("No intent cache found — starting fresh", "INFO")

    valid_set = set(VALID_REASONS)

    # Rows whose reason is NOT a recognised category
    mask_others = ~df["reason"].isin(valid_set)
    others_texts = df.loc[mask_others, "reason"].unique().tolist()

    print(f"[INTENT] Total 'Others' rows  : {mask_others.sum()}")
    print(f"[INTENT] Unique free-text vals: {len(others_texts)}")
    print(f"[INTENT] Samples: {others_texts[:5]}")

    if not others_texts:
        log("No 'Others' rows detected — skipping intent resolution", "INFO")
        return df, pd.DataFrame(), cache   

    log(f"Found {mask_others.sum():,} rows with free-text reasons "
        f"({len(others_texts)} unique)", "WARN")

    # ── Classify (cache-first) ────────────────────────────────────────
    report_rows = []
    new_entries = 0

    for text in others_texts:
        if text in cache:
            mapped = cache[text]
            print(f"[INTENT] CACHE HIT: '{text[:50]}' → '{mapped}'")
        else:
            log(f"Calling Claude for: '{text[:60]}'", "INFO")
            mapped = _call_claude_intent(text)
            cache[text] = mapped
            new_entries += 1
            time.sleep(0.2)     # gentle rate-limit buffer

        report_rows.append({
            "original_text": text,
            "mapped_to":     mapped,
            "rows_affected": int((df["reason"] == text).sum()),
            "source":        "cache" if new_entries == 0 else "claude_api",
        })
        # Replace in dataframe
        df.loc[df["reason"] == text, "reason"] = mapped

    # ── Persist updated cache ─────────────────────────────────────────
    with open(OTHERS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    if new_entries:
        log(f"{new_entries} new mappings added to cache → {OTHERS_CACHE_PATH}", "INFO")
    else:
        log("All mappings served from cache — no API calls made", "INFO")

    # Verify no others remain
    still_others = (~df["reason"].isin(valid_set)).sum()
    if still_others:
        log(f"WARNING: {still_others} rows still have unrecognised reason after mapping", "WARN")
    else:
        log("✔ All free-text reasons successfully mapped to valid categories", "INFO")

    print(f"[INTENT] Unique reasons after resolve: {df['reason'].unique().tolist()}")

    report_df = pd.DataFrame(report_rows)
    
    return df, report_df, cache


# ─────────────────────────────────────────────
#  4. ENCODING  (load pkl or fit & save)
# ─────────────────────────────────────────────
def encode(df: pd.DataFrame):
    log("STEP 4 ▶ Encoding …", "STEP")

    if os.path.exists(TYPE_ENC_PATH) and os.path.exists(REASON_ENC_PATH):
        log("Found existing encoder PKLs → loading …", "INFO")
        le_type   = joblib.load(TYPE_ENC_PATH)
        le_reason = joblib.load(REASON_ENC_PATH)
        print(f"[ENC] type   classes: {le_type.classes_}")
        print(f"[ENC] reason classes: {le_reason.classes_}")

        # Handle unseen labels gracefully
        unseen_types   = set(df["type"].unique())   - set(le_type.classes_)
        unseen_reasons = set(df["reason"].unique()) - set(le_reason.classes_)

        if unseen_types or unseen_reasons:
            log(f"Unseen labels detected → re-fitting encoders", "WARN")
            print(f"[ENC] Unseen type: {unseen_types}, reason: {unseen_reasons}")
            le_type.classes_   = np.union1d(le_type.classes_,   list(unseen_types))
            le_reason.classes_ = np.union1d(le_reason.classes_, list(unseen_reasons))
    else:
        log("No PKL found → fitting new encoders …", "WARN")
        le_type   = LabelEncoder()
        le_reason = LabelEncoder()
        print("=" * 40 + " Encoding " + "=" * 40)
        le_type.fit(df["type"])
        le_reason.fit(df["reason"])
        joblib.dump(le_type,   TYPE_ENC_PATH)
        joblib.dump(le_reason, REASON_ENC_PATH)
        log(f"Encoders saved → {TYPE_ENC_PATH}, {REASON_ENC_PATH}", "INFO")

    df["type"]   = le_type.transform(df["type"])
    df["reason"] = le_reason.transform(df["reason"])
    print("[ENC] Encoded sample:\n", df[["type", "reason"]].head())
    return df, le_type, le_reason

# ─────────────────────────────────────────────
#  5. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    log("STEP 5 ▶ Training model on full data …", "STEP")

    feature_cols = ["trolleyId", "type", "reason", "day", "month", "day_of_week"]
    X = df[feature_cols]
    y = df["difference"]

    print(f"[TRAIN] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[TRAIN] y stats — mean: {y.mean():.2f}, std: {y.std():.2f}, "
          f"min: {y.min()}, max: {y.max()}")

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    y_pred = model.predict(X)
    mae    = mean_absolute_error(y, y_pred)
    r2     = r2_score(y, y_pred)
    print(f"[TRAIN] Train MAE: {mae:.3f}  |  R²: {r2:.4f}")
    log(f"Train MAE={mae:.2f}  R²={r2:.4f}", "INFO")

    joblib.dump(model, MODEL_PATH)
    log(f"Model saved → {MODEL_PATH}", "INFO")
    return model, mae, r2

# ─────────────────────────────────────────────
#  6. PREDICT ALL TROLLEYS × REASONS
# ─────────────────────────────────────────────
def predict_maintenance(raw_df: pd.DataFrame,
                        proc_df: pd.DataFrame,
                        model,
                        le_type: LabelEncoder,
                        le_reason: LabelEncoder) -> pd.DataFrame:
    log("STEP 6 ▶ Predicting maintenance for all trolleys …", "STEP")

    # Latest record for each (trolleyId, reason) from RAW data (for display dates)
    latest = (
        raw_df.sort_values("createdOn")
              .groupby(["trolleyId", "reason"], as_index=False)
              .last()
    )
    print(f"[PRED] Unique (trolleyId, reason) combos: {len(latest)}")

    # Build feature row using processed df stats
    latest["createdOn"] = pd.to_datetime(latest["createdOn"], dayfirst=True).dt.normalize()
    latest["day"]         = latest["createdOn"].dt.day
    latest["month"]       = latest["createdOn"].dt.month
    latest["day_of_week"] = latest["createdOn"].dt.dayofweek

    # Encode type & reason
    latest["type_enc"]   = le_type.transform(latest["type"])
    latest["reason_enc"] = le_reason.transform(latest["reason"])

    feature_cols = ["trolleyId", "type_enc", "reason_enc", "day", "month", "day_of_week"]
    X_pred = latest[feature_cols].copy()
    X_pred.columns = ["trolleyId", "type", "reason", "day", "month", "day_of_week"]

    print(f"[PRED] Prediction input shape: {X_pred.shape}")
    preds = model.predict(X_pred).round().astype(int)
    preds = np.clip(preds, 1, None)   # at least 1 day

    latest["predicted_days"]     = preds
    latest["nextMaintenanceDate"] = latest["createdOn"] + pd.to_timedelta(latest["predicted_days"], unit="d")

    result = latest[["trolleyId", "type", "reason", "createdOn",
                      "predicted_days", "nextMaintenanceDate"]].copy()
    result = result.sort_values(["trolleyId", "predicted_days"])
    result["createdOn"]          = result["createdOn"].dt.date
    result["nextMaintenanceDate"] = result["nextMaintenanceDate"].dt.date

    print(f"[PRED] Result sample:\n{result.head(10)}")
    log(f"Predictions generated for {len(result):,} (trolley, reason) pairs", "INFO")
    return result

# ─────────────────────────────────────────────
#  MAIN STREAMLIT APP
# ─────────────────────────────────────────────
def main():
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "pipeline_done" not in st.session_state:
        st.session_state.pipeline_done = False

    # Header
    st.markdown("""
    <div class='main-header'>
        🔧 Trolley Predictive Maintenance
        <div class='sub-header'>ML-powered • Auto-retrain • Full fleet coverage</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # DB credentials (override env vars for demo)
        with st.expander("🔌 Database Settings", expanded=False):
            db_host = st.text_input("Host",     value=os.getenv("DB_HOST",     "localhost"))
            db_user = st.text_input("User",     value=os.getenv("DB_USER",     "root"))
            db_pass = st.text_input("Password", value=os.getenv("DB_PASSWORD", ""),
                                    type="password")
            db_name = st.text_input("Database", value=os.getenv("DB_DATABASE", "trolley_db"))
            if st.button("💾 Apply DB Settings"):
                os.environ["DB_HOST"]     = db_host
                os.environ["DB_USER"]     = db_user
                os.environ["DB_PASSWORD"] = db_pass
                os.environ["DB_DATABASE"] = db_name
                st.success("Settings applied!")
                load_data.clear()

        st.markdown("---")
        force_retrain = st.checkbox("🔁 Force Retrain Model", value=False,
                                    help="Ignore existing model.pkl and retrain from scratch")
        st.markdown("---")
        run_btn = st.button("▶ RUN FULL PIPELINE", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Pipeline Logs")
        log_placeholder = st.empty()

    # ── PIPELINE ─────────────────────────────
    if run_btn:
        st.session_state.logs = []
        st.session_state.pipeline_done = False

        with st.spinner("Running pipeline …"):

            # ── LOAD ──────────────────────────
            try:
                raw_df = load_data()
            except Exception as e:
                st.error(f"❌ DB Error: {e}")
                log(f"DB connection failed: {e}", "ERR")
                st.stop()

            # ── EDA ───────────────────────────
            eda_df, bp_fig, null_pct, capping_info = run_eda(raw_df.copy())

            # resolve BEFORE preprocess — so clean reasons flow into everything below
            eda_df, others_report, others_cache = resolve_others(eda_df)
            print(f"[FLOW] Reasons after resolve: {eda_df['reason'].unique().tolist()}")

            proc_df = preprocess(eda_df.copy())

            enc_df, le_type, le_reason = encode(proc_df.copy())

            # ── TRAIN ─────────────────────────    
            if force_retrain or not os.path.exists(MODEL_PATH):
                model, mae, r2 = train_model(enc_df)
            else:
                log("Loading existing model from PKL …", "INFO")
                model = joblib.load(MODEL_PATH)
                y_pred_train = model.predict(
                    enc_df[["trolleyId","type","reason","day","month","day_of_week"]])
                mae = mean_absolute_error(enc_df["difference"], y_pred_train)
                r2  = r2_score(enc_df["difference"], y_pred_train)
                log(f"Existing model loaded — MAE={mae:.2f}  R²={r2:.4f}", "INFO")

            # ── PREDICT ───────────────────────
            pred_df = predict_maintenance(
                eda_df.copy(), enc_df, model, le_type, le_reason)

        st.session_state.pipeline_done  = True
        st.session_state.raw_df        = raw_df
        st.session_state.proc_df       = proc_df
        st.session_state.bp_fig        = bp_fig
        st.session_state.null_pct      = null_pct
        st.session_state.capping_info  = capping_info
        st.session_state.mae           = mae
        st.session_state.r2            = r2
        st.session_state.pred_df       = pred_df
        st.session_state.le_type       = le_type
        st.session_state.le_reason     = le_reason
        st.session_state.others_report = others_report

    # ── UPDATE LOG BOX ────────────────────────
    log_html = "<br>".join(st.session_state.logs[-40:]) if st.session_state.logs else "No logs yet."
    log_placeholder.markdown(f"<div class='log-box'>{log_html}</div>", unsafe_allow_html=True)

    # ── RESULTS SECTION ───────────────────────
    if st.session_state.pipeline_done:
        raw_df        = st.session_state.raw_df
        proc_df       = st.session_state.proc_df
        bp_fig        = st.session_state.bp_fig
        null_pct      = st.session_state.null_pct
        capping_info  = st.session_state.capping_info
        mae           = st.session_state.mae
        r2            = st.session_state.r2
        pred_df       = st.session_state.pred_df
        others_report = st.session_state.get("others_report", pd.DataFrame())

        # ── KPI CARDS ─────────────────────────
        st.markdown("<div class='section-title'>📊 Pipeline Summary</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            ("Total Records",      f"{len(raw_df):,}",               c1),
            ("Unique Trolleys",    f"{raw_df['trolleyId'].nunique():,}", c2),
            ("Unique Reasons",     f"{raw_df['reason'].nunique():,}", c3),
            ("Model MAE",          f"{mae:.2f} days",                 c4),
            ("R² Score",           f"{r2:.4f}",                       c5),
        ]
        for label, val, col in cards:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='val'>{val}</div>
                <div class='lbl'>{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── EDA TAB VIEW ──────────────────────
        tab1, tab2, tab3 = st.tabs(["🔍 EDA", "📈 Predictions", "📋 Raw Data"])

        with tab1:
            st.markdown("<div class='section-title'>Null Value Report</div>",
                        unsafe_allow_html=True)
            null_df = pd.DataFrame({
                "Column": null_pct.index,
                "Null %": null_pct.values.round(2),
                "Action": [
                    "—" if p == 0 else ("Drop rows" if p < 30 else "Fill mean/mode")
                    for p in null_pct.values
                ]
            })
            st.dataframe(null_df, use_container_width=True, hide_index=True)

            st.markdown("<div class='section-title'>Outlier Detection — Box Plots</div>",
                        unsafe_allow_html=True)
            st.pyplot(bp_fig, use_container_width=True)

            if capping_info:
                st.markdown("<div class='section-title'>Capping Applied</div>",
                            unsafe_allow_html=True)
                for col, info in capping_info.items():
                    st.markdown(
                        f"<div class='alert-warn'>Column <b>{col}</b>: "
                        f"{info['count']} outliers capped → "
                        f"[{info['lower']:.2f}, {info['upper']:.2f}]</div>",
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='alert-ok'>✔ No outlier capping needed in any column</div>",
                    unsafe_allow_html=True)

            # ── OTHERS INTENT REPORT ──────────────────
            st.markdown("<div class='section-title'>🤖 'Others' Free-Text Intent Resolution</div>",
                        unsafe_allow_html=True)
            if others_report is not None and not others_report.empty:
                total_rows  = others_report["rows_affected"].sum()
                total_unique= len(others_report)
                api_calls   = (others_report["source"] == "claude_api").sum()
                cache_hits  = (others_report["source"] == "cache").sum()

                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.markdown(f"<div class='metric-card'><div class='val' style='color:#ffa500'>{total_rows:,}</div><div class='lbl'>Rows Resolved</div></div>", unsafe_allow_html=True)
                oc2.markdown(f"<div class='metric-card'><div class='val' style='color:#ffa500'>{total_unique}</div><div class='lbl'>Unique Texts</div></div>", unsafe_allow_html=True)
                oc3.markdown(f"<div class='metric-card'><div class='val' style='color:#00e5ff'>{cache_hits}</div><div class='lbl'>Cache Hits</div></div>", unsafe_allow_html=True)
                oc4.markdown(f"<div class='metric-card'><div class='val' style='color:#3fb950'>{api_calls}</div><div class='lbl'>API Calls Made</div></div>", unsafe_allow_html=True)

                st.markdown("")
                st.dataframe(
                    others_report.rename(columns={
                        "original_text": "Original Free Text (from DB)",
                        "mapped_to":     "→ Mapped Category",
                        "rows_affected": "Rows Affected",
                        "source":        "Resolved Via",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown(
                    "<div class='alert-ok'>✔ All 'Others' rows replaced with valid categories "
                    "before encoding & training — model never sees free-text.</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='alert-ok'>✔ No free-text 'Others' rows found in this dataset — "
                    "all reasons are already valid categories.</div>",
                    unsafe_allow_html=True)

            # Feature distributions
            st.markdown("<div class='section-title'>Feature Distributions (processed)</div>",
                        unsafe_allow_html=True)
            dist_cols = ["day", "month", "day_of_week", "difference"]
            dist_cols = [c for c in dist_cols if c in proc_df.columns]
            fig2, axes2 = plt.subplots(1, len(dist_cols),
                                       figsize=(4.5 * len(dist_cols), 3.5),
                                       facecolor="#0d1117")
            for ax, col in zip(axes2, dist_cols):
                ax.set_facecolor("#161b22")
                ax.hist(proc_df[col].dropna(), bins=25,
                        color="#1f6feb", edgecolor="#58a6ff", alpha=0.85)
                ax.set_title(col, color="#c9d1d9", fontsize=9, fontweight="bold")
                ax.tick_params(colors="#8b949e", labelsize=7)
                ax.spines[:].set_color("#21262d")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        with tab2:
            st.markdown("<div class='section-title'>🔮 Predictive Maintenance Schedule — All Trolleys</div>",
                        unsafe_allow_html=True)

            # Filters
            fc1, fc2, fc3 = st.columns([2, 2, 2])
            with fc1:
                t_ids = ["All"] + sorted(pred_df["trolleyId"].unique().tolist())
                sel_t = st.selectbox("Filter by Trolley ID", t_ids)
            with fc2:
                reasons = ["All"] + sorted(pred_df["reason"].unique().tolist())
                sel_r   = st.selectbox("Filter by Reason", reasons)
            with fc3:
                max_days = int(pred_df["predicted_days"].max())
                day_range = st.slider("Max predicted days", 1, max(max_days, 30),
                                      max(max_days, 30))

            disp = pred_df.copy()
            if sel_t != "All":
                disp = disp[disp["trolleyId"] == sel_t]
            if sel_r != "All":
                disp = disp[disp["reason"] == sel_r]
            disp = disp[disp["predicted_days"] <= day_range]

            # Urgency label
            today = pd.Timestamp.today().normalize().date()
            def urgency(row):
                try:
                    delta = (pd.Timestamp(row["nextMaintenanceDate"]).date() - today).days
                except:
                    delta = 999
                if delta <= 3:   return "🔴 CRITICAL"
                if delta <= 7:   return "🟠 WARNING"
                return "🟢 OK"

            disp["urgency"] = disp.apply(urgency, axis=1)
            disp = disp.sort_values(["urgency", "predicted_days"])

            st.dataframe(
                disp.drop(columns=["urgency"]).rename(columns={
                    "trolleyId":           "Trolley ID",
                    "type":                "Type",
                    "reason":              "Reason",
                    "createdOn":           "Last Maintenance",
                    "predicted_days":      "Days Until Next",
                    "nextMaintenanceDate": "Next Maintenance Date",
                }),
                use_container_width=True,
                hide_index=True,
                height=500,
            )

            # Summary stats
            st.markdown("")
            s1, s2, s3 = st.columns(3)
            crit = (disp["urgency"] == "🔴 CRITICAL").sum()
            warn = (disp["urgency"] == "🟠 WARNING").sum()
            ok   = (disp["urgency"] == "🟢 OK").sum()
            s1.markdown(f"<div class='metric-card'><div class='val' style='color:#ff4b4b'>{crit}</div><div class='lbl'>Critical (≤3 days)</div></div>", unsafe_allow_html=True)
            s2.markdown(f"<div class='metric-card'><div class='val' style='color:#ffa500'>{warn}</div><div class='lbl'>Warning (≤7 days)</div></div>", unsafe_allow_html=True)
            s3.markdown(f"<div class='metric-card'><div class='val' style='color:#00e5ff'>{ok}</div><div class='lbl'>OK (>7 days)</div></div>", unsafe_allow_html=True)

            # Download CSV
            st.markdown("")
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Full Predictions CSV",
                               data=csv,
                               file_name="trolley_maintenance_predictions.csv",
                               mime="text/csv")

            # Bar: top 10 trolleys with earliest maintenance
            st.markdown("<div class='section-title'>Top 10 Trolleys — Earliest Maintenance Needed</div>",
                        unsafe_allow_html=True)
            top10 = (pred_df.groupby("trolleyId")["predicted_days"]
                            .min()
                            .nsmallest(10)
                            .reset_index())
            fig3, ax3 = plt.subplots(figsize=(10, 3.5), facecolor="#0d1117")
            ax3.set_facecolor("#161b22")
            bars = ax3.barh(top10["trolleyId"].astype(str),
                            top10["predicted_days"],
                            color=["#ff4b4b" if d <= 3 else "#ffa500" if d <= 7 else "#1f6feb"
                                   for d in top10["predicted_days"]])
            ax3.set_xlabel("Days Until Next Maintenance", color="#8b949e")
            ax3.tick_params(colors="#8b949e", labelsize=8)
            ax3.spines[:].set_color("#21262d")
            ax3.invert_yaxis()
            legend_patches = [
                mpatches.Patch(color="#ff4b4b", label="Critical ≤3d"),
                mpatches.Patch(color="#ffa500", label="Warning ≤7d"),
                mpatches.Patch(color="#1f6feb", label="OK >7d"),
            ]
            ax3.legend(handles=legend_patches, facecolor="#0d1117",
                       labelcolor="#c9d1d9", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

        with tab3:
            st.markdown("<div class='section-title'>Raw DB Data (first 500 rows)</div>",
                        unsafe_allow_html=True)
            st.dataframe(raw_df.head(500), use_container_width=True, hide_index=True)

    else:
        # Welcome state
        st.info("👈 Configure DB settings in the sidebar (if needed) then click **▶ RUN FULL PIPELINE**")
        st.markdown("""
        <div style='background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:2rem;margin-top:1rem;'>
        <h4 style='color:#00e5ff;font-family:Syne,sans-serif'>What this app does</h4>
        <ol style='color:#8b949e;font-family:JetBrains Mono,monospace;font-size:0.85rem;line-height:2'>
            <li>🔌 Connects to MySQL and loads trolley maintenance data</li>
            <li>🔍 EDA — null analysis (drop &lt;30% / fill ≥30%) + boxplot outlier capping</li>
            <li>⚙️ Feature engineering — day, month, day_of_week, next-issue gap</li>
            <li>🏷️ Label-encodes type & reason (loads/saves PKLs automatically)</li>
            <li>🌲 Trains a Random Forest Regressor on full data (200 trees)</li>
            <li>🔮 Predicts days until next maintenance for every (trolleyId × reason)</li>
            <li>🚨 Flags critical / warning / OK urgency in an interactive table</li>
        </ol>
        <p style='color:#8b949e;font-size:0.8rem;font-family:JetBrains Mono,monospace'>
        Re-run anytime new data is added to DB to auto-retrain the model.
        </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()