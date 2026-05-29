"""
Trolley Management Chatbot
- Natural language → SQL (via Azure OpenAI)
- Smart routing: small result → LLM narrates | large result → table/CSV
"""

import os
import json
import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────
# 1. DATABASE CONNECTION
# ─────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:3306/automated_kpi_insights"
    )

# ─────────────────────────────────────────
# 2. AZURE OPENAI CLIENT
# ─────────────────────────────────────────
@st.cache_resource
def get_llm_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ─────────────────────────────────────────
# 3. SCHEMA CONTEXT  (fed to LLM every call)
# ─────────────────────────────────────────
SCHEMA_CONTEXT = """
You are an expert MySQL query generator for a trolley-tracking system.

== TABLES ==

tbl_battery          — Battery warranty per serial number
  batteryId (PK), serialNo, warrantyStartDate, warrantyExpiryDate,
  warrantyDuration, createdOn, modifiedOn, status(1=active)

tbl_device_data      — Raw IoT sensor readings (high-volume, time-series)
  deviceDataId (PK), trolleyId (FK→tbl_trolley), deviceEndpoint,
  firmwareVersion, momentDetection, timestamp, batteryLevel(0-100 %),
  batteryFault, GPSLatitude, GPSLongitude, locationSource, status

tbl_device_history   — Zone entry/exit log per trolley
  historyId (PK), deviceEndpoint ,
  trolleyId (FK→tbl_trolley), geozoneId (FK→tbl_geozones),
  geolayerId, latitude, longitude, createdOn (entry time),
  modifiedOn (exit time), address, locationSource, status

tbl_geozones         — Zone definitions (floors / areas)
  geozoneId (PK), geozoneName, tag (e.g. MTB/STCP), tagId,
  geolayerId, noOfTrolleys, type, status, createdOn, modifiedOn
  !! NEVER SELECT boundary column — it's raw binary data !!

tbl_trolley          — Latest live status of each trolley
  trolleyId (PK), latestdeviceDataId, latestgeozoneId,
  latestBatteryLevel (latest %), serialNo

== KEY RELATIONSHIPS ==
  tbl_trolley.trolleyId       = tbl_device_data.trolleyId
                               = tbl_device_history.trolleyId
  tbl_trolley.deviceEndpoint  = tbl_device_data.deviceEndpoint
                               = tbl_device_history.deviceEndpoint
  tbl_device_history.geozoneId = tbl_geozones.geozoneId
  tbl_trolley.serialNo         = tbl_battery.serialNo
  tbl_device_history.geolayerId = tbl_geozones.geolayerId

== QUERY RULES ==
  • For CURRENT trolley state → use tbl_trolley (+ JOIN tbl_geozones for zone name)
  • For HISTORICAL zone visits  → use tbl_device_history
  • For SENSOR / battery trend  → use tbl_device_data
  • For WARRANTY info           → JOIN tbl_trolley + tbl_battery on serialNo
  • Always add LIMIT 500 unless user specifies a smaller number
  • For aggregates (count, avg, min, max) remove the LIMIT
  • Return ONLY the raw SQL — no markdown, no backticks, no explanation
"""

# ─────────────────────────────────────────
# 4. STEP A — Generate SQL
# ─────────────────────────────────────────
def generate_sql(client: AzureOpenAI, question: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": SCHEMA_CONTEXT},
            {"role": "user",   "content": f"Write a MySQL query for: {question}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # strip accidental markdown fences
    raw = raw.replace("```sql", "").replace("```", "").strip()
    return raw

# ─────────────────────────────────────────
# 5. STEP B — Execute SQL
# ─────────────────────────────────────────
def run_query(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df

# ─────────────────────────────────────────
# 6. STEP C — Route the result
#    Small  (≤10 rows, ≤4 cols) → LLM narrates → text response
#    Large  (>10 rows OR >4 cols) → show as table + offer CSV download
# ─────────────────────────────────────────
SMALL_ROWS = 10
SMALL_COLS = 4

def narrate(client: AzureOpenAI, question: str, df: pd.DataFrame) -> str:
    data_txt = df.to_string(index=False)
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for a trolley management system. "
                    "Answer the user's question in 1-3 clear sentences using the "
                    "database result provided. Be direct and factual."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nDatabase result:\n{data_txt}",
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def route(client: AzureOpenAI, question: str, df: pd.DataFrame) -> dict:
    """
    Returns a structured dict — this is what the API will return to frontend.

    Schema:
    {
        "response_type": "text" | "table",
        "message":       str,           # always present
        "sql_query":     str,           # generated SQL
        "row_count":     int,
        "columns":       list[str],     # present for table type
        "data":          list[dict]     # present for table type  (JSON-serialisable)
    }
    """
    rows, cols = df.shape
    is_small = (rows <= SMALL_ROWS and cols <= SMALL_COLS)

    if rows == 0:
        return {
            "response_type": "text",
            "message": "No data found for your query.",
            "row_count": 0,
            "columns": [],
            "data": [],
        }

    if is_small:
        answer = narrate(client, question, df)
        return {
            "response_type": "text",
            "message": answer,
            "row_count": rows,
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),   # supporting data still included
        }
    else:
        return {
            "response_type": "table",
            "message": f"Found {rows} records. Showing as table below.",
            "row_count": rows,
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }

# ─────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────
def process_question(question: str) -> dict:
    client = get_llm_client()
    engine = get_engine()

    sql = generate_sql(client, question)

    try:
        df = run_query(engine, sql)
    except Exception as e:
        return {
            "response_type": "error",
            "message": f"SQL execution failed: {e}",
            "sql_query": sql,
            "row_count": 0,
            "columns": [],
            "data": [],
        }

    result = route(client, question, df)
    result["sql_query"] = sql
    return result

# ─────────────────────────────────────────
# 8. STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(page_title="Trolley Assistant", page_icon="🛒", layout="wide")

st.markdown("""
    <style>
        .stTextInput > div > div > input { font-size: 1.05rem; }
        .sql-box {
            background: #1e1e2e; color: #cdd6f4; padding: 12px 16px;
            border-radius: 8px; font-family: monospace; font-size: 0.85rem;
            white-space: pre-wrap; margin-top: 8px;
        }
        .badge-text  { background:#d1fae5; color:#065f46; padding:2px 10px; border-radius:20px; font-size:0.82rem; }
        .badge-table { background:#dbeafe; color:#1e40af; padding:2px 10px; border-radius:20px; font-size:0.82rem; }
        .badge-error { background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:20px; font-size:0.82rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 Trolley Management Assistant")
st.caption("Ask anything about trolleys, geozones, battery status, or warranty in plain English.")

question = st.text_input(
    "Your question",
    placeholder="e.g. Which geozone has the most trolleys? / Show all trolleys with battery below 20%",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Generating query and fetching data..."):
        result = process_question(question.strip())

    # ── Show generated SQL ──
    with st.expander("🔍 Generated SQL", expanded=False):
        st.markdown(f'<div class="sql-box">{result.get("sql_query","")}</div>', unsafe_allow_html=True)

    rtype = result["response_type"]

    if rtype == "error":
        st.markdown('<span class="badge-error">ERROR</span>', unsafe_allow_html=True)
        st.error(result["message"])

    elif rtype == "text":
        st.markdown('<span class="badge-text">TEXT ANSWER</span>', unsafe_allow_html=True)
        st.success(result["message"])

        # supporting small table still shown beneath
        if result["data"]:
            with st.expander("Raw data", expanded=False):
                st.dataframe(pd.DataFrame(result["data"]), use_container_width=True)

    elif rtype == "table":
        st.markdown('<span class="badge-table">TABLE</span>', unsafe_allow_html=True)
        st.info(result["message"])

        df_show = pd.DataFrame(result["data"])
        st.dataframe(df_show, use_container_width=True, height=420)

        # CSV download
        csv_bytes = df_show.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="query_result.csv",
            mime="text/csv",
        )

    # ── Structured JSON (for frontend integration reference) ──
    with st.expander("📦 Structured JSON output (for frontend)", expanded=False):
        display_result = {k: v for k, v in result.items() if k != "data"}
        display_result["data"] = result["data"][:3]   # preview first 3 rows
        display_result["data_note"] = "Full data in 'data' array (truncated here)"
        st.json(display_result)