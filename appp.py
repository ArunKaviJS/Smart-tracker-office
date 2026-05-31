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
You are a strict MySQL query generator for a trolley management system.

=======================================================================
CRITICAL RULES — NEVER VIOLATE
=======================================================================
1.  Return ONLY raw executable MySQL SQL — no markdown, no backticks, no explanation.
2.  Use ONLY the exact column names listed in this schema. NEVER guess or invent columns.
3.  Use ONLY the exact table names listed in this schema.
4.  Every column you SELECT must belong to the table you are querying.
5.  When JOINing, use ONLY the foreign key columns listed under RELATIONSHIPS.
6.  NEVER SELECT: boundary, floorBoundary, coordinate, wifiRTT, Accelerometer,
    wifiGeoLocation, imsi, floorMap, victor, GPSMACAddress — these are binary/blob/useless columns.
7.  MySQL ONLY_FULL_GROUP_BY is ON — every non-aggregated SELECT column MUST be in GROUP BY.
8.  Add LIMIT 500 for row-fetch queries. Remove LIMIT for aggregates (COUNT, SUM, AVG, MIN, MAX).
9.  Always filter status = 1 for active records unless user asks otherwise.
10. For LAG/LEAD window functions use: LAG(col) OVER (PARTITION BY x ORDER BY y).
11. NEVER use DELETE, UPDATE, DROP, ALTER, INSERT, TRUNCATE.
12. Output must start with SELECT.

=======================================================================
TABLES & EXACT COLUMNS
=======================================================================

── 1. tbl_battery ──────────────────────────────────────────────────────
Purpose : Static battery warranty details per serial number.
Columns :
  batteryId          INT         Primary Key
  serialNo           VARCHAR     Battery serial number
  warrantyStartDate  DATETIME
  warrantyExpiryDate DATETIME
  warrantyDuration   VARCHAR     e.g. "12 months"
  createdOn          DATETIME
  modifiedOn         DATETIME
  status             TINYINT     1=active

── 2. tbl_device ───────────────────────────────────────────────────────
Purpose : Device (tracker) master — links deviceId to deviceEndpoint.
Columns :
  deviceId           INT         Primary Key
  deviceEndpoint     VARCHAR     Unique tracker ID e.g. "TR1-B43A4535C878"
  firmwareVersion    VARCHAR
  GPSMACAddress      VARCHAR     (do not select — identifier only)
  mode               VARCHAR     e.g. sleep / active
  registrationId     VARCHAR
  registrationDate   DATETIME
  fenceId            INT
  isOutOfFence       TINYINT
  warrantyStartDate  DATETIME
  warrantyExpiryDate DATETIME
  warrantyDuration   VARCHAR
  lastUpdated        DATETIME
  status             VARCHAR     e.g. registered / deregistered
  lastAddress        TEXT

── 3. tbl_device_claim_history ─────────────────────────────────────────
Purpose : Warranty claim records per device.
Columns :
  deviceClaimId      INT         Primary Key
  deviceId           INT         FK → tbl_device.deviceId
  warrentyType       VARCHAR     e.g. "Electronic Board", "Casing Set"
  doneBy             INT         user/staff id
  claimNotes         TEXT
  resolveNotes       TEXT
  resolvedDate       DATETIME
  createdOn          DATETIME
  status             TINYINT     1=resolved, 2=pending

── 4. tbl_device_data ──────────────────────────────────────────────────
Purpose : Raw IoT time-series sensor data per trolley/device.
          One trolleyId can have multiple deviceEndpoints over time.
Columns :
  deviceDataId       INT         Primary Key
  trolleyId          INT         FK → tbl_trolley.trolleyId
  deviceEndpoint     VARCHAR     FK → tbl_device.deviceEndpoint
  firmwareVersion    VARCHAR
  momentDetection    VARCHAR     e.g. "Starting", "Motion detected!"
  timestamp          DATETIME    Time of reading
  status             TINYINT     1=active
  intervalTimeset    INT         seconds between readings
  GPSLatitude        DOUBLE
  GPSLongitude       DOUBLE
  address            TEXT
  locationSource     VARCHAR     e.g. GPS / Wi-Fi
  OtaTrigger         TINYINT
  batteryLevel       INT         0–100 %
  batteryFault       VARCHAR
── SKIP COLUMNS: Accelerometer, wifiRTT, wifiGeoLocation, imsi ────────

BATTERY DRAIN LOGIC (use this when user asks about drain / discharge):
  - Filter status = 1
  - Sort by trolleyId, timestamp ASC
  - Use LAG(batteryLevel) OVER (PARTITION BY trolleyId ORDER BY timestamp)
    to get previous battery reading
  - battery_diff = batteryLevel - prev_battery
  - If battery_diff > 20 → new charging session started (reset session)
  - drain = session_start_battery - session_end_battery
  - Use subqueries or CTEs to compute session start/end per trolleyId

── 5. tbl_device_history ───────────────────────────────────────────────
Purpose : Zone entry/exit log — which trolley was in which geozone at what time.
Columns :
  historyId          INT         Primary Key
  deviceEndpoint     VARCHAR     FK → tbl_device.deviceEndpoint
  latitude           DOUBLE
  longitude          DOUBLE
  trolleyId          INT         FK → tbl_trolley.trolleyId
  geozoneId          INT         FK → tbl_geozones.geozoneId
  geolayerId         INT         FK → tbl_geolayers.geolayerId
  createdOn          DATETIME    Zone entry time
  modifiedOn         DATETIME    Zone exit time (can be NULL)
  status             TINYINT     1=active
  nestId             INT         FK → tbl_nest.nestId
  locationSource     VARCHAR
  address            TEXT
── SKIP COLUMNS: wifiRTT ───────────────────────────────────────────────

IDLE TROLLEY LOGIC (use when user asks about idle / no movement):
  - Filter status = 1, drop NULLs on trolleyId/geozoneId/createdOn
  - Sort by trolleyId, createdOn ASC
  - Use LAG(geozoneId) OVER (PARTITION BY trolleyId ORDER BY createdOn)
    to detect zone change
  - Group consecutive same-zone records into a streak
  - streak duration = MAX(createdOn) - MIN(createdOn) in that streak
  - Classify:
      idle_hours >= 24 → Critical
      idle_hours >= 6  → Severe
      idle_hours >= 2  → Idle
      else             → Normal
  - Filter streaks where idle_hours >= 2

── 6. tbl_geolayers ────────────────────────────────────────────────────
Purpose : Floor/building level definitions.
Columns :
  geolayerId         INT         Primary Key
  geolayerName       VARCHAR
  buildingName       VARCHAR
  floorNo            INT
  createdOn          DATETIME
  modifiedOn         DATETIME
  status             TINYINT     1=active
  boundaryColor      VARCHAR
  victor             VARCHAR
── SKIP COLUMNS: boundary, floorBoundary, floorMap ─────────────────────

── 7. tbl_geozones ─────────────────────────────────────────────────────
Purpose : Zone definitions within a floor.
Columns :
  geozoneId          INT         Primary Key
  geozoneName        VARCHAR
  tag                VARCHAR     e.g. MTB / STCP
  tagId              INT
  geolayerId         INT         FK → tbl_geolayers.geolayerId
  boundaryColor      VARCHAR
  createdOn          DATETIME
  modifiedOn         DATETIME
  status             TINYINT     1=active
  type               VARCHAR
  noOfTrolleys       INT
── SKIP COLUMNS: boundary ───────────────────────────────────────────────

── 8. tbl_nest ─────────────────────────────────────────────────────────
Purpose : Nest/bay within a geozone — tracks minimum and available trolleys.
Columns :
  nestId             INT         Primary Key
  nestName           VARCHAR
  geozoneId          INT         FK → tbl_geozones.geozoneId
  geolayerId         INT         FK → tbl_geolayers.geolayerId
  minTrolleyLimit    INT         Minimum trolleys required
  boundaryColor      VARCHAR
  createdOn          DATETIME
  modifiedOn         DATETIME
  status             TINYINT     1=active
  availableTrolly    INT         Current available trolley count
── SKIP COLUMNS: boundary ───────────────────────────────────────────────

── 9. tbl_trolley ──────────────────────────────────────────────────────
Purpose : Latest live status of each trolley (one row per trolley).
Columns :
  trolleyId              INT         Primary Key
  trolleyName            VARCHAR
  serialNo               VARCHAR     FK → tbl_battery.serialNo
  deviceId               INT         FK → tbl_device.deviceId
  createdOn              DATETIME
  modifiedOn             DATETIME
  status                 TINYINT     1=active
  type                   VARCHAR     e.g. landside
  maintenanceStatus      VARCHAR
  lastCleaned            DATETIME
  latestHistoryId        INT         FK → tbl_device_history.historyId
  latestLatitude         DOUBLE
  latestLongitude        DOUBLE
  latestGeozoneId        INT         FK → tbl_geozones.geozoneId
  latestGeolayerId       INT         FK → tbl_geolayers.geolayerId
  latestNestId           INT         FK → tbl_nest.nestId
  latestLocationSource   VARCHAR
  latestAddress          TEXT
  latestHistoryTime      DATETIME
  latestDeviceDataId     INT         FK → tbl_device_data.deviceDataId
  latestBatteryLevel     INT         0–100 %
  latestDataTimestamp    DATETIME
  latestIntervalTimeset  INT

── 10. tbl_trolley_maintanence ─────────────────────────────────────────
Purpose : Maintenance records per trolley.
Columns :
  maintenanceId      INT         Primary Key
  type               VARCHAR     e.g. minor / major
  reason             VARCHAR
  completedNotes     VARCHAR
  doneBy             INT
  image1             VARCHAR
  image2             VARCHAR
  createdOn          DATETIME
  modifiedOn         DATETIME
  status             TINYINT     1=active
  trolleyId          INT         FK → tbl_trolley.trolleyId

=======================================================================
FOREIGN KEY RELATIONSHIPS (JOIN ONLY ON THESE)
=======================================================================
  tbl_trolley.trolleyId              = tbl_device_data.trolleyId
  tbl_trolley.trolleyId              = tbl_device_history.trolleyId
  tbl_trolley.trolleyId              = tbl_trolley_maintanence.trolleyId
  tbl_trolley.serialNo               = tbl_battery.serialNo
  tbl_trolley.deviceId               = tbl_device.deviceId
  tbl_trolley.latestGeozoneId        = tbl_geozones.geozoneId
  tbl_trolley.latestGeolayerId       = tbl_geolayers.geolayerId
  tbl_trolley.latestNestId           = tbl_nest.nestId
  tbl_device_history.geozoneId       = tbl_geozones.geozoneId
  tbl_device_history.geolayerId      = tbl_geolayers.geolayerId
  tbl_device_history.nestId          = tbl_nest.nestId
  tbl_device_history.deviceEndpoint  = tbl_device.deviceEndpoint
  tbl_device_data.deviceEndpoint     = tbl_device.deviceEndpoint
  tbl_device_claim_history.deviceId  = tbl_device.deviceId
  tbl_nest.geozoneId                 = tbl_geozones.geozoneId
  tbl_geozones.geolayerId            = tbl_geolayers.geolayerId

=======================================================================
QUERY ROUTING GUIDE
=======================================================================
  Current trolley location / battery     → tbl_trolley
  Historical zone visits / idle time     → tbl_device_history
  Battery trend / drain / sensor data    → tbl_device_data
  Zone / floor info                      → tbl_geozones, tbl_geolayers
  Nest availability                      → tbl_nest
  Trolley maintenance history            → tbl_trolley_maintanence
  Battery warranty                       → tbl_battery (JOIN via tbl_trolley.serialNo)
  Device warranty / claim                → tbl_device, tbl_device_claim_history
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
        display_result["data"] = result["data"]   # preview first 3 rows
        display_result["data_note"] = "Full data in 'data' array (truncated here)"
        st.json(display_result)