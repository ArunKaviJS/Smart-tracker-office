import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE    = "automated_kpi_insights"

# ── Azure OpenAI ──────────────────────────────────────────────
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ── Pipeline Thresholds ───────────────────────────────────────
ROW_THRESHOLD   = 20      # rows above this → CSV; below → LLM text
MAX_SQL_RETRIES = 3       # retry SQL generation on execution failure
SAMPLE_ROWS     = 3       # sample rows per table in schema prompt
MAX_TOKENS_SQL  = 600
MAX_TOKENS_ANSWER = 350
EXPORT_DIR      = Path("./exports")

# ── Column-level noise filter (excluded from LLM schema context) ──
# These columns confuse the LLM or carry binary/audit data.
SKIP_COLUMN_NAMES: set[str] = {
    "boundary", "boundaryColor",
    "wifiRTT", "wifiGeoLocation",
    "Accelerometer", "imsi", "OtaTrigger",
    "nestId", "address",
    "createdOn", "modifiedOn",
}

# SQL keywords that must never appear in a generated query
FORBIDDEN_SQL_KEYWORDS: set[str] = {
    "DROP", "DELETE", "UPDATE", "INSERT",
    "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE",
}