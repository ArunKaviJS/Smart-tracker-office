import os
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv()


def get_engine():
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT", 3306)
    DATABASE_NAME = "automated_kpi_insights"

    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DATABASE_NAME}"
    )
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return engine


def run_query(sql: str, params: dict = None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame. Returns empty DataFrame on error."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return pd.DataFrame()


@lru_cache(maxsize=32)
def get_table_columns(table_name: str) -> list:
    """
    Return the actual column names for a table by running DESCRIBE.
    Results are cached per table name for the process lifetime.
    Returns an empty list on error so callers can degrade gracefully.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(f"DESCRIBE `{table_name}`"))
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        print(f"[DB SCHEMA ERROR] Could not DESCRIBE {table_name}: {e}")
        return []


def table_has_column(table_name: str, column: str) -> bool:
    """Convenience check: does this table have a specific column?"""
    return column in get_table_columns(table_name)