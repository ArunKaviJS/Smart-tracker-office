"""
Battery Warranty KPI
--------------------
  - Parse warrantyExpiryDate and compute days_remaining.
  - Classify each battery: Healthy / Warning / Critical / Expired.
  - Correlate with batteryFault from tbl_device_data via device endpoints.
  - Flag accelerated discharge (batteryLevel drops rapidly).
"""

import pandas as pd
from db.connection import run_query
from utils.preprocess import (
    parse_dates, classify_warranty_status, has_active_fault,
)


def _fetch_batteries() -> pd.DataFrame:
    sql = """
        SELECT
            batteryId,
            serialNo,
            warrantyStartDate,
            warrantyExpiryDate,
            warrantyDuration,
            createdOn,
            modifiedOn,
            status
        FROM tbl_battery
        WHERE status = 1
    """
    df = run_query(sql)
    if df.empty:
        return df
    df = parse_dates(df, ["warrantyStartDate", "warrantyExpiryDate", "createdOn", "modifiedOn"])
    return df


def _fetch_device_fault_summary() -> pd.DataFrame:
    """
    Summarise battery-level and fault info per device from device_data.
    We don't have a direct batteryId → deviceEndpoint link in the schema,
    but we include this for dashboard cross-analysis.
    """
    sql = """
        SELECT
            deviceEndpoint,
            AVG(batteryLevel)    AS avg_battery_level,
            MIN(batteryLevel)    AS min_battery_level,
            COUNT(*)             AS reading_count,
            MAX(timestamp)       AS last_seen,
            MAX(batteryFault)    AS last_fault
        FROM tbl_device_data
        WHERE status = 1
        GROUP BY deviceEndpoint
    """
    df = run_query(sql)
    if df.empty:
        return df
    df = parse_dates(df, ["last_seen"])
    return df


def get_battery_warranty_data() -> pd.DataFrame:
    """
    Returns processed battery DataFrame with:
        batteryId, serialNo, warrantyStartDate, warrantyExpiryDate,
        days_remaining, warranty_status, has_fault
    """
    df = _fetch_batteries()
    if df.empty:
        return df

    now = pd.Timestamp.now()
    df["days_remaining"] = (df["warrantyExpiryDate"] - now).dt.days
    df["warranty_status"] = df["days_remaining"].apply(classify_warranty_status)

    # Validate stored duration vs computed duration
    def _validate_duration(row):
        if pd.isna(row["warrantyStartDate"]) or pd.isna(row["warrantyExpiryDate"]):
            return False
        computed_days = (row["warrantyExpiryDate"] - row["warrantyStartDate"]).days
        return abs(computed_days - 365) > 5  # 5-day tolerance for 12-month

    df["duration_mismatch"] = df.apply(_validate_duration, axis=1)

    return df


def get_device_fault_data() -> pd.DataFrame:
    """Returns device-level battery fault summary from device_data."""
    df = _fetch_device_fault_summary()
    if df.empty:
        return df
    df["has_fault"] = df["last_fault"].apply(has_active_fault)
    return df


def get_battery_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    total    = len(df)
    expired  = (df["warranty_status"] == "Expired").sum()
    critical = (df["warranty_status"] == "Critical").sum()
    warning  = (df["warranty_status"] == "Warning").sum()
    healthy  = (df["warranty_status"] == "Healthy").sum()
    mismatch = df.get("duration_mismatch", pd.Series(dtype=bool)).sum()

    avg_days = df["days_remaining"].mean()

    return {
        "total_batteries": total,
        "expired":         int(expired),
        "critical":        int(critical),
        "warning":         int(warning),
        "healthy":         int(healthy),
        "avg_days_left":   round(avg_days, 0) if pd.notna(avg_days) else 0,
        "date_mismatches": int(mismatch),
    }


def get_warranty_status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count per warranty_status — for donut / bar chart.
    Uses groupby().size() to avoid duplicate column names in pandas >=2.0
    where value_counts().reset_index() produces two columns both named 'count'.
    """
    if df.empty:
        return pd.DataFrame()
    order = ["Expired", "Critical", "Warning", "Healthy", "Unknown"]
    result = (
        df.groupby("warranty_status", observed=True)
        .size()
        .rename("count")
        .reindex(order)
        .dropna()
        .astype(int)
        .reset_index()
    )
    # result now has columns: warranty_status, count  — guaranteed unique
    return result


def get_expiry_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly expiry counts — for trend line."""
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2["expiry_month"] = df2["warrantyExpiryDate"].dt.to_period("M").astype(str)
    return (
        df2.groupby("expiry_month")
        .size()
        .reset_index(name="count")
        .sort_values("expiry_month")
    )