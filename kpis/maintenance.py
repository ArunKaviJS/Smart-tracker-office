"""
Trolley Maintenance KPI
-----------------------
Key design decision — resolution time via CONSECUTIVE createdOn:
  - Sort all maintenance records per trolley by createdOn ASC.
  - The "job duration" for record N = createdOn[N+1] − createdOn[N].
  - The LAST record per trolley (no next entry) = currently open job.
  - This avoids relying on modifiedOn which is unreliable.

SLA thresholds:
  - Minor maintenance: 24 hours
  - Major maintenance: 72 hours
"""

import pandas as pd
from db.connection import run_query
from utils.preprocess import (
    parse_dates, deduplicate, normalize_maintenance_reason
)

SLA_HOURS = {"minor": 24, "major": 72}


def _fetch_maintenance() -> pd.DataFrame:
    # tbl_trolley uses latestGeozoneId (not geozoneId) for the trolley's
    # current zone — confirmed from actual table schema.
    sql = """
        SELECT
            m.maintenanceId,
            m.trolleyId,
            m.type,
            m.reason,
            m.completedNotes,
            m.doneBy,
            m.createdOn,
            m.modifiedOn,
            m.status,
            t.latestGeozoneId  AS geozoneId,
            g.geozoneName,
            g.tag
        FROM tbl_trolley_maintenance AS m
        LEFT JOIN tbl_trolley AS t
            ON m.trolleyId = t.trolleyId
        LEFT JOIN tbl_geozones AS g
            ON t.latestGeozoneId = g.geozoneId
        WHERE m.status = 1
        ORDER BY m.trolleyId, m.createdOn ASC
    """
    df = run_query(sql)
    if df.empty:
        return df
    df = parse_dates(df, ["createdOn", "modifiedOn"])
    return df


def _apply_consecutive_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core logic: compute job_duration_hours from consecutive createdOn
    entries per trolleyId, not from modifiedOn.
    """
    df = df.sort_values(["trolleyId", "createdOn"]).reset_index(drop=True)

    # Shift createdOn within each trolley group to get the NEXT record's time
    df["next_record_time"] = (
        df.groupby("trolleyId")["createdOn"].shift(-1)
    )

    df["job_duration_hours"] = (
        (df["next_record_time"] - df["createdOn"])
        .dt.total_seconds() / 3600
    )

    # The last record per trolley = open (no next entry)
    df["is_open"] = df["next_record_time"].isna()

    # SLA breach flag (only for closed records)
    df["sla_hours"] = df["type"].map(SLA_HOURS).fillna(24)
    df["sla_breached"] = (
        ~df["is_open"] &
        (df["job_duration_hours"] > df["sla_hours"])
    )

    return df


def get_maintenance_data() -> pd.DataFrame:
    """
    Returns processed maintenance DataFrame with columns:
        maintenanceId, trolleyId, type, reason, reason_category,
        doneBy, createdOn, is_open, job_duration_hours,
        sla_breached, geozoneName, tag
    """
    df = _fetch_maintenance()
    if df.empty:
        return df

    df = _apply_consecutive_resolution(df)
    df["reason_category"] = df["reason"].apply(normalize_maintenance_reason)

    return df


def get_maintenance_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    total       = len(df)
    open_jobs   = df["is_open"].sum()
    closed_jobs = total - open_jobs
    major_open  = df[df["is_open"] & (df["type"] == "major")].shape[0]
    minor_open  = df[df["is_open"] & (df["type"] == "minor")].shape[0]
    sla_breached= df["sla_breached"].sum()
    avg_hours   = df.dropna(subset=["job_duration_hours"])["job_duration_hours"].mean()

    # Repeat maintenance: trolleys with >1 records
    repeat_trolleys = (df.groupby("trolleyId").size() >= 3).sum()

    return {
        "total_jobs":        total,
        "open_jobs":         int(open_jobs),
        "closed_jobs":       int(closed_jobs),
        "major_open":        int(major_open),
        "minor_open":        int(minor_open),
        "sla_breached":      int(sla_breached),
        "avg_resolution_hrs": round(avg_hours, 1) if pd.notna(avg_hours) else 0,
        "repeat_trolleys":   int(repeat_trolleys),
    }


def get_reason_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Maintenance count by reason category."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["reason_category", "type"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def get_zone_maintenance_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Maintenance count per geozone."""
    if df.empty or "geozoneName" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("geozoneName")
        .agg(
            total_jobs    = ("maintenanceId", "count"),
            open_jobs     = ("is_open", "sum"),
            avg_hours     = ("job_duration_hours", "mean"),
            sla_breached  = ("sla_breached", "sum"),
        )
        .reset_index()
        .sort_values("total_jobs", ascending=False)
    )


def get_resolution_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Daily count of jobs created over time — for trend line."""
    if df.empty:
        return pd.DataFrame()
    df["date"] = df["createdOn"].dt.date
    return (
        df.groupby(["date", "type"])
        .size()
        .reset_index(name="count")
        .sort_values("date")
    )