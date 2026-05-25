"""
Idle Trolley KPI
----------------
Logic: A trolley is "idle" when its most recent momentDetection is NOT
"Motion detected!" for longer than a configurable threshold.

Resolution approach:
  - Deduplicate device_data on (deviceEndpoint, timestamp)
  - For each deviceEndpoint, find last_seen and last_motion_time
  - Compute idle_minutes = now − last_motion_time
  - Join to device_history for last known location → geozone
"""

import pandas as pd
from db.connection import run_query
from utils.preprocess import (
    parse_dates, deduplicate, resolve_trolley_id,
    idle_duration_minutes, classify_idle, has_active_fault,
)


IDLE_THRESHOLDS_MIN = {
    "Active":      30,
    "Soft Idle":   60,
    "Idle":       120,
    "Critical Idle": 9999,
}

MOTION_KEYWORDS = ["motion detected", "moving"]


def _fetch_device_data() -> pd.DataFrame:
    sql = """
        SELECT
            deviceDataId,
            trolleyId,
            deviceEndpoint,
            momentDetection,
            timestamp,
            batteryLevel,
            batteryFault,
            status
        FROM tbl_device_data
        WHERE status = 1
    """
    df = run_query(sql)
    if df.empty:
        return df
    df = parse_dates(df, ["timestamp"])
    df = deduplicate(df, ["deviceEndpoint", "timestamp"])
    df = resolve_trolley_id(df)
    return df


def _fetch_last_location() -> pd.DataFrame:
    """Get most recent geozone per device from device_history."""
    sql = """
        SELECT
            h.deviceEndpoint,
            h.trolleyId,
            h.geozoneId,
            h.latitude,
            h.longitude,
            h.locationSource,
            h.createdOn AS location_time
        FROM tbl_device_history h
        INNER JOIN (
            SELECT deviceEndpoint, MAX(createdOn) AS max_created
            FROM tbl_device_history
            WHERE status = 1
            GROUP BY deviceEndpoint
        ) latest
          ON h.deviceEndpoint = latest.deviceEndpoint
         AND h.createdOn      = latest.max_created
    """
    df = run_query(sql)
    if df.empty:
        return df
    df = parse_dates(df, ["location_time"])
    print('*****location time*****')
    print(df)
    print("**********")
    return df


def _fetch_geozones() -> pd.DataFrame:
    sql = """
        SELECT geozoneId, geozoneName, tag, noOfTrolleys
        FROM tbl_geozones
        WHERE status = 1
    """
    return run_query(sql)


def get_idle_trolley_data() -> pd.DataFrame:
    """
    Returns a processed DataFrame, one row per deviceEndpoint, with:
        deviceEndpoint, trolleyId, last_seen, last_motion_time,
        idle_minutes, idle_status, batteryLevel, has_fault,
        geozoneId, geozoneName, tag, latitude, longitude
    """
    raw        = _fetch_device_data()
    locations  = _fetch_last_location()
    geozones   = _fetch_geozones()

    if raw.empty:
        return pd.DataFrame()

    # ── Derive last_seen and last_motion per device ─────────────────────────
    raw["is_motion"] = raw["momentDetection"].str.lower().str.contains(
        "|".join(MOTION_KEYWORDS), na=False
    )

    last_seen = (
        raw.groupby("deviceEndpoint")
        .agg(
            last_seen       = ("timestamp", "max"),
            last_motion_time= ("timestamp", lambda s: s[raw.loc[s.index, "is_motion"]].max()
                               if raw.loc[s.index, "is_motion"].any() else pd.NaT),
            batteryLevel    = ("batteryLevel", "last"),
            batteryFault    = ("batteryFault", "last"),
            trolleyId       = ("trolleyId", "last"),
        )
        .reset_index()
    )

    # ── Idle duration ────────────────────────────────────────────────────────
    # Use last_motion_time; fall back to last_seen if no motion event found
    last_seen["ref_time"] = last_seen["last_motion_time"].fillna(last_seen["last_seen"])
    last_seen["idle_minutes"] = last_seen["ref_time"].apply(idle_duration_minutes)
    last_seen["idle_status"]  = last_seen["idle_minutes"].apply(classify_idle)
    last_seen["has_fault"]    = last_seen["batteryFault"].apply(has_active_fault)

    # ── Join location ─────────────────────────────────────────────────────────
    if not locations.empty:
        last_seen = last_seen.merge(
            locations[["deviceEndpoint", "geozoneId", "latitude", "longitude", "locationSource"]],
            on="deviceEndpoint", how="left"
        )
    else:
        last_seen["geozoneId"] = None
        last_seen["latitude"]  = None
        last_seen["longitude"] = None
        last_seen["locationSource"] = None

    # ── Join geozone names ────────────────────────────────────────────────────
    if not geozones.empty:
        last_seen = last_seen.merge(
            geozones[["geozoneId", "geozoneName", "tag", "noOfTrolleys"]],
            on="geozoneId", how="left"
        )

    return last_seen


def get_idle_summary(df: pd.DataFrame) -> dict:
    """High-level KPI numbers for metric cards."""
    if df.empty:
        return {}
    total         = len(df)
    critical      = (df["idle_status"] == "Critical Idle").sum()
    idle          = (df["idle_status"] == "Idle").sum()
    soft_idle     = (df["idle_status"] == "Soft Idle").sum()
    active        = (df["idle_status"] == "Active").sum()
    avg_idle_min  = df[df["idle_status"] != "Active"]["idle_minutes"].mean()
    low_batt      = (df["batteryLevel"] < 20).sum() if "batteryLevel" in df.columns else 0

    return {
        "total_trolleys":   total,
        "critical_idle":    int(critical),
        "idle":             int(idle),
        "soft_idle":        int(soft_idle),
        "active":           int(active),
        "avg_idle_minutes": round(avg_idle_min, 1) if pd.notna(avg_idle_min) else 0,
        "low_battery":      int(low_batt),
    }


def get_zone_idle_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Idle counts grouped by geozone — for bar chart."""
    if df.empty or "geozoneName" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["geozoneName", "idle_status"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )