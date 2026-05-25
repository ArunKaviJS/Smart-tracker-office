"""
Tracker / Device Warranty KPI
------------------------------
  - Detect trackers offline > 24 hrs (silent) and > 48 hrs (critical).
  - Compute location failure rate (lat=0, lon=0 readings).
  - Identify outdated firmware versions.
  - Parse wifiRTT for poor signal quality.
  - Surface warranty claim candidates.
"""

import re
import pandas as pd
from db.connection import run_query
from utils.preprocess import parse_dates, deduplicate, parse_wifi_distances, has_active_fault


def _fetch_device_summary() -> pd.DataFrame:
    sql = """
        SELECT
            deviceEndpoint,
            firmwareVersion,
            MAX(timestamp)       AS last_seen,
            AVG(batteryLevel)    AS avg_battery,
            MIN(batteryLevel)    AS min_battery,
            COUNT(*)             AS total_readings,
            SUM(CASE WHEN GPSLatitude  = 0
                      AND GPSLongitude = 0 THEN 1 ELSE 0 END) AS failed_location_count,
            MAX(batteryFault)    AS last_fault,
            MAX(status)          AS status
        FROM tbl_device_data
        GROUP BY deviceEndpoint, firmwareVersion
    """
    df = run_query(sql)
    if df.empty:
        return df
    df = parse_dates(df, ["last_seen"])
    return df


def _fetch_history_quality() -> pd.DataFrame:
    """Wi-Fi RTT signal quality per device."""
    sql = """
        SELECT
            deviceEndpoint,
            COUNT(*) AS total_history,
            SUM(CASE WHEN locationSource = 'GPS' THEN 1 ELSE 0 END) AS gps_count,
            SUM(CASE WHEN locationSource = 'Wi-Fi' THEN 1 ELSE 0 END) AS wifi_count
        FROM tbl_device_history
        WHERE status = 1
        GROUP BY deviceEndpoint
    """
    return run_query(sql)


def get_tracker_data() -> pd.DataFrame:
    """
    Returns one row per tracker (deviceEndpoint) with:
        deviceEndpoint, firmwareVersion, last_seen, hours_offline,
        avg_battery, has_fault, location_fail_rate, offline_status
    """
    df       = _fetch_device_summary()
    hist_q   = _fetch_history_quality()

    if df.empty:
        return df

    # ── Offline duration ─────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    df["hours_offline"] = (now - df["last_seen"]).dt.total_seconds() / 3600
    df["hours_offline"] = df["hours_offline"].clip(lower=0)

    df["offline_status"] = df["hours_offline"].apply(
        lambda h: "Critical (>48h)" if h > 48
        else ("Warning (24–48h)" if h > 24
        else "Online")
    )

    # ── Location failure rate ────────────────────────────────────────────────
    df["location_fail_rate"] = (
        df["failed_location_count"] / df["total_readings"]
    ).fillna(0)

    # ── Fault flag ────────────────────────────────────────────────────────────
    df["has_fault"] = df["last_fault"].apply(has_active_fault)

    # ── Warranty claim candidate ──────────────────────────────────────────────
    df["claim_candidate"] = (
        (df["hours_offline"] > 48) |
        (df["location_fail_rate"] > 0.3) |
        df["has_fault"]
    )

    # ── Join history quality ──────────────────────────────────────────────────
    if not hist_q.empty:
        df = df.merge(hist_q, on="deviceEndpoint", how="left")

    return df


def get_tracker_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    total        = len(df)
    critical     = (df["hours_offline"] > 48).sum()
    warning      = ((df["hours_offline"] > 24) & (df["hours_offline"] <= 48)).sum()
    online       = (df["hours_offline"] <= 24).sum()
    claim_ready  = df["claim_candidate"].sum()
    loc_bad      = (df["location_fail_rate"] > 0.3).sum()
    fault_count  = df["has_fault"].sum()

    fw_versions  = df["firmwareVersion"].nunique()

    return {
        "total_trackers":    total,
        "critical_offline":  int(critical),
        "warning_offline":   int(warning),
        "online":            int(online),
        "claim_candidates":  int(claim_ready),
        "location_failures": int(loc_bad),
        "active_faults":     int(fault_count),
        "firmware_versions": int(fw_versions),
    }


def get_firmware_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("firmwareVersion")
        .agg(count=("deviceEndpoint", "count"),
             avg_offline=("hours_offline", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )


def get_offline_tracker_list(df: pd.DataFrame) -> pd.DataFrame:
    """Filtered list of offline trackers for the detail table."""
    if df.empty:
        return pd.DataFrame()
    cols = ["deviceEndpoint", "firmwareVersion", "last_seen",
            "hours_offline", "offline_status", "has_fault",
            "location_fail_rate", "claim_candidate"]
    return (
        df[df["offline_status"] != "Online"][
            [c for c in cols if c in df.columns]
        ]
        .sort_values("hours_offline", ascending=False)
        .reset_index(drop=True)
    )