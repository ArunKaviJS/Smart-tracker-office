import pandas as pd
import numpy as np
from typing import List, Dict


def zscore_flag(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """
    Return a boolean Series: True where the value is an outlier
    based on Z-score. NaN values are treated as non-anomalies.
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean).abs() / std
    return z > threshold


def iqr_flag(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Return a boolean Series: True where the value is an outlier
    using the IQR fence method.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def generate_idle_insights(df: pd.DataFrame) -> List[Dict]:
    """
    Produce list of insight dicts for the idle trolley KPI.
    Each dict: {severity, title, detail}
    severity: 'critical' | 'warning' | 'info' | 'success'
    """
    insights = []
    if df.empty:
        return insights

    critical_count = (df["idle_status"] == "Critical Idle").sum()
    idle_count     = (df["idle_status"] == "Idle").sum()
    soft_idle      = (df["idle_status"] == "Soft Idle").sum()
    total          = len(df)

    if critical_count > 0:
        insights.append({
            "severity": "critical",
            "title": f"{critical_count} trolleys critically idle (>120 min)",
            "detail": (
                f"{critical_count} of {total} trolleys have not moved for over 2 hours. "
                "Immediate retrieval recommended."
            ),
        })

    if "geozoneName" in df.columns:
        zone_idle = (
            df[df["idle_status"].isin(["Idle", "Critical Idle"])]
            .groupby("geozoneName")
            .size()
            .sort_values(ascending=False)
        )
        if not zone_idle.empty:
            top_zone = zone_idle.index[0]
            top_count = zone_idle.iloc[0]
            insights.append({
                "severity": "warning",
                "title": f"Zone '{top_zone}' has highest idle concentration",
                "detail": f"{top_count} idle trolleys concentrated in this zone.",
            })

    if "batteryLevel" in df.columns:
        low_batt = df[(df["idle_status"] != "Active") & (df["batteryLevel"] < 20)]
        if not low_batt.empty:
            insights.append({
                "severity": "critical",
                "title": f"{len(low_batt)} idle trolleys also have low battery (<20%)",
                "detail": "These devices risk going offline before next use.",
            })

    if idle_count == 0 and critical_count == 0:
        insights.append({
            "severity": "success",
            "title": "All trolleys are active or softly idle",
            "detail": f"{total - soft_idle} trolleys are actively in use.",
        })

    # Anomaly: zone-level idle rate spike
    if "geozoneName" in df.columns and len(df["geozoneName"].dropna().unique()) > 1:
        zone_rates = (
            df.groupby("geozoneName")
            .apply(lambda g: (g["idle_status"] != "Active").mean())
            .rename("idle_rate")
        )
        anomaly_zones = zone_rates[zscore_flag(zone_rates, threshold=1.5)]
        for zone, rate in anomaly_zones.items():
            insights.append({
                "severity": "warning",
                "title": f"Anomalous idle rate in '{zone}'",
                "detail": f"{rate * 100:.0f}% of trolleys in this zone are idle — statistically unusual.",
            })

    return insights


def generate_maintenance_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    if df.empty:
        return insights

    open_jobs   = df[df["is_open"]].shape[0]
    major_open  = df[df["is_open"] & (df["type"] == "major")].shape[0]
    total       = len(df)

    if major_open > 0:
        insights.append({
            "severity": "critical",
            "title": f"{major_open} major maintenance jobs currently open",
            "detail": "Major repairs typically require >24 hours. Escalate if overdue.",
        })

    if open_jobs > 0:
        insights.append({
            "severity": "warning",
            "title": f"{open_jobs} total open maintenance jobs",
            "detail": f"{open_jobs} of {total} maintenance records have no resolution yet.",
        })

    if "job_duration_hours" in df.columns:
        closed = df.dropna(subset=["job_duration_hours"])
        if not closed.empty:
            avg_h = closed["job_duration_hours"].mean()
            sla_breach = closed[
                ((closed["type"] == "minor") & (closed["job_duration_hours"] > 24)) |
                ((closed["type"] == "major") & (closed["job_duration_hours"] > 72))
            ]
            if not sla_breach.empty:
                insights.append({
                    "severity": "warning",
                    "title": f"{len(sla_breach)} jobs breached SLA thresholds",
                    "detail": (
                        f"Minor SLA: 24 hrs, Major SLA: 72 hrs. "
                        f"Average resolution so far: {avg_h:.1f} hrs."
                    ),
                })
            else:
                insights.append({
                    "severity": "success",
                    "title": "All resolved jobs met SLA",
                    "detail": f"Average resolution time: {avg_h:.1f} hours.",
                })

    if "trolleyId" in df.columns:
        repeat = df.groupby("trolleyId").size()
        frequent = repeat[repeat >= 3]
        if not frequent.empty:
            insights.append({
                "severity": "critical",
                "title": f"{len(frequent)} trolleys have 3+ maintenance records",
                "detail": "These trolleys may need full overhaul or retirement.",
            })

    return insights


def generate_battery_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    if df.empty:
        return insights

    expired  = (df["warranty_status"] == "Expired").sum()
    critical = (df["warranty_status"] == "Critical").sum()
    warning  = (df["warranty_status"] == "Warning").sum()

    if expired > 0:
        insights.append({
            "severity": "critical",
            "title": f"{expired} batteries have expired warranties",
            "detail": "These batteries are no longer covered. Initiate replacement claims.",
        })
    if critical > 0:
        insights.append({
            "severity": "critical",
            "title": f"{critical} batteries expiring within 30 days",
            "detail": "Submit warranty claims before expiry to avoid coverage gap.",
        })
    if warning > 0:
        insights.append({
            "severity": "warning",
            "title": f"{warning} batteries expiring within 90 days",
            "detail": "Schedule review and pre-emptive claim filing.",
        })

    if "has_fault" in df.columns:
        faulty = df["has_fault"].sum()
        if faulty > 0:
            insights.append({
                "severity": "critical",
                "title": f"{faulty} batteries reporting active fault codes",
                "detail": "Non-zero fault registers detected. Check cell voltage, temp, and charge circuits.",
            })

    if expired == 0 and critical == 0:
        insights.append({
            "severity": "success",
            "title": "No batteries in urgent warranty status",
            "detail": f"{warning} batteries in 90-day warning window for proactive monitoring.",
        })

    return insights


def generate_tracker_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    if df.empty:
        return insights

    offline  = (df["hours_offline"] > 48).sum()
    silent   = (df["hours_offline"] > 24).sum()
    loc_fail = df.get("location_fail_rate", pd.Series(dtype=float))
    loc_bad  = (loc_fail > 0.3).sum() if not loc_fail.empty else 0

    if offline > 0:
        insights.append({
            "severity": "critical",
            "title": f"{offline} trackers offline for >48 hours",
            "detail": "Extended silence may indicate hardware failure. Initiate warranty claims.",
        })
    if silent > offline:
        insights.append({
            "severity": "warning",
            "title": f"{silent - offline} trackers offline for 24–48 hours",
            "detail": "Monitor closely. May be connectivity issues or low battery.",
        })
    if loc_bad > 0:
        insights.append({
            "severity": "warning",
            "title": f"{loc_bad} trackers have >30% failed location readings",
            "detail": "Wi-Fi radio or GPS module may be defective — warranty eligible.",
        })

    if "firmwareVersion" in df.columns:
        versions = df["firmwareVersion"].value_counts()
        if len(versions) > 1:
            oldest = versions.index[-1]
            count  = versions.iloc[-1]
            insights.append({
                "severity": "info",
                "title": f"{count} trackers on outdated firmware ({oldest})",
                "detail": "Update firmware before filing warranty claims to ensure eligibility.",
            })

    if offline == 0 and silent == 0:
        insights.append({
            "severity": "success",
            "title": "All trackers reporting within 24 hours",
            "detail": "No devices in critical offline state.",
        })

    return insights