import pandas as pd
import re


DATE_COLS_BY_TABLE = {
    "device_data":        ["timestamp"],
    "device_history":     ["createdOn", "modifiedOn"],
    "battery":            ["warrantyStartDate", "warrantyExpiryDate", "createdOn", "modifiedOn"],
    "trolley_maintenance":["createdOn", "modifiedOn"],
    "trolley":            ["createdOn", "modifiedOn"],
    "geozones":           ["createdOn", "modifiedOn"],
}

DATE_FORMAT = "%d-%m-%Y %H:%M"


def parse_dates(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Parse date string columns to datetime. Tries multiple formats."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce")
            # fallback – let pandas infer if the above fails
            mask = df[col].isna() & df[col].astype(str).ne("NaT")
            if mask.any():
                df.loc[mask, col] = pd.to_datetime(
                    df.loc[mask, col], errors="coerce"
                )
    return df


def deduplicate(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """
    Drop exact duplicate rows based on subset columns.
    Keeps the first occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[PREPROCESS] Removed {before - after} duplicate rows on {subset}")
    return df


def resolve_trolley_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    In tbl_device_data trolleyId = 0 means unassigned.
    Flag these rows so callers can decide how to handle them.
    """
    df["trolley_unassigned"] = df["trolleyId"].astype(str).eq("0") | df["trolleyId"].isna()
    return df


def parse_battery_fault(fault_str: str) -> dict:
    """
    Parse batteryFault field like '2A:0000  2B:0000 2F:0001'
    Returns dict {code: int_value}.
    """
    if pd.isna(fault_str) or not fault_str:
        return {}
    pattern = r"([0-9A-Fa-f]{2}):([0-9A-Fa-f]{4})"
    matches = re.findall(pattern, str(fault_str))
    return {code: int(val, 16) for code, val in matches}


def has_active_fault(fault_str: str) -> bool:
    """Return True if any fault code has a non-zero value."""
    codes = parse_battery_fault(fault_str)
    return any(v != 0 for v in codes.values())


def parse_wifi_distances(wifi_str: str) -> list:
    """
    Parse wifiRTT string like '[SSID:... D:36.92 m]'
    Returns list of distances in meters.
    """
    if pd.isna(wifi_str) or not wifi_str:
        return []
    return [float(d) for d in re.findall(r"D:([\d.]+)\s*m", str(wifi_str))]


def idle_duration_minutes(last_motion_time: pd.Timestamp) -> float:
    """Return minutes since last_motion_time. Returns None if NaT."""
    if pd.isna(last_motion_time):
        return None
    delta = pd.Timestamp.now() - last_motion_time
    return round(delta.total_seconds() / 60, 1)


def classify_idle(minutes) -> str:
    if minutes is None:
        return "Unknown"
    if minutes < 30:
        return "Active"
    if minutes < 60:
        return "Soft Idle"
    if minutes < 120:
        return "Idle"
    return "Critical Idle"


def classify_warranty_status(days_remaining) -> str:
    if days_remaining is None:
        return "Unknown"
    if days_remaining < 0:
        return "Expired"
    if days_remaining <= 30:
        return "Critical"
    if days_remaining <= 90:
        return "Warning"
    return "Healthy"


def normalize_maintenance_reason(reason: str) -> str:
    """Map free-text reasons to standard categories."""
    if pd.isna(reason):
        return "Unknown"
    r = reason.lower()
    if any(k in r for k in ["oil", "greas", "lubric"]):
        return "Lubrication"
    if any(k in r for k in ["weld", "broken", "crack", "structur"]):
        return "Structural Damage"
    if any(k in r for k in ["wheel", "castor", "tyre"]):
        return "Wheel Issue"
    if any(k in r for k in ["batter", "charg", "power"]):
        return "Battery / Power"
    if any(k in r for k in ["clean", "wash", "sanitiz"]):
        return "Cleaning"
    if any(k in r for k in ["lock", "brake", "handle"]):
        return "Mechanical"
    return "Other"