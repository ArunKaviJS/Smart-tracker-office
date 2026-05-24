from datetime import datetime

def idle_trolley_detection(movement_df):

    latest_seen = (
        movement_df
        .groupby("trolleyId")["createdOn"]
        .max()
        .reset_index()
    )

    latest_seen["idle_hours"] = (
        datetime.now() -
        latest_seen["createdOn"]
    ).dt.total_seconds() / 3600

    return latest_seen.sort_values(
        by="idle_hours",
        ascending=False
    )