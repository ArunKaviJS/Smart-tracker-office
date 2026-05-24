def generate_insights(
    underperforming_df,
    anomaly_df,
    idle_df
):

    insights = []

    for _, row in underperforming_df.iterrows():

        insights.append({
            "type": "UNDERPERFORMING",
            "message": (
                f"Geozone "
                f"{row['geozoneId']} "
                f"is underperforming"
            )
        })

    for _, row in anomaly_df.iterrows():

        insights.append({
            "type": "ANOMALY",
            "message": (
                f"Anomaly detected "
                f"in geozone "
                f"{row['geozoneId']}"
            )
        })

    for _, row in idle_df.head(10).iterrows():

        if row["idle_hours"] > 12:

            insights.append({
                "type": "IDLE",
                "message": (
                    f"Trolley "
                    f"{row['trolleyId']} "
                    f"idle for "
                    f"{round(row['idle_hours'], 2)} "
                    f"hours"
                )
            })

    return insights