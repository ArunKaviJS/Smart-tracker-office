import pandas as pd

def underperforming_geozones(
    movement_kpi,
    cleaning_kpi
):

    merged = pd.merge(
        movement_kpi,
        cleaning_kpi,
        on="geozoneId",
        how="outer"
    ).fillna(0)

    movement_avg = merged[
        "movement_count"
    ].mean()

    cleaning_avg = merged[
        "cleaning_count"
    ].mean()

    merged["movement_score"] = (
        merged["movement_count"] /
        movement_avg
    )

    merged["cleaning_score"] = (
        merged["cleaning_count"] /
        cleaning_avg
    )

    merged["performance_score"] = (
        merged["movement_score"] * 0.6 +
        merged["cleaning_score"] * 0.4
    )

    result = merged[
        merged["performance_score"] < 1
    ]

    return result.sort_values(
        by="performance_score"
    )