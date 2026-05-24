def movement_count_by_geozone(movement_df):

    result = (
        movement_df
        .groupby("geozoneId")
        .size()
        .reset_index(name="movement_count")
        .sort_values(
            by="movement_count",
            ascending=False
        )
    )

    return result