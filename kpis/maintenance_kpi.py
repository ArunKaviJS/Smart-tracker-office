def maintenance_frequency(maintenance_df):

    result = (
        maintenance_df
        .groupby("trolleyId")
        .size()
        .reset_index(name="maintenance_count")
        .sort_values(
            by="maintenance_count",
            ascending=False
        )
    )

    return result