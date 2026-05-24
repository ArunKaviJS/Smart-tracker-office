def cleaning_count_by_geozone(cleaning_df):

    result = (
        cleaning_df
        .groupby("geozoneId")
        .size()
        .reset_index(name="cleaning_count")
        .sort_values(
            by="cleaning_count",
            ascending=False
        )
    )

    return result


def cleaning_compliance(
    cleaning_df,
    trolley_df
):

    cleaned = cleaning_df["trolleyId"].nunique()

    total = trolley_df["trolleyId"].nunique()

    compliance = (
        cleaned / total
    ) * 100

    return round(compliance, 2)