import pandas as pd

def preprocess_data(data):

    data["movement"]["createdOn"] = pd.to_datetime(
        data["movement"]["createdOn"],
        errors="coerce"
    )

    data["cleaning"]["createdOn"] = pd.to_datetime(
        data["cleaning"]["createdOn"],
        errors="coerce"
    )

    data["maintenance"]["createdOn"] = pd.to_datetime(
        data["maintenance"]["createdOn"],
        errors="coerce"
    )

    return data