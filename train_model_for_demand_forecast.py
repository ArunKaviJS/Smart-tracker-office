# ============================================================
# TROLLEY DEMAND FORECASTING
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("tbl_device_history.csv")

print("Raw Shape :", df.shape)

# ============================================================
# DATETIME PROCESSING
# ============================================================

df["createdOn"] = pd.to_datetime(
    df["createdOn"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

df = df.dropna(subset=["createdOn"])

# ============================================================
# SORT DATA
# ============================================================

df = (
    df.sort_values(
        ["trolleyId", "createdOn"]
    )
    .reset_index(drop=True)
)

# ============================================================
# DETECT STAYS
# ============================================================

df["zone_changed"] = (
    (df["trolleyId"] != df["trolleyId"].shift())
    |
    (df["geozoneId"] != df["geozoneId"].shift())
)

df["stay_id"] = df["zone_changed"].cumsum()

# ============================================================
# BUILD STAY TABLE
# ============================================================

stays = (
    df.groupby("stay_id")
    .agg(
        trolleyId=("trolleyId", "first"),
        geozoneId=("geozoneId", "first"),
        geolayerId=("geolayerId", "first"),
        enter_time=("createdOn", "min"),
        exit_time=("createdOn", "max"),
        ping_count=("historyId", "count")
    )
    .reset_index(drop=True)
)

stays["duration_mins"] = (
    (
        stays["exit_time"]
        -
        stays["enter_time"]
    ).dt.total_seconds()
    / 60
)

stays["is_long_stay"] = (
    stays["duration_mins"] >= 120
).astype(int)

stays = stays[
    stays["geozoneId"] != 0
].copy()

print("Stay Records :", len(stays))

# ============================================================
# CREATE HOURLY SLOTS
# ============================================================

min_slot = stays["enter_time"].min().floor("h")
max_slot = stays["exit_time"].max().ceil("h")

all_slots = pd.date_range(
    start=min_slot,
    end=max_slot,
    freq="1h"
)

records = []

for _, row in stays.iterrows():

    for slot_start in all_slots:

        slot_end = slot_start + pd.Timedelta(hours=1)

        if (
            row["enter_time"] < slot_end
            and
            row["exit_time"] > slot_start
        ):

            records.append({
                "trolleyId": row["trolleyId"],
                "geozoneId": row["geozoneId"],
                "geolayerId": row["geolayerId"],
                "slot_start": slot_start,
                "date": slot_start.date(),
                "hour": slot_start.hour,
                "day_num": slot_start.dayofweek,
                "is_weekend": int(slot_start.dayofweek >= 5)
            })

expanded = pd.DataFrame(records)

print("Expanded Shape :", expanded.shape)

# ============================================================
# TROLLEY COUNT TABLE
# ============================================================

trolley_count = (
    expanded.groupby([
        "date",
        "day_num",
        "is_weekend",
        "hour",
        "slot_start",
        "geozoneId",
        "geolayerId"
    ])["trolleyId"]
    .nunique()
    .reset_index()
    .rename(
        columns={
            "trolleyId": "trolley_count"
        }
    )
)

trolley_count["date"] = pd.to_datetime(
    trolley_count["date"]
)

trolley_count["month"] = (
    trolley_count["date"].dt.month
)

print("Training Dataset Shape :", trolley_count.shape)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

trolley_count["hour_sin"] = np.sin(
    2 * np.pi * trolley_count["hour"] / 24
)

trolley_count["hour_cos"] = np.cos(
    2 * np.pi * trolley_count["hour"] / 24
)

trolley_count["month_sin"] = np.sin(
    2 * np.pi * trolley_count["month"] / 12
)

trolley_count["month_cos"] = np.cos(
    2 * np.pi * trolley_count["month"] / 12
)

trolley_count["day_sin"] = np.sin(
    2 * np.pi * trolley_count["day_num"] / 7
)

trolley_count["day_cos"] = np.cos(
    2 * np.pi * trolley_count["day_num"] / 7
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "day_num",
    "is_weekend",
    "hour",
    "geozoneId",
    "geolayerId",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos"
]

TARGET = "trolley_count"

X = trolley_count[FEATURES]
y = trolley_count[TARGET]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# MODELS
# ============================================================

models = {
    "DecisionTree": DecisionTreeRegressor(
        random_state=42
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    ),

    "AdaBoost": AdaBoostRegressor(
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        random_state=42
    )
}

# ============================================================
# TRAIN & EVALUATE
# ============================================================

results = []

best_model = None
best_score = -999

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4)
    })

    if r2 > best_score:
        best_score = r2
        best_model = model

# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "R2",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df)

# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

print("\nBest Model :", best_model_name)

joblib.dump(
    best_model,
    "best_000_trolley_model.pkl"
)

print("Model Saved Successfully")

# ============================================================
# SAVE TRAINING DATASET
# ============================================================

trolley_count.to_csv(
    "trolley_training_dataset.csv",
    index=False
)

print("Training Dataset Saved")