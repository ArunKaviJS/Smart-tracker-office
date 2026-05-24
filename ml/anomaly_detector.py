from sklearn.ensemble import IsolationForest

def detect_movement_anomalies(
    movement_kpi
):

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    movement_kpi["anomaly"] = (
        model.fit_predict(
            movement_kpi[
                ["movement_count"]
            ]
        )
    )

    anomalies = movement_kpi[
        movement_kpi["anomaly"] == -1
    ]

    return anomalies