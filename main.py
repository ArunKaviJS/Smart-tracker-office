from loaders.data_loader import load_all_tables
import pandas as pd
from preprocessing.preprocess import (
    preprocess_data
)

from kpis.movement_kpi import (
    movement_count_by_geozone
)

from kpis.cleaning_kpi import (
    cleaning_count_by_geozone,
    cleaning_compliance
)

from kpis.maintenance_kpi import (
    maintenance_frequency
)

from kpis.idle_kpi import (
    idle_trolley_detection
)

from kpis.geozone_kpi import (
    underperforming_geozones
)

from ml.anomaly_detector import (
    detect_movement_anomalies
)

from insights.insight_generator import (
    generate_insights
)

from dashboard.dashboard_service import (
    build_dashboard
)

from utils.logger import logger

def main():

    logger.info(
        "Loading Tables"
    )

    data = load_all_tables()
    
    print(data)

    logger.info(
        "Preprocessing Data"
    )

    data = preprocess_data(data)

    trolley_df = data["trolley"]
    movement_df = data["movement"]
    cleaning_df = data["cleaning"]
    maintenance_df = data["maintenance"]

    logger.info(
        "Calculating KPIs"
    )

    movement_kpi = (
        movement_count_by_geozone(
            movement_df
        )
    )

    print('movement kpi',movement_kpi)
    cleaning_kpi = (
        cleaning_count_by_geozone(
            cleaning_df
        )
    )
    print('cleaning api',cleaning_kpi)
    maintenance_kpi = (
        maintenance_frequency(
            maintenance_df
        )
    )
    print('maintenance kpi',maintenance_kpi)
    idle_kpi = (
        idle_trolley_detection(
            movement_df
        )
    )

    print('idle kpi',idle_kpi)
    compliance = (
        cleaning_compliance(
            cleaning_df,
            trolley_df
        )
    )

    print('compliance',compliance)
    print('movenment kpi',movement_kpi)
    print('cleaning kpi',cleaning_kpi)
    
    movement_kpi["geozoneId"] = pd.to_numeric(
    movement_kpi["geozoneId"],
    errors="coerce"
    )

    cleaning_kpi["geozoneId"] = pd.to_numeric(
        cleaning_kpi["geozoneId"],
        errors="coerce"
    )
    
    underperforming = (
        underperforming_geozones(
            movement_kpi,
            cleaning_kpi
        )
    )

    anomalies = (
        detect_movement_anomalies(
            movement_kpi
        )
    )

    insights = generate_insights(
        underperforming,
        anomalies,
        idle_kpi
    )

    dashboard = build_dashboard(
        movement_kpi,
        cleaning_kpi,
        maintenance_kpi,
        compliance
    )

    logger.info(
        "Dashboard Summary"
    )

    print(dashboard)

    logger.info(
        "Generated Insights"
    )

    for insight in insights:

        print(insight)

if __name__ == "__main__":

    main()