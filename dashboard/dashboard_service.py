def build_dashboard(
    movement_kpi,
    cleaning_kpi,
    maintenance_kpi,
    compliance
):

    dashboard = {

        "total_movements":
            int(
                movement_kpi[
                    "movement_count"
                ].sum()
            ),

        "total_cleaning":
            int(
                cleaning_kpi[
                    "cleaning_count"
                ].sum()
            ),

        "total_maintenance":
            int(
                maintenance_kpi[
                    "maintenance_count"
                ].sum()
            ),

        "cleaning_compliance":
            compliance
    }

    return dashboard