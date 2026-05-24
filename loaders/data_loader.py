import pandas as pd

from database.connection import get_connection

def load_all_tables():

    conn = get_connection()

    trolley_df = pd.read_sql(
        "SELECT * FROM tbl_trolley",
        conn
    )

    movement_df = pd.read_sql(
        "SELECT * FROM tbl_device_history",
        conn
    )

    cleaning_df = pd.read_sql(
        "SELECT * FROM tbl_cleaning_history",
        conn
    )

    maintenance_df = pd.read_sql(
        "SELECT * FROM tbl_trolley_maintenance",
        conn
    )

    geozone_df = pd.read_sql(
        "SELECT * FROM tbl_geozones",
        conn
    )

    

    return {
        "trolley": trolley_df,
        "movement": movement_df,
        "cleaning": cleaning_df,
        "maintenance": maintenance_df,
        "geozone": geozone_df
    }