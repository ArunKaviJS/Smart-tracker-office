import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load env variables
load_dotenv()

HOST = os.getenv("DB_HOST")
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
DATABASE = os.getenv("DB_DATABASE")

# Create engine
engine = create_engine(
    f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:3306/{DATABASE}"
)

# List of tables (fixed typo: space removed)
tables = [
    "tbl_battery",
    "tbl_battery_activity",
    "tbl_battery_claim_history",
    "tbl_cleaning_history",
    "tbl_device",
    "tbl_device_data",
    "tbl_device_activity",
    "tbl_device_history",
    "tbl_geolayers",
    "tbl_geozone",
    "tbl_nest",
    "tbl_nest_activity_history",  # FIXED
    "tbl_trolley",
    "tbl_trolley_activity",
    "tbl_trolley_maintainance"
]

# Output folder
output_dir = "csv_exports"
os.makedirs(output_dir, exist_ok=True)

try:
    with engine.connect() as conn:
        print("Connected to DB ✅")

        for table in tables:
            try:
                print(f"Fetching: {table}...")

                query = f"SELECT * FROM {table}"
                df = pd.read_sql(query, conn)

                file_path = os.path.join(output_dir, f"{table}.csv")
                df.to_csv(file_path, index=False)

                print(f"Saved: {file_path}")

            except Exception as table_error:
                print(f"Error in {table}: {table_error}")

except Exception as e:
    print("Connection error:", e)

finally:
    engine.dispose()
    print("Done. Connection closed.")