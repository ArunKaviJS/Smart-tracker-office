import mysql.connector
import csv 
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()

HOST=os.getenv("DB_HOST")
USER=os.getenv("DB_USER")
PASSWORD=os.getenv("DB_PASSWORD")
DATABASE=os.getenv("DB_DATABASE")

###=======================MYSQL==CONNECTION
# db=mysql.connector.connect(
#     host=HOST,
#     user=USER,
#     password=PASSWORD,
#     database=DATABASE, 
# )

# print(db)

# # Load into DataFrame (auto structured)
# df = pd.read_sql("SELECT * FROM tbl_trolley_activity_history", db)

# # Save as clean CSV
# df.to_csv("tbl_trolley_activity_history.csv", index=False)

# print("CSV exported with proper structure!")
# cursor = db.cursor()


engine = create_engine(
    f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:3306/{DATABASE}"
)

print("Connected using SQLAlchemy!")


        
df = pd.read_sql("SELECT * FROM tbl_wifi_fingerprint", engine)
print("tbl_wifi_fingerprint")
print(df.head())

df.to_csv("tbl_wifi_fingerprint.csv", index=False)
