import csv 
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()

HOST=os.getenv("DB_HOST")
USER=os.getenv("DB_USER")
PASSWORD=os.getenv("DB_PASSWORD")
DATABASE='automated_kpi_insights'


###=========sqlAlchemy========
engine = create_engine(
    f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:3306/{DATABASE}"
)

print("Connected using SQLAlchemy!")


        
df = pd.read_sql("Show tables", engine)
print('df')
