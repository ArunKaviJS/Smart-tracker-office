import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_connection():

    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT", 3306)

    DATABASE_NAME = "automated_kpi_insights"

    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DATABASE_NAME}"
    )

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

    return engine