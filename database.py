import pyodbc
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

SERVER = os.getenv("SERVER")
DYNAMIC_USERNAME = os.getenv("DYNAMIC_USERNAME")
DYNAMIC_PASSWORD = os.getenv("DYNAMIC_PASSWORD")
MASTER_TABLE_DB_NAME = os.getenv("MASTER_TABLE_DB_NAME")

def create_connection(database_name: str):
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={database_name};"
        f"UID={DYNAMIC_USERNAME};"
        f"PWD={DYNAMIC_PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=Yes;"
    )
    return pyodbc.connect(conn_str)


async def get_db_connection_dynamic(database_name: str):
    return await asyncio.to_thread(create_connection, database_name)
