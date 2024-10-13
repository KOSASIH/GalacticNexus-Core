import pandas as pd
import psycopg2
import requests

def load_data_from_csv(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    return data

def load_data_from_db(database_name, table_name):
    # Load data from database
    conn = psycopg2.connect(
        dbname=database_name,
        user="username",
        password="password",
        host="host",
        port="port"
    )
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table_name}")
    data = cur.fetchall()
    conn.close()
    return data

def load_data_from_api(api_endpoint):
    # Load data from API
    response = requests.get(api_endpoint)
    data = response.json()
    return data
