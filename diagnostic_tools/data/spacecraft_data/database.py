import sqlite3
import pandas as pd

class SpacecraftDatabase:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name):
        # Create table in database
        self.cursor.execute(f"""
            CREATE TABLE {table_name} (
                spacecraft_id TEXT,
                timestamp TEXT,
                data TEXT
            )
        """)
        self.conn.commit()

    def insert_data(self, table_name, data):
        # Insert data into table
        self.cursor.execute(f"""
            INSERT INTO {table_name} (spacecraft_id, timestamp, data)
            VALUES (?, ?, ?)
        """, data)
        self.conn.commit()

    def retrieve_data(self, table_name):
        # Retrieve data from table
        self.cursor.execute(f"""
            SELECT * FROM {table_name}
        """)
        data = self.cursor.fetchall()
        return data

    def close_connection(self):
        # Close database connection
        self.conn.close()

# Example usage:
db_name = "spacecraft_database.db"
spacecraft_database = SpacecraftDatabase(db_name)
spacecraft_database.create_table("telemetry_data")
data = ("SC-001", "2022-01-01 12:00:00", "temperature=20,humidity=50,pressure=1013")
spacecraft_database.insert_data("telemetry_data", data)
data = spacecraft_database.retrieve_data("telemetry_data")
print(data)
spacecraft_database.close_connection()
