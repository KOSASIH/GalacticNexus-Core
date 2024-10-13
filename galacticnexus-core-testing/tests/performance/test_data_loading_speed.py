import pytest
import time
import pandas as pd
from data_loading import load_data_from_csv, load_data_from_db, load_data_from_api

def test_load_data_from_csv_speed():
    # Test data loading speed from CSV file
    start_time = time.time()
    data = load_data_from_csv("large_data.csv")
    end_time = time.time()
    assert end_time - start_time < 5  # 5 seconds

def test_load_data_from_db_speed():
    # Test data loading speed from database
    start_time = time.time()
    data = load_data_from_db("database_name", "table_name")
    end_time = time.time()
    assert end_time - start_time < 10  # 10 seconds

def test_load_data_from_api_speed():
    # Test data loading speed from API
    start_time = time.time()
    data = load_data_from_api("api_endpoint")
    end_time = time.time()
    assert end_time - start_time < 15  # 15 seconds

def test_load_large_data_from_csv_speed():
    # Test data loading speed from large CSV file
    start_time = time.time()
    data = load_data_from_csv("very_large_data.csv")
    end_time = time.time()
    assert end_time - start_time < 30  # 30 seconds

def test_load_complex_data_from_db_speed():
    # Test data loading speed from complex database query
    start_time = time.time()
    data = load_data_from_db("database_name", "complex_query")
    end_time = time.time()
    assert end_time - start_time < 45  # 45 seconds
