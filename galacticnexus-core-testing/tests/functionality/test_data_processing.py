import pytest
import pandas as pd
from data_processing import validate_data, transform_data, aggregate_data

def test_validate_data_success():
    # Test successful data validation
    data = pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]})
    assert validate_data(data) == True

def test_validate_data_failure():
    # Test failed data validation
    data = pd.DataFrame({"name": ["John", "Mary", "David"], "age": ["25", "31", "42"]})
    assert validate_data(data) == False

def test_transform_data_success():
    # Test successful data transformation
    data = pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]})
    transformed_data = transform_data(data)
    assert transformed_data.equals(pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]}))

def test_transform_data_failure():
    # Test failed data transformation
    data = pd.DataFrame({"name": ["John", "Mary", "David"], "age": ["25", "31", "42"]})
    transformed_data = transform_data(data)
    assert not transformed_data.equals(pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]}))

def test_aggregate_data_success():
    # Test successful data aggregation
    data = pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]})
    aggregated_data = aggregate_data(data)
    assert aggregated_data.equals(pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]}))

def test_aggregate_data_failure():
    # Test failed data aggregation
    data = pd.DataFrame({"name": ["John", "Mary", "David"], "age": ["25", "31", "42"]})
    aggregated_data = aggregate_data(data)
    assert not aggregated_data.equals(pd.DataFrame({"name": ["John", "Mary", "David"], "age": [25, 31, 42]}))
