import pytest
from galacticnexus_core.security import validate_input

def test_validate_input_success():
    # Test successful input validation
    input_data = {"username": "test_user", "password": "test_password"}
    assert validate_input(input_data) == True

def test_validate_input_failure():
    # Test failed input validation
    input_data = {"username": "invalid_user", "password": "invalid_password"}
    assert validate_input(input_data) == False

def test_validate_input_empty():
    # Test input validation with empty input
    input_data = {}
    assert validate_input(input_data) == False

def test_validate_input_invalid_type():
    # Test input validation with invalid input type
    input_data = ["test_user", "test_password"]
    assert validate_input(input_data) == False
