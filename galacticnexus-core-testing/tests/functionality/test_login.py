import pytest
from galacticnexus_core import login

def test_login_success():
    # Test successful login
    username = "test_user"
    password = "test_password"
    assert login(username, password) == True

def test_login_failure():
    # Test failed login
    username = "invalid_user"
    password = "invalid_password"
    assert login(username, password) == False
