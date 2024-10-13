import pytest
import time
from galacticnexus_core import api

def test_api_response_time():
    # Test API response time
    start_time = time.time()
    response = api.get_data()
    end_time = time.time()
    assert end_time - start_time < 5  # 5 seconds
