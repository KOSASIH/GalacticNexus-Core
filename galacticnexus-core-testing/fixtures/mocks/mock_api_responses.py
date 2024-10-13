import json

def mock_get_user_response(user_id):
    """Mock response for getting user details."""
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30
    }

def mock_get_users_response():
    """Mock response for getting a list of users."""
    return [
        {"id": 1, "name": "John Doe", "email": "john.doe@example.com", "age": 30},
        {"id": 2, "name": "Jane Smith", "email": "jane.smith@example.com", "age": 25}
    ]

def mock_create_user_response(user_data):
    """Mock response for creating a new user."""
    return {
        "id": 3,
        "name": user_data["name"],
        "email": user_data["email"],
        "age": user_data["age"]
    }

def mock_error_response(status_code):
    """Mock error response based on status code."""
    error_responses = {
        404: {"error": "User  not found"},
        400: {"error": "Bad request"},
        500: {"error": "Internal server error"}
    }
    return error_responses.get(status_code, {"error": "Unknown error"})

def mock_get_user_response_json(user_id):
    """Return JSON string for getting user details."""
    return json.dumps(mock_get_user_response(user_id))

def mock_get_users_response_json():
    """Return JSON string for getting a list of users."""
    return json.dumps(mock_get_users_response())

def mock_create_user_response_json(user_data):
    """Return JSON string for creating a new user."""
    return json.dumps(mock_create_user_response(user_data))

def mock_error_response_json(status_code):
    """Return JSON string for error response based on status code."""
    return json.dumps(mock_error_response(status_code))
