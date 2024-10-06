# Import the necessary libraries
import jwt
import bcrypt

# Define the gnx_security functions
def generate_token(username):
  token = jwt.encode({'username': username}, 'secret_key', algorithm='HS256')
  return token

def verify_token(token):
  try:
    decoded = jwt.decode(token, 'secret_key', algorithms=['HS256'])
    return decoded
  except jwt.ExpiredSignatureError:
    return None
  except jwt.InvalidTokenError:
    return None

def hash_password(password):
  hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
  return hashed_password

def compare_passwords(password, hashed_password):
  isValid = bcrypt.checkpw(password.encode('utf-8'), hashed_password)
  return isValid

# Export the gnx_security functions
def gnx_security():
  return {'generate_token': generate_token, 'verify_token': verify_token, 'hash_password': hash_password, 'compare_passwords': compare_passwords}
