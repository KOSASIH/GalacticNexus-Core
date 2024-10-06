# Import the necessary libraries
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import bcrypt

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['gnx_db']
collection = db['users']

# Define the gnx_db functions
def create_user(username, password, email):
  hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
  user = {'username': username, 'password': hashed_password, 'email': email}
  collection.insert_one(user)
  return user

def get_user(username):
  user = collection.find_one({'username': username})
  return user

def update_user(username, password, email):
  user = collection.find_one({'username': username})
  if user:
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user['password'] = hashed_password
    user['email'] = email
    collection.update_one({'_id': ObjectId(user['_id'])}, {'$set': user})
  return user

def delete_user(username):
  collection.delete_one({'username': username})

# Export the gnx_db functions
def gnx_db():
  return {'create_user': create_user, 'get_user': get_user, 'update_user': update_user, 'delete_user': delete_user}
