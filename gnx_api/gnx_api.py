# Import the necessary libraries
import flask
from flask import request, jsonify
from flask_cors import CORS
from flask_helmet import Helmet
from flask_mongoengine import MongoEngine
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create the Flask app
app = flask.Flask(__name__)
CORS(app)
Helmet(app)

# Connect to MongoDB
app.config['MONGODB_URI'] = os.environ.get('MONGODB_URI')
db = MongoEngine(app)

# Define the gnx_api parameters
gnxApiParams = {
  # Define the gnx_api parameters here
}

# Define the gnx_api function
def gnxApi(data):
  # Implement the gnx_api function here
  return data

# Export the gnx_api function
app.config['gnx_api'] = gnxApi

# Define the API endpoints
@app.route('/gnx_api', methods=['GET'])
def get_gnx_api():
  return 'GNX API'

@app.route('/gnx_api', methods=['POST'])
def post_gnx_api():
  data = request.get_json()
  result = gnxApi(data)
  return jsonify(result)

@app.route('/register', methods=['POST'])
def register():
  data = request.get_json()
  user = User(username=data['username'], password=data['password'])
  user.save()
  return jsonify({'message': 'User created successfully'})

@app.route('/login', methods=['POST'])
def login():
  data = request.get_json()
  user = User.objects(username=data['username']).first()
  if user and user.check_password(data['password']):
    token = jwt.encode({'user_id': user.id}, os.environ.get('SECRET_KEY'), algorithm='HS256')
    return jsonify({'token': token})
  return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/protected', methods=['GET'])
@jwt_required
def protected():
  return 'Protected route'

# Run the app
if __name__ == '__main__':
  app.run(debug=True)
