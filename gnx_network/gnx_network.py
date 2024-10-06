# Import the necessary libraries
import requests

# Define the gnx_network functions
def get_data(url):
  response = requests.get(url)
  return response.json()

def post_data(url, data):
  response = requests.post(url, json=data)
  return response.json()

# Export the gnx_network functions
def gnx_network():
  return {'get_data': get_data, 'post_data': post_data}
