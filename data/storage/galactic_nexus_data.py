# galactic_nexus_data.py

import os
import json
from datetime import datetime
from cryptography.fernet import Fernet

# Load configuration from environment variables
GALACTIC_NEXUS_DATA_DIR = os.environ['GALACTIC_NEXUS_DATA_DIR']
GALACTIC_NEXUS_DATA_KEY = os.environ['GALACTIC_NEXUS_DATA_KEY']

# Set up Fernet encryption for secure data storage
fernet_key = Fernet(GALACTIC_NEXUS_DATA_KEY)
fernet = Fernet(fernet_key)

# Define a class to represent a Galactic Nexus data object
class GalacticNexusData:
    def __init__(self, data_id, data_type, data_value, timestamp):
        self.data_id = data_id
        self.data_type = data_type
        self.data_value = data_value
        self.timestamp = timestamp

    def to_json(self):
        return {
            'data_id': self.data_id,
            'data_type': self.data_type,
            'data_value': self.data_value,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            json_data['data_id'],
            json_data['data_type'],
            json_data['data_value'],
            datetime.fromisoformat(json_data['timestamp'])
        )

# Load actual data from files
data_dir = GALACTIC_NEXUS_DATA_DIR
data_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

galactic_nexus_data = []
for file in data_files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, 'r') as file:
        encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data.encode()).decode()
    json_data = json.loads(decrypted_data)
    galactic_nexus_data.extend([GalacticNexusData.from_json(data) for data in json_data])

# Print the loaded data
for data in galactic_nexus_data:
    print(data.to_json())

# Example data:
# [
#     {
#         "data_id": "GNX-001",
#         "data_type": "stellar_asset",
#         "data_value": "XLM",
#         "timestamp": "2023-02-15T14:30:00"
#     },
#     {
#         "data_id": "GNX-002",
#         "data_type": "stellar_transaction",
#         "data_value": "0.01 XLM",
#         "timestamp": "2023-02-15T14:31:00"
#     },
#     {
#         "data_id": "GNX-003",
#         "data_type": "galactic_nexus_event",
#         "data_value": "New asset listed",
#         "timestamp": "2023-02-15T14:32:00"
#     }
# ]
