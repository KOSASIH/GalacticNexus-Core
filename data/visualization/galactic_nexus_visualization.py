# galactic_nexus_visualization.py

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from cryptography.fernet import Fernet

# Load configuration from environment variables
GALACTIC_NEXUS_DATA_DIR = os.environ['GALACTIC_NEXUS_DATA_DIR']
GALACTIC_NEXUS_DATA_KEY = os.environ['GALACTIC_NEXUS_DATA_KEY']
GALACTIC_NEXUS_VISUALIZATION_DIR = os.environ['GALACTIC_NEXUS_VISUALIZATION_DIR']

# Set up Fernet encryption for secure data storage
fernet_key = Fernet(GALACTIC_NEXUS_DATA_KEY)
fernet = Fernet(fernet_key)

# Load Galactic Nexus data from files
data_dir = GALACTIC_NEXUS_DATA_DIR
data_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

galactic_nexus_data = []
for file in data_files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, 'r') as file:
        encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data.encode()).decode()
    json_data = json.loads(decrypted_data)
    galactic_nexus_data.extend([{
        'data_id': data['data_id'],
        'data_type': data['data_type'],
        'data_value': data['data_value'],
        'timestamp': datetime.fromisoformat(data['timestamp'])
    } for data in json_data])

# Convert data to Pandas DataFrame
df = pd.DataFrame(galactic_nexus_data)

# Create a line chart of asset values over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='timestamp', y='data_value', hue='data_type', data=df)
plt.title('Galactic Nexus Asset Values Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Asset Value')
plt.savefig(os.path.join(GALACTIC_NEXUS_VISUALIZATION_DIR, 'asset_values_over_time.png'))

# Create a bar chart of transaction volumes by asset type
plt.figure(figsize=(12, 6))
sns.countplot(x='data_type', data=df)
plt.title('Galactic Nexus Transaction Volumes by Asset Type')
plt.xlabel('Asset Type')
plt.ylabel('Transaction Volume')
plt.savefig(os.path.join(GALACTIC_NEXUS_VISUALIZATION_DIR, 'transaction_volumes_by_asset_type.png'))

# Create a heatmap of asset correlations
corr_matrix = df.pivot_table(index='data_id', columns='data_type', values='data_value').corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Galactic Nexus Asset Correlations')
plt.xlabel('Asset ID')
plt.ylabel('Asset ID')
plt.savefig(os.path.join(GALACTIC_NEXUS_VISUALIZATION_DIR, 'asset_correlations.png'))

print('Visualizations saved to', GALACTIC_NEXUS_VISUALIZATION_DIR)
