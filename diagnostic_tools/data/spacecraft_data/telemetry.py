import pandas as pd
import numpy as np
from datetime import datetime

class TelemetryData:
    def __init__(self, spacecraft_id, timestamp, data):
        self.spacecraft_id = spacecraft_id
        self.timestamp = timestamp
        self.data = data

    def to_dataframe(self):
        # Convert telemetry data to pandas DataFrame
        df = pd.DataFrame({
            "spacecraft_id": [self.spacecraft_id],
            "timestamp": [self.timestamp],
            "data": [self.data]
        })
        return df

    def save_to_csv(self, filename):
        # Save telemetry data to CSV file
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

# Example usage:
spacecraft_id = "SC-001"
timestamp = datetime.now()
data = {
    "temperature": 20,
    "humidity": 50,
    "pressure": 1013
}
telemetry_data = TelemetryData(spacecraft_id, timestamp, data)
telemetry_data.save_to_csv("telemetry_data.csv")
