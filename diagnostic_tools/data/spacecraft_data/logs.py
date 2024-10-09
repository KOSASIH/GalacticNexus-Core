import logging
import datetime

class LogData:
    def __init__(self, spacecraft_id, timestamp, log_level, message):
        self.spacecraft_id = spacecraft_id
        self.timestamp = timestamp
        self.log_level = log_level
        self.message = message

    def to_string(self):
        # Convert log data to string
        log_string = f"{self.timestamp} - {self.spacecraft_id} - {self.log_level} - {self.message}"
        return log_string

    def save_to_file(self, filename):
        # Save log data to file
        with open(filename, "a") as f:
            f.write(self.to_string() + "\n")

# Example usage:
spacecraft_id = "SC-001"
timestamp = datetime.now()
log_level = "INFO"
message = "Spacecraft is functioning normally"
log_data = LogData(spacecraft_id, timestamp, log_level, message)
log_data.save_to_file("log_data.log")
