import os
import json
from cryptography.fernet import Fernet

class SecureDataStorage:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)

    def store_data(self, data):
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        with open("secure_data.json", "wb") as f:
            f.write(encrypted_data)

    def retrieve_data(self):
        with open("secure_data.json", "rb") as f:
            encrypted_data = f.read()
        decrypted_data = self.fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode())
