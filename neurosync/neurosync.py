import os
import numpy as np
from qiskit import QuantumCircuit, execute
from neurosync.quantum_key_generator import generate_quantum_key
from neurosync.quantum_cryptography import encrypt_data, decrypt_data
from neurosync.models.neurosync_model import NeuroSyncModel
from neurosync.utils.neurosync_utils import load_config

class NeuroSync:
    def __init__(self, config_file='neurosync_config.py'):
        self.config = load_config(config_file)
        self.model = NeuroSyncModel()

    def generate_quantum_key(self, length):
        return generate_quantum_key(length)

    def encrypt_data(self, data, key):
        return encrypt_data(data, key)

    def decrypt_data(self, encrypted_data, key):
        return decrypt_data(encrypted_data, key)

    def process_transaction(self, data):
        # Generate quantum key
        key = self.generate_quantum_key(self.config.quantum_key_length)

        # Encrypt data
        encrypted_data = self.encrypt_data(data, key)

        # Store encrypted data and key
        self.model.store_encrypted_data(encrypted_data, key)

        return encrypted_data, key

    def verify_transaction(self, encrypted_data, key):
        # Decrypt data
        decrypted_data = self.decrypt_data(encrypted_data, key)

        # Verify transaction
        if self.model.verify_transaction(decrypted_data):
            return True
        else:
            return False

    def get_quantum_key_length(self):
        return self.config.quantum_key_length

    def get_machine_learning_model_path(self):
        return self.config.machine_learning_model_path
