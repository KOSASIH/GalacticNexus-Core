import os
import json
from .interoperability import BlockchainInteroperability
from .security import AdvancedEncryption

class GalacticNexusCore:
    def __init__(self):
        self.config = self.load_config()
        self.blockchain_interoperability = BlockchainInteroperability(self.config)
        self.advanced_encryption = AdvancedEncryption(self.config)

    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'blockchain_platforms.json')) as f:
            return json.load(f)

    def send_data_to_blockchain(self, data, blockchain_platform):
        self.blockchain_interoperability.send_data_to_blockchain(data, blockchain_platform)

    def encrypt_data(self, data):
        return self.advanced_encryption.encrypt_data(data)
