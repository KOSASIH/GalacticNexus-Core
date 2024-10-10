# models/blockchain.py

import hashlib
from flask import jsonify

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.create_genesis_block()

    def create_genesis_block(self):
        # Create the genesis block
        pass

    def add_block(self, block):
        # Add a new block to the blockchain
        pass

    def add_transaction(self, transaction):
        # Add a new transaction to the pending transactions list
        pass

    def mine_pending_transactions(self):
        # Mine the pending transactions and add them to the blockchain
        pass

    def get_chain(self):
        # Return the entire blockchain
        return jsonify({'chain': self.chain})

    def get_pending_transactions(self):
        # Return the pending transactions list
        return jsonify({'pending_transactions': self.pending_transactions})
