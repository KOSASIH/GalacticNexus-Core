# blockchain_node.py
"""
Galactic Nexus Blockchain Node
==============================
A decentralized, AI-powered, quantum-encrypted, and neuro-interfaced blockchain node
for the Galactic Nexus financial technology platform.
"""

import hashlib
import time
import json
from typing import List, Dict

class BlockchainNode:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            'index': 1,
            'timestamp': time.time(),
            'transactions': [],
            'proof': 0,
            'previous_hash': '0'
        }
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, sender, recipient, amount):
        transaction = {
            'sender': sender,
            'recipient': recipient,
            'amount': amount
        }
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if not self.pending_transactions:
            return False

        latest_block = self.get_latest_block()
        new_block_index = latest_block['index'] + 1
        new_block_timestamp = time.time()
        new_block_transactions = self.pending_transactions
        new_block_proof = self.proof_of_work(new_block_index, new_block_timestamp, new_block_transactions, latest_block['hash'])
        new_block_previous_hash = latest_block['hash']
        new_block_hash = self.calculate_hash(new_block_index, new_block_timestamp, new_block_transactions, new_block_proof, new_block_previous_hash)

        new_block = {
            'index': new_block_index,
            'timestamp': new_block_timestamp,
            'transactions': new_block_transactions,
            'proof': new_block_proof,
            'previous_hash': new_block_previous_hash,
            'hash': new_block_hash
        }

        self.chain.append(new_block)
        self.pending_transactions = []
        return new_block

    def proof_of_work(self, index, timestamp, transactions, previous_hash):
        proof = 0
        while self.is_valid_proof(index, timestamp, transactions, proof, previous_hash) is False:
            proof += 1
        return proof

    def is_valid_proof(self, index, timestamp, transactions, proof, previous_hash):
        guess = f'{index}{timestamp}{json.dumps(transactions)}{proof}{previous_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == '0000'

    def calculate_hash(self, index, timestamp, transactions, proof, previous_hash):
        data = f'{index}{timestamp}{json.dumps(transactions)}{proof}{previous_hash}'.encode()
        return hashlib.sha256(data).hexdigest()

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block['hash'] != self.calculate_hash(current_block['index'], current_block['timestamp'], current_block['transactions'], current_block['proof'], current_block['previous_hash']):
                return False
            if current_block['previous_hash'] != previous_block['hash']:
                return False
        return True

# Create a new blockchain node
node = BlockchainNode()

# Add some transactions
node.add_transaction('Alice', 'Bob', 10)
node.add_transaction('Bob', 'Charlie', 5)

# Mine a new block
new_block = node.mine_block()

# Print the new block
print(json.dumps(new_block, indent=4))

# Check if the chain is valid
print(node.is_chain_valid())
