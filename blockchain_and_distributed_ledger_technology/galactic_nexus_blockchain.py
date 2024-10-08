import hashlib
import time
import json
from galactic_nexus_core import GalacticNexusCore

class GalacticNexusBlockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            'index': 1,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': '0',
            'hash': self.calculate_hash('0', [])
        }
        self.chain.append(genesis_block)

    def calculate_hash(self, previous_hash, transactions):
        data = str(previous_hash) + str(transactions)
        return hashlib.sha256(data.encode()).hexdigest()

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if not self.pending_transactions:
            return False

        new_block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.pending_transactions,
            'previous_hash': self.chain[-1]['hash'],
            'hash': self.calculate_hash(self.chain[-1]['hash'], self.pending_transactions)
        }
        self.chain.append(new_block)
        self.pending_transactions = []
        return new_block

    def get_chain(self):
        return self.chain

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block['hash'] != self.calculate_hash(previous_block['hash'], current_block['transactions']):
                return False
        return True

class GalacticNexusSmartContract:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        self.contract_code = ''

    def deploy_contract(self, contract_code):
        self.contract_code = contract_code
        self.blockchain.add_transaction({'type': 'contract', 'code': contract_code})
        self.blockchain.mine_block()

    def execute_contract(self, function_name, args):
        if function_name in self.contract_code:
            return self.contract_code[function_name](*args)
        else:
            return 'Function not found'

class GalacticNexusDistributedLedger:
    def __init__(self, blockchain):
        self.blockchain = blockchain
        self.ledger = {}

    def add_to_ledger(self, key, value):
        self.ledger[key] = value
        self.blockchain.add_transaction({'type': 'ledger', 'key': key, 'value': value})
        self.blockchain.mine_block()

    def get_from_ledger(self, key):
        return self.ledger.get(key)

# Example usage:
blockchain = GalacticNexusBlockchain()
smart_contract = GalacticNexusSmartContract(blockchain)
distributed_ledger = GalacticNexusDistributedLedger(blockchain)

# Deploy a smart contract
contract_code = {
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y
}
smart_contract.deploy_contract(contract_code)

# Execute a smart contract function
result = smart_contract.execute_contract('add', [2, 3])
print(result)  # Output: 5

# Add to the distributed ledger
distributed_ledger.add_to_ledger('key', 'value')
print(distributed_ledger.get_from_ledger('key'))  # Output: value
