import hashlib
import time
import json

def calculate_hash(previous_hash, transactions):
    data = str(previous_hash) + str(transactions)
    return hashlib.sha256(data.encode()).hexdigest()

def is_chain_valid(chain):
    for i in range(1, len(chain)):
        current_block = chain[i]
        previous_block = chain[i - 1]
        if current_block['hash'] != calculate_hash(previous_block['hash'], current_block['transactions']):
            return False
    return True

def deploy_contract(blockchain, contract_code):
    blockchain.add_transaction({'type': 'contract', 'code': contract_code})
    blockchain.mine_block()

def execute_contract(blockchain, contract_code, function_name, args):
    if function_name in contract_code:
        return contract_code[function_name](*args)
    else:
        return 'Function not found'

def add_to_ledger(blockchain, key, value):
    blockchain.add_transaction({'type': 'ledger', 'key': key, 'value': value})
    blockchain.mine_block()

def get_from_ledger(blockchain, key):
    return blockchain.ledger.get(key)
