# ethereum_adapter.py

import web3
from web3.contract import Contract

class EthereumAdapter:
    def __init__(self, node_url, chain_id, contract_address, contract_abi):
        self.web3 = web3.Web3(web3.providers.HttpProvider(node_url))
        self.chain_id = chain_id
        self.contract_address = contract_address
        self.contract_abi = contract_abi
        self.contract = self.web3.eth.contract(address=contract_address, abi=contract_abi)

    def get_block_number(self):
        return self.web3.eth.block_number

    def get_transaction_by_hash(self, tx_hash):
        return self.web3.eth.get_transaction(tx_hash)

    def get_balance(self, address):
        return self.web3.eth.get_balance(address)

    def send_transaction(self, tx):
        return self.web3.eth.send_transaction(tx)

    def call_contract_function(self, function_name, *args):
        return self.contract.functions[function_name](*args).call()

    def send_contract_transaction(self, function_name, *args):
        return self.contract.functions[function_name](*args).transact()
