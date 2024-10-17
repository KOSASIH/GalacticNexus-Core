# project/algorand/main.py

"""
Galactic Nexus Algorand Main

Demonstrates the usage of the Algorand interaction module.
"""

import logging
from project.algorand.core.algorand import (
    get_algorand_indexer,
    get_account_balance,
    send_algorand_transaction,
    get_transaction_details
)
from project.algorand.core.utils import generate_algorand_wallet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Generate a new Algorand wallet
    wallet_mnemonic, wallet_address = generate_algorand_wallet()
    logger.info(f"Wallet generated with address: {wallet_address}")

    # Set up Algorand indexer client
    algod_url = "https://api.algodex.com"
    indexer_url = "https://algoindexer.algodex.com"
    indexer_client = get_algorand_indexer(algod_url, indexer_url)

    # Get account balance
    balance = get_account_balance(wallet_address, indexer_client)
    logger.info(f"Account balance: {balance}")

    # Send a transaction
    recipient_address = "X57BQV7J5AQBQV7J5AQBQV7J5AQBQV7J5AQBQV7J"
    amount = 10000
    sender_private_key = mnemonic.to_private_key(wallet_mnemonic)
    txn_id = send_algorand_transaction(
        wallet_address,
        recipient_address,
        amount,
        sender_private_key,
        indexer_client
    )
    logger.info(f"Transaction sent with ID: {txn_id}")

    # Get transaction details
    txn_info = get_transaction_details(txn_id, indexer_client)
    logger.info(f"Transaction details: {txn_info}")

if __name__ == "__main__":
    main()
