# project/algorand/core/algorand.py

"""
Galactic Nexus Algorand Module

This module provides functionality for interacting with the Algorand blockchain,
including account balance retrieval, transaction sending, and transaction details.
"""

import logging
from algosdk.v2client import indexer
from algosdk import mnemonic, transaction
from algosdk.error import AlgodHTTPError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_algorand_indexer(algod_url: str, indexer_url: str) -> indexer.Indexer:
    """
    Returns an Algorand indexer client.

    Args:
        algod_url (str): Algorand node URL
        indexer_url (str): Algorand indexer URL

    Returns:
        indexer.Indexer: Algorand indexer client
    """
    try:
        indexer_client = indexer.Indexer(algod_url, indexer_url)
        logger.info("Indexer client created successfully.")
        return indexer_client
    except Exception as e:
        logger.error(f"Failed to create indexer client: {e}")
        raise

def get_account_balance(account_address: str, indexer_client: indexer.Indexer) -> int:
    """
    Returns the balance of an Algorand account.

    Args:
        account_address (str): Algorand account address
        indexer_client (indexer.Indexer): Algorand indexer client

    Returns:
        int: Account balance
    """
    try:
        account_info = indexer_client.account_info(account_address)
        balance = account_info['account']['amount']
        logger.info(f"Account balance for {account_address}: {balance}")
        return balance
    except AlgodHTTPError as e:
        logger.error(f"Error retrieving account balance: {e}")
        raise

def send_algorand_transaction(
    sender_address: str,
    recipient_address: str,
    amount: int,
    sender_private_key: str,
    indexer_client: indexer.Indexer
) -> str:
    """
    Sends an Algorand transaction.

    Args:
        sender_address (str): Sender's Algorand account address
        recipient_address (str): Recipient's Algorand account address
        amount (int): Transaction amount
        sender_private_key (str): Sender's private key
        indexer_client (indexer.Indexer): Algorand indexer client

    Returns:
        str: Transaction ID
    """
    try:
        # Create a transaction
        txn = transaction.PaymentTxn(
            sender=sender_address,
            receiver=recipient_address,
            amt=amount,
            sp=indexer_client.suggested_params()
        )
        
        # Sign the transaction
        signed_txn = txn.sign(sender_private_key)
        
        # Send the transaction
        txn_id = indexer_client.send_transaction(signed_txn)
        logger.info(f"Transaction sent with ID: {txn_id}")
        return txn_id
    except AlgodHTTPError as e:
        logger.error(f"Error sending transaction: {e}")
        raise

def get_transaction_details(txn_id: str, indexer_client: indexer.Indexer):
    """
    Retrieves details of a transaction by its ID.

    Args:
        txn_id (str): Transaction ID
        indexer_client (indexer.Indexer): Algorand indexer client

    Returns:
        dict: Transaction details
    """
    try:
        txn_info = indexer_client.transaction_info(txn_id)
        logger.info(f"Transaction details for ID {txn_id}: {txn_info}")
        return txn_info
    except AlgodHTTPError as e:
        logger.error(f"Error retrieving transaction details: {e}")
        raise
