# project/algorand/core/utils.py

"""
Galactic Nexus Utilities Module

This module provides utility functions for generating and managing Algorand wallets.
"""

import logging
from algosdk import mnemonic, account

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_algorand_wallet() -> tuple:
    """
    Generates a new Algorand wallet.

    Returns:
        tuple: Wallet's mnemonic phrase and address
    """
    try:
        wallet_mnemonic, wallet_address = account.generate_account()
        logger.info(f"Wallet generated with address: {wallet_address}")
        return wallet_mnemonic, wallet_address
    except Exception as e:
        logger.error(f"Failed to generate wallet: {e}")
        raise

def get_algorand_address_from_mnemonic(mnemonic_phrase: str) -> str:
    """
    Retrieves an Algorand address from a mnemonic phrase.

    Args:
        mnemonic_phrase (str): Mnemonic phrase

    Returns:
        str: Algorand address
    """
    try:
        wallet_address = account.address_from_mnemonic(mnemonic_phrase)
        logger.info(f"Address retrieved from mnemonic: {wallet_address}")
        return wallet_address
    except Exception as e:
        logger.error(f"Failed to retrieve address from mnemonic: {e}")
        raise
