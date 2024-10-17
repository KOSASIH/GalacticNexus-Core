# project/algorand/performance/performance_optimization.py

import time
from project.algorand.core.algorand import get_account_balance, send_algorand_transaction

def optimize_account_balance_check(address, indexer_client):
    """
    Optimize the account balance check by caching results.
    """
    cache = {}

    def cached_get_account_balance(address):
        if address in cache:
            return cache[address]
        balance = get_account_balance(address, indexer_client)
        cache[address] = balance
        return balance

    return cached_get_account_balance

def batch_send_transactions(transactions, indexer_client):
    """
    Send multiple transactions in a batch to reduce overhead.
    """
    txn_ids = []
    for txn in transactions:
        txn_id = send_algorand_transaction(
            txn['sender'],
            txn['recipient'],
            txn['amount'],
            txn['private_key'],
            indexer_client
        )
        txn_ids.append(txn_id)
    return txn_ids

def measure_execution_time(func, *args, **kwargs):
    """
    Measure the execution time of a function.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
    return result
