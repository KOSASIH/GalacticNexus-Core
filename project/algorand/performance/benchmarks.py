# project/algorand/performance/benchmarks.py

from project.algorand.performance.performance_optimization import (
    optimize_account_balance_check,
    batch_send_transactions,
    measure_execution_time
)
from project.algorand.core.algorand import get_algorand_indexer

def benchmark_account_balance_check(address, indexer_client):
    cached_balance_checker = optimize_account_balance_check(address, indexer_client)
    measure_execution_time(cached_balance_checker)

def benchmark_batch_send_transactions(transactions, indexer_client):
    measure_execution_time(batch_send_transactions, transactions, indexer_client)

if __name__ == "__main__":
    indexer_client = get_algorand_indexer("http://algod_url", "http://indexer_url")
    address = "test_address"
    
    # Benchmark account balance check
    benchmark_account_balance_check(address, indexer_client)

    # Benchmark batch sending transactions
    transactions = [
        {"sender": "sender1", "recipient": "recipient1", "amount": 1000, "private_key": "key1"},
        {"sender": "sender2", "recipient": "recipient2", "amount": 2000, "private_key": "key2"},
    ]
    benchmark_batch_send_transactions(transactions, indexer_client)
