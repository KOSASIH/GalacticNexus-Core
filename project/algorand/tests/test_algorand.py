# project/algorand/tests/test_algorand.py

import unittest
from unittest.mock import patch, MagicMock
from project.algorand.core.algorand import (
    get_algorand_indexer,
    get_account_balance,
    send_algorand_transaction,
    get_transaction_details
)

class TestAlgorandModule(unittest.TestCase):

    @patch('project.algorand.core.algorand.indexer.Indexer')
    def test_get_algorand_indexer(self, mock_indexer):
        mock_indexer.return_value = MagicMock()
        indexer_client = get_algorand_indexer("http://algod_url", "http://indexer_url")
        self.assertIsNotNone(indexer_client)

    @patch('project.algorand.core.algorand.indexer.Indexer')
    def test_get_account_balance(self, mock_indexer):
        mock_indexer.return_value.account_info.return_value = {'account': {'amount': 1000}}
        balance = get_account_balance("test_address", mock_indexer())
        self.assertEqual(balance, 1000)

    @patch('project.algorand.core.algorand.indexer.Indexer')
    def test_send_algorand_transaction(self, mock_indexer):
        mock_indexer.return_value.suggested_params.return_value = MagicMock()
        mock_indexer.return_value.send_transaction.return_value = "txn_id"
        txn_id = send_algorand_transaction("sender_address", "recipient_address", 1000, "private_key", mock_indexer())
        self.assertEqual(txn_id, "txn_id")

    @patch('project.algorand.core.algorand.indexer.Indexer')
    def test_get_transaction_details(self, mock_indexer):
        mock_indexer.return_value.transaction_info.return_value = {'transaction': 'details'}
        txn_info = get_transaction_details("txn_id", mock_indexer())
        self.assertEqual(txn_info, {'transaction': 'details'})

if __name__ == '__main__':
    unittest.main()
