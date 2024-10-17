# project/algorand/tests/test_utils.py

import unittest
from unittest.mock import patch, MagicMock
from project.algorand.core.utils import generate_algorand_wallet, get_algorand_address_from_mnemonic

class TestUtilsModule(unittest.TestCase):

    @patch('project.algorand.core.utils.account.generate_account')
    def test_generate_algorand_wallet(self, mock_generate_account):
        mock_generate_account.return_value = ("mnemonic_phrase", "wallet_address")
        mnemonic, address = generate_algorand_wallet()
        self.assertEqual(mnemonic, "mnemonic_phrase")
        self.assertEqual(address, "wallet_address")

    @patch('project.algorand.core.utils.account.address_from_mnemonic')
    def test_get_algorand_address_from_mnemonic(self, mock_address_from_mnemonic):
        mock_address_from_mnemonic.return_value = "wallet_address"
        address = get_algorand_address_from_mnemonic("mnemonic_phrase")
        self.assertEqual(address, "wallet_address")

if __name__ == '__main__':
    unittest.main()
