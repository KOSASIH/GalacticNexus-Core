import hashlib
import json

class Wallet:
    def __init__(self, private_key):
        self.private_key = private_key
        self.public_key = self.generate_public_key(private_key)

    def generate_public_key(self, private_key):
        # Generate public key from private key using elliptic curve cryptography
        # This is a simplified example and actual implementation may vary
        return hashlib.sha256(private_key.encode()).hexdigest()

    def sign_transaction(self, transaction):
        # Sign transaction using private key
        # This is a simplified example and actual implementation may vary
        return hashlib.sha256((transaction + self.private_key).encode()).hexdigest()

    def get_balance(self, blockchain):
        # Get balance from blockchain
        # This is a simplified example and actual implementation may vary
        return blockchain.get_balance(self.public_key)

# Create a new wallet
wallet = Wallet('my_private_key')

# Sign a transaction
transaction = {'from': wallet.public_key, 'to': 'recipient_public_key', 'amount': 10}
signature = wallet.sign_transaction(json.dumps(transaction))

# Get balance
balance = wallet.get_balance(blockchain)
print(balance)
