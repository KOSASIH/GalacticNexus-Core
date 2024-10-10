class Asset:
    def __init__(self, asset_id, asset_name, asset_type, price, quantity):
        self.asset_id = asset_id
        self.asset_name = asset_name
        self.asset_type = asset_type
        self.price = price
        self.quantity = quantity

    def tokenize(self):
        # Tokenize the asset using the Algorand SDK
        txn = algorand.tokenize_asset(self.asset_id, self.asset_name, self.asset_type, self.price, self.quantity)
        return txn

    def trade(self, buyer_sk, seller_sk, quantity):
        # Trade the asset using the Algorand SDK
        txn = algorand.trade_asset(self.asset_id, buyer_sk, seller_sk, quantity)
        return txn
