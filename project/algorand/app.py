from flask import Flask, request, jsonify
from models.asset import Asset
from models.market import Market
from utils.config import API_KEY, MNEMONIC_PHRASE, NODE_URL

app = Flask(__name__)

# Set up Algorand node connection
algod_client = algod.AlgodClient(API_KEY, NODE_URL)

# Set up wallet
wallet_mnemonic = MNEMONIC_PHRASE
wallet_sk = mnemonic.to_private_key(wallet_mnemonic)
wallet_pk = mnemonic.to_public_key(wallet_mnemonic)

# Create a new asset
@app.route("/create_asset", methods=["POST"])
def create_asset():
    asset_name = request.json["asset_name"]
    asset_type = request.json["asset_type"]
    price = request.json["price"]
    quantity = request.json["quantity"]
    asset = Asset(None, asset_name, asset_type, price, quantity)
    txn = asset.tokenize()
    return jsonify({"txn": txn})

# Trade an asset
@app.route("/trade_asset", methods=["POST"])
def trade_asset():
    asset_id = request.json["asset_id"]
    buyer_sk = request.json["buyer_sk"]
    seller_sk = request.json["seller_sk"]
    quantity = request.json["quantity"]
    asset = Asset(asset_id, None, None, None, None)
    txn = asset.trade(buyer_sk, seller_sk, quantity)
    return jsonify({"txn": txn})

# Get market data
@app.route("/get_market_data", methods=["GET"])
def get_market_data():
    market = Market(None, None, None )
    market_data = market.get_market_data()
    return jsonify({"market_data": market_data})

if __name__ == "__main__":
    app.run(debug=True)
