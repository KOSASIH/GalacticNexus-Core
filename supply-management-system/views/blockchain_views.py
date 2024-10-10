# views/blockchain_views.py

from flask import Blueprint, request, jsonify
from models.blockchain import Blockchain

blockchain_views = Blueprint('blockchain_views', __name__)

@blockchain_views.route('/blockchain', methods=['GET'])
def get_chain():
    blockchain = Blockchain()
    return blockchain.get_chain()

@blockchain_views.route('/blockchain/pending_transactions', methods=['GET'])
def get_pending_transactions():
    blockchain = Blockchain()
    return blockchain.get_pending_transactions()

@blockchain_views.route('/blockchain/mine', methods=['POST'])
def mine_pending_transactions():
    blockchain = Blockchain()
    blockchain.mine_pending_transactions()
    return jsonify({'message': 'Pending transactions mined successfully'})
