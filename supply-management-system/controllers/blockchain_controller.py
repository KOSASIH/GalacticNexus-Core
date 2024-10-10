# controllers/blockchain_controller.py

from flask import Blueprint, request, jsonify
from views.blockchain_views import blockchain_views

blockchain_controller = Blueprint('blockchain_controller', __name__)

@blockchain_controller.route('/blockchain', methods=['GET'])
def get_chain():
    return blockchain_views.get_chain()

@blockchain_controller.route('/blockchain/pending_transactions', methods=['GET'])
def get_pending_transactions():
    return blockchain_views.get_pending_transactions()

@blockchain_controller.route('/blockchain/mine', methods=['POST'])
def mine_pending_transactions():
    return blockchain_views.mine_pending_transactions()
