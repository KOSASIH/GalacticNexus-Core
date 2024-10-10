# routes/blockchain_routes.py

from flask import Blueprint
from controllers.blockchain_controller import blockchain_controller

blockchain_routes = Blueprint('blockchain_routes', __name__)

@blockchain_routes.route('/blockchain', methods=['GET'])
def get_chain():
    return blockchain_controller.get_chain()

@blockchain_routes.route('/blockchain/pending_transactions', methods=['GET'])
def get_pending_transactions():
    return blockchain_controller.get_pending_transactions()

@blockchain_routes.route('/blockchain/mine', methods=['POST'])
def mine_pending_transactions():
    return blockchain_controller.mine_pending_transactions()
