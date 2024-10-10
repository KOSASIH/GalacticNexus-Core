# controllers/ai_controller.py

from flask import Blueprint, request, jsonify
from views.ai_views import ai_views

ai_controller = Blueprint('ai_controller', __name__)

@ai_controller.route('/ai/train', methods=['POST'])
def train_ai_model():
    return ai_views.train_ai_model()

@ai_controller.route('/ai/predict', methods=['POST'])
def predict_with_ai_model():
    return ai_views.predict_with_ai_model()
