# routes/ai_routes.py

from flask import Blueprint
from controllers.ai_controller import ai_controller

ai_routes = Blueprint('ai_routes', __name__)

@ai_routes.route('/ai/train', methods=['POST'])
def train_ai_model():
    return ai_controller.train_ai_model()

@ai_routes.route('/ai/predict', methods=['POST'])
def predict_with_ai_model():
    return ai_controller.predict_with_ai_model()
