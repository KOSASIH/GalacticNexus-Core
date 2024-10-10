# views/ai_views.py

from flask import Blueprint, request, jsonify
from models.ai_model import AIModel

ai_views = Blueprint('ai_views', __name__)

@ai_views.route('/ai/train', methods=['POST'])
def train_ai_model():
    ai_model = AIModel()
    ai_model.train_model()
    return jsonify({'message': 'AI model trained successfully'})

@ai_views.route('/ai/predict', methods=['POST'])
def predict_with_ai_model():
    ai_model = AIModel()
    input_data = request.get_json()
    prediction = ai_model.predict(input_data)
    return jsonify({'prediction': prediction})
