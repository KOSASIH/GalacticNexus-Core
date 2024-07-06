from flask import Flask, request, jsonify
from galactic_nexus_core.neural import NeuralNetwork

app = Flask(__name__)

@app.route('/api/v4/neural', methods=['POST'])
def process_neural_data():
    neural_network = NeuralNetwork()
    data = request.get_json()
    result = neural_network.process_data(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
