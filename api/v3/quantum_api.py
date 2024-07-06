from flask import Flask, request, jsonify
from galactic_nexus_core.quantum import QuantumComputer

app = Flask(__name__)

@app.route('/api/v3/quantum', methods=['POST'])
def process_quantum_data():
    quantum_computer = QuantumComputer()
    data = request.get_json()
    result = quantum_computer.process_data(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
