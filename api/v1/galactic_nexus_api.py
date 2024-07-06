from flask import Flask, request, jsonify
from galactic_nexus_core.logic import GalacticNexusCore

app = Flask(__name__)

@app.route('/api/v1/galactic_nexus', methods=['GET'])
def get_galactic_nexus():
    galactic_nexus_core = GalacticNexusCore()
    data = galactic_nexus_core.get_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
