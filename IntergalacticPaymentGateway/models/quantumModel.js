// quantumModel.js

const mongoose = require('mongoose');
const Qiskit = require('qiskit');

const quantumSchema = new mongoose.Schema({
  circuit: {
    type: String,
    required: true,
  },
  qubits: {
    type: Number,
    required: true,
  },
  gates: {
    type: Array,
    required: true,
  },
  shots: {
    type: Number,
    required: true,
  },
  backend: {
    type: String,
    required: true,
  },
});

quantumSchema.methods.runCircuit = function() {
  const circuit = Qiskit.Circuit.fromQasm(this.circuit);
  const backend = new Qiskit.Aer.get_backend(this.backend);
  const job = backend.run(circuit, shots=this.shots);
  return job.result();
};

const Quantum = mongoose.model('Quantum', quantumSchema);

module.exports = Quantum;
