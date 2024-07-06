import numpy as np
from qiskit import QuantumCircuit, execute

class QubitSimulator:
    def __init__(self):
        self.circuit = QuantumCircuit(2)

    def simulate(self, input_state):
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        job = execute(self.circuit, backend='qasm_simulator', shots=1024)
        result = job.result()
        return result.get_counts()
