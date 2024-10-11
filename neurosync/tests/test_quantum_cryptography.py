import unittest
from neurosync.utils.quantum_utils import measure_quantum_circuit

class TestQuantumCryptography(unittest.TestCase):
    def test_measure_quantum_circuit(self):
        # Test measuring a quantum circuit
        circuit = QuantumCircuit(5)
        circuit.h(range(5))
        circuit.measure_all()
        counts = measure_quantum_circuit(circuit)
        self.assertIsNotNone(counts)

if __name__ == '__main__':
    unittest.main()
