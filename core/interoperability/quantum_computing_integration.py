import qiskit
from qiskit import QuantumCircuit, execute
from quantum_computing_config import QuantumComputingConfig

class QuantumComputingIntegration:
    def __init__(self):
        self.config = QuantumComputingConfig()
        self.backend = self.config.get_backend()

    def run_shor_algorithm(self, N):
        qc = QuantumCircuit(5, 5)
        qc.h(0)
        qc.cx(0, 1)
        qc.u1(np.pi/2, 1)
        qc.u2(np.pi/2, 2)
        qc.u3(np.pi/2, 3)
        qc.measure_all()
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        return counts

    def run_grover_algorithm(self, N):
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.u1(np.pi/2, 1)
        qc.u2(np.pi/2, 2)
        qc.measure_all()
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        return counts

    def integrate_quantum_computing(self, N):
        shor_result = self.run_shor_algorithm(N)
        grover_result = self.run_grover_algorithm(N)
        combined_result = {"Shor's Algorithm": shor_result, "Grover's Algorithm": grover_result}
        return combined_result

# Example usage
qc_integration = QuantumComputingIntegration()
combined_result = qc_integration.integrate_quantum_computing(15)
print("Combined Result:", combined_result)
