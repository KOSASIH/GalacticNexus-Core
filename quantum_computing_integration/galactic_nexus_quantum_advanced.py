from galactic_nexus_quantum import GalacticNexusQuantum
from galactic_nexus_quantum_config import config

# Create a GalacticNexusQuantum instance
quantum = GalacticNexusQuantum(config)

# Perform quantum key distribution with multiple qubits
counts = quantum.quantum_key_distribution(config['num_qubits'] * 2)
print("Quantum Key Distribution with Multiple Qubits:", counts)

# Perform quantum encryption with multiple keys
encrypted_message = quantum.quantum_encryption(config['message'], config['key'] * 2)
print("Quantum Encrypted Message with Multiple Keys:", encrypted_message)

# Perform quantum resistant cryptography with multiple keys
encrypted_message = quantum.quantum_resistant_cryptography(config['message'], config['key'] * 2)
print("Quantum Resistant Encrypted Message with Multiple Keys:", encrypted_message)

# Perform quantum teleportation with multiple qubits
counts = quantum.quantum_teleportation(config['message'], config['key'] * 2)
print("Quantum Teleportation with Multiple Qubits:", counts)

# Perform quantum superdense coding with multiple qubits
counts = quantum.quantum_superdense_coding(config['message'], config['key'] * 2)
print("Quantum Superdense Coding with Multiple Qubits:", counts)

# Perform quantum error correction with multiple qubits
counts = quantum.quantum_error_correction(config['message'], config['key'] * 2)
print("Quantum Error Correction with Multiple Qubits:", counts)

# Perform quantum simulation with multiple qubits
counts = quantum.quantum_simulation(config['message'], config['key'] * 2)
print("Quantum Simulation with Multiple Qubits:", counts)

# Perform quantum machine learning with multiple qubits
counts = quantum.quantum_machine_learning(config['message'], config['key'] * 2)
print("Quantum Machine Learning with Multiple Qubits:", counts)

# Perform quantum neural networks with multiple qubits
counts = quantum.quantum_neural_networks(config['message'], config['key'] * 2)
print("Quantum Neural Networks with Multiple Qubits:", counts)

# Perform quantum optimization with multiple qubits
counts = quantum.quantum_optimization(config['message'], config['key'] * 2)
print("Quantum Optimization with Multiple Qubits:", counts)
