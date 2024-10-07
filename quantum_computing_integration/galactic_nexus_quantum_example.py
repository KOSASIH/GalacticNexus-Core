from galactic_nexus_quantum import GalacticNexusQuantum
from galactic_nexus_quantum_config import config

# Create a GalacticNexusQuantum instance
quantum = GalacticNexusQuantum(config)

# Perform quantum key distribution
counts = quantum.quantum_key_distribution(config['num_qubits'])
print("Quantum Key Distribution:", counts)

# Perform quantum encryption
encrypted_message = quantum.quantum_encryption(config['message'], config['key'])
print("Quantum Encrypted Message:", encrypted_message)

# Perform quantum resistant cryptography
encrypted_message = quantum.quantum_resistant_cryptography(config['message'], config['key'])
print("Quantum Resistant Encrypted Message:", encrypted_message)

# Perform quantum teleportation
counts = quantum.quantum_teleportation(config['message'], config['key'])
print("Quantum Teleportation:", counts)

# Perform quantum superdense coding
counts = quantum.quantum_superdense_coding(config['message'], config['key'])
print("Quantum Superdense Coding:", counts)

# Perform quantum error correction
counts = quantum.quantum_error_correction(config['message'], config['key'])
print("Quantum Error Correction:", counts)

# Perform quantum simulation
counts = quantum.quantum_simulation(config['message'], config['key'])
print("Quantum Simulation:", counts)

# Perform quantum machine learning
counts = quantum.quantum_machine_learning(config['message'], config['key'])
print("Quantum Machine Learning:", counts)

# Perform quantum neural networks
counts = quantum.quantum_neural_networks(config['message'], config['key'])
print("Quantum Neural Networks:", counts)

# Perform quantum optimization
counts = quantum.quantum_optimization(config['message'], config['key'])
print("Quantum Optimization:", counts)
