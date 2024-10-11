import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.providers.aer.noise import depolarizing_error, NoiseModel
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.tools.monitor import job_monitor
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

class QuantumEncryption:
  def __init__(self, key_size=2048):
    self.key_size = key_size
    self.private_key = rsa.generate_private_key(
      public_exponent=65537,
      key_size=self.key_size,
      backend=default_backend()
    )
    self.public_key = self.private_key.public_key()

  def encrypt(self, message):
    # Convert message to binary
    message_binary = message.encode('utf-8')

    # Prepare quantum circuit for encryption
    quantum_circuit = QuantumCircuit(1, 1)
    quantum_circuit.h(0)
    quantum_circuit.barrier()

    # Encrypt message using quantum circuit
    encrypted_message = []
    for bit in message_binary:
      if bit == 0:
        quantum_circuit.i(0)
      else:
        quantum_circuit.x(0)
      quantum_circuit.barrier()
      job = execute(quantum_circuit, Aer.get_backend('qasm_simulator'), shots=1)
      result = job.result()
      counts = result.get_counts(quantum_circuit)
      encrypted_bit = list(counts.keys())[0][0]
      encrypted_message.append(encrypted_bit)

    # Convert encrypted message to bytes
    encrypted_message_bytes = bytes(encrypted_message)

    # Encrypt encrypted message using public key
    encrypted_message_ciphertext = self.public_key.encrypt(
      encrypted_message_bytes,
      padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
      )
    )

    return encrypted_message_ciphertext

  def decrypt(self, encrypted_message_ciphertext):
    # Decrypt encrypted message using private key
    encrypted_message_bytes = self.private_key.decrypt(
      encrypted_message_ciphertext,
      padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
      )
    )

    # Decrypt encrypted message using quantum circuit
    decrypted_message = []
    for encrypted_bit in encrypted_message_bytes:
      quantum_circuit = QuantumCircuit(1, 1)
      quantum_circuit.h(0)
      quantum_circuit.barrier()
      if encrypted_bit == 0:
        quantum_circuit.i(0)
      else:
        quantum_circuit.x(0)
      quantum_circuit.barrier()
      job = execute(quantum_circuit, Aer.get_backend('qasm_simulator'), shots=1)
      result = job.result()
      counts = result.get_counts(quantum_circuit)
      decrypted_bit = list(counts.keys())[0][0]
      decrypted_message.append(decrypted_bit)

    # Convert decrypted message to string
    decrypted_message_string = ''.join([str(bit) for bit in decrypted_message])

    return decrypted_message_string

# Example usage:
if __name__ == '__main__':
  quantum_encryption = QuantumEncryption()

  message = "Hello, Quantum World!"
  encrypted_message_ciphertext = quantum_encryption.encrypt(message)
  print("Encrypted message:", encrypted_message_ciphertext)

  decrypted_message = quantum_encryption.decrypt(encrypted_message_ciphertext)
  print("Decrypted message:", decrypted_message)
