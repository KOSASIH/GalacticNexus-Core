import numpy as np
from neuroquantum_bridge_algorithm import NeuroQuantumBridgeAlgorithm

def main():
  config = json.load(open("config/neuroquantum_bridge_config.json"))
  neuroquantum_bridge_algorithm = NeuroQuantumBridgeAlgorithm(config)
  brain_signal = np.random.rand(100, 100)
  encrypted_output = neuroquantum_bridge_algorithm.integrate_neural_interface_and_quantum_encryption(brain_signal)
  print(encrypted_output)

if __name__ == "__main__":
  main()
