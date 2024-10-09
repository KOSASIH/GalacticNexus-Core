import config
from protocols import bb84, b92, ekert91, quantum_teleportation
from utils import error_correction, key_exchange, quantum_channel

def main():
    config = config.config
    alice_key = np.random.randint(0, 2, size=1024)
    bob_key = np.random.randint(0, 2, size=1024)

    # Quantum key distribution
    shared_key = bb84_protocol(alice_key, bob_key)

    # Error correction
    corrected_key = error_correction.reed_solomon_error_correction(shared_key, 256)

    # Key exchange
    shared_key = key_exchange(corrected_key, bob_key)

    # Quantum teleportation
    teleported_key = quantum_teleportation(shared_key, bob_key)

    print("Shared Key:", shared_key)
    print("Teleported Key:", teleported_key)

if __name__ == "__main__":
    main()
