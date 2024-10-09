import numpy as np

def key_exchange(alice_key, bob_key):
    # Alice and Bob publicly compare their bit strings
    alice_error_rate = np.count_nonzero(alice_key != bob_key) / 1024
    if alice_error_rate > 0.1:
        raise ValueError("Error rate too high")

    # Alice and Bob use the shared key for secure communication
    shared_key = alice_key
    return shared_key
