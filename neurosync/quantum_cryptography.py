import numpy as np

def encrypt_data(data, key):
    # Encrypt data using the quantum key
    encrypted_data = np.bitwise_xor(data, key)
    return encrypted_data

def decrypt_data(encrypted_data, key):
    # Decrypt data using the quantum key
    decrypted_data = np.bitwise_xor(encrypted_data, key)
    return decrypted_data
