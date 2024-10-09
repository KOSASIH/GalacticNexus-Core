import numpy as np

def reed_solomon_error_correction(data, block_size):
    # Encode the data using Reed-Solomon codes
    encoded_data = np.zeros((data.shape[0], block_size), dtype=int)
    for i in range(data.shape[0]):
        encoded_data[i] = np.polyval(data[i], np.arange(block_size))

    # Add errors to the encoded data
    error_rate = 0.1
    errors = np.random.randint(0, 2, size=(data.shape[0], block_size))
    encoded_data += errors

    # Decode the encoded data using Reed-Solomon codes
    decoded_data = np.zeros((data.shape[0], block_size), dtype=int)
    for i in range(data.shape[0]):
        decoded_data[i] = np.polyval(encoded_data[i], np.arange(block_size))

    return decoded_data
