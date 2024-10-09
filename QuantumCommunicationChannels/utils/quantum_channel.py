import numpy as np

def quantum_channel(data, attenuation):
    # Simulate the quantum channel
    noisy_data = np.random.binomial(1, attenuation, size=data.shape)
    return noisy_data
