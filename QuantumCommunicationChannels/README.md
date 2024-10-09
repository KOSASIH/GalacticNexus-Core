# Secure Quantum Communication Channels

This is a quantum-encrypted communication protocol that enables secure, faster-than-light communication between nodes on the network, ensuring the integrity of transactions and data exchange.

## Installation

1. Clone the repository: `git clone https://github.com/KOSASIH/secure-quantum-communication-channels.git`
2. Install the requirements : `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Usage

1. Generate a shared key using quantum key distribution: `bb84_protocol(alice_key, bob_key)`
2. Perform error correction using Reed-Solomon codes: `error_correction.reed_solomon_error_correction(shared_key, 256)`
3. Exchange the shared key using key exchange: `key_exchange(corrected_key, bob_key)`
4. Teleport the shared key using quantum teleportation: `quantum_teleportation(shared_key, bob_key)`

## Protocols

* BB84 Protocol: `bb84.py`
* B92 Protocol: `b92.py`
* Ekert91 Protocol: `ekert91.py`
* Quantum Teleportation: `quantum_teleportation.py`

## Utilities

* Error Correction: `error_correction.py`
* Key Exchange: `key_exchange.py`
* Quantum Channel: `quantum_channel.py`
