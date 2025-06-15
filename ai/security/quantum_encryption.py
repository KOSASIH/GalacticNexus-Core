# ai/security/quantum_encryption.py

import logging
import warnings

try:
    import pqcrypto
    PQCRYPTO_AVAILABLE = True
except ImportError:
    PQCRYPTO_AVAILABLE = False
    warnings.warn("pqcrypto not installed. Quantum-safe encryption simulation only.")

# Example registry of supported algorithms
ENCRYPTION_ALGORITHMS = {
    "kyber": {
        "label": "Kyber (Post-Quantum Lattice-Based)",
        "quantum_safe": True,
        "lib": "pqcrypto" if PQCRYPTO_AVAILABLE else None,
        "recommended": True
    },
    "dilithium": {
        "label": "Dilithium (Post-Quantum Digital Signature)",
        "quantum_safe": True,
        "lib": "pqcrypto" if PQCRYPTO_AVAILABLE else None,
        "recommended": True
    },
    "ntru": {
        "label": "NTRUEncrypt (Post-Quantum Lattice-Based)",
        "quantum_safe": True,
        "lib": "pqcrypto" if PQCRYPTO_AVAILABLE else None,
        "recommended": False
    },
    "aes-256": {
        "label": "AES-256 (Classical Symmetric)",
        "quantum_safe": False,
        "lib": "cryptography",
        "recommended": True
    }
}

def select_encryption_scheme(context):
    """
    Advanced selector for quantum-safe encryption algorithms.
    Uses context awareness and extensibility for new methods.
    """
    logging.info(f"Selecting encryption scheme with context: {context}")

    # High-sensitivity or regulatory contexts demand quantum-safe
    if context.get("quantum_required") or context.get("high_sensitivity"):
        # Prefer strongest available quantum-safe
        for alg in ["kyber", "dilithium", "ntru"]:
            if ENCRYPTION_ALGORITHMS[alg]["quantum_safe"] and ENCRYPTION_ALGORITHMS[alg]["recommended"]:
                if PQCRYPTO_AVAILABLE:
                    logging.info(f"Selected quantum-safe: {ENCRYPTION_ALGORITHMS[alg]['label']}")
                    return ENCRYPTION_ALGORITHMS[alg]['label']
                else:
                    logging.warning("Quantum-safe library not available, falling back.")
        # If not available, fallback
        logging.error("No quantum-safe algorithm available, falling back to AES-256!")
        return ENCRYPTION_ALGORITHMS["aes-256"]["label"]

    # For non-sensitive, classical use
    logging.info("Selected classical: AES-256")
    return ENCRYPTION_ALGORITHMS["aes-256"]["label"]

def list_supported_algorithms():
    """
    List all available encryption algorithms.
    """
    return [{**{"name": key}, **value} for key, value in ENCRYPTION_ALGORITHMS.items()]

def is_quantum_safe(algorithm_name):
    """
    Check if the chosen algorithm is quantum-safe.
    """
    return ENCRYPTION_ALGORITHMS.get(algorithm_name.lower(), {}).get("quantum_safe", False)

# Example usage
if __name__ == "__main__":
    context1 = {"high_sensitivity": True}
    context2 = {"quantum_required": False}
    print("Selected (context1):", select_encryption_scheme(context1))
    print("Selected (context2):", select_encryption_scheme(context2))
    print("Supported algorithms:", list_supported_algorithms())
