import hashlib
import hmac
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class AdvancedEncryption:
    def __init__(self):
        self.key = hashlib.sha256(b"secret_key").digest()
        self.iv = os.urandom(16)

    def encrypt(self, data):
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return encrypted_data

    def decrypt(self, encrypted_data):
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(decrypted_padded_data) + unpadder.finalize()
        return data

    def hmac_sign(self, data):
        hmac_digest = hmac.new(self.key, data, hashlib.sha256).digest()
        return hmac_digest

    def hmac_verify(self, data, hmac_digest):
        expected_hmac_digest = hmac.new(self.key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected_hmac_digest, hmac_digest)
