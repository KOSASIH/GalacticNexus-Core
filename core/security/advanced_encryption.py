import hashlib
import base64

class AdvancedEncryption:
    def __init__(self, config):
        self.config = config

    def encrypt_data(self, data):
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac('sha256', data.encode('utf-8'), salt, 100000)
        encrypted_data = base64.b64encode(key + data.encode('utf-8'))
        return encrypted_data
