import pytest
from galacticnexus_core.security import encrypt, decrypt

def test_encrypt_decrypt_success():
    # Test successful encryption and decryption
    plaintext = "This is a secret message"
    encrypted_text = encrypt(plaintext)
    decrypted_text = decrypt(encrypted_text)
    assert decrypted_text == plaintext

def test_encrypt_decrypt_failure():
    # Test failed encryption and decryption
    plaintext = "This is an invalid message"
    encrypted_text = encrypt(plaintext)
    decrypted_text = decrypt(encrypted_text)
    assert decrypted_text != plaintext

def test_encrypt_decrypt_empty():
    # Test encryption and decryption with empty input
    plaintext = ""
    encrypted_text = encrypt(plaintext)
    decrypted_text = decrypt(encrypted_text)
    assert decrypted_text == plaintext

def test_encrypt_decrypt_invalid_type():
    # Test encryption and decryption with invalid input type
    plaintext = 12345
    encrypted_text = encrypt(plaintext)
    decrypted_text = decrypt(encrypted_text)
    assert decrypted_text != plaintext
