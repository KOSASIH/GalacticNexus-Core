#include <openssl/rsa.h>
#include <openssl/pem.h>

class EncryptionService {
public:
  RSA* generateKeyPair() {
    RSA* rsa = RSA_new();
    BIGNUM* exponent = BN_new();
    BN_set_word(exponent, 65537);
    RSA_generate_key_ex(rsa, 2048, exponent, nullptr);
    return rsa;
  }

  std::vector<uint8_t> encryptData(const std::vector<uint8_t>& data, RSA* publicKey) {
    std::vector<uint8_t> encryptedData;
    int encryptedLength = RSA_size(publicKey);
    encryptedData.resize(encryptedLength);
    int encrypted = RSA_public_encrypt(data.size(), data.data(), encryptedData.data(), publicKey, RSA_PKCS1_OAEP_PADDING);
    return encryptedData;
  }
};
