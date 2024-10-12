// crypto.js
/**
 * @fileoverview Advanced cryptographic utility functions.
 * @author KOSASIH
 */

const crypto = require('crypto');

/**
 * Generates a random cryptographic key.
 * @param {number} size - The size of the key in bytes.
 * @returns {string} The generated key.
 */
function generateKey(size) {
  return crypto.randomBytes(size).toString('hex');
}

/**
 * Hashes a string using the SHA-512 algorithm.
 * @param {string} str - The input string.
 * @returns {string} The hashed string.
 */
function hashStringSha512(str) {
  const hash = crypto.createHash('sha512');
  hash.update(str);
  return hash.digest('hex');
}

/**
 * Encrypts a string using the AES-256-GCM algorithm.
 * @param {string} str - The input string.
 * @param {string} key - The encryption key.
 * @returns {string} The encrypted string.
 */
function encryptStringAes256Gcm(str, key) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  let encrypted = cipher.update(str, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  return iv.toString('hex') + ':' + encrypted + ':' + cipher.getAuthTag().toString('hex');
}

/**
 * Decrypts a string using the AES-256-GCM algorithm.
 * @param {string} str - The input string.
 * @param {string} key - The encryption key.
 * @returns {string} The decrypted string.
 */
function decryptStringAes256Gcm(str, key) {
  const textParts = str.split(':');
  const iv = Buffer.from(textParts.shift(), 'hex');
  const encryptedText = Buffer.from(textParts[0], 'hex');
  const authTag = Buffer.from(textParts[1], 'hex');
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(authTag);
  let decrypted = decipher.update(encryptedText);
  decrypted = Buffer.concat([decrypted, decipher.final()]);
  return decrypted.toString();
}

/**
 * Signs a string using the ECDSA algorithm with a private key.
 * @param {string} str - The input string.
 * @param {string} privateKey - The private key.
 * @returns {string} The signed string.
 */
function signStringEcdsa(str, privateKey) {
  const signer = crypto.createSign('sha256');
  signer.update(str);
  signer.end();
  return signer.sign(privateKey, 'hex');
}

/**
 * Verifies a signed string using the ECDSA algorithm with a public key.
 * @param {string} str - The input string.
 * @param {string} signature - The signature.
 * @param {string} publicKey - The public key.
 * @returns {boolean} True if the signature is valid, false otherwise.
 */
function verifyStringEcdsa(str, signature, publicKey) {
  const verifier = crypto.createVerify('sha256');
  verifier.update(str);
  verifier.end();
  return verifier.verify(publicKey, signature, 'hex');
}

export {
  generateKey,
  hashStringSha512,
  encryptStringAes256Gcm,
  decryptStringAes256Gcm,
  signStringEcdsa,
  verifyStringEcdsa,
};
