// string.js
/**
 * @fileoverview Utility functions for string manipulation.
 * @author KOSASIH
 */

/**
 * Trims a string and converts it to title case.
 * @param {string} str - The input string.
 * @returns {string} The trimmed and title-cased string.
 */
function trimAndTitleCase(str) {
  return str.trim().replace(/\w\S*/g, (word) => {
    return word.charAt(0).toUpperCase() + word.substring(1).toLowerCase();
  });
}

/**
 * Converts a string to snake case.
 * @param {string} str - The input string.
 * @returns {string} The snake-cased string.
 */
function toSnakeCase(str) {
  return str.replace(/([A-Z])/g, '_$1').toLowerCase();
}

/**
 * Converts a string to camel case.
 * @param {string} str - The input string.
 * @returns {string} The camel-cased string.
 */
function toCamelCase(str) {
  return str.replace(/_([a-z])/g, (match, group) => {
    return group.toUpperCase();
  });
}

/**
 * Checks if a string is a valid email address.
 * @param {string} str - The input string.
 * @returns {boolean} True if the string is a valid email address, false otherwise.
 */
function isValidEmail(str) {
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  return emailRegex.test(str);
}

/**
 * Checks if a string is a valid phone number.
 * @param {string} str - The input string.
 * @returns {boolean} True if the string is a valid phone number, false otherwise.
 */
function isValidPhoneNumber(str) {
  const phoneRegex = /^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$/;
  return phoneRegex.test(str);
}

/**
 * Generates a random string of a specified length.
 * @param {number} length - The length of the string to generate.
 * @returns {string} A random string of the specified length.
 */
function generateRandomString(length) {
  const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

/**
 * Hashes a string using the SHA-256 algorithm.
 * @param {string} str - The input string.
 * @returns {string} The hashed string.
 */
function hashString(str) {
  const crypto = require('crypto');
  const hash = crypto.createHash('sha256');
  hash.update(str);
  return hash.digest('hex');
}

/**
 * Encrypts a string using the AES algorithm.
 * @param {string} str - The input string.
 * @param {string} key - The encryption key.
 * @returns {string} The encrypted string.
 */
function encryptString(str, key) {
  const crypto = require('crypto');
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
  let encrypted = cipher.update(str, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  return iv.toString('hex') + ':' + encrypted;
}

/**
 * Decrypts a string using the AES algorithm.
 * @param {string} str - The input string.
 * @param {string} key - The encryption key.
 * @returns {string} The decrypted string.
 */
function decryptString(str, key) {
  const crypto = require('crypto');
  const textParts = str.split(':');
  const iv = Buffer.from(textParts.shift(), 'hex');
  const encryptedText = Buffer.from(textParts.join(':'), 'hex');
  const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
  let decrypted = decipher.update(encryptedText);
  decrypted = Buffer.concat([decrypted, decipher.final()]);
  return decrypted.toString();
}

/**
 * Checks if a string contains a specified substring.
 * @param {string} str - The input string.
 * @param {string} substring - The substring to search for.
 * @returns {boolean} True if the string contains the substring, false otherwise.
 */
function containsSubstring(str, substring) {
  return str.includes(substring);
}

/**
 * Replaces all occurrences of a specified substring in a string.
 * @param {string} str - The input string.
 * @param {string} substring - The substring to replace.
 * @param {string} replacement - The replacement string.
 * @returns {string} The modified string.
 */
function replaceSubstring(str, substring, replacement) {
 return str.replace(new RegExp(substring, 'g'), replacement);
}

/**
 * Converts a string to base64.
 * @param {string} str - The input string.
 * @returns {string} The base64-encoded string.
 */
function toBase64(str) {
  return Buffer.from(str).toString('base64');
}

/**
 * Converts a base64-encoded string to a regular string.
 * @param {string} str - The base64-encoded string.
 * @returns {string} The decoded string.
 */
function fromBase64(str) {
  return Buffer.from(str, 'base64').toString();
}

/**
 * Checks if a string is a valid URL.
 * @param {string} str - The input string.
 * @returns {boolean} True if the string is a valid URL, false otherwise.
 */
function isValidUrl(str) {
  try {
    new URL(str);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Extracts the domain from a URL.
 * @param {string} str - The input URL.
 * @returns {string} The extracted domain.
 */
function extractDomain(str) {
  const url = new URL(str);
  return url.hostname;
}

/**
 * Extracts the protocol from a URL.
 * @param {string} str - The input URL.
 * @returns {string} The extracted protocol.
 */
function extractProtocol(str) {
  const url = new URL(str);
  return url.protocol;
}

export {
  trimAndTitleCase,
  toSnakeCase,
  toCamelCase,
  isValidEmail,
  isValidPhoneNumber,
  generateRandomString,
  hashString,
  encryptString,
  decryptString,
  containsSubstring,
  replaceSubstring,
  toBase64,
  fromBase64,
  isValidUrl,
  extractDomain,
  extractProtocol,
};
