// Import the necessary libraries
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

// Define the gnx_security functions
async function generateToken(username) {
  const token = await jwt.sign({ username }, process.env.SECRET_KEY, { expiresIn: '1h' });
  return token;
}

async function verifyToken(token) {
  try {
    const decoded = await jwt.verify(token, process.env.SECRET_KEY);
    return decoded;
  } catch (error) {
    return null;
  }
}

async function hashPassword(password) {
  const hashedPassword = await bcrypt.hash(password, 10);
  return hashedPassword;
}

async function comparePasswords(password, hashedPassword) {
  const isValid = await bcrypt.compare(password, hashedPassword);
  return isValid;
}

// Export the gnx_security functions
module.exports = { generateToken, verifyToken, hashPassword, comparePasswords };
