// math.js
/**
 * @fileoverview Advanced mathematical utility functions.
 * @author KOSASIH
 */

/**
 * Calculates the factorial of a number.
 * @param {number} n - The input number.
 * @returns {number} The factorial of the number.
 */
function factorial(n) {
  if (n <= 1) return 1;
  return n * factorial(n - 1);
}

/**
 * Calculates the greatest common divisor (GCD) of two numbers.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The GCD of the two numbers.
 */
function gcd(a, b) {
  if (b === 0) return a;
  return gcd(b, a % b);
}

/**
 * Calculates the least common multiple (LCM) of two numbers.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The LCM of the two numbers.
 */
function lcm(a, b) {
  return a * b / gcd(a, b);
}

/**
 * Calculates the nth root of a number.
 * @param {number} base - The base number.
 * @param {number} n - The root index.
 * @returns {number} The nth root of the base number.
 */
function nthRoot(base, n) {
  return Math.pow(base, 1 / n);
}

/**
 * Calculates the modulo operation of two numbers.
 * @param {number} a - The dividend.
 * @param {number} b - The divisor.
 * @returns {number} The remainder of the division.
 */
function mod(a, b) {
  return a % b;
}

/**
 * Calculates the Euclidean distance between two points.
 * @param {number} x1 - The x-coordinate of the first point.
 * @param {number} y1 - The y-coordinate of the first point.
 * @param {number} x2 - The x-coordinate of the second point.
 * @param {number} y2 - The y-coordinate of the second point.
 * @returns {number} The Euclidean distance between the two points.
 */
function euclideanDistance(x1, y1, x2, y2) {
  return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
}

/**
 * Calculates the Manhattan distance between two points.
 * @param {number} x1 - The x-coordinate of the first point.
 * @param {number} y1 - The y-coordinate of the first point.
 * @param {number} x2 - The x-coordinate of the second point.
 * @param {number} y2 - The y-coordinate of the second point.
 * @returns {number} The Manhattan distance between the two points.
 */
function manhattanDistance(x1, y1, x2, y2) {
  return Math.abs(x2 - x1) + Math.abs(y2 - y1);
}

/**
 * Calculates the dot product of two vectors.
 * @param {number[]} a - The first vector.
 * @param {number[]} b - The second vector.
 * @returns {number} The dot product of the two vectors.
 */
function dotProduct(a, b) {
  return a.reduce((acc, val, idx) => acc + val * b[idx], 0);
}

/**
 * Calculates the cross product of two vectors.
 * @param {number[]} a - The first vector.
 * @param {number[]} b - The second vector.
 * @returns {number[]} The cross product of the two vectors.
 */
function crossProduct(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * Calculates the magnitude of a vector.
 * @param {number[]} a - The vector.
 * @returns {number} The magnitude of the vector.
 */
function magnitude(a) {
  return Math.sqrt(a.reduce((acc, val) => acc + Math.pow(val, 2), 0));
}

/**
 * Calculates the unit vector of a vector.
 * @param {number[]} a - The vector.
 * @returns {number[]} The unit vector of the vector.
 */
function unitVector(a) {
  const magnitudeA = magnitude(a);
  return a.map(val => val / magnitudeA);
}

/**
 * Calculates the angle between two vectors.
 * @param {number[]} a - The first vector.
 * @param {number[]} b - The second vector.
 * @returns {number} The angle between the two vectors in radians .
 */
function angleBetweenVectors(a, b) {
  return Math.acos(dotProduct(a, b) / (magnitude(a) * magnitude(b)));
}

/**
 * Calculates the transaction fee for a given amount.
 * @param {number} amount - The transaction amount.
 * @returns {number} The transaction fee.
 */
export function calculateTransactionFee(amount) {
  return amount * INTERGALACTIC_TRANSACTION_FEE;
}

/**
 * Calculates the exchange rate for a given amount.
 * @param {number} amount - The transaction amount.
 * @param {number} exchangeRate - The exchange rate.
 * @returns {number} The exchanged amount.
 */
export function calculateExchangeRate(amount, exchangeRate) {
  return amount * exchangeRate;
                 }
