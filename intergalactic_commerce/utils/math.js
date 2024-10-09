export function calculateTransactionFee(amount) {
  return amount * INTERGALACTIC_TRANSACTION_FEE;
}

export function calculateExchangeRate(amount, exchangeRate) {
  return amount * exchangeRate;
}
