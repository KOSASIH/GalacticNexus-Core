export function formatCurrency(amount, currencyCode) {
  const currencySymbols = {
    GC: '₡',
    ISD: '$',
    USD: '$',
  };

  const symbol = currencySymbols[currencyCode];
  return `${symbol} ${amount.toFixed(2)}`;
}

export function generateUUID() {
  return crypto.randomBytes(16).toString('hex');
}

export function validatePaymentMethod(paymentMethod) {
  if (!paymentMethod || !paymentMethod.id || !paymentMethod.name) {
    throw new InvalidPaymentMethodError('Invalid payment method');
  }
}

export function validateCurrency(currency) {
  if (!currency || !currency.code || !currency.name) {
    throw new InvalidCurrencyError('Invalid currency');
  }
}
