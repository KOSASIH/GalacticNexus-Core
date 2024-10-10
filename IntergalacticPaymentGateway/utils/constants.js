export const APP_NAME = 'GalacticNexus';
export const APP_VERSION = '1.0.0';
export const APP_ENV = process.env.NODE_ENV || 'development';

export const GALACTIC_UNION_API_KEY = 'YOUR_GALACTIC_UNION_API_KEY';
export const GALACTIC_UNION_API_SECRET = 'YOUR_GALACTIC_UNION_API_SECRET';
export const INTERSTELLAR_BANK_API_KEY = 'YOUR_INTERSTELLAR_BANK_API_KEY';
export const INTERSTELLAR_BANK_API_SECRET = 'YOUR_INTERSTELLAR_BANK_API_SECRET';

export const CURRENCY_CODES = {
  GALACTIC_CREDITS: 'GC',
  INTERSTELLAR_DOLLARS: 'ISD',
  EARTH_DOLLARS: 'USD',
};

export const TRANSACTION_STATUSES = {
  PENDING: 'pending',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  FAILED: 'failed',
};

export const PAYMENT_METHOD_TYPES = {
  GALACTIC_UNION: 'galactic_union',
  INTERSTELLAR_BANK: 'interstellar_bank',
  CREDIT_CARD: 'credit_card',
};

export const ERROR_CODES = {
  INVALID_REQUEST: 'invalid_request',
  INVALID_PAYMENT_METHOD: 'invalid_payment_method',
  INVALID_CURRENCY: 'invalid_currency',
  TRANSACTION_FAILED: 'transaction_failed',
};

export const HTTP_STATUS_CODES = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  INTERNAL_SERVER_ERROR: 500,
};
