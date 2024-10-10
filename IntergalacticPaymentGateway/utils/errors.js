class GalacticNexusError extends Error {
  constructor(code, message, statusCode) {
    super(message);
    this.code = code;
    this.statusCode = statusCode;
  }
}

class InvalidRequestError extends GalacticNexusError {
  constructor(message) {
    super('invalid_request', message, 400);
  }
}

class InvalidPaymentMethodError extends GalacticNexusError {
  constructor(message) {
    super('invalid_payment_method', message, 400);
  }
}

class InvalidCurrencyError extends GalacticNexusError {
  constructor(message) {
    super('invalid_currency', message, 400);
  }
}

class TransactionFailedError extends GalacticNexusError {
  constructor(message) {
    super('transaction_failed', message, 500);
  }
}

class UnauthorizedError extends GalacticNexusError {
  constructor(message) {
    super('unauthorized', message, 401);
  }
}

class ForbiddenError extends GalacticNexusError {
  constructor(message) {
    super('forbidden', message, 403);
  }
}

class NotFoundError extends GalacticNexusError {
  constructor(message) {
    super('not_found', message, 404);
  }
}

class InternalServerError extends GalacticNexusError {
  constructor(message) {
    super('internal_server_error', message, 500);
  }
}

export {
  GalacticNexusError,
  InvalidRequestError,
  InvalidPaymentMethodError,
  InvalidCurrencyError,
  TransactionFailedError,
  UnauthorizedError,
  ForbiddenError,
  NotFoundError,
  InternalServerError,
};
