import { Transaction } from '../models/Transaction';
import { PaymentMethod } from '../models/PaymentMethod';
import { Currency } from '../models/Currency';
import { GalacticUnionAPI } from '../integrations/GalacticUnionAPI';
import { InterstellarBankAPI } from '../integrations/InterstellarBankAPI';
import { logger } from '../utils/logger';

class TransactionService {
  async createTransaction(transactionData) {
    try {
      const { amount, paymentMethodId, currencyCode, recipientId } = transactionData;
      const paymentMethod = await this.getPaymentMethod(paymentMethodId);
      const currency = await this.getCurrency(currencyCode);
      const transaction = new Transaction({
        amount,
        paymentMethod,
        currency,
        recipientId,
        status: 'pending',
      });

      // Check if payment method is Galactic Union
      if (paymentMethod.provider === 'galactic_union') {
        const galacticUnionAPI = new GalacticUnionAPI();
        const response = await galacticUnionAPI.initiateTransaction(transaction);
        if (response.success) {
          transaction.status = 'in_progress';
          await transaction.save();
          return transaction;
        } else {
          throw new Error(response.error_message);
        }
      }

      // Check if payment method is Interstellar Bank
      if (paymentMethod.provider === 'interstellar_bank') {
        const interstellarBankAPI = new InterstellarBankAPI();
        const response = await interstellarBankAPI.initiateTransaction(transaction);
        if (response.success) {
          transaction.status = 'in_progress';
          await transaction.save();
          return transaction;
        } else {
          throw new Error(response.error_message);
        }
      }

      throw new Error('Unsupported payment method');
    } catch (error) {
      logger.error(`Error creating transaction: ${error.message}`);
      throw error;
    }
  }

  async getTransaction(transactionId) {
    try {
      const transaction = await Transaction.findById(transactionId);
      if (!transaction) {
        throw new Error('Transaction not found');
      }
      return transaction;
    } catch (error) {
      logger.error(`Error getting transaction: ${error.message}`);
      throw error;
    }
  }

  async updateTransactionStatus(transactionId, status) {
    try {
      const transaction = await Transaction.findById(transactionId);
      if (!transaction) {
        throw new Error('Transaction not found');
      }
      transaction.status = status;
      await transaction.save();
      return transaction;
    } catch (error) {
      logger.error(`Error updating transaction status: ${error.message}`);
      throw error;
    }
  }

  async getPaymentMethod(paymentMethodId) {
    try {
      const paymentMethod = await PaymentMethod.findById(paymentMethodId);
      if (!paymentMethod) {
        throw new Error('Payment method not found');
      }
      return paymentMethod;
    } catch (error) {
      logger.error(`Error getting payment method: ${error.message}`);
      throw error;
    }
  }

  async getCurrency(currencyCode) {
    try {
      const currency = await Currency.findOne({ code: currencyCode });
      if (!currency) {
        throw new Error('Currency not found');
      }
      return currency;
    } catch (error) {
      logger.error(`Error getting currency: ${error.message}`);
      throw error;
    }
  }
}

export const transactionService = new TransactionService();
