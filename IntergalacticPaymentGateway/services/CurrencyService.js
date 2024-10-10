import { Currency } from '../models/Currency';
import { GalacticUnionAPI } from '../integrations/GalacticUnionAPI';
import { InterstellarBankAPI } from '../integrations/InterstellarBankAPI';
import { logger } from '../utils/logger';

class CurrencyService {
  async getCurrencies() {
    try {
      const currencies = await Currency.find();
      return currencies;
    } catch (error) {
      logger.error(`Error getting currencies: ${error.message}`);
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

  async createCurrency(currencyData) {
    try {
      const { code, name, symbol, exchangeRate } = currencyData;
      const currency = new Currency({
        code,
        name,
        symbol,
        exchangeRate,
      });
      await currency.save();
      return currency;
    } catch (error) {
      logger.error(`Error creating currency: ${error.message}`);
      throw error;
    }
  }

  async updateCurrency(currencyCode, currencyData) {
    try {
      const currency = await Currency.findOne({ code: currencyCode });
      if (!currency) {
        throw new Error('Currency not found');
      }
      Object.assign(currency, currencyData);
      await currency.save();
      return currency;
    } catch (error) {
      logger.error(`Error updating currency: ${error.message}`);
      throw error;
    }
  }

  async getExchangeRate(fromCurrencyCode, toCurrencyCode) {
    try {
      const fromCurrency = await this.getCurrency(fromCurrencyCode);
      const toCurrency = await this.getCurrency(toCurrencyCode);
      if (!fromCurrency || !toCurrency) {
        throw new Error('Currency not found');
      }
      const exchangeRate = fromCurrency.exchangeRate / toCurrency.exchangeRate;
      return exchangeRate;
    } catch (error) {
      logger.error(`Error getting exchange rate: ${error.message}`);
      throw error;
    }
  }

  async convertAmount(amount, fromCurrencyCode, toCurrencyCode) {
    try {
      const exchangeRate = await this.getExchangeRate(fromCurrencyCode, toCurrencyCode);
      const convertedAmount = amount * exchangeRate;
      return convertedAmount;
    } catch (error) {
      logger.error(`Error converting amount: ${error.message}`);
      throw error;
    }
  }
}

export const currencyService = new CurrencyService();
