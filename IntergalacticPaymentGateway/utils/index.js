import express from 'express';
import { transactionService } from './services/TransactionService';
import { currencyService } from './services/CurrencyService';
import { paymentMethodService } from './services/PaymentMethodService';
import { errors } from './utils/errors';
import { helpers } from './utils/helpers';

const app = express();

app.use(express.json());

app.post('/transactions', async (req, res) => {
  try {
    const transactionData = req.body;
    const transaction = await transactionService.createTransaction(transactionData);
    res.json(transaction);
  } catch (error) {
    if (error instanceof errors.GalacticNexusError) {
      res.status(error.statusCode).json({ error: error.message });
    } else {
      res.status(500).json({ error: 'Internal Server Error' });
    }
  }
});

app.get('/currencies', async (req, res) => {
  try {
    const currencies = await currencyService.getCurrencies();
    res.json(currencies);
  } catch (error) {
    if (error instanceof errors.GalacticNexusError) {
      res.status(error.statusCode).json({ error: error.message });
    } else {
      res.status(500).json({ error: 'Internal Server Error' });
    }
  }
});

app.get('/payment-methods', async (req, res) => {
  try {
    const paymentMethods = await paymentMethodService.getPaymentMethods();
    res.json(paymentMethods);
  } catch (error) {
    if (error instanceof errors.GalacticNexusError) {
      res.status(error.statusCode).json({ error: error.message });
    } else {
      res.status(500).json({ error: 'Internal Server Error' });
    }
  }
});

app.listen(3000, () => {
  console.log('Galactic Nexus API listening on port 3000');
});
