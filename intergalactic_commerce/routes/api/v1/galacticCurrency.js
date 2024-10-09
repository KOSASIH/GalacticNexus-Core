import express from 'express';
import { GalacticCurrency } from '../database/models';

const router = express.Router();

router.get('/', async (req, res) => {
  const galacticCurrencies = await GalacticCurrency.find().exec();
  res.json(galacticCurrencies);
});

router.post('/', async (req, res) => {
  const { symbol, name, exchangeRate, decimals } = req.body;
  const galacticCurrency = new GalacticCurrency({ symbol, name, exchangeRate, decimals });
  await galacticCurrency.save();
  res.json(galacticCurrency);
});

export default router;
