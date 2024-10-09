import express from 'express';
import { IntergalacticTransaction } from '../database/models';

const router = express.Router();

router.get('/', async (req, res) => {
  const intergalacticTransactions = await IntergalacticTransaction.find().exec();
  res.json(intergalacticTransactions);
});

router.post('/', async (req, res) => {
  const { sender, recipient, amount, currency } = req.body;
  const intergalacticTransaction = new Int ergalacticTransaction({ sender, recipient, amount, currency });
  await intergalacticTransaction.save();
  res.json(intergalacticTransaction);
});

export default router;
