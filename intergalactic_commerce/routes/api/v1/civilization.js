import express from 'express';
import { Civilization } from '../database/models';

const router = express.Router();

router.get('/', async (req, res) => {
  const civilizations = await Civilization.find().exec();
  res.json(civilizations);
});

router.post('/', async (req, res) => {
  const { name, type, planetarySystem } = req.body;
  const civilization = new Civilization({ name, type, planetarySystem });
  await civilization.save();
  res.json(civilization);
});

export default router;
