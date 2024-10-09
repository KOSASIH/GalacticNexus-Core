import express from 'express';
import { PlanetarySystem } from '../database/models';

const router = express.Router();

router.get('/', async (req, res) => {
  const planetarySystems = await PlanetarySystem.find().exec();
  res.json(planetarySystems);
});

router.post('/', async (req, res) => {
  const { name, category, location } = req.body;
  const planetarySystem = new PlanetarySystem({ name, category, location });
  await planetarySystem.save();
  res.json(planetarySystem);
});

export default router;
