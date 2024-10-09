import mongoose from 'mongoose';

const civilizationSchema = new mongoose.Schema({
  name: { type: String, required: true },
  type: { type: String, enum: CIVILIZATION_TYPES },
  planetarySystem: { type: mongoose.Schema.Types.ObjectId, ref: 'PlanetarySystem' },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

export default mongoose.model('Civilization', civilizationSchema);
