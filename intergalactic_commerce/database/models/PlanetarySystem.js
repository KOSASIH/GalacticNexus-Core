import mongoose from 'mongoose';

const planetarySystemSchema = new mongoose.Schema({
  name: { type: String, required: true },
  category: { type: String, enum: PLANETARY_SYSTEM_CATEGORIES },
  location: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

export default mongoose.model('PlanetarySystem', planetarySystemSchema);
