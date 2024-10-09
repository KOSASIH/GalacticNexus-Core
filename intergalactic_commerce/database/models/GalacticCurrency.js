import mongoose from 'mongoose';

const galacticCurrencySchema = new mongoose.Schema({
  symbol: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  exchangeRate: { type: Number, required: true },
  decimals: { type: Number, required: true },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

export default mongoose.model('GalacticCurrency', galacticCurrencySchema);
