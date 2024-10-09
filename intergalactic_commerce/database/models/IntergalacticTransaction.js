import mongoose from 'mongoose';

const intergalacticTransactionSchema = new mongoose.Schema({
  sender: { type: mongoose.Schema.Types.ObjectId, ref: 'Civilization' },
  recipient: { type: mongoose.Schema.Types.ObjectId, ref: 'Civilization' },
  amount: { type: Number, required: true },
  currency: { type: mongoose.Schema.Types.ObjectId, ref: 'GalacticCurrency' },
  transactionFee: { type: Number, required: true },
  status: { type: String, enum: ['pending', 'success', 'failed'] },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

export default mongoose.model('IntergalacticTransaction', intergalacticTransactionSchema);
