// paymentModel.js

const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const paymentSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
  },
  paymentMethod: {
    type: String,
    required: true,
  },
  paymentDate: {
    type: Date,
    default: Date.now,
  },
  amount: {
    type: Number,
    required: true,
  },
  currency: {
    type: String,
    required: true,
  },
  transactionId: {
    type: String,
    required: true,
  },
  cardNumber: {
    type: String,
    required: true,
  },
  cardExpiry: {
    type: String,
    required: true,
  },
  cardCvv: {
    type: String,
    required: true,
  },
});

paymentSchema.methods.generateToken = function() {
  const token = jwt.sign({ _id: this._id }, process.env.SECRET_KEY, {
    expiresIn: '1h',
  });
  return token;
};

paymentSchema.methods.comparePassword = function(password) {
  return bcrypt.compareSync(password, this.password);
};

const Payment = mongoose.model('Payment', paymentSchema);

module.exports = Payment;
