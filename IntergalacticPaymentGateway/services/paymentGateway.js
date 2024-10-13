// Import required modules
const { v4: uuidv4 } = require('uuid');
const { createHash } = require('crypto');
const mongoose = require('mongoose');

// Define the PaymentGateway class
class PaymentGateway {
  constructor() {
    this.payments = mongoose.model('Payment', {
      paymentId: String,
      amount: Number,
      recipient: String,
      paymentHash: String,
    });
  }

  // Create a new payment
  async createPayment(paymentData) {
    try {
      const payment = new this.payments(paymentData);
      await payment.save();
    } catch (error) {
      console.error(error);
      throw error;
    }
  }

  // Get a payment by ID
  async getPayment(paymentId) {
    try {
      const payment = await this.payments.findOne({ paymentId });
      return payment;
    } catch (error) {
      console.error(error);
      throw error;
    }
  }

  // Cancel a payment
  async cancelPayment(paymentData) {
    try {
      await this.payments.deleteOne({ paymentId: paymentData.paymentId });
    } catch (error) {
      console.error(error);
      throw error;
    }
  }
}

// Export the PaymentGateway class
module.exports = PaymentGateway;
