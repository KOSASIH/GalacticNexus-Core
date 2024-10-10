// paymentService.js

const Payment = require('../models/paymentModel');

class PaymentService {
  async createPayment(paymentData) {
    try {
      const payment = new Payment(paymentData);
      await payment.save();
      return payment;
    } catch (err) {
      throw err;
    }
  }

  async getPayments() {
    try {
      const payments = await Payment.find();
      return payments;
    } catch (err) {
      throw err;
    }
  }

  async getPaymentById(paymentId) {
    try {
      const payment = await Payment.findById(paymentId);
      if (!payment) {
        throw new Error('Payment not found');
      }
      return payment;
    } catch (err) {
      throw err;
    }
  }

  async updatePayment(paymentId, paymentData) {
    try {
      const payment = await Payment.findByIdAndUpdate(paymentId, paymentData, {
        new: true,
      });
      if (!payment) {
        throw new Error('Payment not found');
      }
      return payment;
    } catch (err) {
      throw err;
    }
  }

  async deletePayment(paymentId) {
    try {
      const payment = await Payment.findByIdAndRemove(paymentId);
      if (!payment) {
        throw new Error('Payment not found');
      }
      return payment;
    } catch (err) {
      throw err;
    }
  }
}

module.exports = PaymentService;
