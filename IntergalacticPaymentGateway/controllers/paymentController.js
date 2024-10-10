// paymentController.js

const express = require('express');
const router = express.Router();
const Payment = require('../models/paymentModel');

router.post('/create', async (req, res) => {
  try {
    const payment = new Payment(req.body);
    await payment.save();
    res.status(201).send(payment);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get', async (req, res) => {
  try {
    const payments = await Payment.find();
    res.status(200).send(payments);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get/:id', async (req, res) => {
  try {
    const payment = await Payment.findById(req.params.id);
    if (!payment) {
      res.status(404).send({ message: 'Payment not found' });
    } else {
      res.status(200).send(payment);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.put('/update/:id', async (req, res) => {
  try {
    const payment = await Payment.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
    });
    if (!payment) {
      res.status(404).send({ message: 'Payment not found' });
    } else {
      res.status(200).send(payment);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.delete('/delete/:id', async (req, res) => {
  try {
    const payment = await Payment.findByIdAndRemove(req.params.id);
    if (!payment) {
      res.status(404).send({ message: 'Payment not found' });
    } else {
      res.status(200).send({ message: 'Payment deleted successfully' });
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

module.exports = router;
