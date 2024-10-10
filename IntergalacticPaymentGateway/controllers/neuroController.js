// neuroController.js

const express = require('express');
const router = express.Router();
const Neuro = require('../models/neuroModel');
const brain = require('brain.js');

router.post('/create', async (req, res) => {
  try {
    const neuro = new Neuro(req.body);
    await neuro.save();
    res.status(201).send(neuro);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get', async (req, res) => {
  try {
    const neuros = await Neuro.find();
    res.status(200).send(neuros);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get/:id', async (req, res) => {
  try {
    const neuro = await Neuro.findById(req.params.id);
    if (!neuro) {
      res.status(404).send({ message: 'Neuro not found' });
    } else {
      res.status(200).send(neuro);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.put('/update/:id', async (req, res) => {
  try {
    const neuro = await Neuro.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
    });
    if (!neuro) {
      res.status(404).send({ message: 'Neuro not found' });
    } else {
      res.status(200).send(neuro);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.delete('/delete/:id', async (req, res) => {
  try {
    const neuro = await Neuro.findByIdAndRemove(req.params.id);
    if (!neuro) {
      res.status(404).send({ message: 'Neuro not found' });
    } else {
      res.status(200).send({ message: 'Neuro deleted successfully' });
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.post('/train/:id', async (req, res) => {
  try {
    const neuro = await Neuro.findById(req.params.id);
    if (!neuro) {
      res.status(404).send({ message: 'Neuro not found' });
    } else {
      const net = new brain.NeuralNetwork();
      net.fromJSON(neuro.neuralNetwork);
      net.train(neuro.trainingData, neuro.trainingLabels);
      res.status(200).send(net);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

module.exports = router;
