// aiController.js const express = require('express');
const router = express.Router();
const AI = require('../models/aiModel');
const tf = require('@tensorflow/tfjs');

router.post('/create', async (req, res) => {
  try {
    const ai = new AI(req.body);
    await ai.save();
    res.status(201).send(ai);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get', async (req, res) => {
  try {
    const ais = await AI.find();
    res.status(200).send(ais);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get/:id', async (req, res) => {
  try {
    const ai = await AI.findById(req.params.id);
    if (!ai) {
      res.status(404).send({ message: 'AI not found' });
    } else {
      res.status(200).send(ai);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.put('/update/:id', async (req, res) => {
  try {
    const ai = await AI.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
    });
    if (!ai) {
      res.status(404).send({ message: 'AI not found' });
    } else {
      res.status(200).send(ai);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.delete('/delete/:id', async (req, res) => {
  try {
    const ai = await AI.findByIdAndRemove(req.params.id);
    if (!ai) {
      res.status(404).send({ message: 'AI not found' });
    } else {
      res.status(200).send({ message: 'AI deleted successfully' });
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.post('/predict/:id', async (req, res) => {
  try {
    const ai = await AI.findById(req.params.id);
    if (!ai) {
      res.status(404).send({ message: 'AI not found' });
    } else {
      const model = tf.loadModel(ai.modelData);
      const output = model.predict(req.body.input);
      res.status(200).send(output);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

module.exports = router;
