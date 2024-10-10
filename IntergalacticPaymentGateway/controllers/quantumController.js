// quantumController.js

const express = require('express');
const router = express.Router();
const Quantum = require('../models/quantumModel');
const Qiskit = require('qiskit');

router.post('/create', async (req, res) => {
  try {
    const quantum = new Quantum(req.body);
    await quantum.save();
    res.status(201).send(quantum);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get', async (req, res) => {
  try {
    const quantums = await Quantum.find();
    res.status(200).send(quantums);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get/:id', async (req, res) => {
  try {
    const quantum = await Quantum.findById(req.params.id);
    if (!quantum) {
      res.status(404).send({ message: 'Quantum not found' });
    } else {
      res.status(200).send(quantum);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.put('/update/:id', async (req, res) => {
  try {
    const quantum = await Quantum.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
    });
    if (!quantum) {
      res.status(404).send({ message: 'Quantum not found' });
    } else {
      res.status(200).send(quantum);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.delete('/delete/:id', async (req, res) => {
  try {
    const quantum = await Quantum.findByIdAndRemove(req.params.id);
    if (!quantum) {
      res.status(404).send({ message: 'Quantum not found' });
    } else {
      res.status(200).send({ message: 'Quantum deleted successfully' });
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.post('/runCircuit/:id', async (req, res) => {
  try {
    const quantum = await Quantum.findById(req.params.id);
    if (!quantum) {
      res.status (404).send({ message: 'Quantum not found' });
    } else {
      const circuit = Qiskit.Circuit.fromQasm(quantum.circuit);
      const backend = new Qiskit.Aer.get_backend(quantum.backend);
      const job = backend.run(circuit, shots=quantum.shots);
      const result = await job.result();
      res.status(200).send(result);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

module.exports = router;
