// blockchainController.js

const express = require('express');
const router = express.Router();
const Blockchain = require('../models/blockchainModel');
const Web3 = require('web3');

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io'));

router.post('/create', async (req, res) => {
  try {
    const blockchain = new Blockchain(req.body);
    await blockchain.save();
    res.status(201).send(blockchain);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get', async (req, res) => {
  try {
    const blockchains = await Blockchain.find();
    res.status(200).send(blockchains);
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/get/:id', async (req, res) => {
  try {
    const blockchain = await Blockchain.findById(req.params.id);
    if (!blockchain) {
      res.status(404).send({ message: 'Blockchain not found' });
    } else {
      res.status(200).send(blockchain);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.put('/update/:id', async (req, res) => {
  try {
    const blockchain = await Blockchain.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
    });
    if (!blockchain) {
      res.status(404).send({ message: 'Blockchain not found' });
    } else {
      res.status(200).send(blockchain);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.delete('/delete/:id', async (req, res) => {
  try {
    const blockchain = await Blockchain.findByIdAndRemove(req.params.id);
    if (!blockchain) {
      res.status(404).send({ message: 'Blockchain not found' });
    } else {
      res.status(200).send({ message: 'Blockchain deleted successfully' });
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

router.get('/getTransactionReceipt/:id', async (req, res) => {
  try {
    const blockchain = await Blockchain.findById(req.params.id);
    if (!blockchain) {
      res.status(404).send({ message: 'Blockchain not found' });
    } else {
      const transactionReceipt = await web3.eth.getTransactionReceipt(blockchain.transactionHash);
      res.status(200).send(transactionReceipt);
    }
  } catch (err) {
    res.status(400).send(err);
  }
});

module.exports = router;
