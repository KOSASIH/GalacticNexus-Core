// blockchainModel.js

const mongoose = require('mongoose');
const Web3 = require('web3');

const blockchainSchema = new mongoose.Schema({
  blockNumber: {
    type: Number,
    required: true,
  },
  blockHash: {
    type: String,
    required: true,
  },
  transactionHash: {
    type: String,
    required: true,
  },
  fromAddress: {
    type: String,
    required: true,
  },
  toAddress: {
    type: String,
    required: true,
  },
  value: {
    type: Number,
    required: true,
  },
  gasUsed: {
    type: Number,
    required: true,
  },
  gasPrice: {
    type: Number,
    required: true,
  },
  nonce: {
    type: Number,
    required: true,
  },
});

blockchainSchema.methods.getTransactionReceipt = function() {
  const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io'));
  return web3.eth.getTransactionReceipt(this.transactionHash);
};

const Blockchain = mongoose.model('Blockchain', blockchainSchema);

module.exports = Blockchain;
