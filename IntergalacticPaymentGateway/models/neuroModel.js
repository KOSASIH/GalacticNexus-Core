// neuroModel.js

const mongoose = require('mongoose');
const brain = require('brain.js');

const neuroSchema = new mongoose.Schema({
  neuralNetwork: {
    type: String,
    required: true,
  },
  layers: {
    type: Array,
    required: true,
  },
  neurons: {
    type: Number,
    required: true,
  },
  trainingData: {
    type: Buffer,
    required: true,
  },
  trainingLabels: {
    type: Buffer,
    required: true,
  },
});

neuroSchema.methods.train = function() {
  const net = new brain.NeuralNetwork();
  net.fromJSON(this.neuralNetwork);
  net.train(this.trainingData, this.trainingLabels);
  return net;
};

const Neuro = mongoose.model('Neuro', neuroSchema);

module.exports = Neuro;
