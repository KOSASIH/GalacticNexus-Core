// aiModel.js

const mongoose = require('mongoose');
const tf = require('@tensorflow/tfjs');

const aiSchema = new mongoose.Schema({
  modelType: {
    type: String,
    required: true,
  },
  modelData: {
    type: Buffer,
    required: true,
  },
  modelVersion: {
    type: String,
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

aiSchema.methods.predict = function(input) {
  const model = tf.loadModel(this.modelData);
  const output = model.predict(input);
  return output;
};

const AI = mongoose.model('AI', aiSchema);

module.exports = AI;
