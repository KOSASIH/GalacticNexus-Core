// Import the necessary libraries
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

// Connect to MongoDB
mongoose.connect('mongodb://localhost/gnx_db', { useNewUrlParser: true, useUnifiedTopology: true });

// Define the gnx_db schema
const gnxDbSchema = new mongoose.Schema({
  username: String,
  password: String,
  email: String
});

// Define the gnx_db model
const GnxDb = mongoose.model('GnxDb', gnxDbSchema);

// Define the gnx_db functions
async function createUser(username, password, email) {
  const hashedPassword = await bcrypt.hash(password, 10);
  const user = new GnxDb({ username, password: hashedPassword, email });
  await user.save();
  return user;
}

async function getUser(username) {
  const user = await GnxDb.findOne({ username });
  return user;
}

async function updateUser(username, password, email) {
  const user = await GnxDb.findOne({ username });
  if (user) {
    user.password = await bcrypt.hash(password, 10);
    user.email = email;
    await user.save();
  }
  return user;
}

async function deleteUser(username) {
  await GnxDb.deleteOne({ username });
}

// Export the gnx_db functions
module.exports = { createUser, getUser, updateUser, deleteUser };
