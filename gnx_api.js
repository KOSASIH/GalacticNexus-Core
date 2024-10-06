// Import the necessary libraries
const express = require('express');
const app = express();
const port = 3000;
const cors = require('cors');
const helmet = require('helmet');
constmorgan = require('morgan');
const mongoose = require('mongoose');
const passport = require('passport');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true });

// Define the gnx_api parameters
const gnxApiParams = {
  // Define the gnx_api parameters here
};

// Define the gnx_api function
function gnxApi(data) {
  // Implement the gnx_api function here
  return data;
}

// Export the gnx_api function
module.exports = gnxApi;

// Define the API endpoints
app.use(cors());
app.use(helmet());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/gnx_api', (req, res) => {
  res.send('GNX API');
});

app.post('/gnx_api', (req, res) => {
  const data = req.body;
  const result = gnxApi(data);
  res.send(result);
});

app.post('/register', (req, res) => {
  const { username, password } = req.body;
  const hashedPassword = bcrypt.hashSync(password, 10);
  const user = new User({ username, password: hashedPassword });
  user.save((err) => {
    if (err) {
      res.status(400).send({ message: 'User already exists' });
    } else {
      res.send({ message: 'User created successfully' });
    }
  });
});

app.post('/login', (req, res) => {
  const { username, password } = req.body;
  User.findOne({ username }, (err, user) => {
    if (err || !user) {
      res.status(401).send({ message: 'Invalid credentials' });
    } else {
      const isValidPassword = bcrypt.compareSync(password, user.password);
      if (!isValidPassword) {
        res.status(401).send({ message: 'Invalid credentials' });
      } else {
        const token = jwt.sign({ userId: user._id }, process.env.SECRET_KEY, { expiresIn: '1h' });
        res.send({ token });
      }
    }
  });
});

app.get('/protected', passport.authenticate('jwt', { session: false }), (req, res) => {
  res.send('Protected route');
});

// Start the server
app.listen(port, () => {
  console.log(`GNX API listening on port ${port}`);
});
