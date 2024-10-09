const express = require('express');
const app = express();
const blockchain = require('./blockchain');

app.use(express.json());

// Create a new blockchain node
const node = new blockchain.Node();

// Function to handle incoming transactions
app.post('/transactions', (req, res) => {
    const transaction = req.body;
    node.addTransaction(transaction);
    res.send(`Transaction added to the blockchain`);
});

// Function to get the blockchain
app.get('/blockchain', (req, res) => {
    res.send(node.getBlockchain());
});

// Function to mine a new block
app.post('/mine', (req, res) => {
    node.mineBlock();
    res.send(`New block mined`);
});

// Start the server
app.listen(3000, () => {
    console.log('Blockchain node started on port 3000');
});
