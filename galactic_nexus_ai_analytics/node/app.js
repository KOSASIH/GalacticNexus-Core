const express = require('express');
const Web3 = require('web3');
const aiModel = require('./models/ai_model');

const app = express();

app.use(express.json());

const web3 = new Web3(new Web3.providers.HttpProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'));

const aiAnalyticsContract = new web3.eth.Contract(AIAnalytics.abi, '0x...ContractAddress...');

app.post('/train-ai-model', async (req, res) => {
    const assetData = req.body.assetData;
    const marketData = req.body.marketData;

    try {
        await aiModel.train_ai_model(assetData, marketData);
        res.status(201).send({ message: 'AI model trained successfully' });
    } catch (error) {
        res.status(500).send({ message: 'Error training AI model' });
    }
});

app.get('/get-ai-insights', async (req, res) => {
    try {
        const insights = await aiAnalyticsContract.methods.getAIInsights().call();
        res.status(200).send({ insights });
    } catch (error) {
        res.status(500).send({ message: 'Error retrieving AI insights' });
    }
});

app.listen(3000, () => {
    console.log('Server listening on port 3000');
});
