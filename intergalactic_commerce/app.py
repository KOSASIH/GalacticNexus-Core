import express from 'express';
import mongoose from 'mongoose';
import { GalacticCurrency } from './database/models';
import { IntergalacticTransaction } from './database/models';
import { PlanetarySystem } from './database/models';
import { Civilization } from './database/models';
import { encrypt, decrypt } from './utils/crypto';
import { calculateTransactionFee, calculateExchangeRate } from './utils/math';
import { generateUUID } from './utils/string';

const app = express();

app.use(express.json());

app.use('/api/v1/galacticCurrency', require('./routes/api/v1/galacticCurrency'));
app.use('/api/v1/intergalacticTransaction', require('./routes/api/v1/intergalacticTransaction'));
app.use('/api/v1/planetarySystem', require('./routes/api/v1/planetarySystem'));
app.use('/api/v1/civilization', require('./routes/api/v1/civilization'));

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});

mongoose.connect('mongodb://localhost/galacticnexus', { useNewUrlParser: true, useUnifiedTopology: true });
