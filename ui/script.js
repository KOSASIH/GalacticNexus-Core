const getBlockButton = document.getElementById('get-block-button');
const blockNumberSpan = document.getElementById('block-number');
const blockHashSpan = document.getElementById('block-hash');

const startMinerButton = document.getElementById('start-miner-button');
const stopMinerButton = document.getElementById('stop-miner-button');
const minerThreadsSpan = document.getElementById('miner-threads');
const minerIntervalSpan = document.getElementById('miner-interval');

const deployContractButton = document.getElementById('deploy-contract-button');
const contractList = document.getElementById('contract-list');

// Get block function
async function getBlock() {
  const response = await fetch('/api/block');
  const data = await response.json();
  blockNumberSpan.textContent = data.blockNumber;
  blockHashSpan.textContent = data.blockHash;
}

// Start miner function
async function startMiner() {
  const response = await fetch('/api/miner/start');
  const data = await response.json();
  minerThreadsSpan.textContent = data .minerThreads;
  minerIntervalSpan.textContent = data.minerInterval;
}

// Stop miner function
async function stopMiner() {
  const response = await fetch('/api/miner/stop');
  const data = await response.json();
  minerThreadsSpan.textContent = data.minerThreads;
  minerIntervalSpan.textContent = data.minerInterval;
}

// Deploy contract function
async function deployContract() {
  const response = await fetch('/api/contract/deploy');
  const data = await response.json();
  const contractListItem = document.createElement('li');
  contractListItem.textContent = data.contractName;
  contractList.appendChild(contractListItem);
}

// Add event listeners
getBlockButton.addEventListener('click', getBlock);
startMinerButton.addEventListener('click', startMiner);
stopMinerButton.addEventListener('click', stopMiner);
deployContractButton.addEventListener('click', deployContract);
