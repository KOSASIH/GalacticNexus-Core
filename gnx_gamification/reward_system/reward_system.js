// Import the necessary libraries
const Web3 = require('web3');
const ethers = require('ethers');

// Define the reward system functions
async function rewardUser(userAddress, rewardAmount) {
  const rewardContract = await ethers.getContractFactory('RewardContract');
  const rewardContractInstance = await rewardContract.deploy();
  const rewardTx = await rewardContractInstance.rewardUser(userAddress, rewardAmount);
  return rewardTx;
}

async function getRewardBalance(userAddress) {
  const rewardContractInstance = await ethers.getContractFactory('RewardContract');
  const rewardBalance = await rewardContractInstance.getRewardBalance(userAddress);
  return rewardBalance;
}

// Export the reward system functions
module.exports = { rewardUser, getRewardBalance };
