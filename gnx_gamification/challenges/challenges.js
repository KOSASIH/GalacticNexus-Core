// Import the necessary libraries
const Web3 = require('web3');
const ethers = require('ethers');

// Define the challenges functions
async function createChallenge(challengeName, challengeDescription, rewardAmount) {
  const challengeContract = await ethers.getContractFactory('ChallengeContract');
  const challengeContractInstance = await challengeContract.deploy();
  const challengeTx = await challengeContractInstance.createChallenge(challengeName, challengeDescription, rewardAmount);
  return challengeTx;
}

async function completeChallenge(challengeId, userAddress) {
  const challengeContractInstance = await ethers.getContractFactory('ChallengeContract');
  const challengeTx = await challengeContractInstance.completeChallenge(challengeId, userAddress);
  return challengeTx;
}

// Export the challenges functions
module.exports = { createChallenge, completeChallenge };
