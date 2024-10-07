// Import the necessary libraries
const Web3 = require('web3');
const ethers = require('ethers');

// Define the leaderboard functions
async function getLeaderboard() {
  const leaderboardContract = await ethers.getContractFactory('LeaderboardContract');
  const leaderboardContractInstance = await leaderboardContract.deploy();
  const leaderboard = await leaderboardContractInstance.getLeaderboard();
  return leaderboard;
}

async function updateLeaderboard(userAddress, score) {
  const leaderboardContractInstance = await ethers.getContractFactory('LeaderboardContract');
  const leaderboardTx = await leaderboardContractInstance.updateLeaderboard(userAddress, score);
  return leaderboardTx;
}

// Export the leaderboard functions
module.exports = { getLeaderboard, updateLeaderboard };
