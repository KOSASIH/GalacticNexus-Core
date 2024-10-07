// Import the necessary libraries
const Web3 = require('web3');
const ethers = require('ethers');

// Define the voting mechanisms functions
async function createVotingMechanism(tokenAddress, governanceAddress) {
  const votingMechanism = await ethers.getContractFactory('VotingMechanism');
  const votingMechanismInstance = await votingMechanism.deploy(tokenAddress, governanceAddress);
  return votingMechanismInstance;
}

async function castVote(votingMechanismInstance, proposalId, vote) {
  const voteResult = await votingMechanismInstance.castVote(proposalId, vote);
  return voteResult;
}

// Export the voting mechanisms functions
module.exports = { createVotingMechanism, castVote };
