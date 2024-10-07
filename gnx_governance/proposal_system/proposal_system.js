// Import the necessary libraries
const Web3 = require('web3');
const ethers = require('ethers');

// Define the proposal system functions
async function createProposal(tokenAddress, governanceAddress, proposal) {
  const proposalSystem = await ethers.getContractFactory('ProposalSystem');
  const proposalSystemInstance = await proposalSystem.deploy(tokenAddress, governanceAddress);
  const proposalId = await proposalSystemInstance.createProposal(proposal);
  return proposalId;
}

async function getProposal(proposalSystemInstance, proposalId) {
  const proposal = await proposalSystemInstance.getProposal(proposalId);
  return proposal;
}

// Export the proposal system functions
module.exports = { createProposal, getProposal };
