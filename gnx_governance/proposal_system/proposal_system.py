# Import the necessary libraries
import web3
from web3 import Web # Define the proposal system functions
def create_proposal(token_address, governance_address, proposal):
  proposal_system = web3.eth.contract(abi='ProposalSystem', bytecode='ProposalSystem')
  proposal_system_instance = proposal_system.constructor(token_address, governance_address).transact()
  proposal_id = proposal_system_instance.functions.createProposal(proposal).transact()
  return proposal_id

def get_proposal(proposal_system_instance, proposal_id):
  proposal = proposal_system_instance.functions.getProposal(proposal_id).call()
  return proposal

# Export the proposal system functions
def proposal_system():
  return {'create_proposal': create_proposal, 'get_proposal': get_proposal}
