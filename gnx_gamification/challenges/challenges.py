# Import the necessary libraries
import web3
from web3 import Web3

# Define the challenges functions
def create_challenge(challenge_name, challenge_description, reward_amount):
  challenge_contract = web3.eth.contract(abi='ChallengeContract', bytecode='ChallengeContract')
  challenge_contract = challenge_contract.constructor().transact()
  challenge_tx = challenge_contract_instance.functions.createChallenge(challenge_name, challenge_description, reward_amount).transact()
  return challenge_tx

def complete_challenge(challenge_id, user_address):
  challenge_contract_instance = web3.eth.contract(abi='ChallengeContract', bytecode='ChallengeContract')
  challenge_tx = challenge_contract_instance.functions.completeChallenge(challenge_id, user_address).transact()
  return challenge_tx

# Export the challenges functions
def challenges():
  return {'create_challenge': create_challenge, 'complete_challenge': complete_challenge}
