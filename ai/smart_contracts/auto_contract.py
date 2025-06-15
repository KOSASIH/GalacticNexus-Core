# ai/smart_contracts/auto_contract.py

import os
import openai

class AutoContractAI:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        assert self.openai_api_key, "Set your OPENAI_API_KEY environment variable."
        openai.api_key = self.openai_api_key

    def _ask_gpt(self, system_prompt, user_prompt, temperature=0.2):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", generate_contract(self, description):
        prompt = f"Write a production-ready, secure, and well-commented Solidity smart contract for the following purpose:\n\n{description}\n\nUse latest Solidity best practices. Include NatSpec comments. Ensure the contract is compatible with Solidity 0.8+."
        return self._ask_gpt(
            "You are an expert Solidity contract developer and auditor.",
            prompt
        )

    def audit_contract(self, solidity_code):
        prompt = f"Audit and report any vulnerabilities, inefficiencies, or improvements for this Solidity contract:\n\n{solidity_code}"
        return self._ask_gpt(
            "You are a world-class smart contract auditor.",
            prompt
        )

    def summarize_contract(self, solidity_code):
        prompt = f"Summarize the purpose and main features of this Solidity contract in a concise paragraph:\n\n{solidity_code}"
        return self._ask_gpt(
            "You are an expert Solidity contract analyst.",
            prompt
        )

    def verify_contract(self, description, solidity_code):
        prompt = f"Does this Solidity contract accurately and safely implement the following requirements? Reply with YES or NO and explain:\n\nRequirements:\n{description}\n\nContract:\n{solidity_code}"
        return self._ask_gpt(
            "You are an expert Solidity contract reviewer.",
            prompt
        )

# Example usage:
# ai = AutoContractAI()
# solidity_code = ai.generate_contract("A token with burn, mint, and pause features, only owner can mint, anyone can burn.")
# print(solidity_code)
# print(ai.audit_contract(solidity_code))
