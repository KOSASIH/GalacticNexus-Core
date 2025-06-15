# ai/smart_contracts/auto_contract.py
from openai import OpenAI

def generate_contract(description):
    # Requires OpenAI API Key set as environment variable
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a Solidity smart contract developer."},
                  {"role": "user", "content": f"Write a smart contract for: {description}"}]
    )
    return response['choices'][0]['message']['content']
