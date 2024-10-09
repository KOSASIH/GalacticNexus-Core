import requests

class BlockchainInteroperability:
    def __init__(self, config):
        self.config = config

    def send_data_to_blockchain(self, data, blockchain_platform):
        api_url = self.config[blockchain_platform]["api_url"]
        api_key = self.config[blockchain_platform]["api_key"]
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(f"{api_url}/api/data", json=data, headers=headers)
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print("Error sending data")
