#!/bin/bash

# Set the testnet configuration file
CONFIG_FILE="testnet.config.json"

# Set the testnet data directory
DATA_DIR="/path/to/data/dir"

# Set the testnet network ID
NETWORK_ID=12345

# Set the testnet genesis block file
GENESIS_FILE="genesis.json"

# Set the testnet node executable
NODE_EXECUTABLE="node"

# Set the testnet node configuration file
NODE_CONFIG_FILE="node.config.json"

# Function to deploy the testnet
deploy_testnet() {
  # Create the testnet data directory
  mkdir -p $DATA_DIR

  # Initialize the testnet genesis block
  echo "Initializing testnet genesis block..."
  $NODE_EXECUTABLE init --genesis $GENESIS_FILE --datadir $DATA_DIR

  # Start the testnet node
  echo "Starting testnet node..."
  $NODE_EXECUTABLE --datadir $DATA_DIR --networkid $NETWORK_ID --nodiscover --maxpeers 10 --rpc --rpccorsdomain "*" --rpcvhosts "*" --rpcaddr "0.0.0.0" --rpcport 8545 --ws --wsaddr "0.0.0.0" --wsport 8546

  # Wait for the testnet node to start
  echo "Waiting for testnet node to start..."
  sleep 10

  # Deploy the testnet contracts
  echo "Deploying testnet contracts..."
  $NODE_EXECUTABLE deploy --datadir $DATA_DIR --networkid $NETWORK_ID --contracts "contracts/*.sol"

  # Wait for the testnet contracts to deploy
  echo "Waiting for testnet contracts to deploy..."
  sleep 10

  # Start the testnet miner
  echo "Starting testnet miner..."
  $NODE_EXECUTABLE miner --datadir $DATA_DIR --networkid $NETWORK_ID --minerthreads 1 --minerinterval 1000000000

  # Wait for the testnet miner to start
  echo "Waiting for testnet miner to start..."
  sleep 10
}

# Main function
main() {
  # Check if the testnet configuration file exists
  if [ ! -f $CONFIG_FILE ]; then
    echo "Error: Testnet configuration file not found!"
    exit 1
  fi

  # Load the testnet configuration
  echo "Loading testnet configuration..."
  . $CONFIG_FILE

  # Deploy the testnet
  deploy_testnet
}

# Run the main function
main
