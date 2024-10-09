package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/node"
)

// API represents the API implementation
type API struct {
	*node.Node
	*ethclient.Client
	*accounts.Manager
}

// NewAPI creates a new API instance
func NewAPI(config *Config) (*API, error) {
	// Create a new node instance
	node, err := node.NewNode(config.DataDir, config.NetworkId, config.Genesis, config.Validators, config.MaxPeers)
	if err != nil {
		return nil, err
	}

	// Create a new Ethereum client instance
	client := ethclient.NewClient(node.RPC())

	// Create a new account manager instance
	accountMgr := accounts.NewManager(node.AccountManager())

	// Create a new API instance
	api := &API{
		Node:     node,
		Client:   client,
		Manager: accountMgr,
	}

	return api, nil
}

// Start starts the API
func (a *API) Start() error {
	// Start the node
	if err := a.Node.Start(); err != nil {
		return err
	}

	// Start the Ethereum client
	if err := a.Client.Start(); err != nil {
		return err
	}

	// Start the account manager
	if err := a.Manager.Start(); err != nil {
		return err
	}

	return nil
}

// Stop stops the API
func (a *API) Stop() error {
	// Stop the node
	if err := a.Node.Stop(); err != nil {
		return err
	}

	// Stop the Ethereum client
	if err := a.Client.Stop(); err != nil {
		return err
	}

	// Stop the account manager
	if err := a.Manager.Stop(); err != nil {
		return err
	}

	return nil
}

// GetBlock returns the current block
func (a *API) GetBlock(w http.ResponseWriter, r *http.Request) {
	// Get the current block
	block, err := a.Client.BlockByNumber(context.Background(), nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Marshal the block to JSON
	jsonBlock, err := json.Marshal(block)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON block to the response writer
	w.Write(jsonBlock)
}

// StartMiner starts the miner
func (a *API) StartMiner(w http.ResponseWriter, r *http.Request) {
	// Start the miner
	if err := a.Node.Miner().Start(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write a success message to the response writer
	w.Write([]byte("Miner started successfully"))
}

// StopMiner stops the miner
func (a *API) StopMiner(w http.ResponseWriter, r *http.Request) {
	// Stop the miner
	if err := a.Node.Miner().Stop(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	 return
	}

	// Write a success message to the response writer
	w.Write([]byte("Miner stopped successfully"))
}

// DeployContract deploys a contract
func (a *API) DeployContract(w http.ResponseWriter, r *http.Request) {
	// Deploy the contract
	contract, err := a.Manager.DeployContract(context.Background(), "contract.sol")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Marshal the contract to JSON
	jsonContract, err := json.Marshal(contract)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON contract to the response writer
	w.Write(jsonContract)
}

func main() {
	// Load the API configuration
	config, err := LoadConfig("api.config.json")
	if err != nil {
		log.Fatal(err)
	}

	// Create a new API instance
	api, err := NewAPI(config)
	if err != nil {
		log.Fatal(err)
	}

	// Start the API
	if err := api.Start(); err != nil {
		log.Fatal(err)
	}

	// Create a new router instance
	router := mux.NewRouter()

	// Define the API endpoints
	router.HandleFunc("/api/block", api.GetBlock).Methods("GET")
	router.HandleFunc("/api/miner/start", api.StartMiner).Methods("POST")
	router.HandleFunc("/api/miner/stop", api.StopMiner).Methods("POST")
	router.HandleFunc("/api/contract/deploy", api.DeployContract).Methods("POST")

	// Start the router
	log.Fatal(http.ListenAndServe(":8080", router))
}
