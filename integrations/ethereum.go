package integrations

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/ethereum/go-ethereum/accounts"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/ethclient"
)

// EthereumIntegration represents the Ethereum integration
type EthereumIntegration struct {
	*ethclient.Client
	*accounts.Manager
}

// NewEthereumIntegration creates a new Ethereum integration instance
func NewEthereumIntegration(config *Config) (*EthereumIntegration, error) {
	// Create a new Ethereum client instance
	client, err := ethclient.Dial(config.Ethereum.RPCServer)
	if err != nil {
		return nil, err
	}

	// Create a new account manager instance
	accountMgr, err := accounts.NewManager(config.Ethereum.AccountsDir)
	if err != nil {
		return nil, err
	}

	// Create a new Ethereum integration instance
	integration := &EthereumIntegration{
		Client:   client,
		Manager: accountMgr,
	}

	return integration, nil
}

// GetBalance returns the balance of an Ethereum address
func (e *EthereumIntegration) GetBalance(address common.Address) (*big.Int, error) {
	// Get the balance of the address
	balance, err := e.Client.BalanceAt(context.Background(), address, nil)
	if err != nil {
		return nil, err
	}

	return balance, nil
}

// SendTransaction sends an Ethereum transaction
func (e *EthereumIntegration) SendTransaction(from, to common.Address, amount *big.Int) (common.Hash, error) {
	// Create a new transaction instance
	tx, err := e.Client.NewTransaction(from, to, amount)
	if err != nil {
		return common.Hash{}, err
	}

	// Sign the transaction
	sig, err := e.Manager.SignTx(tx)
	if err != nil {
		return common.Hash{}, err
	}

	// Send the transaction
	txid, err := e.Client.SendTransaction(context.Background(), tx)
	if err != nil {
		return common.Hash{}, err
	}

	return txid, nil
}
