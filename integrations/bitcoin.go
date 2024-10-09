package integrations

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/btcsuite/btcd/btcec"
	"github.com/btcsuite/btcd/chaincfg"
	"github.com/btcsuite/btcd/rpcclient"
)

// BitcoinIntegration represents the Bitcoin integration
type BitcoinIntegration struct {
	*rpcclient.Client
	*chaincfg.Params
}

// NewBitcoinIntegration creates a new Bitcoin integration instance
func NewBitcoinIntegration(config *Config) (*BitcoinIntegration, error) {
	// Create a new RPC client instance
	client, err := rpcclient.New(&rpcclient.ConnConfig{
		HttpPostMode: true,
		Dial:         func() (net.Conn, error) { return net.Dial("tcp", config.Bitcoin.RPCServer) },
	}, nil)
	if err != nil {
		return nil, err
	}

	// Create a new chain configuration instance
	params, err := chaincfg.NewParams(config.Bitcoin.Network)
	if err != nil {
		return nil, err
	}

	// Create a new Bitcoin integration instance
	integration := &BitcoinIntegration{
		Client: client,
		Params: params,
	}

	return integration, nil
}

// GetBalance returns the balance of a Bitcoin address
func (b *BitcoinIntegration) GetBalance(address string) (float64, error) {
	// Get the balance of the address
	balance, err := b.Client.GetBalance(address)
	if err != nil {
		return 0, err
	}

	return balance, nil
}

// SendTransaction sends a Bitcoin transaction
func (b *BitcoinIntegration) SendTransaction(from, to string, amount float64) (string, error) {
	// Create a new transaction instance
	tx, err := b.Client.CreateTransaction(from, to, amount)
	if err != nil {
		return "", err
	}

	// Sign the transaction
	sig, err := btcec.SignCompact(btcec.S256(), tx, b.Params)
	if err != nil {
		return "", err
	}

	// Send the transaction
	txid, err := b.Client.SendRawTransaction(tx, sig)
	if err != nil {
		return "", err
	}

	return txid, nil
}
