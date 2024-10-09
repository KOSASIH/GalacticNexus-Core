package markets

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/gorilla/mux"
	"github.com/shopspring/decimal"
)

// Market represents a market
type Market struct {
	ID        string
	Name      string
	BaseAsset  string
	QuoteAsset string
	Price     decimal.Decimal
	Volume    decimal.Decimal
}

// MarketExpansion represents the market expansion
type MarketExpansion struct {
	*Market
	Router *mux.Router
}

// NewMarketExpansion creates a new market expansion instance
func NewMarketExpansion(config *Config) (*MarketExpansion, error) {
	// Create a new market instance
	market := &Market{
		ID:        config.ID,
		Name:      config.Name,
		BaseAsset: config.BaseAsset,
		QuoteAsset: config.QuoteAsset,
	}

	// Create a new router instance
	router := mux.NewRouter()

	// Define the market endpoints
	router.HandleFunc("/markets/{id}", market.GetMarket).Methods("GET")
	router.HandleFunc("/markets/{id}/price", market.GetPrice).Methods("GET")
	router.HandleFunc("/markets/{id}/volume", market.GetVolume).Methods("GET")
	router.HandleFunc("/markets/{id}/buy", market.Buy).Methods("POST")
	router.HandleFunc("/markets/{id}/sell", market.Sell).Methods("POST")

	// Create a new market expansion instance
	expansion := &MarketExpansion{
		Market: market,
		Router: router,
	}

	return expansion, nil
}

// GetMarket returns the market
func (m *Market) GetMarket(w http.ResponseWriter, r *http.Request) {
	// Get the market ID from the request
	vars := mux.Vars(r)
	id := vars["id"]

	// Check if the market ID matches the current market ID
	if id != m.ID {
		http.Error(w, "Invalid market ID", http.StatusBadRequest)
		return
	}

	// Marshal the market to JSON
	jsonMarket, err := json.Marshal(m)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON market to the response writer
	w.Write(jsonMarket)
}

// GetPrice returns the market price
func (m *Market) GetPrice(w http.ResponseWriter, r *http.Request) {
	// Get the market ID from the request
	vars := mux.Vars(r)
	id := vars["id"]

	// Check if the market ID matches the current market ID
	if id != m.ID {
		http.Error(w, "Invalid market ID", http.StatusBadRequest)
		return
	}

	// Marshal the market price to JSON
	jsonPrice, err := json.Marshal(m.Price)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON price to the response writer
	w.Write(jsonPrice)
}

// GetVolume returns the market volume
func (m *Market) GetVolume(w http.ResponseWriter, r *http.Request) {
	// Get the market ID from the request
	vars := mux.Vars(r)
	id := vars["id"]

	// Check if the market ID matches the current market ID
	if id != m.ID {
		http.Error(w, "Invalid market ID", http.StatusBadRequest)
		return
	}

	// Marshal the market volume to JSON
	jsonVolume, err := json.Marshal(m.Volume)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON volume to the response writer
	w.Write(jsonVolume)
}

// Buy buys an asset on the market
func (m *Market) Buy(w http.ResponseWriter, r *http.Request) {
	// Get the market ID from the request
	vars := mux.Vars(r)
	id := vars["id"]

	// Check if the market ID matches the current market ID
	if id != m.ID {
		http.Error(w, "Invalid market ID", http.StatusBadRequest)
		return
	}

	// Get the buy amount from the request
	var buyAmount decimal.Decimal
	err := json.NewDecoder(r.Body).Decode(&buyAmount)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Check if the buy amount is valid
	if buyAmount.LessThanOrEqual(decimal.Zero) {
		http.Error(w, "Invalid buy amount", http.StatusBadRequest)
		return
	}

	// Update the market price and volume
	m.Price = m.Price.Add(buyAmount)
	m.Volume = m.Volume.Add(buyAmount)

	// Marshal the market to JSON
	jsonMarket, err := json.Marshal(m)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON market to the response writer
	w.Write(jsonMarket )
}

// Sell sells an asset on the market
func (m *Market) Sell(w http.ResponseWriter, r *http.Request) {
	// Get the market ID from the request
	vars := mux.Vars(r)
	id := vars["id"]

	// Check if the market ID matches the current market ID
	if id != m.ID {
		http.Error(w, "Invalid market ID", http.StatusBadRequest)
		return
	}

	// Get the sell amount from the request
	var sellAmount decimal.Decimal
	err := json.NewDecoder(r.Body).Decode(&sellAmount)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Check if the sell amount is valid
	if sellAmount.LessThanOrEqual(decimal.Zero) {
		http.Error(w, "Invalid sell amount", http.StatusBadRequest)
		return
	}

	// Update the market price and volume
	m.Price = m.Price.Sub(sellAmount)
	m.Volume = m.Volume.Sub(sellAmount)

	// Marshal the market to JSON
	jsonMarket, err := json.Marshal(m)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write the JSON market to the response writer
	w.Write(jsonMarket)
}
