package main

import (
	"crypto/ecdsa"
	"crypto/rand"
	"encoding/hex"
)

type PaymentProtocol struct {
	privateKey *ecdsa.PrivateKey
}

func NewPaymentProtocol() *PaymentProtocol {
	privateKey, err := ecdsa.GenerateKey(ecdsa.P256(), rand.Reader)
	if err != nil {
		return nil
	}
	return &PaymentProtocol{privateKey: privateKey}
}

func (pp *PaymentProtocol) signTransaction(tx []byte) ([]byte, error) {
	r, s, err := ecdsa.Sign(rand.Reader, pp.privateKey, tx)
	if err != nil {
		return nil, err
	}
	return append(r.Bytes(), s.Bytes()...), nil
}
