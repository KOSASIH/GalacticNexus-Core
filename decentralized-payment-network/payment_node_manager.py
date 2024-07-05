package main

import (
	"fmt"
	"net"
)

type PaymentNodeManager struct {
	nodes map[string]net.Conn
}

func NewPaymentNodeManager() *PaymentNodeManager {
	return &PaymentNodeManager{nodes: make(map[string]net.Conn)}
}

func (pm *PaymentNodeManager) addNode(nodeID string, addr string) error {
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		return err
	}
	pm.nodes[nodeID] = conn
	return nil
}

func (pm *PaymentNodeManager) removeNode(nodeID string) error {
	delete(pm.nodes, nodeID)
	return nil
}
