package main

import (
	"fmt"
	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-chaincode-go/stub"
)

type GalacticNexusChaincode struct {
}

func (c *GalacticNexusChaincode) Init(stub shim.ChaincodeStubInterface) []byte {
	fmt.Println("Galactic Nexus Chaincode initialized")
	return nil
}

func (c *GalacticNexusChaincode) Invoke(stub shim.ChaincodeStubInterface) ([]byte, error) {
	fmt.Println("Galactic Nexus Chaincode invoked")
	function, args := stub.GetFunctionAndParameters()
	if function == "createGalacticNexus" {
		return c.createGalacticNexus(stub, args)
	} else if function == "updateGalacticNexus" {
		return c.updateGalacticNexus(stub, args)
	} else if function == "getGalacticNexus" {
		return c.getGalacticNexus(stub, args)
	} else {
		return nil, errors.New("Invalid function")
	}
}

func (c *GalacticNexusChaincode) createGalacticNexus(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	if len(args) != 5 {
		return nil, errors.New("Invalid number of arguments")
	}
	galacticNexusID := args[0]
	owner := args[1]
	creationDate := args[2]
	data := args[3]
	hash := args[4]
	galacticNexus := &GalacticNexus{
		ID:          galacticNexusID,
		Owner:       owner,
		CreationDate: creationDate,
		Data:        data,
		Hash:        hash,
	}
	err := stub.PutState(galacticNexusID, galacticNexus)
	if err != nil {
		return nil, err
	}
	return []byte("Galactic Nexus created successfully"), nil
}

func (c *GalacticNexusChaincode) updateGalacticNexus(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	if len(args) != 5 {
		return nil, errors.New("Invalid number of arguments")
	}
	galacticNexusID := args[0]
	owner := args[1]
	creationDate := args[2]
	data := args[3]
	hash := args[4]
	galacticNexus, err := c.getGalacticNexus(stub, []string{galacticNexusID})
	if err != nil {
		return nil, err
	}
	galacticNexus.Owner = owner
	galacticNexus.CreationDate = creationDate
	galacticNexus.Data = data
	galacticNexus.Hash = hash
	err = stub.PutState(galacticNexusID, galacticNexus)
	if err != nil {
		return nil, err
	}
	return []byte("Galactic Nexus updated successfully"), nil
}

func (c *GalacticNexusChaincode) getGalacticNexus(stub shim.ChaincodeStubInterface, args []string) ([]byte, error) {
	if len(args) != 1 {
		return nil, errors.New("Invalid number of arguments")
	}
	galacticNexusID := args[0]
	galacticNexus, err := stub.GetState(galacticNexusID)
	if err != nil {
		return nil, err
	}
	return []byte(galacticNexus), nil
}

type GalacticNexus struct {
	ID          string `json:"id"`
	Owner       string `json:"owner"`
	CreationDate string `json:"creationDate"`
	Data        string `json:"data"`
	Hash        string `json:"hash"`
}

func main() {
	fmt.Println("Galactic Nexus Chaincode main function")
}
