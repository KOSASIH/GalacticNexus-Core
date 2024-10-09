pragma solidity ^0.8.0;

contract GalacticNexus {
    // Mapping of user addresses to their balances
    mapping (address => uint256) public balances;

    // Mapping of user addresses to their transaction history
    mapping (address => Transaction[]) public transactionHistory;

    // Struct to represent a transaction
    struct Transaction {
        address sender;
        address recipient;
        uint256 amount;
        uint256 timestamp;
    }

    // Event emitted when a transaction is made
    event TransactionEvent(address indexed sender, address indexed recipient, uint256 amount);

    // Function to transfer tokens
    function transfer(address recipient, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
        emit TransactionEvent(msg.sender, recipient, amount);
        transactionHistory[msg.sender].push(Transaction(msg.sender, recipient, amount, block.timestamp));
    }

    // Function to get user balance
    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }

    // Function to get user transaction history
    function getTransactionHistory(address user) public view returns (Transaction[] memory) {
        return transactionHistory[user];
    }
}
