pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/ownership/Ownable.sol";
import "./GalacticNexusToken.sol";

contract GalacticNexusSmartContract is Ownable {
    GalacticNexusToken public token;
    address public galacticNexusAddress;

    mapping (address => uint256) public userBalances;
    mapping (address => uint256) public userRewards;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);
    event Reward(address indexed user, uint256 amount);

    constructor() public {
        token = new GalacticNexusToken();
        galacticNexusAddress = address(this);
    }

    function deposit(uint256 _amount) public {
        require(token.transferFrom(msg.sender, galacticNexusAddress, _amount), "Transfer failed");
        userBalances[msg.sender] += _amount;
        emit Deposit(msg.sender, _amount);
    }

    function withdraw(uint256 _amount) public {
        require(userBalances[msg.sender] >= _amount, "Insufficient balance");
        require(token.transfer(msg.sender, _amount), "Transfer failed");
        userBalances[msg.sender] -= _amount;
        emit Withdrawal(msg.sender, _amount);
    }

    function reward(address _user, uint256 _amount) public onlyOwner {
        require(token.transfer(_user, _amount), "Transfer failed");
        userRewards[_user] += _amount;
        emit Reward(_user, _amount);
    }

    function getUserBalance(address _user) public view returns (uint256) {
        return userBalances[_user];
    }

    function getUserReward(address _user) public view returns (uint256) {
        return userRewards[_user];
    }
}
