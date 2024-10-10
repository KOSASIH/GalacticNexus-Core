pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract AssetToken {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function mint(address to, uint256 amount) public {
        // Mint new tokens
        // ...
    }

    function transfer(address from, address to, uint256 amount) public {
        // Transfer tokens
        // ...
    }
}
