pragma solidity ^0.8.0;

import "https://github.com/Open Zeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Role.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract GNX_Governance is Ownable {
    using SafeMath for uint256;

    // Governance metadata
    string public name = "Galactic Nexus Governance";
    string public symbol = "GNX-GOV";

    // Governance roles
    mapping (address => mapping (string => bool)) public roles;
    mapping (string => address[]) public roleMembers;

    // Governance events
    event RoleAdded(address indexed member, string indexed role);
    event RoleRemoved(address indexed member, string indexed role);
    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string indexed proposal);
    event ProposalVoted(uint256 indexed proposalId, address indexed voter, bool indexed vote);
    event ProposalExecuted(uint256 indexed proposalId, bool indexed outcome);

    // Governance functions
    constructor() public {
        // Initialize governance roles
        roles[msg.sender]["admin"] = true;
        roleMembers["admin"].push(msg.sender);
    }

    function addRole(address _member, string _role) public onlyOwner {
        require(_member != address(0));
        require(_role != "");

        roles[_member][_role] = true;
        roleMembers[_role].push(_member);

        emit RoleAdded(_member, _role);
    }

    function removeRole(address _member, string _role) public onlyOwner {
        require(_member != address(0));
        require(_role != "");

        roles[_member][_role] = false;
        for (uint256 i = 0; i < roleMembers[_role].length; i++) {
            if (roleMembers[_role][i] == _member) {
                roleMembers[_role][i] = roleMembers[_role][roleMembers[_role].length - 1];
                roleMembers[_role].length--;
                break;
            }
        }

        emit RoleRemoved(_member, _role);
    }

    function createProposal(string _proposal) public onlyOwner {
        require(_proposal != "");

        uint256 proposalId = uint256(keccak256(abi.encodePacked(_proposal)));
        emit ProposalCreated(proposalId, msg.sender, _proposal);
    }

    function voteProposal(uint256 _proposalId, bool _vote) public {
        require(_proposalId != 0);
        require(_vote != false);

        emit ProposalVoted(_proposalId, msg.sender, _vote);
    }

    function executeProposal(uint256 _proposalId, bool _outcome) public onlyOwner {
        require(_proposalId != 0);
        require(_outcome != false);

        emit ProposalExecuted(_proposalId, _outcome);
    }
}
