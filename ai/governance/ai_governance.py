# ai/governance/ai_governance.py

import uuid
import datetime
from collections import defaultdict

class GovernanceAuditTrail:
    def __init__(self):
        self.history = []

    def log(self, proposal_id, proposal, votes, options, outcome, explanation):
        self.history.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "proposal_id": proposal_id,
            "proposal": proposal,
            "votes": dict(votes),
            "options": options,
            "outcome": outcome,
            "explanation": explanation
        })

    def get_history(self):
        return self.history

class AIGovernance:
    def __init__(self):
        self.audit_trail = GovernanceAuditTrail()

    def propose_resolution(self, proposal, votes, weights=None, options=None, consensus_threshold=0.5):
        """
        AI-powered consensus proposal with weighted/multi-option voting and audit trail.
        - proposal: Proposal text
        - votes: {voter_id: choice}
        - weights: {voter_id: float/int} (optional)
        - options: list of valid options (default: ['yes','no','abstain'])
        - consensus_threshold: percent required for passing (default: 0.5, i.e. majority)
        Returns: (outcome, explanation)
        """
        if options is None:
            options = ['yes', 'no', 'abstain']

        # Weighted tally
        tally = defaultdict(float)
        for voter, choice in votes.items():
            if choice not in options:
                continue
            weight = weights.get(voter, 1) if weights else 1
            tally[choice] += weight

        non_abstain_total = sum(tally[opt] for opt in options if opt != 'abstain')
        if non_abstain_total == 0:
            outcome = "No decision (only abstentions or no votes)"
            explanation = f"Votes: {dict(tally)}"
        else:
            winning_option, max_votes = max(
                ((opt, count) for opt, count in tally.items() if opt != 'abstain'),
                key=lambda x: x[1]
            )
            if max_votes > non_abstain_total * consensus_threshold:
                outcome = f"Proposal Passed: {winning_option.upper()}"
            else:
                outcome = "Proposal Rejected (no consensus)"
            explanation = (
                f"Votes: {dict(tally)} | Total non-abstain: {non_abstain_total} | "
                f"Winning: {winning_option.upper()} ({max_votes} votes)"
            )

        proposal_id = str(uuid.uuid4())
        self.audit_trail.log(proposal_id, proposal, votes, options, outcome, explanation)
        return outcome, explanation

    def get_audit_trail(self):
        return self.audit_trail.get_history()

# Example usage:
# governance = AIGovernance()
# proposal = "Should the protocol upgrade to v2.0?"
# votes = {"alice": "yes", "bob": "no", "carol": "yes", "dan": "abstain"}
# weights = {"alice": 2, "bob": 1, "carol": 3, "dan": 1}
# outcome, explanation = governance.propose_resolution(proposal, votes, weights)
# print(outcome)
# print(explanation)
# print(governance.get_audit_trail())
