# ai/swarm/swarm_decision.py

import uuid
import datetime
from collections import Counter, defaultdict

class SwarmDecisionAuditTrail:
    def __init__(self):
        self.history = []

    def log(self, decision_id, agent_decisions, weights, options, outcome, explanation):
        self.history.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "decision_id": decision_id,
            "agent_decisions": list(agent_decisions),
            "weights": dict(weights) if weights else None,
            "options": list(options) if options else None,
            "outcome": outcome,
            "explanation": explanation
        })

    def get_history(self):
        return self.history

class SwarmDecisionEngine:
    def __init__(self):
        self.audit_trail = SwarmDecisionAuditTrail()

    def swarm_vote(
        self, 
        agent_decisions, 
        weights=None, 
        options=None, 
        consensus_threshold=0.5, 
        tie_breaker="random"
    ):
        """
        Advanced swarm voting algorithm.
        - agent_decisions: list of agent votes/decisions (strings or enums)
        - weights: dict {agent_index: float} (optional, for weighted voting)
        - options: list of valid options (optional)
        - consensus_threshold: minimum percent required to win (default: 0.5 for majority)
        - tie_breaker: "random" or "first" (default: random)
        Returns: (outcome, explanation)
        """
        if not agent_decisions:
            outcome = None
            explanation = "No agent decisions provided."
            self.audit_trail.log(str(uuid.uuid4()), agent_decisions, weights, options, outcome, explanation)
            return outcome, explanation

        if options is None:
            options = list(set(agent_decisions))

        # Weighted tally
        tally = defaultdict(float)
        for i, decision in enumerate(agent_decisions):
            if decision not in options:
                continue
            weight = weights.get(i, 1) if weights else 1
            tally[decision] += weight

        total_votes = sum(tally.values())
        if total_votes == 0:
            outcome = None
            explanation = "No valid votes."
            self.audit_trail.log(str(uuid.uuid4()), agent_decisions, weights, options, outcome, explanation)
            return outcome, explanation

        # Find the option(s) with the highest votes
        max_votes = max(tally.values())
        winners = [opt for opt, v in tally.items() if v == max_votes]
        consensus_met = max_votes > total_votes * consensus_threshold

        if len(winners) == 1 and consensus_met:
            outcome = winners[0]
            explanation = (
                f"Winner: {outcome} with {max_votes}/{total_votes} votes. "
                f"Consensus threshold: {consensus_threshold*100:.1f}%"
            )
        elif len(winners) > 1 and consensus_met:
            # Tie-breaking
            import random
            if tie_breaker == "random":
                outcome = random.choice(winners)
                explanation = (
                    f"Tie between {winners}, selected {outcome} by random. "
                    f"Each had {max_votes} votes."
                )
            elif tie_breaker == "first":
                outcome = winners[0]
                explanation = (
                    f"Tie between {winners}, selected {outcome} (first in list)."
                )
            else:
                outcome = None
                explanation = f"Tie between {winners}, no tie-breaker applied."
        else:
            outcome = None
            explanation = (
                f"No decision reached. Top option(s): {winners} with {max_votes}/{total_votes} votes. "
                f"Consensus threshold not met ({consensus_threshold*100:.1f}%)."
            )
        
        decision_id = str(uuid.uuid4())
        self.audit_trail.log(decision_id, agent_decisions, weights, options, outcome, explanation)
        return outcome, explanation

    def get_audit_trail(self):
        return self.audit_trail.get_history()

# Example usage:
# swarm = SwarmDecisionEngine()
# agent_decisions = ["A", "B", "A", "A", "B"]
# weights = {0: 1.0, 1: 2.0, 2: 1.0, 3: 1.0,# printwarm.get_audit_trail())
