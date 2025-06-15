# ai/swarm/swarm_decision.py
def swarm_vote(agent_decisions):
    """
    Simple majority vote across agents.
    """
    from collections import Counter
    return Counter(agent_decisions).most_common(1)[0][0]
