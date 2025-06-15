# ai/self_learning/reinforcement_agent.py
import numpy as np

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, alpha=0.1, gamma=0.99):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += alpha * (td_target - self.q_table[state, action])
