# ai/agents/autonomous_trader.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class TradingDQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(TradingDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class AutonomousTrader:
    def __init__(
        self, 
        state_dim=20,  # Number of input features (customize as needed)
        action_dim=3,  # [0: SELL, 1: HOLD, 2: BUY]
        learning_rate=1e-4, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_min=0.01, 
        epsilon_decay=0.995,
        memory_size=100_000,
        batch_size=256,
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory = deque(maxlen=memory_size)

        self.policy_net = TradingDQN(state_dim, action_dim).to(self.device)
        self.target_net = TradingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.train_step = 0
        self.update_target_every = 500

        self.checkpoint_dir = 'trader_checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _preprocess(self, market_data):
        # Example: Use last N normalized prices as features (customize for your dataset)
        prices = np.array(market_data['prices'][-self.state_dim:])
        prices = (prices - prices.mean()) / (prices.std() + 1e-8)
        features = torch.FloatTensor(prices).unsqueeze(0).to(self.device)
        return features

    def decide(self, market_data, use_greedy=False):
        state = self._preprocess(market_data)
        if np.random.rand() < self.epsilon and not use_greedy:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
            action = q_values.argmax().item()
        return ["SELL", "HOLD", "BUY"][action]

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.cat(next_states),
            torch.FloatTensor(dones).to(self.device)
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        q_eval = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay for exploration
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.train_step += 1

        # Periodically update target network for stability
        if self.train_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename='trader.pth'):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, os.path.join(self.checkpoint_dir, filename))

    def load(self, filename='trader.pth'):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename), map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    def explain_decision(self, market_data):
        """Returns Q-values for each action for explainability."""
        state = self._preprocess(market_data)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return {action: float(q) for action, q in zip(["SELL", "HOLD", "BUY"], q_values.squeeze().cpu().numpy())}

    def risk_management(self, action, portfolio, market_data):
        """
        Integrate risk constraints here (e.g., max drawdown, stop loss).
        """
        if action == "BUY" and portfolio.get('risk', 0) > 0.8:
            return "HOLD"
        return action

# Example usage:
# trader = AutonomousTrader(state_dim=20)
# for timestep in ...:
#     action = trader.decide(market_data)
#     # ...execute action, get reward, next_state, done...
#     state = trader._preprocess(market_data)
#     next_state = trader._preprocess(next_market_data)
#     trader.store_experience(state, action_id, reward, next_state, done)
#     trader.learn()
