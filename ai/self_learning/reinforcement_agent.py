# ai/self_learning/reinforcement_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

class AdvancedRLAgent:
    def __init__(self, state_dim, action_dim, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = deque(maxlen=100_000)
        self.batch_size = 256
        self.gamma = 0.99
        self.epsilon = 1.0  # Exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 1e-4
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.update_target_every = 1000
        self.train_step = 0

        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def choose_action(self, state, use_greedy=False):
        if np.random.rand() < self.epsilon and not use_greedy:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
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

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.train_step += 1

        # Update target network
        if self.train_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename='agent.pth'):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, filename='agent.pth'):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    def log_status(self, episode, reward):
        print(f'Episode: {episode}, Epsilon: {self.epsilon:.4f}, Reward: {reward:.2f}')

# Usage:
# agent = AdvancedRLAgent(state_dim=<your_state_dim>, action_dim=<your_action_dim>)
# for each episode:
#     state = env.reset()
#     done = False
#     while not done:
#         action = agent.choose_action(state)
#         next_state, reward, done, _ = env.step(action)
#         agent.store_transition(state, action, reward, next_state, done)
#         agent.learn()
#         state = next_state
#     agent.log_status(episode, total_reward)
