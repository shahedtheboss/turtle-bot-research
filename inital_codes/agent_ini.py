import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent():
    def __init__(self, state_size=7, action_size=3):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def select_greedy_action(self, state):
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.asarray(states), dtype=torch.float32,
            device=self.device
        )
        actions = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.asarray(next_states), dtype=torch.float32,
            device=self.device
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)


        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    ''''''
    def update_target(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
    ''''''

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * self.epsilon_decay
        )


    
    def save(self, path):
        """Save enough information to resume training later."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)


    def load(self, path):
        """Restore a previously saved checkpoint."""
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]







        