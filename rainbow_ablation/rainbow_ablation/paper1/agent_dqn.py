"""DQN agent for Paper 1 (Mnih et al. 2013/2015).

Sources: inital_codes/agent_ini.py (DQN target) and inital_codes/agent_DDDQN.py
(replay, Huber loss, gradient clipping, save/load). Settings are shared with Rainbow so
that only the algorithm differs between methods (paper1_change_plan.md items 2.1-2.15).

Target (1-step, uniform replay):
    y = r + gamma * (1 - terminal) * max_a' Q_target(s', a')

DDQN (agent_ddqn.py) and D3QN (agent_d3qn.py) subclass this file and override only
`_next_q` (the target) and `_build_network` (the architecture).
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------- NEW CODE ADDED: one place for the hyperparameters shared by DQN/DDQN/D3QN ----------------
# Reason: items 2.4-2.8; equal settings for every method (Rainbow uses the same lr, batch,
# buffer, gamma, target period, warm-up and gradient clip).
SHARED = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "memory_size": 50_000,
    "target_update_every": 250,     # learning steps
    "warmup_transitions": 1_000,
    "grad_clip": 10.0,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 10_000,  # environment steps, linear
    "hidden": 256,
    "head_hidden": 128,
}
# ---------------- END NEW CODE ----------------


class QNetwork(nn.Module):
    """Plain Q-network: 52 -> 256 -> 256 -> 128 -> 7.

    Input  x: float tensor [batch, 52]
    Output  : Q-values     [batch, 7]
    """

    # ---------------- CHANGED CODE (was agent_ini.py:12-22 [7->128->128->out] and agent_DDDQN.py:11-25 [52->1024->1024->out]): ----------------
    # Reason (item 2.2): same trunk width as Rainbow (256-256) and the same head depth
    # (one 128 hidden layer, like Rainbow's value/advantage heads), so capacity is not a confound.
    def __init__(self, state_size, action_size, hidden=256, head_hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, action_size),
        )
    # ---------------- END CHANGED CODE ----------------

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Vanilla DQN: epsilon-greedy, uniform replay, 1-step target with max over the target net."""

    algo = "dqn"

    def __init__(self, state_size=52, action_size=7, **overrides):
        cfg = dict(SHARED, **overrides)
        self.cfg = cfg
        self.state_size, self.action_size = state_size, action_size
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.learning_steps = 0                      # was agent_DDDQN.py:78 (learn_steps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ---------------- CHANGED CODE (was agent_ini.py:43 and agent_DDDQN.py:102, lr=0.001): ----------------
        # Reason (item 2.4): Rainbow's learning rate 1e-4 for every method.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg["learning_rate"])
        # ---------------- END CHANGED CODE ----------------
        self.loss_fn = nn.SmoothL1Loss(reduction="none")   # Huber, as agent_DDDQN.py:104 (per-sample for clarity)

        # ---------------- CHANGED CODE (was agent_ini.py:46,49: buffer 10k, batch 64): ----------------
        # Reason (item 2.5): buffer 50k and batch 128, as Rainbow and D3QN.
        self.memory = deque(maxlen=cfg["memory_size"])
        # ---------------- END CHANGED CODE ----------------

        # ---------------- CHANGED CODE (was agent_DDDQN.py:110-112 [x0.9995 per learning step] and agent_ini.py:51-53 [x0.99 per episode]): ----------------
        # Reason (item 2.8): linear decay over ENVIRONMENT steps, so exploration does not
        # depend on how many gradient updates run per step; 10k steps ~ the guide's 100-episode window.
        self.epsilon = cfg["epsilon_start"]
        # ---------------- END CHANGED CODE ----------------

    # ---------------- NEW CODE ADDED: overridable network builder ----------------
    # Reason: lets D3QN swap in the dueling network without copying the whole agent.
    def _build_network(self):
        return QNetwork(self.state_size, self.action_size,
                        self.cfg["hidden"], self.cfg["head_hidden"])
    # ---------------- END NEW CODE ----------------

    # ---------------- NEW CODE ADDED: epsilon schedule by environment step ----------------
    # Reason (item 2.8). epsilon(t) = max(end, start - (start - end) * t / decay_steps)
    def update_epsilon(self, env_steps_total):
        c = self.cfg
        frac = min(1.0, env_steps_total / c["epsilon_decay_steps"])
        self.epsilon = c["epsilon_start"] + (c["epsilon_end"] - c["epsilon_start"]) * frac
    # ---------------- END NEW CODE ----------------

    def select_action(self, state):
        """Epsilon-greedy. Returns (action, was_random) so training can count random steps."""
        # ---------------- CHANGED CODE (was agent_DDDQN.py:114-122, returned only the action): ----------------
        # Reason (item 3.8): report how many steps were epsilon-random vs the agent's own choice.
        if random.random() < self.epsilon:
            return random.randrange(self.action_size), True
        return self.select_greedy_action(state), False
        # ---------------- END CHANGED CODE ----------------

    def select_greedy_action(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(x).argmax(dim=1).item())

    # ---------------- NEW CODE ADDED: expected Q-values for any batch of states ----------------
    # Reason (item 5.2): overestimation check (predicted max Q at the start state vs the
    # discounted return actually obtained). Same method name exists on the Rainbow wrapper.
    def estimate_q(self, states):
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(x).cpu().numpy()        # [batch, 7]
    # ---------------- END NEW CODE ----------------

    # ---------------- CHANGED CODE (was agent_DDDQN.py:134-135, remember(state, action, reward, next_state, done)): ----------------
    # Reason (item 2.9, bug fix): only goal/collision are terminal. train_DDDQN.py:107 passed
    # `terminated`, which also included the no-progress cut-off, so those targets wrongly
    # dropped the future value. `truncated` is accepted for API parity with Rainbow; with
    # 1-step targets a truncated step simply bootstraps from next_state.
    def remember(self, state, action, reward, next_state, terminal, truncated=False):
        self.memory.append((
            np.asarray(state, dtype=np.float32), int(action), float(reward),
            np.asarray(next_state, dtype=np.float32), float(bool(terminal)),
        ))
    # ---------------- END CHANGED CODE ----------------

    def _next_q(self, next_states):
        """DQN target value: max_a' Q_target(s', a').  Shape [batch]. (was agent_ini.py:106)"""
        return self.target_net(next_states).max(dim=1).values

    def compute_targets(self, rewards, next_states, terminals):
        """y = r + gamma * (1 - terminal) * next_q.  All tensors shape [batch]."""
        with torch.no_grad():
            return rewards + self.gamma * (1.0 - terminals) * self._next_q(next_states)

    def learn(self):
        # ---------------- CHANGED CODE (was agent_DDDQN.py:223, warm-up 2000; agent_ini.py:79, none): ----------------
        # Reason (item 2.7): warm-up 1000 transitions, as Rainbow.
        if len(self.memory) < max(self.batch_size, self.cfg["warmup_transitions"]):
            return None
        # ---------------- END CHANGED CODE ----------------
        batch = random.sample(self.memory, self.batch_size)              # agent_DDDQN.py:247
        states, actions, rewards, next_states, terminals = zip(*batch)
        dev = self.device
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=dev)          # [B, 52]
        actions = torch.as_tensor(actions, dtype=torch.long, device=dev).unsqueeze(1)          # [B, 1]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=dev)                   # [B]
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=dev)
        terminals = torch.as_tensor(terminals, dtype=torch.float32, device=dev)               # [B]

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)                     # [B]
        target_q = self.compute_targets(rewards, next_states, terminals)                      # [B]
        loss = self.loss_fn(current_q, target_q).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()

        self.learning_steps += 1
        # ---------------- CHANGED CODE (was agent_DDDQN.py:466-468 every 500; train_ini.py:122-123 every 2 episodes): ----------------
        # Reason (item 2.6): hard target update every 250 learning steps, as Rainbow.
        if self.learning_steps % self.cfg["target_update_every"] == 0:
            self.update_target()
        # ---------------- END CHANGED CODE ----------------
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------------- CHANGED CODE (was agent_DDDQN.py:488-495, saved only nets/optimizer/epsilon): ----------------
    # Reason (item 2.14): store algo and sizes so evaluate.py can rebuild the right network.
    def save(self, path):
        torch.save({
            "algo": self.algo,
            "state_size": self.state_size, "action_size": self.action_size,
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": float(self.epsilon),
            "learning_steps": self.learning_steps,
            "cfg": dict(self.cfg),          # (review fix 6) native int/float types, so it can rebuild the agent
        }, path)
    # ---------------- END CHANGED CODE ----------------

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.learning_steps = ckpt.get("learning_steps", 0)
