"""Dueling Double DQN (D3QN) agent for Paper 1 (Wang et al. 2016 + van Hasselt et al. 2016).

Same as DDQN (agent_ddqn.py) but with a dueling network:

    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

Shapes: features [B, 256]; V [B, 1]; A [B, 7]; Q [B, 7].
"""

import torch.nn as nn

from dqn_comparison.agent_ddqn import DDQNAgent


class DuelingQNetwork(nn.Module):
    # ---------------- CHANGED CODE (was agent_DDDQN.py:28-73, trunk 52->128->128, heads 128->64): ----------------
    # Reason (items 2.2-2.3): trunk 256-256 and heads 256->128->{1,7}, mirroring Rainbow's
    # value/advantage heads, so the only difference from Rainbow's head is noise/distribution.
    def __init__(self, state_size, action_size, hidden=256, head_hidden=128):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, action_size))
    # ---------------- END CHANGED CODE ----------------

    def forward(self, state):
        features = self.feature_layer(state)                       # agent_DDDQN.py:57
        value = self.value_stream(features)                        # [B, 1]
        advantage = self.advantage_stream(features)                # [B, 7]
        return value + advantage - advantage.mean(dim=1, keepdim=True)   # agent_DDDQN.py:67-70


class D3QNAgent(DDQNAgent):
    algo = "d3qn"

    # ---------------- NEW CODE ADDED: dueling network instead of the plain one ----------------
    # Reason (item 2.13): this is what defines D3QN (agent_DDDQN.py:91-96 used DuelingQNetwork).
    def _build_network(self):
        return DuelingQNetwork(self.state_size, self.action_size,
                               self.cfg["hidden"], self.cfg["head_hidden"])
    # ---------------- END NEW CODE ----------------
