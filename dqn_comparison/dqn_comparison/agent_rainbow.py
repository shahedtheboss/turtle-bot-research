"""Rainbow DQN agent for Paper 1 (Hessel et al. 2018).

The learning code is NOT copied: it is the same RainbowDQNAgent used in the pilot and in
the ablation study (now dqn_comparison/rainbow_core.py, was rainbow_ablation/agent.py; logic identical to mini_dqn_nav/agent_rainbow.py).
This thin subclass only adds the interface Paper 1's shared train/evaluate code expects,
so Rainbow's algorithm stays exactly as it was (no silent change).
"""

import numpy as np
import torch

from dqn_comparison.rainbow_core import RainbowDQNAgent as _BaseRainbow


class RainbowAgent(_BaseRainbow):
    algo = "rainbow"

    # ---------------- NEW CODE ADDED: same interface as the DQN-family agents ----------------
    # Reason (items 3.8 and 5.2): select_action returns (action, was_random); Rainbow never
    # takes epsilon-random actions (NoisyNet explores), so was_random is always False.
    epsilon = 0.0

    def update_epsilon(self, env_steps_total):
        pass

    def select_action(self, state):
        return super().select_action(state), False

    def estimate_q(self, states):
        """Expected Q = sum_z p(z) * z over the C51 support, mean (noise-free) weights. [B, 7]"""
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.q_values(self.policy_net(x)).cpu().numpy()
        self.policy_net.train()
        return q

    def save(self, path):
        super().save(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        ckpt["algo"] = self.algo           # item 2.14: evaluate.py rebuilds from this
        torch.save(ckpt, path)
    # ---------------- END NEW CODE ----------------


def build_rainbow(state_size=52, action_size=7, v_min=-75.0, v_max=125.0, n_step=3, sigma0=0.5):
    """Rainbow with exactly the hyperparameters of the pilot / ablation train.py.
    v_min/v_max: C51 support; defaults are v1. Study v2 scales rewards by 0.01 and passes -0.75 / 1.25
    (CHANGED 2026-11-19: support became parameters; defaults unchanged)."""
    return RainbowAgent(
        state_size=state_size, action_size=action_size,
        learning_rate=1e-4, gamma=0.99, batch_size=128, memory_size=50000,
        n_step=n_step, num_atoms=51, v_min=v_min, v_max=v_max,
        target_update_every=250, warmup_transitions=1000, sigma0=sigma0,   # n_step, sigma0: tuning 2026-11-21
    )
