"""Double DQN agent for Paper 1 (van Hasselt et al. 2016).

Identical to DQN (agent_dqn.py) except the target: the ONLINE network chooses the next
action and the TARGET network evaluates it, which reduces max-operator overestimation.

    a* = argmax_a' Q_online(s', a')
    y  = r + gamma * (1 - terminal) * Q_target(s', a*)
"""

from rainbow_ablation.paper1.agent_dqn import DQNAgent


class DDQNAgent(DQNAgent):
    algo = "ddqn"

    # ---------------- CHANGED CODE (was agent_ini.py:106, max over target net; source agent_DDDQN.py:354-379): ----------------
    # Reason (item 2.12): this is what defines Double DQN.
    def _next_q(self, next_states):
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)    # [B, 1]
        return self.target_net(next_states).gather(1, next_actions).squeeze(1)      # [B]
    # ---------------- END CHANGED CODE ----------------
