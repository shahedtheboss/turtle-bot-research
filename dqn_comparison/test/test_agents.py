"""Unit tests for Paper 1 agents (item 5.7). Run: python3 -m pytest dqn_comparison/test/test_agents.py -q

They check, on tiny hand-made batches, that each agent computes the target its paper
defines, that epsilon follows the linear schedule, and that only true terminals are stored
as terminal. No ROS or Gazebo needed.
"""

import numpy as np
import pytest
import torch

from dqn_comparison.agent_dqn import DQNAgent
from dqn_comparison.agent_ddqn import DDQNAgent
from dqn_comparison.agent_d3qn import D3QNAgent, DuelingQNetwork


class Fixed(torch.nn.Module):
    """Returns a fixed Q table regardless of input: shape [B, A]."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        return self.table[: x.shape[0]]


def _with_nets(agent, online, target):
    agent.policy_net, agent.target_net = Fixed(online), Fixed(target)
    agent.device = torch.device("cpu")
    return agent


ONLINE = [[1.0, 5.0, 2.0], [0.0, 0.0, 9.0]]      # online argmax: action 1, action 2
TARGET = [[3.0, 4.0, 10.0], [7.0, 1.0, 2.0]]     # target max: 10, 7 ; target at online argmax: 4, 2
R = torch.tensor([1.0, 1.0]); NS = torch.zeros(2, 3); T = torch.tensor([0.0, 1.0])


def test_dqn_target_uses_max_of_target_net():
    a = _with_nets(DQNAgent(3, 3), ONLINE, TARGET)
    y = a.compute_targets(R, NS, T)
    assert torch.allclose(y, torch.tensor([1 + 0.99 * 10.0, 1.0]))


def test_ddqn_target_uses_online_argmax_evaluated_by_target():
    a = _with_nets(DDQNAgent(3, 3), ONLINE, TARGET)
    y = a.compute_targets(R, NS, T)
    assert torch.allclose(y, torch.tensor([1 + 0.99 * 4.0, 1.0]))


def test_d3qn_is_double_and_dueling():
    a = D3QNAgent(52, 7)
    assert isinstance(a.policy_net, DuelingQNetwork)
    a = _with_nets(a, ONLINE, TARGET)
    assert torch.allclose(a.compute_targets(R, NS, T), torch.tensor([1 + 0.99 * 4.0, 1.0]))


def test_dueling_aggregation_mean_advantage_is_zero_offset():
    net = DuelingQNetwork(4, 3, hidden=8, head_hidden=4)
    x = torch.randn(5, 4)
    f = net.feature_layer(x)
    v, adv = net.value_stream(f), net.advantage_stream(f)
    q = net(x)
    assert q.shape == (5, 3)
    assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-5)   # mean_a Q = V
    assert torch.allclose(q - q.mean(dim=1, keepdim=True), adv - adv.mean(dim=1, keepdim=True), atol=1e-5)


@pytest.mark.parametrize("t,expected", [(0, 1.0), (5000, 0.525), (10000, 0.05), (50000, 0.05)])
def test_epsilon_linear_in_env_steps(t, expected):
    a = DQNAgent(52, 7)
    a.update_epsilon(t)
    assert a.epsilon == pytest.approx(expected)


def test_only_true_terminals_are_stored_as_terminal():
    a = DQNAgent(52, 7)
    s = np.zeros(52)
    a.remember(s, 0, 0.0, s, terminal=False, truncated=True)   # no-progress / time limit
    a.remember(s, 0, -25.0, s, terminal=True, truncated=False)  # collision
    assert [m[4] for m in a.memory] == [0.0, 1.0]


def test_no_learning_before_warmup_then_learns():
    a = DQNAgent(52, 7)
    s = np.random.rand(52)
    for _ in range(999):
        a.remember(s, 1, 0.1, s, False)
    assert a.learn() is None
    a.remember(s, 1, 0.1, s, False)
    assert isinstance(a.learn(), float) and a.learning_steps == 1


def test_select_action_reports_random_steps():
    a = DQNAgent(52, 7)
    a.epsilon = 1.0
    acts = [a.select_action(np.zeros(52)) for _ in range(50)]
    assert all(r for _, r in acts)
    a.epsilon = 0.0
    assert not a.select_action(np.zeros(52))[1]


def test_save_records_algo(tmp_path):
    for cls, name in ((DQNAgent, "dqn"), (DDQNAgent, "ddqn"), (D3QNAgent, "d3qn")):
        p = tmp_path / f"{name}.pth"
        cls(52, 7).save(str(p))
        ck = torch.load(str(p), weights_only=True)
        assert ck["algo"] == name and ck["state_size"] == 52 and ck["action_size"] == 7


# ---------------- NEW CODE ADDED (review fix 12): Rainbow wrapper, evaluate loader, guide schedule ----------------
def test_rainbow_wrapper_interface(tmp_path):
    from dqn_comparison.agent_rainbow import build_rainbow
    a = build_rainbow()
    act, was_random = a.select_action(np.zeros(52))
    assert 0 <= act < 7 and was_random is False
    assert a.estimate_q(np.zeros(52)).shape == (1, 7)
    assert a.estimate_q(np.zeros((2, 52))).shape == (2, 7)
    p = tmp_path / "rb.pth"
    a.save(str(p))
    assert torch.load(str(p), weights_only=True)["algo"] == "rainbow"


@pytest.mark.parametrize("name", ["dqn", "ddqn", "d3qn", "rainbow"])
def test_evaluate_loader_rebuilds_same_policy(tmp_path, name):
    from dqn_comparison.learning import load_greedy_agent
    from dqn_comparison.learning import AGENTS, probe_states
    a = AGENTS[name]()
    p = tmp_path / f"{name}.pth"
    a.save(str(p))
    b = load_greedy_agent(str(p))
    probes = probe_states(20)
    assert [a.select_greedy_action(s) for s in probes] == [b.select_greedy_action(s) for s in probes]


def test_evaluate_loader_rejects_unknown_algo(tmp_path):
    from dqn_comparison.learning import load_greedy_agent
    p = tmp_path / "x.pth"
    torch.save({"algo": "ppo"}, str(p))
    with pytest.raises(ValueError):
        load_greedy_agent(str(p))


def test_guided_probability_schedule():
    from dqn_comparison.learning import guided_probability
    assert guided_probability(1, False) == 0.0 and guided_probability(150, False) == 0.0
    assert guided_probability(1, True) == pytest.approx(0.8)
    assert guided_probability(100, True) == pytest.approx(0.1)
    assert guided_probability(200, True) == pytest.approx(0.1)


# ---------------- END NEW CODE ----------------
