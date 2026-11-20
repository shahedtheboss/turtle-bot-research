"""Statistics helpers for the v2 analysis (moved unchanged from the v1 analyze.py, version-1 branch):
Wilson interval, IQM, bootstrap CI, Vargha-Delaney A, exact permutation tests, Holm correction.
E* lives in metrics_v2.py.
"""
import math

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def iqm(values):
    """Interquartile mean: mean of the middle 50%.

    Review fix 2: with n = 3 seeds the slice keeps all 3 values, so IQM == mean. The
    summary therefore also reports the median, and the paper must say IQM == mean at n=3.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    mid = v[lo:hi] if hi > lo else v
    return float(mid.mean())


def bootstrap_ci(values, stat=np.mean, reps=10_000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(reps)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))



def vargha_delaney_a(x, y):
    """A = P(X < Y) + 0.5 P(X == Y): probability a random X run learns SOONER than a random Y run.

    0.5 = no difference, 1.0 = X always sooner. Thresholds: 0.56 small, 0.64 medium, 0.71 large.
    """
    x, y = list(x), list(y)
    if not x or not y:
        return float("nan")
    wins = sum((xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    return wins / (len(x) * len(y))


def exact_perm_test_unpaired(x, y):
    """Exact permutation test on ranks (= exact Mann-Whitney), statistic A(x, y).

    Enumerates every way to split the pooled values into groups of len(x) and len(y)
    (5 vs 5: C(10,5) = 252 splits). Returns (A, p_one_sided, p_two_sided):
      p_one_sided = share of splits with A >= observed (H: x learns sooner),
      p_two_sided = share of splits with |A - 0.5| >= |observed - 0.5|.
    """
    from itertools import combinations
    pooled = list(x) + list(y)
    n, k = len(pooled), len(x)
    a_obs = vargha_delaney_a(x, y)
    eps = 1e-12
    ge = two = total = 0
    for idx in combinations(range(n), k):
        s = set(idx)
        a = vargha_delaney_a([pooled[i] for i in idx], [pooled[i] for i in range(n) if i not in s])
        total += 1
        ge += a >= a_obs - eps
        two += abs(a - 0.5) >= abs(a_obs - 0.5) - eps
    return a_obs, ge / total, two / total


def exact_perm_test_paired(x, y):
    """Exact paired sign-flip test on per-seed differences d = y - x (> 0: x learns sooner).

    Under H0 each difference is equally likely + or -, so all 2^n sign patterns are
    enumerated (5 pairs: 32). Statistic = mean difference.
    Returns (mean_diff, p_one_sided, p_two_sided).
    """
    from itertools import product
    d = [float(b) - float(a) for a, b in zip(x, y)]
    obs = float(np.mean(d))
    eps = 1e-12
    ge = two = total = 0
    for signs in product((1, -1), repeat=len(d)):
        m = float(np.mean([s * v for s, v in zip(signs, d)]))
        total += 1
        ge += m >= obs - eps
        two += abs(m) >= abs(obs) - eps
    return obs, ge / total, two / total


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
# ---------------- END NEW CODE ----------------
