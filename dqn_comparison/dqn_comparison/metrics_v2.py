"""Study v2 primary metric E* (decided 2026-11-19, before any main run).

E* = the training episode of the first greedy check that reaches the goal in at least THRESHOLD of the
     N_GOALS check episodes (one per goal) AND whose next check (eval_every episodes later) also does.
     Runs without such a pair are CENSORED at episodes + eval_every (v1: 200 + 20 = 220).
     Hence a run counts as learned only if E* <= episodes - eval_every (the confirming check must exist).

v2: N_GOALS = 4, THRESHOLD = 3 (75 %, the closest to v1's 4/5 = 80 %; one goal may fail, e.g. a goal that a
moving obstacle passes near). v1 used the same definition with 5 check episodes and threshold 4; with those
settings this function reproduces the v1 values (dqn_comparison/test/test_metrics_v2.py).
"""
import csv

N_GOALS = 4
THRESHOLD = 3


def e_star(checks, episodes, eval_every, threshold=THRESHOLD):
    """checks: iterable of (episode, wins), in any order. Returns (E*, censored).

    Two checks only confirm each other if they are consecutive (eval_every episodes apart); a missing
    check therefore breaks the confirmation instead of silently pairing distant checks."""
    checks = sorted((int(e), int(w)) for e, w in checks)
    for (e1, w1), (e2, w2) in zip(checks, checks[1:]):
        if e2 - e1 == eval_every and w1 >= threshold and w2 >= threshold:
            return e1, False
    return episodes + eval_every, True


def e_star_from_csv(evals_csv, episodes, eval_every, threshold=THRESHOLD):
    """E* from a run's *_evals.csv (columns 'episode' and 'wins')."""
    with open(evals_csv) as f:
        return e_star(((r['episode'], r['wins']) for r in csv.DictReader(f)), episodes, eval_every, threshold)
