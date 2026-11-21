# Study v2: pre-registration (frozen by git tag `v2-prereg-1.0`, 2026-11-21, before any main run)

Everything below is fixed before the first main run. Changes after the tag are reported as deviations.

## Design
- **Arms (5):** DQN, DDQN, D3QN, Rainbow, Rainbow + decaying scripted guide (`rainbow_g`).
- **Worlds (5):** DRL stages 2, 3, 4s (stage 4 with static obstacles), 4 (moving obstacles), 6.
- **Seeds:** 1-5 for every arm x world, so 125 runs (`scripts/main_runs_v2.csv`). Tuning seeds 101-102 are never reported as main results.
- **Task:** 4 goals per world, drawn in shuffled blocks from a per-seed RNG (identical goal sequence for every arm with the same seed); same obstacle phase every episode; 52-value state, 7 actions.
- **Budget:** 500 training episodes per run, no early stopping; greedy check (one episode per goal) every 20 episodes.
- **Hyperparameters:** shared v1 settings, except the learning rate tuned per algorithm (`docs/v2_tuning_protocol.md`): DQN 2.5e-4, DDQN 2.5e-4, D3QN 1e-4, Rainbow 1e-4 (also `rainbow_g`).
- **Simulation:** simulated-time stepping, real-time factor cap 3, /clock 200 Hz, 5 ms poll.

## Metrics
- **Primary:** E* = first greedy check with >= 3 of 4 goals reached, confirmed by the next check; runs that never meet it are censored at 520.
- **Secondary:** final success rate (final checkpoint, 5 poses x 4 goals x 2 repeats = 40 episodes); collision rate in that evaluation.

## Analysis plan (statistics code: `stats.py`, `metrics_v2.py`)
- Per world: median and IQM of E* per arm over the 5 seeds; final success with Wilson intervals.
- Comparisons: pairwise exact permutation tests on E* between arms within each world (paired by seed, since seeds share goal sequences), Holm correction over the comparisons of a world; effect size Vargha-Delaney A.
- Across worlds: arm ranks per world and the median of each arm's E* over worlds; no pooled p-value across worlds.
- Censored runs enter the tests with the censoring value 520 (rank-based effect sizes are unaffected by its exact value).
- With 5 seeds the tests have low power; non-significant differences are reported as inconclusive, not as equivalence.

## Machines
- Planned split: lab PC seeds 1-3 (75 runs), laptop seeds 4-5 (50 runs). Whole seeds per machine, so each arm and world is equally affected by machine differences.
- Fallback: if a machine is not available long enough, the other runs the remaining runs (`scripts/run_main_v2.sh <machine> --take-over`). The machine of every run is recorded (hostname in `run_meta.json`) and reported; a machine effect is checked descriptively.
- A run that fails for a technical reason (crash, Gazebo not starting) is rerun with the same settings and reported.
