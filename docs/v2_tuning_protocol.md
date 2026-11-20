# Study v2: hyperparameter tuning protocol (draft 2026-11-20, to be frozen in the pre-registration)

Purpose: give every method the same, documented chance to be well configured before the main comparison
(answering the concern that Rainbow was untuned in v1), without selecting settings on the data that is
later reported.

## Principles
1. **Separate data.** Tuning uses seeds 101-102 (and, if needed, more 1xx seeds); the main study uses seeds 1-5.
   No tuning run is ever reported as a main result.
2. **Equal budget for every algorithm.** Each of the four algorithms (DQN, DDQN, D3QN, Rainbow) gets the same
   number of candidate settings and runs. The guided Rainbow arm uses Rainbow's tuned settings (the guide is not
   a hyperparameter).
3. **Rule fixed in advance.** The selection rule below is written down before any tuning run.
4. **Same task as the main study.** Same episode budget, same E* definition (>= 3 of 4 goals, confirmed), same
   code commit; one tuning world (chosen after the diagnosis; candidate: moving stage 4).

## Search space (planned default: option a)
| Option | Per algorithm | Runs |
|---|---|---|
| **(a) learning rate in {5e-5, 1e-4, 2.5e-4}** for all four algorithms, 2 seeds | 3 settings x 2 seeds | **24** |
| (b) learning rate (3) x one second setting (2) per algorithm, 2 seeds: Rainbow sigma0 in {0.5, 0.25}; DQN/DDQN/D3QN epsilon decay length in {10k, 20k} steps | 6 x 2 | 48 |
Everything not tuned keeps the shared v2 settings (Table II of the paper, plus reward scaling x0.01 and the PER beta schedule).

## Selection rule
For each algorithm, choose the setting with the **lowest median E\*** over its tuning seeds (censored runs count as
budget + check interval). Ties: higher mean final success (20-episode final evaluation). Remaining ties: the
setting closest to the v1 value (1e-4). If no setting learns in any tuning seed, the v1 value is kept and this is
reported.

## Reporting (paper, Methods)
"Hyperparameters were tuned with an equal budget per algorithm: three learning rates (5e-5, 1e-4, 2.5e-4) on two
tuning seeds not used in the main study, in one world, with the main study's episode budget. For each algorithm the
learning rate with the lowest median E* was selected (ties: final success); all tuning runs are reported in the
supplement." (Wording to be adjusted to the option actually used.)

## Status
- [ ] Option (a) or (b) confirmed by the author (default a)
- [ ] Tuning world chosen after the diagnosis
- [ ] Frozen in the pre-registration tag before the first tuning run
