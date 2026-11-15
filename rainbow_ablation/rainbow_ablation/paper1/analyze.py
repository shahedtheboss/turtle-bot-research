"""Paper 1 analysis: tables and figures from results_paper1/ (item 5.6).

    python3 -m rainbow_ablation.paper1.analyze --results-dir results_paper1

Reads every run folder <arm>_s<seed>/ (run_meta.json, *_episodes.csv, *_evals.csv and,
if present, *_final_eval.csv from evaluate.py) plus optional H/ and H_eta/ baselines.
Writes <results-dir>/summary/: summary_runs.csv, summary_configs.csv, per_pose.csv,
summary.md and figures (learning curves vs episode and vs env steps, final success,
overestimation, initial action collapse).

Statistics (Agarwal et al. 2021; Colas et al. 2019):
- per-config final success: mean, IQM (interquartile mean) across seeds, and a 95%
  percentile bootstrap interval over seeds (10,000 resamples). With 3 seeds these
  intervals are wide: report them, do not hide them.
- pooled Wilson 95% interval over all final-evaluation episodes of a config.
- learning speed: AUC of the in-training greedy success curve (mean over checks) and
  E* = first check with >= 4/5 successes confirmed at the next check (censored = episodes+20).
"""

import argparse
import csv
import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["dqn", "dqn_g", "ddqn", "ddqn_g", "d3qn", "d3qn_g", "rainbow", "rainbow_g"]   # review fix 9
_BASE = {"dqn": "DQN", "ddqn": "DDQN", "d3qn": "D3QN", "rainbow": "Rainbow"}
LABEL = {**_BASE, **{k + "_g": v + " + guide" for k, v in _BASE.items()},
         "H": "Heuristic H", "H_eta": "Heuristic H (15% noise)"}


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


def e_star(evals, censor):
    """First check with >= 4/5 wins whose NEXT check also has >= 4/5 (same rule as Paper 2)."""
    ok = [(int(r["episode"]), int(r["wins"]) >= 4) for r in evals]
    for i in range(len(ok) - 1):
        if ok[i][1] and ok[i + 1][1]:
            return ok[i][0], False
    return censor, True


def load_runs(results_dir):
    runs = []
    for meta_path in sorted(glob.glob(os.path.join(results_dir, "*", "*_run_meta.json"))):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("study") != "paper1":
            continue
        d, name = os.path.dirname(meta_path), meta["run_name"]
        runs.append({
            "meta": meta, "name": name, "arm": meta["arm"], "seed": meta["seed"],
            "episodes": read_csv(os.path.join(d, f"{name}_episodes.csv")),
            "evals": read_csv(os.path.join(d, f"{name}_evals.csv")),
            "final": read_csv(os.path.join(d, f"{name}_final_eval.csv")),
        })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_paper1")
    args = ap.parse_args()
    out = os.path.join(args.results_dir, "summary")
    os.makedirs(out, exist_ok=True)
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No Paper 1 runs found in {args.results_dir}")

    # ---------- per-run summary ----------
    run_rows = []
    for r in runs:
        m, eps = r["meta"], r["episodes"]
        censor = int(m["episodes"]) + int(m.get("eval_every", 20))
        est, censored = e_star(r["evals"], censor)
        fin = r["final"]
        k = sum(x["outcome"] == "goal_reached" for x in fin)
        rates = [float(e["success_rate"]) for e in r["evals"]]
        overest = [float(e["overestimation_mean"]) for e in r["evals"]
                   if e.get("overestimation_mean") not in ("", "nan", None)]
        run_rows.append({
            "run": r["name"], "arm": r["arm"], "seed": r["seed"], "status": m.get("status"),
            "episodes_completed": m.get("episodes_completed"),
            "env_steps_total": m.get("env_steps_total"),
            "train_goals": sum(e["outcome"] == "goal_reached" for e in eps),
            "train_collisions": sum(e["outcome"] == "collision_warning" for e in eps),
            "eval_auc": round(float(np.mean(rates)), 4) if rates else "",
            "E_star": est, "E_star_censored": censored,
            "final_k": k, "final_n": len(fin),
            "final_success": round(k / len(fin), 4) if fin else "",
            "final_mean_steps_success": round(float(np.mean(
                [int(x["steps"]) for x in fin if x["outcome"] == "goal_reached"])), 2)
                if k else "",
            "overestimation_last": overest[-1] if overest else "",
            "initial_greedy_top_share": m.get("initial_greedy_top_share"),
            "rtf_mean": m.get("rtf_mean"), "rtf_min": m.get("rtf_min"),
            "wall_time_h": m.get("wall_time_h"), "git_dirty": m.get("git_dirty"),
        })
    with open(os.path.join(out, "summary_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader(); w.writerows(run_rows)

    # ---------- per-config summary ----------
    arms = [a for a in ORDER if any(r["arm"] == a for r in runs)]
    cfg_rows, pose_rows = [], []
    for a in arms:
        rs = [r for r in run_rows if r["arm"] == a]
        fin_rates = [r["final_success"] for r in rs if r["final_success"] != ""]
        k = sum(r["final_k"] for r in rs); n = sum(r["final_n"] for r in rs)
        lo, hi = wilson(k, n)
        blo, bhi = bootstrap_ci(fin_rates)
        aucs = [r["eval_auc"] for r in rs if r["eval_auc"] != ""]
        cfg_rows.append({
            "config": LABEL[a], "arm": a, "seeds": len(rs),
            "final_success_mean": round(float(np.mean(fin_rates)), 3) if fin_rates else "",
            "final_success_iqm": round(iqm(fin_rates), 3) if fin_rates else "",
            "final_success_median": round(float(np.median(fin_rates)), 3) if fin_rates else "",
            "runs_missing_final_eval": len(rs) - len(fin_rates),     # review fix 9
            "bootstrap95_lo": round(blo, 3), "bootstrap95_hi": round(bhi, 3),
            "pooled_k": k, "pooled_n": n,
            "wilson95_lo": round(lo, 3), "wilson95_hi": round(hi, 3),
            "eval_auc_mean": round(float(np.mean(aucs)), 3) if aucs else "",
            "E_star_median": float(np.median([r["E_star"] for r in rs])),
            "E_star_censored_runs": sum(bool(r["E_star_censored"]) for r in rs),
            "train_goals_mean": round(float(np.mean([r["train_goals"] for r in rs])), 1),
            "initial_greedy_top_share_mean": round(float(np.mean(
                [r["initial_greedy_top_share"] for r in rs if r["initial_greedy_top_share"] is not None]
                or [float("nan")])), 3),
        })
        # per-pose success (pooled over seeds)
        fins = [x for r in runs if r["arm"] == a for x in r["final"]]
        for pi in sorted({int(x["pose_index"]) for x in fins}):
            g = [x for x in fins if int(x["pose_index"]) == pi]
            kk = sum(x["outcome"] == "goal_reached" for x in g)
            pose_rows.append({"config": LABEL[a], "pose": pi, "k": kk, "n": len(g),
                              "success": round(kk / len(g), 3)})
    # heuristic baselines (evaluation only)
    for b in ("H", "H_eta"):
        fin = read_csv(os.path.join(args.results_dir, b, f"{b}_final_eval.csv"))
        if fin:
            k = sum(x["outcome"] == "goal_reached" for x in fin); n = len(fin)
            lo, hi = wilson(k, n)
            cfg_rows.append({"config": LABEL[b], "arm": b, "seeds": 1,
                             "final_success_mean": round(k / n, 3), "pooled_k": k, "pooled_n": n,
                             "wilson95_lo": round(lo, 3), "wilson95_hi": round(hi, 3)})
            for pi in sorted({int(x["pose_index"]) for x in fin}):
                g = [x for x in fin if int(x["pose_index"]) == pi]
                kk = sum(x["outcome"] == "goal_reached" for x in g)
                pose_rows.append({"config": LABEL[b], "pose": pi, "k": kk, "n": len(g),
                                  "success": round(kk / len(g), 3)})

    fields = sorted({k for r in cfg_rows for k in r}, key=lambda k: list(cfg_rows[0]).index(k)
                    if k in cfg_rows[0] else 99)
    with open(os.path.join(out, "summary_configs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(cfg_rows)
    if pose_rows:
        with open(os.path.join(out, "per_pose.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pose_rows[0])); w.writeheader(); w.writerows(pose_rows)

    # ---------- figures ----------
    colors = dict(zip(ORDER, plt.cm.tab10.colors))
    for xkey, fname, xlabel in (("episode", "curves_vs_episode.png", "Training episode"),
                                ("env_steps_total", "curves_vs_env_steps.png", "Environment steps")):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for a in arms:
            rs = [r for r in runs if r["arm"] == a and r["evals"]]
            if not rs:
                continue
            L = min(len(r["evals"]) for r in rs)
            ys = np.array([[float(e["success_rate"]) for e in r["evals"][:L]] for r in rs])
            if xkey == "episode":
                xs = [float(e["episode"]) for e in rs[0]["evals"][:L]]   # same check episodes for all seeds
                ax.plot(xs, ys.mean(axis=0), marker="o", color=colors[a], label=f"{LABEL[a]} (n={len(rs)})")
                ax.fill_between(xs, ys.min(axis=0), ys.max(axis=0), color=colors[a], alpha=0.12)
            else:
                # review fix 9: env steps differ per seed, so draw each seed (no x-averaging)
                for j, r in enumerate(rs):
                    ax.plot([float(e[xkey]) for e in r["evals"][:L]], ys[j], marker=".", color=colors[a],
                            alpha=0.7, label=f"{LABEL[a]} (n={len(rs)})" if j == 0 else None)
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel(xlabel); ax.set_ylabel("Greedy success (5 tests)")
        ax.set_title("In-training greedy success: mean over seeds, band = min-max")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    shown = [r for r in cfg_rows if r.get("pooled_n")]
    xs = np.arange(len(shown))
    for i, r in enumerate(shown):
        p = r["pooled_k"] / r["pooled_n"]
        ax.bar(i, p, color=colors.get(r["arm"], "grey"))
        ax.errorbar(i, p, yerr=[[p - r["wilson95_lo"]], [r["wilson95_hi"] - p]], color="k", capsize=4)
        seeds = [x["final_success"] for x in run_rows if x["arm"] == r["arm"] and x["final_success"] != ""]
        ax.scatter([i] * len(seeds), seeds, color="k", s=12, zorder=3)
    ax.set_xticks(xs, [r["config"] for r in shown], rotation=20)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Final greedy success (5 poses x 10)")
    ax.set_title("Final success: bar = pooled, whiskers = Wilson 95%, dots = seeds")
    ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "final_success.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for a in arms:
        rs = [r for r in runs if r["arm"] == a and r["evals"]]
        if not rs:
            continue
        L = min(len(r["evals"]) for r in rs)
        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")
        ys = np.array([[_f(e.get("overestimation_mean")) for e in r["evals"][:L]] for r in rs])
        xs = [int(e["episode"]) for e in rs[0]["evals"][:L]]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # review fix 9: all-NaN columns
                ax.plot(xs, np.nanmean(ys, axis=0), marker="o", color=colors[a], label=LABEL[a])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Training episode"); ax.set_ylabel("max Q(s0) - discounted return")
    ax.set_title("Overestimation at s0 (greedy checks ending in goal/collision only)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, "overestimation.png"), dpi=200); plt.close(fig)

    # ---------- markdown summary ----------
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("# Paper 1 summary (auto-generated by analyze.py)\n\n")
        f.write("| Config | Seeds | Final success mean | Median | IQM | Bootstrap 95% (seeds) | Pooled k/n | Wilson 95% | Eval AUC | E* median (censored) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in cfg_rows:
            f.write(f"| {r['config']} | {r.get('seeds','')} | {r.get('final_success_mean','')} | "
                    f"{r.get('final_success_median','')} | {r.get('final_success_iqm','')} | [{r.get('bootstrap95_lo','')}, {r.get('bootstrap95_hi','')}] | "
                    f"{r.get('pooled_k','')}/{r.get('pooled_n','')} | [{r.get('wilson95_lo','')}, {r.get('wilson95_hi','')}] | "
                    f"{r.get('eval_auc_mean','')} | {r.get('E_star_median','')} ({r.get('E_star_censored_runs','')}) |\n")
        f.write("\nNotes: with 3 seeds IQM equals the mean and the seed bootstrap has at most 10 distinct "
                "resamples; pooled Wilson intervals ignore seed clustering (descriptive only). "
                "Same seed => same initial weights for DQN/DDQN (paired design).\n")
        miss = [r["config"] for r in cfg_rows if r.get("runs_missing_final_eval")]
        f.write("\nConfigs with runs missing a final evaluation: " + (", ".join(miss) if miss else "none") + "\n")
        f.write("\nRuns with git_dirty=True or status != completed: ")
        bad = [r["run"] for r in run_rows if r["git_dirty"] or r["status"] != "completed"]
        f.write(", ".join(bad) if bad else "none")
        f.write("\n")
    print("Wrote", out)


if __name__ == "__main__":
    main()
