"""Paper 1 figures, tables and sensitivity analyses (analysis only; added 2026-11-17).

    python3 -m dqn_comparison.paper_figures --results-dir results --out reports/paper

Reads the same run folders as analyze.py (reuses its loaders and statistics, does NOT change
them) and writes, under --out:
  figures/*.pdf (+ .png)  : publication figures (IEEE column width)
  tables/*.tex            : LaTeX table bodies (\\input into the paper)
  tables/*.csv            : the same numbers as CSV
  numbers.md              : every number quoted in the paper text, in one place

Analyses beyond analyze.py (all post hoc, labelled as such in the paper):
  B2 budget truncation : E* and the planned contrasts recomputed as if the budget were
                         100 / 150 / 200 episodes (uses only checks inside the budget).
  B3 sample efficiency : E* expressed in environment steps and in simulated seconds
                         instead of episodes (guided episodes are shorter).
  C  failure analysis  : end positions of failed final-evaluation episodes on the map;
                         failure modes per arm and pose; mechanism (initial greedy action
                         collapse, share of actions not chosen by the greedy policy).
"""

import argparse
import warnings
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from dqn_comparison.analyze import (LABEL, bootstrap_ci, exact_perm_test_paired,
                                    exact_perm_test_unpaired, holm, iqm, load_runs, read_csv,
                                    vargha_delaney_a, wilson)

ARMS = ["dqn", "ddqn", "d3qn", "d3qn_g", "rainbow", "rainbow_g"]
CHECK_EVERY = 20
POSES = [(0.0, 0.0, 0.0), (-1.5, -0.8, 0.0), (0.0, 1.0, -np.pi / 2), (1.8, -0.2, -np.pi / 2),
         (0.3, -1.8, 0.0)]
GOAL = (2.0, -2.0)
WORLD = "turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_drl_world"
SHORT = {"dqn": "DQN", "ddqn": "DDQN", "d3qn": "D3QN", "d3qn_g": "D3QN\n+G",
         "rainbow": "Rainbow", "rainbow_g": "Rainbow\n+G", "H": "H", "H_eta": "H$_\\eta$"}
OUTCOME_SHORT = {"goal_reached": "goal", "collision_warning": "collision",
                 "no_progress": "no progress", "timeout": "timeout"}

# ---------------- CHANGED (2026-11-17): one visual system for every paper figure, following the
# dataviz skill (palette validated with its validate_palette.js, light mode):
#   algorithms  blue/orange/aqua/yellow (all hard checks PASS; aqua/yellow < 3:1 contrast ->
#               relief by direct labels / axis labels / tables);
#   outcomes    red/violet/green (PASS; red-green in the CVD warn band -> always paired with
#               distinct marker shapes);
#   guide       secondary encoding only (dashed line, hollow marker, lighter bar) - the
#               colour always identifies the algorithm;
#   chrome      hairline solid grid, no dotted rules, text in neutral ink, never series colour.
SURFACE, INK, INK2, GRID, REF = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
HUE = {"dqn": "#2a78d6", "ddqn": "#eb6834", "d3qn": "#1baf7a", "rainbow": "#eda100"}
COLOR = {a: HUE[a.replace("_g", "")] for a in ARMS}
COLOR.update({"H": INK2, "H_eta": REF})
DASH = (0, (3.5, 2))
LS = {a: (DASH if a.endswith("_g") else "-") for a in ARMS}
OUTCOME_STYLE = {"collision_warning": ("x", "#e34948"), "no_progress": ("s", "#6250d6"),
                 "timeout": ("^", "#008300")}
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # skill blue 100->700
# ---------------- END CHANGED

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
COL_W, TXT_W = 3.5, 7.16   # IEEE single column / full text width in inches


# ---------------------------------------------------------------- helpers
def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(out, name, header, rows, tex_rows=None):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    with open(os.path.join(out, "tables", f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(os.path.join(out, "tables", f"{name}.tex"), "w") as f:
        for r in (tex_rows if tex_rows is not None else rows):
            f.write(" & ".join(str(c) for c in r) + r" \\" + "\n")


def e_star_budget(evals, budget):
    """E* using only checks at episode <= budget; censored = budget + 20 (same rule as analyze.e_star)."""
    ok = [(int(r["episode"]), int(r["wins"]) >= 4) for r in evals if int(r["episode"]) <= budget]
    for i in range(len(ok) - 1):
        if ok[i][1] and ok[i + 1][1]:
            return ok[i][0], False
    return budget + CHECK_EVERY, True


def cum_at_episode(episodes, ep, col):
    """Cumulative value of a per-episode column up to and including episode ep."""
    return float(sum(float(r[col]) for r in episodes if int(r["episode"]) <= ep))


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ---------------- NEW CODE ADDED (2026-11-17, review item 5): cluster bootstrap
# A greedy policy started from the same pose is nearly deterministic (117 of 150 seed x pose
# cells give the same outcome in all 10 episodes), so the 250 episodes of a configuration are not
# independent. Resample seeds, then poses within each resampled seed (two-stage cluster bootstrap).
def pose_cells(final_rows):
    by = {}
    for e in final_rows:
        by.setdefault(int(e["pose_index"]), []).append(e["outcome"] == "goal_reached")
    return np.array([np.mean(by[p]) for p in sorted(by)])


def cluster_boot_ci(C, reps=10_000, seed=0):
    """C: seeds x poses matrix of per-cell success rates -> 95% percentile interval of the mean."""
    rng = np.random.default_rng(seed)
    n_s, n_p = C.shape
    B = np.empty(reps)
    for i in range(reps):
        S = C[rng.integers(0, n_s, n_s)]
        B[i] = np.mean([row[rng.integers(0, n_p, n_p)].mean() for row in S])
    return float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))


def pfmt(p):
    """p-values: exact to 4 decimals, trailing zeros dropped (0.004, 0.0625, 0.1875, 1)."""
    return f"{p:.4f}".rstrip("0").rstrip(".")
# ---------------- END NEW CODE


def by_arm(runs):
    return {a: sorted([r for r in runs if r["arm"] == a], key=lambda r: r["seed"]) for a in ARMS}


# ---------------- NEW CODE ADDED (2026-11-17): shared styling helpers (dataviz-skill system)
def style_ax(ax):
    """Recessive chrome: hairline solid y-grid, no box, no tick marks, neutral tick text."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=INK2)


def line(ax, x, y, arm):
    """Series line: algorithm colour; guided arms dashed with hollow markers."""
    g = arm.endswith("_g")
    ax.plot(x, y, color=COLOR[arm], lw=1.5, ls=LS[arm], solid_capstyle="round",
            dash_capstyle="round", zorder=2)
    ax.plot(x, y, ls="none", marker="o", ms=3.6, mfc=SURFACE if g else COLOR[arm],
            mec=COLOR[arm], mew=1.0, zorder=3)


def dot(ax, x, y, arm):
    g = arm.endswith("_g")
    ax.plot(x, y, "o", ms=4.2, mfc=SURFACE if g else COLOR[arm], mec=COLOR[arm] if g else SURFACE,
            mew=1.0 if g else 0.6, zorder=3)


def end_labels(ax, ends, x_end, min_gap=0.075):
    """Direct labels right of the plot; spread apart, with leader lines when ends collide."""
    ends = sorted(ends, key=lambda t: t[1])
    ys = []
    for _, y, _ in ends:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    shift = max(0.0, ys[-1] - 1.04)
    ys = [y - shift for y in ys]
    for (text, y, color), yl in zip(ends, ys):
        ax.plot([x_end + 2, x_end + 9], [y, yl], color=GRID, lw=0.6, clip_on=False, zorder=1)
        ax.plot(x_end + 12, yl, marker="o", ms=3.2, color=color, mec=SURFACE, mew=0.6, clip_on=False, zorder=3)
        ax.text(x_end + 16, yl, text, color=INK, fontsize=7, va="center", ha="left", clip_on=False)


def algo_legend(fig, y=1.04, markers=True):
    """One shared legend: colour = algorithm, line style = guide."""
    m = dict(marker="o", ms=3.6) if markers else {}
    hs = [Line2D([], [], color=HUE[k], lw=1.5, mec=HUE[k], label=LABEL[k], **m) for k in HUE]
    hs += [Line2D([], [], color=INK2, lw=1.2, ls="-", label="unguided"),
           Line2D([], [], color=INK2, lw=1.2, ls=DASH, mfc=SURFACE, mec=INK2, label="+ guide", **m)]
    leg = fig.legend(handles=hs, loc="upper center", ncol=6, frameon=False, fontsize=7,
                     bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK)
# ---------------- END NEW CODE


# ---------------------------------------------------------------- world geometry
def wall_boxes():
    """(x, y, yaw, length, width) of every wall link, parsed from the model SDFs."""
    import xml.etree.ElementTree as ET
    boxes = []
    for m in ("outer_walls", "inner_walls"):
        root = ET.parse(os.path.join(WORLD, m, "model.sdf")).getroot()
        for link in root.iter("link"):
            p = [float(v) for v in link.find("pose").text.split()]
            s = [float(v) for v in link.find("collision/geometry/box/size").text.split()]
            boxes.append((link.get("name"), p[0], p[1], p[5], s[0], s[1]))
    return boxes


def cylinder_path(n, spawn):
    """Key-frame positions of a moving cylinder (offsets from its spawn pose in the world file)."""
    txt = open(os.path.join(WORLD, "obstacle_plugin", f"obstacle{n}.cc")).read()
    pts = re.findall(r"Vector3d\(\s*([-\d.]+),\s*([-\d.]+),", txt)
    return [(spawn[0] + float(x), spawn[1] + float(y)) for x, y in pts]


# CHANGED (2026-11-17, review): maps are drawn in the same orientation as the Gazebo screenshot
# (Fig. 1): up = +x, right = -y. Horizontal axis = y (inverted), vertical axis = x.
SWAP = Affine2D(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def draw_world(ax, boxes, paths, poses=True, legend=False):
    for _, x, y, yaw, L, W in boxes:
        r = Rectangle((-L / 2, -W / 2), L, W, color=INK2, lw=0)
        r.set_transform(Affine2D().rotate(yaw).translate(x, y) + SWAP + ax.transData)
        ax.add_patch(r)
    # CHANGED (2026-11-17): the cylinders were STATIC in every run (Gazebo could not load
    # libobstacle1/2.so; verified live and in ~/.gazebo/server-*/default.log), so draw them
    # at their spawn positions only, not along the plugin key-frame paths.
    for i, path in enumerate(paths):
        ax.add_patch(Circle((path[0][1], path[0][0]), 0.16, fc=REF, ec=REF, lw=0.8,
                            label="static cylinders" if i == 0 else None))
    ax.plot(GOAL[1], GOAL[0], marker="*", ms=9, color=INK, ls="none", label="goal")
    ax.add_patch(Circle((GOAL[1], GOAL[0]), 0.20, fc="none", ec=INK, lw=0.6))
    if poses:
        for i, (x, y, yaw) in enumerate(POSES):
            ax.annotate("", xy=(y + 0.3 * np.sin(yaw), x + 0.3 * np.cos(yaw)), xytext=(y, x),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
            ax.plot(y, x, "o", color=INK2, ms=3)
            ax.text(y - 0.08, x + 0.1, f"P{i}", color=INK, fontsize=7)
    ax.set_xlim(2.6, -2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])


def nearest_wall(x, y, boxes):
    """Name and distance (m) of the nearest wall surface to point (x, y)."""
    best = (None, 1e9)
    for name, bx, by, yaw, L, W in boxes:
        c, s = np.cos(-yaw), np.sin(-yaw)
        lx, ly = c * (x - bx) - s * (y - by), s * (x - bx) + c * (y - by)
        dx, dy = max(abs(lx) - L / 2, 0), max(abs(ly) - W / 2, 0)
        d = float(np.hypot(dx, dy))
        if d < best[1]:
            best = (name, d)
    return best


# ---------------- NEW CODE ADDED (2026-11-17): cylinder-aware obstacle attribution
# The cylinders were static in every run (plugins never loaded), so a collision can be
# attributed to the nearest obstacle surface - wall box or cylinder (r = 0.16 m) - from the
# episode's end position alone. Wall names become readable centre coordinates.
CYL_R = 0.16


def nearest_obstacle(x, y, boxes, cylinders):
    """(label, kind, distance in m) of the nearest obstacle surface to the end point."""
    name, d = nearest_wall(x, y, boxes)
    wall = next(b for b in boxes if b[0] == name)
    best = (f"wall ({wall[1]:g}, {wall[2]:g})" if name.startswith("inner") else "outer wall", "wall", d)
    for cx, cy in cylinders:
        dc = max(float(np.hypot(x - cx, y - cy)) - CYL_R, 0.0)
        if dc < best[2]:
            best = (f"cylinder ({cx:g}, {cy:g})", "cylinder", dc)
    return best
# ---------------- END NEW CODE


# ---------------------------------------------------------------- main
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of all-NaN columns
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="reports/paper")
    args = ap.parse_args()
    out = args.out
    runs = [r for r in load_runs(args.results_dir) if r["meta"].get("status") == "completed"]
    A = by_arm(runs)
    H = {h: read_csv(os.path.join(args.results_dir, h, f"{h}_final_eval.csv")) for h in ("H", "H_eta")}
    budget = int(runs[0]["meta"]["episodes"])
    boxes = wall_boxes()
    paths = [cylinder_path(1, (2.0, 2.0)), cylinder_path(2, (-2.0, -2.0))]
    N = []   # lines for numbers.md

    # ---------- per-run quantities
    for r in runs:
        r["E"], r["cens"] = e_star_budget(r["evals"], budget)
        ev = {int(e["episode"]): e for e in r["evals"]}
        last = r["episodes"][-1]
        if r["cens"]:
            r["E_steps"] = float(last["env_steps_total"])
            r["E_sim"] = cum_at_episode(r["episodes"], budget, "sim_time_s")
        else:
            r["E_steps"] = float(ev[r["E"]]["env_steps_total"])
            r["E_sim"] = cum_at_episode(r["episodes"], r["E"], "sim_time_s")
        f = r["final"]
        r["succ"] = sum(e["outcome"] == "goal_reached" for e in f) / len(f)
        r["coll"] = sum(e["outcome"] == "collision_warning" for e in f) / len(f)
        r["auc"] = float(np.mean([float(e["success_rate"]) for e in r["evals"]]))
        r["train_goals"] = sum(e["outcome"] == "goal_reached" for e in r["episodes"])
        steps = sum(int(e["steps"]) for e in r["episodes"])
        r["guided_share"] = sum(int(e["guided_steps"]) for e in r["episodes"]) / steps
        r["random_share"] = sum(int(e["random_steps"]) for e in r["episodes"]) / steps
        r["env_steps"] = int(last["env_steps_total"])
        r["mean_len"] = steps / len(r["episodes"])

    # ---------------- CHANGED (2026-11-17): all figures restyled with the dataviz-skill system
    # (see the constants block at the top). Data, statistics and file names are unchanged.

    # ================================================================ Fig: world + poses (schematic, not in the paper)
    fig, ax = plt.subplots(figsize=(COL_W * 0.9, COL_W * 0.9))
    draw_world(ax, boxes, paths)
    ax.set_xlabel("y (m), increasing to the left"); ax.set_ylabel("x (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=6)
    save(fig, out, "fig_world_schematic")   # fig_world.* = annotated Gazebo screenshot (reports/paper/make_world_figure.py)

    # ================================================================ Fig: learning curves
    fig, axes = plt.subplots(1, 2, figsize=(TXT_W, 2.55), sharey=True)
    for ax, title, group in ((axes[0], "(a) Unguided algorithms", ["dqn", "ddqn", "d3qn", "rainbow"]),
                             (axes[1], "(b) Effect of the guide", ["d3qn", "d3qn_g", "rainbow", "rainbow_g"])):
        style_ax(ax)
        ax.axhline(0.8, color=REF, lw=0.7, zorder=1)
        ax.text(4, 0.815, "4/5 threshold for $E^*$", color=INK2, fontsize=6.5, va="bottom")
        ends = []
        for a in group:
            eps = np.array([int(e["episode"]) for e in A[a][0]["evals"]])
            M = np.array([[float(e["success_rate"]) for e in r["evals"]] for r in A[a]])
            ax.fill_between(eps, M.min(0), M.max(0), color=COLOR[a], alpha=0.10, lw=0)
            line(ax, eps, M.mean(0), a)
            ends.append((LABEL[a], M.mean(0)[-1], COLOR[a]))
        end_labels(ax, ends, eps[-1])
        ax.set_title(title, loc="left", color=INK, pad=4); ax.set_xlabel("Training episode")
        ax.set_xlim(0, budget); ax.set_xticks(range(0, budget + 1, 40))
        ax.set_ylim(-0.03, 1.06); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("Greedy success\n(5 episodes from $P_0$)")
    algo_legend(fig, y=1.04)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93), w_pad=4.5)
    save(fig, out, "fig_curves")

    # ================================================================ Fig: E* per seed (episodes and env steps)
    fig, axes = plt.subplots(1, 2, figsize=(TXT_W, 2.2))
    for ax, key, lab, cens_val in ((axes[0], "E", "$E^*$ (training episodes)", budget + CHECK_EVERY),
                                   (axes[1], "E_steps", "$E^*$ (environment steps)", None)):
        style_ax(ax)
        if cens_val:
            ax.axhline(cens_val, color=REF, lw=0.7, zorder=1)
            ax.text(len(ARMS) - 0.45, cens_val + 3, "not learned (censored)", ha="right", va="bottom",
                    fontsize=6, color=INK2)
        for i, a in enumerate(ARMS):
            vals = [r[key] for r in A[a]]
            jit = np.linspace(-0.2, 0.2, len(vals))
            for j, r in enumerate(A[a]):
                if r["cens"]:
                    ax.plot(i + jit[j], r[key], "x", color=COLOR[a], ms=4.5, mew=1.2, zorder=3)
                else:
                    dot(ax, i + jit[j], r[key], a)
            ax.plot([i - 0.32, i + 0.32], [np.median(vals)] * 2, color=INK, lw=1.2, zorder=2)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([SHORT[a] for a in ARMS], fontsize=6.5, color=INK)
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylabel(lab)
    axes[0].set_title("(a) Learning speed in episodes (lower = faster)", loc="left", color=INK)
    axes[1].set_title("(b) Learning speed in experience ($\\times$: not learned, total steps)", loc="left", color=INK)
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "fig_estar")

    # ================================================================ Fig: final success
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    style_ax(ax)
    hp = {}
    for h in ("H", "H_eta"):
        hp[h] = sum(e["outcome"] == "goal_reached" for e in H[h]) / len(H[h])
        ax.axhline(hp[h], color=COLOR[h], lw=0.8, zorder=1)
    # labels in the right margin, outside the bars (they collided with Rainbow+G inside the plot)
    ax.text(len(ARMS) - 0.3, hp["H"] + 0.01, f"H {hp['H']:.2f}", ha="left", va="bottom",
            fontsize=6, color=INK2, clip_on=False)
    ax.text(len(ARMS) - 0.3, hp["H_eta"] - 0.01, f"H$_\\eta$ {hp['H_eta']:.2f}", ha="left",
            va="top", fontsize=6, color=INK2, clip_on=False)
    for i, a in enumerate(ARMS):
        k = sum(int(r["succ"] * 50 + 0.5) for r in A[a]); n = 50 * len(A[a])
        lo, hi = cluster_boot_ci(np.array([pose_cells(r["final"]) for r in A[a]]))   # was wilson(k, n)
        ax.bar(i, k / n, width=0.5, color=COLOR[a], alpha=0.35 if a.endswith("_g") else 0.95, lw=0, zorder=2)
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color=INK, capsize=2, lw=0.8, zorder=4)
        ax.plot(i + np.linspace(-0.16, 0.16, len(A[a])), [r["succ"] for r in A[a]], "o", ms=2.6,
                color=INK2, mec=SURFACE, mew=0.5, zorder=5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([SHORT[a] for a in ARMS], fontsize=6.5, color=INK)
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylabel("Final success\n(50 episodes, 5 poses)"); ax.set_ylim(0, 1.05)
    save(fig, out, "fig_final_success")

    # ================================================================ Fig: per-pose heatmap
    labels = ARMS + ["H", "H_eta"]
    M = np.zeros((len(labels), 5))
    for i, a in enumerate(labels):
        eps = H[a] if a in H else [e for r in A[a] for e in r["final"]]
        for p in range(5):
            sel = [e for e in eps if int(e["pose_index"]) == p]
            M[i, p] = sum(e["outcome"] == "goal_reached" for e in sel) / len(sel)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    cmap = LinearSegmentedColormap.from_list("skill_blue", SEQ_RAMP)
    im = ax.pcolormesh(M, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
    ax.invert_yaxis()
    for i in range(M.shape[0]):
        for j in range(5):
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color=SURFACE if M[i, j] > 0.55 else INK)
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels([f"P{j}" + ("\n(train)" if j == 0 else "") for j in range(5)], color=INK)
    ax.set_yticks(np.arange(len(labels)) + 0.5); ax.set_yticklabels([LABEL[a] for a in labels], color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6)
    cb.set_label("success rate", color=INK2)
    save(fig, out, "fig_per_pose")

    # ================================================================ Fig: failure map (end positions)
    fig, axes = plt.subplots(2, 4, figsize=(TXT_W, 3.9))
    panels = ARMS + ["H", "H_eta"]
    fail_rows = []
    for ax, a in zip(axes.flat, panels):
        draw_world(ax, boxes, paths, poses=False)
        eps = H[a] if a in H else [e for r in A[a] for e in r["final"]]
        n_tot = len(eps)
        for oc, (m, c) in OUTCOME_STYLE.items():
            sel = [e for e in eps if e["outcome"] == oc]
            ax.plot([float(e["end_y"]) for e in sel], [float(e["end_x"]) for e in sel], m, color=c,
                    ms=3.2, mew=0.8, mfc=c if m == "x" else "none", alpha=0.85, zorder=4,
                    label=OUTCOME_SHORT[oc] if a == panels[0] else None)
        for i, (x, y, _) in enumerate(POSES):
            ax.plot(y, x, "o", color=INK2, ms=2.4, mec=SURFACE, mew=0.4, zorder=3,
                    label="start poses" if (a == panels[0] and i == 0) else None)
        nf = sum(e["outcome"] != "goal_reached" for e in eps)
        ax.set_title(f"{SHORT[a].replace(chr(10), ' ')}: {nf}/{n_tot} failed", fontsize=6.5, color=INK)
        ax.tick_params(labelsize=6, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for e in eps:
            if e["outcome"] == "goal_reached":
                continue
            name, kind, d = nearest_obstacle(float(e["end_x"]), float(e["end_y"]), boxes,
                                             [p[0] for p in paths])
            fail_rows.append((a, int(e["pose_index"]), e["outcome"], name, d, kind))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False,
               fontsize=7)
    fig.subplots_adjust(bottom=0.1)
    save(fig, out, "fig_failure_map")

    # ================================================================ Fig: mechanism
    fig, axes = plt.subplots(1, 3, figsize=(TXT_W, 2.25))
    # (a) initial greedy action collapse: share of 300 probe states that get the most common action
    style_ax(axes[0])
    axes[0].axhline(1 / 7, color=REF, lw=0.7, zorder=1)
    axes[0].text(-0.45, 1 / 7 + 0.025, "uniform (1/7)", fontsize=6, color=INK2)
    for i, a in enumerate(ARMS):
        v = [r["meta"]["initial_greedy_top_share"] for r in A[a]]
        for x, y in zip(i + np.linspace(-0.18, 0.18, len(v)), v):
            dot(axes[0], x, y, a)
    axes[0].set_xticks(range(len(ARMS)))
    axes[0].set_xticklabels([SHORT[a].replace("\n", "") for a in ARMS], fontsize=6, color=INK,
                            rotation=40, ha="right", rotation_mode="anchor")
    axes[0].set_xlim(-0.6, len(ARMS) - 0.4)
    axes[0].set_ylabel("Top-action share at init."); axes[0].set_ylim(0, 1.06)
    axes[0].set_title("(a) Initial greedy action collapse", loc="left", color=INK)
    # (b) share of steps per episode NOT chosen by the agent's own greedy/noisy policy, (c) length
    k = 10
    for ax, kind, ylab, title in ((axes[1], "nonpolicy", "Share of non-policy actions", "(b) $\\epsilon$-random + guide actions"),
                                  (axes[2], "length", "Steps per episode (10-ep. mean)", "(c) Episode length")):
        style_ax(ax)
        for a in ARMS:
            if kind == "nonpolicy":
                Ms = np.array([[(int(e["random_steps"]) + int(e["guided_steps"])) / max(int(e["steps"]), 1)
                                for e in r["episodes"]] for r in A[a]])
            else:
                Ms = np.array([[int(e["steps"]) for e in r["episodes"]] for r in A[a]])
            sm = np.array([np.convolve(m, np.ones(k) / k, mode="valid") for m in Ms])
            ax.plot(np.arange(k, Ms.shape[1] + 1), sm.mean(0), ls=LS[a], color=COLOR[a], lw=1.3,
                    dash_capstyle="round", solid_capstyle="round")
        ax.set_xlabel("Training episode"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", color=INK); ax.set_xlim(0, budget)
    algo_legend(fig, y=1.06, markers=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5)
    save(fig, out, "fig_mechanism")

    # ================================================================ Fig: value estimates (small multiples)
    fig, axes = plt.subplots(2, 3, figsize=(TXT_W, 3.5), sharex=True, sharey=True)
    for ax, a in zip(axes.flat, ARMS):
        style_ax(ax)
        eps = [int(e["episode"]) for e in A[a][0]["evals"]]
        q = np.array([[float(e["q_start_mean"]) for e in r["evals"]] for r in A[a]])
        g = np.array([[float(e["disc_return_mean"]) for e in r["evals"]] for r in A[a]])
        ok = ~np.all(np.isnan(q), 0)   # Q and return are logged only for terminal check episodes
        eps = np.array(eps)[ok]; q = np.nanmean(q[:, ok], 0); g = np.nanmean(g[:, ok], 0)
        ax.plot(eps, g, color=REF, lw=1.0, marker="s", ms=3, mfc=SURFACE, mec=INK2, mew=0.7, zorder=2)
        line(ax, eps, q, a)
        ax.set_title(LABEL[a], loc="left", color=INK, fontsize=7.5, pad=3)
    for ax in axes[1]:
        ax.set_xlabel("Training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("Value at $P_0$")
    axes[0, 0].set_xlim(0, budget)
    handles = [Line2D([], [], color=INK2, lw=1.5, marker="o", ms=3.6, label="estimated $\\max_a Q(s_0,a)$ (algorithm colour)"),
               Line2D([], [], color=REF, lw=1.0, marker="s", ms=3, mfc=SURFACE, mec=INK2, label="realised discounted return")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "fig_overestimation")
    # ---------------- END CHANGED

    # ================================================================ Fig: trade-off
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for a in ARMS:
        E = np.median([r["E"] for r in A[a]]); S = np.mean([r["succ"] for r in A[a]])
        W = np.mean([r["meta"]["wall_time_h"] for r in A[a]])
        ax.scatter(E, S, s=40 * W, color=COLOR[a], alpha=0.35 if a.endswith("_g") else 0.8,
                   edgecolor=COLOR[a])
        ax.annotate(LABEL[a], (E, S), xytext=(5, 4), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Median E* (episodes; 220 = not learned)"); ax.set_ylabel("Mean final success")
    ax.set_xlim(40, 240); ax.set_ylim(0, 1)
    ax.text(0.02, 0.03, "marker area ∝ wall-clock hours per run", transform=ax.transAxes, fontsize=6)
    save(fig, out, "fig_tradeoff")

    # ================================================================ Table: main results
    hdr = ["Arm", "E* median (censored)", "AUC", "Success mean", "IQM", "ClusterBoot95 (seeds, poses)",
           "Pooled k/n", "Wilson95", "Collision rate", "Steps to goal", "Train goals", "Wall h"]
    rows, tex = [], []
    for a in ARMS:
        R = A[a]; succ = [r["succ"] for r in R]; k = sum(int(s * 50 + 0.5) for s in succ); n = 50 * len(R)
        lo, hi = cluster_boot_ci(np.array([pose_cells(r["final"]) for r in R])); wl, wh = wilson(k, n)
        Es = [r["E"] for r in R]; nc = sum(r["cens"] for r in R)
        st = [int(e["steps"]) for r in R for e in r["final"] if e["outcome"] == "goal_reached"]
        coll = np.mean([r["coll"] for r in R]); auc = np.mean([r["auc"] for r in R])
        tg = np.mean([r["train_goals"] for r in R]); wt = np.mean([r["meta"]["wall_time_h"] for r in R])
        rows.append([LABEL[a], f"{np.median(Es):.0f} ({nc})", fmt(auc), fmt(np.mean(succ)), fmt(iqm(succ)),
                     f"[{lo:.2f}, {hi:.2f}]", f"{k}/{n}", f"[{wl:.2f}, {wh:.2f}]", fmt(coll),
                     fmt(np.mean(st), 1) if st else "--", fmt(tg, 1), fmt(wt, 1)])
        tex.append([LABEL[a], f"{np.median(Es):.0f} ({nc}/5)", fmt(auc), fmt(np.mean(succ)), fmt(iqm(succ)),
                    f"[{lo:.2f}, {hi:.2f}]", fmt(coll), fmt(np.mean(st), 0) if st else "--", fmt(tg, 0)])
    for h in ("H", "H_eta"):
        e = H[h]; k = sum(x["outcome"] == "goal_reached" for x in e); wl, wh = wilson(k, len(e))
        c = sum(x["outcome"] == "collision_warning" for x in e) / len(e)
        st = [int(x["steps"]) for x in e if x["outcome"] == "goal_reached"]
        rows.append([LABEL[h], "--", "--", fmt(k / len(e)), "--", "--", f"{k}/{len(e)}",
                     f"[{wl:.2f}, {wh:.2f}]", fmt(c), fmt(np.mean(st), 1), "--", "--"])
        tex.append([LABEL[h].replace("%", r"\%"), "--", "--", fmt(k / len(e)), "--", "--", fmt(c),
                    fmt(np.mean(st), 0), "--"])
    write_table(out, "tab_main", hdr, rows, tex)

    # ================================================================ Contrasts (main + B2 + B3)
    def best_unguided(key):
        cands = [a for a in ARMS if not a.endswith("_g")]
        return min(cands, key=lambda a: (np.median([r[key] for r in A[a]]),
                                          np.mean([r[key] for r in A[a]]), ARMS.index(a)))

    def contrasts(key, cens_key=None):
        """Same 4 planned contrasts as analyze.py, on per-run quantity `key` (lower = better).
        Censored runs get one common value above every observed value (ranked slowest, tied)."""
        vals = {}
        big = max(r[key] for r in runs) * 10 + 1
        for a in ARMS:
            vals[a] = [big if (cens_key and r[cens_key]) else r[key] for r in A[a]]
        b = best_unguided(key)
        spec = [("C1", "rainbow_g", "rainbow", "unpaired"), ("C2", "d3qn_g", "d3qn", "unpaired"),
                ("C3", "ddqn", "dqn", "paired"), ("C4", "rainbow_g", b, "unpaired")]
        res = []
        for cid, x, y, kind in spec:
            if kind == "paired":
                _, p1, p2 = exact_perm_test_paired(vals[x], vals[y])
                av = vargha_delaney_a(vals[x], vals[y])
            else:
                av, p1, p2 = exact_perm_test_unpaired(vals[x], vals[y])
            res.append([cid, x, y, kind, av, p1, p2])
        ph = holm([r[5] for r in res])
        return [r + [h] for r, h in zip(res, ph)]

    hdr = ["Analysis", "ID", "X", "Y", "Test", "A", "p one-sided", "p two-sided", "p Holm"]
    allrows, tex = [], []
    B2 = {}
    for bud in (100, 150, budget):
        for r in runs:
            r[f"E{bud}"], r[f"c{bud}"] = e_star_budget(r["evals"], bud)
        B2[bud] = contrasts(f"E{bud}", f"c{bud}")
        for c in B2[bud]:
            allrows.append([f"E* episodes, budget {bud}"] + c[:4] + [f"{c[4]:.2f}", f"{c[5]:.4f}", f"{c[6]:.4f}", f"{c[7]:.4f}"])
    B3 = {"E_steps": contrasts("E_steps", "cens"), "E_sim": contrasts("E_sim", "cens")}
    for k, lab in (("E_steps", "E* env steps"), ("E_sim", "E* sim seconds")):
        for c in B3[k]:
            allrows.append([lab] + c[:4] + [f"{c[4]:.2f}", f"{c[5]:.4f}", f"{c[6]:.4f}", f"{c[7]:.4f}"])
    write_table(out, "tab_contrasts_all", hdr, allrows, [])
    names = {"C1": "Rainbow+guide vs Rainbow", "C2": "D3QN+guide vs D3QN", "C3": "DDQN vs DQN (paired)",
             "C4": "Rainbow+guide vs best unguided"}
    # CHANGED (2026-11-17, review): one p format everywhere; Holm also for the step-based tests;
    # for the paired C3 the effect is the within-pair probability of superiority (ties = 1/2).
    dq = {r["seed"]: r["E"] for r in A["dqn"]}; dd = {r["seed"]: r["E"] for r in A["ddqn"]}
    paired_ps = np.mean([1.0 if dd[s] < dq[s] else 0.5 if dd[s] == dq[s] else 0.0 for s in dq])
    for c in B2[budget]:
        st = [x for x in B3["E_steps"] if x[0] == c[0]][0]
        eff = f"{paired_ps:.2f}$^\\dagger$" if c[0] == "C3" else f"{c[4]:.2f}"
        tex.append([c[0], names[c[0]] + (f" ({LABEL[c[2]]})" if c[0] == "C4" else ""),
                    eff, pfmt(c[5]), pfmt(c[7]), pfmt(st[5]), pfmt(st[7])])
    with open(os.path.join(out, "tables", "tab_contrasts.tex"), "w") as f:
        for r in tex:
            f.write(" & ".join(r) + r" \\" + "\n")

    # ================================================================ B2 table: learned fraction and median E* by budget
    hdr = ["Arm"] + [f"B={b}: learned, median E*" for b in (100, 150, budget)]
    rows, tex = [], []
    for a in ARMS:
        row = [LABEL[a]]
        for b in (100, 150, budget):
            Es = [r[f"E{b}"] for r in A[a]]; nl = sum(not r[f"c{b}"] for r in A[a])
            row.append(f"{nl}/5, {np.median(Es):.0f}")
        rows.append(row); tex.append(row)
    write_table(out, "tab_budget", hdr, rows, tex)

    # ================================================================ B3 table: E* in steps / sim time
    hdr = ["Arm", "median E* episodes", "median E* env steps", "median E* sim min", "mean ep. length", "total env steps"]
    rows = []
    for a in ARMS:
        R = A[a]; c = any(r["cens"] for r in R)
        rows.append([LABEL[a], f"{np.median([r['E'] for r in R]):.0f}",
                     f"{np.median([r['E_steps'] for r in R]):.0f}" + (" (cens.)" if all(r["cens"] for r in R) else ""),
                     f"{np.median([r['E_sim'] for r in R]) / 60:.1f}",
                     f"{np.mean([r['mean_len'] for r in R]):.1f}", f"{np.mean([r['env_steps'] for r in R]):.0f}"])
    write_table(out, "tab_sample_eff", hdr, rows)

    # ================================================================ failure-mode table
    # CHANGED (2026-11-17): collisions attributed to walls OR static cylinders (was walls only)
    NEAR = 0.25
    hdr = ["Arm", "goal", "collision", "no progress", "timeout",
           f"collisions within {NEAR} m of wall / cylinder / neither", "most common obstacle"]
    rows = []
    for a in labels:
        eps = H[a] if a in H else [e for r in A[a] for e in r["final"]]
        cnt = {k: sum(e["outcome"] == k for e in eps) for k in OUTCOME_SHORT}
        cw = [f for f in fail_rows if f[0] == a and f[2] == "collision_warning"]
        nwall = sum(f[4] < NEAR and f[5] == "wall" for f in cw)
        ncyl = sum(f[4] < NEAR and f[5] == "cylinder" for f in cw)
        near = [f[3] for f in cw if f[4] < NEAR]
        top = max(set(near), key=near.count) + f" ({near.count(max(set(near), key=near.count))})" if near else "--"
        rows.append([LABEL[a], cnt["goal_reached"], cnt["collision_warning"], cnt["no_progress"], cnt["timeout"],
                     f"{nwall} / {ncyl} / {len(cw) - nwall - ncyl}", top])
    write_table(out, "tab_failures", hdr, rows,
                [[r[0].replace("%", r"\%"), r[1], r[2], r[3], r[4], r[5],
                  r[6].replace("(-", "($-$").replace(", -", ", $-$")] for r in rows])
    write_table(out, "tab_per_pose", ["Arm"] + [f"P{j}" for j in range(5)],
                [[LABEL[a]] + [f"{M[i, j]:.2f}" for j in range(5)] for i, a in enumerate(labels)],
                [[LABEL[a].replace("%", r"\%")] + [f"{M[i, j]:.2f}" for j in range(5)] for i, a in enumerate(labels)])

    # ================================================================ numbers.md
    N.append("# Numbers for the paper (auto-generated by paper_figures.py; do not edit by hand)\n")
    N.append(f"Runs: {len(runs)} completed; budget {budget} episodes; checks every {CHECK_EVERY}.\n")
    N.append("## Per-run E* (episodes / env steps / sim min), final success, collision rate\n")
    for a in ARMS:
        N.append(f"- **{LABEL[a]}**: " + "; ".join(
            f"s{r['seed']}: {r['E']}{'c' if r['cens'] else ''} / {r['E_steps']:.0f} / {r['E_sim'] / 60:.1f} min, "
            f"succ {r['succ']:.2f}, coll {r['coll']:.2f}, guided {r['guided_share']:.2f}, eps-rand {r['random_share']:.2f}, "
            f"init top-share {r['meta']['initial_greedy_top_share']:.2f}" for r in A[a]))
    N.append("\n## Planned contrasts, all analyses (one-sided p, Holm within each analysis)\n")
    N.append("| Analysis | ID | X | Y | A | p1 | p Holm |\n|---|---|---|---|---|---|---|")
    for r in allrows:
        N.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5]} | {r[6]} | {r[8]} |")
    N.append("\n## Budget truncation (learned/5, median E*)\n")
    N.append("| Arm | B=100 | B=150 | B=200 |\n|---|---|---|---|")
    for r in write_rows_budget(A, budget):
        N.append("| " + " | ".join(r) + " |")
    N.append("\n## Training experience (means over seeds)\n")
    N.append("| Arm | env steps | mean ep. len | train goals | guided share | eps-random share | RTF mean | RTF min | wall h |\n|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        R = A[a]
        N.append(f"| {LABEL[a]} | {np.mean([r['env_steps'] for r in R]):.0f} | {np.mean([r['mean_len'] for r in R]):.1f} | "
                 f"{np.mean([r['train_goals'] for r in R]):.1f} | {np.mean([r['guided_share'] for r in R]):.3f} | "
                 f"{np.mean([r['random_share'] for r in R]):.3f} | {np.mean([r['meta']['rtf_mean'] for r in R]):.3f} | "
                 f"{min(r['meta']['rtf_min'] for r in R):.3f} | {np.mean([r['meta']['wall_time_h'] for r in R]):.2f} |")
    N.append("\n## Failure modes (final evaluation)\n")
    N.append("| Arm | goal | collision | no progress | timeout | coll. wall / cyl. / neither (<0.25 m) | most common obstacle |\n|---|---|---|---|---|---|---|")
    for r in rows:
        N.append("| " + " | ".join(str(c) for c in r) + " |")
    with open(os.path.join(out, "numbers.md"), "w") as f:
        f.write("\n".join(N) + "\n")
    print(f"Wrote figures, tables and numbers.md to {out}")


def write_rows_budget(A, budget):
    out = []
    for a in ARMS:
        row = [LABEL[a]]
        for b in (100, 150, budget):
            Es = [r[f"E{b}"] for r in A[a]]; nl = sum(not r[f"c{b}"] for r in A[a])
            row.append(f"{nl}/5, {np.median(Es):.0f}")
        out.append(row)
    return out


if __name__ == "__main__":
    main()
