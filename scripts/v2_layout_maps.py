"""Draw the v2 layouts: walls, obstacle paths, obstacle positions at episode start (t = W),
start pose, 4 shared goals and the 4 per-world test poses.  python3 scripts/v2_layout_maps.py OUT.png"""
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dqn_comparison'))
from dqn_comparison import worlds_v2 as W  # noqa: E402

TITLES = {1: 'Stage 1: empty', 2: 'Stage 2: 4 static', 3: 'Stage 3: 4 rotating', 4: 'Stage 4: walls + 2 moving',
          6: 'Stage 6: 6 moving'}
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
for ax, st in zip(axes, [1, 2, 3, 4, 6]):
    lay = W.LAYOUTS[st]
    for cx, cy, yaw, L, T in W.WORLDS[st]['walls']:
        r = Rectangle((cx - L / 2, cy - T / 2), L, T, color='#5f5e5a')
        r.set_transform(Affine2D().rotate_around(cx, cy, yaw) + ax.transData)
        ax.add_patch(r)
    ts = [i * 0.5 for i in range(int(400 / 0.5))]
    n = len(W.obstacles_at(st, 0))
    for k in range(n):                                       # obstacle paths over the loop
        pts = [W.obstacles_at(st, t)[k][:2] for t in ts]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color='#eb6834', lw=0.8, alpha=0.5)
    for ox, oy, r in W.obstacles_at(st, lay['W']):           # positions when each episode starts
        ax.add_patch(Circle((ox, oy), r, color='#eb6834'))
    for i, (gx, gy) in enumerate(lay['goals']):
        ax.plot(gx, gy, marker='*', ms=13, color='#1baf7a', mec='black', mew=0.5)
        ax.annotate(f'G{i + 1}', (gx, gy), xytext=(4, 4), textcoords='offset points', fontsize=7)
    for i, (x, y, yaw) in enumerate([lay['start']] + lay['tests']):
        c = '#2a78d6' if i == 0 else '#8a8a8a'
        ax.arrow(x, y, 0.3 * math.cos(yaw), 0.3 * math.sin(yaw), width=0.04, color=c, length_includes_head=True)
        ax.annotate('S' if i == 0 else f'T{i}', (x, y), xytext=(-12, -12), textcoords='offset points', fontsize=7)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_aspect('equal')
    ax.set_title(f'{TITLES[st]}  (W = {lay["W"]:g} s)', fontsize=9); ax.tick_params(labelsize=7)
fig.suptitle('v2 layouts: start S (blue), goals G1-G4 (green, shared), test poses T1-T4 (grey, per world); '
             'orange = obstacles at episode start, lines = their paths', fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=150)
print('written', sys.argv[1])
