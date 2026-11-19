"""Study v2, step B4: check / search start poses, test poses, goals and warm-up W per world.

Rules (fixed before searching):
  walls     : start/test poses >= 0.50 m from any wall surface, goals >= 0.40 m
  obstacles : at the episode start (animation time W) every MOVING obstacle surface >= 1.0 m from the
              start pose and from every test pose; STATIC obstacles (stage 2) >= 0.50 m, like walls
              (revised 2026-11-19 before any run: 1.0 m for static cylinders 2 m apart left almost no space)
  goals     : SAME 4 goals in every world; never on an obstacle path: over the whole loop, obstacle
              surface >= 0.20 m from the goal; 1.5-3.6 m from the start; spread over directions
              (relaxed 2026-11-19 before any run from >= 0.30 m and 2.0 m: with the stricter rule only
              two directions were valid in all five worlds)
  tests     : 4 test poses PER WORLD (one shared set was impossible), farthest-point selection among
              valid poses that are >= 1.0 m from the start and >= 0.8 m from every goal; heading
              towards the arena centre
  W         : smallest warm-up in 5..60 s (1 s steps) that satisfies the obstacle rule (static worlds: 0)

  python3 scripts/v2_world_layout.py            # print candidates and check the proposed layout
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dqn_comparison'))
from dqn_comparison import worlds_v2 as W  # noqa: E402

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'turtlebot3_simulations', 'turtlebot3_gazebo',
                          'models', 'turtlebot3_drl_world', 'obstacle_plugin')
LOOP_T = [i * 0.5 for i in range(int(400 / 0.5))]      # 0-400 s covers every loop at least twice


def check_keyframes():
    """The copied keyframes must equal the .cc plugin sources exactly."""
    for name, (period, keys) in W.ANIM.items():
        t = open(os.path.join(PLUGIN_DIR, f'{name}.cc')).read()
        p = float(re.search(r'PoseAnimation\("[^"]*",\s*([0-9.]+)', t).group(1))
        ks = [(float(a), float(b), float(c)) for a, b, c in re.findall(
            r'CreateKeyFrame\(([0-9.]+)\);\s*key->Translation\(ignition::math::Vector3d\(([-0-9. ]+),([-0-9. ]+),', t)]
        assert p == period and ks == [tuple(map(float, k)) for k in keys], f'{name}: keyframes differ from .cc'
    print('keyframes: identical to the plugin .cc files')


def min_path_clearance(stage, x, y):
    return min(W.obstacle_distance(stage, x, y, t) for t in LOOP_T)


def pose_ok(stage, x, y, w):
    obstacle_min = 0.50 if W.WORLDS[stage]['group'] == 'static' else 1.0
    return W.wall_distance(stage, x, y) >= 0.50 and W.obstacle_distance(stage, x, y, w) >= obstacle_min


def goal_ok(stage, gx, gy, start):
    d = math.hypot(gx - start[0], gy - start[1])
    return W.wall_distance(stage, gx, gy) >= 0.40 and min_path_clearance(stage, gx, gy) >= 0.20 and 1.5 <= d <= 3.6


def smallest_w(stage, poses):
    if not W.WORLDS[stage]['anim'] and W.WORLDS[stage]['group'] != 'rotating':
        return 0.0
    for w in range(5, 61):
        if all(pose_ok(stage, x, y, w) for x, y, _ in poses):
            return float(w)
    return None


def report(stage, layout):
    start, tests, goals = layout['start'], layout['tests'], layout['goals']
    w = smallest_w(stage, [start] + tests)
    print(f'\nstage {stage}: W = {w}')
    for name, (x, y, yaw) in [('start', start)] + [(f'test{i + 1}', p) for i, p in enumerate(tests)]:
        ow = W.obstacle_distance(stage, x, y, w or 0)
        print(f'  {name:6s} ({x:5.2f},{y:5.2f}, {math.degrees(yaw):5.0f} deg)  wall {W.wall_distance(stage, x, y):.2f}  '
              f'obstacle at W {ow:.2f}  {"OK" if pose_ok(stage, x, y, w or 0) else "FAIL"}')
    for i, (gx, gy) in enumerate(goals):
        print(f'  goal{i + 1}  ({gx:5.2f},{gy:5.2f})  wall {W.wall_distance(stage, gx, gy):.2f}  '
              f'min path clearance {min_path_clearance(stage, gx, gy):.2f}  dist {math.hypot(gx - start[0], gy - start[1]):.2f}  '
              f'{"OK" if goal_ok(stage, gx, gy, start) else "FAIL"}')
    return w


def candidates(stage, start):
    grid = [i * 0.25 for i in range(-8, 9)]
    return [(x, y) for x in grid for y in grid if goal_ok(stage, x, y, start)]


if __name__ == '__main__':
    check_keyframes()
    from dqn_comparison.worlds_v2 import LAYOUTS     # proposed layouts (added after the first search)
    for st, lay in LAYOUTS.items():
        report(st, lay)
