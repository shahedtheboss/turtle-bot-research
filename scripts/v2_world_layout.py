"""Study v2, step B4: rule-based choice and check of start pose, goals, test poses and warm-up W.

Obstacle positions come from worlds_v2.obstacles_at(), i.e. the RECORDED real paths (B4b).

Rules (revision 3, 2026-11-19, before any run; history below):
  start     : (0, 0, 0) in every world (as v1)
  walls     : start/test poses >= 0.50 m from any wall surface, goals >= 0.40 m
  obstacles : at the episode start (animation time W) every MOVING obstacle surface >= 1.0 m from the start
              and from every test pose; STATIC obstacles (stage 2) >= 0.50 m, like walls
  goals     : the SAME 4 goals in every world, 1.5-3.6 m from the start; in every world an obstacle surface is
              within 0.30 m of the goal for at most 20 % of the obstacle loop ("blocked fraction");
              selection: first the goal with the smallest worst-case blocked fraction, then repeatedly the
              goal whose direction (seen from the start) is farthest from the chosen ones; ties -> smaller
              worst-case blocked fraction, then coordinates
  tests     : 4 test poses PER WORLD, >= 1.0 m from the start and >= 0.8 m from every goal; selection: first
              the pose farthest from the start, then farthest-point (largest distance to chosen poses);
              heading towards the arena centre
  W         : smallest warm-up in 5..60 s (1 s steps) satisfying the obstacle rule for start + test poses
              (worlds without moving obstacles: 0)
Grid: 0.25 m. Loop sampling for the blocked fraction: 0-400 s every 0.5 s.

History (all before any training run):
  rev 1: moving AND static obstacles >= 1.0 m; goals never on a path (>= 0.30 m), 2.0-3.6 m from start
  rev 2: static obstacles 0.5 m (stage-2 cylinders 2 m apart left no space); goal path clearance 0.20 m and
         1.5 m minimum distance (only two directions were valid in all worlds)
  rev 3: real recorded paths showed stage-6 obstacles pass within 0.3 m of every point for >= 10.9 % of the
         loop, so "never on a path" is impossible -> blocked fraction <= 20 %; whole selection made rule-based

  python3 scripts/v2_world_layout.py            # check worlds_v2.LAYOUTS against the rules
  python3 scripts/v2_world_layout.py --search   # run the rule-based selection and print the layouts
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dqn_comparison'))
from dqn_comparison import worlds_v2 as W  # noqa: E402

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'turtlebot3_simulations', 'turtlebot3_gazebo',
                          'models', 'turtlebot3_drl_world', 'obstacle_plugin')
LOOP_T = [i * 0.5 for i in range(int(400 / 0.5))]
GRID = [i * 0.25 for i in range(-8, 9)]
STAGES = [1, 2, 3, 4, 6]
START = (0.0, 0.0, 0.0)
BLOCK_DIST, BLOCK_MAX = 0.30, 0.20


def check_keyframes():
    """The copied keyframes must equal the .cc plugin sources exactly."""
    for name, (period, keys) in W.ANIM.items():
        t = open(os.path.join(PLUGIN_DIR, f'{name}.cc')).read()
        p = float(re.search(r'PoseAnimation\("[^"]*",\s*([0-9.]+)', t).group(1))
        ks = [(float(a), float(b), float(c)) for a, b, c in re.findall(
            r'CreateKeyFrame\(([0-9.]+)\);\s*key->Translation\(ignition::math::Vector3d\(([-0-9. ]+),([-0-9. ]+),', t)]
        assert p == period and ks == [tuple(map(float, k)) for k in keys], f'{name}: keyframes differ from .cc'
    print('keyframes: identical to the plugin .cc files')


_blocked_cache = {}


def blocked_fraction(stage, x, y):
    key = (stage, x, y)
    if key not in _blocked_cache:
        _blocked_cache[key] = sum(W.obstacle_distance(stage, x, y, t) < BLOCK_DIST for t in LOOP_T) / len(LOOP_T)
    return _blocked_cache[key]


def pose_ok(stage, x, y, w):
    obstacle_min = 0.50 if W.WORLDS[stage]['group'] == 'static' else 1.0
    return W.wall_distance(stage, x, y) >= 0.50 and W.obstacle_distance(stage, x, y, w) >= obstacle_min


def goal_ok(stage, gx, gy, start=START):
    d = math.hypot(gx - start[0], gy - start[1])
    return (W.wall_distance(stage, gx, gy) >= 0.40 and 1.5 <= d <= 3.6
            and blocked_fraction(stage, gx, gy) <= BLOCK_MAX)


def smallest_w(stage, poses):
    if not W.WORLDS[stage]['anim'] and W.WORLDS[stage]['group'] != 'rotating':
        return 0.0
    for w in range(5, 61):
        if all(pose_ok(stage, x, y, w) for x, y, _ in poses):
            return float(w)
    return None


def _angle_gap(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def search_goals():
    cand = [(x, y) for x in GRID for y in GRID if all(goal_ok(st, x, y) for st in STAGES)]
    worst = {p: max(blocked_fraction(st, *p) for st in STAGES) for p in cand}
    ang = {p: math.atan2(p[1], p[0]) for p in cand}
    chosen = [min(cand, key=lambda p: (worst[p], p))]
    while len(chosen) < 4:
        chosen.append(min((p for p in cand if p not in chosen),
                          key=lambda p: (-min(_angle_gap(ang[p], ang[q]) for q in chosen), worst[p], p)))
    return chosen, len(cand)


def search_tests(stage, goals):
    w0 = smallest_w(stage, [START])
    cand = [(x, y) for x in GRID for y in GRID if math.hypot(x, y) >= 1.0
            and min(math.hypot(x - gx, y - gy) for gx, gy in goals) >= 0.8 and pose_ok(stage, x, y, w0)]
    chosen = [max(cand, key=lambda p: (math.hypot(*p), p))]
    while len(chosen) < 4:
        chosen.append(max((p for p in cand if p not in chosen),
                          key=lambda p: (min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in chosen), p)))
    tests = [(x, y, round(math.atan2(-y, -x), 4)) for x, y in chosen]
    return tests, smallest_w(stage, [START] + tests), len(cand)


def report(stage, layout):
    start, tests, goals = layout['start'], layout['tests'], layout['goals']
    w = smallest_w(stage, [start] + tests)
    ok = w == layout['W']
    print(f'\nstage {stage}: W = {w} (layout {layout["W"]}) {"OK" if ok else "FAIL"}')
    for name, (x, y, yaw) in [('start', start)] + [(f'test{i + 1}', p) for i, p in enumerate(tests)]:
        good = pose_ok(stage, x, y, layout['W'])
        ok &= good
        print(f'  {name:6s} ({x:5.2f},{y:5.2f}, {math.degrees(yaw):5.0f} deg)  wall {W.wall_distance(stage, x, y):.2f}  '
              f'obstacle at W {W.obstacle_distance(stage, x, y, layout["W"]):.2f}  {"OK" if good else "FAIL"}')
    for i, (gx, gy) in enumerate(goals):
        good = goal_ok(stage, gx, gy, start)
        ok &= good
        print(f'  goal{i + 1}  ({gx:5.2f},{gy:5.2f})  wall {W.wall_distance(stage, gx, gy):.2f}  '
              f'blocked {blocked_fraction(stage, gx, gy) * 100:4.1f} %  dist {math.hypot(gx - start[0], gy - start[1]):.2f}  '
              f'{"OK" if good else "FAIL"}')
    return ok


if __name__ == '__main__':
    check_keyframes()
    if '--search' in sys.argv:
        goals, n = search_goals()
        print(f'goals ({n} valid in all worlds): {goals}')
        for st in STAGES:
            tests, w, n = search_tests(st, goals)
            print(f'stage {st}: W = {w}, test poses ({n} candidates): {tests}')
    else:
        all_ok = all(report(st, lay) for st, lay in W.LAYOUTS.items())
        print('\nALL LAYOUTS OK' if all_ok else '\nSOME LAYOUTS FAIL')
