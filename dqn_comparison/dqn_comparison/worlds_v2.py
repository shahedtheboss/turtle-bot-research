"""Study v2: geometry and obstacle motion of the five DRL stage worlds (1, 2, 3, 4, 6).

Everything here is DATA copied from the Gazebo model/world files and the obstacle plugin sources
(turtlebot3_simulations/turtlebot3_gazebo/...), plus small geometry helpers. It is used to choose
start poses, goals and the warm-up time W (scripts/v2_world_layout.py) and, later, by training and
evaluation. tests/test_worlds_v2.py checks the copied keyframes against the .cc plugin files.

Coordinates: Gazebo world frame, metres. Animation time restarts at 0 after /reset_simulation.

CORRECTED 2026-11-19 (B4b): each moving obstacle is a NESTED model. The outer model follows the keyframes
(that is what /get_entity_state reports), the cylinder sits inside it at the <include> <pose> offset
(OFFSETS below), and Gazebo interpolates between keyframes with smooth curves, not straight lines.
The real cylinder positions were therefore RECORDED from /obstacle/odom (gazebo_ros_p3d, world frame) and
are used from data/v2_obstacle_paths.csv; a LiDAR check confirmed /obstacle/odom is what the robot sees
(scripts/v2_obstacle_truth_check.py). anim_position() below is the outer frame only, kept for reference.
"""
import csv
import math
import os

ARENA_HALF = 2.345            # inner face of the outer walls (centre 2.42, thickness 0.15)
ROBOT_RADIUS = 0.105          # TurtleBot3 Burger footprint, approx. circumradius

# ---- walls: (centre_x, centre_y, yaw_rad, length, thickness) ----------------------------------------
OUTER_WALLS = [(-2.42, 0.0, math.pi / 2, 5.0, 0.15), (0.0, 2.42, 0.0, 5.0, 0.15),
               (2.42, 0.0, -math.pi / 2, 5.0, 0.15), (0.0, -2.42, math.pi, 5.0, 0.15)]
INNER_WALLS_STAGE4 = [(-2.00, -1.50, 0.0, 1.0, 0.15), (-0.50, -2.00, -math.pi / 2, 1.0, 0.15),
                      (1.00, -1.00, math.pi / 2, 1.0, 0.15), (1.20, 1.90, -math.pi / 2, 1.0, 0.15),
                      (1.90, 0.40, 0.0, 1.0, 0.15), (-0.50, 1.50, 0.0, 1.0, 0.15),
                      (-1.20, 0.09, -math.pi / 2, 1.0, 0.15)]

# ---- moving obstacles: keyframes (t [s], x, y), linear interpolation, looped ------------------------
# Copied from models/turtlebot3_drl_world/obstacle_plugin/obstacleN.cc (PoseAnimation, loop=true).
ANIM = {
    'obstacle1': (160.0, [(0, 0, 0), (10, -0.5, -1), (50, -3.5, -1), (70, -3.7, -3), (90, -3.5, -1),
                          (130, -0.5, -1), (140, 0, 0), (160, 0, 0)]),
    'obstacle2': (140.0, [(0, 0, 0), (10, 0.7, 0.2), (40, 2.5, 3.5), (55, 0.3, 3.5), (85, 3.5, 1.8),
                          (100, 3.5, 0), (110, 2, 0.5), (115, 1.5, 1), (120, 1, 0.5), (125, 0.5, 0.1),
                          (130, 0, 0), (140, 0, 0)]),
    'obstacle3': (165.0, [(0, 0, 0), (10, -1, 0.2), (40, -2, 1), (55, -3.5, 0), (85, -2.5, 1.5),
                          (110, 0, 0), (130, -1, 2), (145, -2, 1), (165, 0, 0)]),
    'obstacle4': (170.0, [(0, 0, 0), (10, 0, -3.2), (30, 2, -2.7), (40, 0, 0), (60, 0, -3.2),
                          (80, 2, -2.7), (110, 0, 0), (130, 0, -3.2), (150, 2, -2.7), (170, 0, 0)]),
    'obstacle5': (205.0, [(0, 0, 0), (10, 0.7, 1), (40, 2.5, 2), (55, 0, 2), (85, 0, -1), (110, 2, 0),
                          (125, 4, 0), (145, 3, -2), (170, 2, 0), (185, 2, 2), (205, 0, 0)]),
    'obstacle6': (170.0, [(0, 0, 0), (10, -1, 0), (40, -1, 2), (55, -1.5, 0), (85, 0, 2),
                          (120, -4, -1.8), (130, -3, -1.8), (145, -2.5, 1), (170, 0, 0)]),
}
ANIM_RADIUS = 0.16

# Stage 2: static group; stage 3: the same group rotating about (0, 0), one turn per 40 s (obstacles.cc).
GROUP_CYLINDERS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
GROUP_RADIUS = 0.15
GROUP_PERIOD = 40.0

WORLDS = {
    1: {'walls': OUTER_WALLS, 'group': None, 'anim': []},
    2: {'walls': OUTER_WALLS, 'group': 'static', 'anim': []},
    3: {'walls': OUTER_WALLS, 'group': 'rotating', 'anim': []},
    4: {'walls': OUTER_WALLS + INNER_WALLS_STAGE4, 'group': None, 'anim': ['obstacle1', 'obstacle2']},
    6: {'walls': OUTER_WALLS, 'group': None,
        'anim': ['obstacle1', 'obstacle2', 'obstacle3', 'obstacle4', 'obstacle5', 'obstacle6']},
}


# ---- v2 layouts: produced by `python3 scripts/v2_world_layout.py --search` (rules rev 3 in its header) ---
# Re-chosen 2026-11-19 on the RECORDED obstacle paths (the rev-1/2 layouts used wrong positions).
# start and goals are the same in every world; test poses are per world; W = warm-up after reset
# (animation time at which every episode starts); poses are (x, y, yaw_rad).
START = (0.0, 0.0, 0.0)
GOALS = [(1.75, -1.0), (-1.75, 1.0), (0.5, 1.75), (-1.75, -0.5)]
_OPEN_TESTS = [(1.75, 1.75, -2.3562), (-1.75, -1.75, 0.7854), (1.25, -1.75, 2.191), (-1.25, 1.75, -0.9505)]
LAYOUTS = {
    1: {'W': 0.0, 'start': START, 'goals': GOALS, 'tests': _OPEN_TESTS},
    2: {'W': 0.0, 'start': START, 'goals': GOALS, 'tests': _OPEN_TESTS},
    3: {'W': 5.0, 'start': START, 'goals': GOALS, 'tests': _OPEN_TESTS},
    4: {'W': 5.0, 'start': START, 'goals': GOALS,
        'tests': [(0.25, -1.75, 1.7127), (0.75, 0.75, -2.3562), (-1.0, -1.0, 0.7854), (1.0, 0.0, -3.1416)]},
    6: {'W': 5.0, 'start': START, 'goals': GOALS,
        'tests': [(-1.25, 1.75, -0.9505), (0.5, -1.75, 1.8491), (0.75, 0.75, -2.3562), (-1.0, -1.0, 0.7854)]},
}


# Nested <include> pose of each cylinder inside its animated model (world files, stages 4 and 6).
OFFSETS = {4: {'obstacle1': (2.0, 2.0), 'obstacle2': (-2.0, -2.0)},
           6: {'obstacle1': (2.0, 2.0), 'obstacle2': (-2.0, -2.0), 'obstacle3': (2.0, -2.0),
               'obstacle4': (-2.0, 2.0), 'obstacle5': (-2.0, 0.0), 'obstacle6': (2.0, 0.0)}}

_PATHS = None
PATH_DT = 0.1      # table resolution [s]; table covers 0-400 s, longer than every loop period


def _paths():
    """Recorded cylinder positions {(stage, name): [(x, y), ...]} and stage-3 yaw {(3,'group_yaw'): [yaw, ...]}."""
    global _PATHS
    if _PATHS is None:
        _PATHS = {}
        path = os.path.join(os.path.dirname(__file__), 'data', 'v2_obstacle_paths.csv')
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (int(r['stage']), r['name'])
                val = float(r['x_or_yaw']) if r['y'] == '' else (float(r['x_or_yaw']), float(r['y']))
                _PATHS.setdefault(key, []).append(val)
    return _PATHS


def recorded_position(stage, name, t):
    """Real (x, y) of an animated cylinder at animation time t, from the recording (loops are identical)."""
    table = _paths()[(stage, name)]
    i = round((t % ANIM[name][0]) / PATH_DT)
    return table[min(i, len(table) - 1)]


def recorded_group_yaw(t):
    table = _paths()[(3, 'group_yaw')]
    return table[min(round((t % GROUP_PERIOD) / PATH_DT), len(table) - 1)]


def anim_position(name, t):
    """(x, y) of the OUTER animated frame at time t, linear between keyframes (reference only, not the cylinder)."""
    period, keys = ANIM[name]
    t = t % period
    for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
    return keys[-1][1], keys[-1][2]


def obstacles_at(stage, t):
    """List of (x, y, radius) for every obstacle of the stage at animation time t."""
    w = WORLDS[stage]
    out = []
    if w['group'] == 'static':
        out += [(x, y, GROUP_RADIUS) for x, y in GROUP_CYLINDERS]
    elif w['group'] == 'rotating':
        a = recorded_group_yaw(t)              # recorded: eased, not constant speed
        out += [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), GROUP_RADIUS)
                for x, y in GROUP_CYLINDERS]
    out += [(*recorded_position(stage, n, t), ANIM_RADIUS) for n in w['anim']]
    return out


def wall_distance(stage, x, y):
    """Distance from (x, y) to the nearest wall surface (0 inside a wall)."""
    best = float('inf')
    for cx, cy, yaw, length, thick in WORLDS[stage]['walls']:
        dx, dy = x - cx, y - cy                                   # into the wall's own frame
        u = dx * math.cos(yaw) + dy * math.sin(yaw)
        v = -dx * math.sin(yaw) + dy * math.cos(yaw)
        ex = max(abs(u) - length / 2, 0.0)
        ey = max(abs(v) - thick / 2, 0.0)
        best = min(best, math.hypot(ex, ey))
    return best


def obstacle_distance(stage, x, y, t):
    """Distance from (x, y) to the nearest obstacle surface at time t (inf if none)."""
    obs = obstacles_at(stage, t)
    return min((math.hypot(x - ox, y - oy) - r for ox, oy, r in obs), default=float('inf'))
