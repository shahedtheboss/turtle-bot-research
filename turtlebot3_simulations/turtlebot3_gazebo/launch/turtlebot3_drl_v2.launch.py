#!/usr/bin/env python3
# ---------------- NEW FILE (2026-11-19, study v2) ----------------
# Launch one DRL stage world for study v2. The v1 launch files are unchanged.
#
#   ros2 launch turtlebot3_gazebo turtlebot3_drl_v2.launch.py stage:=4 rtf_cap:=3 clock_rate:=1000 gui:=false
#
# What it changes compared with turtlebot3_drl_stageN.launch.py:
#   1. Speed cap: a copy of the world file is written with real_time_update_rate = 1000 * rtf_cap.
#      max_step_size stays 0.001 s, so the physics per step is identical; only wall-clock speed changes.
#      rtf_cap:=0 removes the cap (as fast as the CPU allows).
#   2. /clock rate: gazebo_ros publishes /clock at clock_rate Hz (default 10 Hz -> 100 ms ticks,
#      too coarse for 0.5 s actions timed on simulated time; 1000 Hz -> 1 ms ticks).
#   3. Moving obstacles: the obstacle plugin folder is added to GAZEBO_PLUGIN_PATH, so the
#      animated obstacles of stages 3-6 and 8-9 move (in v1 they were not found and stayed static).
# Use together with DQN_SIM_TIME_STEP=1 in the training code, so actions last 0.5 s of simulated time.

import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

V2_STAGES = ('1', '2', '3', '4', '6')     # worlds used in study v2 (all 5 x 5 m arena)
TMP_DIR = '/tmp/drlnav_v2'


def _prepare(context):
    """Write the speed-capped world copy and the clock parameter file; return gzserver launch."""
    stage = LaunchConfiguration('stage').perform(context)
    rtf_cap = float(LaunchConfiguration('rtf_cap').perform(context))
    clock_rate = float(LaunchConfiguration('clock_rate').perform(context))
    if stage not in V2_STAGES:
        raise RuntimeError(f'stage must be one of {V2_STAGES}, got {stage!r}')
    if rtf_cap < 0 or clock_rate <= 0:
        raise RuntimeError('rtf_cap must be >= 0 and clock_rate > 0')

    pkg = get_package_share_directory('turtlebot3_gazebo')
    model = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    src = os.path.join(pkg, 'worlds', f'turtlebot3_drl_stage{stage}', f'{model}.model')
    text = open(src).read()

    rate = 1000.0 * rtf_cap      # steps per wall-clock second; 0 = unlimited
    text, n = re.subn(r'<real_time_update_rate>[^<]*</real_time_update_rate>',
                      f'<real_time_update_rate>{rate:g}</real_time_update_rate>', text)
    if n != 1 or '<max_step_size>0.001</max_step_size>' not in text:
        raise RuntimeError(f'{src}: expected one real_time_update_rate and max_step_size 0.001')

    # /get_entity_state and /set_entity_state (used to teleport the robot at reset) come from the
    # gazebo_ros_state world plugin. Only the stage-4 world declares it; add the same block to the
    # copy of the other worlds (the original world files stay unchanged).
    if 'libgazebo_ros_state.so' not in text:
        state_plugin = ('<world name="default">\n  <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">\n'
                        '   <ros>\n     <namespace>/</namespace>\n    </ros>\n'
                        '   <update_rate>10.0</update_rate>\n  </plugin>')
        text, n = re.subn(r'<world name="default">', state_plugin, text)
        if n != 1:
            raise RuntimeError(f'{src}: could not insert gazebo_ros_state plugin')

    os.makedirs(TMP_DIR, exist_ok=True)
    world = os.path.join(TMP_DIR, f'stage{stage}_{model}_cap{rtf_cap:g}.model')
    params = os.path.join(TMP_DIR, f'clock_{clock_rate:g}hz.yaml')
    # Write via a temporary name + rename, so parallel slots never read a half-written file.
    for path, content in ((world, text),
                          # publish_rate must be a float in YAML ("1000.0"); "1000" is an int and gzserver aborts
                          (params, f'gazebo:\n  ros__parameters:\n    publish_rate: {clock_rate:.1f}\n')):
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            f.write(content)
        os.replace(tmp, path)

    with open('/tmp/drlnav_current_stage.txt', 'w') as f:   # kept for compatibility with v1 tools
        f.write(f'{stage}\n')

    gazebo_ros = get_package_share_directory('gazebo_ros')
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world, 'pause': 'false', 'params_file': params}.items())]


def generate_launch_description():
    pkg = get_package_share_directory('turtlebot3_gazebo')
    plugin_dir = os.path.join(pkg, 'models', 'turtlebot3_drl_world', 'obstacle_plugin', 'lib')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    return LaunchDescription([
        DeclareLaunchArgument('stage', default_value='4', description='DRL stage world: 1, 2, 3, 4 or 6'),
        DeclareLaunchArgument('rtf_cap', default_value='3', description='Max real-time factor (0 = unlimited)'),
        DeclareLaunchArgument('clock_rate', default_value='1000', description='/clock publish rate [Hz]'),
        DeclareLaunchArgument('gui', default_value='false', description='Start gzclient'),
        AppendEnvironmentVariable('GAZEBO_PLUGIN_PATH', plugin_dir),
        OpaqueFunction(function=_prepare),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items()),
    ])
