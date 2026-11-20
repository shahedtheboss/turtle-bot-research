#!/usr/bin/env bash
# Study v2, step B3: run the sim-time equivalence check in three modes, each in a fresh headless
# Gazebo (stage 1, empty arena), slot 7 isolation (ROS domain 37, Gazebo port 11352).
#   rt_wall : real time, wall-clock action timing (v1 behaviour)
#   rt_sim  : real time, simulated-time action timing
#   x3_sim  : 3x speed cap, simulated-time action timing (v2 setup)
# Output: results_smoke/b3/<mode>.csv
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export ROS_DOMAIN_ID=37 ROS_LOCALHOST_ONLY=1 GAZEBO_MASTER_URI=http://localhost:11352 TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
source "$HOME/pytorch-env/bin/activate"
mkdir -p results_smoke/b3 log/slots

run_mode() {   # mode rtf_cap sim_flag
    local mode=$1 cap=$2 sim=$3
    setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_v2.launch.py stage:=1 rtf_cap:=$cap clock_rate:="${V2_CLOCK_HZ:-1000}" \
        > "log/slots/b3_${mode}_gazebo.log" 2>&1 &
    local gz=$!
    deadline=$(( $(date +%s) + 120 ))
    until timeout 10 ros2 topic echo /scan --once --no-daemon > /dev/null 2>&1; do
        [ "$(date +%s)" -ge "$deadline" ] && { echo "Gazebo not ready ($mode)" >&2; kill -INT -- -$gz; exit 1; }
        sleep 2
    done
    sleep 3
    DQN_SIM_TIME_STEP=$sim python3 scripts/sim_time_equivalence.py --mode "$mode" --repeats "${REPEATS:-3}" --out "results_smoke/b3/${mode}${TAG:-}.csv" || true
    kill -INT -- -$gz 2>/dev/null || true; sleep 5; kill -KILL -- -$gz 2>/dev/null || true
}

MODES="${MODES:-rt_wall rt_sim x3_sim}"
for m in $MODES; do
    case $m in rt_wall) run_mode rt_wall 1 0 ;; rt_sim) run_mode rt_sim 1 1 ;; x3_sim) run_mode x3_sim 3 1 ;; x2_sim) run_mode x2_sim 2 1 ;; esac
done
echo "gzserver left: $(pgrep -c gzserver || true)"
