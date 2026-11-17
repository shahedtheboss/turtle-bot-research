#!/usr/bin/env bash
# Post hoc Q-along-trajectory probe (analysis only) in its own isolated, headless Gazebo slot.
# One greedy episode from P0 per final epsilon-greedy checkpoint; logs Q-values and rewards per step.
#   scripts/run_q_probe_slot.sh [SLOT]      # default slot 1
# Output: results/q_trajectory/q_trajectory.csv
set -eo pipefail
SLOT=${1:-1}
case "$SLOT" in 1|2|3|4|5|6) ;; *) echo "SLOT must be 1-6" >&2; exit 1 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- isolation for this slot (identical to run_slot.sh) ----
export ROS_DOMAIN_ID=$((30 + SLOT))
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="http://localhost:$((11345 + SLOT))"
export TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
source "$HOME/pytorch-env/bin/activate"

mkdir -p "$REPO/log/slots" "$REPO/results/q_trajectory"
GZ_LOG="$REPO/log/slots/q_probe_gazebo.log"
setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py gui:=false > "$GZ_LOG" 2>&1 &
GZ_PGID=$!
cleanup() {
    kill -INT -- -"$GZ_PGID" 2>/dev/null || true
    sleep 5
    kill -KILL -- -"$GZ_PGID" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$(( $(date +%s) + 120 ))
until timeout 10 ros2 topic echo /scan --once --no-daemon > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then echo "Gazebo did not publish /scan within 120 s; see $GZ_LOG" >&2; exit 1; fi
    sleep 2
done
sleep 3
python3 scripts/q_trajectory_probe.py --out results/q_trajectory/q_trajectory.csv
