#!/usr/bin/env bash
# Study v2: one training run + its final evaluation in an isolated, headless Gazebo slot.
#
#   scripts/run_slot_v2.sh SLOT WORLD ARM SEED [extra train_v2 flags...]
#   scripts/run_slot_v2.sh 1 4 d3qn 1
#   EVAL_REPEATS=1 scripts/run_slot_v2.sh 2 4s rainbow_g 9 --episodes 3 --eval-every 3 --results-dir results_smoke/b11
#
# Slot isolation as in v1: ROS_DOMAIN_ID = 30 + SLOT, Gazebo port 11345 + SLOT (slots 1-8; use more than 6 only
# after the long load test). Gazebo: turtlebot3_drl_v2.launch.py stage:=WORLD rtf_cap:=$V2_RTF_CAP (default 3).
# Steps: start Gazebo, wait for /scan, train (train_v2), evaluate the FINAL checkpoint (evaluate_v2,
# EVAL_REPEATS x 20 episodes, default 2), stop ONLY this slot's Gazebo (matched by its port; also on Ctrl+C).
set -eo pipefail
if [ $# -lt 4 ]; then echo "usage: $0 SLOT(1-8) WORLD(2|3|4s|4|6) ARM SEED [train_v2 flags]" >&2; exit 1; fi
SLOT=$1; WORLD=$2; ARM=$3; SEED=$4; shift 4
case "$SLOT" in 1|2|3|4|5|6|7|8) ;; *) echo "SLOT must be 1-8" >&2; exit 1 ;; esac
case "$WORLD" in 2|3|4s|4|6) ;; *) echo "WORLD must be 2, 3, 4s, 4 or 6" >&2; exit 1 ;; esac
case "$ARM" in dqn|ddqn|d3qn|rainbow|rainbow_g) ;; *) echo "ARM must be dqn ddqn d3qn rainbow rainbow_g" >&2; exit 1 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export ROS_DOMAIN_ID=$((30 + SLOT)) ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="http://localhost:$((11345 + SLOT))"
export RAINBOW_SLOT="$SLOT" RAINBOW_PARALLEL="${RAINBOW_PARALLEL:-unknown}" TURTLEBOT3_MODEL=burger
export DQN_SIM_TIME_STEP=1 V2_RTF_CAP="${V2_RTF_CAP:-3}"
# CPU tweak options (2026-11-20): /clock rate and wait-poll interval; defaults = original values
export V2_CLOCK_HZ="${V2_CLOCK_HZ:-1000}" DQN_SIM_POLL_S="${DQN_SIM_POLL_S:-0.001}"
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
source "$HOME/pytorch-env/bin/activate"

RUN_NAME="w${WORLD}_${ARM}_s${SEED}"; RESULTS_DIR="results_v2"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in --run-name) RUN_NAME="${args[$((i + 1))]}" ;; --results-dir) RESULTS_DIR="${args[$((i + 1))]}" ;; esac
done
RUN_DIR="$REPO/$RESULTS_DIR/$RUN_NAME"
mkdir -p "$REPO/log/slots"
GZ_LOG="$REPO/log/slots/${RUN_NAME}_gazebo.log"

stop_slot_gazebo() {   # by port, not process group (the launch parent can exit before its children)
    for p in $(pgrep gzserver || true); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GAZEBO_MASTER_URI=$GAZEBO_MASTER_URI" && kill -INT "$p" || true
    done
    for p in $(pgrep -f "robot_state_publisher" || true); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" && kill -INT "$p" || true
    done
}
cleanup() {
    echo "[slot $SLOT] stopping this slot's Gazebo"
    stop_slot_gazebo; sleep 5; stop_slot_gazebo
    [ -d "$RUN_DIR" ] && cp "$GZ_LOG" "$RUN_DIR/${RUN_NAME}_gazebo.log" 2>/dev/null || true
}
trap cleanup EXIT

echo "[slot $SLOT] $RUN_NAME world=$WORLD domain=$ROS_DOMAIN_ID gazebo=$GAZEBO_MASTER_URI rtf_cap=$V2_RTF_CAP"
setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_v2.launch.py stage:="$WORLD" rtf_cap:="$V2_RTF_CAP" \
    clock_rate:="$V2_CLOCK_HZ" gui:=false \
    > "$GZ_LOG" 2>&1 &
deadline=$(( $(date +%s) + 120 ))
until timeout 10 ros2 topic echo /scan --once --no-daemon > /dev/null 2>&1; do
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[slot $SLOT] Gazebo did not publish /scan within 120 s; see $GZ_LOG" >&2; exit 1; }
    sleep 2
done
sleep 3

python3 -m dqn_comparison.train_v2 --world "$WORLD" --arm "$ARM" --seed "$SEED" "$@"
python3 -m dqn_comparison.evaluate_v2 --world "$WORLD" --model "$RUN_DIR/${RUN_NAME}_model_final.pth" \
    --repeats "${EVAL_REPEATS:-2}" --label "$RUN_NAME" --out-csv "$RUN_DIR/${RUN_NAME}_final_eval.csv"
