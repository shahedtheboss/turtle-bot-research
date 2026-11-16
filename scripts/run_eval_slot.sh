#!/usr/bin/env bash
# Run the FINAL EVALUATION of one finished run (or a heuristic baseline) in its own
# isolated, headless Gazebo slot. Same isolation as run_slot.sh (training); separate
# file so the training script is never edited while training runs use it.
#
#   scripts/run_eval_slot.sh SLOT RUN          # RUN = <arm>_s<seed>, e.g. ddqn_s2
#   scripts/run_eval_slot.sh SLOT H            # heuristic baseline
#   scripts/run_eval_slot.sh SLOT H_eta        # heuristic baseline with 15% noise
#
# The evaluation itself is exactly the commands.md §8 command (50 greedy episodes,
# 5 poses x 10, final checkpoint); only WHERE it runs (which slot) changes.
# Outputs: results/<RUN>/<RUN>_final_eval.csv/.png and <RUN>_eval_gazebo.log

set -eo pipefail
if [ $# -ne 2 ]; then echo "usage: $0 SLOT(1-6) RUN|H|H_eta" >&2; exit 1; fi
SLOT=$1; RUN=$2
case "$SLOT" in 1|2|3|4|5|6) ;; *) echo "SLOT must be 1-6" >&2; exit 1 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RUN_DIR="$REPO/results/$RUN"

case "$RUN" in
    H)     mkdir -p "$RUN_DIR"
           EVAL_ARGS=(--policy heuristic --episodes 50 --label H) ;;
    H_eta) mkdir -p "$RUN_DIR"
           EVAL_ARGS=(--policy heuristic --guide-noise 0.15 --seed 1 --episodes 50 --label H_eta) ;;
    *)     MODEL="$RUN_DIR/${RUN}_model_final.pth"
           [ -f "$MODEL" ] || { echo "no final checkpoint: $MODEL" >&2; exit 1; }
           EVAL_ARGS=(--model "$MODEL" --episodes 50 --label "$RUN" --plot "$RUN_DIR/${RUN}_final_eval.png") ;;
esac
EVAL_ARGS+=(--out-csv "$RUN_DIR/${RUN}_final_eval.csv")

# ---- isolation for this slot (identical to run_slot.sh) ----
export ROS_DOMAIN_ID=$((30 + SLOT))
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="http://localhost:$((11345 + SLOT))"
export TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
source "$HOME/pytorch-env/bin/activate"

mkdir -p "$REPO/log/slots"
GZ_LOG="$REPO/log/slots/${RUN}_eval_gazebo.log"
echo "[eval slot $SLOT] run=$RUN domain=$ROS_DOMAIN_ID gazebo=$GAZEBO_MASTER_URI"

setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py gui:=false > "$GZ_LOG" 2>&1 &
GZ_PGID=$!
cleanup() {
    echo "[eval slot $SLOT] stopping this slot's Gazebo"
    kill -INT -- -"$GZ_PGID" 2>/dev/null || true
    sleep 5
    kill -KILL -- -"$GZ_PGID" 2>/dev/null || true
    cp "$GZ_LOG" "$RUN_DIR/${RUN}_eval_gazebo.log" 2>/dev/null || true
}
trap cleanup EXIT

echo "[eval slot $SLOT] waiting for Gazebo (/scan)..."
deadline=$(( $(date +%s) + 120 ))
until timeout 10 ros2 topic echo /scan --once --no-daemon > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[eval slot $SLOT] Gazebo did not publish /scan within 120 s; see $GZ_LOG" >&2; exit 1
    fi
    sleep 2
done
echo "[eval slot $SLOT] Gazebo ready"
sleep 3

python3 -m dqn_comparison.evaluate "${EVAL_ARGS[@]}"
