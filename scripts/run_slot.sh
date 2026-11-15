#!/usr/bin/env bash
# Run one training run in its own isolated, headless Gazebo ("slot").
#
#   scripts/run_slot.sh SLOT ARM SEED [extra train.py flags...]
#   scripts/run_slot.sh 1 N 1
#   scripts/run_slot.sh 2 G 1 --episodes 5 --eval-every 5 --run-name smoke_G_slot2 --results-dir results_smoke
#
# Each slot gets its own ROS 2 domain (topics/services cannot cross) and its own
# Gazebo master port, so up to 3 runs can share the laptop. The script starts
# gzserver without the GUI, waits for /scan, trains, then stops ONLY its own
# Gazebo (also on Ctrl+C or a crash). The Gazebo log is copied into the run folder.

set -eo pipefail

if [ $# -lt 3 ]; then
    echo "usage: [TRAIN_MODULE=rainbow_ablation.paper1.train] $0 SLOT(1-3) ARM SEED [extra flags]" >&2
    echo "  ablation arms: G N R;  paper1 arms: dqn ddqn d3qn d3qn_g rainbow rainbow_g" >&2
    exit 1
fi
SLOT=$1; ARM=$2; SEED=$3; shift 3
case "$SLOT" in 1|2|3) ;; *) echo "SLOT must be 1, 2 or 3" >&2; exit 1 ;; esac

# ---- NEW CODE ADDED (review fix 10): check the arm matches the training module BEFORE booting Gazebo ----
if [ "${TRAIN_MODULE:-rainbow_ablation.train}" = "rainbow_ablation.paper1.train" ]; then
    case "$ARM" in dqn|ddqn|d3qn|d3qn_g|rainbow|rainbow_g) ;; *) echo "paper1 ARM must be dqn ddqn d3qn d3qn_g rainbow rainbow_g" >&2; exit 1 ;; esac
else
    case "$ARM" in G|N|R) ;; *) echo "ablation ARM must be G, N or R (for Paper 1 set TRAIN_MODULE=rainbow_ablation.paper1.train)" >&2; exit 1 ;; esac
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---- isolation for this slot ----
export ROS_DOMAIN_ID=$((30 + SLOT))
export ROS_LOCALHOST_ONLY=1
export GAZEBO_MASTER_URI="http://localhost:$((11345 + SLOT))"
export RAINBOW_SLOT="$SLOT"
export RAINBOW_PARALLEL="${RAINBOW_PARALLEL:-unknown}"
export TURTLEBOT3_MODEL=burger

source /opt/ros/humble/setup.bash
source "$REPO/install/setup.bash"
source "$HOME/pytorch-env/bin/activate"

# Work out the run folder the same way train.py does, to store the Gazebo log there.
RUN_NAME="${ARM}_s${SEED}"
# ---- CHANGED CODE: Paper 1 writes to results_paper1 by default ----
if [ "${TRAIN_MODULE:-}" = "rainbow_ablation.paper1.train" ]; then RESULTS_DIR="results_paper1"; else RESULTS_DIR="results"; fi
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --run-name) RUN_NAME="${args[$((i + 1))]}" ;;
        --results-dir) RESULTS_DIR="${args[$((i + 1))]}" ;;
        --run-name=*) RUN_NAME="${args[$i]#--run-name=}" ;;          # review fix 10
        --results-dir=*) RESULTS_DIR="${args[$i]#--results-dir=}" ;;
    esac
done
mkdir -p "$REPO/log/slots"
GZ_LOG="$REPO/log/slots/${RUN_NAME}_gazebo.log"

echo "[slot $SLOT] run=$RUN_NAME domain=$ROS_DOMAIN_ID gazebo=$GAZEBO_MASTER_URI"

# Start headless Gazebo in its own process group so we can stop exactly it.
setsid ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py gui:=false \
    > "$GZ_LOG" 2>&1 &
GZ_PGID=$!

cleanup() {
    echo "[slot $SLOT] stopping this slot's Gazebo"
    kill -INT -- -"$GZ_PGID" 2>/dev/null || true
    sleep 5
    kill -KILL -- -"$GZ_PGID" 2>/dev/null || true
    case "$RESULTS_DIR" in /*) RUN_DIR="$RESULTS_DIR/$RUN_NAME" ;; *) RUN_DIR="$REPO/$RESULTS_DIR/$RUN_NAME" ;; esac   # review fix 10
    if [ -d "$RUN_DIR" ]; then
        cp "$GZ_LOG" "$RUN_DIR/${RUN_NAME}_gazebo.log" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[slot $SLOT] waiting for Gazebo (/scan)..."
if ! timeout 90 ros2 topic echo /scan --once > /dev/null 2>&1; then
    echo "[slot $SLOT] Gazebo did not publish /scan within 90 s; see $GZ_LOG" >&2
    exit 1
fi
sleep 3   # let odometry and the clock start too

# ---- NEW CODE ADDED: TRAIN_MODULE picks the study (Paper 1: rainbow_ablation.paper1.train) ----
TRAIN_MODULE="${TRAIN_MODULE:-rainbow_ablation.train}"
python3 -m "$TRAIN_MODULE" --arm "$ARM" --seed "$SEED" "$@"
