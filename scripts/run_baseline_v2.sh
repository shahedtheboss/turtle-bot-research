#!/usr/bin/env bash
# Study v2 baselines (deviation 4): the scripted guide alone, evaluated with the final-evaluation protocol of the
# main runs (5 poses x 4 goals x 2 repeats per world), no training.
#   H       : heuristic, no noise; one run per world (deterministic policy)          -> results_v2/w<world>_heuristic/
#   H_noisy : heuristic with the training guide's 15% noise, seeds 1-5 per world       -> results_v2/w<world>_heuristic_noisy_s<seed>/
# 5 + 25 = 30 evaluations, PARALLEL slots at once (default 4). Restartable: runs with a final-eval CSV are skipped.
#
#   scripts/run_baseline_v2.sh
#   DRY_RUN=1 scripts/run_baseline_v2.sh
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PARALLEL="${PARALLEL:-4}"
RESULTS_DIR=results_v2
export V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005       # same simulation settings as the main runs

jobs=()
for world in 2 3 4s 4 6; do jobs+=("w${world}_heuristic $world heuristic 0"); done
for seed in 1 2 3 4 5; do for world in 2 3 4s 4 6; do
    jobs+=("w${world}_heuristic_noisy_s${seed} $world heuristic_noisy $seed")
done; done

is_done() { [ -f "$RESULTS_DIR/$1/$1_final_eval.csv" ]; }
slot_busy() {
    pgrep -f "run_slot_v2.sh $1 " > /dev/null && return 0
    for p in $(pgrep gzserver || true); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GAZEBO_MASTER_URI=http://localhost:$((11345 + $1))" && return 0
    done
    return 1
}

if [ "${DRY_RUN:-0}" = 1 ]; then
    n=0; for j in "${jobs[@]}"; do read -r name _ <<< "$j"; is_done "$name" || { echo "would run: $j"; n=$((n + 1)); }; done
    echo "$n of ${#jobs[@]} baseline runs to do"; exit 0
fi

CLAIMS="$(mktemp -d)"
trap 'rm -rf "$CLAIMS"' EXIT
worker() {
    local slot=$1
    for j in "${jobs[@]}"; do
        read -r name world arm seed <<< "$j"
        is_done "$name" && continue
        mkdir "$CLAIMS/$name" 2> /dev/null || continue
        while slot_busy "$slot"; do sleep 30; done
        echo "[slot $slot] $(date '+%F %T') start $name"
        RAINBOW_PARALLEL=$PARALLEL scripts/run_slot_v2.sh "$slot" "$world" "$arm" "$seed" \
            --results-dir "$RESULTS_DIR" --run-name "$name" > "log/slots/${name}.log" 2>&1
        echo "[slot $slot] $(date '+%F %T') end $name (exit $?)"
    done
}

mkdir -p log/slots "$RESULTS_DIR"
echo "baselines parallel=$PARALLEL commit=$(git rev-parse --short HEAD) $(date '+%F %T')"
for ((s = 1; s <= PARALLEL; s++)); do worker "$s" & sleep 20; done
wait
echo "baselines finished $(date '+%F %T')"
