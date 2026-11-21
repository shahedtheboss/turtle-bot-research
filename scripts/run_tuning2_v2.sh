#!/usr/bin/env bash
# Study v2, second tuning stage (docs/v2_tuning_protocol.md, 2026-11-21): 2 extra settings x 2 values per algorithm,
# learning rate fixed at the first-stage choice (TUNED_LR), 2 tuning seeds -> 32 runs in world 4, each stopped once
# E* is met. Runs already finished are skipped, so the script can be restarted.
#
#   scripts/run_tuning2_v2.sh                 # PARALLEL=4 by default
#   DRY_RUN=1 scripts/run_tuning2_v2.sh       # list the jobs only
# Output: results_tune2_v2/t2_w4_<arm>_<setting>_s<seed>/
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PARALLEL="${PARALLEL:-4}"
EPISODES=500
WORLD=4
RESULTS_DIR=results_tune2_v2
export V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005      # CPU setting, as in every v2 run (DQN_DEVICE=cuda via run_slot_v2.sh)

# job = "arm name flags..."; per seed, settings interleaved across algorithms so each slot mixes Rainbow and the rest
jobs=()
for seed in 101 102; do
    for i in 0 1 2 3; do
        ed=$([ $((i / 2)) = 0 ] && echo 10000 || echo 20000); tu=$([ $((i % 2)) = 0 ] && echo 250 || echo 1000)
        sg=$([ $((i / 2)) = 0 ] && echo 0.5 || echo 0.25);   ns=$([ $((i % 2)) = 0 ] && echo 3 || echo 1)
        for arm in dqn ddqn d3qn; do
            jobs+=("$arm t2_w${WORLD}_${arm}_ed${ed}_tu${tu}_s${seed} $seed --eps-decay $ed --target-update $tu")
        done
        jobs+=("rainbow t2_w${WORLD}_rainbow_sg${sg}_ns${ns}_s${seed} $seed --sigma0 $sg --n-step $ns")
    done
done

is_done() {
    local m="$RESULTS_DIR/$1/$1_run_meta.json"
    [ -f "$m" ] && grep -q '"status": "completed"' "$m" && [ -f "$RESULTS_DIR/$1/$1_final_eval.csv" ]
}
slot_busy() {
    pgrep -f "run_slot_v2.sh $1 " > /dev/null && return 0
    for p in $(pgrep gzserver || true); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GAZEBO_MASTER_URI=http://localhost:$((11345 + $1))" && return 0
    done
    return 1
}

if [ "${DRY_RUN:-0}" = 1 ]; then
    for j in "${jobs[@]}"; do read -r arm name _ <<< "$j"; is_done "$name" || echo "would run: $j"; done
    echo "${#jobs[@]} jobs"; exit 0
fi

CLAIMS="$(mktemp -d)"; trap 'rm -rf "$CLAIMS"' EXIT
worker() {
    local slot=$1
    for j in "${jobs[@]}"; do
        read -r arm name seed flags <<< "$j"
        is_done "$name" && continue
        mkdir "$CLAIMS/$name" 2> /dev/null || continue
        while slot_busy "$slot"; do sleep 30; done
        echo "[slot $slot] $(date '+%F %T') start $name"
        # shellcheck disable=SC2086
        RAINBOW_PARALLEL=$PARALLEL scripts/run_slot_v2.sh "$slot" "$WORLD" "$arm" "$seed" \
            --episodes "$EPISODES" --eval-every 20 $flags --stop-when-learned \
            --results-dir "$RESULTS_DIR" --run-name "$name" --overwrite > "log/slots/${name}.log" 2>&1
        echo "[slot $slot] $(date '+%F %T') end $name (exit $?)"
    done
}
mkdir -p log/slots "$RESULTS_DIR"
echo "tuning stage 2: parallel=$PARALLEL commit=$(git rev-parse --short HEAD) $(date '+%F %T')"
for ((s = 1; s <= PARALLEL; s++)); do worker "$s" & sleep 20; done
wait
echo "tuning stage 2 finished $(date '+%F %T')"
