#!/usr/bin/env bash
# Study v2 main runs: work through scripts/main_runs_v2.csv (125 runs = 5 worlds x 5 arms x 5 seeds) in
# PARALLEL isolated Gazebo slots. Each run: 500 episodes (no early stop), greedy check every 20, final eval.
#
#   scripts/run_main_v2.sh labpc                 # this machine's planned runs (seeds 1-3, 75 runs)
#   scripts/run_main_v2.sh laptop                # seeds 4-5, 50 runs
#   scripts/run_main_v2.sh laptop --take-over    # own runs first, then EVERY other run not done here
#   PARALLEL=6 scripts/run_main_v2.sh labpc      # slots at once (default 4; max 8)
#   DRY_RUN=1 scripts/run_main_v2.sh laptop --take-over   # only print what would run
#
# Planned split (pre-registered): lab PC seeds 1-3, laptop seeds 4-5. Whole seeds per machine, so every
# arm and world gets the same share of each machine and a machine difference cannot favour one method.
# Fallback (also pre-registered): if a machine is not available long enough, the other machine runs the
# remaining runs with --take-over. A run is "done" when results_v2/<run>/ has run_meta.json with status
# completed AND the final-eval CSV; copy finished run folders between machines BEFORE a take-over, so
# nothing is run twice. Every run records its machine (hostname) and git commit in run_meta.json.
# Restartable: done runs are skipped, runs still in progress are skipped, interrupted runs start again.
set -o pipefail
if [ $# -lt 1 ]; then echo "usage: $0 labpc|laptop [--take-over]" >&2; exit 1; fi
MACHINE=$1; TAKE_OVER=0; [ "${2:-}" = "--take-over" ] && TAKE_OVER=1
case "$MACHINE" in labpc|laptop) ;; *) echo "machine must be labpc or laptop" >&2; exit 1 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PARALLEL="${PARALLEL:-4}"
EPISODES=500
RESULTS_DIR=results_v2
# CPU setting adopted 2026-11-20 (docs/v2_tuning_protocol.md): identical for every tuning and main run
export V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005

# job list: own rows first (file order), then, with --take-over, the other machine's rows
jobs=()
while IFS=, read -r name world arm seed machine; do
    machine=${machine%$'\r'}                      # tolerate Windows line endings
    [ "$name" = run_name ] && continue
    [ "$machine" = "$MACHINE" ] && jobs+=("$name $world $arm $seed")
done < scripts/main_runs_v2.csv
if [ $TAKE_OVER = 1 ]; then
    while IFS=, read -r name world arm seed machine; do
        machine=${machine%$'\r'}
        [ "$name" = run_name ] && continue
        [ "$machine" != "$MACHINE" ] && jobs+=("$name $world $arm $seed")
    done < scripts/main_runs_v2.csv
fi

is_done() {
    local m="$RESULTS_DIR/$1/$1_run_meta.json"
    [ -f "$m" ] && grep -q '"status": "completed"' "$m" && [ -f "$RESULTS_DIR/$1/$1_final_eval.csv" ]
}
slot_busy() {   # a run script, or its Gazebo, still uses this slot (cleanup stops Gazebo twice, 5 s apart)
    pgrep -f "run_slot_v2.sh $1 " > /dev/null && return 0
    for p in $(pgrep gzserver || true); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GAZEBO_MASTER_URI=http://localhost:$((11345 + $1))" && return 0
    done
    return 1
}

if [ "${DRY_RUN:-0}" = 1 ]; then
    n=0; for j in "${jobs[@]}"; do read -r name _ <<< "$j"; is_done "$name" || { echo "would run: $j"; n=$((n + 1)); }; done
    echo "$n runs to do on $MACHINE (take-over=$TAKE_OVER), ${#jobs[@]} in its list"; exit 0
fi

# one shared queue: a worker claims the next job with mkdir (atomic), so slots never take the same run
CLAIMS="$(mktemp -d)"
trap 'rm -rf "$CLAIMS"' EXIT
worker() {
    local slot=$1
    for j in "${jobs[@]}"; do
        read -r name world arm seed <<< "$j"
        is_done "$name" && continue
        pgrep -f -- "--run-name $name " > /dev/null && continue        # running from another launcher
        mkdir "$CLAIMS/$name" 2> /dev/null || continue                   # another slot took it
        while slot_busy "$slot"; do sleep 30; done
        echo "[slot $slot] $(date '+%F %T') start $name"
        RAINBOW_PARALLEL=$PARALLEL scripts/run_slot_v2.sh "$slot" "$world" "$arm" "$seed" \
            --episodes "$EPISODES" --eval-every 20 --results-dir "$RESULTS_DIR" --run-name "$name" --overwrite \
            > "log/slots/${name}.log" 2>&1
        echo "[slot $slot] $(date '+%F %T') end $name (exit $?)"
    done
}

mkdir -p log/slots "$RESULTS_DIR"
echo "machine=$MACHINE take-over=$TAKE_OVER parallel=$PARALLEL commit=$(git rev-parse --short HEAD) $(date '+%F %T')"
for ((s = 1; s <= PARALLEL; s++)); do
    worker "$s" &
    sleep 20          # stagger Gazebo start-ups
done
wait
echo "main runs finished on $MACHINE $(date '+%F %T')"
