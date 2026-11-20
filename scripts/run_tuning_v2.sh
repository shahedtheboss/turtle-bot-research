#!/usr/bin/env bash
# Study v2 tuning (docs/v2_tuning_protocol.md, option a): 4 algorithms x 3 learning rates x 2 tuning seeds
# = 24 runs (20 after dropping lr 5e-5 on seed 102) in world 4 (moving), each stopped once E* is met (--stop-when-learned), PARALLEL slots at once.
#
#   scripts/run_tuning_v2.sh            # all 24 runs, 4 slots
#   PARALLEL=3 scripts/run_tuning_v2.sh
#
# Runs already finished (status completed in run_meta.json) or still running are skipped, so the script can be
# restarted; a worker waits until its slot's Gazebo from an earlier launcher has stopped.
# Output: results_tune_v2/tune_w4_<arm>_lr<lr>_s<seed>/
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PARALLEL="${PARALLEL:-4}"
EPISODES="${EPISODES:-500}"
WORLD=4
# CPU tweak (adopted 2026-11-20): /clock 200 Hz, 5 ms wait poll; ~15% faster at 4 slots, end poses within 0.6 cm of
# the v1 real-time reference (results_smoke/b3/x3_sim_r10_tweak.csv). Main runs must use the same values.
export V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005

# job list; order mixes algorithms so every slot gets a similar share of slow (Rainbow) runs
jobs=()
for seed in 101 102; do
    for lr in 1e-4 5e-5 2.5e-4; do
        # dropped 2026-11-20 by the author for time (protocol deviation, see docs/v2_tuning_protocol.md):
        # learning rate 5e-5 on seed 102 (only seed 101 is run at 5e-5)
        [ "$lr" = 5e-5 ] && [ "$seed" = 102 ] && continue
        for arm in dqn ddqn d3qn rainbow; do
            jobs+=("$arm $lr $seed")
        done
    done
done

slot_busy() {   # is this slot still used by an earlier launcher's run (its script, or its Gazebo)?
    # the script check matters: run_slot_v2's cleanup stops its Gazebo twice, 5 s apart, so "no Gazebo" alone
    # let a new run start in between and get killed (2026-11-20, two lost starts)
    pgrep -f "run_slot_v2.sh $1 " > /dev/null && return 0
    for p in $(pgrep gzserver || true); do
        tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GAZEBO_MASTER_URI=http://localhost:$((11345 + $1))" && return 0
    done
    return 1
}

worker() {   # slot: takes every PARALLEL-th job, starting at job (slot - 1)
    local slot=$1
    for ((i = slot - 1; i < ${#jobs[@]}; i += PARALLEL)); do
        read -r arm lr seed <<< "${jobs[$i]}"
        name="tune_w${WORLD}_${arm}_lr${lr}_s${seed}"
        meta="results_tune_v2/$name/${name}_run_meta.json"
        if [ -f "$meta" ] && grep -q '"status": "completed"' "$meta" \
                && [ -f "results_tune_v2/$name/${name}_final_eval.csv" ]; then
            echo "[slot $slot] skip $name (done)"; continue
        fi
        if pgrep -f -- "--run-name $name " > /dev/null; then
            echo "[slot $slot] skip $name (already running from an earlier launcher)"; continue
        fi
        while slot_busy "$slot"; do sleep 30; done      # an earlier launcher's run still uses this slot
        echo "[slot $slot] $(date +%T) start $name"
        RAINBOW_PARALLEL=$PARALLEL scripts/run_slot_v2.sh "$slot" "$WORLD" "$arm" "$seed" \
            --episodes "$EPISODES" --eval-every 20 --lr "$lr" --stop-when-learned \
            --results-dir results_tune_v2 --run-name "$name" --overwrite \
            > "log/slots/${name}.log" 2>&1
        echo "[slot $slot] $(date +%T) end $name (exit $?)"
    done
}

mkdir -p log/slots results_tune_v2
for ((s = 1; s <= PARALLEL; s++)); do
    worker "$s" &
    sleep 20          # stagger Gazebo start-ups
done
wait
echo "tuning finished $(date +%T)"
