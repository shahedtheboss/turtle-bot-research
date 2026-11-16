#!/usr/bin/env bash
# Run the remaining DQN-comparison batches one after another, unattended.
#
#   scripts/run_queue.sh            # start (normally launched detached, see commands.md §7)
#   DRY_RUN=1 scripts/run_queue.sh  # print what it would do, start nothing
#
# What it does, in order:
#   1. Blocks system suspend, idle sleep and the lid switch for as long as it runs
#      (systemd-inhibit; no sudo needed). The screen may still blank/lock; that's harmless.
#   2. Waits until no run_slot.sh is running (i.e. the batch you started by hand has finished).
#   3. For each batch below: waits until enough RAM is free, starts its runs in parallel
#      (one isolated slot each, exactly the same run_slot.sh command you'd type), waits for
#      all of them, then moves on. A failed run is logged and the queue continues.
# Logs: log/queue/queue.log (overview) and log/queue/<arm>_s<seed>.out (one per run).
# Stop everything: pkill -f run_queue.sh; pkill -f run_slot.sh   (each slot cleans up its Gazebo)
#
# Batch order = commands.md §7 (pre-registered plan, slots rotated between batches).

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOG_DIR="$REPO/log/queue"; mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue.log"
DRY_RUN="${DRY_RUN:-0}"

# Each batch is a space-separated list of "arm:seed", in slot order 1..n.
BATCHES=(
  "rainbow_g:3 rainbow:3 d3qn_g:3 d3qn:3 ddqn:3 dqn:3"          # batch 3
  "d3qn_g:1 rainbow:1 rainbow_g:1"                               # batch 4
  "ddqn:4 dqn:4 rainbow_g:4 rainbow:4 d3qn_g:4 d3qn:4"          # batch 5
  "d3qn_g:5 rainbow_g:5 dqn:5 ddqn:5 d3qn:5 rainbow:5"          # batch 6
)
RAM_PER_RUN_MB=1400     # measured: ~1.07 GB python + 0.27 GB gzserver
RAM_MARGIN_MB=1500

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

# Re-run ourselves under systemd-inhibit so nothing suspends the machine while we work.
if [ -z "${QUEUE_INHIBITED:-}" ] && [ "$DRY_RUN" != "1" ]; then
    export QUEUE_INHIBITED=1
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch --who="dqn_comparison queue" \
        --why="training batches running" --mode=block "$0" "$@"
fi

mem_available_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }

wait_for_no_slots() {
    while pgrep -f "scripts/run_slot.sh" > /dev/null; do sleep 60; done
}

log "queue started (pid $$, dry_run=$DRY_RUN, ${#BATCHES[@]} batches)"
if [ "$DRY_RUN" != "1" ]; then
    log "waiting for the currently running batch to finish..."
    wait_for_no_slots
    sleep 30   # let the last Gazebo shut down fully
fi

b=0
for batch in "${BATCHES[@]}"; do
    b=$((b + 1)); read -r -a runs <<< "$batch"; n=${#runs[@]}
    need=$(( n * RAM_PER_RUN_MB + RAM_MARGIN_MB ))
    log "batch $b/${#BATCHES[@]}: ${runs[*]} (parallel=$n, needs ~${need} MB RAM)"
    if [ "$DRY_RUN" != "1" ]; then
        while [ "$(mem_available_mb)" -lt "$need" ]; do
            log "  only $(mem_available_mb) MB available, waiting (close Chrome?)"; sleep 120
        done
    fi
    pids=(); names=()
    slot=0
    for r in "${runs[@]}"; do
        slot=$((slot + 1)); arm="${r%%:*}"; seed="${r##*:}"; name="${arm}_s${seed}"
        if [ -d "results/$name" ] && [ -n "$(ls -A "results/$name" 2>/dev/null)" ]; then
            log "  skip $name: results/$name already has files (re-run by hand with --run-name ${name}_rerun)"
            continue
        fi
        cmd="RAINBOW_PARALLEL=$n scripts/run_slot.sh $slot $arm $seed"
        if [ "$DRY_RUN" = "1" ]; then log "  would run: $cmd"; continue; fi
        log "  start slot $slot: $name"
        RAINBOW_PARALLEL=$n setsid scripts/run_slot.sh "$slot" "$arm" "$seed" > "$LOG_DIR/$name.out" 2>&1 &
        pids+=($!); names+=("$name")
        sleep 20   # stagger Gazebo start-ups
    done
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then log "  done: ${names[$i]}"
        else log "  FAILED (exit $?): ${names[$i]}, see $LOG_DIR/${names[$i]}.out"; fi
    done
    [ "$DRY_RUN" != "1" ] && sleep 30
done
log "queue finished"
