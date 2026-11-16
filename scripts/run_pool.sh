#!/usr/bin/env bash
# Slot-pool scheduler for the remaining DQN-comparison runs (replaces run_queue.sh's
# batch-by-batch waiting). Keeps up to MAX_SLOTS runs going: whenever a slot (1..6) is
# free and enough RAM is available, it starts the NEXT run in the pre-registered order.
# Runs already running (e.g. started by hand) keep their slots and are left alone.
#
#   scripts/run_pool.sh             # normally launched detached (see commands.md §7)
#   DRY_RUN=1 scripts/run_pool.sh   # print the pending list and free slots, start nothing
#
# Same per-run command as by hand: RAINBOW_PARALLEL=6 scripts/run_slot.sh SLOT ARM SEED.
# Blocks suspend / idle sleep / lid switch while it runs (systemd-inhibit, no sudo).
# Skips any run whose results folder already has files.
# Logs: log/queue/pool.log and log/queue/<arm>_s<seed>.out
# Stop: pkill -f run_pool.sh; pkill -f run_slot.sh

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOG_DIR="$REPO/log/queue"; mkdir -p "$LOG_DIR"
PLOG="$LOG_DIR/pool.log"
DRY_RUN="${DRY_RUN:-0}"
MAX_SLOTS=6
RAM_PER_RUN_MB=1400
RAM_MARGIN_MB=1500

# Run order, as arm:seed. Pre-registered batch order (commands.md §7), EXCEPT that the
# unguided Rainbow runs (slowest, ~1.5 min/episode) are moved to the end (user request,
# 2026-11-16 15:50) so all other arms finish first. Runs with existing results are skipped.
PENDING=(
  rainbow_g:3 d3qn_g:3 d3qn:3 ddqn:3 dqn:3
  d3qn_g:1 rainbow_g:1
  ddqn:4 dqn:4 rainbow_g:4 d3qn_g:4 d3qn:4
  d3qn_g:5 rainbow_g:5 dqn:5 ddqn:5 d3qn:5
  rainbow:3 rainbow:1 rainbow:4 rainbow:5
)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PLOG"; }

if [ -z "${POOL_INHIBITED:-}" ] && [ "$DRY_RUN" != "1" ]; then
    export POOL_INHIBITED=1
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch --who="dqn_comparison pool" \
        --why="training runs in progress" --mode=block "$0" "$@"
fi

mem_available_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }

# Slots currently used by any run_slot.sh (ours or started by hand): "run_slot.sh SLOT ARM SEED".
used_slots() {
    pgrep -af "scripts/run_slot.sh" | grep -v pgrep | \
        sed -n 's/.*scripts\/run_slot\.sh \([1-6]\) .*/\1/p' | sort -u
}
free_slot() {
    local used; used="$(used_slots)"
    for s in $(seq 1 "$MAX_SLOTS"); do
        echo "$used" | grep -qx "$s" || { echo "$s"; return; }
    done
}

# Drop runs that already have results (finished, failed or running).
todo=()
for r in "${PENDING[@]}"; do
    name="${r%%:*}_s${r##*:}"
    if [ -d "results/$name" ] && [ -n "$(ls -A "results/$name" 2>/dev/null)" ]; then continue; fi
    todo+=("$r")
done
log "pool started (pid $$, dry_run=$DRY_RUN): ${#todo[@]} runs to do: ${todo[*]}"
log "slots in use now: $(used_slots | tr '\n' ' ')"
if [ "$DRY_RUN" = "1" ]; then log "next free slot: $(free_slot)"; exit 0; fi

i=0
while [ "$i" -lt "${#todo[@]}" ]; do
    slot="$(free_slot)"
    need=$(( RAM_PER_RUN_MB + RAM_MARGIN_MB ))
    if [ -z "$slot" ]; then sleep 60; continue; fi
    if [ "$(mem_available_mb)" -lt "$need" ]; then
        log "slot $slot free but only $(mem_available_mb) MB RAM available; waiting"; sleep 120; continue
    fi
    r="${todo[$i]}"; arm="${r%%:*}"; seed="${r##*:}"; name="${arm}_s${seed}"
    log "start slot $slot: $name  ($((i + 1))/${#todo[@]})"
    RAINBOW_PARALLEL=$MAX_SLOTS setsid scripts/run_slot.sh "$slot" "$arm" "$seed" \
        > "$LOG_DIR/$name.out" 2>&1 &
    i=$((i + 1))
    sleep 30   # let this slot register (pgrep) and stagger Gazebo start-ups
done
log "all runs started; waiting for them to finish"
wait
log "pool finished"
