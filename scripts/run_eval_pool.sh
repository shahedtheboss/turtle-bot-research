#!/usr/bin/env bash
# Evaluation pool: starts the final evaluation of each training run as soon as that run
# has COMPLETED and one of the 6 slots is free. Shares the 6 slots with training
# (run_slot.sh), so training and evaluation together never exceed 6 Gazebos.
# Also evaluates the heuristic baselines H and H_eta.
#
#   scripts/run_eval_pool.sh             # normally launched detached (commands.md §8)
#   DRY_RUN=1 scripts/run_eval_pool.sh   # show what is ready / waiting, start nothing
#
# A run is "ready" when results/<run>/<run>_run_meta.json says status=completed and
# results/<run>/<run>_final_eval.csv does not exist yet. Each target is started once.
# Exits when every target was started (then waits for them), or when nothing is left
# that could still become ready (no training running).
# Logs: log/queue/eval_pool.log and log/queue/eval_<run>.out
# Stop: kill the pool by PID (see commands.md), then the run_eval_slot.sh PIDs.

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOG_DIR="$REPO/log/queue"; mkdir -p "$LOG_DIR"
PLOG="$LOG_DIR/eval_pool.log"
DRY_RUN="${DRY_RUN:-0}"
MAX_SLOTS=6
RAM_PER_RUN_MB=1400
RAM_MARGIN_MB=1500

TARGETS=(H H_eta)
for S in 1 2 3 4 5; do for A in dqn ddqn d3qn d3qn_g rainbow rainbow_g; do TARGETS+=("${A}_s${S}"); done; done

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PLOG"; }

if [ -z "${EVAL_POOL_INHIBITED:-}" ] && [ "$DRY_RUN" != "1" ]; then
    export EVAL_POOL_INHIBITED=1
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch --who="dqn_comparison eval pool" \
        --why="final evaluations in progress" --mode=block "$0" "$@"
fi

mem_available_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
# Slots used by training (run_slot.sh) OR evaluation (run_eval_slot.sh).
used_slots() {
    pgrep -af "scripts/run_(eval_)?slot\.sh [1-6] " | grep -v pgrep | \
        sed -n 's/.*scripts\/run_\(eval_\)\{0,1\}slot\.sh \([1-6]\) .*/\2/p' | sort -u
}
free_slot() {
    local used; used="$(used_slots)"
    for s in $(seq 1 "$MAX_SLOTS"); do echo "$used" | grep -qx "$s" || { echo "$s"; return; }; done
}
training_running() { pgrep -f "scripts/run_slot\.sh [1-6] " > /dev/null; }
has_eval() { [ -f "results/$1/$1_final_eval.csv" ]; }
is_ready() {   # H/H_eta always ready; runs once training completed
    case "$1" in H|H_eta) return 0 ;; esac
    grep -q '"status": *"completed"' "results/$1/$1_run_meta.json" 2>/dev/null
}

declare -A started
for t in "${TARGETS[@]}"; do has_eval "$t" && started[$t]=done; done

log "eval pool started (pid $$, dry_run=$DRY_RUN)"
if [ "$DRY_RUN" = "1" ]; then
    for t in "${TARGETS[@]}"; do
        if [ -n "${started[$t]:-}" ]; then st="has eval"; elif is_ready "$t"; then st="READY"; else st="waiting (training not completed)"; fi
        log "  $t: $st"
    done
    log "slots in use: $(used_slots | tr '\n' ' ')  next free: $(free_slot)"
    exit 0
fi

while :; do
    todo=0; next=""
    for t in "${TARGETS[@]}"; do
        [ -n "${started[$t]:-}" ] && continue
        todo=$((todo + 1))
        if [ -z "$next" ] && is_ready "$t"; then next="$t"; fi
    done
    [ "$todo" -eq 0 ] && break
    if [ -z "$next" ]; then
        if ! training_running; then log "nothing ready and no training running: $todo target(s) can never become ready; stopping"; break; fi
        sleep 60; continue
    fi
    slot="$(free_slot)"
    if [ -z "$slot" ]; then sleep 60; continue; fi
    if [ "$(mem_available_mb)" -lt $(( RAM_PER_RUN_MB + RAM_MARGIN_MB )) ]; then
        log "slot $slot free but only $(mem_available_mb) MB RAM available; waiting"; sleep 120; continue
    fi
    log "start eval slot $slot: $next  ($todo left incl. this one)"
    setsid scripts/run_eval_slot.sh "$slot" "$next" > "$LOG_DIR/eval_$next.out" 2>&1 &
    started[$next]=$!
    sleep 30   # let it register (pgrep) and stagger Gazebo start-ups
done
log "all evaluations started; waiting for them to finish"
wait
for t in "${TARGETS[@]}"; do has_eval "$t" || log "MISSING final eval: $t (see $LOG_DIR/eval_$t.out)"; done
log "eval pool finished"
