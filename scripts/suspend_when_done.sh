#!/usr/bin/env bash
# Suspend the laptop once ALL training and evaluation work has finished.
#
#   setsid nohup scripts/suspend_when_done.sh > log/queue/suspend_watch.out 2>&1 < /dev/null &
#
# "Finished" = no run_pool.sh, run_eval_pool.sh, run_slot.sh or run_eval_slot.sh process
# for 2 checks in a row (2 min apart). Then it logs a summary and runs `systemctl suspend`
# (allowed without sudo for the logged-in user: login1 CanSuspend = yes).
# It holds NO inhibitor itself. Cancel: kill its PID (pgrep -af suspend_when_done).
# Log: log/queue/suspend_watch.log

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOG="$REPO/log/queue/suspend_watch.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

busy() { pgrep -f "scripts/run_(eval_)?(pool|slot)\.sh" > /dev/null; }

log "watcher started (pid $$): will suspend when all training and evaluation processes are gone"
idle_checks=0
while [ "$idle_checks" -lt 2 ]; do
    sleep 120
    if busy; then idle_checks=0; else idle_checks=$((idle_checks + 1)); fi
done

n_eval=$(ls results/*/*_final_eval.csv 2>/dev/null | wc -l)
n_done=$(grep -l '"status": *"completed"' results/*_s[1-5]/*_run_meta.json 2>/dev/null | wc -l)
log "all work finished: $n_done completed training runs, $n_eval final evaluations; suspending"
tail -3 log/queue/eval_pool.log >> "$LOG" 2>/dev/null
sync
systemctl suspend
log "suspend command returned (exit $?)"
