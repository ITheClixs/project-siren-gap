#!/bin/bash
# Keeps 53_resume_s8_decodes.sh alive. The first run of this chain died silently at
# shard_011008 after its throughput fell from 210 s to 824 s per shard -- not OOM (the generator
# holds ~570 MB against 66% free) and not a crash, since nothing was written to the log. The
# surviving explanation is the process losing its parent, which nohup does not always prevent.
#
# Every step in the chain is resumable: 03_generate_inrbench skips completed shards, the W12
# artifacts are guarded by their own existence checks, and 48_s8_sweep.py refuses to append rows
# already in the ledger. So relaunching is safe as long as nothing is mid-flight, which the
# chain's own busy() guard already handles.
#
# Launch detached:  nohup caffeinate -i bash scripts/54_chain_watchdog.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG=results/s8/run_master.log
WATCH=results/s8/run_watchdog.log
CHAIN=scripts/53_resume_s8_decodes.sh

for _ in $(seq 1 288); do  # 288 * 5 min = 24 h ceiling
  sleep 300
  if grep -q "resume chain complete" "$LOG" 2>/dev/null; then
    echo "$(date): chain complete, watchdog exiting" >> "$WATCH"
    exit 0
  fi
  if pgrep -f "$CHAIN" >/dev/null 2>&1; then
    continue
  fi
  # The chain shell is gone. If a step is still running on its own, leave it alone -- the
  # relaunched chain would wait on busy() anyway, but starting a second shell is pointless.
  if pgrep -f "(03_generate_inrbench|48_s8_sweep|35_s5_pareto)[.]py" >/dev/null 2>&1; then
    echo "$(date): chain shell gone but a step is still running; waiting" >> "$WATCH"
    continue
  fi
  echo "$(date): chain not running and not complete -- relaunching" >> "$WATCH"
  nohup caffeinate -i bash "$CHAIN" >/dev/null 2>&1 &
  sleep 60
done
echo "$(date): watchdog hit its 24 h ceiling" >> "$WATCH"
