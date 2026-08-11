#!/bin/bash
# Restarts a corpus chain that has *degraded*, not just died.
#
# Three times now, long MPS runs on this machine have driven swap to ~80% and collapsed shard
# throughput by one to two orders of magnitude (265 s -> 2470 s, 145 s -> 22300 s) while the
# process stayed alive with a small and shrinking RSS. The existing watchdog only detects a dead
# process, so it slept through all three. Roughly nineteen hours of compute went to this.
#
# The generator is resumable and skips completed shards, and a fresh process gets a clean
# allocator, so the fix is to notice the stall and restart. A restart costs one in-flight shard.
#
# Usage:  nohup caffeinate -i bash scripts/69_stall_watchdog.sh <chain-script> <log> &
set -u
cd "$(dirname "$0")/.."
CHAIN="${1:?chain script}"
LOG="${2:?log file}"
WATCH=results/stall_watchdog.log
MIN_IDLE=900          # never intervene before this many seconds of silence
FACTOR=4              # or this multiple of the recent median shard time, whichever is larger

median_shard_seconds() {
  grep -oE "in [0-9]+\.[0-9]+s" "$LOG" 2>/dev/null | tail -20 | grep -oE "[0-9]+\.[0-9]+" \
    | sort -n | awk '{a[NR]=$1} END {if (NR) printf "%d", a[int((NR+1)/2)]; else print 0}'
}

# A resuming chain writes no shard line while it rescans completed shards and loads the dataset,
# which on this machine can exceed twenty minutes. Judging idleness from the moment of arming
# therefore kills healthy jobs -- the first version of this script did exactly that. The clock
# only starts once the *current* run has written a shard line of its own.
GRACE=2400
armed_at=$(date +%s)
baseline=$(grep -c "shard_" "$LOG" 2>/dev/null || echo 0)

echo "$(date): watching $CHAIN via $LOG (grace ${GRACE}s, baseline ${baseline} shards)" >> "$WATCH"
while true; do
  sleep 120
  if ! pgrep -f "$CHAIN" >/dev/null 2>&1; then
    grep -q "complete" "$LOG" 2>/dev/null && { echo "$(date): chain finished" >> "$WATCH"; exit 0; }
    echo "$(date): chain gone, relaunching" >> "$WATCH"
    nohup caffeinate -i bash "$CHAIN" >/dev/null 2>&1 &
    sleep 300; continue
  fi

  now=$(date +%s)
  # still inside the startup grace window, and this run has not written a shard yet
  if [ "$(grep -c "shard_" "$LOG" 2>/dev/null || echo 0)" -le "$baseline" ]; then
    [ $(( now - armed_at )) -lt "$GRACE" ] && continue
  fi

  med=$(median_shard_seconds)
  [ "${med:-0}" -gt 0 ] || continue
  limit=$(( med * FACTOR )); [ "$limit" -lt "$MIN_IDLE" ] && limit=$MIN_IDLE
  idle=$(( $(date +%s) - $(stat -f %m "$LOG") ))

  if [ "$idle" -gt "$limit" ]; then
    swap=$(sysctl -n vm.swapusage | grep -oE "used = [0-9.]+M" | grep -oE "[0-9.]+")
    echo "$(date): STALL - idle ${idle}s > ${limit}s (median ${med}s), swap ${swap}M; restarting" \
      >> "$WATCH"
    pkill -f "$CHAIN"; sleep 2
    pkill -f "03_generate_inrbench"; sleep 15
    nohup caffeinate -i bash "$CHAIN" >/dev/null 2>&1 &
    armed_at=$(date +%s)
    baseline=$(grep -c "shard_" "$LOG" 2>/dev/null || echo 0)
    sleep 600   # let the fresh process get past its first shard before judging it again
  fi
done
