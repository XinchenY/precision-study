#!/usr/bin/env bash
# GPU-A30 机会主义计时: load<15 窗口开火一轮批量; 每小时心跳(可见死亡时刻)
set -u; cd "$(dirname "$0")"
exec >> logs/mhd/perf_baseline/gpu_watch2.log 2>&1
exec 8>/tmp/xy382_gpu_watch.lock
flock -n 8 || exit 0
echo "== gpu-watch2 start $(date -Is) pid $$"
n=0
while :; do
  L=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  if [ "$L" -lt 8 ]; then
    echo "$(date -Is) quiet (load $L), firing GPU battery"
    ./run_baseline_perf_gpu.sh
    echo "$(date -Is) battery done"; break
  fi
  n=$((n+1)); [ $((n % 6)) -eq 0 ] && echo "$(date -Is) heartbeat load $L"
  sleep 600
done
