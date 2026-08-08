#!/usr/bin/env bash
# =============================================================================
#  load 门控性能战役 (setsid 脱离会话跑): 只在 load1 < 15 的窗口开火
#  优先级: GPU 批量一轮 -> CPU 512²×3/精度 -> 512² 若被拒重试 -> 256²/384² 补齐
#  每个 CPU rep 前后都查 load; rep 被拒 (ratio<=0.95) 自动重试 (每配置至多 6 发)
#  日志: logs/mhd/perf_baseline/watcher.log; CSV 沿用两个 battery 的格式
# =============================================================================
set -u
cd "$(dirname "$0")"
LOG=logs/mhd/perf_baseline
CSV=$LOG/cpu_timing.csv
SCR=/tmp/perf_quiet_$$
mkdir -p "$LOG" "$SCR"
exec >> $LOG/watcher.log 2>&1
echo "== watcher start $(date -Is) pid $$"

quiet() { [ "$(cut -d. -f1 /proc/loadavg)" -lt 15 ]; }
wait_quiet() { until quiet; do sleep 600; done; }

cpu_rep() {  # prec N rep_label -> 0 accepted / 1 rejected
  PREC=$1; N=$2; R=$3
  BIN=bin/mhd2d_${PREC}_eps1e-4_timed
  LOGF=$LOG/cpu_${PREC}_N${N}_q${R}.log
  ISO=$(date -Is)
  OUTDIR=$SCR/run /usr/bin/time -f "%e %U %S" -o $SCR/t.txt \
    ./$BIN $N $N 0.5 0.25 muscl hlld 0.18 > $LOGF 2>&1
  RC=$?
  read E U S < $SCR/t.txt
  LOOPW=$(grep -oP 'Time loop wall time = \K[0-9.]+' $LOGF | head -1)
  STEPS=$(grep -oP 'Done\. Steps: \K[0-9]+' $LOGF | head -1)
  FB=$(grep -oP 'fallbacks = \K[0-9]+' $LOGF | head -1)
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  RATIO=$(python3 -c "print(f'{(${U:-0}+${S:-0})/max(${E:-1},0.001):.3f}')")
  ACC=$(python3 -c "print('yes' if $RATIO>0.95 and $RC==0 else 'no')")
  echo "$ISO,cpu,$PREC,$N,q$R,${LOOPW:-},${E:-},${U:-},${S:-},$RATIO,$LOAD,${STEPS:-},${FB:-},$ACC" >> "$CSV"
  rm -rf $SCR/run
  [ "$ACC" = "yes" ]
}

collect() {  # prec N need -> 收满 need 个 accepted, 每次前置 wait_quiet, 至多 6 发
  PREC=$1; N=$2; NEED=$3; got=0
  for try in 1 2 3 4 5 6; do
    [ $got -ge $NEED ] && break
    wait_quiet
    echo "$(date -Is) firing cpu $PREC $N try$try (load $(cut -d' ' -f1 /proc/loadavg))"
    if cpu_rep $PREC $N $try; then got=$((got+1)); fi
  done
  echo "$(date -Is) cpu $PREC $N done: $got/$NEED accepted"
}

wait_quiet
echo "$(date -Is) quiet window, GPU battery"
./run_baseline_perf_gpu.sh

collect fp64 512 3
collect fp32 512 3
collect fp64 256 4
collect fp32 256 4
collect fp64 384 3
collect fp32 384 3
echo "== watcher complete $(date -Is)"
rm -rf $SCR
