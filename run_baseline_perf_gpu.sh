#!/usr/bin/env bash
# =============================================================================
#  Ch10 §10.4 baseline performance — GPU 批量 (A30, ε=1e-4)
#  6 配置 = {fp64,fp32} × {256,384,512}; 每配置 2 warm-up + 10 timed reps
#  记录: loop_wall(二进制自报, 时间环) / e2e wall / load / steps / fallbacks
#  GPU 侧无 cpu_ratio 判据 (host 大部分在等), 污染靠 IQR 事后识别
#  输出: logs/mhd/perf_baseline/gpu_timing.csv
# =============================================================================
set -u
cd "$(dirname "$0")"
exec 9>/tmp/xy382_gpu_battery.lock
flock -n 9 || { echo "another battery instance running, abort"; exit 1; }
if pgrep -u xy382 -f mhd2d_glm_gpu > /dev/null; then echo "GPU busy with our own solver, abort"; exit 1; fi
LOG=logs/mhd/perf_baseline
CSV=$LOG/gpu_timing.csv
SCRATCH=/tmp/claude-65995/-lsc-zeushome-xy382-project/ff12f7e1-a411-4f66-bbb2-6e79e18bec62/scratchpad/perf_gpu
mkdir -p "$LOG" "$SCRATCH"
[ -f "$CSV" ] || echo "iso,backend,prec,N,rep,loop_wall_s,e2e_wall_s,load1,steps,fallbacks" > "$CSV"

for PREC in fp64 fp32; do
  BIN=bin/mhd2d_glm_gpu_${PREC}_eps1e-4
  for N in 256 384 512; do
    for w in 1 2; do
      OUTDIR=$SCRATCH ./$BIN $N $N 0.5 0.25 muscl hlld 0.18 > /dev/null 2>&1
      rm -rf $SCRATCH/*.dat
    done
    for r in $(seq 1 10); do
      LOGF=$LOG/gpu_${PREC}_N${N}_rep${r}.log
      ISO=$(date -Is)
      TMPT=$SCRATCH/timing.txt
      OUTDIR=$SCRATCH /usr/bin/time -f "%e" -o $TMPT \
        ./$BIN $N $N 0.5 0.25 muscl hlld 0.18 > $LOGF 2>&1
      E=$(cat $TMPT)
      LOOPW=$(grep -oP 'Time loop wall time = \K[0-9.]+' $LOGF | head -1)
      STEPS=$(grep -oP 'Done\. Steps: \K[0-9]+' $LOGF | head -1)
      FB=$(grep -oP 'fallbacks = \K[0-9]+' $LOGF | head -1)
      LOAD=$(cut -d' ' -f1 /proc/loadavg)
      echo "$ISO,gpu,$PREC,$N,$r,${LOOPW:-},${E:-},$LOAD,${STEPS:-},${FB:-}" >> "$CSV"
      rm -rf $SCRATCH/*.dat
    done
  done
done
echo "GPU battery complete"
