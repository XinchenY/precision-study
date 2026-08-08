#!/usr/bin/env bash
# GPU baseline v2 batch — 2026-07-23
# 判据放宽(用户裁定):不再等 load<15;验收条件 = 无竞争 GPU 进程。
#   nvidia-smi 因 driver/library mismatch (535.288 kernel / 535.309 userspace) 不可用,
#   竞争检测改为每 rep 记录 fuser /dev/nvidia0 的持有进程数(发射前采样,应为 0),
#   同时记录主机 load1,供"主机 load 无碍"论证(range 紧 => 质疑不攻自破)。
# 6 配置 = {fp64,fp32} x {256,384,512}²,各 2 warmup + 10 timed reps。
# 口径:loop_wall_s = 二进制打印的 "Time loop wall time"(纯时间环,与 CPU timed 补丁同口径)。
# 输出:新文件 gpu_timing_v2.csv + 每 rep 原始 log;不触碰任何既有数据文件。
set -u
PROJ=/lsc/zeushome/xy382/project
BIN=$PROJ/bin
LOGD=$PROJ/logs/mhd/perf_baseline
CSV=$LOGD/gpu_timing_v2.csv
BATCHLOG=$LOGD/gpu_v2_batch.log
WORK=${GPU_V2_WORKDIR:?set GPU_V2_WORKDIR}   # .dat 落在 scratch,不污染 results/
mkdir -p "$WORK"
cd "$WORK" || exit 1

[ -f "$CSV" ] || echo "iso,backend,prec,N,rep,loop_wall_s,e2e_wall_s,load1,gpu_procs,steps,fallbacks,exit" > "$CSV"
echo "== gpu-v2 start $(date -Is) pid $$ (load $(cut -d' ' -f1 /proc/loadavg))" >> "$BATCHLOG"

for prec in fp64 fp32; do
  for N in 256 384 512; do
    exe=$BIN/mhd2d_glm_gpu_${prec}_eps1e-4
    for w in 1 2; do
      "$exe" "$N" "$N" 0.5 0.25 muscl hlld 0.18 > /dev/null 2>&1
    done
    echo "$(date -Is) $prec N$N warmup x2 done" >> "$BATCHLOG"
    for r in 1 2 3 4 5 6 7 8 9 10; do
      lg=$LOGD/gpu_${prec}_N${N}_v2r${r}.log
      load=$(cut -d' ' -f1 /proc/loadavg)
      gp=$(fuser /dev/nvidia0 2>/dev/null | wc -w)
      t0=$(date +%s.%N)
      "$exe" "$N" "$N" 0.5 0.25 muscl hlld 0.18 > "$lg" 2>&1
      rc=$?
      t1=$(date +%s.%N)
      e2e=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
      loop=$(sed -n 's/^Time loop wall time = \([0-9.]*\) s.*/\1/p' "$lg")
      steps=$(sed -n 's/^Done\. Steps: \([0-9]*\),.*/\1/p' "$lg")
      fb=$(sed -n 's/^HLLD->HLL fallbacks = \([0-9]*\) .*/\1/p' "$lg")
      echo "$(date -Is),gpu,$prec,$N,v2r${r},${loop:-NA},${e2e},${load},${gp},${steps:-NA},${fb:-NA},${rc}" >> "$CSV"
      echo "$(date -Is) $prec N$N r$r loop=${loop:-NA}s e2e=${e2e}s load=$load gpu_procs=$gp rc=$rc" >> "$BATCHLOG"
    done
  done
done
echo "== gpu-v2 complete $(date -Is)" >> "$BATCHLOG"
