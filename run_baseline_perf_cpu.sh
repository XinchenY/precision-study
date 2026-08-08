#!/usr/bin/env bash
# =============================================================================
#  Ch10 §10.4 baseline performance — CPU 单精度流 (每个精度一个实例, 串行 reps)
#  用法: ./run_baseline_perf_cpu.sh fp64|fp32
#  规程: 先 192² 逐位验证计时补丁无害 (与 9.2 正式数据比 md5), 失败即中止;
#        然后 256²×5 / 384²×3 / 512²×3 reps, 每个 rep 记录:
#        loop_wall(二进制自报) / e2e wall / user+sys CPU / cpu_ratio / load /
#        steps / fallbacks; cpu_ratio<0.95 标记 rejected (不自动重跑, 事后补)
#  输出: logs/mhd/perf_baseline/cpu_timing.csv (+ 每 rep .log)
# =============================================================================
set -u
cd "$(dirname "$0")"
PREC=${1:?usage: $0 fp64|fp32}
BIN=bin/mhd2d_${PREC}_eps1e-4_timed
REF=results/mhd/orszag_tang_large_eps1e-4/orszag_tang_glm_muscl_hlld_${PREC}_N192x192.dat
LOG=logs/mhd/perf_baseline
CSV=$LOG/cpu_timing.csv
SCRATCH=/tmp/claude-65995/-lsc-zeushome-xy382-project/ff12f7e1-a411-4f66-bbb2-6e79e18bec62/scratchpad/perf_$PREC
mkdir -p "$LOG" "$SCRATCH"
command -v /usr/bin/time >/dev/null || { echo "no /usr/bin/time"; exit 1; }
[ -f "$CSV" ] || echo "iso,backend,prec,N,rep,loop_wall_s,e2e_wall_s,user_s,sys_s,cpu_ratio,load1,steps,fallbacks,accepted" > "$CSV"

# ── 逐位验证: 计时补丁不得改变任何数据位 ───────────────────────────────────
OUTDIR=$SCRATCH/verify ./$BIN 192 192 0.5 0.25 muscl hlld 0.18 > $LOG/cpu_${PREC}_verify.log 2>&1 \
  || { echo "$PREC verify run failed"; exit 2; }
M1=$(grep -v '^#' $SCRATCH/verify/orszag_tang_glm_muscl_hlld_${PREC}_N192x192.dat | md5sum | cut -d' ' -f1)
M2=$(grep -v '^#' $REF | md5sum | cut -d' ' -f1)
rm -rf $SCRATCH/verify
if [ "$M1" != "$M2" ]; then echo "$PREC BITWISE VERIFY FAILED ($M1 vs $M2) — aborting"; exit 3; fi
echo "$PREC bitwise verify OK"

run_reps() {
  N=$1; REPS=$2
  for r in $(seq 1 $REPS); do
    LOGF=$LOG/cpu_${PREC}_N${N}_rep${r}.log
    TMPT=$SCRATCH/timing.txt
    ISO=$(date -Is)
    OUTDIR=$SCRATCH/run /usr/bin/time -f "%e %U %S" -o $TMPT \
      ./$BIN $N $N 0.5 0.25 muscl hlld 0.18 > $LOGF 2>&1
    RC=$?
    read E U S < $TMPT
    LOOPW=$(grep -oP 'Time loop wall time = \K[0-9.]+' $LOGF | head -1)
    STEPS=$(grep -oP 'Done\. Steps: \K[0-9]+' $LOGF | head -1)
    FB=$(grep -oP 'fallbacks = \K[0-9]+' $LOGF | head -1)
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    RATIO=$(python3 -c "print(f'{(${U:-0}+${S:-0})/max(${E:-1},0.001):.3f}')")
    ACC=$(python3 -c "print('yes' if $RATIO>0.95 and $RC==0 else 'no')")
    echo "$ISO,cpu,$PREC,$N,$r,${LOOPW:-},${E:-},${U:-},${S:-},$RATIO,$LOAD,${STEPS:-},${FB:-},$ACC" >> "$CSV"
    rm -rf $SCRATCH/run
  done
}

run_reps 256 5
run_reps 384 3
run_reps 512 3
echo "$PREC stream complete"
