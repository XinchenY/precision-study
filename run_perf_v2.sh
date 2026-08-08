#!/usr/bin/env bash
# CPU 计时收集 v2: 不设 load 前置门槛(load>=55 才等), 以 cpu_ratio>0.95 事后判收
set -u
cd "$(dirname "$0")"
LOG=logs/mhd/perf_baseline; CSV=$LOG/cpu_timing.csv
SCR=/tmp/perfv2_$$; mkdir -p $SCR
exec >> $LOG/watcher2.log 2>&1
echo "== v2 start $(date -Is) pid $$"
cpu_rep() {
  PREC=$1; N=$2; R=$3
  BIN=bin/mhd2d_${PREC}_eps1e-4_timed
  LOGF=$LOG/cpu_${PREC}_N${N}_v2r${R}.log
  ISO=$(date -Is)
  OUTDIR=$SCR/run /usr/bin/time -f "%e %U %S" -o $SCR/t.txt \
    ./$BIN $N $N 0.5 0.25 muscl hlld 0.18 > $LOGF 2>&1
  RC=$?; read E U S < $SCR/t.txt
  LOOPW=$(grep -oP 'Time loop wall time = \K[0-9.]+' $LOGF | head -1)
  STEPS=$(grep -oP 'Done\. Steps: \K[0-9]+' $LOGF | head -1)
  FB=$(grep -oP 'fallbacks = \K[0-9]+' $LOGF | head -1)
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  RATIO=$(python3 -c "print(f'{(${U:-0}+${S:-0})/max(${E:-1},0.001):.3f}')")
  ACC=$(python3 -c "print('yes' if $RATIO>0.95 and $RC==0 else 'no')")
  echo "$ISO,cpu,$PREC,$N,v2r$R,${LOOPW:-},${E:-},${U:-},${S:-},$RATIO,$LOAD,${STEPS:-},${FB:-},$ACC" >> "$CSV"
  rm -rf $SCR/run
  [ "$ACC" = "yes" ]
}
collect() {
  PREC=$1; N=$2; NEED=$3; MAXTRY=$4; got=0; try=0
  while [ $got -lt $NEED ] && [ $try -lt $MAXTRY ]; do
    L=$(cut -d. -f1 /proc/loadavg)
    if [ "$L" -ge 55 ]; then echo "$(date -Is) load $L, wait"; sleep 900; continue; fi
    try=$((try+1))
    echo "$(date -Is) fire $PREC $N try$try (load $L)"
    if cpu_rep $PREC $N $try; then got=$((got+1)); fi
  done
  echo "$(date -Is) $PREC $N: $got/$NEED accepted"
}
collect fp64 512 3 8
collect fp32 512 3 8
collect fp64 384 3 6
collect fp32 384 3 6
collect fp64 256 4 6
collect fp32 256 4 6
echo "== v2 complete $(date -Is)"
rm -rf $SCR
