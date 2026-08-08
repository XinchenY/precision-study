#!/usr/bin/env bash
# CPU 变体安静门禁电池 @ cerberus3: 每 10 min 探 load1, <4.0 才开火;
# 轮转 O2/O3/Ofast x3, 每发前复检门禁, 超时 24h 放弃。
set -u; cd /lsc/zeushome/xy382/project
exec 7>/tmp/xy382_cerb_var_quiet.lock; flock -n 7 || exit 0
L=logs/mhd/perf_baseline; S=/tmp/cerbvarq_$$; mkdir -p $S
exec >> $L/variants_cerberus3_quiet.log 2>&1
echo "== quiet-gate armed $(date -Is) host=$(hostname) gate: load1<4.0, timeout 24h"
DEADLINE=$(( $(date +%s) + 86400 ))
gate() { python3 -c "import sys;sys.exit(0 if float(open('/proc/loadavg').read().split()[0])<4.0 else 1)"; }
until gate; do
  [ $(date +%s) -ge $DEADLINE ] && { echo "== gate timeout $(date -Is), giving up"; rm -rf $S; exit 0; }
  sleep 600
done
echo "== gate open $(date -Is) load1=$(cut -d' ' -f1 /proc/loadavg)"
echo "iso,round,variant,loop_wall_s,e2e_s,cpu_ratio,load1" > $L/variants_cerberus3_quiet.csv
declare -A BIN=( [O2]=bin/mhd2d_fp64_eps1e-4_timed [O3]=bin/mhd2d_fp64_O3_eps1e-4_timed [Ofast]=bin/mhd2d_fp64_Ofast_eps1e-4_timed )
for r in 1 2 3; do
  for cfg in O2 O3 Ofast; do
    until gate; do
      [ $(date +%s) -ge $DEADLINE ] && { echo "== load returned mid-battery, aborting $(date -Is)"; rm -rf $S; exit 0; }
      echo "-- gate closed before r$r $cfg, waiting ($(date -Is) load1=$(cut -d' ' -f1 /proc/loadavg))"; sleep 600
    done
    OUTDIR=$S/run /usr/bin/time -f "%e %U %S" -o $S/t.txt ./${BIN[$cfg]} 512 512 0.5 0.25 muscl hlld 0.18 > $S/o.log 2>&1
    read E U SY < $S/t.txt
    W=$(grep -oP 'wall time = \K[0-9.]+' $S/o.log)
    R=$(python3 -c "print(f'{(${U:-0}+${SY:-0})/max(${E:-1},0.001):.3f}')" 2>/dev/null || echo "")
    echo "$(date -Is),$r,$cfg,${W:-},${E:-},${R:-},$(cut -d' ' -f1 /proc/loadavg)" >> $L/variants_cerberus3_quiet.csv
    rm -rf $S/run
  done
  echo "round $r done $(date -Is)"
done
rm -rf $S; echo "== cerberus3 quiet variants complete $(date -Is)"
