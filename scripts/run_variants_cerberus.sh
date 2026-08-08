#!/usr/bin/env bash
# CPU 变体战役 @ cerberus3 (同款 Xeon 4314, 空闲): 轮转 O2/O3/Ofast x3
set -u; cd /lsc/zeushome/xy382/project
exec 7>/tmp/xy382_cerb_var.lock; flock -n 7 || exit 0
L=logs/mhd/perf_baseline; S=/tmp/cerbvar_$$; mkdir -p $S
exec >> $L/variants_cerberus3.log 2>&1
echo "== cerberus3 variants start $(date -Is) host=$(hostname) nproc=$(nproc)"
lscpu | grep "Model name"
echo "iso,round,variant,loop_wall_s,e2e_s,cpu_ratio,load1" > $L/variants_cerberus3.csv
declare -A BIN=( [O2]=bin/mhd2d_fp64_eps1e-4_timed [O3]=bin/mhd2d_fp64_O3_eps1e-4_timed [Ofast]=bin/mhd2d_fp64_Ofast_eps1e-4_timed )
for r in 1 2 3; do
  for cfg in O2 O3 Ofast; do
    OUTDIR=$S/run /usr/bin/time -f "%e %U %S" -o $S/t.txt ./${BIN[$cfg]} 512 512 0.5 0.25 muscl hlld 0.18 > $S/o.log 2>&1
    read E U SY < $S/t.txt
    W=$(grep -oP 'wall time = \K[0-9.]+' $S/o.log)
    R=$(python3 -c "print(f'{(${U:-0}+${SY:-0})/max(${E:-1},0.001):.3f}')" 2>/dev/null || echo "")
    echo "$(date -Is),$r,$cfg,${W:-},${E:-},${R:-},$(cut -d' ' -f1 /proc/loadavg)" >> $L/variants_cerberus3.csv
    rm -rf $S/run
  done
  echo "round $r done $(date -Is)"
done
rm -rf $S; echo "== cerberus3 variants complete $(date -Is)"
