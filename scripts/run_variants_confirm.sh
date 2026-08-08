#!/usr/bin/env bash
# 真空定音鼓 @ cerberus3 (发射时 load 0.05): O3 x3 + O2/Ofast 补第三发, 交错排列
set -u; cd /lsc/zeushome/xy382/project
exec 7>/tmp/xy382_var_confirm.lock; flock -n 7 || exit 0
L=logs/mhd/perf_baseline; S=/tmp/varcf_$$; mkdir -p $S
exec >> $L/variants_confirm_cerberus3.log 2>&1
echo "== confirm battery start $(date -Is) host=$(hostname) load1=$(cut -d' ' -f1 /proc/loadavg)"
CSV=$L/variants_confirm_cerberus3.csv
[ -f $CSV ] || echo "iso,seq,variant,loop_wall_s,e2e_s,cpu_ratio,load1" > $CSV
declare -A BIN=( [O2]=bin/mhd2d_fp64_eps1e-4_timed [O3]=bin/mhd2d_fp64_O3_eps1e-4_timed [Ofast]=bin/mhd2d_fp64_Ofast_eps1e-4_timed )
i=0
for cfg in O3 O2 O3 Ofast O3; do
  i=$((i+1))
  OUTDIR=$S/run /usr/bin/time -f "%e %U %S" -o $S/t.txt ./${BIN[$cfg]} 512 512 0.5 0.25 muscl hlld 0.18 > $S/o.log 2>&1
  read E U SY < $S/t.txt
  W=$(grep -oP 'wall time = \K[0-9.]+' $S/o.log)
  R=$(python3 -c "print(f'{(${U:-0}+${SY:-0})/max(${E:-1},0.001):.3f}')" 2>/dev/null || echo "")
  echo "$(date -Is),$i,$cfg,${W:-},${E:-},${R:-},$(cut -d' ' -f1 /proc/loadavg)" >> $CSV
  rm -rf $S/run
done
rm -rf $S; echo "== confirm battery complete $(date -Is)"
