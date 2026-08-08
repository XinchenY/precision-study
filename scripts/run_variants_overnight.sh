#!/usr/bin/env bash
# CPU 变体通宵连跑: 不设门禁, 连续轮转 O2/O3/Ofast 直到 deadline(5h)或 6 轮;
# 每发记 load, 早晨按预登记判据挑轮 (O2 控制发 <= 1811 s, 即冻结基线 +3%)。
set -u; cd /lsc/zeushome/xy382/project
H=$(hostname -s)
exec 7>/tmp/xy382_var_overnight_$H.lock; flock -n 7 || exit 0
L=logs/mhd/perf_baseline; S=/tmp/varov_$$; mkdir -p $S
exec >> $L/variants_overnight_$H.log 2>&1
echo "== overnight variants start $(date -Is) host=$H nproc=$(nproc)"
DEADLINE=$(( $(date +%s) + 18000 ))
CSV=$L/variants_overnight_$H.csv
[ -f $CSV ] || echo "iso,round,variant,loop_wall_s,e2e_s,cpu_ratio,load1" > $CSV
declare -A BIN=( [O2]=bin/mhd2d_fp64_eps1e-4_timed [O3]=bin/mhd2d_fp64_O3_eps1e-4_timed [Ofast]=bin/mhd2d_fp64_Ofast_eps1e-4_timed )
for r in 1 2 3 4 5 6; do
  [ $(date +%s) -ge $DEADLINE ] && break
  for cfg in O2 O3 Ofast; do
    OUTDIR=$S/run /usr/bin/time -f "%e %U %S" -o $S/t.txt ./${BIN[$cfg]} 512 512 0.5 0.25 muscl hlld 0.18 > $S/o.log 2>&1
    read E U SY < $S/t.txt
    W=$(grep -oP 'wall time = \K[0-9.]+' $S/o.log)
    R=$(python3 -c "print(f'{(${U:-0}+${SY:-0})/max(${E:-1},0.001):.3f}')" 2>/dev/null || echo "")
    echo "$(date -Is),$r,$cfg,${W:-},${E:-},${R:-},$(cut -d' ' -f1 /proc/loadavg)" >> $CSV
    rm -rf $S/run
  done
  echo "round $r done $(date -Is) load1=$(cut -d' ' -f1 /proc/loadavg)"
done
rm -rf $S; echo "== overnight variants complete $(date -Is)"
