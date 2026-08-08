#!/usr/bin/env bash
# =============================================================================
#  定稿 OT 批量 / Definitive Orszag-Tang battery — final merged HLLD code
#  (统一容差 1e-6 + fallback 计数器版, 2026-07-14)
#
#  跑什么: OT {128,192,256,384}^2 x {fp64,fp32}, CFL=0.25, GLM, MUSCL+HLLD
#  输出:   staging/ot_final_20260714/*.dat  (确认无误后再提升进 results/)
#  日志:   staging/ot_final_20260714/logs/*.log
#
#  用法:   nohup ./run_ot_final_battery.sh > staging/ot_final_20260714/battery.log 2>&1 &
# =============================================================================
set -u
cd "$(dirname "$0")"

STAGE="staging/ot_final_20260714"
mkdir -p "$STAGE/logs"

N_LIST="128 192 256 384"
CFL=0.25
T_END=0.5

# 二进制必须已用最终版代码编译好 (bin/mhd2d_glm_cpu_{fp64,fp32})
for prec in fp64 fp32; do
  [ -x "bin/mhd2d_glm_cpu_$prec" ] || { echo "missing bin/mhd2d_glm_cpu_$prec"; exit 1; }
done

echo "Battery start: $(date)"
pids=""
for prec in fp64 fp32; do
  for N in $N_LIST; do
    log="$STAGE/logs/ot_${prec}_N${N}.log"
    OUTDIR="$STAGE" nice -n 19 "./bin/mhd2d_glm_cpu_$prec" \
      "$N" "$N" "$T_END" "$CFL" muscl hlld 0.18 > "$log" 2>&1 &
    pids="$pids $!"
    echo "launched $prec N=$N (pid $!)"
  done
done

fail=0
for pid in $pids; do
  wait "$pid" || fail=$((fail+1))
done

echo ""
echo "Battery done: $(date)   failures: $fail"
echo "================ summary ================"
for prec in fp64 fp32; do
  for N in $N_LIST; do
    log="$STAGE/logs/ot_${prec}_N${N}.log"
    done_line=$(grep -m1 "^Done" "$log" 2>/dev/null || echo "DID NOT FINISH")
    fb_line=$(grep -m1 "HLLD->HLL fallbacks" "$log" 2>/dev/null || true)
    err_line=$(grep -m1 "^Error" "$log" 2>/dev/null || true)
    echo "[$prec N=$N] $done_line  ${fb_line:-}  ${err_line:-}"
  done
done
