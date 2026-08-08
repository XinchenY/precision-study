#!/usr/bin/env bash
# =============================================================================
#  Ch10 Step 2a-2c: VPREC 位宽扫描 (1D Brio-Wu N=800, ε=1e-4 协议)
#    Verificarlo VPREC backend, --mode=ob (确定性最近舍入)
#    t ∈ {16,20,23,28,32,40,52}; 基线 = 同 clang18 同选项原生 FP64
#    输出: results/mhd/vprec/scan1d/{native_fp64,tNN}/
#    日志: logs/mhd/vprec_scan1d/ (含 fallback 行与 exit code)
#  用法: ./run_vprec_scan1d.sh
# =============================================================================
set -u
cd "$(dirname "$0")"
VFC=/lsc/opt/verificarlo-2.4.0/bin/verificarlo-c++
CXX=/usr/bin/clang++
EPS=-DMHD_HLLD_TOLERANCE_VALUE=1e-4
OUT=results/mhd/vprec/scan1d
LOG=logs/mhd/vprec_scan1d
mkdir -p "$OUT" "$LOG"

echo "== build (eps=1e-4) =="
$CXX -O2 -std=c++14 $EPS -I include src/mhd/mhd1d_cpu.cpp -o bin/vprec_step1/mhd1d_clang18_fp64_eps1e-4 || exit 1
$VFC  -O2 -std=c++14 $EPS -I include src/mhd/mhd1d_cpu.cpp -o bin/vprec_step1/mhd1d_vfc_fp64_eps1e-4  || exit 1
echo "OK"

echo "== baseline (native fp64, eps=1e-4) =="
OUTDIR=$OUT/native_fp64 ./bin/vprec_step1/mhd1d_clang18_fp64_eps1e-4 800 0.1 0.4 muscl hlld \
  > $LOG/native_fp64.log 2>&1 || { echo "baseline failed"; exit 1; }
BASE=$OUT/native_fp64/brio_wu_muscl_hlld_fp64_N800.dat

for t in 16 20 23 28 32 40 52; do
  echo "== t=$t =="
  VFC_BACKENDS="libinterflop_vprec.so --precision-binary64=$t --mode=ob" \
  OUTDIR=$OUT/t$t ./bin/vprec_step1/mhd1d_vfc_fp64_eps1e-4 800 0.1 0.4 muscl hlld \
    > $LOG/t$t.log 2>&1
  rc=$?
  echo "exit=$rc" >> $LOG/t$t.log
  if [ $rc -eq 0 ]; then
    python3 analysis/mhd/compare_precision.py $BASE $OUT/t$t/brio_wu_muscl_hlld_fp64_N800.dat > $LOG/cmp_t$t.txt
  else
    echo "RUN FAILED rc=$rc" > $LOG/cmp_t$t.txt
  fi
  grep -E "Done|fallbacks =|exit=" $LOG/t$t.log | tail -3
done
echo "== scan complete =="