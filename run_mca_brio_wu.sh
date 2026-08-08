#!/usr/bin/env bash
# =============================================================================
#  MHD MCA 战役 (1D Brio-Wu) — 与 Euler branch_sensitivity.sh 同协议
#    Verificarlo MCA: --mode=mca, 虚拟精度 = 原生精度, N_SAMPLES=30
#    配置矩阵: {hlld, hll} x {fp64, fp32}
#    输出: results/mhd/mca/brio_wu/{prec}/{solver}/sample_NN.dat
#    日志: logs/mhd/mca_brio_wu/
#  用法:  ./run_mca_brio_wu.sh          # 编译 + 全部 4 配置 x 30 样本 (~5 分钟)
# =============================================================================
set -u
cd "$(dirname "$0")"

VFC=/lsc/opt/verificarlo-2.4.0/bin/verificarlo-c++
SRC=src/mhd/mhd1d_cpu.cpp
N=800; T_END=0.1; CFL=0.4; ORDER=muscl
N_SAMPLES=30
PAR=16                       # 并行度 / parallel batch size

OUT=results/mhd/mca/brio_wu
LOG=logs/mhd/mca_brio_wu
mkdir -p "$LOG"

# ── 编译 / build (verificarlo 插桩) ─────────────────────────────────────────
echo "== build =="
$VFC -O2 -std=c++14 -I include            "$SRC" -o bin/mhd1d_vfc      || exit 1
$VFC -O2 -std=c++14 -I include -DUSE_FLOAT "$SRC" -o bin/mhd1d_vfc_fp32 || exit 1
echo "OK"

# ── 采样 / sampling ──────────────────────────────────────────────────────────
TOTAL_FAIL=0
for prec in fp64 fp32; do
  if [ "$prec" = "fp32" ]; then
    bin=bin/mhd1d_vfc_fp32; bits="--precision-binary32=24"
  else
    bin=bin/mhd1d_vfc;      bits="--precision-binary64=53"
  fi
  for solver in hlld hll; do
    outdir="$OUT/$prec/$solver"
    mkdir -p "$outdir"
    echo "== $prec / $solver : $N_SAMPLES samples =="
    t0=$(date +%s)
    for ((s=1; s<=N_SAMPLES; s++)); do
      (
        tmp=$(mktemp -d "$outdir/tmp_XXXX")
        if VFC_BACKENDS="libinterflop_mca.so --mode=mca $bits" \
           OUTDIR="$tmp" nice -n 19 "./$bin" $N $T_END $CFL $ORDER $solver \
           > "$LOG/${prec}_${solver}_s${s}.log" 2>&1; then
          mv "$tmp"/brio_wu_*.dat "$outdir/$(printf 'sample_%02d.dat' $s)"
        fi
        rmdir "$tmp" 2>/dev/null
      ) &
      [ $((s % PAR)) -eq 0 ] && wait
    done
    wait
    t1=$(date +%s)
    n_ok=$(ls "$outdir"/sample_*.dat 2>/dev/null | wc -l)
    echo "OK ($((t1-t0))s, $n_ok/$N_SAMPLES samples)"
    [ "$n_ok" -lt "$N_SAMPLES" ] && TOTAL_FAIL=$((TOTAL_FAIL + N_SAMPLES - n_ok))
  done
done

echo ""
echo "== summary =="
for prec in fp64 fp32; do for solver in hlld hll; do
  n=$(ls "$OUT/$prec/$solver"/sample_*.dat 2>/dev/null | wc -l)
  printf "  %s/%s: %d samples\n" "$prec" "$solver" "$n"
done; done
[ "$TOTAL_FAIL" -gt 0 ] && echo "WARN: $TOTAL_FAIL sample run(s) failed (see $LOG/)"
exit 0
