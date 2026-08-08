#!/usr/bin/env bash
# =============================================================================
#  sanity_check.sh  —  verificarlo 工具链完整性验证
#
#  用法:
#    bash sanity_check.sh           # 默认 fp64
#    bash sanity_check.sh --prec fp32
#    bash sanity_check.sh --prec fp64
#
#  验证内容:
#    1. 依赖检查   (g++, verificarlo-c++, python3)
#    2. 编译检查
#    3. 运行检查   (g++ 版本能正常产出结果)
#    4. IEEE 验证  (verificarlo+IEEE 与 g++ 结果的 L∞ 在合理范围内)
#
#  输出:
#    终端显示 PASS/FAIL
#    logs/euler/{PREC}/sanity_check.log
# =============================================================================

set -euo pipefail

# ── 参数解析 ─────────────────────────────────────────────────
PREC="fp64"
while [[ $# -gt 0 ]]; do
  case $1 in
    --prec) PREC="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "$PREC" == "fp64" ]]; then
  FLOAT_FLAG=""
  BIN="bin/riemann1d"
  VFC_BIN_OUT="bin/riemann1d_vfc"
  DAT_SUFFIX="fp64_200"
  THRESHOLD="1e-12"    # fp64: 1000 × machine_epsilon(2.22e-16) ≈ 2.22e-13，取整到 1e-12
elif [[ "$PREC" == "fp32" ]]; then
  FLOAT_FLAG="-DUSE_FLOAT"
  BIN="bin/riemann1d_fp32"
  VFC_BIN_OUT="bin/riemann1d_vfc_fp32"
  DAT_SUFFIX="fp32_200"
  THRESHOLD="1e-4"     # fp32: 1000 × machine_epsilon(1.19e-07) ≈ 1.19e-4，取整到 1e-4
else
  echo "Unknown precision: $PREC. Use fp64 or fp32."; exit 1
fi

LOG="logs/euler/${PREC}/sanity_check.log"
mkdir -p "logs/euler/${PREC}" bin results

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " Sanity Check [$PREC] — $(date)"
echo " Host: $(hostname)"
echo " CPU:  $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "============================================================"

PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

# ── Step 1: 依赖检查 ─────────────────────────────────────────
echo ""
echo "--- Step 1: Dependencies ---"

for cmd in g++ python3; do
  command -v "$cmd" &>/dev/null && ok "$cmd: $(command -v $cmd)" || fail "$cmd not found"
done

VFC_BIN=""
for candidate in verificarlo-c++ /lsc/opt/verificarlo-2.4.0/bin/verificarlo-c++; do
  if command -v "$candidate" &>/dev/null 2>&1; then
    VFC_BIN="$candidate"; break
  fi
done
[ -n "$VFC_BIN" ] && ok "verificarlo-c++: $VFC_BIN" || { fail "verificarlo-c++ not found"; exit 1; }

# ── Step 2: 编译 ─────────────────────────────────────────────
echo ""
echo "--- Step 2: Compilation [$PREC] ---"

g++ -O2 -std=c++14 $FLOAT_FLAG -I include src/euler/riemann1d_cpu.cpp -o "$BIN" 2>&1 \
  && ok "g++ → $BIN" || { fail "g++ compilation failed"; exit 1; }

"$VFC_BIN" -O2 -std=c++14 $FLOAT_FLAG -I include src/euler/riemann1d_cpu.cpp -o "$VFC_BIN_OUT" 2>&1 \
  && ok "verificarlo → $VFC_BIN_OUT" || { fail "verificarlo compilation failed"; exit 1; }

# ── Step 3: 运行检查 ─────────────────────────────────────────
echo ""
echo "--- Step 3: Run check (g++, Test 1 HLLC) ---"

"./$BIN" 200 1 hllc \
  && ok "g++ binary ran successfully" || { fail "g++ binary failed"; exit 1; }
cp "results/euler/toro1_hllc_${DAT_SUFFIX}.dat" /tmp/_sc_ref.dat

# ── Step 4: IEEE 验证 ─────────────────────────────────────────
echo ""
echo "--- Step 4: IEEE backend verification ---"

VFC_BACKENDS="libinterflop_ieee.so" "./$VFC_BIN_OUT" 200 1 hllc
cp "results/euler/toro1_hllc_${DAT_SUFFIX}.dat" /tmp/_sc_vfc_ieee.dat

RESULT=$(python3 analysis/euler/compare_norms.py /tmp/_sc_ref.dat /tmp/_sc_vfc_ieee.dat)
echo "$RESULT"

LINF=$(echo "$RESULT" | awk '/rho/ {print $NF}')

python3 - <<EOF
linf = float("$LINF")
thr  = $THRESHOLD
if linf <= thr:
    print(f"  L∞(rho) = {linf:.3e}  ≤  {thr:.0e}  →  PASS")
    exit(0)
else:
    print(f"  L∞(rho) = {linf:.3e}  >  {thr:.0e}  →  FAIL")
    exit(1)
EOF

[ $? -eq 0 ] \
  && ok "IEEE backend L∞ within threshold ($THRESHOLD)" \
  || fail "IEEE backend L∞ exceeds threshold"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " [$PREC]  $PASS PASSED, $FAIL FAILED"
[ "$FAIL" -eq 0 ] \
  && echo " Status: ALL CHECKS PASSED — safe to proceed with MCA" \
  || echo " Status: SOME CHECKS FAILED — do not proceed with MCA"
echo " Log: $LOG"
echo "============================================================"

[ "$FAIL" -eq 0 ]
