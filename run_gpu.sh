#!/usr/bin/env bash
# =============================================================================
#  run_gpu.sh  —  GPU 1D Riemann solver: 编译 + 运行全部测试
#
#  运行后产生:
#    results/euler/gpu/fp64/toro{1-5}_{hllc|hll}_fp64_200.dat   (10 个文件)
#    results/euler/gpu/fp32/toro{1-5}_{hllc|hll}_fp32_200.dat   (10 个文件)
#
#  用法:
#    bash run_gpu.sh           # 编译 + 运行 fp64 和 fp32
#    bash run_gpu.sh --fp64    # 只运行 fp64
#    bash run_gpu.sh --fp32    # 只运行 fp32
#    bash run_gpu.sh --norun   # 只编译, 不运行
# =============================================================================

set -euo pipefail

DO_FP64=true
DO_FP32=true
DO_RUN=true

for arg in "$@"; do
  case "$arg" in
    --fp64)  DO_FP32=false ;;
    --fp32)  DO_FP64=false ;;
    --norun) DO_RUN=false  ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

# ── NVCC 路径 ─────────────────────────────────────────────────────────────────
NVCC=${NVCC:-nvcc}
if ! command -v "$NVCC" &>/dev/null; then
  for candidate in /usr/local/cuda/bin/nvcc /usr/bin/nvcc; do
    [ -x "$candidate" ] && { NVCC="$candidate"; break; }
  done
fi
echo "NVCC: $NVCC"
"$NVCC" --version | head -1

NVCC_FLAGS="-O2 -std=c++14 -I include --generate-code arch=compute_80,code=sm_80"
# A30 = sm_86; 兼容 sm_80 也可运行。如需精确匹配 A30 改为 arch=compute_86,code=sm_86

mkdir -p bin results/euler/gpu/fp64 results/euler/gpu/fp32

# ── 编译 ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Compilation ==="

if $DO_FP64; then
  echo "  Compiling fp64 ..."
  "$NVCC" $NVCC_FLAGS src/euler/riemann1d_gpu.cu -o bin/riemann1d_gpu
  echo "  [OK] bin/riemann1d_gpu"
fi

if $DO_FP32; then
  echo "  Compiling fp32 ..."
  "$NVCC" $NVCC_FLAGS -DUSE_FLOAT src/euler/riemann1d_gpu.cu -o bin/riemann1d_gpu_fp32
  echo "  [OK] bin/riemann1d_gpu_fp32"
fi

$DO_RUN || { echo "  --norun: skipping execution"; exit 0; }

# ── 运行 ──────────────────────────────────────────────────────────────────────
TESTS="1 2 3 4 5"
SOLVERS="hllc hll"

echo ""
echo "=== Running Tests ==="

if $DO_FP64; then
  echo "--- fp64 ---"
  for tid in $TESTS; do
    for slv in $SOLVERS; do
      printf "  Test %d  %-4s  fp64 ... " "$tid" "$slv"
      ./bin/riemann1d_gpu 200 "$tid" "$slv" 2>&1 | tail -1
    done
  done
fi

if $DO_FP32; then
  echo "--- fp32 ---"
  for tid in $TESTS; do
    for slv in $SOLVERS; do
      printf "  Test %d  %-4s  fp32 ... " "$tid" "$slv"
      ./bin/riemann1d_gpu_fp32 200 "$tid" "$slv" 2>&1 | tail -1
    done
  done
fi

echo ""
echo "=== Done ==="
echo "Results in:"
echo "  results/euler/gpu/fp64/  ($(ls results/euler/gpu/fp64/*.dat 2>/dev/null | wc -l) files)"
echo "  results/euler/gpu/fp32/  ($(ls results/euler/gpu/fp32/*.dat 2>/dev/null | wc -l) files)"
