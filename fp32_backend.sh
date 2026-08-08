#!/usr/bin/env bash
# =============================================================================
#  fp32_backend.sh — Step 5: FP32 backend experiments
#
#  Closes the 2x2 CPU/GPU x FP64/FP32 design by mirroring the FP64
#  Section 5.3.3 case list at single precision:
#
#    1D:  Toro tests 1-5 (HLLC, N=200)
#    2D:  shock-bubble (500x197), Riemann cfg6 and cfg12 (400x400)
#
#  Output (re-uses existing fp32 directory layout — no new top-level dirs):
#    results/euler/ieee/fp32/      — CPU FP32  1D
#    results/euler/gpu/fp32/       — GPU FP32  1D
#    results/euler/2d/fp32/        — CPU FP32  2D shock-bubble + Riemann
#    results/euler/2d/fp32/gpu/    — GPU FP32  2D Riemann (CPU writes alongside)
#
#  IMPORTANT: This script ALWAYS recompiles binaries before running. The OUTDIR
#  env-var mechanism only exists in the post-Step-4 source code, so the older
#  binaries on disk (Apr-May 4) won't respect it. --run-only assumes you ran
#  --build-only first in this session.
#
#  Usage:
#    bash fp32_backend.sh                # full: compile + run all 16 cases
#    bash fp32_backend.sh --build-only   # compile only
#    bash fp32_backend.sh --run-only     # assumes binaries already fresh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

NVCC=${NVCC:-nvcc}
NVCC_ARCH="--generate-code arch=compute_80,code=sm_80"
COMMON="-std=c++14 -I include -DUSE_FLOAT"

DO_BUILD=true
DO_RUN=true
for arg in "$@"; do
  case "$arg" in
    --build-only) DO_RUN=false ;;
    --run-only)   DO_BUILD=false ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

mkdir -p bin logs/euler/fp32 \
         results/euler/ieee/fp32 results/euler/gpu/fp32 results/euler/2d/fp32/gpu

# ── Compile ──────────────────────────────────────────────────────────────────
if $DO_BUILD; then
  echo "=========================================="
  echo " COMPILE (FP32 = -DUSE_FLOAT)"
  echo "=========================================="
  log="logs/euler/fp32/compile.log"
  : > "$log"

  # CPU
  for pair in "src/euler/riemann1d_cpu.cpp:riemann1d_fp32" \
              "src/euler/euler2d_cpu.cpp:euler2d_cpu_fp32" \
              "src/euler/riemann2d_cpu.cpp:riemann2d_fp32"; do
    IFS=':' read -r src out <<< "$pair"
    printf "  g++  %-30s -> bin/%s ... " "$src" "$out"
    g++ -O2 $COMMON "$src" -o "bin/$out" >> "$log" 2>&1
    echo "OK"
  done

  # GPU
  for pair in "src/euler/riemann1d_gpu.cu:riemann1d_gpu_fp32" \
              "src/euler/euler2d_cuda.cu:euler2d_gpu_fp32" \
              "src/euler/riemann2d_gpu.cu:riemann2d_gpu_fp32"; do
    IFS=':' read -r src out <<< "$pair"
    printf "  nvcc %-30s -> bin/%s ... " "$src" "$out"
    $NVCC -O2 $COMMON $NVCC_ARCH "$src" -o "bin/$out" >> "$log" 2>&1
    echo "OK"
  done
fi

# ── Run ──────────────────────────────────────────────────────────────────────
if $DO_RUN; then
  echo ""
  echo "=========================================="
  echo " RUN"
  echo "=========================================="
  log="logs/euler/fp32/run.log"
  : > "$log"

  TESTS_1D="1 2 3 4 5"
  CFGS_2D="6 12"

  echo ""
  echo "--- CPU FP32 1D ---"
  for t in $TESTS_1D; do
    printf "  Toro %d HLLC ... " "$t"
    OUTDIR=results/euler/ieee/fp32 ./bin/riemann1d_fp32 200 "$t" hllc >> "$log" 2>&1
    echo "OK"
  done

  echo ""
  echo "--- GPU FP32 1D ---"
  for t in $TESTS_1D; do
    printf "  Toro %d HLLC ... " "$t"
    OUTDIR=results/euler/gpu/fp32 ./bin/riemann1d_gpu_fp32 200 "$t" hllc >> "$log" 2>&1
    echo "OK"
  done

  echo ""
  echo "--- CPU FP32 2D ---"
  printf "  shock-bubble    ... "
  OUTDIR=results/euler/2d/fp32 ./bin/euler2d_cpu_fp32 >> "$log" 2>&1
  echo "OK"
  for c in $CFGS_2D; do
    printf "  Riemann cfg%-2d   ... " "$c"
    OUTDIR=results/euler/2d/fp32 ./bin/riemann2d_fp32 400 "$c" >> "$log" 2>&1
    echo "OK"
  done

  echo ""
  echo "--- GPU FP32 2D ---"
  printf "  shock-bubble    ... "
  OUTDIR=results/euler/2d/fp32 ./bin/euler2d_gpu_fp32 >> "$log" 2>&1
  echo "OK"
  for c in $CFGS_2D; do
    printf "  Riemann cfg%-2d   ... " "$c"
    OUTDIR=results/euler/2d/fp32/gpu ./bin/riemann2d_gpu_fp32 400 "$c" >> "$log" 2>&1
    echo "OK"
  done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " SUMMARY"
echo "=========================================="
echo "  CPU FP32 1D  (results/euler/ieee/fp32/):     $(ls results/euler/ieee/fp32/*_hllc_fp32_*.dat 2>/dev/null | wc -l) files"
echo "  GPU FP32 1D  (results/euler/gpu/fp32/):      $(ls results/euler/gpu/fp32/*_hllc_fp32_*.dat 2>/dev/null | wc -l) files"
echo "  CPU FP32 2D  (results/euler/2d/fp32/):       $(ls results/euler/2d/fp32/{cpu_shock_bubble,riemann2d}*.dat 2>/dev/null | wc -l) files"
echo "  GPU FP32 2D  (results/euler/2d/fp32/, gpu/): $(ls results/euler/2d/fp32/gpu_shock_bubble.dat results/euler/2d/fp32/gpu/*.dat 2>/dev/null | wc -l) files"
echo ""
echo "Next: see end of Step 5 instructions in chat for compare commands."
