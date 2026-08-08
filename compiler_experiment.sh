#!/usr/bin/env bash
# =============================================================================
#  compiler_experiment.sh — Build + run all (build × case) combinations for
#                           compiler-flag sensitivity analysis.
#
#  Builds:
#    cpu_O2       = g++ -O2                    (baseline, IEEE 754 strict)
#    cpu_O3       = g++ -O3                    (still IEEE 754 strict)
#    cpu_Ofast    = g++ -Ofast                 (breaks IEEE: -ffast-math, ...)
#    gpu_default  = nvcc -O2                   (default device math)
#    gpu_fastmath = nvcc -O2 --use_fast_math   (fast device intrinsics)
#
#  Cases:
#    toro5            = 1D Toro Test 5 (two strong shocks, N=200)        [~ 1 s]
#    shock_bubble     = 2D Haas–Sturtevant shock-bubble                  [~15 min CPU]
#    riemann2d_cfg12  = 2D Liska-Wendroff cfg12 (2 shocks + 2 contacts)  [~10 min CPU]
#
#  Output layout (canonicalised filenames, same across all builds):
#    bin/<case>_<build>                                       — 15 binaries
#    results/euler/compiler/fp64/<build>/<canonical_file>           — 3 files per build
#    logs/euler/compiler/<build>_<phase>.log                        — compile/run logs
#
#  Source code uses the OUTDIR env var (default behaviour preserved when unset),
#  so this script does not modify any .cpp / .cu file at runtime.
#
#  Usage:
#    bash compiler_experiment.sh                # full: compile + run all 15
#    bash compiler_experiment.sh --build-only   # compile only
#    bash compiler_experiment.sh --run-only     # run only (binaries must exist)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── Configuration ────────────────────────────────────────────────────────────
NVCC=${NVCC:-nvcc}
NVCC_ARCH="--generate-code arch=compute_80,code=sm_80"
COMMON="-std=c++14 -I include"

# Build name -> "compiler|flags"
declare -A BUILD_FLAGS=(
  [cpu_O2]="g++|-O2"
  [cpu_O3]="g++|-O3"
  [cpu_Ofast]="g++|-Ofast"
  [gpu_default]="nvcc|-O2"
  [gpu_fastmath]="nvcc|-O2 --use_fast_math"
)
BUILD_ORDER=(cpu_O2 cpu_O3 cpu_Ofast gpu_default gpu_fastmath)

# Case -> source file (selected based on whether build is cpu_* or gpu_*)
declare -A SRC_CPU=(
  [toro5]="src/euler/riemann1d_cpu.cpp"
  [shock_bubble]="src/euler/euler2d_cpu.cpp"
  [riemann2d_cfg12]="src/euler/riemann2d_cpu.cpp"
)
declare -A SRC_GPU=(
  [toro5]="src/euler/riemann1d_gpu.cu"
  [shock_bubble]="src/euler/euler2d_cuda.cu"
  [riemann2d_cfg12]="src/euler/riemann2d_gpu.cu"
)

# Case -> binary args
declare -A CASE_ARGS=(
  [toro5]="200 5 hllc"
  [shock_bubble]=""
  [riemann2d_cfg12]="400 12"
)

# Case -> raw output filename produced by binary (CPU vs GPU differ for shock_bubble)
declare -A RAW_FILE_CPU=(
  [toro5]="toro5_hllc_fp64_200.dat"
  [shock_bubble]="cpu_shock_bubble.dat"
  [riemann2d_cfg12]="riemann2d_cfg12_2S2C_hllc_400.dat"
)
declare -A RAW_FILE_GPU=(
  [toro5]="toro5_hllc_fp64_200.dat"
  [shock_bubble]="gpu_shock_bubble.dat"
  [riemann2d_cfg12]="riemann2d_cfg12_2S2C_hllc_400.dat"
)

# Case -> canonical filename (same across all builds, for easy cross-build diff)
declare -A CANON=(
  [toro5]="toro5_hllc_fp64_200.dat"
  [shock_bubble]="shock_bubble.dat"
  [riemann2d_cfg12]="riemann2d_cfg12_2S2C_hllc_400.dat"
)

CASES=(toro5 shock_bubble riemann2d_cfg12)

# ── CLI ──────────────────────────────────────────────────────────────────────
DO_BUILD=true
DO_RUN=true
for arg in "$@"; do
  case "$arg" in
    --build-only) DO_RUN=false ;;
    --run-only)   DO_BUILD=false ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

mkdir -p bin logs/euler/compiler

# ── Compile ──────────────────────────────────────────────────────────────────
if $DO_BUILD; then
  echo "=========================================="
  echo " COMPILE phase"
  echo "=========================================="
  for build in "${BUILD_ORDER[@]}"; do
    IFS='|' read -r compiler flags <<< "${BUILD_FLAGS[$build]}"
    log="logs/euler/compiler/${build}_compile.log"
    : > "$log"
    echo ""
    echo "[$build]  compiler=$compiler  flags='$flags'"

    for c in "${CASES[@]}"; do
      if [[ $build == cpu_* ]]; then
        src="${SRC_CPU[$c]}"
      else
        src="${SRC_GPU[$c]}"
      fi
      out="bin/${c}_${build}"
      printf "  %-22s -> %-40s ... " "$c" "$out"
      if [[ $compiler == g++ ]]; then
        g++ $flags $COMMON "$src" -o "$out" >> "$log" 2>&1
      else
        $NVCC $flags $COMMON $NVCC_ARCH "$src" -o "$out" >> "$log" 2>&1
      fi
      echo "OK"
    done
  done
fi

# ── Run ──────────────────────────────────────────────────────────────────────
if $DO_RUN; then
  echo ""
  echo "=========================================="
  echo " RUN phase"
  echo "=========================================="
  for build in "${BUILD_ORDER[@]}"; do
    outdir="results/euler/compiler/fp64/${build}"
    mkdir -p "$outdir"
    log="logs/euler/compiler/${build}_run.log"
    : > "$log"
    echo ""
    echo "[$build]  outdir=$outdir"

    for c in "${CASES[@]}"; do
      bin="bin/${c}_${build}"
      printf "  %-22s ... " "$c"
      t0=$(date +%s)
      # Don't let one failure abort the experiment (Ofast may NaN on pathological cases)
      if OUTDIR="$outdir" "./$bin" ${CASE_ARGS[$c]} >> "$log" 2>&1; then
        t1=$(date +%s)
        if [[ $build == cpu_* ]]; then
          raw="${RAW_FILE_CPU[$c]}"
        else
          raw="${RAW_FILE_GPU[$c]}"
        fi
        canon="${CANON[$c]}"
        if [[ "$raw" != "$canon" && -f "$outdir/$raw" ]]; then
          mv "$outdir/$raw" "$outdir/$canon"
        fi
        if [[ -f "$outdir/$canon" ]]; then
          echo "OK ($((t1-t0))s) -> $canon"
        else
          echo "FAIL (binary returned 0 but no output file; see $log)"
        fi
      else
        t1=$(date +%s)
        echo "FAIL ($((t1-t0))s) (see $log)"
      fi
    done
  done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " SUMMARY"
echo "=========================================="
for build in "${BUILD_ORDER[@]}"; do
  d="results/euler/compiler/fp64/${build}"
  if [[ -d "$d" ]]; then
    n=$(ls "$d" 2>/dev/null | wc -l)
    echo "  $build: $n files in $d"
  fi
done
echo ""
echo "Next step:"
echo "  python3 analysis/euler/compare_compilers.py"
