#!/usr/bin/env bash
# =============================================================================
#  branch_sensitivity.sh — MCA with floating-point comparison instrumentation
#                          (--inst-fcmp) for Section 5.4 Branch Sensitivity.
#
#  Compares against the existing MCA baseline at results/euler/mca/{prec}/...
#  by re-running the same Toro 1-5 HLLC configurations under MCA, but with
#  Verificarlo's --inst-fcmp flag enabled so that comparison operations are
#  also subject to MCA perturbation.
#
#  Output: results/euler/mca_fcmp/{fp32,fp64}/test{N}_{name}/hllc/sample_{NN}.dat
#  (Same structure as existing results/euler/mca/, so analysis/euler/mca_analysis.py
#   works after a one-line MCA_DIR override.)
#
#  Usage:
#    bash branch_sensitivity.sh                # compile + run all (FP32+FP64)
#    bash branch_sensitivity.sh --build-only   # compile only
#    bash branch_sensitivity.sh --run-only     # run only (binaries must exist)
#    bash branch_sensitivity.sh --prec fp32    # only FP32
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── Locate Verificarlo ───────────────────────────────────────────────────────
VFC=${VFC_BIN:-verificarlo-c++}
if ! command -v "$VFC" &>/dev/null; then
  for cand in /lsc/opt/verificarlo-2.4.0/bin/verificarlo-c++; do
    [ -x "$cand" ] && { VFC="$cand"; break; }
  done
fi
echo "Using Verificarlo: $VFC"

# ── CLI ──────────────────────────────────────────────────────────────────────
DO_BUILD=true
DO_RUN=true
PRECS="fp32 fp64"
N_SAMPLES=30
NX=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only) DO_RUN=false; shift ;;
    --run-only)   DO_BUILD=false; shift ;;
    --prec)       PRECS="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

mkdir -p bin logs/euler/branch_sensitivity

declare -A TESTNAME=(
  [1]="test1_sod"  [2]="test2_123"  [3]="test3_left_blast"
  [4]="test4_right_blast"  [5]="test5_two_shocks"
)

# ── Compile ──────────────────────────────────────────────────────────────────
if $DO_BUILD; then
  echo ""
  echo "=========================================="
  echo " COMPILE Verificarlo binaries with --inst-fcmp"
  echo "=========================================="
  log="logs/euler/branch_sensitivity/compile.log"
  : > "$log"

  for prec in $PRECS; do
    flag=""; suffix=""
    [[ $prec == fp32 ]] && { flag="-DUSE_FLOAT"; suffix="_fp32"; }
    out="bin/riemann1d_vfc_fcmp${suffix}"
    printf "  %-8s -> %-40s ... " "$prec" "$out"
    "$VFC" --inst-fcmp -O2 -std=c++14 $flag -I include \
      src/euler/riemann1d_cpu.cpp -o "$out" >> "$log" 2>&1
    echo "OK"
  done
fi

# ── Run ──────────────────────────────────────────────────────────────────────
if $DO_RUN; then
  echo ""
  echo "=========================================="
  echo " RUN MCA with branch instrumentation"
  echo "=========================================="
  log="logs/euler/branch_sensitivity/run.log"
  : > "$log"

  TOTAL_FAIL=0

  for prec in $PRECS; do
    suffix=""; bits_arg=""
    if [[ $prec == fp32 ]]; then
      suffix="_fp32"
      bits_arg="--precision-binary32=24"
    else
      bits_arg="--precision-binary64=53"
    fi
    bin="bin/riemann1d_vfc_fcmp${suffix}"

    if [[ ! -x "$bin" ]]; then
      echo "  ERROR: $bin not found. Run --build-only first or omit --run-only."
      exit 1
    fi

    for tid in 1 2 3 4 5; do
      tname="${TESTNAME[$tid]}"
      outdir="results/euler/mca_fcmp/${prec}/${tname}/hllc"
      mkdir -p "$outdir"
      raw="$outdir/toro${tid}_hllc_${prec}_${NX}.dat"

      printf "  %-5s %-22s ... " "$prec" "$tname"
      t0=$(date +%s)
      n_ok=0; n_fail=0

      for ((s=1; s<=N_SAMPLES; s++)); do
        sample=$(printf "sample_%02d.dat" "$s")
        if OUTDIR="$outdir" \
           VFC_BACKENDS="libinterflop_mca.so --mode=mca $bits_arg" \
           "./$bin" "$NX" "$tid" hllc >> "$log" 2>&1 \
           && [[ -f "$raw" ]]; then
          mv "$raw" "$outdir/$sample"
          n_ok=$((n_ok+1))
        else
          n_fail=$((n_fail+1))
          [[ -f "$raw" ]] && rm -f "$raw"
        fi
      done

      t1=$(date +%s)
      if [[ $n_fail -eq 0 ]]; then
        echo "OK ($((t1-t0))s, $n_ok/$N_SAMPLES samples)"
      else
        echo "PARTIAL ($((t1-t0))s, $n_ok ok, $n_fail FAIL)"
        TOTAL_FAIL=$((TOTAL_FAIL+n_fail))
      fi
    done
  done

  echo ""
  if [[ $TOTAL_FAIL -gt 0 ]]; then
    echo "WARN: $TOTAL_FAIL sample run(s) failed (see logs/euler/branch_sensitivity/run.log)"
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " SUMMARY"
echo "=========================================="
for prec in $PRECS; do
  for tid in 1 2 3 4 5; do
    tname="${TESTNAME[$tid]}"
    d="results/euler/mca_fcmp/${prec}/${tname}/hllc"
    if [[ -d "$d" ]]; then
      n=$(ls "$d"/sample_*.dat 2>/dev/null | wc -l)
      printf "  %s/%s: %d samples\n" "$prec" "$tname" "$n"
    fi
  done
done

cat <<'EOF'

Next: compare baseline MCA vs --inst-fcmp MCA significant digits.
  Baseline:    python3 analysis/euler/mca_analysis.py --prec fp32
               (reads results/euler/mca/fp32/...)
  fcmp:        edit analysis/euler/mca_analysis.py to set
                 MCA_DIR = f"results/euler/mca_fcmp/{PREC_NAME}"
               then run with --prec fp32 again, compare summary text.
EOF
