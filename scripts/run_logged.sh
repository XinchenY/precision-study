#!/usr/bin/env bash
# =============================================================================
#  Ch10 运行台账 wrapper: 每个 run 自动追加一行到 logs/mhd/ch10_runs.csv
#  用法:
#    scripts/run_logged.sh <step> <case> <outdir> <logfile> -- <command ...>
#  例:
#    VFC_BACKENDS="libinterflop_vprec.so --precision-binary64=23 --mode=ob" \
#    scripts/run_logged.sh 2d ot192_t23 results/mhd/vprec/scan2d/t23 \
#      logs/mhd/vprec_scan2d/t23.log -- \
#      ./bin/vprec_step1/mhd2d_vfc_fp64_eps1e-4 192 192 0.5 0.25 muscl hlld 0.18
#  记录列: start_iso,step,case,command,vfc_backends,outdir,wall_s,exit,steps,
#          fallbacks,logfile
# =============================================================================
set -u
cd "$(dirname "$0")/.."
LEDGER=logs/mhd/ch10_runs.csv
[ -f "$LEDGER" ] || echo "start_iso,step,case,command,vfc_backends,outdir,wall_s,exit,steps,fallbacks,logfile" > "$LEDGER"

STEP=$1; CASE=$2; OUT=$3; LOGF=$4; shift 4
[ "$1" = "--" ] && shift

START_ISO=$(date -Is)
t0=$(date +%s)
OUTDIR=$OUT "$@" > "$LOGF" 2>&1
RC=$?
t1=$(date +%s)
WALL=$((t1 - t0))

STEPS=$(grep -oP 'Done\. Steps: \K[0-9]+' "$LOGF" | tail -1)
FB=$(grep -oP 'fallbacks = \K[0-9]+' "$LOGF" | tail -1)
CMD=$(printf '%s ' "$@" | sed 's/,/;/g; s/ $//')
ENVSTR=$(echo "${VFC_BACKENDS:-}" | sed 's/,/;/g')

echo "$START_ISO,$STEP,$CASE,\"$CMD\",\"$ENVSTR\",$OUT,$WALL,$RC,${STEPS:-},${FB:-},$LOGF" >> "$LEDGER"
echo "exit=$RC wall=${WALL}s" >> "$LOGF"
exit $RC