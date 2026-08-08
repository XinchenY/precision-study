#!/usr/bin/env bash
# =============================================================================
#  Ch10 §10.3: 2D OT 192² region 扫描一条龙 (setsid 脱会话跑)
#  编译(clang-20, ε=1e-4) -> V1/V3 逐位验证(失败中止) -> 7 region + all +
#  keep{flux,update} 组合 -> 原生 fp32 参照 -> 汇总表
#  进度: logs/mhd/raptor_scan/ot2d_progress.log
#  汇总: logs/mhd/raptor_scan/ot2d_table.txt
# =============================================================================
set -u
cd "$(dirname "$0")"
LOG=logs/mhd/raptor_scan
OUT=results/mhd/raptor
exec >> $LOG/ot2d_progress.log 2>&1
echo "== 2D scan start $(date -Is)"

C20=/lsc/opt/llvm-20.1/bin/clang++
E=-DMHD_HLLD_TOLERANCE_VALUE=1e-4
$C20 -O2 -std=c++14 $E -I include src/mhd/mhd2d_glm_cpu.cpp -o bin/raptor/mhd2d_orig_clang20 || exit 1
$C20 -O2 -std=c++14 $E -DUSE_FLOAT -I include src/mhd/mhd2d_glm_cpu.cpp -o bin/raptor/mhd2d_orig_clang20_fp32 || exit 1
$C20 -O2 -std=c++14 $E -I include src/mhd/mhd2d_glm_raptor.cpp -o bin/raptor/mhd2d_extracted || exit 1
scripts/raptor-clang++ -O2 -std=c++14 $E -DUSE_RAPTOR -I include src/mhd/mhd2d_glm_raptor.cpp -o bin/raptor/mhd2d_raptor || exit 1
echo "builds OK $(date -Is)"

R="192 192 0.5 0.25 muscl hlld 0.18"
OUTDIR=$OUT/ot2d_baseline ./bin/raptor/mhd2d_orig_clang20 $R > $LOG/ot2d_orig.log 2>&1 || { echo ABORT-origrun; exit 2; }
B=$OUT/ot2d_baseline/orszag_tang_glm_muscl_hlld_fp64_N192x192.dat
OUTDIR=$OUT/ot2d_ext ./bin/raptor/mhd2d_extracted $R none > $LOG/ot2d_ext.log 2>&1 || { echo ABORT-extrun; exit 2; }
M1=$(grep -v '^#' $B | md5sum | cut -d' ' -f1)
M2=$(grep -v '^#' $OUT/ot2d_ext/orszag_tang_glm_muscl_hlld_fp64_N192x192_r-none.dat | md5sum | cut -d' ' -f1)
[ "$M1" = "$M2" ] || { echo "ABORT V1 MISMATCH $M1 $M2"; exit 3; }
echo "V1 bitwise OK $(date -Is)"
OUTDIR=$OUT/ot2d_rapnone ./bin/raptor/mhd2d_raptor $R none > $LOG/ot2d_rapnone.log 2>&1 || { echo ABORT-rapnone; exit 2; }
M3=$(grep -v '^#' $OUT/ot2d_rapnone/orszag_tang_glm_muscl_hlld_fp64_N192x192_r-none.dat | md5sum | cut -d' ' -f1)
[ "$M1" = "$M3" ] || { echo "ABORT V3 MISMATCH $M1 $M3"; exit 3; }
echo "V3 bitwise OK $(date -Is)"

OUTDIR=$OUT/ot2d_f32 ./bin/raptor/mhd2d_orig_clang20_fp32 $R > $LOG/ot2d_f32.log 2>&1
python3 analysis/mhd/compare_precision.py $B $OUT/ot2d_f32/orszag_tang_glm_muscl_hlld_fp32_N192x192.dat > $LOG/ot2d_cmp_nativefp32.txt 2>&1
echo "native fp32 ref done $(date -Is)"

CONFIGS="recovery cfl recon flux rhs glm update"
for r in $CONFIGS; do
  OUTDIR=$OUT/ot2d_r_$r ./bin/raptor/mhd2d_raptor $R $r > $LOG/ot2d_$r.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    python3 analysis/mhd/compare_precision.py $B $OUT/ot2d_r_$r/orszag_tang_glm_muscl_hlld_fp64_N192x192_r-$r.dat > $LOG/ot2d_cmp_$r.txt 2>&1
  else
    echo "REGION $r CRASHED rc=$rc" > $LOG/ot2d_cmp_$r.txt
  fi
  echo "region $r done rc=$rc $(date -Is)"
done

for c in "recovery,cfl,recon,flux,rhs,glm,update" "recovery,cfl,recon,rhs,glm"; do
  tag=$(echo $c | tr ',' '+')
  OUTDIR=$OUT/ot2d_c_$tag ./bin/raptor/mhd2d_raptor $R $c > $LOG/ot2d_c_$tag.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    python3 analysis/mhd/compare_precision.py $B $OUT/ot2d_c_$tag/orszag_tang_glm_muscl_hlld_fp64_N192x192_r-$tag.dat > $LOG/ot2d_cmp_c_$tag.txt 2>&1
  else
    echo "COMBO $tag CRASHED rc=$rc" > $LOG/ot2d_cmp_c_$tag.txt
  fi
  echo "combo $tag done rc=$rc $(date -Is)"
done

python3 - <<'EOF' > logs/mhd/raptor_scan/ot2d_table.txt
import re, glob
def maxrel(path):
    vals = {}
    try:
        for line in open(path):
            m = re.match(r"\s*(\w+)\s+[\d.e+-]+\s+[\d.e+-]+\s+([\d.e+-]+)\s+([\d.e+-]+)\s*$", line)
            if m and m.group(1) not in ("vz", "Bz"):
                vals[m.group(1)] = (float(m.group(3)), float(m.group(2)))
    except FileNotFoundError:
        return None
    return vals
rows = [("native_fp32", "ot2d_cmp_nativefp32.txt")]
rows += [(r, f"ot2d_cmp_{r}.txt") for r in
         ["recovery","cfl","recon","flux","rhs","glm","update"]]
rows += [("all7", "ot2d_cmp_c_recovery+cfl+recon+flux+rhs+glm+update.txt"),
         ("keepFU", "ot2d_cmp_c_recovery+cfl+recon+rhs+glm.txt")]
print(f"{'config':12} {'max relL2':>12} {'(var)':>6} {'max Linf':>12}")
for name, f in rows:
    v = maxrel(f"logs/mhd/raptor_scan/{f}")
    if not v:
        print(f"{name:12} MISSING/CRASHED"); continue
    var = max(v, key=lambda k: v[k][0])
    linf = max(x[1] for x in v.values())
    print(f"{name:12} {v[var][0]:12.2e} {var:>6} {linf:12.2e}")
EOF
echo "== 2D scan complete $(date -Is)"
