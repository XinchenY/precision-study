#!/usr/bin/env bash
# =============================================================================
#  verify_hlld_fix.sh — HLLD scale-aware + 正压回退修复的完整验证组
#
#  A) 非回归 (必须逐位相同, 否则 9.2.2 表格需重生成):
#       OT 128²/192² × fp64/fp32 @ CFL=0.25  vs  results/mhd/orszag_tang/ 现有数据
#  B) 修复效果 (旧代码会崩的配置, 新代码应跑通):
#       OT 256²/384² × fp64/fp32 @ CFL=0.25 和 @ CFL=0.4
#
#  用法:  bash verify_hlld_fix.sh          # 编译 + 全部 12 个 run (并行)
#  输出:  /tmp/hlld_fix_verify/            # 本机临时, 不碰 results/
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

OUT=/tmp/hlld_fix_verify
rm -rf "$OUT"; mkdir -p "$OUT/nr" "$OUT/fix" "$OUT/logs"

echo "=== 编译 (含修复的最新源码) ==="
g++ -O2 -std=c++14 -I include src/mhd/mhd2d_glm_cpu.cpp -o "$OUT/glm_fp64" || exit 1
g++ -O2 -std=c++14 -DUSE_FLOAT -I include src/mhd/mhd2d_glm_cpu.cpp -o "$OUT/glm_fp32" || exit 1
echo "  OK (fp64 + fp32)"

# ── A) 非回归: 128/192 @0.25, 并行 ───────────────────────────────────────────
echo "=== 启动 A) 非回归 runs (4 个, 并行) ==="
for N in 128 192; do
  for prec in fp64 fp32; do
    ( OUTDIR="$OUT/nr" nice -n 19 "$OUT/glm_${prec}" $N $N 0.5 0.25 muscl hlld \
        > "$OUT/logs/nr_${prec}_N${N}.log" 2>&1 ) &
  done
done

# ── B) 修复效果: 256/384 × 两精度 × 两 CFL, 并行 ─────────────────────────────
echo "=== 启动 B) 修复效果 runs (8 个, 并行; 384² 较慢) ==="
for cfl in 0.25 0.4; do
  for N in 256 384; do
    for prec in fp64 fp32; do
      tag="${prec}_N${N}_cfl${cfl}"
      ( OUTDIR="$OUT/fix/cfl${cfl}" nice -n 19 "$OUT/glm_${prec}" $N $N 0.5 $cfl muscl hlld \
          > "$OUT/logs/fix_${tag}.log" 2>&1 ) &
    done
  done
done

echo "等待全部 12 个 run 完成 ..."
wait
echo

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo "==================================================================="
echo " A) 非回归 (期望: 全部'逐位相同' → 论文 9.2.2 表格原封不动)"
echo "==================================================================="
for N in 128 192; do
  for prec in fp64 fp32; do
    new="$OUT/nr/orszag_tang_glm_muscl_hlld_${prec}_N${N}x${N}.dat"
    old="results/mhd/orszag_tang/orszag_tang_glm_muscl_hlld_${prec}_N${N}x${N}.dat"
    if [ -f "$new" ] && diff -q "$new" "$old" >/dev/null 2>&1; then
      echo "  ${prec} ${N}²: 逐位相同 ✓ (新 guard 未触发)"
    elif [ -f "$new" ]; then
      echo "  ${prec} ${N}²: 有差异 → guard 在此分辨率已触发, 表格需用新数据重生成"
    else
      echo "  ${prec} ${N}²: 没跑完!? 见 $OUT/logs/nr_${prec}_N${N}.log"
    fi
  done
done
echo
echo "==================================================================="
echo " B) 修复效果 (旧代码: fp32 崩 256²@2340 / fp64 崩 384²@3600, 均 CFL 0.25)"
echo "==================================================================="
for cfl in 0.25 0.4; do
  for N in 256 384; do
    for prec in fp64 fp32; do
      log="$OUT/logs/fix_${prec}_N${N}_cfl${cfl}.log"
      if grep -q 'Saved:' "$log" 2>/dev/null; then
        st=$(grep -oE 'Steps: [0-9]+' "$log" | grep -oE '[0-9]+' | head -1)
        echo "  CFL=$cfl  ${prec} ${N}²: 跑通 ✓ (${st} steps)"
      else
        err=$(grep -iE 'error|non-physical' "$log" 2>/dev/null | head -1)
        echo "  CFL=$cfl  ${prec} ${N}²: 崩 ✗  ${err}"
      fi
    done
  done
done
echo
echo "结论提示: 若 A 全同 + B@0.25 全通 → 修复成立且论文数字零改动;"
echo "          B@0.4 若 fp64 仍崩 → 0.4 的失稳不在 HLLD 退化层, 正式设置维持 0.25。"
