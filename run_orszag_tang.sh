#!/usr/bin/env bash
# =============================================================================
#  run_orszag_tang.sh — 2D 理想 MHD Orszag-Tang(GLM 散度清理版):编译 + 跑
#                       2D ideal-MHD Orszag-Tang (GLM cleaning): build + run
# =============================================================================
#
#  用的是 GLM 版求解器 mhd2d_glm_cpu(9 变量,含 ψ 散度清理)—— 这是正式方法,
#  不是 base(不清散度)那个。
#  Uses the GLM solver mhd2d_glm_cpu (9-var, ψ divergence cleaning) — the real
#  method, not the base (no-cleaning) one.
#
#  矩阵 = 精度(fp64, fp32)。求解器固定 HLLD + MUSCL-Hancock,gamma=5/3。
#  Matrix = precision {fp64, fp32}. Solver fixed to HLLD + MUSCL-Hancock.
#
#  输出 / output:
#    results/mhd/orszag_tang/orszag_tang_glm_{order}_{solver}_{prec}_N{n}x{n}.dat
#    logs/mhd/orszag_tang_glm_{order}_{solver}_{prec}_N{n}x{n}.log
#    (.dat 列 / columns: x y rho vx vy vz p Bx By Bz E divB psi)
#
#  用法 / Usage:
#    bash run_orszag_tang.sh                 # 编译 + 跑 4 个分辨率(128/192/256/384)
#                                            # = fp32-vs-fp64 分辨率研究一次出齐
#    bash run_orszag_tang.sh --build-only
#    bash run_orszag_tang.sh --run-only
#    N_LIST="512" bash run_orszag_tang.sh    # 单独加更大分辨率
#
#  注意 / NOTE:  256²/384² 计算量大(384² 约为 128² 的几十倍),请在 lovelace
#  上跑(空闲、快,且与 athena 逐位相同)。athena 负载高只适合小分辨率验证。
#  fp32 在 384² 可能崩 —— 脚本对单个失败不中断,fp64 与其余分辨率照常产出。
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── 可调参数(环境变量可覆盖)/ tunable (env-overridable) ─────────────────────
N_LIST=${N_LIST:-"128 192 256 384"}   # 分辨率研究: 4 个点 / resolution study (fp32-vs-fp64 趋势)
PRECS=${PRECS:-"fp64 fp32"}      # 精度 / precisions
SOLVER=${SOLVER:-hlld}           # 求解器(GLM 用 hlld)/ Riemann solver
ORDER=${ORDER:-muscl}            # 重构 / reconstruction
T_END=${T_END:-0.5}              # 终止时间(OT 标准)/ final time
CFL=${CFL:-0.25}                 # CFL(GLM 常用略低)/ CFL number

SRC="src/mhd/mhd2d_glm_cpu.cpp"
CXX=${CXX:-g++}
CXXFLAGS="-O2 -std=c++14 -I include"   # 注意:无 -march,保证跨机逐位一致

# ── CLI ──────────────────────────────────────────────────────────────────────
DO_BUILD=true; DO_RUN=true
for arg in "$@"; do
  case "$arg" in
    --build-only) DO_RUN=false ;;
    --run-only)   DO_BUILD=false ;;
    *) echo "未知参数 / unknown flag: $arg"; exit 1 ;;
  esac
done

mkdir -p bin results/mhd/orszag_tang logs/mhd

# ── 编译 / BUILD ──────────────────────────────────────────────────────────────
if $DO_BUILD; then
  echo "========== 编译 / COMPILE (GLM) =========="
  for prec in $PRECS; do
    flag=""; [ "$prec" = "fp32" ] && flag="-DUSE_FLOAT"
    out="bin/mhd2d_glm_cpu_${prec}"
    printf "  %-5s -> %-26s ... " "$prec" "$out"
    $CXX $CXXFLAGS $flag "$SRC" -o "$out"
    echo "OK"
  done
fi

# ── 运行 / RUN ────────────────────────────────────────────────────────────────
if $DO_RUN; then
  echo "========== 运行 / RUN  数据->results/mhd/orszag_tang  日志->logs/mhd =========="
  n_ok=0; n_fail=0
  for prec in $PRECS; do
    bin="bin/mhd2d_glm_cpu_${prec}"
    [ -x "$bin" ] || { echo "  缺二进制 / missing: $bin (先 --build-only)"; exit 1; }
    for N in $N_LIST; do
      tag="glm_${ORDER}_${SOLVER}_${prec}_N${N}x${N}"
      log="logs/mhd/orszag_tang_${tag}.log"
      printf "  %-5s N=%-4s %-4s ... " "$prec" "$N" "$SOLVER"
      # 参数顺序 / arg order:  nx ny t_end cfl order solver
      if OUTDIR="results/mhd/orszag_tang" "./$bin" "$N" "$N" "$T_END" "$CFL" "$ORDER" "$SOLVER" \
           > "$log" 2>&1; then
        steps=$(grep -oE '[Ss]teps?[: ]+[0-9]+' "$log" | grep -oE '[0-9]+' | tail -1 || true)
        db=$(grep -oiE 'divB[^0-9-]*[-0-9.eE+]+' "$log" | tail -1 || true)
        echo "OK (steps=${steps:-?}) -> orszag_tang_${tag}.dat"
        n_ok=$((n_ok+1))
      else
        echo "FAIL / 崩了 (见 $log)"    # fp32 可能崩,不中断
        n_fail=$((n_fail+1))
      fi
    done
  done
  echo "  ----- 完成 / done: $n_ok ok, $n_fail failed -----"
fi

# ── 小结 / SUMMARY ────────────────────────────────────────────────────────────
echo "========== 结果 / RESULTS =========="
echo "  数据 / data (results/mhd/orszag_tang/):"
ls -1 results/mhd/orszag_tang/*.dat 2>/dev/null | sed 's/^/    /' || echo "    (无 / none)"
echo "  日志 / logs (logs/mhd/):"
ls -1 logs/mhd/orszag_tang_*.log 2>/dev/null | sed 's/^/    /' || echo "    (无 / none)"
echo ""
echo "  画等高线图 / contour plot:"
echo "    python3 analysis/mhd/plot_orszag_tang_stone_contours.py \\"
echo "      results/mhd/orszag_tang/<file>.dat  analysis/mhd/figures/<name>.png"
