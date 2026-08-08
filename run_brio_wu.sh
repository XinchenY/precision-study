#!/usr/bin/env bash
# =============================================================================
#  run_brio_wu.sh — 1D 理想 MHD Brio-Wu 激波管:一键编译 + 跑全套
#                   1D ideal-MHD Brio-Wu shock tube: build + run the full matrix
# =============================================================================
#
#  这个脚本做三件事 / This script does three things:
#
#    (1) 编译 CPU 求解器的 fp64 和 fp32 两个版本
#        Compile the CPU solver in both fp64 and fp32
#
#    (2) 跑一个矩阵 = 分辨率 × 精度 × 黎曼求解器
#        Run a matrix = resolution × precision × Riemann solver
#          N       ∈ { 800 }          网格格子数     / grid cells
#          prec    ∈ { fp64, fp32 }   浮点精度       / precision
#          solver  ∈ { hll,  hlld }   黎曼求解器     / Riemann solver
#        固定量 / fixed:  t_end=0.1, CFL=0.4, 重构=MUSCL-Hancock (Brio-Wu 标准)
#
#    (3) 数据存 results/mhd/brio_wu/ , 日志存 logs/mhd/  (两者分开,不混)
#        Data -> results/mhd/brio_wu/ , logs -> logs/mhd/  (kept separate)
#
#  文件名由求解器自己决定 / filenames are chosen by the solver:
#    results/mhd/brio_wu/brio_wu_{order}_{solver}_{prec}_N{N}.dat
#    logs/mhd/brio_wu_{order}_{solver}_{prec}_N{N}.log
#
#  用法 / Usage:
#    bash run_brio_wu.sh                # 全套:先编译再跑 / build then run (default)
#    bash run_brio_wu.sh --build-only   # 只编译 / compile only
#    bash run_brio_wu.sh --run-only     # 只跑(二进制须已存在) / run only
#
#  想改矩阵?用环境变量覆盖,不必改脚本 / override via env vars, e.g.:
#    N_LIST="800 1600" SOLVERS="hlld" bash run_brio_wu.sh
#
#  在 lovelace 和 athena 上跑法完全一样(共享文件系统)。
#  Runs identically on lovelace and athena (shared filesystem).
# =============================================================================

set -euo pipefail          # 出错即停 / 未定义变量报错 / 管道里的错误不被吞掉
cd "$(dirname "$0")"       # 切到脚本所在目录(= 项目根),让下面的相对路径都成立
                           # cd into the script's own dir (= project root)

# ── 可调参数(都能用同名环境变量覆盖)/ Tunable settings (env-overridable) ──────
N_LIST=${N_LIST:-"800"}          # 网格分辨率(默认只 800;想加分辨率: N_LIST="400 800")
PRECS=${PRECS:-"fp64 fp32"}      # 精度列表       / precisions
SOLVERS=${SOLVERS:-"hll hlld"}   # 求解器列表     / Riemann solvers
ORDER=${ORDER:-muscl}            # 重构: muscl(二阶) 或 first(一阶) / reconstruction
T_END=${T_END:-0.1}              # 终止时间 / final time  (Brio-Wu 标准 = 0.1)
CFL=${CFL:-0.4}                  # CFL 数   / CFL number

SRC="src/mhd/mhd1d_cpu.cpp"      # 源文件 / source file
CXX=${CXX:-g++}                  # 编译器 / compiler
CXXFLAGS="-O2 -std=c++14 -I include"   # 优化 / C++ 标准 / 头文件搜索路径

# ── 命令行开关 / Command-line flags ──────────────────────────────────────────
DO_BUILD=true
DO_RUN=true
for arg in "$@"; do
  case "$arg" in
    --build-only) DO_RUN=false ;;
    --run-only)   DO_BUILD=false ;;
    *) echo "未知参数 / unknown flag: $arg"; exit 1 ;;
  esac
done

# 先把输出目录建好(mkdir -p:不存在才建,已存在不报错)
# Create output dirs up front (mkdir -p is idempotent)
mkdir -p bin results/mhd/brio_wu logs/mhd

# ── 编译阶段 / BUILD phase ────────────────────────────────────────────────────
if $DO_BUILD; then
  echo "=========================================="
  echo " 编译 / COMPILE"
  echo "=========================================="
  for prec in $PRECS; do
    # fp32 需要 -DUSE_FLOAT 把 Real 切成 float;fp64 不加,默认就是 double
    # fp32 needs -DUSE_FLOAT (Real=float); fp64 adds nothing (Real=double)
    flag=""
    [ "$prec" = "fp32" ] && flag="-DUSE_FLOAT"
    out="bin/mhd1d_cpu_${prec}"
    printf "  %-5s -> %-24s ... " "$prec" "$out"
    $CXX $CXXFLAGS $flag "$SRC" -o "$out"
    echo "OK"
  done
fi

# ── 运行阶段 / RUN phase ──────────────────────────────────────────────────────
if $DO_RUN; then
  echo ""
  echo "=========================================="
  echo " 运行 / RUN   数据->results/mhd/brio_wu   日志->logs/mhd"
  echo "=========================================="
  n_ok=0
  n_fail=0
  for prec in $PRECS; do
    bin="bin/mhd1d_cpu_${prec}"
    if [ ! -x "$bin" ]; then
      echo "  缺二进制 / missing binary: $bin  (先跑 bash run_brio_wu.sh --build-only)"
      exit 1
    fi
    for N in $N_LIST; do
      for solver in $SOLVERS; do
        tag="${ORDER}_${solver}_${prec}_N${N}"
        log="logs/mhd/brio_wu_${tag}.log"
        printf "  %-5s N=%-5s %-4s ... " "$prec" "$N" "$solver"

        # 关键一行:OUTDIR 指定 .dat 落到哪;程序的屏幕输出(stdout+stderr)重定向进日志
        # Key line: OUTDIR sets where the .dat goes; console output goes to the log
        # 参数顺序 / arg order:  N  t_end  CFL  order  solver
        if OUTDIR="results/mhd/brio_wu" "./$bin" "$N" "$T_END" "$CFL" "$ORDER" "$solver" \
             > "$log" 2>&1; then
          # 从日志里抓步数,确认真的跑完了 / pull step count to confirm it finished
          steps=$(grep -oE 'Steps: [0-9]+' "$log" | grep -oE '[0-9]+' | head -1 || true)
          echo "OK (steps=${steps:-?}) -> brio_wu_${tag}.dat"
          n_ok=$((n_ok + 1))
        else
          # 某一格失败(比如 fp32 出 NaN)不中断整个矩阵,继续跑剩下的
          # One failing case (e.g. fp32 NaN) does NOT abort the matrix; keep going
          echo "FAIL  (原因见 / see $log)"
          n_fail=$((n_fail + 1))
        fi
      done
    done
  done
  echo "------------------------------------------"
  echo "  完成 / done:  $n_ok 成功 / ok,  $n_fail 失败 / failed"
fi

# ── 小结 / SUMMARY ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " 结果清单 / RESULTS"
echo "=========================================="
echo "  数据 / data  (results/mhd/brio_wu/):"
ls -1 results/mhd/brio_wu/*.dat 2>/dev/null | sed 's/^/    /' || echo "    (无 / none)"
echo "  日志 / logs  (logs/mhd/):"
ls -1 logs/mhd/brio_wu_*.log 2>/dev/null | sed 's/^/    /' || echo "    (无 / none)"
echo ""
echo "  画图示例 / plot example:"
echo "    python3 analysis/mhd/plot_brio_wu.py \\"
echo "      results/mhd/brio_wu/brio_wu_muscl_hlld_fp64_N800.dat \\"
echo "      analysis/mhd/figures/brio_wu_hlld_fp64_N800.png"
