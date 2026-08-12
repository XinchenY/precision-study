# Compressible-Flow Solvers & Floating-Point Precision Study
# 可压缩流体求解器与浮点精度研究

**English** | [中文说明见下半部分](#中文说明)

---

## Overview

This repository contains the complete code base of an MPhil project studying
**how floating-point precision affects finite-volume solvers for compressible
flow**. It provides:

- **Solvers.** 1D/2D Euler (HLL/HLLC) and 1D/2D ideal MHD with mixed
  GLM divergence control (HLL/HLLD, MUSCL–Hancock), written once as
  header-only numerical kernels and compiled for both **CPU (g++)** and
  **GPU (CUDA)** from the same source, so the two backends are bit-comparable.
- **Precision instrumentation.**
  - *Global*: FP64 vs FP32 builds (one `-DUSE_FLOAT` switch), and
    Verificarlo **MCA/VPREC** bit-width scans (how many mantissa bits are
    enough?).
  - *Per-stage*: **RAPTOR** experiments that lower the arithmetic of one
    solver stage at a time (recovery / CFL / reconstruction / flux / update
    / GLM) to binary32, identifying which stages tolerate FP32;
    a **CUDA mixed-precision solver** then realises the proposed allocation
    in hardware.
  - *Compiler-level*: `-O3`, `-Ofast`, `-fmad=false`, `--use_fast_math`
    accuracy and runtime controls.
- **Nonlinear robustness tests.** Orszag–Tang vortex, Brio–Wu and Sod shock
  tubes, and a Kelvin–Helmholtz test with a deterministic single-mode
  perturbation for a fair FP32-vs-FP64 comparison deep into the nonlinear
  regime.
- **A measurement discipline.** Bitwise regression gates for every code
  change, interleaved timing batteries with pre-registered acceptance
  criteria for shared-node benchmarking, and a provenance ledger
  (`logs/mhd/ch10_runs.csv`) from which **every number in the write-up can be
  traced back to a command**.

## Repository layout

```
include/   header-only numerical kernels (euler/, mhd/), shared Real/HD macros
src/       main programs: {euler,mhd} × {CPU .cpp, GPU .cu},
           *_raptor.cpp (stage-level precision), mhd2d_glm_gpu_mixed.cu
analysis/  Python plotting & comparison tools (compare_precision.py, figures/)
scripts/   experiment batteries, watchers, manifest generator
logs/      run logs, timing CSVs, comparison tables, provenance ledger
docs/      REPRODUCING.md — build matrix, run conventions, table→data map
*.sh       top-level experiment drivers (run from repository root)
```

Raw simulation output (`results/`, ~1.5 GB) and compiled binaries (`bin/`)
are **not** tracked; `docs/REPRODUCING.md` gives the exact commands to
regenerate them and the md5 manifest to verify bitwise agreement.

## Quick start

```bash
# CPU, FP64 (baseline -O2; add -DUSE_FLOAT for FP32; swap -O2 for -O3/-Ofast)
g++ -O2 -std=c++14 -I include -DMHD_HLLD_TOLERANCE_VALUE=1.0e-4 \
    src/mhd/mhd2d_glm_cpu.cpp -o bin/mhd2d_fp64

# GPU (requires sm_80-class device, e.g. NVIDIA A30)
nvcc -O2 -std=c++14 -arch=sm_80 -I include -DMHD_HLLD_TOLERANCE_VALUE=1.0e-4 \
    src/mhd/mhd2d_glm_gpu.cu -o bin/mhd2d_glm_gpu_fp64

# Orszag-Tang, 512², t=0.5           # Kelvin-Helmholtz, t=5
./bin/mhd2d_glm_gpu_fp64 512 512 0.5 0.25 muscl hlld 0.18
./bin/mhd2d_glm_gpu_fp64 512 512 5.0 0.25 muscl hlld 0.18 kh

# Compare two runs variable-by-variable (relative L2)
python3 analysis/mhd/compare_precision.py fp64.dat fp32.dat
```

Full build matrix (RAPTOR, VPREC, compiler variants) and the
table-by-table reproduction map: **[docs/REPRODUCING.md](docs/REPRODUCING.md)**.

## Requirements

- g++ ≥ 9 (C++14), CUDA toolkit with `sm_80` support, Python 3 + NumPy +
  Matplotlib.
- Optional, for the precision-emulation experiments only:
  [Verificarlo](https://github.com/verificarlo/verificarlo) 2.4.0 (VPREC/MCA)
  and [RAPTOR](https://github.com/RIKEN-RCCS/RAPTOR) with clang 20.

---

## 中文说明

本仓库是一个 MPhil 项目的完整代码:研究**浮点精度对可压缩流体有限体积求解器
的影响**。

- **求解器**:1D/2D Euler(HLL/HLLC)与 1D/2D 理想 MHD + mixed GLM 散度控制
  (HLL/HLLD,MUSCL–Hancock)。数值内核只写一份(header-only),CPU(g++)与
  GPU(CUDA)编译同一份源码,两个后端可逐位对比。
- **精度实验三层**:
  - *全局*:FP64/FP32 双构建(`-DUSE_FLOAT` 一键切换)+ Verificarlo
    **MCA/VPREC** 位宽扫描(多少位尾数才够?);
  - *按阶段*:**RAPTOR** 每次只把一个求解阶段(守恒量恢复/CFL/重构/通量/更新/
    GLM)的运算降到 binary32,找出哪些阶段能容忍 FP32;再用
    **CUDA mixed-precision 求解器**把该分配真实现到硬件上验证;
  - *编译器级*:`-O3`/`-Ofast`/`-fmad=false`/`--use_fast_math`
    的精度与运行时对照。
- **非线性稳健性**:Orszag–Tang 涡、Brio–Wu/Sod 激波管、以及带确定性单模扰动的
  Kelvin–Helmholtz 测试(FP32 与 FP64 逐位同扰动,公平对比深入非线性阶段)。
- **测量纪律**:每次改码过逐位回归门禁;共享节点计时用轮转交错电池 +
  预登记验收判据;中央台账 `logs/mhd/ch10_runs.csv` 让**正文每个数字都能反查到
  命令级**。

目录结构、编译矩阵、"表格→数据→脚本"映射见
**[docs/REPRODUCING.md](docs/REPRODUCING.md)**。
原始数据(`results/`,约 1.5 GB)与二进制(`bin/`)不入库,按手册命令可重生成,
并用 md5 清单验证逐位一致。
