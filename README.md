# 可压缩流体求解器 — 浮点数值精度研究

单一代码库，统一框架，**按"关注点 × 物理模块"两层组织**：每个 `include/ src/ analysis/ results/`
下再分 `euler/`（已完成）与 `mhd/`（进行中）。研究浮点运算在不同精度/硬件/编译器下对解的
可靠性影响，用 Verificarlo Monte-Carlo Arithmetic (MCA) 量化有效十进制位数 `s_d = -log₁₀|σ/μ|`。

## 目录结构

```
include/
  base.hpp                                                   共享样板: Real 类型 + HD 宏
  euler/   config.hpp euler_common.hpp riemann.hpp          Euler 数值内核
  mhd/     mhd_common.hpp mhd_glm_common.hpp                 MHD 数值内核 (8 变量 + 磁场)
src/
  euler/   riemann1d/euler2d/riemann2d × {cpu.cpp, gpu.cu}   6 个主程序
  mhd/     mhd1d_cpu.cpp mhd2d_cpu.cpp mhd2d_glm_cpu.cpp      MHD 主程序 (目前仅 CPU)
analysis/
  euler/   *.py + fp32/ fp64/ figures/                       分析脚本与图
  mhd/     plot_brio_wu.py plot_orszag_tang*.py              MHD 画图 (命令行参数式)
results/
  euler/   mca/ ieee/ gpu/ 2d/ compiler/                     Euler 结果
  mhd/     brio_wu/ orszag_tang/                             MHD 结果 (OUTDIR 默认落此)
logs/
  euler/   fp32/ fp64/ compiler/ branch_sensitivity/         Euler 运行日志
  mhd/     brio_wu_*.log                                     MHD 运行日志 (stdout 重定向)
*.sh                                                          实验脚本（从项目根运行）
archive/   euler_audit_table57.tar.gz                        Table 5.7 复现审计快照
```

## 统一框架约定（Euler 与 MHD 通用）

- **单一数值内核，CPU/GPU 共用**：头文件放 `include/<module>/`，全部标 `__host__ __device__`（`HD` 宏），
  `.cpp`(g++) 与 `.cu`(nvcc) 编译同一份源码，保证算法逐位一致。
- **头文件引用**：编译带 `-I include`，源码写 `#include "euler/riemann.hpp"` / `"mhd/mhd_common.hpp"`（与源码所在深度无关）。
- **精度开关**：`-DUSE_FLOAT` 一键切 `typedef Real`（double ↔ float）。
- **输出重定向**：源码读 `OUTDIR` 环境变量（默认落到 `results/<module>/...`），脚本无需改源码即可分流。
- **共享样板 `include/base.hpp`**：只含 `Real` 类型 + `HD` 宏，Euler 与 MHD 都可 `#include "base.hpp"`。
  MHD 头文件用它拿 Real/HD，**不** `#include` Euler 专属的 `config.hpp`（那里还有 GAMMA/NX/domain
  等 shock-bubble 物理宏，MHD 不需要；MHD 物理参数自带，如 `MHD_GAMMA`、运行时 `nx/ghost`）。

## 实验脚本（Euler，从项目根运行）

| 脚本 | 内容 |
|------|------|
| `sanity_check.sh`        | Verificarlo 工具链验证（IEEE 后端 L∞ 阈值） |
| `run_gpu.sh`             | GPU 1D 全测试（Toro 1–5 × HLLC/HLL × FP64/FP32） |
| `fp32_backend.sh`        | CPU/GPU × FP64/FP32 的 2×2 矩阵（1D + 2D） |
| `compiler_experiment.sh` | 编译器敏感性：O2/O3/Ofast × nvcc default/fastmath |
| `branch_sensitivity.sh`  | MCA `--inst-fcmp`（浮点比较也受扰动） |

分析入口：`python3 analysis/euler/mca_analysis.py --prec fp64`（或 `fp32`）。

## results/euler/ 里有什么

| 子目录 | 内容 |
|--------|------|
| `results/euler/mca/{fp32,fp64}/test{1-5}/{hllc,hll}/sample_*.dat` | **600 个 MCA 采样**（核心数据，重跑昂贵） |
| `results/euler/ieee/{fp32,fp64}/`      | CPU IEEE 基准解 (1D) |
| `results/euler/gpu/{fp32,fp64}/`       | GPU 解 (1D) |
| `results/euler/2d/{fp32,fp64}/`        | 2D 解（激波-气泡 + Riemann） |
| `results/euler/compiler/fp64/<build>/` | 编译器敏感性结果（5 builds × 3 cases） |

## 快速复现（Euler）

```bash
bash sanity_check.sh --prec fp64                      # 验证工具链（已测通过）
bash run_gpu.sh                                       # 重建并跑 GPU 1D
python3 analysis/euler/mca_analysis.py --prec fp64    # 有效位数分析
```

> 二进制统一输出到项目根的 `bin/`（构建脚本自动创建，已在 .gitignore 语义上视为可再生产物）。

## 提交/复现指引 (2026-08 更新)

- 状态更新:MHD 已含 GPU 后端(`src/mhd/*.cu`,含 mixed-precision 与 KH 版)、
  RAPTOR/VPREC 精度实验、完整计时协议。
- **复现手册:`docs/REPRODUCING.md`**(编译矩阵、运行约定、表格→数据映射、门禁纪律)。
- 中央台账:`logs/mhd/ch10_runs.csv`;大数据不入库,md5 清单 `data/MANIFEST.md5`
  (`scripts/make_manifest.sh` 生成)。
