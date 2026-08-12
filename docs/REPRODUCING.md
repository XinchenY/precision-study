# 复现论文结果 / Reproducing the thesis results

所有正文数字都能沿「表格 → 台账 → CSV/cmp 文件 → 原始命令」反查。
Central ledger: `logs/mhd/ch10_runs.csv`（每次运行的时间、命令、输出 md5、接受/拒绝原因）。

`results/` 不入库（约 1.5 GB）：原始结果已省略；关键 `.dat` 的校验/验证请使用仓内脚本重现。
Raw `results/` are omitted from the repository (approx. 1.5 GB); raw results are omitted and reproducible from the provided scripts.

## 1. 编译矩阵 / 1. Build matrix

### CPU(g++,MHD 2D GLM;ε 协议统一 1e-4)

```bash
g++ -O2 -std=c++14 -I include -DMHD_HLLD_TOLERANCE_VALUE=1.0e-4 \
    src/mhd/mhd2d_glm_cpu.cpp -o bin/mhd2d_fp64_eps1e-4          # 基线 -O2
# 变体: 把 -O2 换成 -O3 / -Ofast;FP32: 加 -DUSE_FLOAT
```

CPU (g++, MHD 2D GLM; ε protocol unified 1e-4)

Variants: replace -O2 with -O3 / -Ofast; for FP32 add -DUSE_FLOAT.

### GPU(nvcc,必须 -arch=sm_80;A30)

```bash
nvcc -O2 -std=c++14 -arch=sm_80 -I include -DMHD_HLLD_TOLERANCE_VALUE=1.0e-4 \
    src/mhd/mhd2d_glm_gpu.cu -o bin/mhd2d_glm_gpu_fp64_eps1e-4
# FP32: 加 -DUSE_FLOAT;变体: 加 -fmad=false 或 --use_fast_math
# mixed(存储 FP64、stage 级 FP32): 同 flags 编 src/mhd/mhd2d_glm_gpu_mixed.cu
```

GPU (nvcc, require -arch=sm_80; tested on A30)

For FP32 add -DUSE_FLOAT; variants: -fmad=false or --use_fast_math. Mixed builds store FP64 but use FP32 at some stages — see src/mhd/mhd2d_glm_gpu_mixed.cu.

### RAPTOR(stage 级精度仿真;clang 20.1 + RAPTOR 插件)

```bash
clang++ -O2 -std=c++14 -I include src/mhd/mhd1d_raptor.cpp \
    -o bin/raptor/mhd1d_extracted                                # V1 门禁参照
raptor-clang++ -O2 -std=c++14 -DUSE_RAPTOR -I include \
    src/mhd/mhd1d_raptor.cpp -o bin/raptor/mhd1d_raptor          # 插桩版
# 2D: mhd2d_glm_raptor.cpp;Euler Sod: src/euler/riemann1d_raptor.cpp
```

RAPTOR (stage-precision simulation; clang 20.1 + RAPTOR plugin)

## 2. 运行约定 / 2. Run conventions

```bash
./bin/mhd2d_glm_gpu_fp64_eps1e-4 512 512 0.5 0.25 muscl hlld 0.18        # OT
./bin/mhd2d_glm_gpu_fp64_prob_eps1e-4 512 512 5.0 0.25 muscl hlld 0.18 kh # KH
./bin/raptor/mhd1d_raptor 800 0.1 0.4 muscl hlld recovery,flux            # RAPTOR
```

参数/Parameters: `nx ny t_end cfl order solver alpha [problem|regions]`。
输出目录用环境变量 `OUTDIR` 重定向(默认 `results/…`)。

The output directory can be redirected with the OUTDIR environment variable (default `results/…`).

## 3. 正文表格/图 → 数据来源 / 3. Tables/Figures → Data sources

| 表/图 | 数据文件 | 生成脚本/命令 |
|---|---|---|
| Table 9.9(编译器/浮点控制精度) | `logs/mhd/gpu_backend_20260718/cmp_*.txt` | compare_precision.py 对成对 .dat |
| Table 10.1(512² 计时) | `logs/mhd/perf_baseline/*.csv` | `scripts/run_variants_*.sh`、`run_gpu_baseline_v2.sh` |
| VPREC 扫描（Section 10.3; Tables 10.3--10.4） | `logs/mhd/vprec_scan{1d,2d}/` | `scripts/run_vprec_scan1d.sh` |
| Table 10.6/10.7(RAPTOR) | `logs/mhd/raptor_scan/*.txt` | `run_raptor_2d_scan.sh` + 第 2 节命令 |
| mixed 实测(10.5) | `logs/mhd/perf_baseline/mixed_rr.csv`、`logs/mhd/raptor_scan/ot2d512_cmp_mixed_gpu.txt` | mixed 二进制 + compare_precision.py |
| KH 三联图 | `results/mhd/gpu/kh/*.dat`、`logs/mhd/kh/` | 第 2 节 KH 命令 ×2 精度 → `analysis/mhd/plot_kh_triptych.py` |

Mapping from tables/figures to source data files and generation scripts. See the table above for the exact file paths and commands.

## 4. 门禁纪律(改代码后必须过) / 4. Acceptance gates (must pass after code changes)

1. 计时/改动版二进制:OT 192² fp64 **数据行 md5** 必须等于冻结参考 `d54b0169…` (`grep -v '^#' out.dat | md5sum`), HLLD fallback 计数 13372；
2. RAPTOR:V1(region 化重构逐位同原版)、V3(插桩+空 region 列表逐位同 V1)；
3. 计时:共享节点上按 `cpu_ratio>0.95` + 同轮 O2 对照复现冻结基线 ±3% 验收 (判据先于数据登记进台账)。

1. Timing/modified binaries: OT 192² fp64 data row MD5 must match the frozen reference (`d54b0169…`) — compute with `grep -v '^#' out.dat | md5sum`. HLLD fallback count: 13372.
2. RAPTOR: V1 (regionized reconstruction bitwise-equal to original), V3 (instrumented + empty region list bitwise-equal to V1).
3. Timing: reproduce baseline within ±3% on a shared node under `cpu_ratio>0.95` with the same O2 round (acceptance criteria recorded in the ledger prior to data registration).
