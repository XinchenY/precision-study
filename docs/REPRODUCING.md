# Reproducing the thesis results

所有正文数字都能沿「表格 → 台账 → CSV/cmp 文件 → 原始命令」反查。
中央台账:`logs/mhd/ch10_runs.csv`(每次运行的时间、命令、输出 md5、接受/拒绝原因)。
`results/` 不入库(1.5 GB):关键 `.dat` 的 md5 见 `data/MANIFEST.md5`
(由 `scripts/make_manifest.sh` 生成),按下述命令重生成后比对 md5 即可验证逐位一致。

## 1. 编译矩阵

### CPU(g++,MHD 2D GLM;ε 协议统一 1e-4)

```bash
g++ -O2 -std=c++14 -I include -DMHD_HLLD_TOLERANCE_VALUE=1.0e-4 \
    src/mhd/mhd2d_glm_cpu.cpp -o bin/mhd2d_fp64_eps1e-4          # 基线 -O2
# 变体: 把 -O2 换成 -O3 / -Ofast;FP32: 加 -DUSE_FLOAT
```

### GPU(nvcc,必须 -arch=sm_80;A30)

```bash
nvcc -O2 -std=c++14 -arch=sm_80 -I include -DMHD_HLLD_TOLERANCE_VALUE=1.0e-4 \
    src/mhd/mhd2d_glm_gpu.cu -o bin/mhd2d_glm_gpu_fp64_eps1e-4
# FP32: 加 -DUSE_FLOAT;变体: 加 -fmad=false 或 --use_fast_math
# mixed(存储 FP64、stage 级 FP32): 同 flags 编 src/mhd/mhd2d_glm_gpu_mixed.cu
```

### RAPTOR(stage 级精度仿真;clang 20.1 + RAPTOR 插件)

```bash
clang++ -O2 -std=c++14 -I include src/mhd/mhd1d_raptor.cpp \
    -o bin/raptor/mhd1d_extracted                                # V1 门禁参照
raptor-clang++ -O2 -std=c++14 -DUSE_RAPTOR -I include \
    src/mhd/mhd1d_raptor.cpp -o bin/raptor/mhd1d_raptor          # 插桩版
# 2D: mhd2d_glm_raptor.cpp;Euler Sod: src/euler/riemann1d_raptor.cpp
```

### VPREC(全局位宽扫描;Verificarlo 2.4.0 + clang 18)

见 `scripts/run_vprec_scan1d.sh` / `logs/mhd/vprec_*/`(位宽约定探针:
`scripts/vprec_convention_probe.cpp`)。

## 2. 运行约定

```bash
./bin/mhd2d_glm_gpu_fp64_eps1e-4 512 512 0.5 0.25 muscl hlld 0.18        # OT
./bin/mhd2d_glm_gpu_fp64_prob_eps1e-4 512 512 5.0 0.25 muscl hlld 0.18 kh # KH
./bin/raptor/mhd1d_raptor 800 0.1 0.4 muscl hlld recovery,flux            # RAPTOR
```

参数:`nx ny t_end cfl order solver alpha [problem|regions]`;
输出目录用环境变量 `OUTDIR` 重定向(默认 `results/…`)。

## 3. 正文表格/图 → 数据来源

| 表/图 | 数据文件 | 生成脚本/命令 |
|---|---|---|
| Table 9.9(编译器/浮点控制精度) | `logs/mhd/gpu_backend_20260718/cmp_*.txt` | compare_precision.py 对成对 .dat |
| Table 10.1(512² 计时) | `logs/mhd/perf_baseline/*.csv` | `scripts/run_variants_*.sh`、`run_gpu_baseline_v2.sh` |
| VPREC 扫描(10.2) | `logs/mhd/vprec_scan{1d,2d}/` | `scripts/run_vprec_scan1d.sh` |
| Table 10.6/10.7(RAPTOR) | `logs/mhd/raptor_scan/*.txt` | `run_raptor_2d_scan.sh` + 第 2 节命令 |
| mixed 实测(10.5) | `logs/mhd/perf_baseline/mixed_rr.csv`、`raptor_scan/ot2d512_cmp_mixed_gpu.txt` | mixed 二进制 + compare_precision.py |
| KH 三联图 | `results/mhd/gpu/kh/*.dat`、`logs/mhd/kh/` | 第 2 节 KH 命令 ×2 精度 → `analysis/mhd/plot_kh_triptych.py` |

## 4. 门禁纪律(改代码后必须过)

1. 计时/改动版二进制:OT 192² fp64 **数据行 md5** 必须等于冻结参考
   `d54b0169…`(`grep -v '^#' out.dat | md5sum`),HLLD fallback 计数 13372;
2. RAPTOR:V1(region 化重构逐位同原版)、V3(插桩+空 region 列表逐位同 V1);
3. 计时:共享节点上按 `cpu_ratio>0.95` + 同轮 O2 对照复现冻结基线 ±3% 验收
   (判据先于数据登记进台账)。
