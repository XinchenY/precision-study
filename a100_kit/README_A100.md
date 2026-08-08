# A100 计时包(Ch10 §10.1 用)

自带一切,拷到 A100 机器解包即用。目标:6 配置({fp64,fp32} × {256²,384²,512²})
的 GPU 计时(median±IQR)+ 与 A30 的一致性交叉校验。

## 0. 包内容

```
include/                      共享头文件(原样)
src/mhd2d_glm_gpu.cu          2D OT 求解器(CUDA)
compare_precision.py          误差对比工具(备用)
a30_reference_checksums.txt   A30 参考 md5(交叉校验用)
a100_bench.sh                 一键脚本(编译+校验+计时)
```

## 1. 前置检查(记录进论文的机器信息)

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version        # 记录 CUDA 版本!
```

**CUDA 版本注意**:A30 侧用 CUDA 12.9 编译。若 A100 机器主版本不同,
生成的机器码舍入路径可能不同,逐位校验(第 3 步)允许失败,
届时退用 compare_precision.py 检查 relL2 ~1e-15 量级即可(fp64)。

## 2. 编译(A100 同为 sm_80,命令与 A30 完全一致)

```bash
nvcc -O2 -std=c++14 -arch=sm_80 -DMHD_HLLD_TOLERANCE_VALUE=1e-4 \
     -I include src/mhd2d_glm_gpu.cu -o mhd2d_gpu_fp64
nvcc -O2 -std=c++14 -arch=sm_80 -DMHD_HLLD_TOLERANCE_VALUE=1e-4 -DUSE_FLOAT \
     -I include src/mhd2d_glm_gpu.cu -o mhd2d_gpu_fp32
```

## 3. 一致性校验(论文 6c 声明的依据)

```bash
OUTDIR=out ./mhd2d_gpu_fp64 192 192 0.5 0.25 muscl hlld 0.18
grep -v '^#' out/orszag_tang_glm_muscl_hlld_fp64_N192x192.dat | md5sum
# 与 a30_reference_checksums.txt 中 fp64 行比对; fp32 同理
# md5 相同 → 论文写 "A100 与 A30 输出逐位一致, 继承 Ch9 误差刻画"
# 不同(且 CUDA 版本不同)→ 用 compare_precision.py 报 relL2, 如实写
# 回退计数也应一致: fp64=13372, fp32=13355(stdout 里有)
```

## 4. 计时(或直接 `./a100_bench.sh` 全自动)

每配置 2 次 warm-up 后 ≥10 次计时,记录求解器自报的
"Time loop wall time"(时间环口径,与 A30/CPU 一致)。机器空闲时跑
(A100 若独占,天然干净;若共享,记录 uptime 负载)。

```bash
for p in fp64 fp32; do for N in 256 384 512; do
  for w in 1 2; do OUTDIR=/tmp/a100run ./mhd2d_gpu_$p $N $N 0.5 0.25 muscl hlld 0.18 >/dev/null; done
  for r in $(seq 1 10); do
    OUTDIR=/tmp/a100run ./mhd2d_gpu_$p $N $N 0.5 0.25 muscl hlld 0.18 \
      | grep -oP 'Time loop wall time = \K[0-9.]+' \
      | xargs -I{} echo "$(date -Is),a100,$p,$N,$r,{}" >> a100_timing.csv
    rm -rf /tmp/a100run
  done
done; done
```

## 5. 带回来的东西

- `a100_timing.csv`(60 行)
- 第 1 步的机器信息输出
- 第 3 步的两个 md5 + 回退计数
- 任一 512² run 的完整 stdout(留档)

把这四样发回来,10.1 的 A100 列即可并入,6c 声明照校验结果写。
