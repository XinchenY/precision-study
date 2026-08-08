#!/usr/bin/env bash
# A100 一键: 编译 + 一致性校验 + 6 配置计时 (详见 README_A100.md)
set -u; cd "$(dirname "$0")"
echo "== machine info ==" | tee a100_info.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a a100_info.txt
nvcc --version | tail -1 | tee -a a100_info.txt
nvcc -O2 -std=c++14 -arch=sm_80 -DMHD_HLLD_TOLERANCE_VALUE=1e-4 -I include src/mhd2d_glm_gpu.cu -o mhd2d_gpu_fp64 || exit 1
nvcc -O2 -std=c++14 -arch=sm_80 -DMHD_HLLD_TOLERANCE_VALUE=1e-4 -DUSE_FLOAT -I include src/mhd2d_glm_gpu.cu -o mhd2d_gpu_fp32 || exit 1
echo "== consistency check ==" | tee -a a100_info.txt
for p in fp64 fp32; do
  OUTDIR=chk ./mhd2d_gpu_$p 192 192 0.5 0.25 muscl hlld 0.18 > chk_$p.log 2>&1
  M=$(grep -v '^#' chk/orszag_tang_glm_muscl_hlld_${p}_N192x192.dat | md5sum | cut -d' ' -f1)
  FB=$(grep -oP 'fallbacks = \K[0-9]+' chk_$p.log)
  echo "$p md5=$M fallbacks=$FB" | tee -a a100_info.txt
done
echo "(对照 a30_reference_checksums.txt)" | tee -a a100_info.txt
echo "iso,backend,prec,N,rep,loop_wall_s" > a100_timing.csv
for p in fp64 fp32; do for N in 256 384 512; do
  for w in 1 2; do OUTDIR=/tmp/a100run ./mhd2d_gpu_$p $N $N 0.5 0.25 muscl hlld 0.18 > /dev/null 2>&1; rm -rf /tmp/a100run; done
  for r in $(seq 1 10); do
    W=$(OUTDIR=/tmp/a100run ./mhd2d_gpu_$p $N $N 0.5 0.25 muscl hlld 0.18 2>/dev/null | grep -oP 'Time loop wall time = \K[0-9.]+')
    echo "$(date -Is),a100,$p,$N,$r,$W" >> a100_timing.csv
    rm -rf /tmp/a100run
  done
  echo "$p N=$N done"
done; done
echo "ALL DONE -> a100_timing.csv, a100_info.txt"
