#!/usr/bin/env python3
"""
compare_precision.py — fp32 vs fp64 逐点精度对比 (每个变量的 L1 / L2 / L∞)
                       fp32 vs fp64 pointwise accuracy (L1 / L2 / L-inf per variable)

用法 / Usage:
  python3 analysis/mhd/compare_precision.py  FP64.dat  FP32.dat

自动识别布局 / auto-detects layout by column count:
  10 列 → 1D Brio-Wu:  x  rho vx vy vz p Bx By Bz E
  13 列 → 2D OT-GLM:    x y  rho vx vy vz p Bx By Bz E divB psi

范数定义(逐点差 d = fp64 - fp32, N 个格点, 单位域):
  L1   = mean(|d|)          平均绝对差
  L2   = sqrt(mean(d^2))    均方根(RMS)
  Linf = max(|d|)           最大逐点差
  relL2 = L2 / RMS(fp64)    相对误差(无量纲, 论文常报这个)
"""
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python3 analysis/mhd/compare_precision.py FP64.dat FP32.dat")
    sys.exit(1)

a = np.loadtxt(sys.argv[1])
b = np.loadtxt(sys.argv[2])
if a.shape != b.shape:
    print(f"ERROR: shape mismatch {a.shape} vs {b.shape}")
    sys.exit(1)

ncol = a.shape[1]
if ncol == 10:      # 1D Brio-Wu
    names = ["rho", "vx", "vy", "vz", "p", "Bx", "By", "Bz", "E"]
    cols  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    layout = "1D Brio-Wu"
elif ncol == 13:    # 2D Orszag-Tang GLM
    names = ["rho", "vx", "vy", "vz", "p", "Bx", "By", "Bz", "E"]
    cols  = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    layout = "2D OT-GLM"
else:
    print(f"ERROR: unknown layout, ncol={ncol}")
    sys.exit(1)

N = a.shape[0]
print(f"# 布局/layout = {layout}   点数/N = {N}   列/cols = {ncol}")
print(f"# fp64 = {sys.argv[1].split('/')[-1]}")
print(f"# fp32 = {sys.argv[2].split('/')[-1]}")
print(f"{'var':>4} {'L1':>12} {'L2(RMS)':>12} {'Linf':>12} {'relL2':>10}")
print("-" * 56)
for nm, c in zip(names, cols):
    d = a[:, c] - b[:, c]
    L1   = np.mean(np.abs(d))
    L2   = np.sqrt(np.mean(d**2))
    Linf = np.max(np.abs(d))
    ref  = np.sqrt(np.mean(a[:, c]**2))
    rel  = (L2 / ref) if ref > 0 else 0.0
    print(f"{nm:>4} {L1:12.4e} {L2:12.4e} {Linf:12.4e} {rel:10.2e}")

# GLM 特有: divB 的控制在两种精度下差多少 (只有 2D 有 divB 列)
if ncol == 13:
    d64 = np.abs(a[:, 11]); d32 = np.abs(b[:, 11])
    print("-" * 56)
    print(f"# divB 控制 / divergence control (col 12):")
    print(f"#   fp64:  L1={np.mean(d64):.3e}  Linf={np.max(d64):.3e}")
    print(f"#   fp32:  L1={np.mean(d32):.3e}  Linf={np.max(d32):.3e}")
