#!/usr/bin/env python3
"""
compare_fp32_fp64.py  —  比较 fp32 与 fp64 IEEE 基线解的 L1/L2/L∞ 误差

fp32 和 fp64 在同一个 N=200 均匀网格上，但 x 坐标的浮点表示不同，
因此按列索引对齐（不做 x 坐标一致性检查）。

用法:
  python3 analysis/euler/compare_fp32_fp64.py [--solver hllc|hll|both]

输出:
  终端打印 LaTeX-ready 表格 + 保存到 analysis/euler/fp32_vs_fp64_norms.txt
"""

import numpy as np
import os
import sys

SOLVER_FLAG = "both"
for i, a in enumerate(sys.argv):
    if a == "--solver" and i + 1 < len(sys.argv):
        SOLVER_FLAG = sys.argv[i + 1]

TESTS = [
    ("test1_sod",         1, "Sod"),
    ("test2_123",         2, "123"),
    ("test3_left_blast",  3, "LBlast"),
    ("test4_right_blast", 4, "RBlast"),
    ("test5_two_shocks",  5, "TwoShocks"),
]
SOLVERS = ["hllc", "hll"] if SOLVER_FLAG == "both" else [SOLVER_FLAG]
VARS    = ["rho", "u", "p"]
COL     = {"rho": 1, "u": 2, "p": 3}

def norms(a, b):
    d  = np.abs(a - b)
    L1   = d.mean()
    L2   = np.sqrt((d**2).mean())
    Linf = d.max()
    return L1, L2, Linf

lines = []
lines.append("fp32 vs fp64 IEEE baseline  (L1=(1/N)Σ|e|, L2=RMS, L∞=max|e|)")
lines.append(f"{'Test':<12} {'Solver':>6}  {'var':>3}  {'L1':>12}  {'L2':>12}  {'L∞':>12}")
lines.append("-" * 65)

for dirname, tid, label in TESTS:
    for solver in SOLVERS:
        f64 = f"results/euler/ieee/fp64/toro{tid}_{solver}_fp64_200.dat"
        f32 = f"results/euler/ieee/fp32/toro{tid}_{solver}_fp32_200.dat"
        if not os.path.exists(f64) or not os.path.exists(f32):
            print(f"  Missing: {f64} or {f32}")
            continue
        A = np.loadtxt(f64)
        B = np.loadtxt(f32)
        first = True
        for v in VARS:
            c = COL[v]
            l1, l2, linf = norms(A[:, c], B[:, c])
            tag = f"{label}" if first else ""
            stag = f"{solver}" if first else ""
            lines.append(f"  {tag:<12} {stag:>6}  {v:>3}  {l1:12.4e}  {l2:12.4e}  {linf:12.4e}")
            first = False

output = "\n".join(lines)
print(output)

os.makedirs("analysis", exist_ok=True)
with open("analysis/euler/fp32_vs_fp64_norms.txt", "w") as f:
    f.write(output + "\n")
print("\nSaved: analysis/euler/fp32_vs_fp64_norms.txt")
