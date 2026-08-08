#!/usr/bin/env python3
"""
plot_fp32_vs_fp64.py  —  fp32 vs fp64 IEEE 基线物理解对比

每个测试一张图 (2×2: rho, u, p, e), 共 5 张:
  analysis/euler/figures/fp32_vs_fp64_test{N}.pdf

HLLC solver, fp64 (蓝实线) vs fp32 (红虚线)

用法:
  python3 analysis/euler/plot_fp32_vs_fp64.py [--solver hllc|hll]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams.update({"font.family": "serif", "font.size": 12})

SOLVER = "hllc"
for i, a in enumerate(sys.argv):
    if a == "--solver" and i + 1 < len(sys.argv):
        SOLVER = sys.argv[i + 1]

GAMMA   = 1.4
FIG_DIR = "analysis/euler/figures"
os.makedirs(FIG_DIR, exist_ok=True)

TESTS = [
    ("test1_sod",         1, "Sod Shock Tube"),
    ("test2_123",         2, "123 Problem (Two Rarefactions)"),
    ("test3_left_blast",  3, "Left Blast Wave ($p_L/p_R = 10^5$)"),
    ("test4_right_blast", 4, "Right Blast Wave ($p_R/p_L = 10^4$)"),
    ("test5_two_shocks",  5, "Two Strong Shocks"),
]
VAR_TITLE = ["Density", "Velocity", "Pressure", "Internal energy"]
VAR_LABEL = [r"$\rho$", r"$u$", r"$p$", r"$e$"]

STYLE64 = dict(color="#1f77b4", ls="-",  lw=1.5, label="fp64")
STYLE32 = dict(color="#d62728", ls="--", lw=1.5, label="fp32")

for dirname, tid, title in TESTS:
    f64_path = f"results/euler/ieee/fp64/toro{tid}_{SOLVER}_fp64_200.dat"
    f32_path = f"results/euler/ieee/fp32/toro{tid}_{SOLVER}_fp32_200.dat"

    if not os.path.exists(f64_path) or not os.path.exists(f32_path):
        print(f"  Missing data for test {tid}"); continue

    A64 = np.loadtxt(f64_path)
    A32 = np.loadtxt(f32_path)

    x64  = A64[:, 0]
    x32  = A32[:, 0]
    rho64, u64, p64 = A64[:, 1], A64[:, 2], A64[:, 3]
    rho32, u32, p32 = A32[:, 1], A32[:, 2], A32[:, 3]
    e64 = p64 / ((GAMMA - 1.0) * rho64)
    e32 = p32 / ((GAMMA - 1.0) * rho32)

    vals64 = [rho64, u64, p64, e64]
    vals32 = [rho32, u32, p32, e32]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.40})
    fig.suptitle(f"{title}  ({SOLVER.upper()})", fontsize=13, y=1.01)

    for ax, v64, v32, vtitle in zip(axes.ravel(), vals64, vals32, VAR_TITLE):
        ax.plot(x64, v64, **STYLE64)
        ax.plot(x32, v32, **STYLE32)
        ax.set_xlabel("Position", fontsize=13)
        ax.set_ylabel(vtitle, fontsize=13)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10)

    outfile = os.path.join(FIG_DIR, f"fp32_vs_fp64_test{tid}.pdf")
    fig.savefig(outfile, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {outfile}")

print("Done.")
