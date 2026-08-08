#!/usr/bin/env python3
"""
plot_sig_digits.py  —  可视化 MCA 有效十进制位数

每个测试两张图，共 10 张:
  solution_test{N}.pdf    : 物理解 (rho, u, p, e) 来自 ieee/ 参考
  sig_digits_test{N}.pdf  : 有效十进制位数，HLLC (实线) vs HLL (虚线)

用法:
  python3 analysis/euler/plot_sig_digits.py
输出:
  analysis/euler/figures/solution_test{N}.pdf
  analysis/euler/figures/sig_digits_test{N}.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # 无显示器环境
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

LOG10_2  = np.log10(2)
GAMMA    = 1.4

# ── 精度参数 ──────────────────────────────────────────────────
PREC = "fp64"
for i, arg in enumerate(sys.argv):
    if arg == "--prec" and i + 1 < len(sys.argv):
        PREC = sys.argv[i + 1]

if PREC == "fp64":
    SD_MAX = 53.0 * LOG10_2    # ≈ 15.95
    PREC_SUFFIX = "fp64_200"
elif PREC == "fp32":
    SD_MAX = 24.0 * LOG10_2    # ≈ 7.22
    PREC_SUFFIX = "fp32_200"
else:
    print(f"Unknown precision: {PREC}"); sys.exit(1)

IEEE_DIR = f"results/euler/ieee/{PREC}"
ANA_DIR  = f"analysis/euler/{PREC}"
FIG_DIR  = os.path.join(ANA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TESTS = [
    ("test1_sod",         1, "Sod Shock Tube"),
    ("test2_123",         2, "123 Problem (Two Rarefactions)"),
    ("test3_left_blast",  3, "Left Blast Wave ($p_L/p_R = 10^5$)"),
    ("test4_right_blast", 4, "Right Blast Wave ($p_R/p_L = 10^4$)"),
    ("test5_two_shocks",  5, "Two Strong Shocks"),
]
VARS      = ["rho", "u", "p", "e"]
VAR_LABEL = [r"$\rho$", r"$u$", r"$p$", r"$e$"]
VAR_TITLE = ["Density", "Velocity", "Pressure", "Internal energy"]
SOLVER_STYLE = {
    "hllc": dict(color="#1f77b4", ls="-",  lw=1.5, label="HLLC"),
    "hll" : dict(color="#ff7f0e", ls="--", lw=1.5, label="HLL"),
}

for dirname, tid, title in TESTS:
    # ── 读取数据 ──────────────────────────────────────────────
    ieee_hllc = np.loadtxt(f"{IEEE_DIR}/toro{tid}_hllc_{PREC_SUFFIX}.dat")
    x = ieee_hllc[:, 0]

    sd = {}
    for solver in ["hllc", "hll"]:
        f = f"{ANA_DIR}/sig_digits_{dirname}_{solver}.dat"
        if not os.path.exists(f):
            print(f"  Missing: {f}  (run mca_analysis.py first)")
            continue
        sd[solver] = np.loadtxt(f)   # cols: x sd_rho sd_u sd_p sd_e

    if len(sd) == 0:
        continue

    # ── 物理解：2行 × 2列 ────────────────────────────────────
    rho = ieee_hllc[:, 1]
    u = ieee_hllc[:, 2]
    p = ieee_hllc[:, 3]
    e = p / ((GAMMA - 1.0) * rho)
    ref_vals = [rho, u, p, e]

    fig_sol, axes_sol = plt.subplots(2, 2, figsize=(9.5, 7.0),
                                     gridspec_kw={"hspace": 0.38, "wspace": 0.38})
    for ax, vals, label in zip(axes_sol.ravel(), ref_vals, VAR_TITLE):
        ax.plot(x, vals, color="black", marker=".", ls="None", ms=2.2)
        ax.set_xlabel("Position", fontsize=15)
        ax.set_ylabel(label, fontsize=15)
        ax.tick_params(direction="in", top=True, right=True)

    outfile_sol = os.path.join(FIG_DIR, f"solution_test{tid}.pdf")
    fig_sol.savefig(outfile_sol, bbox_inches="tight", dpi=150)
    plt.close(fig_sol)
    print(f"Saved: {outfile_sol}")

    # ── 有效十进制位数：2行 × 2列 ────────────────────────────
    fig_sd, axes_sd = plt.subplots(2, 2, figsize=(9.5, 7.0),
                                   gridspec_kw={"hspace": 0.38, "wspace": 0.38})
    for ax, vname, vl, vtitle in zip(axes_sd.ravel(), VARS, VAR_LABEL, VAR_TITLE):
        col = VARS.index(vname) + 1

        for solver, sdata in sd.items():
            sd_vals = sdata[:, col]
            ax.plot(x, sd_vals, **SOLVER_STYLE[solver])

        # fp64 理论上限
        ax.axhline(SD_MAX, color="gray", ls=":", lw=1.0, label=f"{PREC} max ({SD_MAX:.1f})")
        # 诊断线：相对 MCA 标准差超过 1e-3
        ax.axhline(3, color="red", ls=":", lw=0.8, alpha=0.6,
                   label=r"$s_d=3$ ($\sigma/|\mu|=10^{-3}$)")

        ax.set_xlabel("Position", fontsize=13)
        ax.set_ylabel("Significant decimal digits", fontsize=11)
        ax.set_title(vtitle, fontsize=12)
        ymin = min(np.nanmin(sdata[:, col]) for sdata in sd.values())
        ax.set_ylim(min(-1.0, ymin - 0.5), SD_MAX + 0.7)
        ax.set_yticks([0, 3, 6, 9, 12, 15])
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in", top=True, right=True)
        if vname == "rho":
            ax.legend(fontsize=8, loc="lower left")

    outfile = os.path.join(FIG_DIR, f"sig_digits_test{tid}.pdf")
    fig_sd.savefig(outfile, bbox_inches="tight", dpi=150)
    plt.close(fig_sd)
    print(f"Saved: {outfile}")

print("Done.")
