#!/usr/bin/env python3
"""
plot_brio_wu.py — plot the 1D ideal-MHD Brio-Wu shock tube result.

This is a validation/inspection plot for the new CPU FP64 MHD solver.  It reads
the dat file written by src/mhd1d_cpu.cpp:

  x rho vx vy vz p Bx By Bz E

Usage:
  python3 analysis/mhd/plot_brio_wu.py INPUT.dat OUTPUT.png [--style points|line|both|step]

The figure follows the common Brio-Wu reference layout:
  rho on the top row, then vx/vy, then By/p.

Example:
  python3 analysis/mhd/plot_brio_wu.py \
    /private/tmp/mhd_brio_wu_check/brio_wu_hll_fp64_N400.dat \
    /private/tmp/mhd_brio_wu_check/brio_wu_hll_fp64_N400.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def marker_stride(npoints: int) -> int:
    """Keep point markers readable on high-resolution runs.

    The numerical solution is cell-centred.  Markers make that discreteness
    visible, but plotting every marker for N=10000 would hide the profile.
    数值解是单元中心值；点可以显示离散性，但高分辨率时需要抽稀。
    """
    if npoints <= 900:
        return 1
    return max(1, npoints // 450)


def main() -> int:
    if len(sys.argv) not in (3, 5):
        print("Usage: python3 analysis/mhd/plot_brio_wu.py INPUT.dat OUTPUT.png [--style points|line|both|step]")
        return 1

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    style = "points"
    if len(sys.argv) == 5:
        if sys.argv[3] != "--style" or sys.argv[4] not in ("points", "line", "both", "step"):
            print("Optional argument must be: --style points|line|both|step")
            return 1
        style = sys.argv[4]

    if not os.path.exists(in_path):
        print(f"Missing input file: {in_path}")
        return 1

    data = np.loadtxt(in_path)
    if data.ndim != 2 or data.shape[1] < 10:
        print("Expected at least 10 columns: x rho vx vy vz p Bx By Bz E")
        return 1

    x = data[:, 0]
    rho = data[:, 1]
    vx = data[:, 2]
    vy = data[:, 3]
    p = data[:, 5]
    bx = data[:, 6]
    by = data[:, 7]

    print(f"Loaded: {in_path}")
    print(f"Rows: {len(x)}")
    print(f"rho range: [{rho.min():.6e}, {rho.max():.6e}]")
    print(f"vx range:  [{vx.min():.6e}, {vx.max():.6e}]")
    print(f"vy range:  [{vy.min():.6e}, {vy.max():.6e}]")
    print(f"p range:   [{p.min():.6e}, {p.max():.6e}]")
    print(f"Bx range:  [{bx.min():.6e}, {bx.max():.6e}]")
    print(f"By range:  [{by.min():.6e}, {by.max():.6e}]")
    print(f"Plot style: {style}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig = plt.figure(figsize=(8.0, 9.2))
    grid = fig.add_gridspec(3, 4, hspace=0.34, wspace=0.90)
    axes = [
        fig.add_subplot(grid[0, 1:3]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[2, 0:2]),
        fig.add_subplot(grid[2, 2:4]),
    ]
    panels = [
        (rho, r"$\rho$"),
        (vx, r"$v_x$"),
        (vy, r"$v_y$"),
        (by, r"$B_y$"),
        (p, r"$p$"),
    ]
    stride = marker_stride(len(x))

    for ax, (y, ylabel) in zip(axes, panels):
        if style == "step":
            ax.step(x, y, where="mid", color="black", lw=1.0)
        elif style == "line":
            ax.plot(x, y, color="black", lw=1.1)
        elif style == "points":
            ax.plot(x, y, linestyle="None", marker=".", color="black", ms=2.4, markevery=stride)
        else:
            ax.plot(x, y, color="black", lw=0.75, alpha=0.75)
            ax.plot(x, y, linestyle="None", marker=".", color="black", ms=2.0, markevery=stride)
        ax.set_ylabel(ylabel, rotation=0, labelpad=12)
        ax.tick_params(direction="in", top=True, right=True)

    axes[0].set_xlabel("x")
    for ax in axes[1:3]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    for ax in axes[3:]:
        ax.set_xlabel("x")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
