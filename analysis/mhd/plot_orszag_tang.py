#!/usr/bin/env python3
"""
Plot a 2D Orszag-Tang output written by src/mhd2d_cpu.cpp.

Usage:
  python3 analysis/mhd/plot_orszag_tang.py INPUT.dat OUTPUT.png

Input columns:
  x y rho vx vy vz p Bx By Bz E divB
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 analysis/mhd/plot_orszag_tang.py INPUT.dat OUTPUT.png")
        return 1

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    if not os.path.exists(in_path):
        print(f"Missing input file: {in_path}")
        return 1

    data = np.loadtxt(in_path, comments="#")
    if data.ndim != 2 or data.shape[1] < 12:
        print("Expected columns: x y rho vx vy vz p Bx By Bz E divB")
        return 1

    x = data[:, 0]
    y = data[:, 1]
    xs = np.unique(x)
    ys = np.unique(y)
    nx = len(xs)
    ny = len(ys)
    if nx * ny != data.shape[0]:
        print("Could not infer a structured grid from x/y coordinates.")
        return 1

    rho = data[:, 2].reshape(ny, nx)
    p = data[:, 6].reshape(ny, nx)
    bx = data[:, 7].reshape(ny, nx)
    by = data[:, 8].reshape(ny, nx)
    bz = data[:, 9].reshape(ny, nx)
    divb = data[:, 11].reshape(ny, nx)
    magnetic_pressure = 0.5 * (bx * bx + by * by + bz * bz)

    print(f"Loaded: {in_path}")
    print(f"Grid: {nx} x {ny}")
    print(f"rho range: [{rho.min():.6e}, {rho.max():.6e}]")
    print(f"p range:   [{p.min():.6e}, {p.max():.6e}]")
    print(f"|divB| max: {np.max(np.abs(divb)):.6e}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": False,
    })

    fields = [
        (rho, r"$\rho$"),
        (p, r"$p$"),
        (magnetic_pressure, r"$|B|^2/2$"),
        (np.abs(divb), r"$|\nabla\cdot B|$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    for ax, (field, label) in zip(axes.flat, fields):
        image = ax.imshow(field, origin="lower", extent=extent, aspect="equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
