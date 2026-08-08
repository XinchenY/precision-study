#!/usr/bin/env python3
"""
Plot Orszag-Tang contours in the same variable layout as Stone et al. Fig. 22.

Usage:
  python3 analysis/mhd/plot_orszag_tang_stone_contours.py INPUT.dat OUTPUT.png

Input columns:
  x y rho vx vy vz p Bx By Bz E divB [psi]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_grid(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 12:
        raise ValueError("Expected columns: x y rho vx vy vz p Bx By Bz E divB [psi]")

    x = data[:, 0]
    y = data[:, 1]
    xs = np.unique(x)
    ys = np.unique(y)
    nx = len(xs)
    ny = len(ys)
    if nx * ny != data.shape[0]:
        raise ValueError("Could not infer a structured grid from x/y coordinates.")

    def field(col):
        return data[:, col].reshape(ny, nx)

    return xs, ys, {
        "rho": field(2),
        "vx": field(3),
        "vy": field(4),
        "vz": field(5),
        "p": field(6),
        "bx": field(7),
        "by": field(8),
        "bz": field(9),
        "divb": field(11),
    }


def levels_for(field, count=30):
    vmin = float(np.min(field))
    vmax = float(np.max(field))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.linspace(vmin - 1.0, vmax + 1.0, count)
    return np.linspace(vmin, vmax, count)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analysis/mhd/plot_orszag_tang_stone_contours.py INPUT.dat OUTPUT.png")
        return 1

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    if not os.path.exists(in_path):
        print(f"Missing input file: {in_path}")
        return 1

    xs, ys, f = load_grid(in_path)
    magnetic_pressure = 0.5 * (f["bx"] ** 2 + f["by"] ** 2 + f["bz"] ** 2)
    specific_kinetic_energy = 0.5 * (f["vx"] ** 2 + f["vy"] ** 2 + f["vz"] ** 2)

    fields = [
        ("DENSITY", f["rho"]),
        ("PRESSURE", f["p"]),
        ("MAGNETIC PRESSURE", magnetic_pressure),
        ("SPECIFIC KINETIC ENERGY", specific_kinetic_energy),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0), constrained_layout=True)
    xgrid, ygrid = np.meshgrid(xs, ys)

    for ax, (title, field) in zip(axes.flat, fields):
        ax.contour(xgrid, ygrid, field, levels=levels_for(field), colors="0.10", linewidths=0.45)
        ax.set_title(title, color="0.45", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_xticks([])
        ax.set_yticks([])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded: {in_path}")
    print(f"Grid: {len(xs)} x {len(ys)}")
    print(f"min rho = {np.min(f['rho']):.8e}, min p = {np.min(f['p']):.8e}")
    print(f"mean |divB| = {np.mean(np.abs(f['divb'])):.8e}")
    print(f"max |divB| = {np.max(np.abs(f['divb'])):.8e}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
