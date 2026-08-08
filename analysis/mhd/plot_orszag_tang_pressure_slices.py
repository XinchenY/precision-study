#!/usr/bin/env python3
"""
Plot Orszag-Tang pressure slices following Stone et al. Fig. 23.

Usage:
  python3 analysis/mhd/plot_orszag_tang_pressure_slices.py INPUT.dat OUTPUT.png
  python3 analysis/mhd/plot_orszag_tang_pressure_slices.py LOW.dat OUTPUT.png HIGH.dat

Input columns:
  x y rho vx vy vz p Bx By Bz E divB [psi]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SLICE_YS = (0.3125, 0.427)


def read_grid(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Expected Orszag-Tang columns in {path}")
    x = data[:, 0]
    y = data[:, 1]
    xs = np.unique(x)
    ys = np.unique(y)
    nx = len(xs)
    ny = len(ys)
    if nx * ny != len(data):
        raise ValueError(f"Could not infer structured grid from {path}")
    p = data[:, 6].reshape(ny, nx)
    return xs, ys, p


def nearest_slice(xs, ys, pressure, y_target):
    j = int(np.argmin(np.abs(ys - y_target)))
    return xs, pressure[j, :], ys[j]


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 analysis/mhd/plot_orszag_tang_pressure_slices.py LOW.dat OUTPUT.png [HIGH.dat]")
        return 1

    low_path = sys.argv[1]
    out_path = sys.argv[2]
    high_path = sys.argv[3] if len(sys.argv) == 4 else None

    xs_low, ys_low, p_low = read_grid(low_path)
    high = read_grid(high_path) if high_path else None

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    labels = ("top", "bottom")

    for ax, y_target, label in zip(axes, SLICE_YS, labels):
        x_low, p_slice_low, y_used_low = nearest_slice(xs_low, ys_low, p_low, y_target)

        if high is not None:
            xs_high, ys_high, p_high = high
            x_high, p_slice_high, y_used_high = nearest_slice(xs_high, ys_high, p_high, y_target)
            ax.plot(x_high, p_slice_high, color="black", linewidth=0.9,
                    label=f"high grid, y={y_used_high:.4f}")
            ax.plot(x_low, p_slice_low, linestyle="none", marker="s",
                    markersize=3.0, markerfacecolor="none", markeredgewidth=0.6,
                    color="black", label=f"low grid, y={y_used_low:.4f}")
        else:
            ax.plot(x_low, p_slice_low, color="black", linewidth=0.8,
                    marker="s", markersize=2.5, markerfacecolor="none",
                    markeredgewidth=0.5)

        ax.set_ylabel(r"$p$")
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_on()
        ax.set_title(f"{label}: y = {y_target}", fontsize=10)

    axes[-1].set_xlabel(r"$x$")
    if high is not None:
        axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
