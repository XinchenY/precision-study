#!/usr/bin/env python3
"""
Plot density evolution panels for Orszag-Tang outputs.

Usage:
  python3 analysis/mhd/plot_orszag_tang_evolution.py OUTPUT.png INPUT1.dat INPUT2.dat ...
"""

import os
import re
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_time(path: str) -> Optional[float]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = re.search(r"\bt\s+([0-9.eE+-]+)", line)
            if match:
                return float(match.group(1))
    return None


def load_density(path: str):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 12:
        raise ValueError(f"Expected Orszag-Tang columns in {path}")

    x = data[:, 0]
    y = data[:, 1]
    xs = np.unique(x)
    ys = np.unique(y)
    nx = len(xs)
    ny = len(ys)
    if nx * ny != data.shape[0]:
        raise ValueError(f"Could not infer structured grid in {path}")

    rho = data[:, 2].reshape(ny, nx)
    return xs, ys, rho, read_time(path)


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: python3 analysis/mhd/plot_orszag_tang_evolution.py OUTPUT.png INPUT1.dat INPUT2.dat ...")
        return 1

    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    loaded = [load_density(path) for path in in_paths]
    rho_min = min(item[2].min() for item in loaded)
    rho_max = max(item[2].max() for item in loaded)

    n = len(loaded)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
    })

    last_image = None
    for idx, (ax, (xs, ys, rho, time_value)) in enumerate(zip(axes, loaded)):
        extent = [xs.min(), xs.max(), ys.min(), ys.max()]
        last_image = ax.imshow(rho, origin="lower", extent=extent, aspect="equal",
                               vmin=rho_min, vmax=rho_max)
        label = f"t = {time_value:.2f}" if time_value is not None else os.path.basename(in_paths[idx])
        ax.set_title(f"({chr(ord('a') + idx)}) {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in axes[n:]:
        ax.axis("off")

    if last_image is not None:
        fig.colorbar(last_image, ax=axes[:n], fraction=0.046, pad=0.04, label=r"$\rho$")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
