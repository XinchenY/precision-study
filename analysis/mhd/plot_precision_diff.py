#!/usr/bin/env python3
"""
plot_precision_diff.py — fp32 vs fp64 逐点误差场 (2D OT-GLM)
                         spatial map of |fp64 - fp32| for the OT-GLM solution.

把 fp64 密度结构(左上, 显示电流片位置)和 |Δρ|/|Δp|/|ΔBy| 的误差场
(对数色标)放在一起 —— 一眼看出 fp32 误差是不是集中在电流片上。

Usage:
  python3 analysis/mhd/plot_precision_diff.py  FP64.dat  FP32.dat  OUT.png  [gray]
  第 4 个参数给 "gray"(或 grey/bw)则输出黑白版,匹配论文等高线图的单色风格。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def load(path):
    d = np.loadtxt(path, comments="#")
    xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
    if len(xs) * len(ys) != d.shape[0]:
        raise ValueError(f"{path}: not a structured grid")
    return xs, ys, d


def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: plot_precision_diff.py FP64.dat FP32.dat OUT.png [gray]")
        return 1

    gray = len(sys.argv) == 5 and sys.argv[4].lower() in ("gray", "grey", "bw")
    cmap_err = "Greys" if gray else "inferno"   # 误差: 高=黑 / high=dark
    cmap_ref = "gray_r" if gray else "viridis"  # 参考结构

    xs, ys, a = load(sys.argv[1])   # fp64
    _,  _,  b = load(sys.argv[2])   # fp32
    if a.shape != b.shape:
        print(f"shape mismatch {a.shape} vs {b.shape}")
        return 1
    nx, ny = len(xs), len(ys)
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]

    def f(dat, col):
        return dat[:, col].reshape(ny, nx)

    panels = [
        ("fp64 density (structure)",   f(a, 2),                    False),
        (r"$|\Delta\rho|$  fp64-fp32", np.abs(f(a, 2) - f(b, 2)),  True),
        (r"$|\Delta p|$  fp64-fp32",   np.abs(f(a, 6) - f(b, 6)),  True),
        (r"$|\Delta B_y|$  fp64-fp32", np.abs(f(a, 8) - f(b, 8)),  True),
    ]

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6), constrained_layout=True)
    for ax, (title, field, islog) in zip(axes.flat, panels):
        if islog:
            vmax = float(field.max())
            vmin = vmax * 1e-4 if vmax > 0 else 1e-12
            im = ax.imshow(np.clip(field, vmin, vmax), origin="lower", extent=ext,
                           norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap_err,
                           aspect="equal")
        else:
            im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap_ref,
                           aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    os.makedirs(os.path.dirname(sys.argv[3]) or ".", exist_ok=True)
    fig.savefig(sys.argv[3], dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"grid {nx}x{ny}")
    print(f"max|d_rho| = {np.abs(f(a,2)-f(b,2)).max():.3e}")
    print(f"max|d_p|   = {np.abs(f(a,6)-f(b,6)).max():.3e}")
    print(f"max|d_By|  = {np.abs(f(a,8)-f(b,8)).max():.3e}")
    print(f"Saved: {sys.argv[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
