#!/usr/bin/env python3
"""
plot_ot_512_jz.py — 大网格差异场的空间归属: 电流片还是激波?
  2x2:
    左上 = |J_z| (FP64)          — 电流片位置 (J_z = dBy/dx - dBx/dy)
    左下 = 压缩 -div(v) (FP64)   — 激波位置 (clip >= 0)
    右上 = |ΔB_x| + |J_z| 等值线 — 误差 vs 电流片
    右下 = |Δρ|  + |J_z| 等值线 — 误差 vs 电流片
  差异 panel 用 viridis 对数色标, 下限 = 峰值/1e3 (背景压黑, 同 Fig 9.2 风格)。

Usage:
  python3 analysis/mhd/plot_ot_512_jz.py FP64.dat FP32.dat OUT.png [full]
  默认 1x2 (论文版: 两个误差 panel + J_z 等值线); 加 "full" 出 2x2 分析版。
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

if len(sys.argv) not in (4, 5):
    print("Usage: plot_ot_512_jz.py FP64.dat FP32.dat OUT.png [full]"); sys.exit(1)
FULL = len(sys.argv) == 5 and sys.argv[4] == "full"

a = np.loadtxt(sys.argv[1], comments="#"); b = np.loadtxt(sys.argv[2], comments="#")
xs, ys = np.unique(a[:, 0]), np.unique(a[:, 1]); nx, ny = len(xs), len(ys)
ext = [xs.min(), xs.max(), ys.min(), ys.max()]
dx, dy = xs[1] - xs[0], ys[1] - ys[0]
f = lambda d, c: d[:, c].reshape(ny, nx)   # axis0 = y, axis1 = x

# 周期边界的中心差分 / periodic central differences
ddx = lambda q: (np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)) / (2 * dx)
ddy = lambda q: (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2 * dy)

vx64, vy64 = f(a, 3), f(a, 4)
bx64, by64 = f(a, 7), f(a, 8)
jz = ddx(by64) - ddy(bx64)                 # z 向电流密度 (FP64)
comp = np.clip(-(ddx(vx64) + ddy(vy64)), 0, None)   # 压缩 (激波处 > 0)

d_bx = np.abs(f(a, 7) - f(b, 7))
d_rho = np.abs(f(a, 2) - f(b, 2))

X, Y = np.meshgrid(xs, ys)
jz_abs = np.abs(jz)
jz_levels = [0.25 * jz_abs.max(), 0.5 * jz_abs.max()]   # 电流片等值线

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})

if FULL:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.8), constrained_layout=True)

    ax = axes[0, 0]
    im = ax.imshow(jz_abs, origin="lower", extent=ext, cmap="magma", aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$|J_z|$ (FP64)", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = axes[1, 0]
    im = ax.imshow(comp, origin="lower", extent=ext, cmap="magma", aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"$-\nabla\!\cdot\!\mathbf{v}$ (FP64, clipped $\geq 0$)", fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    err_axes = [axes[0, 1], axes[1, 1]]
else:
    fig, err_axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)

# --- 差异场 + J_z 等值线 ---
panels = [(err_axes[0], d_bx, r"$|B_x^{\mathrm{FP64}}-B_x^{\mathrm{FP32}}|$"),
          (err_axes[1], d_rho, r"$|\rho^{\mathrm{FP64}}-\rho^{\mathrm{FP32}}|$")]
for ax, d, cblab in panels:
    vmax = d.max(); vmin = vmax * 1e-3
    im = ax.imshow(np.clip(d, vmin, vmax), origin="lower", extent=ext,
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", aspect="equal")
    ax.contour(X, Y, jz_abs, levels=jz_levels, colors="white",
               linewidths=0.35, alpha=0.85)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cblab, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")

fig.savefig(sys.argv[3], dpi=160)
print(f"Grid {nx}x{ny}. Saved: {sys.argv[3]}")
