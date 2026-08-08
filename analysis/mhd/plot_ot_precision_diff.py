#!/usr/bin/env python3
"""
plot_ot_precision_diff.py — OT 的 FP32-FP64 差值图 (论文 Fig 5.10 同款视觉语言)
  2x2: 左上 = FP64 密度 30 条等高线参考 (Stone 风格)
       其余 = |Δρ|, |Δp|, |ΔB_y| 的 viridis 对数热图

Usage:
  python3 analysis/mhd/plot_ot_precision_diff.py FP64.dat FP32.dat OUT.png
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

if len(sys.argv) != 4:
    print("Usage: plot_ot_precision_diff.py FP64.dat FP32.dat OUT.png"); sys.exit(1)

a = np.loadtxt(sys.argv[1], comments="#"); b = np.loadtxt(sys.argv[2], comments="#")
xs, ys = np.unique(a[:,0]), np.unique(a[:,1]); nx, ny = len(xs), len(ys)
ext = [xs.min(), xs.max(), ys.min(), ys.max()]
f = lambda d,c: d[:,c].reshape(ny,nx)
rho64 = f(a,2)
X, Y = np.meshgrid(xs, ys)

plt.rcParams.update({"font.family":"serif","font.size":10, "axes.linewidth":0.8})
fig, axes = plt.subplots(2,2, figsize=(9.6,8.8), constrained_layout=True)

ax = axes[0,0]
ax.contour(X, Y, rho64, levels=np.linspace(rho64.min(), rho64.max(), 30),
           colors="0.10", linewidths=0.45)
ax.set_title(r"$\rho$ (FP64)")
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

panels = [(axes[0,1], r"$|\rho^{\mathrm{FP64}}-\rho^{\mathrm{FP32}}|$", 2),
          (axes[1,0], r"$|p^{\mathrm{FP64}}-p^{\mathrm{FP32}}|$",       6),
          (axes[1,1], r"$|B_y^{\mathrm{FP64}}-B_y^{\mathrm{FP32}}|$",   8)]
for ax, cblab, col in panels:
    d = np.abs(f(a,col)-f(b,col))
    vmax = d.max(); vmin = vmax*1e-4
    im = ax.imshow(np.clip(d,vmin,vmax), origin="lower", extent=ext,
                   norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", aspect="equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cblab, fontsize=9)

fig.savefig(sys.argv[3], dpi=160)
print(f"Grid {nx}x{ny}. Saved: {sys.argv[3]}")
