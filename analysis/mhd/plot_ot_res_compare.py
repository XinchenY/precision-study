#!/usr/bin/env python3
"""
plot_ot_res_compare.py — 同协议(safeguarded, eps=1e-4)跨分辨率差异场对比
  1-2-2 布局:
    第 1 行 (单图) = rho (FP64, 512^2) 30 条黑色等高线 — 流场参考
    第 2 行 = |Δρ|  @192^2  |  |Δρ|  @512^2
    第 3 行 = |ΔB_x|@192^2  |  |ΔB_x|@512^2
  每个差异 panel: viridis 对数色标 (各自独立, 下限=峰值/1e3),
  白色细线 = 本分辨率 FP64 的 |J_z| 等值线 (25%/50% 峰值)。

Usage:
  python3 analysis/mhd/plot_ot_res_compare.py \
      FP64_192.dat FP32_192.dat FP64_512.dat FP32_512.dat OUT.png
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

if len(sys.argv) != 6:
    print("Usage: plot_ot_res_compare.py FP64_192 FP32_192 FP64_512 FP32_512 OUT.png")
    sys.exit(1)


def load_pair(f64, f32):
    a = np.loadtxt(f64, comments="#"); b = np.loadtxt(f32, comments="#")
    xs, ys = np.unique(a[:, 0]), np.unique(a[:, 1])
    nx, ny = len(xs), len(ys)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    g = lambda d, c: d[:, c].reshape(ny, nx)
    ddx = lambda q: (np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)) / (2 * dx)
    ddy = lambda q: (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2 * dy)
    jz = np.abs(ddx(g(a, 8)) - ddy(g(a, 7)))          # |J_z| (FP64)
    return dict(xs=xs, ys=ys, nx=nx,
                ext=[xs.min(), xs.max(), ys.min(), ys.max()],
                rho64=g(a, 2), jz=jz,
                d_rho=np.abs(g(a, 2) - g(b, 2)),
                d_bx=np.abs(g(a, 7) - g(b, 7)))


lo = load_pair(sys.argv[1], sys.argv[2])   # 192^2
hi = load_pair(sys.argv[3], sys.argv[4])   # 512^2

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig = plt.figure(figsize=(9.6, 13.2), constrained_layout=True)
gs = fig.add_gridspec(3, 2)

# ---- 第 1 行: 参考流场 (rho, FP64, 512^2) ----
ax = fig.add_subplot(gs[0, :])
X, Y = np.meshgrid(hi["xs"], hi["ys"])
ax.contour(X, Y, hi["rho64"],
           levels=np.linspace(hi["rho64"].min(), hi["rho64"].max(), 30),
           colors="0.10", linewidths=0.45)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"$\rho$ (FP64, $512^2$)")

# ---- 第 2/3 行: 差异场 (每行共用一个色标, 跨 5 decade, 颜色可横向比较) ----
rows = [("d_rho", r"$|\Delta\rho|$"), ("d_bx", r"$|\Delta B_x|$")]
cols = [(lo, r"$192^2$"), (hi, r"$512^2$")]
for r, (key, lab) in enumerate(rows):
    vmax = max(lo[key].max(), hi[key].max())
    vmin = vmax * 1e-5
    norm = LogNorm(vmin=vmin, vmax=vmax)
    row_axes = []
    for c, (dat, title) in enumerate(cols):
        ax = fig.add_subplot(gs[r + 1, c])
        im = ax.imshow(np.clip(dat[key], vmin, vmax), origin="lower",
                       extent=dat["ext"], norm=norm,
                       cmap="viridis", aspect="equal")
        Xc, Yc = np.meshgrid(dat["xs"], dat["ys"])
        ax.contour(Xc, Yc, dat["jz"],
                   levels=[0.25 * dat["jz"].max(), 0.5 * dat["jz"].max()],
                   colors="white", linewidths=0.35, alpha=0.85)
        if r == 0:
            ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        row_axes.append(ax)
    cb = fig.colorbar(im, ax=row_axes, fraction=0.023, pad=0.02)
    cb.set_label(lab + r" (shared scale)", fontsize=9)

fig.savefig(sys.argv[5], dpi=160)
print(f"Saved: {sys.argv[5]}")
