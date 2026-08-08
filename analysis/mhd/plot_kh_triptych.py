#!/usr/bin/env python3
# KH 非线性阶段 FP64/FP32 三联图: 左 FP64 rho, 中 FP32 rho (同色标), 右 |Δrho| (独立 LogNorm).
# 数据: mhd2d_glm_gpu (device 1), 512², t=5, muscl+hlld, ε=1e-4, 确定性单模扰动.
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N = 512
d64 = np.loadtxt("results/mhd/gpu/kh/kh_glm_muscl_hlld_fp64_N512x512.dat")
d32 = np.loadtxt("results/mhd/gpu/kh/kh_glm_muscl_hlld_fp32_N512x512.dat")
g = lambda a, k: a[:, k].reshape(N, N)
x, y = g(d64, 0), g(d64, 1)
r64, r32 = g(d64, 2), g(d32, 2)
diff = np.abs(r32 - r64)

vmin = min(r64.min(), r32.min())
vmax = max(r64.max(), r32.max())
rel = np.linalg.norm(r32 - r64) / np.linalg.norm(r64)
print(f"rho: min64={r64.min():.4f} max64={r64.max():.4f}  relL2(rho)={rel:.3e}  max|d|={diff.max():.3e}")

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), sharey=True,
                         constrained_layout=True)
for ax, f, title in [(axes[0], r64, r"FP64: $\rho$"),
                     (axes[1], r32, r"FP32: $\rho$")]:
    im = ax.pcolormesh(x, y, f, cmap="viridis", vmin=vmin, vmax=vmax,
                       shading="auto")
    ax.set_title(title + rf"  ($t=5$)", fontsize=12)
    ax.set_xlabel("x")
    ax.set_aspect("equal")
fig.colorbar(im, ax=axes[:2], shrink=0.9, pad=0.015, label=r"$\rho$")

dpos = diff[diff > 0]
vmin_d = max(dpos.min(), diff.max() * 1e-8) if dpos.size else 1e-12
imd = axes[2].pcolormesh(x, y, np.maximum(diff, vmin_d),
                         norm=LogNorm(vmin=vmin_d, vmax=diff.max()),
                         cmap="magma", shading="auto")
axes[2].set_title(r"$|\rho_{\mathrm{FP32}} - \rho_{\mathrm{FP64}}|$", fontsize=12)
axes[2].set_xlabel("x")
axes[2].set_aspect("equal")
fig.colorbar(imd, ax=axes[2], shrink=0.9, pad=0.03,
             label=r"$|\Delta\rho|$")
axes[0].set_ylabel("y")

out = "analysis/mhd/figures/kh_fp32_fp64_triptych.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved:", out)
