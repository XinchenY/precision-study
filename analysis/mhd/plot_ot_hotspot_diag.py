#!/usr/bin/env python3
# Fig 9.2 热斑身份作证图: |Δρ| 背景 + 左图叠压缩等值线(激波), 右图叠 |Jz|
# 等值线(电流片)。亮斑与压缩极值重合、与主电流片不重合。
# 数据: 9.1 plain 192² 对 (与 Fig 9.2 同源)。
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

N = 192
d64 = np.loadtxt("results/mhd/orszag_tang/orszag_tang_glm_muscl_hlld_fp64_N192x192.dat")
d32 = np.loadtxt("results/mhd/orszag_tang/orszag_tang_glm_muscl_hlld_fp32_N192x192.dat")
g = lambda a, k: a[:, k].reshape(N, N)
x, y = g(d64, 0), g(d64, 1)
vx, vy = g(d64, 3), g(d64, 4)
Bx, By = g(d64, 7), g(d64, 8)
drho = np.abs(g(d64, 2) - g(d32, 2))
h = 1.0 / N
Jz = np.gradient(By, h, axis=1) - np.gradient(Bx, h, axis=0)
comp = -(np.gradient(vx, h, axis=1) + np.gradient(vy, h, axis=0))  # 压缩为正

fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=True)
for ax, field, levels, color, name in [
        (axes[0], comp, [40, 80], "#FF3333", r"compression $-\nabla\!\cdot v$"),
        (axes[1], np.abs(Jz), [50, 90], "#FF9500", r"current density $|J_z|$")]:
    im = ax.pcolormesh(x, y, drho, norm=LogNorm(vmin=3e-7, vmax=2e-3),
                       cmap="viridis", shading="auto")
    cs = ax.contour(x, y, field, levels=levels, colors=color,
                    linewidths=[0.7, 1.3])
    ax.set_title(rf"$|\Delta\rho|$ with {name} contours", fontsize=11)
    ax.set_xlabel("x")
    ax.set_aspect("equal")
    # 热斑标记
    for (hx, hy) in [(0.815, 0.112), (0.185, 0.888)]:
        ax.add_patch(plt.Circle((hx, hy), 0.055, fill=False, color="white",
                                lw=1.6, ls="--"))
axes[0].set_ylabel("y")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label(r"$|\rho^{\mathrm{FP64}}-\rho^{\mathrm{FP32}}|$")
fig.savefig("analysis/mhd/figures/ot_hotspot_diagnosis.png", dpi=180,
            bbox_inches="tight")
print("saved: analysis/mhd/figures/ot_hotspot_diagnosis.png")
