#!/usr/bin/env python3
"""
plot_brio_wu_solver_compare.py — 确定性 HLLD vs HLL 对比 (MCA 小节的铺垫图)
  上 = 全域 rho 剖面 (HLLD 实心点, HLL 空心圈)
  下 = compound wave / contact / slow shock 区域放大 (x 窗口可调)
Usage:
  python3 analysis/mhd/plot_brio_wu_solver_compare.py [fp64|fp32] [OUT.png]
"""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREC = sys.argv[1] if len(sys.argv) > 1 else "fp64"
OUT  = sys.argv[2] if len(sys.argv) > 2 else f"analysis/mhd/figures/brio_wu_solver_compare_{PREC}.png"

# 波位置虚线 (与 Fig 9.1 一致; 按你论文里的值微调) / dashed wave markers
WAVES = {"compound": 0.470, "contact": 0.560, "slow shock": 0.645}
ZOOM  = (0.40, 0.72)   # 放大窗口 / zoom window

D = "results/mhd/brio_wu"
hlld = np.loadtxt(f"{D}/brio_wu_muscl_hlld_{PREC}_N800.dat")
hll  = np.loadtxt(f"{D}/brio_wu_muscl_hll_{PREC}_N800.dat")

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=True)

for ax, xlim in ((axes[0], None), (axes[1], ZOOM)):
    ax.plot(hlld[:, 0], hlld[:, 1], ".", color="C0", ms=3, label="HLLD")
    ax.plot(hll[:, 0],  hll[:, 1],  "o", mfc="none", mec="C1", ms=3.5,
            mew=0.7, label="HLL")
    for name, xw in WAVES.items():
        ax.axvline(xw, color="0.55", ls="--", lw=0.7)
    ax.set_ylabel(r"$\rho$")
    if xlim:
        ax.set_xlim(*xlim)
        lo = (hlld[:, 0] >= xlim[0]) & (hlld[:, 0] <= xlim[1])
        pad = 0.05 * (hlld[lo, 1].max() - hlld[lo, 1].min())
        ax.set_ylim(hlld[lo, 1].min() - pad, hlld[lo, 1].max() + pad)
axes[0].legend(frameon=False, loc="upper right")
for name, xw in WAVES.items():
    axes[0].text(xw, axes[0].get_ylim()[1], name, ha="center", va="bottom",
                 fontsize=8, color="0.4")
# 上图标出下图的放大区域 / mark the zoom window of the lower panel
lo = (hlld[:, 0] >= ZOOM[0]) & (hlld[:, 0] <= ZOOM[1])
y0, y1 = hlld[lo, 1].min(), hlld[lo, 1].max()
pad = 0.06 * (y1 - y0)
axes[0].add_patch(plt.Rectangle((ZOOM[0], y0 - pad), ZOOM[1] - ZOOM[0],
                                (y1 - y0) + 2 * pad, fill=False,
                                edgecolor="0.35", lw=0.9, ls="-"))
axes[0].annotate("enlarged below", xy=(ZOOM[1], y0), fontsize=8,
                 color="0.35", ha="right", va="top",
                 xytext=(ZOOM[1] - 0.005, y0 - 2.5 * pad))
axes[1].set_xlabel("x")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=160)
print(f"Saved: {OUT}")
