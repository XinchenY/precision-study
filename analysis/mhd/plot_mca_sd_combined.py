#!/usr/bin/env python3
"""
plot_mca_sd_combined.py — FP64 + FP32 的 s_rho 剖面合并成一张 1-1-1 图
  上 = 确定性 rho 剖面 (FP64)
  中 = s_rho (FP64): HLLD vs HLL, 上限线 15.95
  下 = s_rho (FP32): HLLD vs HLL, 上限线 7.22
  两个 s_d panel 的 y 轴取相同跨度 (各自上限往下 WINDOW 位), 便于直接
  看出 "损失剖面形状相同, 整体平移 ~8.7 位"。
Usage:
  python3 analysis/mhd/plot_mca_sd_combined.py
"""
import os, re, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_PRIMARY = 30
REL_TOL = 1e-2
WINDOW = 5.5      # 每个 s_d panel 显示的位数跨度 / y-span in digits
WAVES = {"compound": 0.470, "contact": 0.560, "slow shock": 0.645}
FIG_DIR = "analysis/mhd/figures"


def load_samples(dirpath):
    files = glob.glob(os.path.join(dirpath, "sample_*.dat"))
    files.sort(key=lambda f: int(re.search(r"sample_(\d+)", f).group(1)))
    return np.array([np.loadtxt(f) for f in files[:N_PRIMARY]])


def sig_digits(vals, sd_max):
    mean = np.mean(vals, axis=0)
    std = np.std(vals, axis=0, ddof=1)
    domain_max = np.nanmax(np.abs(mean))
    near_zero = (domain_max > 0) & (np.abs(mean) < REL_TOL * domain_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)
        sd = np.where(ratio > 0, -np.log10(ratio), sd_max)
    sd = np.minimum(sd, sd_max)
    sd[near_zero] = np.nan
    return sd


det = np.loadtxt("results/mhd/brio_wu/brio_wu_muscl_hlld_fp64_N800.dat")
x = det[:, 0]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.8), constrained_layout=True,
                         sharex=True)

axes[0].plot(x, det[:, 1], ".", color="black", ms=2.5)
axes[0].set_ylabel(r"Density $\rho$")
for name, xw in WAVES.items():
    axes[0].text(xw, axes[0].get_ylim()[1], name, ha="center", va="bottom",
                 fontsize=9, color="0.4")

for ax, prec, bits in ((axes[1], "fp64", 53), (axes[2], "fp32", 24)):
    sd_max = bits * np.log10(2)
    for solver, style in (("hlld", dict(marker=".", color="C0", ms=3, ls="")),
                          ("hll", dict(marker="o", mfc="none", mec="C1",
                                       ms=3.5, mew=0.7, ls=""))):
        data = load_samples(f"results/mhd/mca/brio_wu/{prec}/{solver}")
        ax.plot(x, sig_digits(data[:, :, 1], sd_max), label=solver.upper(),
                **style)
    ax.axhline(sd_max, color="0.6", ls=":", lw=0.9)
    ax.text(0.012, sd_max - 0.12,
            rf"$s_\max^{{\mathrm{{{prec.upper()}}}}}\approx{sd_max:.1f}$",
            va="top", fontsize=11, color="0.35")
    ax.set_ylim(sd_max - WINDOW, sd_max + 0.45)
    ax.set_ylabel(rf"$s_{{\rho,i}}$ ({prec.upper()})")

for ax in axes:
    for name, xw in WAVES.items():
        ax.axvline(xw, color="0.55", ls="--", lw=0.7)
axes[1].legend(frameon=False, loc="lower left")
axes[2].set_xlabel(r"Position $x$")

out = f"{FIG_DIR}/mca_brio_wu_sd_rho_combined.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
