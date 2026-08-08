#!/usr/bin/env python3
# Ch10 fig:vprec-scan — VPREC 位宽扫描, R(t) 口径 (1D Brio-Wu N=800)
#   R(t) = relL2(t) / relL2(native FP32), 主线取变量最大值, 细灰线为逐变量。
#   判据元素: 水平线 R=1 (native FP32 parity), 竖线 t*=23。
#   数据: logs/mhd/vprec_scan1d/cmp_tNN.txt; fp32 参照 = Step1 clang18 逐变量。
#   输出: analysis/mhd/figures/vprec_scan_relL2.png
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "logs/mhd/vprec_scan1d"
TS = [16, 20, 23, 28, 32, 40, 52]
VARS = ["rho", "vx", "vy", "p", "By", "E"]
FP32 = {"rho": 9.30e-7, "vx": 8.18e-6, "vy": 6.57e-6,
        "p": 2.27e-6, "By": 1.50e-6, "E": 1.39e-6}

def read_cmp(path):
    vals = {}
    for line in open(path):
        m = re.match(r"\s*(\w+)\s+[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s+([\d.e+-]+)\s*$", line)
        if m and m.group(1) in VARS:
            vals[m.group(1)] = float(m.group(2))
    return vals

table = {t: read_cmp(f"{LOG}/cmp_t{t}.txt") for t in TS}
ratios = {t: {v: table[t][v] / FP32[v] for v in VARS} for t in TS}
Rmax = [max(ratios[t].values()) for t in TS]

fig, ax = plt.subplots(figsize=(7.6, 4.8))

# R>1 区底色 (t < 23): 劣于 native FP32
ax.axvspan(14.8, 23, color="0.92", zorder=0)
ax.text(18.9, 3.0e-7, "worse than\nnative FP32", ha="center",
        fontsize=10, color="0.35")

# 主线: 最大比值 R(t)
ax.semilogy(TS, Rmax, marker="s", color="#EE6677", lw=2.0, ms=7,
            mec="white", mew=0.7, zorder=4,
            label=r"$R(t)=\max_{q}\; \mathrm{rel}L_2^{\,t}(q)\,/\,\mathrm{rel}L_2^{\,\mathrm{FP32}}(q)$")

# t=23 与 t=20 的数值标注 (关键点无争议)
ax.annotate(r"$R(23)=0.97$", xy=(23, 0.974), xytext=(26.5, 6e-2),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))
ax.annotate(r"$R(20)=20$", xy=(20, 19.8), xytext=(24.5, 40),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8))

# 判据线: R = 1 (native FP32 parity)
ax.axhline(1.0, color="k", ls="--", lw=1.3, zorder=3)
ax.text(53.0, 0.55, "native FP32 parity  ($R=1$)", ha="right", va="top",
        fontsize=10)

# 门槛位宽竖线 t* = 23
ax.axvline(23, color="k", ls="-.", lw=1.4, zorder=3)
ax.text(23.6, 1.2e-10, "lowest passing tested setting\n" + r"$t=23$ (binary32 significand)",
        fontsize=10.5, va="bottom", ha="left")

ax.set_xlabel("VPREC significand width $t$ (bits)", fontsize=11)
ax.set_ylabel(r"$R(t)$: error relative to native FP32", fontsize=11)
ax.set_xticks(TS)
ax.set_xticklabels(["16", "20", "23", "28", "32", "40", "52\nbinary64"])
ax.set_xlim(14.8, 53.5)
ax.set_ylim(5e-11, 5e3)
ax.grid(alpha=0.25, which="major")
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
ax.tick_params(labelsize=10)

fig.tight_layout()
os.makedirs("analysis/mhd/figures", exist_ok=True)
fig.savefig("analysis/mhd/figures/vprec_scan_relL2.png", dpi=200)
print("saved: analysis/mhd/figures/vprec_scan_relL2.png")
