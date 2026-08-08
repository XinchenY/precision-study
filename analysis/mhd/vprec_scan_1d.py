#!/usr/bin/env python3
# Ch10 Step 2b: 汇总 VPREC 位宽扫描 (1D Brio-Wu) 的 relL2-vs-t 表与单对数图。
#   输入: logs/mhd/vprec_scan1d/cmp_tNN.txt (compare_precision.py 输出)
#   输出: analysis/mhd/vprec/scan1d_table.txt
#         analysis/mhd/figures/vprec_scan1d_relL2.png
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "logs/mhd/vprec_scan1d"
TS = [16, 20, 23, 28, 32, 40, 52]
VARS = ["rho", "vx", "vy", "p", "By", "E"]          # vz/Bx/Bz 恒为零, 不报
THRESHOLD = 1.0e-5                                   # 2c 预定义门槛
# Step 1 原生 FP32 vs FP64 (clang18) 参考水平, 见 logs/mhd/vprec_step1/
FP32_REF = {"rho": 9.30e-7, "vx": 8.18e-6, "vy": 6.57e-6,
            "p": 2.27e-6, "By": 1.50e-6, "E": 1.39e-6}

def read_cmp(path):
    if not os.path.exists(path):
        return None
    vals = {}
    for line in open(path):
        m = re.match(r"\s*(\w+)\s+[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s+([\d.e+-]+)\s*$", line)
        if m and m.group(1) in VARS:
            vals[m.group(1)] = float(m.group(2))
    return vals if len(vals) == len(VARS) else None

table = {}
for t in TS:
    vals = read_cmp(f"{LOG}/cmp_t{t}.txt")
    table[t] = vals

os.makedirs("analysis/mhd/vprec", exist_ok=True)
with open("analysis/mhd/vprec/scan1d_table.txt", "w") as f:
    f.write("# VPREC scan, 1D Brio-Wu N=800, mode=ob, eps=1e-4; relL2 vs baseline (clang18 fp64)\n")
    f.write("# threshold (2c): all vars <= 1e-5 and run completed\n")
    f.write(f"{'t':>4} " + "".join(f"{v:>12}" for v in VARS) + "   pass\n")
    for t in TS:
        vals = table[t]
        if vals is None:
            f.write(f"{t:>4} " + "CRASHED".rjust(12 * len(VARS)) + "   no\n")
            continue
        ok = all(vals[v] <= THRESHOLD for v in VARS)
        f.write(f"{t:>4} " + "".join(f"{vals[v]:>12.3e}" for v in VARS)
                + ("   yes\n" if ok else "   no\n"))
print(open("analysis/mhd/vprec/scan1d_table.txt").read())

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for v in VARS:
    ts = [t for t in TS if table[t] is not None]
    ax.semilogy(ts, [table[t][v] for t in ts], "o-", label=v, lw=1.4, ms=4)
ax.axhline(THRESHOLD, color="k", ls="--", lw=1.0)
ax.text(TS[-1], THRESHOLD * 1.35, "threshold $10^{-5}$", ha="right", fontsize=9)
ax.axvline(23, color="gray", ls=":", lw=1.0)
ax.text(23.3, ax.get_ylim()[0] * 2, "binary32 (t=23)", fontsize=9, color="gray")
ax.set_xlabel("VPREC fraction precision t (bits)")
ax.set_ylabel(r"relative $L_2$ vs FP64 baseline")
ax.set_xticks(TS)
ax.grid(alpha=0.3, which="both")
ax.legend(ncol=3, fontsize=9)
fig.tight_layout()
os.makedirs("analysis/mhd/figures", exist_ok=True)
fig.savefig("analysis/mhd/figures/vprec_scan1d_relL2.png", dpi=160)
print("saved: analysis/mhd/figures/vprec_scan1d_relL2.png")