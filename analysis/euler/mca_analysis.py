#!/usr/bin/env python3
"""
mca_analysis.py  —  计算 MCA 30次采样的有效十进制位数
s_d = -log10(|σ / μ|)  =  s_bits × log10(2)

用法:
  python3 analysis/euler/mca_analysis.py --prec fp64
  python3 analysis/euler/mca_analysis.py --prec fp32

输出:
  analysis/euler/fp64/sig_digits_summary.txt  (或 fp32/)
  analysis/euler/fp64/sig_digits_{test}_{solver}.dat
"""

import numpy as np
import glob
import os
import sys

# ── 精度参数 ──────────────────────────────────────────────────
PREC = "fp64"
for i, arg in enumerate(sys.argv):
    if arg == "--prec" and i + 1 < len(sys.argv):
        PREC = sys.argv[i + 1]

if PREC == "fp64":
    MANTISSA_BITS = 53
    PREC_NAME     = "fp64"
elif PREC == "fp32":
    MANTISSA_BITS = 24
    PREC_NAME     = "fp32"
else:
    print(f"Unknown precision: {PREC}. Use fp64 or fp32."); sys.exit(1)

LOG10_2  = np.log10(2)
SD_MAX   = MANTISSA_BITS * LOG10_2     # fp64≈15.95  fp32≈7.22
GAMMA    = 1.4
N_SAMPLES = 30

MCA_DIR  = f"results/euler/mca/{PREC_NAME}"
IEEE_DIR = f"results/euler/ieee/{PREC_NAME}"
OUT_DIR  = f"analysis/euler/{PREC_NAME}"

TESTS = [
    ("test1_sod",         1, "Sod shock tube"),
    ("test2_123",         2, "123 problem (two rarefactions)"),
    ("test3_left_blast",  3, "Left blast wave (p=1000/0.01)"),
    ("test4_right_blast", 4, "Right blast wave (p=0.01/100)"),
    ("test5_two_shocks",  5, "Two strong shocks"),
]
SOLVERS = ["hllc", "hll"]
VARS    = ["rho", "u", "p", "e"]

# ── 函数 ─────────────────────────────────────────────────────
def load_samples(dirpath, n=N_SAMPLES):
    files = sorted(glob.glob(os.path.join(dirpath, "sample_*.dat")))
    if len(files) != n:
        raise RuntimeError(f"Expected {n} samples in {dirpath}, found {len(files)}")
    return np.array([np.loadtxt(f) for f in files])

def values_for_var(data, vname):
    if vname == "rho": return data[:, :, 1]
    if vname == "u":   return data[:, :, 2]
    if vname == "p":   return data[:, :, 3]
    if vname == "e":
        return data[:, :, 3] / ((GAMMA - 1.0) * data[:, :, 1])
    raise ValueError(f"unknown variable: {vname}")

REL_TOL = 1e-2   # |mean| < REL_TOL × max|mean| → μ→0，结果无意义，设为 NaN

def sig_digits(data, vname):
    vals = values_for_var(data, vname)
    mean = np.mean(vals, axis=0)
    std  = np.std(vals,  axis=0, ddof=1)

    # μ→0 判断: 当前点的 |mean| 小于全域最大值的 REL_TOL 倍
    domain_max = np.nanmax(np.abs(mean))
    near_zero  = (domain_max > 0) & (np.abs(mean) < REL_TOL * domain_max)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)
        sd    = np.where(ratio > 0, -np.log10(ratio), SD_MAX)

    sd[near_zero] = np.nan   # μ→0 的点设为 NaN，不参与统计和绘图
    return sd

# ── 主循环 ───────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

summary_lines = []
summary_lines.append(f"Precision: {PREC_NAME}  |  significant decimal digits  (max ≈ {SD_MAX:.2f})")
header = (f"{'Test':<25} {'Solver':>6}  {'var':>3}"
          f"  {'mean_sd':>8}  {'min_sd':>8}  {'median_sd':>9}  {'#sd<3':>6}")
summary_lines.append(header)
summary_lines.append("-" * len(header))

for (dirname, tid, desc), solver in [(t, s) for t in TESTS for s in SOLVERS]:
    dirpath = os.path.join(MCA_DIR, dirname, solver)
    try:
        data = load_samples(dirpath)
    except RuntimeError as e:
        print(f"  SKIP: {e}"); continue

    x = data[0, :, 0]
    sd_all = {v: sig_digits(data, v) for v in VARS}

    out_arr = np.column_stack([x] + [sd_all[v] for v in VARS])
    outfile = os.path.join(OUT_DIR, f"sig_digits_{dirname}_{solver}.dat")
    np.savetxt(outfile, out_arr,
               header="x  sd_rho  sd_u  sd_p  sd_e  [decimal digits]",
               fmt="%.6e")

    label = dirname
    for vname in VARS:
        sd = sd_all[vname]
        fin = sd[np.isfinite(sd)]
        n_nan  = np.sum(np.isnan(sd))
        line = (f"  {label:<25} {solver:>6}  {vname:>3}"
                f"  {np.mean(fin):8.3f}  {np.min(fin):8.3f}"
                f"  {np.median(fin):9.3f}"
                f"  {np.sum(fin < 3):6d}  (NaN={n_nan})")
        summary_lines.append(line)
        label = ""

summary_lines.append("")
output = "\n".join(summary_lines)
print(output)
with open(os.path.join(OUT_DIR, "sig_digits_summary.txt"), "w") as f:
    f.write(output + "\n")
print(f"Saved: {OUT_DIR}/sig_digits_summary.txt")
print(f"Saved: {OUT_DIR}/sig_digits_<test>_<solver>.dat")
