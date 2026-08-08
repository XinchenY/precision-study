#!/usr/bin/env python3
"""
mca_brio_wu.py — MHD 1D Brio-Wu 的 MCA 分析 (与 Euler mca_analysis.py 同协议)
  s_d = -log10(sigma/|mu|), 逐格点逐变量, 超过格式上限 SD_MAX 处截断到上限。
  统一 mask: 用确定性 FP64-HLLD 参考解定义 |q_ref| >= 1e-2 * max|q_ref|,
  四个配置 (求解器 x 精度) 全部在同一批格点上统计; 各变量保留比例一并输出。
  表格统计量: mean / p05 / median (min 及其位置仅作内部记录)。
  主分析固定用前 30 个样本 (与 Report 1 同规格); 样本 >30 时做收敛检查。
输出:
  analysis/mhd/{prec}/mca_summary.txt              — 汇总表
  analysis/mhd/figures/mca_brio_wu_sd_rho_{prec}.png — Fig 5.12 同款 (rho)
Usage:
  python3 analysis/mhd/mca_brio_wu.py [fp64|fp32]
"""
import sys, os, re, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREC = sys.argv[1] if len(sys.argv) > 1 else "fp32"
MANTISSA_BITS = 24 if PREC == "fp32" else 53
SD_MAX  = MANTISSA_BITS * np.log10(2)      # fp32≈7.22, fp64≈15.95
N_PRIMARY = 30                              # 主分析样本数, 与 Report 1 一致
REL_TOL = 1e-2

MCA_DIR = f"results/mhd/mca/brio_wu/{PREC}"
DET_DIR = "results/mhd/brio_wu"
OUT_DIR = f"analysis/mhd/{PREC}"
FIG_DIR = "analysis/mhd/figures"
SOLVERS = ["hlld", "hll"]
# Brio-Wu .dat 列: x rho vx vy vz p Bx By Bz E
VARS = {"rho": 1, "vx": 2, "vy": 3, "p": 5, "By": 7, "E": 9}
WAVES = {"compound": 0.470, "contact": 0.560, "slow shock": 0.645}


def load_samples(dirpath):
    """按编号数值排序读入全部样本 (sample_100 不会被字符串排序插错位)。"""
    files = glob.glob(os.path.join(dirpath, "sample_*.dat"))
    files.sort(key=lambda f: int(re.search(r"sample_(\d+)", f).group(1)))
    if len(files) < 2:
        raise RuntimeError(f"need >=2 samples in {dirpath}, found {len(files)}")
    return np.array([np.loadtxt(f) for f in files]), len(files)


def sig_digits(vals):
    """vals: (n_samples, n_cells) -> s_d per cell (未做空间 mask, 上限截断)。"""
    mean = np.mean(vals, axis=0)
    std = np.std(vals, axis=0, ddof=1)          # 样本标准差, ddof=1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(mean) > 0, std / np.abs(mean), np.nan)
        sd = np.where(ratio > 0, -np.log10(ratio), SD_MAX)
    return np.minimum(sd, SD_MAX)


# ── 统一参考 mask: 确定性 FP64-HLLD 解 (与精度/求解器配置无关) ────────────────
ref = np.loadtxt(f"{DET_DIR}/brio_wu_muscl_hlld_fp64_N800.dat")
x_ref = ref[:, 0]
MASKS = {v: np.abs(ref[:, c]) >= REL_TOL * np.max(np.abs(ref[:, c]))
         for v, c in VARS.items()}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

lines = [f"MCA Brio-Wu (N=800, t=0.1, CFL=0.4, mode=mca, primary n={N_PRIMARY})",
         f"Precision: {PREC}  |  significant decimal digits (SD_MAX ≈ {SD_MAX:.2f},"
         f" values above SD_MAX truncated)",
         "Verificarlo 2.4.0; 30 independent runs; sigma = sample std (ddof=1)",
         "",
         "Common mask (deterministic FP64-HLLD reference, |q_ref| >= 1e-2 max|q_ref|):"]
ncell = ref.shape[0]
for v in VARS:
    n = int(MASKS[v].sum())
    lines.append(f"  {v:3s}: retained {n}/{ncell} = {100.0 * n / ncell:.1f}%")
lines.append("")

header = (f"{'Solver':>6} {'var':>4} {'n':>4}  {'mean_sd':>8} {'p05_sd':>7}"
          f" {'median_sd':>9} {'min_sd':>7} {'x_min':>7} {'#cap':>5}")
lines += [header, "-" * len(header)]

sd_rho = {}
for solver in SOLVERS:
    data, n_all = load_samples(os.path.join(MCA_DIR, solver))
    prim = data[:N_PRIMARY]
    for vname, col in VARS.items():
        sd = sig_digits(prim[:, :, col])
        if vname == "rho":
            sd_rho[solver] = np.where(MASKS[vname], sd, np.nan)
        valid = MASKS[vname] & np.isfinite(sd)
        vals = sd[valid]
        i_min = np.flatnonzero(valid)[np.argmin(vals)]
        lines.append(f"{solver:>6} {vname:>4} {min(n_all, N_PRIMARY):>4}"
                     f"  {vals.mean():>8.2f} {np.percentile(vals, 5):>7.2f}"
                     f" {np.median(vals):>9.2f} {vals.min():>7.2f}"
                     f" {x_ref[i_min]:>7.3f} {int((vals >= SD_MAX).sum()):>5d}")
    # 收敛检查: 样本多于 N_PRIMARY 时, 前 30 vs 全样本 (同一 mask)
    if n_all > N_PRIMARY:
        for vname, col in VARS.items():
            d30 = sig_digits(data[:N_PRIMARY, :, col])
            dall = sig_digits(data[:, :, col])
            diff = np.abs(dall - d30)[MASKS[vname]]
            ok = diff[np.isfinite(diff)]
            lines.append(f"    [convergence {solver}/{vname}: n=30 vs n={n_all}]"
                         f"  median|Δsd|={np.median(ok):.3f}  max|Δsd|={ok.max():.3f}")

summary = "\n".join(lines)
print(summary)
with open(f"{OUT_DIR}/mca_summary.txt", "w") as f:
    f.write(summary + "\n")

# ── Fig 5.12 同款: 上=确定性 rho 剖面, 下=s_rho (HLLD vs HLL) ────────────────
det = np.loadtxt(f"{DET_DIR}/brio_wu_muscl_hlld_{PREC}_N800.dat")
x = det[:, 0]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.6, 7.4), constrained_layout=True,
                               sharex=True)
ax0.plot(x, det[:, 1], ".", color="black", ms=2.5)
ax0.set_ylabel(r"Density $\rho$")
ax1.plot(x, sd_rho["hlld"], ".", color="C0", ms=3, label="HLLD")
ax1.plot(x, sd_rho["hll"], "o", mfc="none", mec="C1", ms=3.5, mew=0.7, label="HLL")
ax1.axhline(SD_MAX, color="0.6", ls=":", lw=0.8)
ax1.text(0.01, SD_MAX, rf"$s_\max^{{\mathrm{{{PREC.upper()}}}}}\approx{SD_MAX:.1f}$",
         va="bottom", fontsize=10, color="0.4")
for ax in (ax0, ax1):
    for name, xw in WAVES.items():
        ax.axvline(xw, color="0.55", ls="--", lw=0.7)
for name, xw in WAVES.items():
    ax0.text(xw, ax0.get_ylim()[1], name, ha="center", va="bottom",
             fontsize=8, color="0.4")
ax1.set_ylim(0, np.ceil(SD_MAX) + 1)
ax1.set_ylabel(r"$s_{\rho,i}$ (significant digits)")
ax1.set_xlabel(r"Position $x$")
ax1.legend(frameon=False, loc="lower left")

out = f"{FIG_DIR}/mca_brio_wu_sd_rho_{PREC}.png"
fig.savefig(out, dpi=160)
print(f"Saved: {out}")
