#!/usr/bin/env python3
"""
plot_2d_fp32_vs_fp64.py  —  2D shock-bubble fp32 vs fp64 comparison

Generates:
  1. analysis/euler/figures/2d_density_fp64.pdf         — fp64 density field
  2. analysis/euler/figures/2d_density_fp32.pdf         — fp32 density field
  3. analysis/euler/figures/2d_density_diff_fp32_fp64.pdf — |fp64 - fp32| pointwise
  4. prints L1/L2/Linf norms of the difference

Usage:
  python3 analysis/euler/plot_2d_fp32_vs_fp64.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIG_DIR = "analysis/euler/figures"
os.makedirs(FIG_DIR, exist_ok=True)

F64 = "results/euler/2d/fp64/cpu_shock_bubble.dat"
F32 = "results/euler/2d/fp32/cpu_shock_bubble.dat"

for f in [F64, F32]:
    if not os.path.exists(f):
        print(f"Missing: {f}"); exit(1)

print("Loading data ...")
A64 = np.loadtxt(F64)
A32 = np.loadtxt(F32)

# columns: x y rho u v p
x64   = A64[:, 0];  y64  = A64[:, 1]
rho64 = A64[:, 2]
rho32 = A32[:, 2]

# Infer grid size from unique x values
NX = len(np.unique(x64))
NY = len(np.unique(y64))
print(f"Grid: {NX} x {NY}")

X   = x64.reshape(NX, NY)
Y   = y64.reshape(NX, NY)
R64 = rho64.reshape(NX, NY)
R32 = rho32.reshape(NX, NY)
DIFF = np.abs(R64 - R32)

# ── norms ─────────────────────────────────────────────────────────────────
d = DIFF.ravel()
print(f"\nDensity |fp64 - fp32|:")
print(f"  L1   = {d.mean():.4e}")
print(f"  L2   = {np.sqrt((d**2).mean()):.4e}")
print(f"  Linf = {d.max():.4e}")

# ── plot 1: fp64 density ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
cf = ax.contourf(X, Y, R64, levels=50, cmap="plasma")
plt.colorbar(cf, ax=ax, label=r"$\rho$ (kg/m³)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(r"Density $\rho^{\mathrm{FP64}}$,  $\tau=19$")
ax.set_aspect("equal")
out = os.path.join(FIG_DIR, "2d_density_fp64.pdf")
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── plot 2: fp32 density ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
cf = ax.contourf(X, Y, R32, levels=50, cmap="plasma")
plt.colorbar(cf, ax=ax, label=r"$\rho$ (kg/m³)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(r"Density $\rho^{\mathrm{FP32}}$,  $\tau=19$")
ax.set_aspect("equal")
out = os.path.join(FIG_DIR, "2d_density_fp32.pdf")
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out}")

# ── plot 3: pointwise difference |fp64 - fp32| ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
cf = ax.contourf(X, Y, DIFF, levels=50, cmap="hot_r")
plt.colorbar(cf, ax=ax, label=r"$|\rho^{\mathrm{FP64}} - \rho^{\mathrm{FP32}}|$")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(r"Pointwise density difference $|\rho^{\mathrm{FP64}} - \rho^{\mathrm{FP32}}|$,  $\tau=19$")
ax.set_aspect("equal")
out = os.path.join(FIG_DIR, "2d_density_diff_fp32_fp64.pdf")
fig.savefig(out, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved: {out}")

print("\nDone.")
