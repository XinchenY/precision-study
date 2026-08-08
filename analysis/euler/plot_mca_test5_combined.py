#!/usr/bin/env python3
"""
plot_mca_test5_combined.py — Two-panel stacked figure for MCA section (5.4):

  Top panel  : Test 5 density solution (FP32 baseline)
  Bottom panel: pointwise mean significant digits s_rho from 30 MCA samples,
                HLLC vs HLL, shared x-axis.

Vertical dashed lines mark the contact and right shock positions
(auto-detected from the largest |drho/dx| peaks).

Output: analysis/euler/fp32/figures/mca_test5_solution_and_significance.pdf

Usage:  python3 analysis/euler/plot_mca_test5_combined.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
EPS_FP32 = 1.19209289551e-7
S_MAX_FP32 = 24 * np.log10(2)   # ≈ 7.22
S_FLOOR = 3.0                   # threshold for "catastrophic" drop

# ---- Load data ----
sol = np.loadtxt('results/euler/ieee/fp32/toro5_hllc_fp32_200.dat')
x = sol[:, 0]
rho = sol[:, 1]

s_hllc = np.loadtxt('analysis/euler/fp32/sig_digits_test5_two_shocks_hllc.dat')
s_hll  = np.loadtxt('analysis/euler/fp32/sig_digits_test5_two_shocks_hll.dat')
sx = s_hllc[:, 0]
s_rho_hllc = s_hllc[:, 1]
s_rho_hll  = s_hll[:, 1]

# ---- Auto-detect top-3 gradient peaks with minimum separation ----
# Test 5 has shock-contact-shock structure; we detect three local maxima of
# |drho/dx| separated by at least MIN_SEP cells to avoid double-counting.
grad = np.abs(np.gradient(rho))
MIN_SEP = 8  # cells (~ 4% of domain at N=200)
threshold = np.percentile(grad, 80)

# Find local maxima above threshold
candidates = []
for i in range(1, len(grad) - 1):
    if grad[i] > threshold and grad[i] >= grad[i-1] and grad[i] >= grad[i+1]:
        candidates.append((grad[i], i))
candidates.sort(reverse=True)

# Greedy: keep highest peak first, then accept next only if separated enough
chosen = []
for g, i in candidates:
    if all(abs(i - j) >= MIN_SEP for _, j in chosen):
        chosen.append((g, i))
    if len(chosen) == 3:
        break
disc_x = sorted([x[i] for _, i in chosen])
print(f"Discontinuity positions auto-detected at: {disc_x}")

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.5),
                               sharex=True,
                               gridspec_kw={'height_ratios': [1, 1.2],
                                            'hspace': 0.08})

# Top: solution (points, not lines)
ax1.plot(x, rho, 'o', color='black', markersize=2.0, markeredgewidth=0)
ax1.set_ylabel(r'Density $\rho$', fontsize=11)
ax1.set_xlim(0, 1)
ax1.grid(alpha=0.3)
ax1.tick_params(labelbottom=False)

# Vertical dashed lines at discontinuity positions (no in-figure labels;
# physical interpretation is given in the caption).
for xc in disc_x:
    ax1.axvline(xc, color='gray', linestyle='--', lw=0.9, alpha=0.7)

# Bottom: significant digits (points, not lines)
# HLLC = filled blue circles; HLL = open orange circles (distinguishable when overlapping)
ax2.plot(sx, s_rho_hllc, 'o', color='C0', markersize=2.8,
         markeredgewidth=0, label='HLLC')
ax2.plot(sx, s_rho_hll,  'o', color='C1', markersize=2.8,
         markerfacecolor='none', markeredgewidth=0.8, label='HLL')

# Reference lines
ax2.axhline(S_MAX_FP32, color='gray', linestyle=':', lw=0.9, alpha=0.7)
# ax2.axhline(S_FLOOR,    color='red',  linestyle=':', lw=0.9, alpha=0.7)
ax2.text(0.01, S_MAX_FP32, r' $s_{\max}^{\rm FP32}\!\approx\!7.2$',
         va='bottom', fontsize=8, color='dimgray')
# ax2.text(0.01, S_FLOOR, r' $s = 3$ floor',
#          va='bottom', fontsize=8, color='red')

# Same vertical dashed lines in the bottom panel
for xc in disc_x:
    ax2.axvline(xc, color='gray', linestyle='--', lw=0.9, alpha=0.7)

ax2.set_xlabel('Position $x$', fontsize=11)
ax2.set_ylabel(r'$s_{\rho,i}$  (significant digits)', fontsize=11)
ax2.set_ylim(0, 9)
ax2.set_xlim(0, 1)
ax2.grid(alpha=0.3)
ax2.legend(loc='lower left', fontsize=10, frameon=True)

plt.tight_layout()
out_dir = 'analysis/euler/fp32/figures'
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, 'mca_test5_solution_and_significance.pdf')
out_png = os.path.join(out_dir, 'mca_test5_solution_and_significance.png')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, bbox_inches='tight', dpi=160)
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')
