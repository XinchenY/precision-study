#!/usr/bin/env python3
"""
compare_2d.py — Backend consistency for 2D problems (CPU vs GPU)

For a pair of 2D output files in the 6-column format
    x  y  rho  u  v  p
(blank lines between x-blocks are tolerated), compute L1/L2/L∞ of
|CPU - GPU| on rho, u, v, p.

Usage:
  python3 analysis/euler/compare_2d.py CPU.dat GPU.dat
  python3 analysis/euler/compare_2d.py CPU.dat GPU.dat --tag "shock-bubble FP64"
  python3 analysis/euler/compare_2d.py CPU.dat GPU.dat --tex
"""

import argparse
import os
import sys
import numpy as np

VAR_NAMES = ["rho", "u", "v", "p"]
VAR_COLS  = {"rho": 2, "u": 3, "v": 4, "p": 5}


def norms(a, b):
    d = np.abs(a - b)
    return d.mean(), np.sqrt((d ** 2).mean()), d.max()


def load_2d(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # np.loadtxt skips blank lines and "# ..." comment lines by default
    return np.loadtxt(path, comments="#")


def fmt_sci(x):
    return f"{x:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu", help="CPU .dat file")
    ap.add_argument("gpu", help="GPU .dat file")
    ap.add_argument("--tag", default="", help="Optional label printed above the table")
    ap.add_argument("--tex", action="store_true", help="Also emit LaTeX table")
    ap.add_argument("--allow-grid-mismatch", action="store_true",
                    help="Tolerate small (x,y) differences (use when comparing across precisions)")
    args = ap.parse_args()

    try:
        A = load_2d(args.cpu)
        B = load_2d(args.gpu)
    except FileNotFoundError as e:
        print(f"Missing file: {e}", file=sys.stderr)
        sys.exit(1)

    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}", file=sys.stderr)
        sys.exit(1)

    # 坐标检查 — backend 对比要求严格相等; 跨精度对比放宽到 rtol=1e-5
    grid_strict = (np.array_equal(A[:, 0], B[:, 0]) and np.array_equal(A[:, 1], B[:, 1]))
    if not grid_strict:
        if args.allow_grid_mismatch:
            grid_loose = (np.allclose(A[:, 0], B[:, 0], rtol=1e-5, atol=0)
                          and np.allclose(A[:, 1], B[:, 1], rtol=1e-5, atol=0))
            if not grid_loose:
                print("ERROR: (x,y) grids differ by more than rtol=1e-5 — likely different grids!",
                      file=sys.stderr)
                sys.exit(1)
        else:
            print("WARNING: (x,y) grids differ between files", file=sys.stderr)
            print("         (pass --allow-grid-mismatch if comparing across precisions)",
                  file=sys.stderr)
            sys.exit(1)

    rows = []
    for var in VAR_NAMES:
        c = VAR_COLS[var]
        l1, l2, linf = norms(A[:, c], B[:, c])
        rows.append((var, l1, l2, linf))

    title = args.tag if args.tag else f"{args.cpu}  vs  {args.gpu}"
    print()
    print(f"=== CPU vs GPU backend consistency  ({title}) ===")
    print(f"  N points = {A.shape[0]}")
    print()
    header = f"{'Var':<4} {'L1':>12} {'L2':>12} {'Linf':>12}"
    print(header)
    print("-" * len(header))
    for var, l1, l2, linf in rows:
        print(f"{var:<4} {fmt_sci(l1):>12} {fmt_sci(l2):>12} {fmt_sci(linf):>12}")

    if args.tex:
        print()
        print(f"% LaTeX table — {title}")
        print(r"\begin{tabular}{lccc}")
        print(r"\toprule")
        print(r"Variable & $L_1$ & $L_2$ & $L_\infty$ \\")
        print(r"\midrule")
        for var, l1, l2, linf in rows:
            print(f"${var}$ & {fmt_sci(l1)} & {fmt_sci(l2)} & {fmt_sci(linf)} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
