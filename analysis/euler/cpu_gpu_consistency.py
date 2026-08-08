#!/usr/bin/env python3
"""
cpu_gpu_consistency.py — Backend consistency table (CPU vs GPU, FP64)

For each Toro test (1-5) with the chosen solver, compute L1/L2/Linf of
|CPU - GPU| on rho, u, p. Reproduces the style of paper Table 5.7 but
extended to all 5 tests.

Usage:
  python3 analysis/euler/cpu_gpu_consistency.py                  # default: hllc, fp64, N=200
  python3 analysis/euler/cpu_gpu_consistency.py --solver hll
  python3 analysis/euler/cpu_gpu_consistency.py --tex            # also emit a LaTeX-ready table
  python3 analysis/euler/cpu_gpu_consistency.py --prec fp32

File layout assumed:
  CPU  -> results/euler/ieee/{prec}/toro{T}_{solver}_{prec}_{N}.dat
  GPU  -> results/euler/gpu/{prec}/toro{T}_{solver}_{prec}_{N}.dat
"""

import argparse
import os
import sys
import numpy as np

VAR_NAMES = ["rho", "u", "p"]
VAR_COLS  = {"rho": 1, "u": 2, "p": 3}
TEST_LABELS = {
    1: "Sod",
    2: "123 problem",
    3: "Left blast",
    4: "Right blast",
    5: "Two shocks",
}


def norms(a, b):
    d = np.abs(a - b)
    return d.mean(), np.sqrt((d ** 2).mean()), d.max()


def load_pair(cpu_path, gpu_path):
    if not os.path.exists(cpu_path):
        raise FileNotFoundError(f"CPU file missing: {cpu_path}")
    if not os.path.exists(gpu_path):
        raise FileNotFoundError(f"GPU file missing: {gpu_path}")
    A = np.loadtxt(cpu_path)
    B = np.loadtxt(gpu_path)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    if not np.array_equal(A[:, 0], B[:, 0]):
        # x-grid差异通常意味着采样格点不同 — backend对比前必须排除
        raise ValueError(f"x-grid differs between {cpu_path} and {gpu_path}")
    return A, B


def fmt_sci(x):
    return f"{x:.2e}"


def print_text_table(rows, solver, prec, N):
    print()
    print(f"=== CPU vs GPU backend consistency  ({solver.upper()}, {prec}, N={N}) ===")
    print()
    header = f"{'Test':<14} {'Var':<4} {'L1':>12} {'L2':>12} {'Linf':>12}"
    print(header)
    print("-" * len(header))
    for tid, var, l1, l2, linf in rows:
        label = TEST_LABELS[tid] if var == "rho" else ""
        print(f"{label:<14} {var:<4} {fmt_sci(l1):>12} {fmt_sci(l2):>12} {fmt_sci(linf):>12}")
        if var == "p":
            print()


def print_tex_table(rows, solver, prec, N):
    print()
    print(f"% LaTeX table — CPU/GPU {solver.upper()} {prec} N={N}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Test & Variable & $L_1$ & $L_2$ & $L_\infty$ \\")
    print(r"\midrule")
    last_tid = None
    for tid, var, l1, l2, linf in rows:
        label = TEST_LABELS[tid] if tid != last_tid else ""
        if tid != last_tid and last_tid is not None:
            print(r"\midrule")
        print(f"{label} & ${var}$ & {fmt_sci(l1)} & {fmt_sci(l2)} & {fmt_sci(linf)} \\\\")
        last_tid = tid
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="hllc", choices=["hllc", "hll"])
    ap.add_argument("--prec", default="fp64", choices=["fp64", "fp32"])
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--tests", default="1,2,3,4,5",
                    help="Comma-separated test IDs (default: 1,2,3,4,5)")
    ap.add_argument("--cpu-dir", default="results/euler/ieee")
    ap.add_argument("--gpu-dir", default="results/euler/gpu")
    ap.add_argument("--tex", action="store_true",
                    help="Also print a LaTeX-ready table")
    args = ap.parse_args()

    tests = [int(t) for t in args.tests.split(",") if t.strip()]
    rows = []
    missing = []

    for tid in tests:
        fname = f"toro{tid}_{args.solver}_{args.prec}_{args.N}.dat"
        cpu_p = os.path.join(args.cpu_dir, args.prec, fname)
        gpu_p = os.path.join(args.gpu_dir, args.prec, fname)
        try:
            A, B = load_pair(cpu_p, gpu_p)
        except FileNotFoundError as e:
            missing.append(str(e))
            continue
        for var in VAR_NAMES:
            c = VAR_COLS[var]
            l1, l2, linf = norms(A[:, c], B[:, c])
            rows.append((tid, var, l1, l2, linf))

    if missing:
        print("Missing files (skipped):", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        print(file=sys.stderr)

    if not rows:
        print("No comparable files found.", file=sys.stderr)
        sys.exit(1)

    print_text_table(rows, args.solver, args.prec, args.N)
    if args.tex:
        print_tex_table(rows, args.solver, args.prec, args.N)


if __name__ == "__main__":
    main()
