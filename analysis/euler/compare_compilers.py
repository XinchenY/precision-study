#!/usr/bin/env python3
"""
compare_compilers.py — Compare compiler-flag variants vs a baseline build.

For each test case in results/euler/compiler/fp64/<build>/<canonical_file>,
compute L1 / L2 / L_inf of |variant - baseline| on all primitive variables.
Auto-detects 1D (4-col) vs 2D (6-col) data.

Default baseline: cpu_O2 (IEEE-strict reference)
Default variants: cpu_O3, cpu_Ofast, gpu_default, gpu_fastmath
Default cases:    toro5, shock_bubble, riemann2d_cfg12

Usage:
    python3 analysis/euler/compare_compilers.py                      # full per-case tables
    python3 analysis/euler/compare_compilers.py --summary            # compact L_inf grid
    python3 analysis/euler/compare_compilers.py --tex                # also LaTeX output
    python3 analysis/euler/compare_compilers.py --baseline gpu_default
    python3 analysis/euler/compare_compilers.py --builds cpu_Ofast,gpu_fastmath
"""

import argparse
import os
import sys
import numpy as np

ALL_CASES = ["toro5", "shock_bubble", "riemann2d_cfg12"]
ALL_BUILDS = ["cpu_O2", "cpu_O3", "cpu_Ofast", "gpu_default", "gpu_fastmath"]

CASE_FILE = {
    "toro5":            "toro5_hllc_fp64_200.dat",
    "shock_bubble":     "shock_bubble.dat",
    "riemann2d_cfg12":  "riemann2d_cfg12_2S2C_hllc_400.dat",
}

# Column maps by number of columns in the file (auto-detected)
VARS_BY_NCOLS = {
    4: [("rho", 1), ("u", 2), ("p", 3)],            # 1D: x rho u p
    6: [("rho", 2), ("u", 3), ("v", 4), ("p", 5)],  # 2D: x y rho u v p
}


def norms(a, b):
    d = np.abs(a - b)
    return d.mean(), np.sqrt((d ** 2).mean()), d.max()


def load_dat(path):
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, comments="#")


def compare_pair(baseline_path, variant_path):
    A = load_dat(baseline_path)
    B = load_dat(variant_path)
    if A is None or B is None:
        return None, "missing file"
    if A.shape != B.shape:
        return None, f"shape mismatch {A.shape} vs {B.shape}"
    ncols = A.shape[1]
    if ncols not in VARS_BY_NCOLS:
        return None, f"unrecognised column count {ncols}"
    rows = []
    for vname, c in VARS_BY_NCOLS[ncols]:
        l1, l2, linf = norms(A[:, c], B[:, c])
        rows.append((vname, l1, l2, linf))
    return rows, None


def fmt(x):
    if x == 0.0:
        return "0"
    return f"{x:.2e}"


def print_full_tables(results, cases, builds, baseline):
    for c in cases:
        case_rows = [(b, v, l1, l2, li) for (cc, b, v), (l1, l2, li) in results.items() if cc == c]
        if not case_rows:
            continue
        print()
        print(f"=== {c}  vs baseline={baseline}  (FP64) ===")
        print()
        header = f"{'build':<14} {'var':<5}  {'L1':>11}  {'L2':>11}  {'Linf':>11}"
        print(header)
        print("-" * len(header))
        for b in builds:
            entries = [(v, l1, l2, li) for (bb, v, l1, l2, li) in case_rows if bb == b]
            if not entries:
                continue
            for i, (v, l1, l2, li) in enumerate(entries):
                prefix = b if i == 0 else ""
                print(f"{prefix:<14} {v:<5}  {fmt(l1):>11}  {fmt(l2):>11}  {fmt(li):>11}")
            print()


def print_summary(results, cases, builds, baseline):
    # Build the (case, var) row index in deterministic order
    row_keys = []
    seen = set()
    for c in cases:
        for (cc, b, v) in results:
            if cc == c and (c, v) not in seen:
                seen.add((c, v))
                row_keys.append((c, v))
    print()
    print(f"=== L_inf  vs baseline={baseline}  (FP64) ===")
    print()
    header = f"{'case':<18} {'var':<5}"
    for b in builds:
        header += f"  {b:>13}"
    print(header)
    print("-" * len(header))
    for c, v in row_keys:
        row = f"{c:<18} {v:<5}"
        for b in builds:
            if (c, b, v) in results:
                row += f"  {fmt(results[(c, b, v)][2]):>13}"
            else:
                row += f"  {'--':>13}"
        print(row)


def print_tex(results, cases, builds, baseline):
    print()
    print(f"% LaTeX summary: L_inf relative to baseline {baseline}")
    align = "l l " + "c " * len(builds)
    print(f"\\begin{{tabular}}{{{align.strip()}}}")
    print(r"\toprule")
    hdr = "Case & Var"
    for b in builds:
        hdr += " & " + b.replace("_", r"\_")
    print(hdr + r" \\")
    print(r"\midrule")
    seen = set()
    for c in cases:
        for (cc, b0, v) in results:
            if cc != c or (c, v) in seen:
                continue
            seen.add((c, v))
            row = f"{c.replace('_', ' ')} & ${v}$"
            for b in builds:
                row += " & " + (fmt(results[(c, b, v)][2]) if (c, b, v) in results else "--")
            print(row + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/euler/compiler/fp64")
    ap.add_argument("--baseline", default="cpu_O2")
    ap.add_argument("--builds", default="cpu_O3,cpu_Ofast,gpu_default,gpu_fastmath")
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    ap.add_argument("--summary", action="store_true",
                    help="Compact L_inf-only table (rows=case+var, cols=builds)")
    ap.add_argument("--tex", action="store_true", help="Also emit LaTeX table")
    args = ap.parse_args()

    cases  = [c for c in args.cases.split(",")  if c.strip()]
    builds = [b for b in args.builds.split(",") if b.strip()]

    results = {}     # (case, build, var) -> (L1, L2, Linf)
    skipped = []

    for c in cases:
        if c not in CASE_FILE:
            skipped.append(f"unknown case: {c}")
            continue
        base_path = os.path.join(args.root, args.baseline, CASE_FILE[c])
        for b in builds:
            var_path = os.path.join(args.root, b, CASE_FILE[c])
            rows, err = compare_pair(base_path, var_path)
            if rows is None:
                skipped.append(f"{c}/{b}: {err}  ({var_path})")
                continue
            for vname, l1, l2, linf in rows:
                results[(c, b, vname)] = (l1, l2, linf)

    if skipped:
        print("Skipped:", file=sys.stderr)
        for s in skipped:
            print(f"  {s}", file=sys.stderr)
        print(file=sys.stderr)

    if not results:
        print("No comparable files found.", file=sys.stderr)
        sys.exit(1)

    if args.summary:
        print_summary(results, cases, builds, args.baseline)
    else:
        print_full_tables(results, cases, builds, args.baseline)

    if args.tex:
        print_tex(results, cases, builds, args.baseline)


if __name__ == "__main__":
    main()
