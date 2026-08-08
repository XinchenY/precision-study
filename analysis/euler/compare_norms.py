#!/usr/bin/env python3
"""
compare_norms.py  —  计算两个 .dat 文件之间的 L1 / L2 / L∞ 范数
用法:
  python3 analysis/euler/compare_norms.py  fileA.dat  fileB.dat
  python3 analysis/euler/compare_norms.py  fileA.dat  fileB.dat  --col rho
列说明: col 0=x, 1=rho, 2=u, 3=p  (默认对 rho/u/p 全部计算)
"""

import numpy as np
import sys

def norms(a, b):
    d = np.abs(a - b)
    N = len(d)
    L1   = d.mean()                    # (1/N) Σ|eᵢ|
    L2   = np.sqrt((d**2).mean())      # √((1/N) Σeᵢ²)  = RMS
    Linf = d.max()                     # max|eᵢ|  (与 N 无关，不变)
    return L1, L2, Linf

def main():
    if len(sys.argv) < 3:
        print("Usage: compare_norms.py fileA fileB [--col rho|u|p]")
        sys.exit(1)

    fA, fB = sys.argv[1], sys.argv[2]
    A = np.loadtxt(fA)
    B = np.loadtxt(fB)

    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}")
        sys.exit(1)

    allow_mismatch = "--allow-grid-mismatch" in sys.argv
    if not np.allclose(A[:, 0], B[:, 0], atol=0, rtol=0):
        if allow_mismatch:
            # cross-precision compare: x written with different digits, but row order matches
            if np.allclose(A[:, 0], B[:, 0], rtol=1e-5, atol=0):
                print(f"  x-grid: OK (within rtol=1e-5; cross-precision)")
            else:
                print("ERROR: x coordinates differ by more than rtol=1e-5 — likely different grids!")
                sys.exit(1)
        else:
            print("WARNING: x coordinates differ — files may not be from the same grid!")
            print("         (pass --allow-grid-mismatch if comparing across precisions)")
            sys.exit(1)
    else:
        print(f"  x-grid: OK (identical)")

    col_filter = None
    if "--col" in sys.argv:
        col_filter = sys.argv[sys.argv.index("--col") + 1]

    col_map = {"rho": 1, "u": 2, "p": 3}
    cols = col_map if col_filter is None else {col_filter: col_map[col_filter]}

    print(f"Comparing: {fA}")
    print(f"      vs : {fB}")
    print(f"  N points: {A.shape[0]}")
    print(f"{'var':>5}  {'L1':>14}  {'L2':>14}  {'L∞':>14}")
    print("-" * 55)
    for name, c in cols.items():
        l1, l2, linf = norms(A[:, c], B[:, c])
        print(f"  {name:>3}  {l1:14.6e}  {l2:14.6e}  {linf:14.6e}")

if __name__ == "__main__":
    main()
