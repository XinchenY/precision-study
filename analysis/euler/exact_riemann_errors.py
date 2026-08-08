#!/usr/bin/env python3
"""
exact_riemann_errors.py  —  Density (and u, p) errors vs exact Riemann solution

Implements exact Riemann solver (Toro 2009, Ch.4) via Newton-Raphson iteration.
Reads numerical results from results/euler/ieee/fp64/ and computes L1/L2/Linf errors.

Usage:
  python3 analysis/euler/exact_riemann_errors.py [--solver hllc|hll|both]

Output:
  Terminal table + analysis/euler/exact_riemann_norms.txt
"""

import numpy as np
import os
import sys

GAMMA = 1.4

# ── Exact Riemann solver ────────────────────────────────────────────────────

def _sound(rho, p):
    return np.sqrt(GAMMA * p / rho)

def _fK(p, rhoK, pK, aK):
    """Pressure function for left (K=L) or right (K=R) state."""
    AK = 2.0 / ((GAMMA + 1.0) * rhoK)
    BK = (GAMMA - 1.0) / (GAMMA + 1.0) * pK
    if p > pK:                           # shock
        return (p - pK) * np.sqrt(AK / (p + BK))
    else:                                # rarefaction
        exp = (GAMMA - 1.0) / (2.0 * GAMMA)
        return (2.0 * aK / (GAMMA - 1.0)) * ((p / pK)**exp - 1.0)

def _fK_deriv(p, rhoK, pK, aK):
    AK = 2.0 / ((GAMMA + 1.0) * rhoK)
    BK = (GAMMA - 1.0) / (GAMMA + 1.0) * pK
    if p > pK:                           # shock
        return np.sqrt(AK / (p + BK)) * (1.0 - (p - pK) / (2.0 * (p + BK)))
    else:                                # rarefaction
        exp = -(GAMMA + 1.0) / (2.0 * GAMMA)
        return (p / pK)**exp / (rhoK * aK)

def _find_pstar(rhoL, uL, pL, rhoR, uR, pR, tol=1e-12, max_iter=100):
    aL = _sound(rhoL, pL)
    aR = _sound(rhoR, pR)
    # initial guess: primitive variable Riemann (PVRS, Toro eq 9.28)
    p0 = max(0.0, 0.5*(pL + pR) - 0.125*(uR - uL)*(rhoL + rhoR)*(aL + aR))
    p = max(p0, 1e-10)
    for _ in range(max_iter):
        f  = _fK(p, rhoL, pL, aL) + _fK(p, rhoR, pR, aR) + (uR - uL)
        df = _fK_deriv(p, rhoL, pL, aL) + _fK_deriv(p, rhoR, pR, aR)
        dp = -f / df
        p = max(p + dp, 1e-10)
        if abs(dp) / (0.5 * (p + max(p - dp, 1e-10))) < tol:
            break
    ustar = 0.5*(uL + uR) + 0.5*(_fK(p, rhoR, pR, aR) - _fK(p, rhoL, pL, aL))
    return p, ustar

def _sample(p_star, u_star, rhoL, uL, pL, rhoR, uR, pR, S):
    """Sample exact solution at speed S = (x - x0)/t."""
    aL = _sound(rhoL, pL)
    aR = _sound(rhoR, pR)

    if S <= u_star:                      # left of contact
        if p_star > pL:                  # left shock
            ML = np.sqrt(((GAMMA+1)/(2*GAMMA)) * p_star/pL + (GAMMA-1)/(2*GAMMA))
            SL = uL - aL * ML
            if S <= SL:
                return rhoL, uL, pL
            else:
                coeff = rhoL * (SL - uL) / (SL - u_star)
                return coeff, u_star, p_star
        else:                            # left rarefaction
            SHL = uL - aL                # head
            a_star_L = aL * (p_star/pL)**((GAMMA-1)/(2*GAMMA))
            STL = u_star - a_star_L     # tail
            if S <= SHL:
                return rhoL, uL, pL
            elif S <= STL:
                exp1 = 2.0/(GAMMA+1)
                exp2 = (GAMMA-1)/(GAMMA+1)
                rho_fan = rhoL * (2/(GAMMA+1) + (GAMMA-1)/((GAMMA+1)*aL)*(uL-S))**(2/(GAMMA-1))
                u_fan   = 2/(GAMMA+1) * (aL + (GAMMA-1)/2*uL + S)
                p_fan   = pL * (2/(GAMMA+1) + (GAMMA-1)/((GAMMA+1)*aL)*(uL-S))**(2*GAMMA/(GAMMA-1))
                return rho_fan, u_fan, p_fan
            else:
                rho_star_L = rhoL * (p_star/pL)**(1/GAMMA)
                return rho_star_L, u_star, p_star
    else:                                # right of contact
        if p_star > pR:                  # right shock
            MR = np.sqrt(((GAMMA+1)/(2*GAMMA)) * p_star/pR + (GAMMA-1)/(2*GAMMA))
            SR = uR + aR * MR
            if S >= SR:
                return rhoR, uR, pR
            else:
                coeff = rhoR * (SR - uR) / (SR - u_star)
                return coeff, u_star, p_star
        else:                            # right rarefaction
            SHR = uR + aR
            a_star_R = aR * (p_star/pR)**((GAMMA-1)/(2*GAMMA))
            STR = u_star + a_star_R
            if S >= SHR:
                return rhoR, uR, pR
            elif S >= STR:
                rho_fan = rhoR * (2/(GAMMA+1) - (GAMMA-1)/((GAMMA+1)*aR)*(uR-S))**(2/(GAMMA-1))
                u_fan   = 2/(GAMMA+1) * (-aR + (GAMMA-1)/2*uR + S)
                p_fan   = pR * (2/(GAMMA+1) - (GAMMA-1)/((GAMMA+1)*aR)*(uR-S))**(2*GAMMA/(GAMMA-1))
                return rho_fan, u_fan, p_fan
            else:
                rho_star_R = rhoR * (p_star/pR)**(1/GAMMA)
                return rho_star_R, u_star, p_star

def exact_solution(rhoL, uL, pL, rhoR, uR, pR, x0, T, x_arr):
    """Evaluate exact solution at all points in x_arr at time T."""
    p_star, u_star = _find_pstar(rhoL, uL, pL, rhoR, uR, pR)
    n = len(x_arr)
    rho_ex = np.empty(n)
    u_ex   = np.empty(n)
    p_ex   = np.empty(n)
    for i, xi in enumerate(x_arr):
        S = (xi - x0) / T
        r, u, p = _sample(p_star, u_star, rhoL, uL, pL, rhoR, uR, pR, S)
        rho_ex[i] = r
        u_ex[i]   = u
        p_ex[i]   = p
    return rho_ex, u_ex, p_ex

# ── Test cases (Toro Table 4.1) ─────────────────────────────────────────────

TESTS = [
    # (name_short, tid, rhoL, uL, pL, rhoR, uR, pR, T_end, x0, label)
    ("Sod",        1, 1.0,     0.0,      1.0,    0.125,   0.0,      0.1,    0.25,  0.5, "1 (Sod)"),
    ("123",        2, 1.0,    -2.0,      0.4,    1.0,     2.0,      0.4,    0.15,  0.5, "2 (123 problem)"),
    ("LBlast",     3, 1.0,     0.0,   1000.0,    1.0,     0.0,      0.01,   0.012, 0.5, "3 (left blast)"),
    ("RBlast",     4, 1.0,     0.0,      0.01,   1.0,     0.0,    100.0,    0.035, 0.5, "4 (right blast)"),
    ("TwoShocks",  5, 5.99924, 19.5975, 460.894, 5.99242,-6.19633, 46.095,  0.035, 0.4, "5 (two shocks)"),
]

# ── Main ─────────────────────────────────────────────────────────────────────

SOLVER_FLAG = "hllc"
for i, a in enumerate(sys.argv):
    if a == "--solver" and i + 1 < len(sys.argv):
        SOLVER_FLAG = sys.argv[i + 1]

SOLVERS = ["hllc", "hll"] if SOLVER_FLAG == "both" else [SOLVER_FLAG]
PRECS   = [("fp64", "results/euler/ieee/fp64"), ("fp32", "results/euler/ieee/fp32")]

def norms(a, b):
    d    = np.abs(a - b)
    L1   = d.mean()
    L2   = np.sqrt((d**2).mean())
    Linf = d.max()
    return L1, L2, Linf

lines = []
lines.append("Density errors vs exact Riemann solution  (N=200 IEEE, Lovelace)")
lines.append("  L1 = mean|err|,  L2 = RMS,  L∞ = max|err|")
lines.append("")
lines.append(f"{'Test':<20} {'Solver':>6}  {'Prec':>5}  {'L1':>12}  {'L2':>12}  {'L∞':>12}")
lines.append("-" * 78)

for (short, tid, rhoL, uL, pL, rhoR, uR, pR, T_end, x0, label) in TESTS:
    for solver in SOLVERS:
        first_row = True
        for (prec, datadir) in PRECS:
            fpath = f"{datadir}/toro{tid}_{solver}_{prec}_200.dat"
            if not os.path.exists(fpath):
                print(f"  Missing: {fpath}"); continue
            A = np.loadtxt(fpath)
            x_num   = A[:, 0]
            rho_num = A[:, 1]

            rho_ex, _, _ = exact_solution(rhoL, uL, pL, rhoR, uR, pR,
                                          x0, T_end, x_num)
            l1, l2, linf = norms(rho_num, rho_ex)

            tag_test   = label   if first_row else ""
            tag_solver = solver  if first_row else ""
            lines.append(f"  {tag_test:<20} {tag_solver:>6}  {prec:>5}  {l1:12.3e}  {l2:12.3e}  {linf:12.3e}")
            first_row = False

output = "\n".join(lines)
print(output)

os.makedirs("analysis", exist_ok=True)
with open("analysis/euler/exact_riemann_norms.txt", "w") as f:
    f.write(output + "\n")
print("\nSaved: analysis/euler/exact_riemann_norms.txt")
