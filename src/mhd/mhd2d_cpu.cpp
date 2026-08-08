/**
 * ============================================================================
 *  2D ideal-MHD Orszag-Tang vortex (CPU, FP64 by default)
 * ============================================================================
 *
 *  This file is the first two-dimensional MHD validation driver.  It is kept
 *  separate from the Euler solvers and from the 1D Brio-Wu driver.
 *
 *  Default case:
 *    Orszag-Tang vortex on [0,1] x [0,1]
 *    gamma = 5/3
 *    periodic boundaries
 *    MUSCL reconstruction + SSPRK2 time integration
 *    HLL flux by default, HLLD optional
 *
 *  Usage:
 *    g++ -O2 -std=c++14 -I include src/mhd2d_cpu.cpp -o bin/mhd2d_cpu
 *    ./bin/mhd2d_cpu
 *    ./bin/mhd2d_cpu 128 128 0.5 0.3 muscl hll
 *    ./bin/mhd2d_cpu 128 128 0.2 0.25 muscl hlld
 *    ./bin/mhd2d_cpu 128 128 0.5 0.3 first hll
 *
 *  Output:
 *    ${OUTDIR:-results/mhd/orszag_tang}/orszag_tang_{order}_{solver}_fp64_NxNy.dat
 *
 *  Columns:
 *    x y rho vx vy vz p Bx By Bz E divB
 * ============================================================================
 */

#ifndef MHD_GAMMA_VALUE
#define MHD_GAMMA_VALUE (5.0 / 3.0)
#endif

#include "mhd/mhd_common.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

typedef std::array<Real, MHD_NVAR> MHDState;

static int index2D(int i, int j, int nx_total) {
  return j * nx_total + i;
}

static void copyStateToArray(const MHDState& state, Real U[MHD_NVAR]) {
  for (int k = 0; k < MHD_NVAR; k++) U[k] = state[k];
}

static MHDState arrayToState(const Real U[MHD_NVAR]) {
  MHDState state;
  for (int k = 0; k < MHD_NVAR; k++) state[k] = U[k];
  return state;
}

static void stateToPrimitive(const MHDState& state, Real W[MHD_NVAR]) {
  Real U[MHD_NVAR];
  copyStateToArray(state, U);
  mhdConsToPrim(U, W);
}

static MHDState primitiveArrayToState(const Real W[MHD_NVAR]) {
  Real U[MHD_NVAR];
  mhdPrimToCons(W, U);
  return arrayToState(U);
}

static bool primitiveIsUsable(const Real W[MHD_NVAR]) {
  if (W[MHD_PR_RHO] <= MHD_RHO_FLOOR || W[MHD_PR_P] <= MHD_P_FLOOR) return false;
  for (int k = 0; k < MHD_NVAR; k++) {
    if (!std::isfinite((double)W[k])) return false;
  }
  return true;
}

static bool stateIsUsable(const MHDState& state) {
  Real U[MHD_NVAR], W[MHD_NVAR];
  copyStateToArray(state, U);
  if (U[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  mhdConsToPrim(U, W);
  return primitiveIsUsable(W);
}

static void applyPeriodicBoundary(std::vector<MHDState>& U,
                                  int nx,
                                  int ny,
                                  int ghost) {
  const int nx_total = nx + 2 * ghost;
  const int ny_total = ny + 2 * ghost;

  for (int j = ghost; j < ny + ghost; j++) {
    for (int g = 0; g < ghost; g++) {
      U[index2D(g, j, nx_total)] = U[index2D(nx + g, j, nx_total)];
      U[index2D(nx + ghost + g, j, nx_total)] = U[index2D(ghost + g, j, nx_total)];
    }
  }

  for (int i = 0; i < nx_total; i++) {
    for (int g = 0; g < ghost; g++) {
      U[index2D(i, g, nx_total)] = U[index2D(i, ny + g, nx_total)];
      U[index2D(i, ny + ghost + g, nx_total)] = U[index2D(i, ghost + g, nx_total)];
    }
  }

  (void)ny_total;
}

static void reconstructPrimitive(const Real Wm[MHD_NVAR],
                                  const Real W0[MHD_NVAR],
                                  const Real Wp[MHD_NVAR],
                                  Real Wleft[MHD_NVAR],
                                  Real Wright[MHD_NVAR]) {
  for (int k = 0; k < MHD_NVAR; k++) {
    Real slope = mhdMinmod(W0[k] - Wm[k], Wp[k] - W0[k]);
    Wleft[k] = W0[k] - (Real)0.5 * slope;
    Wright[k] = W0[k] + (Real)0.5 * slope;
  }

  if (!primitiveIsUsable(Wleft) || !primitiveIsUsable(Wright)) {
    for (int k = 0; k < MHD_NVAR; k++) {
      Wleft[k] = W0[k];
      Wright[k] = W0[k];
    }
  }
}

static void xFaceStates(const std::vector<MHDState>& U,
                        int i,
                        int j,
                        int nx_total,
                        bool use_muscl,
                        Real UL[MHD_NVAR],
                        Real UR[MHD_NVAR]) {
  if (!use_muscl) {
    copyStateToArray(U[index2D(i, j, nx_total)], UL);
    copyStateToArray(U[index2D(i + 1, j, nx_total)], UR);
    return;
  }

  Real Wim[MHD_NVAR], Wi[MHD_NVAR], Wip[MHD_NVAR];
  Real Wj[MHD_NVAR], Wjp[MHD_NVAR], Wjpp[MHD_NVAR];
  Real Wleft_i[MHD_NVAR], Wright_i[MHD_NVAR];
  Real Wleft_ip[MHD_NVAR], Wright_ip[MHD_NVAR];

  stateToPrimitive(U[index2D(i - 1, j, nx_total)], Wim);
  stateToPrimitive(U[index2D(i, j, nx_total)], Wi);
  stateToPrimitive(U[index2D(i + 1, j, nx_total)], Wip);
  stateToPrimitive(U[index2D(i, j, nx_total)], Wj);
  stateToPrimitive(U[index2D(i + 1, j, nx_total)], Wjp);
  stateToPrimitive(U[index2D(i + 2, j, nx_total)], Wjpp);

  reconstructPrimitive(Wim, Wi, Wip, Wleft_i, Wright_i);
  reconstructPrimitive(Wj, Wjp, Wjpp, Wleft_ip, Wright_ip);

  Real Uleft[MHD_NVAR], Uright[MHD_NVAR];
  mhdPrimToCons(Wright_i, Uleft);
  mhdPrimToCons(Wleft_ip, Uright);
  for (int k = 0; k < MHD_NVAR; k++) {
    UL[k] = Uleft[k];
    UR[k] = Uright[k];
  }
}

static void yFaceStates(const std::vector<MHDState>& U,
                        int i,
                        int j,
                        int nx_total,
                        bool use_muscl,
                        Real UL[MHD_NVAR],
                        Real UR[MHD_NVAR]) {
  if (!use_muscl) {
    copyStateToArray(U[index2D(i, j, nx_total)], UL);
    copyStateToArray(U[index2D(i, j + 1, nx_total)], UR);
    return;
  }

  Real Wjm[MHD_NVAR], Wj[MHD_NVAR], Wjp[MHD_NVAR];
  Real Wk[MHD_NVAR], Wkp[MHD_NVAR], Wkpp[MHD_NVAR];
  Real Wleft_j[MHD_NVAR], Wright_j[MHD_NVAR];
  Real Wleft_jp[MHD_NVAR], Wright_jp[MHD_NVAR];

  stateToPrimitive(U[index2D(i, j - 1, nx_total)], Wjm);
  stateToPrimitive(U[index2D(i, j, nx_total)], Wj);
  stateToPrimitive(U[index2D(i, j + 1, nx_total)], Wjp);
  stateToPrimitive(U[index2D(i, j, nx_total)], Wk);
  stateToPrimitive(U[index2D(i, j + 1, nx_total)], Wkp);
  stateToPrimitive(U[index2D(i, j + 2, nx_total)], Wkpp);

  reconstructPrimitive(Wjm, Wj, Wjp, Wleft_j, Wright_j);
  reconstructPrimitive(Wk, Wkp, Wkpp, Wleft_jp, Wright_jp);

  Real Uleft[MHD_NVAR], Uright[MHD_NVAR];
  mhdPrimToCons(Wright_j, Uleft);
  mhdPrimToCons(Wleft_jp, Uright);
  for (int k = 0; k < MHD_NVAR; k++) {
    UL[k] = Uleft[k];
    UR[k] = Uright[k];
  }
}

static void computeRhs(std::vector<MHDState> U,
                       std::vector<MHDState>& rhs,
                       int nx,
                       int ny,
                       int ghost,
                       Real dx,
                       Real dy,
                       bool use_muscl,
                       bool use_hlld) {
  const int nx_total = nx + 2 * ghost;
  const int ny_total = ny + 2 * ghost;
  const int total = nx_total * ny_total;

  applyPeriodicBoundary(U, nx, ny, ghost);

  std::vector<MHDState> Fx(total);
  std::vector<MHDState> Gy(total);
  rhs.assign(total, MHDState());

  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost - 1; i < nx + ghost; i++) {
      Real UL[MHD_NVAR], UR[MHD_NVAR], F[MHD_NVAR];
      xFaceStates(U, i, j, nx_total, use_muscl, UL, UR);
      if (use_hlld) {
        mhdHlldFluxX(UL, UR, F);
      } else {
        mhdHllFluxX(UL, UR, F);
      }
      for (int k = 0; k < MHD_NVAR; k++) Fx[index2D(i, j, nx_total)][k] = F[k];
    }
  }

  for (int j = ghost - 1; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real UL[MHD_NVAR], UR[MHD_NVAR], G[MHD_NVAR];
      yFaceStates(U, i, j, nx_total, use_muscl, UL, UR);
      if (use_hlld) {
        mhdHlldFluxY(UL, UR, G);
      } else {
        mhdHllFluxY(UL, UR, G);
      }
      for (int k = 0; k < MHD_NVAR; k++) Gy[index2D(i, j, nx_total)][k] = G[k];
    }
  }

  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      int id = index2D(i, j, nx_total);
      int im = index2D(i - 1, j, nx_total);
      int jm = index2D(i, j - 1, nx_total);
      for (int k = 0; k < MHD_NVAR; k++) {
        rhs[id][k] = - (Fx[id][k] - Fx[im][k]) / dx
                    - (Gy[id][k] - Gy[jm][k]) / dy;
      }
    }
  }

  (void)ny_total;
}

static Real maxSignalRate(const std::vector<MHDState>& U,
                          int nx,
                          int ny,
                          int ghost,
                          Real dx,
                          Real dy) {
  const int nx_total = nx + 2 * ghost;
  Real max_rate = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real W[MHD_NVAR];
      stateToPrimitive(U[index2D(i, j, nx_total)], W);
      Real rate = (mhdAbs(W[MHD_PR_VX]) + mhdFastSpeedX(W)) / dx
                + (mhdAbs(W[MHD_PR_VY]) + mhdFastSpeedY(W)) / dy;
      max_rate = mhdMax(max_rate, rate);
    }
  }
  return max_rate;
}

static bool allInteriorStatesUsable(const std::vector<MHDState>& U,
                                    int nx,
                                    int ny,
                                    int ghost,
                                    int& bad_i,
                                    int& bad_j) {
  const int nx_total = nx + 2 * ghost;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      if (!stateIsUsable(U[index2D(i, j, nx_total)])) {
        bad_i = i - ghost;
        bad_j = j - ghost;
        return false;
      }
    }
  }
  return true;
}

static Real divergenceB(const std::vector<MHDState>& U,
                        int i,
                        int j,
                        int nx_total,
                        Real dx,
                        Real dy) {
  Real bxp = U[index2D(i + 1, j, nx_total)][MHD_BX];
  Real bxm = U[index2D(i - 1, j, nx_total)][MHD_BX];
  Real byp = U[index2D(i, j + 1, nx_total)][MHD_BY];
  Real bym = U[index2D(i, j - 1, nx_total)][MHD_BY];
  return (bxp - bxm) / ((Real)2.0 * dx) + (byp - bym) / ((Real)2.0 * dy);
}

int main(int argc, char* argv[]) {
  int nx = 128;
  int ny = 128;
  Real t_end = (Real)0.5;
  Real cfl = (Real)0.3;
  std::string order = "muscl";
  std::string solver = "hll";

  if (argc > 1) nx = std::atoi(argv[1]);
  if (argc > 2) ny = std::atoi(argv[2]);
  if (argc > 3) t_end = (Real)std::atof(argv[3]);
  if (argc > 4) cfl = (Real)std::atof(argv[4]);
  if (argc > 5) order = argv[5];
  if (argc > 6) solver = argv[6];

  if (nx <= 0 || ny <= 0) {
    std::cerr << "Error: Nx and Ny must be positive." << std::endl;
    return 1;
  }
  if (t_end <= (Real)0.0) {
    std::cerr << "Error: t_end must be positive." << std::endl;
    return 1;
  }
  if (cfl <= (Real)0.0 || cfl >= (Real)1.0) {
    std::cerr << "Error: CFL must be in (0, 1)." << std::endl;
    return 1;
  }
  if (order != "muscl" && order != "first") {
    std::cerr << "Error: order must be 'muscl' or 'first'." << std::endl;
    return 1;
  }
  if (solver != "hll" && solver != "hlld") {
    std::cerr << "Error: solver must be 'hll' or 'hlld'." << std::endl;
    return 1;
  }

  const bool use_muscl = (order == "muscl");
  const bool use_hlld = (solver == "hlld");
  const int ghost = 2;
  const int nx_total = nx + 2 * ghost;
  const int ny_total = ny + 2 * ghost;
  const int total = nx_total * ny_total;
  const Real x_min = (Real)0.0;
  const Real x_max = (Real)1.0;
  const Real y_min = (Real)0.0;
  const Real y_max = (Real)1.0;
  const Real dx = (x_max - x_min) / (Real)nx;
  const Real dy = (y_max - y_min) / (Real)ny;
  const Real pi = std::acos((Real)-1.0);
  const Real two_pi = (Real)2.0 * pi;
  const Real inv_sqrt_4pi = (Real)1.0 / std::sqrt((Real)4.0 * pi);

  std::cout << "===== 2D ideal MHD Orszag-Tang vortex =====" << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL") << std::endl;
  std::cout << "Reconstruction/time: "
            << (use_muscl ? "MUSCL + SSPRK2" : "first-order + SSPRK2")
            << std::endl;
  std::cout << "Precision: " << (int)(sizeof(Real) * 8) << " bit"
            << (sizeof(Real) == 8 ? " (fp64)" : " (fp32)") << std::endl;
  std::cout << "gamma = " << MHD_GAMMA << std::endl;
  std::cout << "Nx x Ny = " << nx << " x " << ny
            << ", CFL = " << cfl << ", t_end = " << t_end << std::endl;

  std::vector<MHDState> U(total);
  std::vector<MHDState> rhs(total), rhs2(total), U1(total);

  for (int j = 0; j < ny_total; j++) {
    for (int i = 0; i < nx_total; i++) {
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;

      Real W[MHD_NVAR];
      W[MHD_PR_RHO] = (Real)25.0 / ((Real)36.0 * pi);
      W[MHD_PR_VX]  = -std::sin(two_pi * y);
      W[MHD_PR_VY]  =  std::sin(two_pi * x);
      W[MHD_PR_VZ]  = (Real)0.0;
      W[MHD_PR_P]   = (Real)5.0 / ((Real)12.0 * pi);
      W[MHD_PR_BX]  = -std::sin(two_pi * y) * inv_sqrt_4pi;
      W[MHD_PR_BY]  =  std::sin((Real)2.0 * two_pi * x) * inv_sqrt_4pi;
      W[MHD_PR_BZ]  = (Real)0.0;

      U[index2D(i, j, nx_total)] = primitiveArrayToState(W);
    }
  }
  applyPeriodicBoundary(U, nx, ny, ghost);

  Real t = (Real)0.0;
  int step = 0;
  while (t < t_end) {
    Real max_rate = maxSignalRate(U, nx, ny, ghost, dx, dy);
    if (max_rate <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal rate." << std::endl;
      return 2;
    }

    Real dt = cfl / max_rate;
    if (t + dt > t_end) dt = t_end - t;

    computeRhs(U, rhs, nx, ny, ghost, dx, dy, use_muscl, use_hlld);
    for (int j = ghost; j < ny + ghost; j++) {
      for (int i = ghost; i < nx + ghost; i++) {
        int id = index2D(i, j, nx_total);
        for (int k = 0; k < MHD_NVAR; k++) U1[id][k] = U[id][k] + dt * rhs[id][k];
      }
    }
    applyPeriodicBoundary(U1, nx, ny, ghost);

    int bad_i = -1, bad_j = -1;
    if (!allInteriorStatesUsable(U1, nx, ny, ghost, bad_i, bad_j)) {
      std::cerr << "Error: non-physical RK stage at step " << step
                << ", cell (" << bad_i << ", " << bad_j << ")" << std::endl;
      return 3;
    }

    computeRhs(U1, rhs2, nx, ny, ghost, dx, dy, use_muscl, use_hlld);
    for (int j = ghost; j < ny + ghost; j++) {
      for (int i = ghost; i < nx + ghost; i++) {
        int id = index2D(i, j, nx_total);
        for (int k = 0; k < MHD_NVAR; k++) {
          U[id][k] = (Real)0.5 * U[id][k]
                   + (Real)0.5 * (U1[id][k] + dt * rhs2[id][k]);
        }
      }
    }
    applyPeriodicBoundary(U, nx, ny, ghost);

    if (!allInteriorStatesUsable(U, nx, ny, ghost, bad_i, bad_j)) {
      std::cerr << "Error: non-physical state at step " << step
                << ", cell (" << bad_i << ", " << bad_j << ")" << std::endl;
      return 4;
    }

    t += dt;
    step++;
  }

  Real div_l1 = (Real)0.0;
  Real div_linf = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real divb = divergenceB(U, i, j, nx_total, dx, dy);
      div_l1 += mhdAbs(divb);
      div_linf = mhdMax(div_linf, mhdAbs(divb));
    }
  }
  div_l1 /= (Real)(nx * ny);

  std::cout << "Done. Steps: " << step << ", final t = " << t << std::endl;
  std::cout << "divB L1 = " << div_l1 << ", divB Linf = " << div_linf << std::endl;

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/orszag_tang");
  std::system(("mkdir -p " + out_dir).c_str());

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::ostringstream fname;
  fname << out_dir << "/orszag_tang_" << order << "_" << solver << "_"
        << prec_name << "_N" << nx << "x" << ny << ".dat";

  std::ofstream out(fname.str());
  if (!out) {
    std::cerr << "Error: failed to open output file: " << fname.str() << std::endl;
    return 5;
  }

  out << "# Orszag-Tang ideal MHD\n";
  out << "# nx " << nx << " ny " << ny << " t " << t << " gamma " << MHD_GAMMA
      << " cfl " << cfl << " order " << order << " solver " << solver << "\n";
  out << "# columns: x y rho vx vy vz p Bx By Bz E divB\n";
  out << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10);

  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      int id = index2D(i, j, nx_total);
      Real Ui[MHD_NVAR], Wi[MHD_NVAR];
      copyStateToArray(U[id], Ui);
      mhdConsToPrim(Ui, Wi);
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;
      Real divb = divergenceB(U, i, j, nx_total, dx, dy);

      out << x << " "
          << y << " "
          << Wi[MHD_PR_RHO] << " "
          << Wi[MHD_PR_VX] << " "
          << Wi[MHD_PR_VY] << " "
          << Wi[MHD_PR_VZ] << " "
          << Wi[MHD_PR_P] << " "
          << Wi[MHD_PR_BX] << " "
          << Wi[MHD_PR_BY] << " "
          << Wi[MHD_PR_BZ] << " "
          << Ui[MHD_E] << " "
          << divb << "\n";
    }
  }

  std::cout << "Saved: " << fname.str() << std::endl;
  return 0;
}
