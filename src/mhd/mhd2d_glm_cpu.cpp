/**
 * ============================================================================
 *  2D ideal-MHD Orszag-Tang vortex with mixed GLM cleaning
 * ============================================================================
 *
 *  This is the cell-centred 9-variable GLM driver for the two-dimensional MHD
 *  validation run:
 *
 *    U = (rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz, psi).
 *
 *  The physical MHD flux uses HLLD by default.  The normal magnetic field and
 *  psi are coupled through the mixed-GLM subsystem, and psi is damped using
 *  c_p^2 / c_h = alpha.
 *
 *  Usage:
 *    g++ -O2 -std=c++14 -I include src/mhd2d_glm_cpu.cpp -o bin/mhd2d_glm_cpu
 *    ./bin/mhd2d_glm_cpu
 *    ./bin/mhd2d_glm_cpu 128 128 0.5 0.25 muscl hlld 0.18
 *
 *  Output columns:
 *    x y rho vx vy vz p Bx By Bz E divB psi
 * ============================================================================
 */

#ifndef MHD_GAMMA_VALUE
#define MHD_GAMMA_VALUE (5.0 / 3.0)
#endif

#include "mhd/mhd_glm_common.hpp"

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
#include <chrono>

typedef std::array<Real, MHD_GLM_NVAR> GlmState;

// HLLD->HLL fallback 统计: 按方向和原因计数, 结束时写入 stdout 和 .dat 头。
// Tallies HLLD->HLL fallbacks by direction and reason; reported at the end
// of the run and recorded in the .dat header for reproducibility.
struct HlldFallbackStats {
  unsigned long long x_evaluations;
  unsigned long long y_evaluations;
  unsigned long long degenerate;
  unsigned long long nonphysical;

  HlldFallbackStats()
      : x_evaluations(0), y_evaluations(0), degenerate(0), nonphysical(0) {}

  void recordX(int status) {
    x_evaluations++;
    recordReason(status);
  }

  void recordY(int status) {
    y_evaluations++;
    recordReason(status);
  }

  void recordReason(int status) {
    if (status == MHD_HLLD_FALLBACK_DEGENERATE) degenerate++;
    if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) nonphysical++;
  }

  unsigned long long evaluations() const { return x_evaluations + y_evaluations; }
  unsigned long long fallbacks() const { return degenerate + nonphysical; }
};

static int index2D(int i, int j, int nx_total) {
  return j * nx_total + i;
}

static void copyStateToArray(const GlmState& state, Real U[MHD_GLM_NVAR]) {
  for (int k = 0; k < MHD_GLM_NVAR; k++) U[k] = state[k];
}

static GlmState arrayToState(const Real U[MHD_GLM_NVAR]) {
  GlmState state;
  for (int k = 0; k < MHD_GLM_NVAR; k++) state[k] = U[k];
  return state;
}

static void stateToPrimitive(const GlmState& state, Real P[MHD_GLM_NVAR]) {
  Real U[MHD_GLM_NVAR], W[MHD_NVAR], psi;
  copyStateToArray(state, U);
  mhdGlmConsToPrim(U, W, psi);
  for (int k = 0; k < MHD_NVAR; k++) P[k] = W[k];
  P[MHD_GLM_PSI] = psi;
}

static GlmState primitiveToState(const Real P[MHD_GLM_NVAR]) {
  Real W[MHD_NVAR], U[MHD_GLM_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) W[k] = P[k];
  mhdGlmPrimToCons(W, P[MHD_GLM_PSI], U);
  return arrayToState(U);
}

static bool primitiveIsUsable(const Real P[MHD_GLM_NVAR]) {
  if (P[MHD_PR_RHO] <= MHD_RHO_FLOOR || P[MHD_PR_P] <= MHD_P_FLOOR) return false;
  for (int k = 0; k < MHD_GLM_NVAR; k++) {
    if (!std::isfinite((double)P[k])) return false;
  }
  return true;
}

static bool stateIsUsable(const GlmState& state) {
  Real P[MHD_GLM_NVAR];
  if (state[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  stateToPrimitive(state, P);
  return primitiveIsUsable(P);
}

static void applyPeriodicBoundary(std::vector<GlmState>& U,
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

static void reconstructPrimitive(const Real Pm[MHD_GLM_NVAR],
                                  const Real P0[MHD_GLM_NVAR],
                                  const Real Pp[MHD_GLM_NVAR],
                                  Real Pleft[MHD_GLM_NVAR],
                                  Real Pright[MHD_GLM_NVAR]) {
  for (int k = 0; k < MHD_GLM_NVAR; k++) {
    Real slope = mhdMinmod(P0[k] - Pm[k], Pp[k] - P0[k]);
    Pleft[k] = P0[k] - (Real)0.5 * slope;
    Pright[k] = P0[k] + (Real)0.5 * slope;
  }

  if (!primitiveIsUsable(Pleft) || !primitiveIsUsable(Pright)) {
    for (int k = 0; k < MHD_GLM_NVAR; k++) {
      Pleft[k] = P0[k];
      Pright[k] = P0[k];
    }
  }
}

static void xFaceStates(const std::vector<GlmState>& U,
                        int i,
                        int j,
                        int nx_total,
                        bool use_muscl,
                        Real UL[MHD_GLM_NVAR],
                        Real UR[MHD_GLM_NVAR]) {
  if (!use_muscl) {
    copyStateToArray(U[index2D(i, j, nx_total)], UL);
    copyStateToArray(U[index2D(i + 1, j, nx_total)], UR);
    return;
  }

  Real Pim[MHD_GLM_NVAR], Pi[MHD_GLM_NVAR], Pip[MHD_GLM_NVAR];
  Real Pj[MHD_GLM_NVAR], Pjp[MHD_GLM_NVAR], Pjpp[MHD_GLM_NVAR];
  Real Pleft_i[MHD_GLM_NVAR], Pright_i[MHD_GLM_NVAR];
  Real Pleft_ip[MHD_GLM_NVAR], Pright_ip[MHD_GLM_NVAR];

  stateToPrimitive(U[index2D(i - 1, j, nx_total)], Pim);
  stateToPrimitive(U[index2D(i, j, nx_total)], Pi);
  stateToPrimitive(U[index2D(i + 1, j, nx_total)], Pip);
  stateToPrimitive(U[index2D(i, j, nx_total)], Pj);
  stateToPrimitive(U[index2D(i + 1, j, nx_total)], Pjp);
  stateToPrimitive(U[index2D(i + 2, j, nx_total)], Pjpp);

  reconstructPrimitive(Pim, Pi, Pip, Pleft_i, Pright_i);
  reconstructPrimitive(Pj, Pjp, Pjpp, Pleft_ip, Pright_ip);

  GlmState left = primitiveToState(Pright_i);
  GlmState right = primitiveToState(Pleft_ip);
  copyStateToArray(left, UL);
  copyStateToArray(right, UR);
}

static void yFaceStates(const std::vector<GlmState>& U,
                        int i,
                        int j,
                        int nx_total,
                        bool use_muscl,
                        Real UL[MHD_GLM_NVAR],
                        Real UR[MHD_GLM_NVAR]) {
  if (!use_muscl) {
    copyStateToArray(U[index2D(i, j, nx_total)], UL);
    copyStateToArray(U[index2D(i, j + 1, nx_total)], UR);
    return;
  }

  Real Pjm[MHD_GLM_NVAR], Pj[MHD_GLM_NVAR], Pjp[MHD_GLM_NVAR];
  Real Pk[MHD_GLM_NVAR], Pkp[MHD_GLM_NVAR], Pkpp[MHD_GLM_NVAR];
  Real Pleft_j[MHD_GLM_NVAR], Pright_j[MHD_GLM_NVAR];
  Real Pleft_jp[MHD_GLM_NVAR], Pright_jp[MHD_GLM_NVAR];

  stateToPrimitive(U[index2D(i, j - 1, nx_total)], Pjm);
  stateToPrimitive(U[index2D(i, j, nx_total)], Pj);
  stateToPrimitive(U[index2D(i, j + 1, nx_total)], Pjp);
  stateToPrimitive(U[index2D(i, j, nx_total)], Pk);
  stateToPrimitive(U[index2D(i, j + 1, nx_total)], Pkp);
  stateToPrimitive(U[index2D(i, j + 2, nx_total)], Pkpp);

  reconstructPrimitive(Pjm, Pj, Pjp, Pleft_j, Pright_j);
  reconstructPrimitive(Pk, Pkp, Pkpp, Pleft_jp, Pright_jp);

  GlmState left = primitiveToState(Pright_j);
  GlmState right = primitiveToState(Pleft_jp);
  copyStateToArray(left, UL);
  copyStateToArray(right, UR);
}

static void computeRhs(std::vector<GlmState> U,
                       std::vector<GlmState>& rhs,
                       int nx,
                       int ny,
                       int ghost,
                       Real dx,
                       Real dy,
                       bool use_muscl,
                       bool use_hlld,
                       Real ch,
                       Real alpha,
                       HlldFallbackStats* fallback_stats) {
  const int nx_total = nx + 2 * ghost;
  const int ny_total = ny + 2 * ghost;
  const int total = nx_total * ny_total;
  const Real damping_rate = ch / mhdMax(alpha, (Real)1.0e-12);

  applyPeriodicBoundary(U, nx, ny, ghost);
  std::vector<GlmState> Fx(total);
  std::vector<GlmState> Gy(total);
  rhs.assign(total, GlmState());

  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost - 1; i < nx + ghost; i++) {
      Real UL[MHD_GLM_NVAR], UR[MHD_GLM_NVAR], F[MHD_GLM_NVAR];
      xFaceStates(U, i, j, nx_total, use_muscl, UL, UR);
      int status = mhdGlmFluxX(UL, UR, ch, use_hlld, F);
      if (use_hlld && fallback_stats) fallback_stats->recordX(status);
      for (int k = 0; k < MHD_GLM_NVAR; k++) Fx[index2D(i, j, nx_total)][k] = F[k];
    }
  }

  for (int j = ghost - 1; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real UL[MHD_GLM_NVAR], UR[MHD_GLM_NVAR], G[MHD_GLM_NVAR];
      yFaceStates(U, i, j, nx_total, use_muscl, UL, UR);
      int status = mhdGlmFluxY(UL, UR, ch, use_hlld, G);
      if (use_hlld && fallback_stats) fallback_stats->recordY(status);
      for (int k = 0; k < MHD_GLM_NVAR; k++) Gy[index2D(i, j, nx_total)][k] = G[k];
    }
  }

  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      int id = index2D(i, j, nx_total);
      int im = index2D(i - 1, j, nx_total);
      int jm = index2D(i, j - 1, nx_total);
      for (int k = 0; k < MHD_GLM_NVAR; k++) {
        rhs[id][k] = - (Fx[id][k] - Fx[im][k]) / dx
                    - (Gy[id][k] - Gy[jm][k]) / dy;
      }
      rhs[id][MHD_GLM_PSI] -= damping_rate * U[id][MHD_GLM_PSI];
    }
  }

  (void)ny_total;
}

static Real maxPhysicalSignalSpeed(const std::vector<GlmState>& U,
                                   int nx,
                                   int ny,
                                   int ghost) {
  const int nx_total = nx + 2 * ghost;
  Real max_speed = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real P[MHD_GLM_NVAR];
      stateToPrimitive(U[index2D(i, j, nx_total)], P);
      Real sx = mhdAbs(P[MHD_PR_VX]) + mhdFastSpeedX(P);
      Real sy = mhdAbs(P[MHD_PR_VY]) + mhdFastSpeedY(P);
      max_speed = mhdMax(max_speed, mhdMax(sx, sy));
    }
  }
  return max_speed;
}

static Real maxSignalRate(const std::vector<GlmState>& U,
                          int nx,
                          int ny,
                          int ghost,
                          Real dx,
                          Real dy,
                          Real ch) {
  const int nx_total = nx + 2 * ghost;
  Real max_rate = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real P[MHD_GLM_NVAR];
      stateToPrimitive(U[index2D(i, j, nx_total)], P);
      Real sx = mhdMax(mhdAbs(P[MHD_PR_VX]) + mhdFastSpeedX(P), ch);
      Real sy = mhdMax(mhdAbs(P[MHD_PR_VY]) + mhdFastSpeedY(P), ch);
      max_rate = mhdMax(max_rate, sx / dx + sy / dy);
    }
  }
  return max_rate;
}

static bool allInteriorStatesUsable(const std::vector<GlmState>& U,
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

static Real divergenceB(const std::vector<GlmState>& U,
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

// 崩溃诊断: 打印失败格点的完整状态 (能量分解、divB、位置), 便于定位
// update-level 正定性丢失。只在 run 失败时调用, 不影响正常运行的输出。
// Failure diagnostics: dump the offending cell's full state (energy split,
// divB, location).  Called only when a run aborts.
static void printFailureDiagnostics(const char* label,
                                    const std::vector<GlmState>& U,
                                    int nx,
                                    int ny,
                                    int ghost,
                                    int bad_i,
                                    int bad_j,
                                    Real dx,
                                    Real dy,
                                    Real t,
                                    Real dt,
                                    int step,
                                    Real ch) {
  const int nx_total = nx + 2 * ghost;
  const int i = bad_i + ghost;
  const int j = bad_j + ghost;
  const int id = index2D(i, j, nx_total);

  Real P[MHD_GLM_NVAR];
  stateToPrimitive(U[id], P);
  Real rho = U[id][MHD_RHO];
  Real rho_safe = mhdMax(rho, MHD_RHO_FLOOR);
  Real vx = U[id][MHD_MX] / rho_safe;
  Real vy = U[id][MHD_MY] / rho_safe;
  Real vz = U[id][MHD_MZ] / rho_safe;
  Real bx = U[id][MHD_BX];
  Real by = U[id][MHD_BY];
  Real bz = U[id][MHD_BZ];
  Real e = U[id][MHD_E];
  Real kinetic = (Real)0.5 * rho_safe * (vx*vx + vy*vy + vz*vz);
  Real magnetic = (Real)0.5 * (bx*bx + by*by + bz*bz);
  Real internal = e - kinetic - magnetic;
  Real p = (MHD_GAMMA - (Real)1.0) * internal;
  Real divb = divergenceB(U, i, j, nx_total, dx, dy);
  Real x = ((Real)bad_i + (Real)0.5) * dx;
  Real y = ((Real)bad_j + (Real)0.5) * dy;

  std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10);
  std::cerr << "Failure diagnostics (" << label << ")\n";
  std::cerr << "  step=" << step << " t=" << t << " dt=" << dt
            << " ch=" << ch << "\n";
  std::cerr << "  cell=(" << bad_i << ", " << bad_j << ")"
            << " x=" << x << " y=" << y << "\n";
  std::cerr << "  rho_raw=" << rho << " rho_prim=" << P[MHD_PR_RHO]
            << " p=" << p << " p_prim=" << P[MHD_PR_P] << "\n";
  std::cerr << "  vx=" << vx << " vy=" << vy << " vz=" << vz << "\n";
  std::cerr << "  Bx=" << bx << " By=" << by << " Bz=" << bz
            << " psi=" << U[id][MHD_GLM_PSI] << "\n";
  std::cerr << "  E=" << e << " kinetic=" << kinetic
            << " magnetic=" << magnetic << " internal=" << internal << "\n";
  std::cerr << "  divB=" << divb << "\n";
}

int main(int argc, char* argv[]) {
  int nx = 128;
  int ny = 128;
  Real t_end = (Real)0.5;
  Real cfl = (Real)0.25;
  std::string order = "muscl";
  std::string solver = "hlld";
  Real alpha = (Real)0.18;

  if (argc > 1) nx = std::atoi(argv[1]);
  if (argc > 2) ny = std::atoi(argv[2]);
  if (argc > 3) t_end = (Real)std::atof(argv[3]);
  if (argc > 4) cfl = (Real)std::atof(argv[4]);
  if (argc > 5) order = argv[5];
  if (argc > 6) solver = argv[6];
  if (argc > 7) alpha = (Real)std::atof(argv[7]);

  if (nx <= 0 || ny <= 0 || t_end <= (Real)0.0) {
    std::cerr << "Error: invalid grid or final time." << std::endl;
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
  if (alpha <= (Real)0.0) {
    std::cerr << "Error: alpha must be positive." << std::endl;
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

  std::cout << "===== 2D ideal MHD Orszag-Tang vortex with mixed GLM =====" << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL") << std::endl;
  std::cout << "Reconstruction/time: "
            << (use_muscl ? "MUSCL + SSPRK2" : "first-order + SSPRK2")
            << std::endl;
  std::cout << "Precision: " << (int)(sizeof(Real) * 8) << " bit"
            << (sizeof(Real) == 8 ? " (fp64)" : " (fp32)") << std::endl;
  std::cout << "gamma = " << MHD_GAMMA << ", alpha = " << alpha << std::endl;
  std::cout << "Nx x Ny = " << nx << " x " << ny
            << ", CFL = " << cfl << ", t_end = " << t_end << std::endl;

  std::vector<GlmState> U(total);
  std::vector<GlmState> U1(total);
  std::vector<GlmState> rhs(total), rhs2(total);

  for (int j = 0; j < ny_total; j++) {
    for (int i = 0; i < nx_total; i++) {
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;

      Real P[MHD_GLM_NVAR];
      P[MHD_PR_RHO] = (Real)25.0 / ((Real)36.0 * pi);
      P[MHD_PR_VX]  = -std::sin(two_pi * y);
      P[MHD_PR_VY]  =  std::sin(two_pi * x);
      P[MHD_PR_VZ]  = (Real)0.0;
      P[MHD_PR_P]   = (Real)5.0 / ((Real)12.0 * pi);
      P[MHD_PR_BX]  = -std::sin(two_pi * y) * inv_sqrt_4pi;
      P[MHD_PR_BY]  =  std::sin((Real)2.0 * two_pi * x) * inv_sqrt_4pi;
      P[MHD_PR_BZ]  = (Real)0.0;
      P[MHD_GLM_PSI] = (Real)0.0;

      U[index2D(i, j, nx_total)] = primitiveToState(P);
    }
  }
  applyPeriodicBoundary(U, nx, ny, ghost);

  Real t = (Real)0.0;
  int step = 0;
  Real last_ch = (Real)0.0;
  HlldFallbackStats hlld_stats;

  auto wall_t0 = std::chrono::steady_clock::now();
  while (t < t_end) {
    Real physical_speed = maxPhysicalSignalSpeed(U, nx, ny, ghost);
    Real ch = mhdMax(physical_speed, (Real)1.0e-12);
    last_ch = ch;
    Real max_rate = maxSignalRate(U, nx, ny, ghost, dx, dy, ch);
    if (max_rate <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal rate." << std::endl;
      return 2;
    }

    Real dt = cfl / max_rate;
    if (t + dt > t_end) dt = t_end - t;

    computeRhs(U, rhs, nx, ny, ghost, dx, dy, use_muscl, use_hlld, ch, alpha,
               &hlld_stats);
    for (int j = ghost; j < ny + ghost; j++) {
      for (int i = ghost; i < nx + ghost; i++) {
        int id = index2D(i, j, nx_total);
        for (int k = 0; k < MHD_GLM_NVAR; k++) {
          U1[id][k] = U[id][k] + dt * rhs[id][k];
        }
      }
    }
    applyPeriodicBoundary(U1, nx, ny, ghost);

    int bad_i = -1, bad_j = -1;
    if (!allInteriorStatesUsable(U1, nx, ny, ghost, bad_i, bad_j)) {
      std::cerr << "Error: non-physical RK stage at step " << step
                << ", cell (" << bad_i << ", " << bad_j << ")" << std::endl;
      printFailureDiagnostics("RK stage", U1, nx, ny, ghost,
                              bad_i, bad_j, dx, dy, t, dt, step, ch);
      return 3;
    }

    computeRhs(U1, rhs2, nx, ny, ghost, dx, dy, use_muscl, use_hlld, ch, alpha,
               &hlld_stats);
    for (int j = ghost; j < ny + ghost; j++) {
      for (int i = ghost; i < nx + ghost; i++) {
        int id = index2D(i, j, nx_total);
        for (int k = 0; k < MHD_GLM_NVAR; k++) {
          U[id][k] = (Real)0.5 * U[id][k]
                   + (Real)0.5 * (U1[id][k] + dt * rhs2[id][k]);
        }
      }
    }
    applyPeriodicBoundary(U, nx, ny, ghost);

    if (!allInteriorStatesUsable(U, nx, ny, ghost, bad_i, bad_j)) {
      std::cerr << "Error: non-physical state at step " << step
                << ", cell (" << bad_i << ", " << bad_j << ")" << std::endl;
      printFailureDiagnostics("full RK state", U, nx, ny, ghost,
                              bad_i, bad_j, dx, dy, t, dt, step, ch);
      return 4;
    }

    t += dt;
    step++;
  }
  auto wall_t1 = std::chrono::steady_clock::now();
  double loop_wall = std::chrono::duration<double>(wall_t1 - wall_t0).count();

  Real div_l1 = (Real)0.0;
  Real div_linf = (Real)0.0;
  Real psi_linf = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      int id = index2D(i, j, nx_total);
      Real divb = divergenceB(U, i, j, nx_total, dx, dy);
      div_l1 += mhdAbs(divb);
      div_linf = mhdMax(div_linf, mhdAbs(divb));
      psi_linf = mhdMax(psi_linf, mhdAbs(U[id][MHD_GLM_PSI]));
    }
  }
  div_l1 /= (Real)(nx * ny);

  std::cout << "Done. Steps: " << step << ", final t = " << t << std::endl;
  std::cout << "Time loop wall time = " << loop_wall << " s ("
            << ((double)nx * ny * step / loop_wall / 1.0e6)
            << " Mcell-steps/s)" << std::endl;
  std::cout << "c_h(last) = " << last_ch << ", alpha = " << alpha << std::endl;
  std::cout << "divB L1 = " << div_l1 << ", divB Linf = " << div_linf
            << ", psi Linf = " << psi_linf << std::endl;
  if (use_hlld) {
    double fallback_fraction = hlld_stats.evaluations() > 0
        ? (double)hlld_stats.fallbacks() / (double)hlld_stats.evaluations()
        : 0.0;
    std::cout << "HLLD flux evaluations = " << hlld_stats.evaluations()
              << " (x=" << hlld_stats.x_evaluations
              << ", y=" << hlld_stats.y_evaluations << ")" << std::endl;
    std::cout << "HLLD relative fallback tolerance = "
              << std::scientific << (double)mhdHlldTolerance()
              << std::defaultfloat << std::endl;
    std::cout << "HLLD->HLL fallbacks = " << hlld_stats.fallbacks()
              << " (degenerate=" << hlld_stats.degenerate
              << ", nonphysical=" << hlld_stats.nonphysical
              << "), fraction = " << std::scientific << fallback_fraction
              << std::defaultfloat << std::endl;
  }

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/orszag_tang_glm");
  std::system(("mkdir -p " + out_dir).c_str());

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::ostringstream fname;
  fname << out_dir << "/orszag_tang_glm_" << order << "_" << solver << "_"
        << prec_name << "_N" << nx << "x" << ny << ".dat";

  std::ofstream out(fname.str());
  if (!out) {
    std::cerr << "Error: failed to open output file: " << fname.str() << std::endl;
    return 5;
  }

  out << "# Orszag-Tang ideal MHD with mixed GLM\n";
  out << "# nx " << nx << " ny " << ny << " t " << t << " gamma " << MHD_GAMMA
      << " cfl " << cfl << " order " << order << " solver " << solver
      << " alpha " << alpha << " ch_last " << last_ch << "\n";
  if (use_hlld) {
    double fallback_fraction = hlld_stats.evaluations() > 0
        ? (double)hlld_stats.fallbacks() / (double)hlld_stats.evaluations()
        : 0.0;
    out << "# hlld_evaluations " << hlld_stats.evaluations()
        << " hlld_tolerance " << mhdHlldTolerance()
        << " hlld_x_evaluations " << hlld_stats.x_evaluations
        << " hlld_y_evaluations " << hlld_stats.y_evaluations
        << " hlld_fallbacks " << hlld_stats.fallbacks()
        << " hlld_degenerate " << hlld_stats.degenerate
        << " hlld_nonphysical " << hlld_stats.nonphysical
        << " hlld_fallback_fraction " << fallback_fraction << "\n";
  }
  out << "# columns: x y rho vx vy vz p Bx By Bz E divB psi\n";
  out << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10);

  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      int id = index2D(i, j, nx_total);
      Real U9[MHD_GLM_NVAR], W[MHD_NVAR], psi;
      copyStateToArray(U[id], U9);
      mhdGlmConsToPrim(U9, W, psi);
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;
      Real divb = divergenceB(U, i, j, nx_total, dx, dy);

      out << x << " "
          << y << " "
          << W[MHD_PR_RHO] << " "
          << W[MHD_PR_VX] << " "
          << W[MHD_PR_VY] << " "
          << W[MHD_PR_VZ] << " "
          << W[MHD_PR_P] << " "
          << W[MHD_PR_BX] << " "
          << W[MHD_PR_BY] << " "
          << W[MHD_PR_BZ] << " "
          << U9[MHD_E] << " "
          << divb << " "
          << psi << "\n";
    }
  }

  std::cout << "Saved: " << fname.str() << std::endl;
  return 0;
}
