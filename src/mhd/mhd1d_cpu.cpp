/**
 * ============================================================================
 *   1D ideal MHD Brio-Wu shock tube (CPU, FP64 by default)
 *   1D 理想 MHD Brio-Wu 激波管 (CPU，默认 FP64)
 * ============================================================================
 *
 *  English:
 *  This is the first MHD development file.  It does not modify the Euler
 *  solver and it does not call the Euler HLL/HLLC flux functions.  It can be
 *  compiled and run with only this file plus include/mhd_common.hpp.
 *
 *  中文:
 *  这是第一个 MHD 开发文件。它不修改 Euler solver，也不调用 Euler 的 HLL/HLLC
 *  通量函数。当前版本只依赖本文件和 include/mhd_common.hpp 就可以编译运行。
 *
 *  What is reused from the Euler code style / 从 Euler 保持一致的部分:
 *    - finite-volume update loop / 有限体积更新循环
 *    - transmissive ghost cells / 透射边界 ghost cell
 *    - CFL time-step logic / CFL 时间步
 *    - OUTDIR output convention / OUTDIR 输出习惯
 *    - Real precision typedef from config.hpp / 复用 config.hpp 里的 Real 精度类型
 *
 *  What is new for MHD / MHD 重新写的部分:
 *    - 8 primitive/conserved variables / 8 个原始变量和守恒变量
 *    - ideal-MHD energy with magnetic energy / 含磁能的 MHD 总能量
 *    - ideal-MHD flux and fast magnetosonic wave speed / MHD 通量和快磁声速
 *    - HLL flux over the MHD state vector / 作用在 MHD 8 变量上的 HLL 通量
 *
 *  Brio-Wu initial condition / Brio-Wu 初始条件:
 *    gamma = 2
 *    x in [0, 1], discontinuity at x = 0.5
 *    left:  rho=1,     p=1,   v=(0,0,0), B=(0.75,  1, 0)
 *    right: rho=0.125, p=0.1, v=(0,0,0), B=(0.75, -1, 0)
 *
 *  Usage / 用法:
 *    g++ -O2 -std=c++14 -I include src/mhd1d_cpu.cpp -o bin/mhd1d_cpu
 *    ./bin/mhd1d_cpu                             # N=400, t_end=0.1, CFL=0.4
 *    ./bin/mhd1d_cpu 800 0.1 0.4 muscl hll      # MUSCL-Hancock + HLL
 *    ./bin/mhd1d_cpu 800 0.1 0.4 muscl hlld     # MUSCL-Hancock + HLLD
 *    ./bin/mhd1d_cpu 800 0.1 0.4 first hll      # first-order HLL fallback
 *
 *  Output / 输出:
 *    ${OUTDIR:-results/mhd/brio_wu}/brio_wu_{order}_{solver}_fp64_N{N}.dat
 *
 *  Columns / 输出列:
 *    x rho vx vy vz p Bx By Bz E
 * ============================================================================
 */

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

// HLLD->HLL fallback 统计: 记录触发次数与原因, 结束时写入 stdout 和 .dat 头。
// Tallies HLLD->HLL fallbacks by reason; reported at the end of the run and
// recorded in the .dat header for reproducibility.
struct HlldFallbackStats {
  unsigned long long evaluations;
  unsigned long long degenerate;
  unsigned long long nonphysical;

  HlldFallbackStats() : evaluations(0), degenerate(0), nonphysical(0) {}

  void record(int status) {
    evaluations++;
    if (status == MHD_HLLD_FALLBACK_DEGENERATE) degenerate++;
    if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) nonphysical++;
  }

  unsigned long long fallbacks() const { return degenerate + nonphysical; }
};

// Primitive Brio-Wu state used only for clean initial-condition setup.
// 只用于清楚地写 Brio-Wu 初始条件；真正更新的是 conserved state。
struct BrioWuState {
  Real rho;
  Real vx, vy, vz;
  Real p;
  Real bx, by, bz;
};

static void makePrimitive(const BrioWuState& s, Real W[MHD_NVAR]) {
  W[MHD_PR_RHO] = s.rho;
  W[MHD_PR_VX]  = s.vx;
  W[MHD_PR_VY]  = s.vy;
  W[MHD_PR_VZ]  = s.vz;
  W[MHD_PR_P]   = s.p;
  W[MHD_PR_BX]  = s.bx;
  W[MHD_PR_BY]  = s.by;
  W[MHD_PR_BZ]  = s.bz;
}

static MHDState primitiveToState(const BrioWuState& s) {
  Real W[MHD_NVAR], U[MHD_NVAR];
  makePrimitive(s, W);
  mhdPrimToCons(W, U);

  MHDState state;
  for (int k = 0; k < MHD_NVAR; k++) state[k] = U[k];
  return state;
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

// Same boundary style as the Euler test programs: copy the nearest interior
// state into ghost cells.  This is a transmissive/outflow boundary.
// 和 Euler 测试程序一样的边界处理：把最近的内部格点复制到 ghost cell。
static void applyTransmissiveBoundary(std::vector<MHDState>& U, int nx, int ghost) {
  int total = nx + 2 * ghost;
  for (int g = 0; g < ghost; g++) {
    int left_inner = ghost;
    int right_inner = total - 1 - ghost;
    int right_ghost = total - 1 - g;

    U[g] = U[left_inner];
    U[right_ghost] = U[right_inner];
  }
}

// Minimal runtime safety check for the first MHD baseline.
// 第一版的基本安全检查：rho 和 p 必须为正，所有变量必须是有限数。
static bool stateIsUsable(const MHDState& state) {
  Real U[MHD_NVAR], W[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) U[k] = state[k];

  if (U[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  mhdConsToPrim(U, W);

  if (W[MHD_PR_RHO] <= (Real)0.0 || W[MHD_PR_P] <= (Real)0.0) return false;
  for (int k = 0; k < MHD_NVAR; k++) {
    if (!std::isfinite((double)U[k]) || !std::isfinite((double)W[k])) return false;
  }
  return true;
}

// MUSCL reconstruction in primitive variables.
// The Bx slope is forced to zero because Bx is constant in 1D Brio-Wu.
// 用原始变量做 MUSCL 重构。Brio-Wu 的 Bx 在 1D 中应保持常数，因此不重构 Bx。
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

  Wleft[MHD_PR_BX] = W0[MHD_PR_BX];
  Wright[MHD_PR_BX] = W0[MHD_PR_BX];

  if (!primitiveIsUsable(Wleft) || !primitiveIsUsable(Wright)) {
    for (int k = 0; k < MHD_NVAR; k++) {
      Wleft[k] = W0[k];
      Wright[k] = W0[k];
    }
  }
}

// Hancock predictor: evolve both reconstructed faces of one cell by half a time
// step using the flux difference across that cell.
// Hancock 预测步：用该 cell 左右重构面通量差，把两个面都推进半个时间步。
static void hancockPredictCell(const Real Wleft[MHD_NVAR],
                               const Real Wright[MHD_NVAR],
                               Real factor,
                               MHDState& UleftHalf,
                               MHDState& UrightHalf) {
  Real UL[MHD_NVAR], UR[MHD_NVAR];
  Real FL[MHD_NVAR], FR[MHD_NVAR];
  mhdPrimToCons(Wleft, UL);
  mhdPrimToCons(Wright, UR);
  mhdFluxX(UL, FL);
  mhdFluxX(UR, FR);

  Real ULH[MHD_NVAR], URH[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) {
    Real correction = factor * (FR[k] - FL[k]);
    ULH[k] = UL[k] - correction;
    URH[k] = UR[k] - correction;
  }

  Real WLH[MHD_NVAR], WRH[MHD_NVAR];
  mhdConsToPrim(ULH, WLH);
  mhdConsToPrim(URH, WRH);

  if (ULH[MHD_RHO] <= MHD_RHO_FLOOR || !primitiveIsUsable(WLH)) {
    for (int k = 0; k < MHD_NVAR; k++) ULH[k] = UL[k];
  }
  if (URH[MHD_RHO] <= MHD_RHO_FLOOR || !primitiveIsUsable(WRH)) {
    for (int k = 0; k < MHD_NVAR; k++) URH[k] = UR[k];
  }

  UleftHalf = arrayToState(ULH);
  UrightHalf = arrayToState(URH);
}

int main(int argc, char* argv[]) {
  int nx = 400;
  Real t_end = (Real)0.1;
  Real cfl = (Real)0.4;
  std::string order = "muscl";
  std::string solver = "hll";

  if (argc > 1) nx = std::atoi(argv[1]);
  if (argc > 2) t_end = (Real)std::atof(argv[2]);
  if (argc > 3) cfl = (Real)std::atof(argv[3]);
  if (argc > 4) order = argv[4];
  if (argc > 5) solver = argv[5];

  if (nx <= 0) {
    std::cerr << "Error: N must be positive." << std::endl;
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
  // ghost=2 matches the Euler code and leaves room for future MUSCL-Hancock.
  // 当前一阶 HLL 只需要较少 ghost cell，但保留 2 层以便后续加 MUSCL-Hancock。
  const int total = nx + 2 * ghost;
  const Real x_min = (Real)0.0;
  const Real x_max = (Real)1.0;
  const Real x0 = (Real)0.5;
  const Real dx = (x_max - x_min) / (Real)nx;

  const BrioWuState left = {
    (Real)1.0,
    (Real)0.0, (Real)0.0, (Real)0.0,
    (Real)1.0,
    (Real)0.75, (Real)1.0, (Real)0.0
  };

  const BrioWuState right = {
    (Real)0.125,
    (Real)0.0, (Real)0.0, (Real)0.0,
    (Real)0.1,
    (Real)0.75, (Real)-1.0, (Real)0.0
  };

  std::cout << "===== 1D ideal MHD Brio-Wu shock tube =====" << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL") << std::endl;
  std::cout << "Reconstruction: " << (use_muscl ? "MUSCL-Hancock" : "first-order")
            << std::endl;
  std::cout << "Precision: " << (int)(sizeof(Real) * 8) << " bit"
            << (sizeof(Real) == 8 ? " (fp64)" : " (fp32)") << std::endl;
  std::cout << "gamma = " << MHD_GAMMA << std::endl;
  std::cout << "N = " << nx << ", dx = " << dx << ", CFL = " << cfl << std::endl;
  std::cout << "T_end = " << t_end << ", x0 = " << x0 << std::endl;
  std::cout << "Left:  rho=1 p=1 Bx=0.75 By=1 Bz=0" << std::endl;
  std::cout << "Right: rho=0.125 p=0.1 Bx=0.75 By=-1 Bz=0" << std::endl;

  std::vector<MHDState> U(total);
  std::vector<MHDState> F(total);
  std::vector<MHDState> UleftHalf(total), UrightHalf(total);

  // Initialise conserved variables from the Brio-Wu primitive states.
  // 从 Brio-Wu 原始变量初始化守恒变量；这里会调用 MHD 总能量公式。
  MHDState UL = primitiveToState(left);
  MHDState UR = primitiveToState(right);

  for (int i = 0; i < total; i++) {
    Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
    U[i] = (x < x0) ? UL : UR;
  }

  Real t = (Real)0.0;
  int step = 0;
  HlldFallbackStats hlld_stats;

  while (t < t_end) {
    applyTransmissiveBoundary(U, nx, ghost);

    // CFL step: max signal speed is |vx| + fast magnetosonic speed.
    // CFL 时间步：最大信号速度为 |vx| + 快磁声速。
    Real max_speed = (Real)0.0;
    for (int i = ghost; i < nx + ghost; i++) {
      Real Ui[MHD_NVAR], Wi[MHD_NVAR];
      for (int k = 0; k < MHD_NVAR; k++) Ui[k] = U[i][k];
      mhdConsToPrim(Ui, Wi);
      Real speed = mhdAbs(Wi[MHD_PR_VX]) + mhdFastSpeedX(Wi);
      max_speed = mhdMax(max_speed, speed);
    }

    if (max_speed <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal speed." << std::endl;
      return 2;
    }

    Real dt = cfl * dx / max_speed;
    if (t + dt > t_end) dt = t_end - t;

    if (use_muscl) {
      Real factor = (Real)0.5 * dt / dx;
      for (int i = ghost - 1; i <= nx + ghost; i++) {
        Real Wm[MHD_NVAR], W0[MHD_NVAR], Wp[MHD_NVAR];
        Real Wleft[MHD_NVAR], Wright[MHD_NVAR];
        stateToPrimitive(U[i - 1], Wm);
        stateToPrimitive(U[i], W0);
        stateToPrimitive(U[i + 1], Wp);
        reconstructPrimitive(Wm, W0, Wp, Wleft, Wright);
        hancockPredictCell(Wleft, Wright, factor, UleftHalf[i], UrightHalf[i]);
      }
    }

    // Interface fluxes.  MUSCL-Hancock uses the predicted right face of cell i
    // and predicted left face of cell i+1.  The first-order fallback uses cell
    // averages directly.
    // 界面通量：MUSCL-Hancock 使用左 cell 的右半步面值和右 cell 的左半步面值；
    // first-order fallback 则直接使用 cell average。
    for (int i = ghost - 1; i < nx + ghost; i++) {
      Real UiL[MHD_NVAR], UiR[MHD_NVAR], Fi[MHD_NVAR];
      const MHDState& leftState = use_muscl ? UrightHalf[i] : U[i];
      const MHDState& rightState = use_muscl ? UleftHalf[i + 1] : U[i + 1];
      copyStateToArray(leftState, UiL);
      copyStateToArray(rightState, UiR);
      if (use_hlld) {
        hlld_stats.record(mhdHlldFluxX(UiL, UiR, Fi));
      } else {
        mhdHllFluxX(UiL, UiR, Fi);
      }
      for (int k = 0; k < MHD_NVAR; k++) F[i][k] = Fi[k];
    }

    // Conservative finite-volume update:
    // U_i^{n+1} = U_i^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    // 守恒型有限体积更新，结构上和 Euler solver 一致。
    Real ratio = dt / dx;
    for (int i = ghost; i < nx + ghost; i++) {
      for (int k = 0; k < MHD_NVAR; k++) {
        U[i][k] -= ratio * (F[i][k] - F[i - 1][k]);
      }
    }

    for (int i = ghost; i < nx + ghost; i++) {
      if (!stateIsUsable(U[i])) {
        std::cerr << "Error: non-physical state at step " << step
                  << ", cell " << (i - ghost) << std::endl;
        return 3;
      }
    }

    t += dt;
    step++;
  }

  std::cout << "Done. Steps: " << step << ", final t = " << t << std::endl;
  if (use_hlld) {
    double fallback_fraction = hlld_stats.evaluations > 0
        ? (double)hlld_stats.fallbacks() / (double)hlld_stats.evaluations
        : 0.0;
    std::cout << "HLLD flux evaluations = " << hlld_stats.evaluations << std::endl;
    std::cout << "HLLD relative fallback tolerance = "
              << std::scientific << (double)mhdHlldTolerance()
              << std::defaultfloat << std::endl;
    std::cout << "HLLD->HLL fallbacks = " << hlld_stats.fallbacks()
              << " (degenerate=" << hlld_stats.degenerate
              << ", nonphysical=" << hlld_stats.nonphysical
              << "), fraction = " << std::scientific << fallback_fraction
              << std::defaultfloat << std::endl;
  }

  // Keep the same OUTDIR convention as the Euler programs, but use a separate
  // MHD result folder by default to avoid mixing Euler and MHD outputs.
  // 保持 Euler 程序的 OUTDIR 习惯，但默认输出到单独的 MHD 文件夹，避免混淆。
  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/brio_wu");
  std::system(("mkdir -p " + out_dir).c_str());

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::ostringstream fname;
  fname << out_dir << "/brio_wu_" << order << "_" << solver << "_"
        << prec_name << "_N" << nx << ".dat";

  std::ofstream out(fname.str());
  if (!out) {
    std::cerr << "Error: failed to open output file: " << fname.str() << std::endl;
    return 4;
  }

  out << "# Brio-Wu ideal MHD\n";
  out << "# nx " << nx << " t " << t << " gamma " << MHD_GAMMA
      << " cfl " << cfl << " order " << order << " solver " << solver << "\n";
  if (use_hlld) {
    double fallback_fraction = hlld_stats.evaluations > 0
        ? (double)hlld_stats.fallbacks() / (double)hlld_stats.evaluations
        : 0.0;
    out << "# hlld_evaluations " << hlld_stats.evaluations
        << " hlld_tolerance " << mhdHlldTolerance()
        << " hlld_fallbacks " << hlld_stats.fallbacks()
        << " hlld_degenerate " << hlld_stats.degenerate
        << " hlld_nonphysical " << hlld_stats.nonphysical
        << " hlld_fallback_fraction " << fallback_fraction << "\n";
  }
  out << "# columns: x rho vx vy vz p Bx By Bz E\n";
  out << std::scientific << std::setprecision(std::numeric_limits<Real>::max_digits10);
  for (int i = ghost; i < nx + ghost; i++) {
    Real Ui[MHD_NVAR], Wi[MHD_NVAR];
    for (int k = 0; k < MHD_NVAR; k++) Ui[k] = U[i][k];
    mhdConsToPrim(Ui, Wi);

    Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
    out << x << " "
        << Wi[MHD_PR_RHO] << " "
        << Wi[MHD_PR_VX] << " "
        << Wi[MHD_PR_VY] << " "
        << Wi[MHD_PR_VZ] << " "
        << Wi[MHD_PR_P] << " "
        << Wi[MHD_PR_BX] << " "
        << Wi[MHD_PR_BY] << " "
        << Wi[MHD_PR_BZ] << " "
        << Ui[MHD_E] << "\n";
  }

  std::cout << "Saved: " << fname.str() << std::endl;
  return 0;
}
