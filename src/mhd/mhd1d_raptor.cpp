/* ============================================================================
 *  mhd1d_raptor.cpp — Brio-Wu 1D, region 化重构版 (Ch10 §10.3 RAPTOR 实验)
 *
 *  与 src/mhd/mhd1d_cpu.cpp 数值逐位等价的机械重构: 把时间步内的六个阶段
 *  提取为独立的 C 风格函数 (region), 使 RAPTOR 能按 region 生成降精度克隆:
 *    recovery : 守恒 -> 原始变量 (全场一次, 从各 region 调用链中提出)
 *    cfl      : 最大信号速度归约
 *    recon    : MUSCL 原始变量重构 (含 minmod 与可用性回退)
 *    hancock  : Hancock 半步预测 (含 prim<->cons 与通量差)
 *    flux     : 界面 HLLD/HLL 通量 (含 fallback 判据与计数)
 *    update   : 守恒有限体积更新
 *  正定性检查与 dt 计算留在 main (诊断/2 flop, 不参与 region 截断)。
 *
 *  两种编译形态:
 *    原生 (提取合法性验证, 必须与原版 clang-20 build 数据行逐位相同):
 *      /lsc/opt/llvm-20.1/bin/clang++ -O2 -std=c++14 -I include \
 *          src/mhd/mhd1d_raptor.cpp -o bin/raptor/mhd1d_extracted
 *    RAPTOR 插桩 (region 截断实验):
 *      scripts/raptor-clang++ -O2 -std=c++14 -DUSE_RAPTOR -I include \
 *          src/mhd/mhd1d_raptor.cpp -o bin/raptor/mhd1d_raptor
 *
 *  用法 (前 5 个参数与原版相同, 第 6 个为逗号分隔的截断 region 列表):
 *    ./bin/raptor/mhd1d_raptor 800 0.1 0.4 muscl hlld recovery,flux
 *    region 名 ∈ {recovery,cfl,recon,hancock,flux,update}; 缺省/none = 全 FP64。
 *    截断模式固定为 IEEE binary32 运算 (__raptor_truncate_op_func(f,64,0,32))。
 * ==========================================================================*/

#include "mhd/mhd_common.hpp"
#ifdef USE_RAPTOR
#include <raptor/raptor.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define NV MHD_NVAR

/* ── 与原版逐行对应的辅助函数 ─────────────────────────────────────────────── */
static bool primitiveIsUsable(const Real W[NV]) {
  if (W[MHD_PR_RHO] <= MHD_RHO_FLOOR || W[MHD_PR_P] <= MHD_P_FLOOR) return false;
  for (int k = 0; k < NV; k++) {
    if (!std::isfinite((double)W[k])) return false;
  }
  return true;
}

static bool stateIsUsable(const Real* U8) {
  Real U[NV], W[NV];
  for (int k = 0; k < NV; k++) U[k] = U8[k];
  if (U[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  mhdConsToPrim(U, W);
  if (W[MHD_PR_RHO] <= (Real)0.0 || W[MHD_PR_P] <= (Real)0.0) return false;
  for (int k = 0; k < NV; k++) {
    if (!std::isfinite((double)U[k]) || !std::isfinite((double)W[k])) return false;
  }
  return true;
}

static void reconstructPrimitive(const Real Wm[NV], const Real W0[NV],
                                 const Real Wp[NV], Real Wleft[NV],
                                 Real Wright[NV]) {
  for (int k = 0; k < NV; k++) {
    Real slope = mhdMinmod(W0[k] - Wm[k], Wp[k] - W0[k]);
    Wleft[k] = W0[k] - (Real)0.5 * slope;
    Wright[k] = W0[k] + (Real)0.5 * slope;
  }
  Wleft[MHD_PR_BX] = W0[MHD_PR_BX];
  Wright[MHD_PR_BX] = W0[MHD_PR_BX];
  if (!primitiveIsUsable(Wleft) || !primitiveIsUsable(Wright)) {
    for (int k = 0; k < NV; k++) {
      Wleft[k] = W0[k];
      Wright[k] = W0[k];
    }
  }
}

static void hancockPredictCell(const Real Wleft[NV], const Real Wright[NV],
                               Real factor, Real* ULH_out, Real* URH_out) {
  Real UL[NV], UR[NV], FL[NV], FR[NV];
  mhdPrimToCons(Wleft, UL);
  mhdPrimToCons(Wright, UR);
  mhdFluxX(UL, FL);
  mhdFluxX(UR, FR);

  Real ULH[NV], URH[NV];
  for (int k = 0; k < NV; k++) {
    Real correction = factor * (FR[k] - FL[k]);
    ULH[k] = UL[k] - correction;
    URH[k] = UR[k] - correction;
  }

  Real WLH[NV], WRH[NV];
  mhdConsToPrim(ULH, WLH);
  mhdConsToPrim(URH, WRH);
  if (ULH[MHD_RHO] <= MHD_RHO_FLOOR || !primitiveIsUsable(WLH)) {
    for (int k = 0; k < NV; k++) ULH[k] = UL[k];
  }
  if (URH[MHD_RHO] <= MHD_RHO_FLOOR || !primitiveIsUsable(WRH)) {
    for (int k = 0; k < NV; k++) URH[k] = UR[k];
  }
  for (int k = 0; k < NV; k++) { ULH_out[k] = ULH[k]; URH_out[k] = URH[k]; }
}

/* ── 六个 region 函数 (C 风格签名, noinline, RAPTOR 克隆的单位) ─────────────── */
extern "C" {

__attribute__((noinline))
void regionRecovery(const Real* U, Real* W, int total) {
  for (int i = 0; i < total; i++) {
    Real Ui[NV], Wi[NV];
    for (int k = 0; k < NV; k++) Ui[k] = U[i * NV + k];
    mhdConsToPrim(Ui, Wi);
    for (int k = 0; k < NV; k++) W[i * NV + k] = Wi[k];
  }
}

__attribute__((noinline))
Real regionCfl(const Real* W, int nx, int ghost) {
  Real max_speed = (Real)0.0;
  for (int i = ghost; i < nx + ghost; i++) {
    const Real* Wi = &W[i * NV];
    Real speed = mhdAbs(Wi[MHD_PR_VX]) + mhdFastSpeedX(Wi);
    max_speed = mhdMax(max_speed, speed);
  }
  return max_speed;
}

__attribute__((noinline))
void regionRecon(const Real* W, Real* WLf, Real* WRf, int nx, int ghost) {
  for (int i = ghost - 1; i <= nx + ghost; i++) {
    reconstructPrimitive(&W[(i - 1) * NV], &W[i * NV], &W[(i + 1) * NV],
                         &WLf[i * NV], &WRf[i * NV]);
  }
}

__attribute__((noinline))
void regionHancock(const Real* WLf, const Real* WRf, Real factor,
                   Real* UleftHalf, Real* UrightHalf, int nx, int ghost) {
  for (int i = ghost - 1; i <= nx + ghost; i++) {
    hancockPredictCell(&WLf[i * NV], &WRf[i * NV], factor,
                       &UleftHalf[i * NV], &UrightHalf[i * NV]);
  }
}

__attribute__((noinline))
void regionFlux(const Real* U, const Real* UleftHalf, const Real* UrightHalf,
                Real* F, int nx, int ghost, int use_muscl, int use_hlld,
                unsigned long long* counters) {
  for (int i = ghost - 1; i < nx + ghost; i++) {
    Real UiL[NV], UiR[NV], Fi[NV];
    const Real* ls = use_muscl ? &UrightHalf[i * NV] : &U[i * NV];
    const Real* rs = use_muscl ? &UleftHalf[(i + 1) * NV] : &U[(i + 1) * NV];
    for (int k = 0; k < NV; k++) { UiL[k] = ls[k]; UiR[k] = rs[k]; }
    if (use_hlld) {
      int status = mhdHlldFluxX(UiL, UiR, Fi);
      counters[0]++;
      if (status == MHD_HLLD_FALLBACK_DEGENERATE) counters[1]++;
      if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) counters[2]++;
    } else {
      mhdHllFluxX(UiL, UiR, Fi);
    }
    for (int k = 0; k < NV; k++) F[i * NV + k] = Fi[k];
  }
}

__attribute__((noinline))
void regionUpdate(Real* U, const Real* F, Real ratio, int nx, int ghost) {
  for (int i = ghost; i < nx + ghost; i++) {
    for (int k = 0; k < NV; k++) {
      U[i * NV + k] -= ratio * (F[i * NV + k] - F[(i - 1) * NV + k]);
    }
  }
}

}  /* extern "C" */

/* ── region 指针表: 默认原函数, 被选中的 region 换成 RAPTOR 截断克隆 ───────── */
typedef void (*RecoveryFn)(const Real*, Real*, int);
typedef Real (*CflFn)(const Real*, int, int);
typedef void (*ReconFn)(const Real*, Real*, Real*, int, int);
typedef void (*HancockFn)(const Real*, const Real*, Real, Real*, Real*, int, int);
typedef void (*FluxFn)(const Real*, const Real*, const Real*, Real*, int, int,
                       int, int, unsigned long long*);
typedef void (*UpdateFn)(Real*, const Real*, Real, int, int);

int main(int argc, char* argv[]) {
  int nx = 400;
  Real t_end = (Real)0.1;
  Real cfl = (Real)0.4;
  std::string order = "muscl";
  std::string solver = "hll";
  std::string regions = "none";

  if (argc > 1) nx = std::atoi(argv[1]);
  if (argc > 2) t_end = (Real)std::atof(argv[2]);
  if (argc > 3) cfl = (Real)std::atof(argv[3]);
  if (argc > 4) order = argv[4];
  if (argc > 5) solver = argv[5];
  if (argc > 6) regions = argv[6];

  if (order != "muscl" && order != "first") {
    std::cerr << "Error: order must be 'muscl' or 'first'." << std::endl;
    return 1;
  }
  if (solver != "hll" && solver != "hlld") {
    std::cerr << "Error: solver must be 'hll' or 'hlld'." << std::endl;
    return 1;
  }
  const int use_muscl = (order == "muscl") ? 1 : 0;
  const int use_hlld = (solver == "hlld") ? 1 : 0;

  auto wants = [&](const char* name) {
    return regions.find(name) != std::string::npos;
  };

  RecoveryFn fRecovery = regionRecovery;
  CflFn fCfl = regionCfl;
  ReconFn fRecon = regionRecon;
  HancockFn fHancock = regionHancock;
  FluxFn fFlux = regionFlux;
  UpdateFn fUpdate = regionUpdate;

#ifdef USE_RAPTOR
  if (wants("recovery")) fRecovery = __raptor_truncate_op_func(regionRecovery, 64, 0, 32);
  if (wants("cfl"))      fCfl      = __raptor_truncate_op_func(regionCfl, 64, 0, 32);
  if (wants("recon"))    fRecon    = __raptor_truncate_op_func(regionRecon, 64, 0, 32);
  if (wants("hancock"))  fHancock  = __raptor_truncate_op_func(regionHancock, 64, 0, 32);
  if (wants("flux"))     fFlux     = __raptor_truncate_op_func(regionFlux, 64, 0, 32);
  if (wants("update"))   fUpdate   = __raptor_truncate_op_func(regionUpdate, 64, 0, 32);
#else
  if (regions != "none" && regions != "") {
    std::cerr << "Warning: built without USE_RAPTOR, region list ignored."
              << std::endl;
  }
#endif

  const int ghost = 2;
  const int total = nx + 2 * ghost;
  const Real x_min = (Real)0.0, x_max = (Real)1.0, x0 = (Real)0.5;
  const Real dx = (x_max - x_min) / (Real)nx;

  std::cout << "===== 1D ideal MHD Brio-Wu shock tube (region build) ====="
            << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL") << std::endl;
  std::cout << "Reconstruction: "
            << (use_muscl ? "MUSCL-Hancock" : "first-order") << std::endl;
  std::cout << "Precision: " << (int)(sizeof(Real) * 8) << " bit"
            << (sizeof(Real) == 8 ? " (fp64)" : " (fp32)") << std::endl;
  std::cout << "Truncated regions: " << regions << std::endl;
  std::cout << "gamma = " << MHD_GAMMA << std::endl;
  std::cout << "N = " << nx << ", dx = " << dx << ", CFL = " << cfl << std::endl;

  /* Brio-Wu 初始条件, 与原版一致 */
  Real WLp[NV] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0};
  Real WRp[NV] = {0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0};
  Real ULc[NV], URc[NV];
  mhdPrimToCons(WLp, ULc);
  mhdPrimToCons(WRp, URc);

  std::vector<Real> U((size_t)total * NV), W((size_t)total * NV);
  std::vector<Real> F((size_t)total * NV);
  std::vector<Real> WLf((size_t)total * NV), WRf((size_t)total * NV);
  std::vector<Real> UleftHalf((size_t)total * NV), UrightHalf((size_t)total * NV);
  for (int i = 0; i < total; i++) {
    Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
    const Real* src = (x < x0) ? ULc : URc;
    for (int k = 0; k < NV; k++) U[(size_t)i * NV + k] = src[k];
  }

  Real t = (Real)0.0;
  int step = 0;
  unsigned long long counters[3] = {0, 0, 0};

  while (t < t_end) {
    /* transmissive BC (与原版一致) */
    for (int g = 0; g < ghost; g++) {
      int right_ghost = total - 1 - g;
      for (int k = 0; k < NV; k++) {
        U[(size_t)g * NV + k] = U[(size_t)ghost * NV + k];
        U[(size_t)right_ghost * NV + k] =
            U[(size_t)(total - 1 - ghost) * NV + k];
      }
    }

    fRecovery(U.data(), W.data(), total);

    Real max_speed = fCfl(W.data(), nx, ghost);
    if (max_speed <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal speed." << std::endl;
      return 2;
    }
    Real dt = cfl * dx / max_speed;
    if (t + dt > t_end) dt = t_end - t;

    if (use_muscl) {
      fRecon(W.data(), WLf.data(), WRf.data(), nx, ghost);
      fHancock(WLf.data(), WRf.data(), (Real)0.5 * dt / dx,
               UleftHalf.data(), UrightHalf.data(), nx, ghost);
    }

    fFlux(U.data(), UleftHalf.data(), UrightHalf.data(), F.data(), nx, ghost,
          use_muscl, use_hlld, counters);

    fUpdate(U.data(), F.data(), dt / dx, nx, ghost);

    for (int i = ghost; i < nx + ghost; i++) {
      if (!stateIsUsable(&U[(size_t)i * NV])) {
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
    unsigned long long fb = counters[1] + counters[2];
    double frac = counters[0] > 0 ? (double)fb / (double)counters[0] : 0.0;
    std::cout << "HLLD flux evaluations = " << counters[0] << std::endl;
    std::cout << "HLLD relative fallback tolerance = "
              << std::scientific << (double)mhdHlldTolerance()
              << std::defaultfloat << std::endl;
    std::cout << "HLLD->HLL fallbacks = " << fb
              << " (degenerate=" << counters[1]
              << ", nonphysical=" << counters[2]
              << "), fraction = " << std::scientific << frac
              << std::defaultfloat << std::endl;
  }

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/raptor/brio_wu");
  if (std::system(("mkdir -p " + out_dir).c_str()) != 0) { /* ignore */ }

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::string tag = (regions == "none" || regions.empty()) ? "none" : regions;
  for (char& c : tag) if (c == ',') c = '+';
  std::ostringstream fname;
  fname << out_dir << "/brio_wu_" << order << "_" << solver << "_" << prec_name
        << "_N" << nx << "_r-" << tag << ".dat";

  std::ofstream out(fname.str());
  if (!out) {
    std::cerr << "Error: failed to open output file: " << fname.str()
              << std::endl;
    return 4;
  }
  out << "# Brio-Wu ideal MHD (region build)\n";
  out << "# nx " << nx << " t " << t << " gamma " << MHD_GAMMA << " cfl " << cfl
      << " order " << order << " solver " << solver
      << " truncated_regions " << tag << "\n";
  out << "# columns: x rho vx vy vz p Bx By Bz E\n";
  out << std::scientific
      << std::setprecision(std::numeric_limits<Real>::max_digits10);
  for (int i = ghost; i < nx + ghost; i++) {
    Real Ui[NV], Wi[NV];
    for (int k = 0; k < NV; k++) Ui[k] = U[(size_t)i * NV + k];
    mhdConsToPrim(Ui, Wi);
    Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
    out << x << " " << Wi[MHD_PR_RHO] << " " << Wi[MHD_PR_VX] << " "
        << Wi[MHD_PR_VY] << " " << Wi[MHD_PR_VZ] << " " << Wi[MHD_PR_P] << " "
        << Wi[MHD_PR_BX] << " " << Wi[MHD_PR_BY] << " " << Wi[MHD_PR_BZ]
        << " " << Ui[MHD_E] << "\n";
  }
  std::cout << "Saved: " << fname.str() << std::endl;
  return 0;
}
