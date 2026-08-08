/* ============================================================================
 *  mhd2d_glm_raptor.cpp — 2D OT + mixed GLM, region 化重构 (Ch10 §10.3)
 *
 *  与 src/mhd/mhd2d_glm_cpu.cpp 数值逐位等价的机械重构, 七 region:
 *    recovery : 守恒 -> 原始 (全场一次/级; cfl 与 recon 消费同一份 P)
 *    cfl      : 物理信号速度归约 + 信号率归约 (两个函数同属一 region)
 *    recon    : x/y 方向逐胞 MUSCL 面值重构 + 面值 prim->cons
 *    flux     : x/y 界面 GLM-HLLD/HLL 通量 (含 fallback 判据计数)
 *    rhs      : 通量散度差分
 *    glm      : psi 阻尼源项 (含 damping_rate = ch/max(alpha,eps) 的除法)
 *    update   : SSPRK2 两级组合
 *  BC 复制与正定性检查留在 main (胶水/诊断, 不截断)。
 *  逐位等价依据: 原版逐界面重复计算的 prim 转换/重构值被逐胞霍升复用
 *  (同函数同输入 -> 同值); 阻尼拆为独立循环 (逐胞独立, 顺序无关)。
 *
 *  原生:  /lsc/opt/llvm-20.1/bin/clang++ -O2 -std=c++14 \
 *           -DMHD_HLLD_TOLERANCE_VALUE=1e-4 -I include \
 *           src/mhd/mhd2d_glm_raptor.cpp -o bin/raptor/mhd2d_extracted
 *  插桩:  scripts/raptor-clang++ ... -DUSE_RAPTOR ... -o bin/raptor/mhd2d_raptor
 *  用法:  ./mhd2d_raptor nx ny t_end cfl order solver alpha [regions]
 * ==========================================================================*/

#define MHD_GAMMA_VALUE (5.0 / 3.0)
#include "mhd/mhd_glm_common.hpp"
#ifdef USE_RAPTOR
#include <raptor/raptor.h>
#endif

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define NV MHD_GLM_NVAR

static inline int id2(int i, int j, int nxt) { return j * nxt + i; }

/* ── 与原版逐行对应的辅助 ─────────────────────────────────────────────────── */
static void stateToPrim9(const Real* U9, Real P[NV]) {
  Real W[MHD_NVAR], psi;
  mhdGlmConsToPrim(U9, W, psi);
  for (int k = 0; k < MHD_NVAR; k++) P[k] = W[k];
  P[MHD_GLM_PSI] = psi;
}

static void prim9ToState(const Real P[NV], Real U9[NV]) {
  Real W[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) W[k] = P[k];
  mhdGlmPrimToCons(W, P[MHD_GLM_PSI], U9);
}

static bool prim9IsUsable(const Real P[NV]) {
  if (P[MHD_PR_RHO] <= MHD_RHO_FLOOR || P[MHD_PR_P] <= MHD_P_FLOOR) return false;
  for (int k = 0; k < NV; k++) {
    if (!std::isfinite((double)P[k])) return false;
  }
  return true;
}

static bool state9IsUsable(const Real* U9) {
  Real P[NV];
  if (U9[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  stateToPrim9(U9, P);
  return prim9IsUsable(P);
}

static void reconPrim9(const Real Pm[NV], const Real P0[NV], const Real Pp[NV],
                       Real Pl[NV], Real Pr[NV]) {
  for (int k = 0; k < NV; k++) {
    Real slope = mhdMinmod(P0[k] - Pm[k], Pp[k] - P0[k]);
    Pl[k] = P0[k] - (Real)0.5 * slope;
    Pr[k] = P0[k] + (Real)0.5 * slope;
  }
  if (!prim9IsUsable(Pl) || !prim9IsUsable(Pr)) {
    for (int k = 0; k < NV; k++) { Pl[k] = P0[k]; Pr[k] = P0[k]; }
  }
}

/* ── 七个 region ─────────────────────────────────────────────────────────── */
extern "C" {

__attribute__((noinline))
void m2RegionRecovery(const Real* U, Real* P, int total) {
  for (int c = 0; c < total; c++) stateToPrim9(&U[(size_t)c * NV], &P[(size_t)c * NV]);
}

__attribute__((noinline))
Real m2RegionCflSpeed(const Real* P, int nx, int ny, int ghost) {
  const int nxt = nx + 2 * ghost;
  Real max_speed = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      const Real* Pc = &P[(size_t)id2(i, j, nxt) * NV];
      Real sx = mhdAbs(Pc[MHD_PR_VX]) + mhdFastSpeedX(Pc);
      Real sy = mhdAbs(Pc[MHD_PR_VY]) + mhdFastSpeedY(Pc);
      max_speed = mhdMax(max_speed, mhdMax(sx, sy));
    }
  }
  return max_speed;
}

__attribute__((noinline))
Real m2RegionCflRate(const Real* P, int nx, int ny, int ghost, Real dx,
                     Real dy, Real ch) {
  const int nxt = nx + 2 * ghost;
  Real max_rate = (Real)0.0;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      const Real* Pc = &P[(size_t)id2(i, j, nxt) * NV];
      Real sx = mhdMax(mhdAbs(Pc[MHD_PR_VX]) + mhdFastSpeedX(Pc), ch);
      Real sy = mhdMax(mhdAbs(Pc[MHD_PR_VY]) + mhdFastSpeedY(Pc), ch);
      max_rate = mhdMax(max_rate, sx / dx + sy / dy);
    }
  }
  return max_rate;
}

/* 逐胞 x/y 面值: 重构 + prim->cons (原版 x/yFaceStates 的霍升形式) */
__attribute__((noinline))
void m2RegionRecon(const Real* P, Real* UXL, Real* UXR, Real* UYL, Real* UYR,
                   int nx, int ny, int ghost) {
  const int nxt = nx + 2 * ghost;
  Real Pl[NV], Pr[NV];
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost - 1; i <= nx + ghost; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      reconPrim9(&P[(size_t)id2(i - 1, j, nxt) * NV], &P[c],
                 &P[(size_t)id2(i + 1, j, nxt) * NV], Pl, Pr);
      prim9ToState(Pl, &UXL[c]);
      prim9ToState(Pr, &UXR[c]);
    }
  }
  for (int j = ghost - 1; j <= ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      reconPrim9(&P[(size_t)id2(i, j - 1, nxt) * NV], &P[c],
                 &P[(size_t)id2(i, j + 1, nxt) * NV], Pl, Pr);
      prim9ToState(Pl, &UYL[c]);
      prim9ToState(Pr, &UYR[c]);
    }
  }
}

/* 一阶路径: 面值 = 胞平均 (原版 use_muscl=false 分支) */
__attribute__((noinline))
void m2RegionFacesFirstOrder(const Real* U, Real* UXL, Real* UXR, Real* UYL,
                             Real* UYR, int nx, int ny, int ghost) {
  const int nxt = nx + 2 * ghost;
  const int nyt = ny + 2 * ghost;
  for (int j = 0; j < nyt; j++) {
    for (int i = 0; i < nxt; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      for (int k = 0; k < NV; k++) {
        UXL[c + k] = U[c + k]; UXR[c + k] = U[c + k];
        UYL[c + k] = U[c + k]; UYR[c + k] = U[c + k];
      }
    }
  }
}

__attribute__((noinline))
void m2RegionFlux(const Real* UXL, const Real* UXR, const Real* UYL,
                  const Real* UYR, Real* Fx, Real* Gy, int nx, int ny,
                  int ghost, Real ch, int use_hlld,
                  unsigned long long* counters) {
  const int nxt = nx + 2 * ghost;
  Real UL[NV], UR[NV], F[NV];
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost - 1; i < nx + ghost; i++) {
      for (int k = 0; k < NV; k++) {
        UL[k] = UXR[(size_t)id2(i, j, nxt) * NV + k];
        UR[k] = UXL[(size_t)id2(i + 1, j, nxt) * NV + k];
      }
      int status = mhdGlmFluxX(UL, UR, ch, use_hlld != 0, F);
      if (use_hlld) {
        counters[0]++;
        if (status == MHD_HLLD_FALLBACK_DEGENERATE) counters[2]++;
        if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) counters[3]++;
      }
      for (int k = 0; k < NV; k++) Fx[(size_t)id2(i, j, nxt) * NV + k] = F[k];
    }
  }
  for (int j = ghost - 1; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      for (int k = 0; k < NV; k++) {
        UL[k] = UYR[(size_t)id2(i, j, nxt) * NV + k];
        UR[k] = UYL[(size_t)id2(i, j + 1, nxt) * NV + k];
      }
      int status = mhdGlmFluxY(UL, UR, ch, use_hlld != 0, F);
      if (use_hlld) {
        counters[1]++;
        if (status == MHD_HLLD_FALLBACK_DEGENERATE) counters[2]++;
        if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) counters[3]++;
      }
      for (int k = 0; k < NV; k++) Gy[(size_t)id2(i, j, nxt) * NV + k] = F[k];
    }
  }
}

__attribute__((noinline))
void m2RegionRhs(const Real* Fx, const Real* Gy, Real* rhs, int nx, int ny,
                 int ghost, Real dx, Real dy) {
  const int nxt = nx + 2 * ghost;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      size_t im = (size_t)id2(i - 1, j, nxt) * NV;
      size_t jm = (size_t)id2(i, j - 1, nxt) * NV;
      for (int k = 0; k < NV; k++) {
        rhs[c + k] = -(Fx[c + k] - Fx[im + k]) / dx
                     - (Gy[c + k] - Gy[jm + k]) / dy;
      }
    }
  }
}

/* psi 阻尼: 含 damping_rate 的除法 (原版在 computeRhs 内) */
__attribute__((noinline))
void m2RegionGlm(const Real* U, Real* rhs, int nx, int ny, int ghost,
                 Real ch, Real alpha) {
  const int nxt = nx + 2 * ghost;
  Real damping_rate = ch / mhdMax(alpha, (Real)1.0e-12);
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      rhs[c + MHD_GLM_PSI] -= damping_rate * U[c + MHD_GLM_PSI];
    }
  }
}

__attribute__((noinline))
void m2RegionStage1(const Real* U, const Real* rhs, Real* U1, Real dt,
                    int nx, int ny, int ghost) {
  const int nxt = nx + 2 * ghost;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      for (int k = 0; k < NV; k++) U1[c + k] = U[c + k] + dt * rhs[c + k];
    }
  }
}

__attribute__((noinline))
void m2RegionStage2(Real* U, const Real* U1, const Real* rhs2, Real dt,
                    int nx, int ny, int ghost) {
  const int nxt = nx + 2 * ghost;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      size_t c = (size_t)id2(i, j, nxt) * NV;
      for (int k = 0; k < NV; k++) {
        U[c + k] = (Real)0.5 * U[c + k]
                 + (Real)0.5 * (U1[c + k] + dt * rhs2[c + k]);
      }
    }
  }
}

}  /* extern "C" */

typedef void (*M2RecoveryFn)(const Real*, Real*, int);
typedef Real (*M2SpeedFn)(const Real*, int, int, int);
typedef Real (*M2RateFn)(const Real*, int, int, int, Real, Real, Real);
typedef void (*M2ReconFn)(const Real*, Real*, Real*, Real*, Real*, int, int, int);
typedef void (*M2FluxFn)(const Real*, const Real*, const Real*, const Real*,
                         Real*, Real*, int, int, int, Real, int,
                         unsigned long long*);
typedef void (*M2RhsFn)(const Real*, const Real*, Real*, int, int, int, Real, Real);
typedef void (*M2GlmFn)(const Real*, Real*, int, int, int, Real, Real);
typedef void (*M2Stage1Fn)(const Real*, const Real*, Real*, Real, int, int, int);
typedef void (*M2Stage2Fn)(Real*, const Real*, const Real*, Real, int, int, int);

static void applyPeriodic(Real* U, int nx, int ny, int ghost) {
  const int nxt = nx + 2 * ghost;
  for (int j = ghost; j < ny + ghost; j++) {
    for (int g = 0; g < ghost; g++) {
      for (int k = 0; k < NV; k++) {
        U[(size_t)id2(g, j, nxt) * NV + k] =
            U[(size_t)id2(nx + g, j, nxt) * NV + k];
        U[(size_t)id2(nx + ghost + g, j, nxt) * NV + k] =
            U[(size_t)id2(ghost + g, j, nxt) * NV + k];
      }
    }
  }
  for (int i = 0; i < nxt; i++) {
    for (int g = 0; g < ghost; g++) {
      for (int k = 0; k < NV; k++) {
        U[(size_t)id2(i, g, nxt) * NV + k] =
            U[(size_t)id2(i, ny + g, nxt) * NV + k];
        U[(size_t)id2(i, ny + ghost + g, nxt) * NV + k] =
            U[(size_t)id2(i, ghost + g, nxt) * NV + k];
      }
    }
  }
}

int main(int argc, char* argv[]) {
  int nx = 128, ny = 128;
  Real t_end = (Real)0.5;
  Real cfl = (Real)0.25;
  std::string order = "muscl";
  std::string solver = "hlld";
  Real alpha = (Real)0.18;
  std::string regions = "none";

  if (argc > 1) nx = std::atoi(argv[1]);
  if (argc > 2) ny = std::atoi(argv[2]);
  if (argc > 3) t_end = (Real)std::atof(argv[3]);
  if (argc > 4) cfl = (Real)std::atof(argv[4]);
  if (argc > 5) order = argv[5];
  if (argc > 6) solver = argv[6];
  if (argc > 7) alpha = (Real)std::atof(argv[7]);
  if (argc > 8) regions = argv[8];

  if (order != "muscl" && order != "first") return 1;
  if (solver != "hll" && solver != "hlld") return 1;
  if (alpha <= (Real)0.0) return 1;
  const int use_muscl = (order == "muscl") ? 1 : 0;
  const int use_hlld = (solver == "hlld") ? 1 : 0;

  auto wants = [&](const char* name) {
    return regions.find(name) != std::string::npos;
  };

  M2RecoveryFn fRecovery = m2RegionRecovery;
  M2SpeedFn fSpeed = m2RegionCflSpeed;
  M2RateFn fRate = m2RegionCflRate;
  M2ReconFn fRecon = m2RegionRecon;
  M2FluxFn fFlux = m2RegionFlux;
  M2RhsFn fRhs = m2RegionRhs;
  M2GlmFn fGlm = m2RegionGlm;
  M2Stage1Fn fStage1 = m2RegionStage1;
  M2Stage2Fn fStage2 = m2RegionStage2;

#ifdef USE_RAPTOR
  if (wants("recovery")) fRecovery = __raptor_truncate_op_func(m2RegionRecovery, 64, 0, 32);
  if (wants("cfl")) {
    fSpeed = __raptor_truncate_op_func(m2RegionCflSpeed, 64, 0, 32);
    fRate  = __raptor_truncate_op_func(m2RegionCflRate, 64, 0, 32);
  }
  if (wants("chs")) fSpeed = __raptor_truncate_op_func(m2RegionCflSpeed, 64, 0, 32);
  if (wants("chr")) fRate  = __raptor_truncate_op_func(m2RegionCflRate, 64, 0, 32);
  if (wants("recon"))  fRecon  = __raptor_truncate_op_func(m2RegionRecon, 64, 0, 32);
  if (wants("flux"))   fFlux   = __raptor_truncate_op_func(m2RegionFlux, 64, 0, 32);
  if (wants("rhs"))    fRhs    = __raptor_truncate_op_func(m2RegionRhs, 64, 0, 32);
  if (wants("glm"))    fGlm    = __raptor_truncate_op_func(m2RegionGlm, 64, 0, 32);
  if (wants("update")) {
    fStage1 = __raptor_truncate_op_func(m2RegionStage1, 64, 0, 32);
    fStage2 = __raptor_truncate_op_func(m2RegionStage2, 64, 0, 32);
  }
#else
  if (regions != "none" && regions != "") {
    std::cerr << "Warning: built without USE_RAPTOR, region list ignored."
              << std::endl;
  }
#endif

  const int ghost = 2;
  const int nxt = nx + 2 * ghost;
  const int nyt = ny + 2 * ghost;
  const int total = nxt * nyt;
  const Real x_min = (Real)0.0, x_max = (Real)1.0;
  const Real y_min = (Real)0.0, y_max = (Real)1.0;
  const Real dx = (x_max - x_min) / (Real)nx;
  const Real dy = (y_max - y_min) / (Real)ny;
  const Real pi = std::acos((Real)-1.0);
  const Real two_pi = (Real)2.0 * pi;
  const Real inv_sqrt_4pi = (Real)1.0 / std::sqrt((Real)4.0 * pi);

  std::cout << "===== 2D OT mixed GLM (region build) =====" << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL")
            << ", order: " << order << std::endl;
  std::cout << "Truncated regions: " << regions << std::endl;
  std::cout << "Nx x Ny = " << nx << " x " << ny << ", CFL = " << cfl
            << ", t_end = " << t_end << ", alpha = " << alpha << std::endl;

  std::vector<Real> U((size_t)total * NV);
  for (int j = 0; j < nyt; j++) {
    for (int i = 0; i < nxt; i++) {
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;
      Real P[NV], U9[NV];
      P[MHD_PR_RHO] = (Real)25.0 / ((Real)36.0 * pi);
      P[MHD_PR_VX] = -std::sin(two_pi * y);
      P[MHD_PR_VY] = std::sin(two_pi * x);
      P[MHD_PR_VZ] = (Real)0.0;
      P[MHD_PR_P] = (Real)5.0 / ((Real)12.0 * pi);
      P[MHD_PR_BX] = -std::sin(two_pi * y) * inv_sqrt_4pi;
      P[MHD_PR_BY] = std::sin((Real)2.0 * two_pi * x) * inv_sqrt_4pi;
      P[MHD_PR_BZ] = (Real)0.0;
      P[MHD_GLM_PSI] = (Real)0.0;
      prim9ToState(P, U9);
      for (int k = 0; k < NV; k++) U[(size_t)id2(i, j, nxt) * NV + k] = U9[k];
    }
  }
  applyPeriodic(U.data(), nx, ny, ghost);

  std::vector<Real> Pm((size_t)total * NV), rhs((size_t)total * NV);
  std::vector<Real> U1((size_t)total * NV);
  std::vector<Real> UXL((size_t)total * NV), UXR((size_t)total * NV);
  std::vector<Real> UYL((size_t)total * NV), UYR((size_t)total * NV);
  std::vector<Real> Fx((size_t)total * NV), Gy((size_t)total * NV);

  Real t = (Real)0.0, last_ch = (Real)0.0;
  int step = 0;
  unsigned long long counters[4] = {0, 0, 0, 0};

  auto evalRhs = [&](Real* Ustage) {
    /* 原版 computeRhs 的 value 拷贝 + BC 等价于对已 BC 的 Ustage 原位求值 */
    applyPeriodic(Ustage, nx, ny, ghost);
    fRecovery(Ustage, Pm.data(), total);
    if (use_muscl) {
      fRecon(Pm.data(), UXL.data(), UXR.data(), UYL.data(), UYR.data(),
             nx, ny, ghost);
    } else {
      m2RegionFacesFirstOrder(Ustage, UXL.data(), UXR.data(), UYL.data(),
                              UYR.data(), nx, ny, ghost);
    }
    fFlux(UXL.data(), UXR.data(), UYL.data(), UYR.data(), Fx.data(),
          Gy.data(), nx, ny, ghost, last_ch, use_hlld, counters);
    fRhs(Fx.data(), Gy.data(), rhs.data(), nx, ny, ghost, dx, dy);
    fGlm(Ustage, rhs.data(), nx, ny, ghost, last_ch, alpha);
  };

  while (t < t_end) {
    fRecovery(U.data(), Pm.data(), total);
    Real speed = fSpeed(Pm.data(), nx, ny, ghost);
    Real ch = mhdMax(speed, (Real)1.0e-12);
    last_ch = ch;
    Real rate = fRate(Pm.data(), nx, ny, ghost, dx, dy, ch);
    if (rate <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal rate." << std::endl;
      return 2;
    }
    Real dt = cfl / rate;
    if (t + dt > t_end) dt = t_end - t;

    evalRhs(U.data());
    fStage1(U.data(), rhs.data(), U1.data(), dt, nx, ny, ghost);
    applyPeriodic(U1.data(), nx, ny, ghost);
    for (int j = ghost; j < ny + ghost; j++) {
      for (int i = ghost; i < nx + ghost; i++) {
        if (!state9IsUsable(&U1[(size_t)id2(i, j, nxt) * NV])) {
          std::cerr << "Error: non-physical RK stage at step " << step
                    << ", cell (" << (i - ghost) << ", " << (j - ghost) << ")"
                    << std::endl;
          return 3;
        }
      }
    }

    evalRhs(U1.data());
    fStage2(U.data(), U1.data(), rhs.data(), dt, nx, ny, ghost);
    applyPeriodic(U.data(), nx, ny, ghost);
    for (int j = ghost; j < ny + ghost; j++) {
      for (int i = ghost; i < nx + ghost; i++) {
        if (!state9IsUsable(&U[(size_t)id2(i, j, nxt) * NV])) {
          std::cerr << "Error: non-physical state at step " << step
                    << ", cell (" << (i - ghost) << ", " << (j - ghost) << ")"
                    << std::endl;
          return 4;
        }
      }
    }

    t += dt;
    step++;
  }

  std::cout << "Done. Steps: " << step << ", final t = " << t << std::endl;
  std::cout << "c_h(last) = " << last_ch << ", alpha = " << alpha << std::endl;

  Real div_l1 = (Real)0.0, div_linf = (Real)0.0, psi_linf = (Real)0.0;
  auto Uat = [&](int i, int j, int k) -> Real {
    return U[(size_t)id2(i, j, nxt) * NV + k];
  };
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real divb = (Uat(i + 1, j, MHD_BX) - Uat(i - 1, j, MHD_BX))
                      / ((Real)2.0 * dx)
                + (Uat(i, j + 1, MHD_BY) - Uat(i, j - 1, MHD_BY))
                      / ((Real)2.0 * dy);
      div_l1 += mhdAbs(divb);
      div_linf = mhdMax(div_linf, mhdAbs(divb));
      psi_linf = mhdMax(psi_linf, mhdAbs(Uat(i, j, MHD_GLM_PSI)));
    }
  }
  div_l1 /= (Real)(nx * ny);
  std::cout << "divB L1 = " << div_l1 << ", divB Linf = " << div_linf
            << ", psi Linf = " << psi_linf << std::endl;

  unsigned long long evals = counters[0] + counters[1];
  if (use_hlld) {
    unsigned long long fb = counters[2] + counters[3];
    double frac = evals > 0 ? (double)fb / (double)evals : 0.0;
    std::cout << "HLLD flux evaluations = " << evals << std::endl;
    std::cout << "HLLD->HLL fallbacks = " << fb
              << " (degenerate=" << counters[2]
              << ", nonphysical=" << counters[3]
              << "), fraction = " << std::scientific << frac
              << std::defaultfloat << std::endl;
  }

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/raptor/ot2d");
  if (std::system(("mkdir -p " + out_dir).c_str()) != 0) { /* ignore */ }
  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::string tag = (regions == "none" || regions.empty()) ? "none" : regions;
  for (char& c : tag) if (c == ',') c = '+';
  std::ostringstream fname;
  fname << out_dir << "/orszag_tang_glm_" << order << "_" << solver << "_"
        << prec_name << "_N" << nx << "x" << ny << "_r-" << tag << ".dat";
  std::ofstream out(fname.str());
  if (!out) return 5;
  out << "# Orszag-Tang ideal MHD with mixed GLM (region build)\n";
  out << "# nx " << nx << " ny " << ny << " t " << t << " gamma " << MHD_GAMMA
      << " cfl " << cfl << " order " << order << " solver " << solver
      << " alpha " << alpha << " ch_last " << last_ch
      << " truncated_regions " << tag << "\n";
  out << "# columns: x y rho vx vy vz p Bx By Bz E divB psi\n";
  out << std::scientific
      << std::setprecision(std::numeric_limits<Real>::max_digits10);
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real U9[NV], W[MHD_NVAR], psi;
      for (int k = 0; k < NV; k++) U9[k] = U[(size_t)id2(i, j, nxt) * NV + k];
      mhdGlmConsToPrim(U9, W, psi);
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;
      Real divb = (Uat(i + 1, j, MHD_BX) - Uat(i - 1, j, MHD_BX))
                      / ((Real)2.0 * dx)
                + (Uat(i, j + 1, MHD_BY) - Uat(i, j - 1, MHD_BY))
                      / ((Real)2.0 * dy);
      out << x << " " << y << " " << W[MHD_PR_RHO] << " " << W[MHD_PR_VX]
          << " " << W[MHD_PR_VY] << " " << W[MHD_PR_VZ] << " " << W[MHD_PR_P]
          << " " << W[MHD_PR_BX] << " " << W[MHD_PR_BY] << " " << W[MHD_PR_BZ]
          << " " << U9[MHD_E] << " " << divb << " " << psi << "\n";
    }
  }
  std::cout << "Saved: " << fname.str() << std::endl;
  return 0;
}
