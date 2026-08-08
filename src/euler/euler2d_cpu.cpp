/**
 * ============================================================================
 *      2D 欧拉方程求解器 (MUSCL-Hancock + HLLC) — CPU 版本
 * ============================================================================
 *
 * 核心数值函数来自 include/ 下的共享头文件:
 *   config.hpp       — 全局参数
 *   euler_common.hpp — 变量转换, 声速, minmod
 *   riemann.hpp      — HLLC solver, Hancock 演化, MUSCL 重构
 *
 * 本文件只包含: SoA 数据结构, 边界条件, 初始条件,
 *              时间步 CFL, 全场重构/演化/通量/更新, main
 *
 * ============================================================================
 */

#include "euler/riemann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

// ============================================================================
//  索引函数 (行优先)
// ============================================================================
inline int idx(int i, int j) { return j * TOTAL_X + i; }

// ============================================================================
//  SoA 数据结构
// ============================================================================
struct ConservativeField {
  std::vector<Real> rho, rhou, rhov, E;
  ConservativeField(int size) : rho(size), rhou(size), rhov(size), E(size) {}
};

struct PrimitiveField {
  std::vector<Real> rho, u, v, p;
  PrimitiveField(int size) : rho(size), u(size), v(size), p(size) {}
};

struct FluxField {
  std::vector<Real> f0, f1, f2, f3;
  FluxField(int size) : f0(size), f1(size), f2(size), f3(size) {}
};

// ============================================================================
//  全场 守恒→原始
// ============================================================================
void conservativeToPrimitive(const ConservativeField &U, PrimitiveField &W) {
  for (int j = 0; j < TOTAL_Y; j++) {
    for (int i = 0; i < TOTAL_X; i++) {
      int id = idx(i, j);
      Real Upt[4] = {U.rho[id], U.rhou[id], U.rhov[id], U.E[id]};
      Real Wpt[4];
      consToPrim(Upt, Wpt);  // 共享函数
      W.rho[id] = Wpt[0];
      W.u[id]   = Wpt[1];
      W.v[id]   = Wpt[2];
      W.p[id]   = Wpt[3];
    }
  }
}

// ============================================================================
//  边界条件 (透射)
// ============================================================================
void applyBoundaryConditions(ConservativeField &U) {
  // 左右边界
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int g = 0; g < GHOST; g++) {
      int idL_ghost = idx(g, j), idL_inner = idx(GHOST, j);
      U.rho[idL_ghost] = U.rho[idL_inner];
      U.rhou[idL_ghost] = U.rhou[idL_inner];
      U.rhov[idL_ghost] = U.rhov[idL_inner];
      U.E[idL_ghost] = U.E[idL_inner];

      int idR_ghost = idx(TOTAL_X - 1 - g, j);
      int idR_inner = idx(TOTAL_X - 1 - GHOST, j);
      U.rho[idR_ghost] = U.rho[idR_inner];
      U.rhou[idR_ghost] = U.rhou[idR_inner];
      U.rhov[idR_ghost] = U.rhov[idR_inner];
      U.E[idR_ghost] = U.E[idR_inner];
    }
  }
  // 上下边界
  for (int g = 0; g < GHOST; g++) {
    for (int i = 0; i < TOTAL_X; i++) {
      int idB_ghost = idx(i, g), idB_inner = idx(i, GHOST);
      U.rho[idB_ghost] = U.rho[idB_inner];
      U.rhou[idB_ghost] = U.rhou[idB_inner];
      U.rhov[idB_ghost] = U.rhov[idB_inner];
      U.E[idB_ghost] = U.E[idB_inner];

      int idT_ghost = idx(i, TOTAL_Y - 1 - g);
      int idT_inner = idx(i, TOTAL_Y - 1 - GHOST);
      U.rho[idT_ghost] = U.rho[idT_inner];
      U.rhou[idT_ghost] = U.rhou[idT_inner];
      U.rhov[idT_ghost] = U.rhov[idT_inner];
      U.E[idT_ghost] = U.E[idT_inner];
    }
  }
}

// ============================================================================
//  初始条件: 激波-气泡
// ============================================================================
void initializeShockBubble(ConservativeField &U, PrimitiveField &W) {
  Real M = MACH_NUM, g = GAMMA;
  Real rho1 = RHO_AIR, p1 = P_ATM;
  Real c1 = std::sqrt(g * p1 / rho1);

  Real rho2 = rho1 * ((g + 1) * M * M) / ((g - 1) * M * M + 2);
  Real p2 = p1 * (2 * g * M * M - (g - 1)) / (g + 1);
  Real u2 = c1 * M * (1 - rho1 / rho2);

  std::cout << "===== Shock-Bubble Problem =====" << std::endl;
  std::cout << "Pre-shock:  rho=" << rho1 << ", p=" << p1 << std::endl;
  std::cout << "Post-shock: rho=" << rho2 << ", p=" << p2 << ", u=" << u2 << std::endl;
  std::cout << "================================" << std::endl;

  Real R2 = BUBBLE_R * BUBBLE_R;

  for (int j = 0; j < TOTAL_Y; j++) {
    Real y = Y_MIN + (j - GHOST + 0.5) * DY;
    for (int i = 0; i < TOTAL_X; i++) {
      Real x = X_MIN + (i - GHOST + 0.5) * DX;
      int id = idx(i, j);

      Real rho, u, v, p;
      Real dx_b = x - BUBBLE_X;
      Real dy_b = y - BUBBLE_Y;
      Real dist_sq = dx_b * dx_b + dy_b * dy_b;

      if (dist_sq < R2) {
        rho = RHO_HELIUM; u = 0.0; v = 0.0; p = p1;
      } else if (x < SHOCK_X) {
        rho = rho2; u = u2; v = 0.0; p = p2;
      } else {
        rho = rho1; u = 0.0; v = 0.0; p = p1;
      }

      W.rho[id] = rho; W.u[id] = u; W.v[id] = v; W.p[id] = p;
      Real Wpt[4] = {rho, u, v, p};
      Real Upt[4];
      primToCons(Wpt, Upt);  // 共享函数
      U.rho[id] = Upt[0]; U.rhou[id] = Upt[1];
      U.rhov[id] = Upt[2]; U.E[id] = Upt[3];
    }
  }
}

// ============================================================================
//  CFL 时间步
// ============================================================================
Real computeTimeStep(const PrimitiveField &W) {
  Real max_sx = (Real)0.0, max_sy = (Real)0.0;
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int i = GHOST; i < NX + GHOST; i++) {
      int id = idx(i, j);
      Real c = soundSpeed(W.p[id], W.rho[id]);  // 共享函数
      max_sx = std::max(max_sx, std::abs(W.u[id]) + c);
      max_sy = std::max(max_sy, std::abs(W.v[id]) + c);
    }
  }
  return CFL_NUM * std::min(DX / max_sx, DY / max_sy);
}

// ============================================================================
//  MUSCL 重构 (全场) — 调用共享 muscl_recon
// ============================================================================
void musclReconstruct_x(const PrimitiveField &W, PrimitiveField &WL,
                        PrimitiveField &WR) {
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int i = 1; i < TOTAL_X - 1; i++) {
      int id = idx(i, j), idL = idx(i - 1, j), idR = idx(i + 1, j);

      muscl_recon(W.rho[idL], W.rho[id], W.rho[idR], WL.rho[id], WR.rho[id]);
      muscl_recon(W.u[idL],   W.u[id],   W.u[idR],   WL.u[id],   WR.u[id]);
      muscl_recon(W.v[idL],   W.v[id],   W.v[idR],   WL.v[id],   WR.v[id]);
      muscl_recon(W.p[idL],   W.p[id],   W.p[idR],   WL.p[id],   WR.p[id]);
    }
  }
}

void musclReconstruct_y(const PrimitiveField &W, PrimitiveField &WB,
                        PrimitiveField &WT) {
  for (int j = 1; j < TOTAL_Y - 1; j++) {
    for (int i = GHOST; i < NX + GHOST; i++) {
      int id = idx(i, j), idB = idx(i, j - 1), idT = idx(i, j + 1);

      muscl_recon(W.rho[idB], W.rho[id], W.rho[idT], WB.rho[id], WT.rho[id]);
      muscl_recon(W.u[idB],   W.u[id],   W.u[idT],   WB.u[id],   WT.u[id]);
      muscl_recon(W.v[idB],   W.v[id],   W.v[idT],   WB.v[id],   WT.v[id]);
      muscl_recon(W.p[idB],   W.p[id],   W.p[idT],   WB.p[id],   WT.p[id]);
    }
  }
}

// ============================================================================
//  Hancock 半步演化 (全场) — 调用共享 hancock_evolve
// ============================================================================
void halfTimeEvolution_x(const PrimitiveField &WL_face,
                         const PrimitiveField &WR_face,
                         PrimitiveField &WL_half, PrimitiveField &WR_half,
                         Real dt) {
  Real factor = (Real)0.5 * dt / DX;
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int i = 1; i < TOTAL_X - 1; i++) {
      int id = idx(i, j);
      Real Wlf[4] = {WL_face.rho[id], WL_face.u[id], WL_face.v[id], WL_face.p[id]};
      Real Wrf[4] = {WR_face.rho[id], WR_face.u[id], WR_face.v[id], WR_face.p[id]};

      Real WLh[4], WRh[4];
      hancock_evolve(Wlf, Wlf, Wrf, WLh, factor, 1, 2);  // 共享函数
      hancock_evolve(Wrf, Wlf, Wrf, WRh, factor, 1, 2);

      WL_half.rho[id] = WLh[0]; WL_half.u[id] = WLh[1];
      WL_half.v[id]   = WLh[2]; WL_half.p[id] = WLh[3];
      WR_half.rho[id] = WRh[0]; WR_half.u[id] = WRh[1];
      WR_half.v[id]   = WRh[2]; WR_half.p[id] = WRh[3];
    }
  }
}

void halfTimeEvolution_y(const PrimitiveField &WB_face,
                         const PrimitiveField &WT_face,
                         PrimitiveField &WB_half, PrimitiveField &WT_half,
                         Real dt) {
  Real factor = (Real)0.5 * dt / DY;
  for (int j = 1; j < TOTAL_Y - 1; j++) {
    for (int i = GHOST; i < NX + GHOST; i++) {
      int id = idx(i, j);
      Real Wbf[4] = {WB_face.rho[id], WB_face.u[id], WB_face.v[id], WB_face.p[id]};
      Real Wtf[4] = {WT_face.rho[id], WT_face.u[id], WT_face.v[id], WT_face.p[id]};

      Real WBh[4], WTh[4];
      hancock_evolve(Wbf, Wbf, Wtf, WBh, factor, 2, 1);  // 共享函数
      hancock_evolve(Wtf, Wbf, Wtf, WTh, factor, 2, 1);

      WB_half.rho[id] = WBh[0]; WB_half.u[id] = WBh[1];
      WB_half.v[id]   = WBh[2]; WB_half.p[id] = WBh[3];
      WT_half.rho[id] = WTh[0]; WT_half.u[id] = WTh[1];
      WT_half.v[id]   = WTh[2]; WT_half.p[id] = WTh[3];
    }
  }
}

// ============================================================================
//  HLLC 通量计算 (全场) — 调用共享 hllc_flux_x / hllc_flux_y
// ============================================================================
void hllcFluxField_x(const PrimitiveField &WL, const PrimitiveField &WR,
                     FluxField &flux) {
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int i = GHOST - 1; i < NX + GHOST; i++) {
      int idL = idx(i, j), idR = idx(i + 1, j);

      Real wl[4] = {WL.rho[idL], WL.u[idL], WL.v[idL], WL.p[idL]};
      Real wr[4] = {WR.rho[idR], WR.u[idR], WR.v[idR], WR.p[idR]};
      Real f[4];
      hllc_flux_x(wl, wr, f);  // 共享函数

      flux.f0[idL] = f[0]; flux.f1[idL] = f[1];
      flux.f2[idL] = f[2]; flux.f3[idL] = f[3];
    }
  }
}

void hllcFluxField_y(const PrimitiveField &WB, const PrimitiveField &WT,
                     FluxField &flux) {
  for (int j = GHOST - 1; j < NY + GHOST; j++) {
    for (int i = GHOST; i < NX + GHOST; i++) {
      int idB = idx(i, j), idT = idx(i, j + 1);

      Real wb[4] = {WB.rho[idB], WB.u[idB], WB.v[idB], WB.p[idB]};
      Real wt[4] = {WT.rho[idT], WT.u[idT], WT.v[idT], WT.p[idT]};
      Real g[4];
      hllc_flux_y(wb, wt, g);  // 共享函数

      flux.f0[idB] = g[0]; flux.f1[idB] = g[1];
      flux.f2[idB] = g[2]; flux.f3[idB] = g[3];
    }
  }
}

// ============================================================================
//  更新守恒变量
// ============================================================================
void updateConservative_x(ConservativeField &U, const FluxField &flux, Real dt) {
  Real factor = dt / DX;
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int i = GHOST; i < NX + GHOST; i++) {
      int id = idx(i, j), id_lf = idx(i - 1, j);
      U.rho[id]  -= factor * (flux.f0[id] - flux.f0[id_lf]);
      U.rhou[id] -= factor * (flux.f1[id] - flux.f1[id_lf]);
      U.rhov[id] -= factor * (flux.f2[id] - flux.f2[id_lf]);
      U.E[id]    -= factor * (flux.f3[id] - flux.f3[id_lf]);
    }
  }
}

void updateConservative_y(ConservativeField &U, const FluxField &flux, Real dt) {
  Real factor = dt / DY;
  for (int j = GHOST; j < NY + GHOST; j++) {
    for (int i = GHOST; i < NX + GHOST; i++) {
      int id = idx(i, j), id_bf = idx(i, j - 1);
      U.rho[id]  -= factor * (flux.f0[id] - flux.f0[id_bf]);
      U.rhou[id] -= factor * (flux.f1[id] - flux.f1[id_bf]);
      U.rhov[id] -= factor * (flux.f2[id] - flux.f2[id_bf]);
      U.E[id]    -= factor * (flux.f3[id] - flux.f3[id_bf]);
    }
  }
}

// ============================================================================
//  主程序
// ============================================================================
int main() {
  std::cout << "Grid: " << NX << " x " << NY << std::endl;
  std::cout << "Domain: x=[" << X_MIN << ", " << X_MAX << "], y=["
            << Y_MIN << ", " << Y_MAX << "]" << std::endl;
  std::cout << "T_end = " << T_END << "s" << std::endl << std::endl;

  int sz = TOTAL_SIZE;

  ConservativeField U(sz);
  PrimitiveField W(sz);

  PrimitiveField WL_face(sz), WR_face(sz);
  PrimitiveField WL_half(sz), WR_half(sz);

  FluxField flux_x(sz), flux_y(sz);

  initializeShockBubble(U, W);
  applyBoundaryConditions(U);

  Real t = (Real)0.0;
  int step = 0;

  while (t < T_END) {
    conservativeToPrimitive(U, W);
    Real dt = computeTimeStep(W);
    if (t + dt > T_END) dt = T_END - t;

    // fp32 安全: dt 太小无法推进 t 时直接退出
    if (dt <= (Real)0.0 || t + dt == t) break;

    // 交替方向分裂
    if (step % 2 == 0) {
      // X → Y
      musclReconstruct_x(W, WL_face, WR_face);
      halfTimeEvolution_x(WL_face, WR_face, WL_half, WR_half, dt);
      hllcFluxField_x(WR_half, WL_half, flux_x);
      updateConservative_x(U, flux_x, dt);
      applyBoundaryConditions(U);

      conservativeToPrimitive(U, W);

      musclReconstruct_y(W, WL_face, WR_face);
      halfTimeEvolution_y(WL_face, WR_face, WL_half, WR_half, dt);
      hllcFluxField_y(WR_half, WL_half, flux_y);
      updateConservative_y(U, flux_y, dt);
      applyBoundaryConditions(U);
    } else {
      // Y → X
      musclReconstruct_y(W, WL_face, WR_face);
      halfTimeEvolution_y(WL_face, WR_face, WL_half, WR_half, dt);
      hllcFluxField_y(WR_half, WL_half, flux_y);
      updateConservative_y(U, flux_y, dt);
      applyBoundaryConditions(U);

      conservativeToPrimitive(U, W);

      musclReconstruct_x(W, WL_face, WR_face);
      halfTimeEvolution_x(WL_face, WR_face, WL_half, WR_half, dt);
      hllcFluxField_x(WR_half, WL_half, flux_x);
      updateConservative_x(U, flux_x, dt);
      applyBoundaryConditions(U);
    }

    t += dt;
    step++;
    if (step % 100 == 0)
      std::cout << "Step " << step << ": t = " << t << std::endl;
  }

  std::cout << "\nDone. Steps: " << step << std::endl;

  // 输出结果
  conservativeToPrimitive(U, W);
  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : (std::string("results/euler/2d/") + prec_name);
  std::system(("mkdir -p " + out_dir).c_str());
  std::ostringstream fname;
  fname << out_dir << "/cpu_shock_bubble.dat";
  std::ofstream out(fname.str());
  out << std::scientific;
  out.precision(std::numeric_limits<Real>::max_digits10);
  out << "# Time = " << t << "\n";
  out << "# Steps = " << step << "\n";
  out << "# T_END = " << (double)T_END << "\n";
  for (int i = GHOST; i < NX + GHOST; i++) {
    Real x = X_MIN + (i - GHOST + 0.5) * DX;
    for (int j = GHOST; j < NY + GHOST; j++) {
      Real y = Y_MIN + (j - GHOST + 0.5) * DY;
      int id = idx(i, j);
      out << x << " " << y << " " << W.rho[id] << " " << W.u[id] << " "
          << W.v[id] << " " << W.p[id] << "\n";
    }
    out << "\n";
  }
  out.close();
  std::cout << "Results saved to '" << fname.str() << "'" << std::endl;

  return 0;
}
