/**
 * ============================================================================
 *   2D Riemann 问题测试 (MUSCL-Hancock + HLLC)
 * ============================================================================
 *
 * 使用与 shock-bubble solver 完全相同的共享数值函数 (riemann.hpp)
 * 精度: 由 config.hpp 中的 typedef Real 控制 (double 或 float)
 *
 * 参考: Liska, R. & Wendroff, B. (2003)
 *       "Comparison of Several Difference Schemes on 1D and 2D Test Problems
 *        for the Euler Equations"
 *       SIAM J. Sci. Comput., 25(3), 995-1017, Table 4.3
 *
 * 四象限初始条件:
 *   Config 3:  4 个激波交汇         (T=0.3)
 *   Config 6:  4 个接触间断 + 涡旋  (T=0.3)
 *   Config 12: 2 激波 + 2 接触间断  (T=0.25)
 *
 * 用法:
 *   ./bin/riemann2d              → Config 6, 400x400
 *   ./bin/riemann2d 400 3        → Config 3, 400x400
 *   ./bin/riemann2d 800 12       → Config 12, 800x800
 *
 * 输出: results/riemann2d_cfg{C}_{prec}_{N}.dat
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
#include <string>
#include <vector>

// ============================================================================
//  四象限初始条件 (Liska & Wendroff 2003, Table 4.3)
//  象限编号:
//    Q1 = 右上 (x>0.5, y>0.5)   Q2 = 左上 (x<0.5, y>0.5)
//    Q3 = 左下 (x<0.5, y<0.5)   Q4 = 右下 (x>0.5, y<0.5)
// ============================================================================
struct QuadState {
  Real rho, u, v, p;
};

struct Config2D {
  const char* name;
  QuadState Q1, Q2, Q3, Q4;  // 右上, 左上, 左下, 右下
  Real T_end;
};

Config2D configs[] = {
  // Config 3: 4 个激波交汇
  {"cfg3_4shocks",
   {(Real)0.5197, (Real)-0.6259, (Real)-0.3109, (Real)0.4},    // Q1 右上
   {(Real)1.0,    (Real)0.1,     (Real)-0.3109, (Real)1.0},    // Q2 左上
   {(Real)0.8,    (Real)0.1,     (Real)0.4276,  (Real)0.4},    // Q3 左下
   {(Real)0.5197, (Real)-0.6259, (Real)0.4276,  (Real)0.4},    // Q4 右下
   (Real)0.3},

  // Config 6: 4 个接触间断 + 涡旋 (最经典, 展示精度差异效果好)
  {"cfg6_4contacts",
   {(Real)1.0, (Real)0.75,  (Real)-0.5, (Real)1.0},   // Q1 右上
   {(Real)2.0, (Real)0.75,  (Real)0.5,  (Real)1.0},   // Q2 左上
   {(Real)1.0, (Real)-0.75, (Real)0.5,  (Real)1.0},   // Q3 左下
   {(Real)3.0, (Real)-0.75, (Real)-0.5, (Real)1.0},   // Q4 右下
   (Real)0.3},

  // Config 12: 2 激波 + 2 接触间断
  {"cfg12_2S2C",
   {(Real)0.5313, (Real)0.0,    (Real)0.0,    (Real)0.4},   // Q1 右上
   {(Real)1.0,    (Real)0.7276, (Real)0.0,    (Real)1.0},   // Q2 左上
   {(Real)0.8,    (Real)0.0,    (Real)0.0,    (Real)1.0},   // Q3 左下
   {(Real)1.0,    (Real)0.0,    (Real)0.7276, (Real)1.0},   // Q4 右下
   (Real)0.25},
};

// 把 config_id (3/6/12) 映射到数组索引
int configIndex(int cfg_id) {
  switch (cfg_id) {
    case 3:  return 0;
    case 6:  return 1;
    case 12: return 2;
    default: return -1;
  }
}

// ============================================================================
//  主程序 — 自带网格参数, 不依赖 config.hpp 的网格宏
// ============================================================================
int main(int argc, char* argv[]) {

  // 参数: [N] [config_id]
  int N = 400;         // 默认 400x400
  int cfg_id = 6;      // 默认 Config 6

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) cfg_id = std::atoi(argv[2]);

  int ci = configIndex(cfg_id);
  if (ci < 0) {
    std::cerr << "Error: config_id must be 3, 6, or 12" << std::endl;
    return 1;
  }

  const Config2D& cfg = configs[ci];

  // ---- 本地网格参数 (正方形 [0,1]x[0,1]) ----
  const int NX2 = N, NY2 = N;
  const int GH = 2;
  const int TX = NX2 + 2 * GH;
  const int TY = NY2 + 2 * GH;
  const int SZ = TX * TY;
  const Real x_min = (Real)0.0, x_max = (Real)1.0;
  const Real y_min = (Real)0.0, y_max = (Real)1.0;
  const Real dx = (x_max - x_min) / (Real)NX2;
  const Real dy = (y_max - y_min) / (Real)NY2;
  const Real cfl = (Real)0.8;

  // 索引函数
  auto ID = [&](int i, int j) { return j * TX + i; };

  int prec_bits = (int)(sizeof(Real) * 8);
  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
#ifdef USE_HLL
  const char* solver_name = "hll";
#else
  const char* solver_name = "hllc";
#endif

  std::cout << "===== 2D Riemann Problem =====" << std::endl;
  std::cout << "Config: " << cfg.name << " (id=" << cfg_id << ")" << std::endl;
  std::cout << "Precision: " << prec_bits << " bit (" << prec_name << ")" << std::endl;
  std::cout << "Solver: " << solver_name << std::endl;
  std::cout << "Grid: " << NX2 << " x " << NY2 << ", dx=" << dx << std::endl;
  std::cout << "T_end = " << cfg.T_end << std::endl;
  std::cout << "==============================" << std::endl;

  // ---- 分配 SoA ----
  std::vector<Real> rho(SZ), rhou(SZ), rhov(SZ), E(SZ);  // 守恒量
  std::vector<Real> Wr(SZ), Wu(SZ), Wv(SZ), Wp(SZ);      // 原始量
  // MUSCL 重构面值 + Hancock 半步
  std::vector<Real> WLr(SZ),WLu(SZ),WLv(SZ),WLp(SZ);
  std::vector<Real> WRr(SZ),WRu(SZ),WRv(SZ),WRp(SZ);
  std::vector<Real> WLhr(SZ),WLhu(SZ),WLhv(SZ),WLhp(SZ);
  std::vector<Real> WRhr(SZ),WRhu(SZ),WRhv(SZ),WRhp(SZ);
  // 通量
  std::vector<Real> Fx0(SZ),Fx1(SZ),Fx2(SZ),Fx3(SZ);
  std::vector<Real> Fy0(SZ),Fy1(SZ),Fy2(SZ),Fy3(SZ);

  // ---- 初始条件: 四象限 ----
  for (int j = 0; j < TY; j++) {
    Real y = y_min + (j - GH + (Real)0.5) * dy;
    for (int i = 0; i < TX; i++) {
      Real x = x_min + (i - GH + (Real)0.5) * dx;
      int id = ID(i, j);

      const QuadState* qs;
      if      (x >= (Real)0.5 && y >= (Real)0.5) qs = &cfg.Q1;  // 右上
      else if (x <  (Real)0.5 && y >= (Real)0.5) qs = &cfg.Q2;  // 左上
      else if (x <  (Real)0.5 && y <  (Real)0.5) qs = &cfg.Q3;  // 左下
      else                                        qs = &cfg.Q4;  // 右下

      Wr[id] = qs->rho; Wu[id] = qs->u; Wv[id] = qs->v; Wp[id] = qs->p;

      Real W4[4] = {qs->rho, qs->u, qs->v, qs->p};
      Real U4[4];
      primToCons(W4, U4);
      rho[id] = U4[0]; rhou[id] = U4[1]; rhov[id] = U4[2]; E[id] = U4[3];
    }
  }

  // ---- 宏函数: 守恒→原始, 边界, CFL ----
  auto cons2prim = [&]() {
    for (int j = 0; j < TY; j++)
      for (int i = 0; i < TX; i++) {
        int id = ID(i,j);
        Real U4[4] = {rho[id], rhou[id], rhov[id], E[id]};
        Real W4[4];
        consToPrim(U4, W4);
        Wr[id]=W4[0]; Wu[id]=W4[1]; Wv[id]=W4[2]; Wp[id]=W4[3];
      }
  };

  // 透射边界
  auto applyBC = [&]() {
    for (int j = GH; j < NY2+GH; j++) {
      for (int g = 0; g < GH; g++) {
        int gL = ID(g,j), iL = ID(GH,j);
        rho[gL]=rho[iL]; rhou[gL]=rhou[iL]; rhov[gL]=rhov[iL]; E[gL]=E[iL];
        int gR = ID(TX-1-g,j), iR = ID(TX-1-GH,j);
        rho[gR]=rho[iR]; rhou[gR]=rhou[iR]; rhov[gR]=rhov[iR]; E[gR]=E[iR];
      }
    }
    for (int g = 0; g < GH; g++)
      for (int i = 0; i < TX; i++) {
        int gB = ID(i,g), iB = ID(i,GH);
        rho[gB]=rho[iB]; rhou[gB]=rhou[iB]; rhov[gB]=rhov[iB]; E[gB]=E[iB];
        int gT = ID(i,TY-1-g), iT = ID(i,TY-1-GH);
        rho[gT]=rho[iT]; rhou[gT]=rhou[iT]; rhov[gT]=rhov[iT]; E[gT]=E[iT];
      }
  };

  auto computeDt = [&]() {
    Real mx = (Real)0, my = (Real)0;
    for (int j = GH; j < NY2+GH; j++)
      for (int i = GH; i < NX2+GH; i++) {
        int id = ID(i,j);
        Real c = soundSpeed(Wp[id], Wr[id]);
        mx = std::max(mx, std::abs(Wu[id]) + c);
        my = std::max(my, std::abs(Wv[id]) + c);
      }
    return cfl * std::min(dx / mx, dy / my);
  };

  // ---- 重构 + 演化 + 通量 + 更新 (x方向) ----
  auto sweepX = [&](Real dt) {
    // MUSCL x
    for (int j = GH; j < NY2+GH; j++)
      for (int i = 1; i < TX-1; i++) {
        int id=ID(i,j), iL=ID(i-1,j), iR=ID(i+1,j);
        muscl_recon(Wr[iL],Wr[id],Wr[iR], WLr[id],WRr[id]);
        muscl_recon(Wu[iL],Wu[id],Wu[iR], WLu[id],WRu[id]);
        muscl_recon(Wv[iL],Wv[id],Wv[iR], WLv[id],WRv[id]);
        muscl_recon(Wp[iL],Wp[id],Wp[iR], WLp[id],WRp[id]);
      }
    // Hancock x
    Real fac = (Real)0.5 * dt / dx;
    for (int j = GH; j < NY2+GH; j++)
      for (int i = 1; i < TX-1; i++) {
        int id = ID(i,j);
        Real Wlf[4]={WLr[id],WLu[id],WLv[id],WLp[id]};
        Real Wrf[4]={WRr[id],WRu[id],WRv[id],WRp[id]};
        Real WLh[4], WRh[4];
        hancock_evolve(Wlf, Wlf, Wrf, WLh, fac, 1, 2);
        hancock_evolve(Wrf, Wlf, Wrf, WRh, fac, 1, 2);
        WLhr[id]=WLh[0]; WLhu[id]=WLh[1]; WLhv[id]=WLh[2]; WLhp[id]=WLh[3];
        WRhr[id]=WRh[0]; WRhu[id]=WRh[1]; WRhv[id]=WRh[2]; WRhp[id]=WRh[3];
      }
    // HLLC x
    for (int j = GH; j < NY2+GH; j++)
      for (int i = GH-1; i < NX2+GH; i++) {
        int idL=ID(i,j), idR=ID(i+1,j);
        Real wl[4]={WRhr[idL],WRhu[idL],WRhv[idL],WRhp[idL]};
        Real wr[4]={WLhr[idR],WLhu[idR],WLhv[idR],WLhp[idR]};
        Real f[4];
#ifdef USE_HLL
        hll_flux_x(wl, wr, f);
#else
        hllc_flux_x(wl, wr, f);
#endif
        Fx0[idL]=f[0]; Fx1[idL]=f[1]; Fx2[idL]=f[2]; Fx3[idL]=f[3];
      }
    // 更新 x
    Real rx = dt / dx;
    for (int j = GH; j < NY2+GH; j++)
      for (int i = GH; i < NX2+GH; i++) {
        int id=ID(i,j), il=ID(i-1,j);
        rho[id]  -= rx*(Fx0[id]-Fx0[il]);
        rhou[id] -= rx*(Fx1[id]-Fx1[il]);
        rhov[id] -= rx*(Fx2[id]-Fx2[il]);
        E[id]    -= rx*(Fx3[id]-Fx3[il]);
      }
  };

  // ---- 重构 + 演化 + 通量 + 更新 (y方向) ----
  auto sweepY = [&](Real dt) {
    // MUSCL y
    for (int j = 1; j < TY-1; j++)
      for (int i = GH; i < NX2+GH; i++) {
        int id=ID(i,j), iB=ID(i,j-1), iT=ID(i,j+1);
        muscl_recon(Wr[iB],Wr[id],Wr[iT], WLr[id],WRr[id]);
        muscl_recon(Wu[iB],Wu[id],Wu[iT], WLu[id],WRu[id]);
        muscl_recon(Wv[iB],Wv[id],Wv[iT], WLv[id],WRv[id]);
        muscl_recon(Wp[iB],Wp[id],Wp[iT], WLp[id],WRp[id]);
      }
    // Hancock y
    Real fac = (Real)0.5 * dt / dy;
    for (int j = 1; j < TY-1; j++)
      for (int i = GH; i < NX2+GH; i++) {
        int id = ID(i,j);
        Real Wbf[4]={WLr[id],WLu[id],WLv[id],WLp[id]};
        Real Wtf[4]={WRr[id],WRu[id],WRv[id],WRp[id]};
        Real WBh[4], WTh[4];
        hancock_evolve(Wbf, Wbf, Wtf, WBh, fac, 2, 1);
        hancock_evolve(Wtf, Wbf, Wtf, WTh, fac, 2, 1);
        WLhr[id]=WBh[0]; WLhu[id]=WBh[1]; WLhv[id]=WBh[2]; WLhp[id]=WBh[3];
        WRhr[id]=WTh[0]; WRhu[id]=WTh[1]; WRhv[id]=WTh[2]; WRhp[id]=WTh[3];
      }
    // HLLC y
    for (int j = GH-1; j < NY2+GH; j++)
      for (int i = GH; i < NX2+GH; i++) {
        int idB=ID(i,j), idT=ID(i,j+1);
        Real wb[4]={WRhr[idB],WRhu[idB],WRhv[idB],WRhp[idB]};
        Real wt[4]={WLhr[idT],WLhu[idT],WLhv[idT],WLhp[idT]};
        Real g[4];
#ifdef USE_HLL
        hll_flux_y(wb, wt, g);
#else
        hllc_flux_y(wb, wt, g);
#endif
        Fy0[idB]=g[0]; Fy1[idB]=g[1]; Fy2[idB]=g[2]; Fy3[idB]=g[3];
      }
    // 更新 y
    Real ry = dt / dy;
    for (int j = GH; j < NY2+GH; j++)
      for (int i = GH; i < NX2+GH; i++) {
        int id=ID(i,j), ib=ID(i,j-1);
        rho[id]  -= ry*(Fy0[id]-Fy0[ib]);
        rhou[id] -= ry*(Fy1[id]-Fy1[ib]);
        rhov[id] -= ry*(Fy2[id]-Fy2[ib]);
        E[id]    -= ry*(Fy3[id]-Fy3[ib]);
      }
  };

  // ---- 时间推进 (交替方向分裂) ----
  applyBC();

  Real t = (Real)0.0;
  int step = 0;

  while (t < cfg.T_end) {
    cons2prim();
    Real dt = computeDt();
    if (t + dt > cfg.T_end) dt = cfg.T_end - t;
    if (dt <= (Real)0.0 || t + dt == t) break;

    if (step % 2 == 0) {
      sweepX(dt); applyBC(); cons2prim();
      sweepY(dt); applyBC();
    } else {
      sweepY(dt); applyBC(); cons2prim();
      sweepX(dt); applyBC();
    }

    t += dt;
    step++;
    if (step % 200 == 0)
      std::cout << "Step " << step << ": t = " << t << std::endl;
  }

  std::cout << "\nDone. Steps: " << step << std::endl;

  // ---- 输出 ----
  cons2prim();

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env) : (std::string("results/euler/2d/") + prec_name);
  std::system(("mkdir -p " + out_dir).c_str());
  std::ostringstream fname;
  fname << out_dir << "/riemann2d_" << cfg.name << "_" << solver_name << "_" << N << ".dat";

  std::ofstream out(fname.str());
  out << std::scientific;
  out.precision(std::numeric_limits<Real>::max_digits10);
  for (int j = GH; j < NY2+GH; j++) {
    for (int i = GH; i < NX2+GH; i++) {
      Real x = x_min + (i - GH + (Real)0.5) * dx;
      Real y = y_min + (j - GH + (Real)0.5) * dy;
      int id = ID(i,j);
      out << x << " " << y << " " << Wr[id] << " "
          << Wu[id] << " " << Wv[id] << " " << Wp[id] << "\n";
    }
  }
  out.close();
  std::cout << "Saved: " << fname.str() << std::endl;

  return 0;
}
