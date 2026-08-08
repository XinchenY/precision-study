/**
 * ============================================================================
 *   1D Riemann 测试 (MUSCL-Hancock + HLLC/HLL)
 * ============================================================================
 *
 * 使用与 2D solver 完全相同的共享函数 (riemann.hpp)
 * 精度: 由 config.hpp 中的 typedef Real 控制 (double 或 float)
 *
 * Toro (2009) Table 4.1 — 5 个标准 Riemann 问题:
 *   Test 1: Sod shock tube  (标准 Sod)
 *   Test 2: 123 problem     (two rarefactions, near-vacuum)
 *   Test 3: Left blast wave (pL=1000, pR=0.01)
 *   Test 4: Right blast wave (pL=0.01, pR=100)
 *   Test 5: Two strong shocks collision
 *
 * 用法:
 *   ./bin/riemann1d              → Test 1 (Sod), N=200, HLLC
 *   ./bin/riemann1d 200 3 hll    → Test 3, N=200, HLL solver
 *   ./bin/riemann1d 400 1 hllc   → Test 1, N=400, HLLC solver
 *
 * 输出: results/toro{T}_{solver}_{prec}_{N}.dat
 * ============================================================================
 */

#include "euler/riemann.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
//  Toro Table 4.1: 5 个标准 Riemann 问题
//  {name, rhoL, uL, pL, rhoR, uR, pR, T_end, x0}
// ============================================================================
struct TestCase {
  const char* name;
  Real rhoL, uL, pL;
  Real rhoR, uR, pR;
  Real T_end;
  Real x0;   // 间断初始位置
};

// tests 是一个数组, 存了 5 个 TestCase 结构体
// tests[0] = Test 1, tests[1] = Test 2, ..., tests[4] = Test 5
TestCase tests[] = {
  // Test 1: Sod shock tube (标准 Sod)
  {"Sod",
   (Real)1.0,      (Real)0.0,   (Real)1.0,
   (Real)0.125,    (Real)0.0,   (Real)0.1,
   (Real)0.25,     (Real)0.5},

  // Test 2: 123 problem (two rarefactions, near-vacuum)
  {"123_problem",
   (Real)1.0,      (Real)-2.0,  (Real)0.4,
   (Real)1.0,      (Real)2.0,   (Real)0.4,
   (Real)0.15,     (Real)0.5},

  // Test 3: Left blast wave (pL=1000, pR=0.01)
  {"Left_blast",
   (Real)1.0,      (Real)0.0,   (Real)1000.0,
   (Real)1.0,      (Real)0.0,   (Real)0.01,
   (Real)0.012,    (Real)0.5},

  // Test 4: Right blast wave (pL=0.01, pR=100)
  {"Right_blast",
   (Real)1.0,      (Real)0.0,   (Real)0.01,
   (Real)1.0,      (Real)0.0,   (Real)100.0,
   (Real)0.035,    (Real)0.5},

  // Test 5: Two strong shocks collision
  {"Two_shocks",
   (Real)5.99924,  (Real)19.5975,    (Real)460.894,
   (Real)5.99242,  (Real)-6.19633,   (Real)46.0950,
   (Real)0.035,    (Real)0.4},
};

int main(int argc, char* argv[]) {
  // 参数: [N] [test_id] [solver]
  int NX1D = 200;
  int test_id = 1;           // 默认 Test 1 (Sod)
  std::string solver = "hllc";

  if (argc > 1) NX1D = std::atoi(argv[1]);
  if (argc > 2) test_id = std::atoi(argv[2]);
  if (argc > 3) solver = argv[3];

  if (test_id < 1 || test_id > 5) {
    std::cerr << "Error: test_id must be 1-5 (Toro Table 4.1)" << std::endl;
    return 1;
  }

  bool use_hll = (solver == "hll" || solver == "HLL");

  const TestCase& tc = tests[test_id - 1];  // test_id=1 → tests[0]

  const int GHOST1D = 2;
  const int TOTAL1D = NX1D + 2 * GHOST1D;
  const Real X1D_MIN = (Real)0.0, X1D_MAX = (Real)1.0;
  const Real DX1D = (X1D_MAX - X1D_MIN) / (Real)NX1D;
  const Real CFL1D = (Real)0.8;

  int prec_bits = (int)(sizeof(Real) * 8);
  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";

  std::cout << "===== 1D Riemann Test =====" << std::endl;
  std::cout << "Test: " << tc.name << " (id=" << test_id << ")" << std::endl;
  std::cout << "Solver: " << (use_hll ? "HLL" : "HLLC") << std::endl;
  std::cout << "Precision: " << prec_bits << " bit (" << prec_name << ")" << std::endl;
  std::cout << "N = " << NX1D << ", dx = " << DX1D << std::endl;
  std::cout << "Left:  rho=" << tc.rhoL << " u=" << tc.uL << " p=" << tc.pL << std::endl;
  std::cout << "Right: rho=" << tc.rhoR << " u=" << tc.uR << " p=" << tc.pR << std::endl;
  std::cout << "T_end = " << tc.T_end << ", x0 = " << tc.x0 << std::endl;

  // 守恒变量: rho, rho*u, E
  std::vector<Real> U_rho(TOTAL1D), U_rhou(TOTAL1D), U_E(TOTAL1D);

  // 初始条件
  for (int i = 0; i < TOTAL1D; i++) {
    Real x = X1D_MIN + (Real)(i - GHOST1D + 0.5) * DX1D;
    Real rho, u, p;
    if (x < tc.x0) {
      rho = tc.rhoL; u = tc.uL; p = tc.pL;
    } else {
      rho = tc.rhoR; u = tc.uR; p = tc.pR;
    }
    U_rho[i]  = rho;
    U_rhou[i] = rho * u;
    U_E[i]    = p / (GAMMA - (Real)1.0) + (Real)0.5 * rho * u * u;
  }

  std::vector<Real> F_rho(TOTAL1D), F_rhou(TOTAL1D), F_E(TOTAL1D);

  Real t = (Real)0.0;
  int step = 0;

  while (t < tc.T_end) {
    // 边界条件 (透射)
    for (int g = 0; g < GHOST1D; g++) {
      U_rho[g]  = U_rho[GHOST1D];
      U_rhou[g] = U_rhou[GHOST1D]; 
      U_E[g]    = U_E[GHOST1D];

      int ir = TOTAL1D - 1 - g;
      int inner = TOTAL1D - 1 - GHOST1D;
      U_rho[ir]  = U_rho[inner];
      U_rhou[ir] = U_rhou[inner];
      U_E[ir]    = U_E[inner];
    }

    // CFL: dt = CFL * dx / max(|u| + c)
    Real max_speed = (Real)0.0;
    for (int i = GHOST1D; i < NX1D + GHOST1D; i++) {
      Real rho = U_rho[i], ru = U_rhou[i], E = U_E[i];
      Real u = ru / rho;
      Real p = (GAMMA - (Real)1.0) * (E - (Real)0.5 * rho * u * u);
      Real c = soundSpeed(p, rho);
      max_speed = std::max(max_speed, std::abs(u) + c);
    }
    Real dt = CFL1D * DX1D / max_speed;
    if (t + dt > tc.T_end) dt = tc.T_end - t;

    // MUSCL + Hancock + HLLC/HLL
    for (int i = GHOST1D - 1; i < NX1D + GHOST1D; i++) {
      int i_LL = i - 1, i_L = i, i_0 = i + 1, i_R = i + 2;

      auto toPrim = [&](int idx, Real W[3]) {
        Real rho = U_rho[idx], ru = U_rhou[idx], E = U_E[idx];
        W[0] = rho;
        W[1] = ru / rho;
        W[2] = (GAMMA - (Real)1.0) * (E - (Real)0.5 * ru * ru / rho);
      };

      Real W_LL[3], W_L[3], W_0[3], W_R[3];
      toPrim(i_LL, W_LL);
      toPrim(i_L,  W_L);
      toPrim(i_0,  W_0);
      toPrim(i_R,  W_R);

      // MUSCL 重构
      Real WL_Lface[3], WL_Rface[3], W0_Lface[3], W0_Rface[3];
      for (int k = 0; k < 3; k++) {
        muscl_recon(W_LL[k], W_L[k], W_0[k], WL_Lface[k], WL_Rface[k]);
        muscl_recon(W_L[k],  W_0[k], W_R[k], W0_Lface[k], W0_Rface[k]);
      }

      // Hancock 半步 (1D)
      Real factor = (Real)0.5 * dt / DX1D;

      auto hancock1d = [&](const Real Wface[3], const Real Wlf[3],
                           const Real Wrf[3], Real Whalf[3]) {
        Real Uf[3] = {Wface[0], Wface[0]*Wface[1],
          (Real)(Wface[2]/(GAMMA-(Real)1.0) + (Real)0.5*Wface[0]*Wface[1]*Wface[1])};

        Real rL = Wlf[0], uL = Wlf[1], pL = Wlf[2];
        Real rR = Wrf[0], uR = Wrf[1], pR = Wrf[2];
        Real EL = pL/(GAMMA-(Real)1.0) + (Real)0.5*rL*uL*uL;
        Real ER = pR/(GAMMA-(Real)1.0) + (Real)0.5*rR*uR*uR;

        Real dF[3];
        dF[0] = rR*uR - rL*uL;
        dF[1] = (rR*uR*uR + pR) - (rL*uL*uL + pL);
        dF[2] = (ER+pR)*uR - (EL+pL)*uL;

        Real Uh[3];
        for (int k = 0; k < 3; k++) Uh[k] = Uf[k] - factor * dF[k];

        Whalf[0] = Uh[0];
        Whalf[1] = Uh[1] / Uh[0];
        Whalf[2] = (GAMMA-(Real)1.0) * (Uh[2] - (Real)0.5*Uh[1]*Uh[1]/Uh[0]);
      };

      Real WL_half[3], W0_half[3];
      hancock1d(WL_Rface, WL_Lface, WL_Rface, WL_half);
      hancock1d(W0_Lface, W0_Lface, W0_Rface, W0_half);

      // Riemann solver: HLLC 或 HLL
      Real f[3];
      if (use_hll) {
        hll_flux_1d(WL_half, W0_half, f);
      } else {
        hllc_flux_1d(WL_half, W0_half, f);
      }

      F_rho[i]  = f[0];
      F_rhou[i] = f[1];
      F_E[i]    = f[2];
    }

    Real ratio = dt / DX1D;
    for (int i = GHOST1D; i < NX1D + GHOST1D; i++) {
      U_rho[i]  -= ratio * (F_rho[i]  - F_rho[i-1]);
      U_rhou[i] -= ratio * (F_rhou[i] - F_rhou[i-1]);
      U_E[i]    -= ratio * (F_E[i]    - F_E[i-1]);
    }

    t += dt;
    step++;
  }

  std::cout << "Done. Steps: " << step << std::endl;

  // 输出: ${OUTDIR:-results}/toro{T}_{solver}_{prec}_{N}.dat
  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env) : std::string("results/euler");
  std::system(("mkdir -p " + out_dir).c_str());
  std::ostringstream fname;
  fname << out_dir << "/toro" << test_id << "_" << solver << "_" << prec_name << "_" << NX1D << ".dat";
  std::ofstream out(fname.str());
  out << std::scientific;
  out.precision(std::numeric_limits<Real>::max_digits10);
  for (int i = GHOST1D; i < NX1D + GHOST1D; i++) {
    Real x = X1D_MIN + (Real)(i - GHOST1D + 0.5) * DX1D;
    Real rho = U_rho[i], ru = U_rhou[i], E = U_E[i];
    Real u = ru / rho;
    Real p = (GAMMA - (Real)1.0) * (E - (Real)0.5 * rho * u * u);
    out << x << " " << rho << " " << u << " " << p << "\n";
  }
  out.close();
  std::cout << "Saved: " << fname.str() << std::endl;
  return 0;
}
