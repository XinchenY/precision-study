/* ============================================================================
 *  riemann1d_raptor.cpp — Euler 1D Riemann (Sod), region 化重构 (Ch10 §10.3 3e)
 *
 *  与 src/euler/riemann1d_cpu.cpp 数值逐位等价的机械重构, 六 region 与 MHD 1D
 *  版一一对应: recovery / cfl / recon / hancock / flux / update。
 *  注意: cfl region 保留原版自己的内联压强表达式 (0.5*rho*u*u), 它与 toPrim
 *  的 (0.5*ru*ru/rho) 代数等价但舍入路径不同, 不能共用 recovery 的 W。
 *
 *  原生 (提取合法性验证):
 *    /lsc/opt/llvm-20.1/bin/clang++ -O2 -std=c++14 -I include \
 *        src/euler/riemann1d_raptor.cpp -o bin/raptor/riemann1d_extracted
 *  RAPTOR 插桩:
 *    scripts/raptor-clang++ -O2 -std=c++14 -DUSE_RAPTOR -I include \
 *        src/euler/riemann1d_raptor.cpp -o bin/raptor/riemann1d_raptor
 *
 *  用法: ./riemann1d_raptor [N] [test_id] [solver] [regions]
 *    regions = 逗号分隔 {recovery,cfl,recon,hancock,flux,update} / none
 * ==========================================================================*/

#include "euler/riemann.hpp"
#ifdef USE_RAPTOR
#include <raptor/raptor.h>
#endif

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct TestCase {
  const char* name;
  Real rhoL, uL, pL;
  Real rhoR, uR, pR;
  Real T_end;
  Real x0;
};

static TestCase tests[] = {
  {"Sod", (Real)1.0, (Real)0.0, (Real)1.0,
   (Real)0.125, (Real)0.0, (Real)0.1, (Real)0.25, (Real)0.5},
  {"123_problem", (Real)1.0, (Real)-2.0, (Real)0.4,
   (Real)1.0, (Real)2.0, (Real)0.4, (Real)0.15, (Real)0.5},
  {"Left_blast", (Real)1.0, (Real)0.0, (Real)1000.0,
   (Real)1.0, (Real)0.0, (Real)0.01, (Real)0.012, (Real)0.5},
  {"Right_blast", (Real)1.0, (Real)0.0, (Real)0.01,
   (Real)1.0, (Real)0.0, (Real)100.0, (Real)0.035, (Real)0.5},
  {"Two_shocks", (Real)5.99924, (Real)19.5975, (Real)460.894,
   (Real)5.99242, (Real)-6.19633, (Real)46.0950, (Real)0.035, (Real)0.4},
};

/* ── 六个 region 函数 (与原版逐运算对应) ─────────────────────────────────── */
extern "C" {

/* toPrim 逐胞一次: W[1]=ru/rho, W[2]=(g-1)*(E-0.5*ru*ru/rho), 与原版一致 */
__attribute__((noinline))
void eRegionRecovery(const Real* Ur, const Real* Uru, const Real* UE,
                     Real* Wr, Real* Wu, Real* Wp, int total) {
  for (int i = 0; i < total; i++) {
    Real rho = Ur[i], ru = Uru[i], E = UE[i];
    Wr[i] = rho;
    Wu[i] = ru / rho;
    Wp[i] = (GAMMA - (Real)1.0) * (E - (Real)0.5 * ru * ru / rho);
  }
}

/* CFL: 原版内联表达式 (u=ru/rho; p=(g-1)*(E-0.5*rho*u*u)), 不用 W */
__attribute__((noinline))
Real eRegionCfl(const Real* Ur, const Real* Uru, const Real* UE,
                int nx, int ghost) {
  Real max_speed = (Real)0.0;
  for (int i = ghost; i < nx + ghost; i++) {
    Real rho = Ur[i], ru = Uru[i], E = UE[i];
    Real u = ru / rho;
    Real p = (GAMMA - (Real)1.0) * (E - (Real)0.5 * rho * u * u);
    Real c = soundSpeed(p, rho);
    max_speed = std::max(max_speed, std::abs(u) + c);
  }
  return max_speed;
}

/* 每胞 MUSCL 重构 (原版对每个界面重复计算相邻胞, 值相同; 此处每胞一次) */
__attribute__((noinline))
void eRegionRecon(const Real* Wr, const Real* Wu, const Real* Wp,
                  Real* Lr, Real* Lu, Real* Lp, Real* Rr, Real* Ru, Real* Rp,
                  int nx, int ghost) {
  for (int i = ghost - 1; i <= nx + ghost; i++) {
    muscl_recon(Wr[i - 1], Wr[i], Wr[i + 1], Lr[i], Rr[i]);
    muscl_recon(Wu[i - 1], Wu[i], Wu[i + 1], Lu[i], Ru[i]);
    muscl_recon(Wp[i - 1], Wp[i], Wp[i + 1], Lp[i], Rp[i]);
  }
}

/* Hancock 半步: 每胞两个面值 (原版 hancock1d 的逐运算复制) */
__attribute__((noinline))
void eRegionHancock(const Real* Lr, const Real* Lu, const Real* Lp,
                    const Real* Rr, const Real* Ru, const Real* Rp,
                    Real factor,
                    Real* HLr, Real* HLu, Real* HLp,
                    Real* HRr, Real* HRu, Real* HRp, int nx, int ghost) {
  for (int i = ghost - 1; i <= nx + ghost; i++) {
    Real rL = Lr[i], uL = Lu[i], pL = Lp[i];
    Real rR = Rr[i], uR = Ru[i], pR = Rp[i];
    Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rL * uL * uL;
    Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rR * uR * uR;
    Real dF0 = rR * uR - rL * uL;
    Real dF1 = (rR * uR * uR + pR) - (rL * uL * uL + pL);
    Real dF2 = (ER + pR) * uR - (EL + pL) * uL;

    /* 左面半步 (原版 hancock1d(W0_Lface, ...) 分支: Uf 由左面值构造) */
    {
      Real Uf0 = rL, Uf1 = rL * uL;
      Real Uf2 = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rL * uL * uL;
      Real Uh0 = Uf0 - factor * dF0;
      Real Uh1 = Uf1 - factor * dF1;
      Real Uh2 = Uf2 - factor * dF2;
      HLr[i] = Uh0;
      HLu[i] = Uh1 / Uh0;
      HLp[i] = (GAMMA - (Real)1.0) * (Uh2 - (Real)0.5 * Uh1 * Uh1 / Uh0);
    }
    /* 右面半步 */
    {
      Real Uf0 = rR, Uf1 = rR * uR;
      Real Uf2 = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rR * uR * uR;
      Real Uh0 = Uf0 - factor * dF0;
      Real Uh1 = Uf1 - factor * dF1;
      Real Uh2 = Uf2 - factor * dF2;
      HRr[i] = Uh0;
      HRu[i] = Uh1 / Uh0;
      HRp[i] = (GAMMA - (Real)1.0) * (Uh2 - (Real)0.5 * Uh1 * Uh1 / Uh0);
    }
  }
}

__attribute__((noinline))
void eRegionFlux(const Real* HLr, const Real* HLu, const Real* HLp,
                 const Real* HRr, const Real* HRu, const Real* HRp,
                 Real* Fr, Real* Fru, Real* FE, int nx, int ghost,
                 int use_hll) {
  for (int i = ghost - 1; i < nx + ghost; i++) {
    Real WL[3] = {HRr[i], HRu[i], HRp[i]};
    Real WR[3] = {HLr[i + 1], HLu[i + 1], HLp[i + 1]};
    Real f[3];
    if (use_hll) hll_flux_1d(WL, WR, f);
    else hllc_flux_1d(WL, WR, f);
    Fr[i] = f[0];
    Fru[i] = f[1];
    FE[i] = f[2];
  }
}

__attribute__((noinline))
void eRegionUpdate(Real* Ur, Real* Uru, Real* UE, const Real* Fr,
                   const Real* Fru, const Real* FE, Real ratio,
                   int nx, int ghost) {
  for (int i = ghost; i < nx + ghost; i++) {
    Ur[i] -= ratio * (Fr[i] - Fr[i - 1]);
    Uru[i] -= ratio * (Fru[i] - Fru[i - 1]);
    UE[i] -= ratio * (FE[i] - FE[i - 1]);
  }
}

}  /* extern "C" */

typedef void (*ERecoveryFn)(const Real*, const Real*, const Real*, Real*,
                            Real*, Real*, int);
typedef Real (*ECflFn)(const Real*, const Real*, const Real*, int, int);
typedef void (*EReconFn)(const Real*, const Real*, const Real*, Real*, Real*,
                         Real*, Real*, Real*, Real*, int, int);
typedef void (*EHancockFn)(const Real*, const Real*, const Real*, const Real*,
                           const Real*, const Real*, Real, Real*, Real*, Real*,
                           Real*, Real*, Real*, int, int);
typedef void (*EFluxFn)(const Real*, const Real*, const Real*, const Real*,
                        const Real*, const Real*, Real*, Real*, Real*, int,
                        int, int);
typedef void (*EUpdateFn)(Real*, Real*, Real*, const Real*, const Real*,
                          const Real*, Real, int, int);

int main(int argc, char* argv[]) {
  int NX1D = 200;
  int test_id = 1;
  std::string solver = "hllc";
  std::string regions = "none";

  if (argc > 1) NX1D = std::atoi(argv[1]);
  if (argc > 2) test_id = std::atoi(argv[2]);
  if (argc > 3) solver = argv[3];
  if (argc > 4) regions = argv[4];

  if (test_id < 1 || test_id > 5) {
    std::cerr << "Error: test_id must be 1-5" << std::endl;
    return 1;
  }
  bool use_hll = (solver == "hll" || solver == "HLL");
  const TestCase& tc = tests[test_id - 1];

  auto wants = [&](const char* name) {
    return regions.find(name) != std::string::npos;
  };

  ERecoveryFn fRecovery = eRegionRecovery;
  ECflFn fCfl = eRegionCfl;
  EReconFn fRecon = eRegionRecon;
  EHancockFn fHancock = eRegionHancock;
  EFluxFn fFlux = eRegionFlux;
  EUpdateFn fUpdate = eRegionUpdate;

#ifdef USE_RAPTOR
  if (wants("recovery")) fRecovery = __raptor_truncate_op_func(eRegionRecovery, 64, 0, 32);
  if (wants("cfl"))      fCfl      = __raptor_truncate_op_func(eRegionCfl, 64, 0, 32);
  if (wants("recon"))    fRecon    = __raptor_truncate_op_func(eRegionRecon, 64, 0, 32);
  if (wants("hancock"))  fHancock  = __raptor_truncate_op_func(eRegionHancock, 64, 0, 32);
  if (wants("flux"))     fFlux     = __raptor_truncate_op_func(eRegionFlux, 64, 0, 32);
  if (wants("update"))   fUpdate   = __raptor_truncate_op_func(eRegionUpdate, 64, 0, 32);
#else
  if (regions != "none" && regions != "") {
    std::cerr << "Warning: built without USE_RAPTOR, region list ignored."
              << std::endl;
  }
#endif

  const int GHOST1D = 2;
  const int TOTAL1D = NX1D + 2 * GHOST1D;
  const Real X1D_MIN = (Real)0.0, X1D_MAX = (Real)1.0;
  const Real DX1D = (X1D_MAX - X1D_MIN) / (Real)NX1D;
  const Real CFL1D = (Real)0.8;

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::cout << "===== 1D Riemann Test (region build) =====" << std::endl;
  std::cout << "Test: " << tc.name << " (id=" << test_id << ")" << std::endl;
  std::cout << "Solver: " << (use_hll ? "HLL" : "HLLC") << std::endl;
  std::cout << "Truncated regions: " << regions << std::endl;
  std::cout << "N = " << NX1D << ", dx = " << DX1D << std::endl;

  std::vector<Real> U_rho(TOTAL1D), U_rhou(TOTAL1D), U_E(TOTAL1D);
  for (int i = 0; i < TOTAL1D; i++) {
    Real x = X1D_MIN + (Real)(i - GHOST1D + 0.5) * DX1D;
    Real rho, u, p;
    if (x < tc.x0) { rho = tc.rhoL; u = tc.uL; p = tc.pL; }
    else { rho = tc.rhoR; u = tc.uR; p = tc.pR; }
    U_rho[i] = rho;
    U_rhou[i] = rho * u;
    U_E[i] = p / (GAMMA - (Real)1.0) + (Real)0.5 * rho * u * u;
  }

  std::vector<Real> F_rho(TOTAL1D), F_rhou(TOTAL1D), F_E(TOTAL1D);
  std::vector<Real> Wr(TOTAL1D), Wu(TOTAL1D), Wp(TOTAL1D);
  std::vector<Real> Lr(TOTAL1D), Lu(TOTAL1D), Lp(TOTAL1D);
  std::vector<Real> Rr(TOTAL1D), Ru(TOTAL1D), Rp(TOTAL1D);
  std::vector<Real> HLr(TOTAL1D), HLu(TOTAL1D), HLp(TOTAL1D);
  std::vector<Real> HRr(TOTAL1D), HRu(TOTAL1D), HRp(TOTAL1D);

  Real t = (Real)0.0;
  int step = 0;

  while (t < tc.T_end) {
    for (int g = 0; g < GHOST1D; g++) {
      U_rho[g] = U_rho[GHOST1D];
      U_rhou[g] = U_rhou[GHOST1D];
      U_E[g] = U_E[GHOST1D];
      int ir = TOTAL1D - 1 - g;
      int inner = TOTAL1D - 1 - GHOST1D;
      U_rho[ir] = U_rho[inner];
      U_rhou[ir] = U_rhou[inner];
      U_E[ir] = U_E[inner];
    }

    Real max_speed = fCfl(U_rho.data(), U_rhou.data(), U_E.data(),
                          NX1D, GHOST1D);
    Real dt = CFL1D * DX1D / max_speed;
    if (t + dt > tc.T_end) dt = tc.T_end - t;

    fRecovery(U_rho.data(), U_rhou.data(), U_E.data(),
              Wr.data(), Wu.data(), Wp.data(), TOTAL1D);
    fRecon(Wr.data(), Wu.data(), Wp.data(), Lr.data(), Lu.data(), Lp.data(),
           Rr.data(), Ru.data(), Rp.data(), NX1D, GHOST1D);
    fHancock(Lr.data(), Lu.data(), Lp.data(), Rr.data(), Ru.data(), Rp.data(),
             (Real)0.5 * dt / DX1D, HLr.data(), HLu.data(), HLp.data(),
             HRr.data(), HRu.data(), HRp.data(), NX1D, GHOST1D);
    fFlux(HLr.data(), HLu.data(), HLp.data(), HRr.data(), HRu.data(),
          HRp.data(), F_rho.data(), F_rhou.data(), F_E.data(), NX1D, GHOST1D,
          use_hll ? 1 : 0);
    fUpdate(U_rho.data(), U_rhou.data(), U_E.data(), F_rho.data(),
            F_rhou.data(), F_E.data(), dt / DX1D, NX1D, GHOST1D);

    t += dt;
    step++;
  }

  std::cout << "Done. Steps: " << step << std::endl;

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/euler/raptor");
  if (std::system(("mkdir -p " + out_dir).c_str()) != 0) { /* ignore */ }
  std::string tag = (regions == "none" || regions.empty()) ? "none" : regions;
  for (char& c : tag) if (c == ',') c = '+';
  std::ostringstream fname;
  fname << out_dir << "/toro" << test_id << "_" << solver << "_" << prec_name
        << "_" << NX1D << "_r-" << tag << ".dat";
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
