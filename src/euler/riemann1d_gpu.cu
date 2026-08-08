/**
 * ============================================================================
 *   riemann1d_gpu.cu  —  1D Riemann Test on GPU (MUSCL-Hancock + HLLC/HLL)
 * ============================================================================
 *
 *  注意: config.hpp 中有 #define NX 500, GHOST 2, DX ... 等宏 (针对2D)
 *  本文件所有局部变量全部用小写或 1D 后缀命名, 避免宏展开冲突.
 *
 *  输出: results/gpu/{fp64|fp32}/toro{T}_{solver}_{prec}_{N}.dat
 * ============================================================================
 */

#include "euler/riemann.hpp"   // Real, muscl_recon, hllc_flux_1d, hll_flux_1d
                                    // (已含 __host__ __device__ 标注)
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                              \
              __FILE__, __LINE__, cudaGetErrorString(_e));                    \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// ============================================================================
//  Toro Table 4.1 — 5 标准 Riemann 问题
// ============================================================================
struct TestCase {
  const char* name;
  Real rhoL, uL, pL;
  Real rhoR, uR, pR;
  Real Tend, x0;          // 避免与 config.hpp 的 T_END 宏冲突
};

static TestCase tests[] = {
  {"Sod",
   (Real)1.0,     (Real)0.0,    (Real)1.0,
   (Real)0.125,   (Real)0.0,    (Real)0.1,
   (Real)0.25,    (Real)0.5},
  {"123_problem",
   (Real)1.0,     (Real)-2.0,   (Real)0.4,
   (Real)1.0,     (Real)2.0,    (Real)0.4,
   (Real)0.15,    (Real)0.5},
  {"Left_blast",
   (Real)1.0,     (Real)0.0,    (Real)1000.0,
   (Real)1.0,     (Real)0.0,    (Real)0.01,
   (Real)0.012,   (Real)0.5},
  {"Right_blast",
   (Real)1.0,     (Real)0.0,    (Real)0.01,
   (Real)1.0,     (Real)0.0,    (Real)100.0,
   (Real)0.035,   (Real)0.5},
  {"Two_shocks",
   (Real)5.99924, (Real)19.5975, (Real)460.894,
   (Real)5.99242, (Real)-6.19633,(Real)46.0950,
   (Real)0.035,   (Real)0.4},
};

// ============================================================================
//  Device helper: Hancock 半步 (1D)
//  Wf — 要演化的面值, Wl/Wr — 该 cell 左右面值, Wh — 输出
// ============================================================================
__device__ __forceinline__ void hancock1d_dev(
    const Real Wf[3], const Real Wl[3], const Real Wr[3],
    Real Wh[3], Real fac)
{
  Real Uf0 = Wf[0];
  Real Uf1 = Wf[0] * Wf[1];
  Real Uf2 = Wf[2] / (GAMMA-(Real)1.0) + (Real)0.5*Wf[0]*Wf[1]*Wf[1];

  Real rL=Wl[0], uL=Wl[1], pL=Wl[2];
  Real rR=Wr[0], uR=Wr[1], pR=Wr[2];
  Real EL = pL/(GAMMA-(Real)1.0) + (Real)0.5*rL*uL*uL;
  Real ER = pR/(GAMMA-(Real)1.0) + (Real)0.5*rR*uR*uR;

  Real dF0 = rR*uR - rL*uL;
  Real dF1 = (rR*uR*uR+pR) - (rL*uL*uL+pL);
  Real dF2 = (ER+pR)*uR - (EL+pL)*uL;

  Real Uh0 = Uf0 - fac*dF0;
  Real Uh1 = Uf1 - fac*dF1;
  Real Uh2 = Uf2 - fac*dF2;

  Wh[0] = Uh0;
  Wh[1] = Uh1/Uh0;
  Wh[2] = (GAMMA-(Real)1.0)*(Uh2 - (Real)0.5*Uh1*Uh1/Uh0);
}

// ============================================================================
//  Kernel 1: transmissive 边界条件
//  参数全用小写 ng/tot 避免与 config 宏冲突
//  Launch: <<<1, ng>>>
// ============================================================================
__global__ void bcKernel(Real* rho, Real* rhou, Real* E, int ng, int tot)
{
  int g = threadIdx.x;
  if (g >= ng) return;
  rho[g]       = rho[ng];       rhou[g]       = rhou[ng];       E[g]       = E[ng];
  rho[tot-1-g] = rho[tot-1-ng]; rhou[tot-1-g] = rhou[tot-1-ng]; E[tot-1-g] = E[tot-1-ng];
}

// ============================================================================
//  Kernel 2: max 波速归约 (shared memory)
//  参数 ng/nx 避免宏冲突
//  Launch: <<<nblk, blksz, blksz*sizeof(Real)>>>
// ============================================================================
__global__ void maxSpeedKernel(
    const Real* rho, const Real* rhou, const Real* E,
    Real* d_out, int ng, int nx)
{
  extern __shared__ Real sdata[];
  int tid = threadIdx.x;
  int i   = blockIdx.x * blockDim.x + tid + ng;

  Real val = (Real)0.0;
  if (i < nx + ng) {
    Real r = rho[i], u = rhou[i]/r;
    Real p = (GAMMA-(Real)1.0)*(E[i] - (Real)0.5*r*u*u);
    Real c = sqrt(GAMMA*p/r);
    val = fabs(u) + c;
  }
  sdata[tid] = val;
  __syncthreads();

  for (int s = blockDim.x>>1; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid+s]);
    __syncthreads();
  }
  if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// ============================================================================
//  Kernel 3: 界面通量 (每线程一条界面)
//  界面 i: ng-1 ≤ i ≤ nx+ng-1, 使用 cells i-1..i+2
//  Launch: <<<ceil((nx+1)/blksz), blksz>>>
// ============================================================================
__global__ void fluxKernel(
    const Real* U_rho, const Real* U_rhou, const Real* U_E,
    Real* F_rho, Real* F_rhou, Real* F_E,
    Real dt, Real ds, int ng, int nx, int use_hll)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i   = idx + (ng - 1);
  if (i >= nx + ng) return;

  // 4 cells → 原始变量
  Real W_LL[3], W_L[3], W_0[3], W_R[3];
#define TOPRIM(k, W)                                                          \
  { Real _r=U_rho[k],_ru=U_rhou[k],_E=U_E[k];                               \
    (W)[0]=_r; (W)[1]=_ru/_r;                                                 \
    (W)[2]=(GAMMA-(Real)1.0)*(_E-(Real)0.5*_ru*_ru/_r); }
  TOPRIM(i-1, W_LL) TOPRIM(i,   W_L)
  TOPRIM(i+1, W_0)  TOPRIM(i+2, W_R)
#undef TOPRIM

  // MUSCL 重构
  Real WL_Lf[3], WL_Rf[3], W0_Lf[3], W0_Rf[3];
  for (int k = 0; k < 3; k++) {
    muscl_recon(W_LL[k], W_L[k], W_0[k], WL_Lf[k], WL_Rf[k]);
    muscl_recon(W_L[k],  W_0[k], W_R[k], W0_Lf[k], W0_Rf[k]);
  }

  // Hancock 半步 (与 CPU 一致)
  Real fac = (Real)0.5 * dt / ds;
  Real WL_half[3], W0_half[3];
  hancock1d_dev(WL_Rf, WL_Lf, WL_Rf, WL_half, fac);
  hancock1d_dev(W0_Lf, W0_Lf, W0_Rf, W0_half, fac);

  Real f[3];
  if (use_hll) hll_flux_1d(WL_half, W0_half, f);
  else         hllc_flux_1d(WL_half, W0_half, f);

  F_rho[i]=f[0]; F_rhou[i]=f[1]; F_E[i]=f[2];
}

// ============================================================================
//  Kernel 4: 守恒变量更新
//  Launch: <<<ceil(nx/blksz), blksz>>>
// ============================================================================
__global__ void updateKernel(
    Real* U_rho, Real* U_rhou, Real* U_E,
    const Real* F_rho, const Real* F_rhou, const Real* F_E,
    Real ratio, int ng, int nx)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i   = idx + ng;
  if (i >= nx + ng) return;
  U_rho[i]  -= ratio*(F_rho[i]  - F_rho[i-1]);
  U_rhou[i] -= ratio*(F_rhou[i] - F_rhou[i-1]);
  U_E[i]    -= ratio*(F_E[i]    - F_E[i-1]);
}

// ============================================================================
//  main
//  所有 1D 参数用 _1d 后缀或小写, 避免与 config.hpp 的 NX/GHOST/DX 宏冲突
// ============================================================================
int main(int argc, char* argv[]) {
  int nx1d = 200; int test_id = 1; std::string solver = "hllc";
  if (argc > 1) nx1d    = std::atoi(argv[1]);
  if (argc > 2) test_id = std::atoi(argv[2]);
  if (argc > 3) solver  = argv[3];

  if (test_id < 1 || test_id > 5) {
    std::cerr << "Error: test_id must be 1-5\n"; return 1;
  }
  bool use_hll = (solver == "hll" || solver == "HLL");
  const TestCase& tc = tests[test_id - 1];

  const int  ng1d   = 2;                        // ghost cells (避开 GHOST 宏)
  const int  tot1d  = nx1d + 2 * ng1d;
  const Real xmin1d = (Real)0.0;
  const Real xmax1d = (Real)1.0;
  const Real ds1d   = (xmax1d - xmin1d) / (Real)nx1d;  // 避开 DX 宏
  const Real cfl1d  = (Real)0.8;
  const char* prec  = (sizeof(Real) == 8) ? "fp64" : "fp32";

  std::cout << "===== 1D Riemann GPU =====\n"
            << "Test:      " << tc.name << "  (id=" << test_id << ")\n"
            << "Solver:    " << (use_hll ? "HLL" : "HLLC") << "\n"
            << "Precision: " << prec << "\n"
            << "N=" << nx1d << "  dx=" << ds1d << "\n";

  // ── 初始条件 ──────────────────────────────────────────────────────────────
  std::vector<Real> h_rho(tot1d), h_rhou(tot1d), h_E(tot1d);
  for (int i = 0; i < tot1d; i++) {
    Real x = xmin1d + (Real)(i - ng1d + 0.5) * ds1d;
    Real r, u, p;
    if (x < tc.x0) { r=tc.rhoL; u=tc.uL; p=tc.pL; }
    else            { r=tc.rhoR; u=tc.uR; p=tc.pR; }
    h_rho[i]  = r;
    h_rhou[i] = r*u;
    h_E[i]    = p/(GAMMA-(Real)1.0) + (Real)0.5*r*u*u;
  }

  // ── device 分配 & 上传 ────────────────────────────────────────────────────
  Real *d_rho, *d_rhou, *d_E, *d_Frho, *d_Frhou, *d_FE;
  size_t nb = (size_t)tot1d * sizeof(Real);
  CUDA_CHECK(cudaMalloc(&d_rho,   nb)); CUDA_CHECK(cudaMalloc(&d_rhou,  nb));
  CUDA_CHECK(cudaMalloc(&d_E,     nb)); CUDA_CHECK(cudaMalloc(&d_Frho,  nb));
  CUDA_CHECK(cudaMalloc(&d_Frhou, nb)); CUDA_CHECK(cudaMalloc(&d_FE,    nb));
  CUDA_CHECK(cudaMemcpy(d_rho,  h_rho.data(),  nb, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rhou, h_rhou.data(), nb, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_E,    h_E.data(),    nb, cudaMemcpyHostToDevice));

  const int blksz  = 256;
  const int nblk   = (nx1d + blksz - 1) / blksz;
  Real* d_part; CUDA_CHECK(cudaMalloc(&d_part, (size_t)nblk * sizeof(Real)));
  std::vector<Real> h_part(nblk);

  const int fblks = ((nx1d+1) + blksz-1) / blksz;
  const int ublks = (nx1d     + blksz-1) / blksz;

  // ── 时间推进 ──────────────────────────────────────────────────────────────
  Real t = (Real)0.0;
  int  step = 0;
  while (t < tc.Tend) {
    bcKernel<<<1, ng1d>>>(d_rho, d_rhou, d_E, ng1d, tot1d);
    CUDA_CHECK(cudaGetLastError());

    maxSpeedKernel<<<nblk, blksz, (size_t)blksz*sizeof(Real)>>>(
        d_rho, d_rhou, d_E, d_part, ng1d, nx1d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_part.data(), d_part,
                          (size_t)nblk*sizeof(Real), cudaMemcpyDeviceToHost));
    Real vmax = (Real)0.0;
    for (int b = 0; b < nblk; b++) vmax = std::max(vmax, h_part[b]);

    Real dt = cfl1d * ds1d / vmax;
    if (t + dt > tc.Tend) dt = tc.Tend - t;

    fluxKernel<<<fblks, blksz>>>(
        d_rho, d_rhou, d_E, d_Frho, d_Frhou, d_FE,
        dt, ds1d, ng1d, nx1d, use_hll ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    updateKernel<<<ublks, blksz>>>(
        d_rho, d_rhou, d_E, d_Frho, d_Frhou, d_FE,
        dt/ds1d, ng1d, nx1d);
    CUDA_CHECK(cudaGetLastError());

    t += dt; step++;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "Done. Steps: " << step << "\n";

  // ── 拷回 & 输出 ───────────────────────────────────────────────────────────
  CUDA_CHECK(cudaMemcpy(h_rho.data(),  d_rho,  nb, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_rhou.data(), d_rhou, nb, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_E.data(),    d_E,    nb, cudaMemcpyDeviceToHost));

  const char* out_env = std::getenv("OUTDIR");
  std::string dir = out_env ? std::string(out_env) : (std::string("results/euler/gpu/") + prec);
  system(("mkdir -p " + dir).c_str());

  std::ostringstream fname;
  fname << dir << "/toro" << test_id << "_" << solver
        << "_" << prec << "_" << nx1d << ".dat";
  std::ofstream out(fname.str());
  if (!out) { std::cerr << "Cannot open: " << fname.str() << "\n"; return 1; }
  out << std::scientific;
  out.precision(std::numeric_limits<Real>::max_digits10);
  for (int i = ng1d; i < nx1d + ng1d; i++) {
    Real x  = xmin1d + (Real)(i - ng1d + 0.5) * ds1d;
    Real r  = h_rho[i], ru = h_rhou[i], Ev = h_E[i];
    Real u  = ru/r;
    Real p  = (GAMMA-(Real)1.0)*(Ev - (Real)0.5*r*u*u);
    out << x << " " << r << " " << u << " " << p << "\n";
  }
  out.close();
  std::cout << "Saved: " << fname.str() << "\n";

  cudaFree(d_rho); cudaFree(d_rhou); cudaFree(d_E);
  cudaFree(d_Frho); cudaFree(d_Frhou); cudaFree(d_FE);
  cudaFree(d_part);
  return 0;
}
