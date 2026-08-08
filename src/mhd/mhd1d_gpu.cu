/* ============================================================================
 *  mhd1d_gpu.cu — 1D ideal MHD Brio-Wu shock tube on the GPU (CUDA)
 *
 *  与 src/mhd/mhd1d_cpu.cpp 逐步对应的 CUDA 版本: 同一份 include/mhd/
 *  mhd_common.hpp 提供全部 __host__ __device__ 通量/换算函数, 本文件只写
 *  kernel 驱动。算法与 CPU 完全一致:
 *    transmissive BC -> CFL(dt) -> MUSCL-Hancock 预测 -> HLLD/HLL 通量
 *    (含 HLL fallback 状态码, 设备端原子计数) -> 守恒更新 -> 正定性检查
 *  数据布局: U[cell*8 + k] (cell-major), 与设备函数的 Real[8] 接口直接兼容。
 *
 *  Build (必须 -arch=sm_80, 见项目记录; 不开 --use_fast_math):
 *    nvcc -O2 -std=c++14 -arch=sm_80 -I include src/mhd/mhd1d_gpu.cu \
 *         -o bin/mhd1d_gpu_fp64
 *    nvcc -O2 -std=c++14 -arch=sm_80 -I include -DUSE_FLOAT \
 *         src/mhd/mhd1d_gpu.cu -o bin/mhd1d_gpu_fp32
 *  Run (与 CPU 相同的参数约定):
 *    OUTDIR=results/mhd/gpu/brio_wu ./bin/mhd1d_gpu_fp64 800 0.1 0.4 muscl hlld
 * ==========================================================================*/

#include "mhd/mhd_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << cudaGetErrorString(err) << std::endl;                     \
      std::exit(9);                                                          \
    }                                                                        \
  } while (0)

static const int BLK = 128;

// ── 与 CPU 版逐行对应的设备端辅助函数 ────────────────────────────────────────
__device__ bool primitiveIsUsableDev(const Real W[MHD_NVAR]) {
  if (W[MHD_PR_RHO] <= MHD_RHO_FLOOR || W[MHD_PR_P] <= MHD_P_FLOOR) return false;
  for (int k = 0; k < MHD_NVAR; k++) {
    if (!isfinite((double)W[k])) return false;
  }
  return true;
}

__device__ bool stateIsUsableDev(const Real U[MHD_NVAR]) {
  Real W[MHD_NVAR];
  if (U[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  mhdConsToPrim(U, W);
  if (W[MHD_PR_RHO] <= (Real)0.0 || W[MHD_PR_P] <= (Real)0.0) return false;
  for (int k = 0; k < MHD_NVAR; k++) {
    if (!isfinite((double)U[k]) || !isfinite((double)W[k])) return false;
  }
  return true;
}

__device__ void reconstructPrimitiveDev(const Real Wm[MHD_NVAR],
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
  if (!primitiveIsUsableDev(Wleft) || !primitiveIsUsableDev(Wright)) {
    for (int k = 0; k < MHD_NVAR; k++) {
      Wleft[k] = W0[k];
      Wright[k] = W0[k];
    }
  }
}

__device__ void hancockPredictCellDev(const Real Wleft[MHD_NVAR],
                                      const Real Wright[MHD_NVAR],
                                      Real factor,
                                      Real ULH[MHD_NVAR],
                                      Real URH[MHD_NVAR]) {
  Real UL[MHD_NVAR], UR[MHD_NVAR];
  Real FL[MHD_NVAR], FR[MHD_NVAR];
  mhdPrimToCons(Wleft, UL);
  mhdPrimToCons(Wright, UR);
  mhdFluxX(UL, FL);
  mhdFluxX(UR, FR);

  for (int k = 0; k < MHD_NVAR; k++) {
    Real correction = factor * (FR[k] - FL[k]);
    ULH[k] = UL[k] - correction;
    URH[k] = UR[k] - correction;
  }

  Real WLH[MHD_NVAR], WRH[MHD_NVAR];
  mhdConsToPrim(ULH, WLH);
  mhdConsToPrim(URH, WRH);
  if (ULH[MHD_RHO] <= MHD_RHO_FLOOR || !primitiveIsUsableDev(WLH)) {
    for (int k = 0; k < MHD_NVAR; k++) ULH[k] = UL[k];
  }
  if (URH[MHD_RHO] <= MHD_RHO_FLOOR || !primitiveIsUsableDev(WRH)) {
    for (int k = 0; k < MHD_NVAR; k++) URH[k] = UR[k];
  }
}

// ── kernels ─────────────────────────────────────────────────────────────────
// Transmissive BC: 把最近的内部格点复制进 ghost cells。 Launch: <<<1, ghost>>>
__global__ void bcKernel(Real* U, int ghost, int total) {
  int g = threadIdx.x;
  if (g >= ghost) return;
  int left_inner = ghost;
  int right_inner = total - 1 - ghost;
  int right_ghost = total - 1 - g;
  for (int k = 0; k < MHD_NVAR; k++) {
    U[g * MHD_NVAR + k] = U[left_inner * MHD_NVAR + k];
    U[right_ghost * MHD_NVAR + k] = U[right_inner * MHD_NVAR + k];
  }
}

// 每 block 归约出局部最大信号速度 |vx|+cf, 部分结果拷回 host 求全局最大。
__global__ void maxSpeedKernel(const Real* U, int ghost, int nx, Real* part) {
  extern __shared__ unsigned char smem_raw[];
  Real* smem = reinterpret_cast<Real*>(smem_raw);
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;      // 0..nx-1 (interior index)
  Real speed = (Real)0.0;
  if (i < nx) {
    Real W[MHD_NVAR];
    mhdConsToPrim(&U[(size_t)(i + ghost) * MHD_NVAR], W);
    speed = mhdAbs(W[MHD_PR_VX]) + mhdFastSpeedX(W);
  }
  smem[tid] = speed;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = mhdMax(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  if (tid == 0) part[blockIdx.x] = smem[0];
}

// MUSCL-Hancock 预测: i in [ghost-1, nx+ghost] (含端点), 与 CPU 循环一致。
__global__ void predictKernel(const Real* U, Real* Uleft, Real* Uright,
                              Real factor, int ghost, int nx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + (ghost - 1);
  if (i > nx + ghost) return;
  Real Wm[MHD_NVAR], W0[MHD_NVAR], Wp[MHD_NVAR];
  Real Wl[MHD_NVAR], Wr[MHD_NVAR];
  mhdConsToPrim(&U[(size_t)(i - 1) * MHD_NVAR], Wm);
  mhdConsToPrim(&U[(size_t)i * MHD_NVAR], W0);
  mhdConsToPrim(&U[(size_t)(i + 1) * MHD_NVAR], Wp);
  reconstructPrimitiveDev(Wm, W0, Wp, Wl, Wr);
  hancockPredictCellDev(Wl, Wr, factor,
                        &Uleft[(size_t)i * MHD_NVAR],
                        &Uright[(size_t)i * MHD_NVAR]);
}

// 界面通量: i in [ghost-1, nx+ghost), F[i] = flux(i, i+1)。
// counters = {evaluations, degenerate, nonphysical} (仅 HLLD 时计数)。
__global__ void fluxKernel(const Real* U, const Real* Uleft, const Real* Uright,
                           Real* F, int ghost, int nx, int use_muscl,
                           int use_hlld, unsigned long long* counters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + (ghost - 1);
  if (i >= nx + ghost) return;
  const Real* UL = use_muscl ? &Uright[(size_t)i * MHD_NVAR]
                             : &U[(size_t)i * MHD_NVAR];
  const Real* UR = use_muscl ? &Uleft[(size_t)(i + 1) * MHD_NVAR]
                             : &U[(size_t)(i + 1) * MHD_NVAR];
  Real UiL[MHD_NVAR], UiR[MHD_NVAR], Fi[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) { UiL[k] = UL[k]; UiR[k] = UR[k]; }
  if (use_hlld) {
    int status = mhdHlldFluxX(UiL, UiR, Fi);
    atomicAdd(&counters[0], 1ULL);
    if (status == MHD_HLLD_FALLBACK_DEGENERATE) atomicAdd(&counters[1], 1ULL);
    if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) atomicAdd(&counters[2], 1ULL);
  } else {
    mhdHllFluxX(UiL, UiR, Fi);
  }
  for (int k = 0; k < MHD_NVAR; k++) F[(size_t)i * MHD_NVAR + k] = Fi[k];
}

// 守恒更新 + 正定性检查: bad_cell 用 atomicMin 记录最左侧坏格点。
__global__ void updateKernel(Real* U, const Real* F, Real ratio,
                             int ghost, int nx, int* bad_cell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + ghost;
  if (i >= nx + ghost) return;
  Real Ui[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) {
    Ui[k] = U[(size_t)i * MHD_NVAR + k]
          - ratio * (F[(size_t)i * MHD_NVAR + k]
                     - F[(size_t)(i - 1) * MHD_NVAR + k]);
    U[(size_t)i * MHD_NVAR + k] = Ui[k];
  }
  if (!stateIsUsableDev(Ui)) atomicMin(bad_cell, i - ghost);
}

// ── host driver ─────────────────────────────────────────────────────────────
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

  const int ghost = 2;
  const int total = nx + 2 * ghost;
  const Real x_min = (Real)0.0, x_max = (Real)1.0;
  const Real x0 = (Real)0.5;
  const Real dx = (x_max - x_min) / (Real)nx;

  std::cout << "===== 1D ideal MHD Brio-Wu shock tube (GPU) =====" << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL") << std::endl;
  std::cout << "Reconstruction: "
            << (use_muscl ? "MUSCL-Hancock" : "first-order") << std::endl;
  std::cout << "Precision: " << (int)(sizeof(Real) * 8) << " bit"
            << (sizeof(Real) == 8 ? " (fp64)" : " (fp32)") << std::endl;
  std::cout << "gamma = " << MHD_GAMMA << std::endl;
  std::cout << "N = " << nx << ", dx = " << dx << ", CFL = " << cfl << std::endl;

  // Brio-Wu 初始条件 (与 CPU 相同)。
  Real WL[MHD_NVAR] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0};
  Real WR[MHD_NVAR] = {0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0};
  Real UL[MHD_NVAR], UR[MHD_NVAR];
  mhdPrimToCons(WL, UL);
  mhdPrimToCons(WR, UR);

  std::vector<Real> hU((size_t)total * MHD_NVAR);
  for (int i = 0; i < total; i++) {
    Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
    const Real* src = (x < x0) ? UL : UR;
    for (int k = 0; k < MHD_NVAR; k++) hU[(size_t)i * MHD_NVAR + k] = src[k];
  }

  const size_t nbytes = (size_t)total * MHD_NVAR * sizeof(Real);
  Real *dU, *dF, *dUl, *dUr, *dPart;
  int* dBad;
  unsigned long long* dCnt;
  const int nblk_speed = (nx + BLK - 1) / BLK;
  CUDA_CHECK(cudaMalloc(&dU, nbytes));
  CUDA_CHECK(cudaMalloc(&dF, nbytes));
  CUDA_CHECK(cudaMalloc(&dUl, nbytes));
  CUDA_CHECK(cudaMalloc(&dUr, nbytes));
  CUDA_CHECK(cudaMalloc(&dPart, (size_t)nblk_speed * sizeof(Real)));
  CUDA_CHECK(cudaMalloc(&dBad, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dCnt, 3 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(dCnt, 0, 3 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemcpy(dU, hU.data(), nbytes, cudaMemcpyHostToDevice));

  std::vector<Real> hPart(nblk_speed);
  Real t = (Real)0.0;
  int step = 0;
  const int INT_BIG = 0x7fffffff;

  while (t < t_end) {
    bcKernel<<<1, ghost>>>(dU, ghost, total);

    maxSpeedKernel<<<nblk_speed, BLK, BLK * sizeof(Real)>>>(dU, ghost, nx,
                                                            dPart);
    CUDA_CHECK(cudaMemcpy(hPart.data(), dPart,
                          (size_t)nblk_speed * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    Real max_speed = (Real)0.0;
    for (int b = 0; b < nblk_speed; b++)
      max_speed = mhdMax(max_speed, hPart[b]);
    if (max_speed <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal speed." << std::endl;
      return 2;
    }
    Real dt = cfl * dx / max_speed;
    if (t + dt > t_end) dt = t_end - t;

    if (use_muscl) {
      int n = (nx + ghost) - (ghost - 1) + 1;   // cells ghost-1 .. nx+ghost
      predictKernel<<<(n + BLK - 1) / BLK, BLK>>>(
          dU, dUl, dUr, (Real)0.5 * dt / dx, ghost, nx);
    }

    {
      int n = (nx + ghost) - (ghost - 1);   // interfaces ghost-1 .. nx+ghost-1
      fluxKernel<<<(n + BLK - 1) / BLK, BLK>>>(
          dU, dUl, dUr, dF, ghost, nx, use_muscl, use_hlld, dCnt);
    }

    CUDA_CHECK(cudaMemcpy(dBad, &INT_BIG, sizeof(int), cudaMemcpyHostToDevice));
    updateKernel<<<(nx + BLK - 1) / BLK, BLK>>>(dU, dF, dt / dx, ghost, nx,
                                                dBad);
    int bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, dBad, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad != INT_BIG) {
      std::cerr << "Error: non-physical state at step " << step
                << ", cell " << bad << std::endl;
      return 3;
    }

    t += dt;
    step++;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long hCnt[3] = {0, 0, 0};
  CUDA_CHECK(cudaMemcpy(hCnt, dCnt, 3 * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hU.data(), dU, nbytes, cudaMemcpyDeviceToHost));

  std::cout << "Done. Steps: " << step << ", final t = " << t << std::endl;
  if (use_hlld) {
    unsigned long long fb = hCnt[1] + hCnt[2];
    double frac = hCnt[0] > 0 ? (double)fb / (double)hCnt[0] : 0.0;
    std::cout << "HLLD flux evaluations = " << hCnt[0] << std::endl;
    std::cout << "HLLD relative fallback tolerance = "
              << std::scientific << (double)mhdHlldTolerance()
              << std::defaultfloat << std::endl;
    std::cout << "HLLD->HLL fallbacks = " << fb
              << " (degenerate=" << hCnt[1] << ", nonphysical=" << hCnt[2]
              << "), fraction = " << std::scientific << frac
              << std::defaultfloat << std::endl;
  }

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/gpu/brio_wu");
  std::string mk = "mkdir -p " + out_dir;
  if (std::system(mk.c_str()) != 0) { /* ignore */ }

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::ostringstream fname;
  fname << out_dir << "/brio_wu_" << order << "_" << solver << "_"
        << prec_name << "_N" << nx << ".dat";

  std::ofstream out(fname.str());
  if (!out) {
    std::cerr << "Error: failed to open output file: " << fname.str()
              << std::endl;
    return 4;
  }
  out << "# Brio-Wu ideal MHD (GPU)\n";
  out << "# nx " << nx << " t " << t << " gamma " << MHD_GAMMA
      << " cfl " << cfl << " order " << order << " solver " << solver << "\n";
  if (use_hlld) {
    unsigned long long fb = hCnt[1] + hCnt[2];
    double frac = hCnt[0] > 0 ? (double)fb / (double)hCnt[0] : 0.0;
    out << "# hlld_evaluations " << hCnt[0]
        << " hlld_tolerance " << mhdHlldTolerance()
        << " hlld_fallbacks " << fb
        << " hlld_degenerate " << hCnt[1]
        << " hlld_nonphysical " << hCnt[2]
        << " hlld_fallback_fraction " << frac << "\n";
  }
  out << "# columns: x rho vx vy vz p Bx By Bz E\n";
  out << std::scientific
      << std::setprecision(std::numeric_limits<Real>::max_digits10);
  for (int i = ghost; i < nx + ghost; i++) {
    Real Ui[MHD_NVAR], Wi[MHD_NVAR];
    for (int k = 0; k < MHD_NVAR; k++) Ui[k] = hU[(size_t)i * MHD_NVAR + k];
    mhdConsToPrim(Ui, Wi);
    Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
    out << x << " " << Wi[MHD_PR_RHO] << " " << Wi[MHD_PR_VX] << " "
        << Wi[MHD_PR_VY] << " " << Wi[MHD_PR_VZ] << " " << Wi[MHD_PR_P] << " "
        << Wi[MHD_PR_BX] << " " << Wi[MHD_PR_BY] << " " << Wi[MHD_PR_BZ]
        << " " << Ui[MHD_E] << "\n";
  }
  std::cout << "Saved: " << fname.str() << std::endl;

  cudaFree(dU); cudaFree(dF); cudaFree(dUl); cudaFree(dUr);
  cudaFree(dPart); cudaFree(dBad); cudaFree(dCnt);
  return 0;
}