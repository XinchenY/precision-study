/* ============================================================================
 *  mhd2d_glm_gpu.cu — 2D Orszag-Tang vortex, ideal MHD + mixed GLM (CUDA)
 *
 *  与 src/mhd/mhd2d_glm_cpu.cpp 逐步对应: 同一份 include/mhd/ 头文件提供
 *  全部 __host__ __device__ 通量/换算函数。算法一致:
 *    periodic BC -> ch/dt (两次归约) -> SSPRK2 两级
 *    每级: BC -> x/y 界面通量 (MUSCL 重构, GLM 9 变量, HLLD/HLL + fallback
 *    原子计数) -> rhs (含 psi 阻尼) -> 更新 -> 正定性检查
 *  时间环带 wall-clock 计时, 供性能测试直接使用。
 *
 *  Build (必须 -arch=sm_80; 不开 --use_fast_math):
 *    nvcc -O2 -std=c++14 -arch=sm_80 -I include src/mhd/mhd2d_glm_gpu.cu \
 *         -o bin/mhd2d_glm_gpu_fp64
 *    nvcc -O2 -std=c++14 -arch=sm_80 -I include -DUSE_FLOAT \
 *         src/mhd/mhd2d_glm_gpu.cu -o bin/mhd2d_glm_gpu_fp32
 *  Run (与 CPU 相同的参数约定):
 *    ./bin/mhd2d_glm_gpu_fp64 192 192 0.5 0.25 muscl hlld 0.18
 * ==========================================================================*/

#define MHD_GAMMA_VALUE (5.0 / 3.0)
#include "mhd/mhd_glm_common.hpp"

#include <chrono>
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
#define NV MHD_GLM_NVAR   /* 9 */

__host__ __device__ inline int idx2(int i, int j, int nx_total) {
  return j * nx_total + i;
}

// ── 与 CPU 版逐行对应的设备端辅助函数 ────────────────────────────────────────
__device__ void stateToPrimitiveDev(const Real* U9, Real P[NV]) {
  Real W[MHD_NVAR], psi;
  mhdGlmConsToPrim(U9, W, psi);
  for (int k = 0; k < MHD_NVAR; k++) P[k] = W[k];
  P[MHD_GLM_PSI] = psi;
}

__device__ void primitiveToStateDev(const Real P[NV], Real U9[NV]) {
  Real W[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) W[k] = P[k];
  mhdGlmPrimToCons(W, P[MHD_GLM_PSI], U9);
}

__device__ bool primitiveIsUsableDev(const Real P[NV]) {
  if (P[MHD_PR_RHO] <= MHD_RHO_FLOOR || P[MHD_PR_P] <= MHD_P_FLOOR) return false;
  for (int k = 0; k < NV; k++) {
    if (!isfinite((double)P[k])) return false;
  }
  return true;
}

__device__ bool stateIsUsableDev(const Real* U9) {
  Real P[NV];
  if (U9[MHD_RHO] <= MHD_RHO_FLOOR) return false;
  stateToPrimitiveDev(U9, P);
  return primitiveIsUsableDev(P);
}

__device__ void reconstructPrimitiveDev(const Real Pm[NV], const Real P0[NV],
                                        const Real Pp[NV], Real Pleft[NV],
                                        Real Pright[NV]) {
  for (int k = 0; k < NV; k++) {
    Real slope = mhdMinmod(P0[k] - Pm[k], Pp[k] - P0[k]);
    Pleft[k] = P0[k] - (Real)0.5 * slope;
    Pright[k] = P0[k] + (Real)0.5 * slope;
  }
  if (!primitiveIsUsableDev(Pleft) || !primitiveIsUsableDev(Pright)) {
    for (int k = 0; k < NV; k++) {
      Pleft[k] = P0[k];
      Pright[k] = P0[k];
    }
  }
}

// x 方向界面 (i,j)-(i+1,j) 的左右面值, 与 CPU xFaceStates 一致。
__device__ void xFaceStatesDev(const Real* U, int i, int j, int nx_total,
                               int use_muscl, Real UL[NV], Real UR[NV]) {
  if (!use_muscl) {
    const Real* a = &U[(size_t)idx2(i, j, nx_total) * NV];
    const Real* b = &U[(size_t)idx2(i + 1, j, nx_total) * NV];
    for (int k = 0; k < NV; k++) { UL[k] = a[k]; UR[k] = b[k]; }
    return;
  }
  Real Pim[NV], Pi[NV], Pip[NV], Pipp[NV];
  Real Pl_i[NV], Pr_i[NV], Pl_ip[NV], Pr_ip[NV];
  stateToPrimitiveDev(&U[(size_t)idx2(i - 1, j, nx_total) * NV], Pim);
  stateToPrimitiveDev(&U[(size_t)idx2(i, j, nx_total) * NV], Pi);
  stateToPrimitiveDev(&U[(size_t)idx2(i + 1, j, nx_total) * NV], Pip);
  stateToPrimitiveDev(&U[(size_t)idx2(i + 2, j, nx_total) * NV], Pipp);
  reconstructPrimitiveDev(Pim, Pi, Pip, Pl_i, Pr_i);
  reconstructPrimitiveDev(Pi, Pip, Pipp, Pl_ip, Pr_ip);
  primitiveToStateDev(Pr_i, UL);
  primitiveToStateDev(Pl_ip, UR);
}

// y 方向界面 (i,j)-(i,j+1), 与 CPU yFaceStates 一致。
__device__ void yFaceStatesDev(const Real* U, int i, int j, int nx_total,
                               int use_muscl, Real UL[NV], Real UR[NV]) {
  if (!use_muscl) {
    const Real* a = &U[(size_t)idx2(i, j, nx_total) * NV];
    const Real* b = &U[(size_t)idx2(i, j + 1, nx_total) * NV];
    for (int k = 0; k < NV; k++) { UL[k] = a[k]; UR[k] = b[k]; }
    return;
  }
  Real Pjm[NV], Pj[NV], Pjp[NV], Pjpp[NV];
  Real Pl_j[NV], Pr_j[NV], Pl_jp[NV], Pr_jp[NV];
  stateToPrimitiveDev(&U[(size_t)idx2(i, j - 1, nx_total) * NV], Pjm);
  stateToPrimitiveDev(&U[(size_t)idx2(i, j, nx_total) * NV], Pj);
  stateToPrimitiveDev(&U[(size_t)idx2(i, j + 1, nx_total) * NV], Pjp);
  stateToPrimitiveDev(&U[(size_t)idx2(i, j + 2, nx_total) * NV], Pjpp);
  reconstructPrimitiveDev(Pjm, Pj, Pjp, Pl_j, Pr_j);
  reconstructPrimitiveDev(Pj, Pjp, Pjpp, Pl_jp, Pr_jp);
  primitiveToStateDev(Pr_j, UL);
  primitiveToStateDev(Pl_jp, UR);
}

// ── kernels ─────────────────────────────────────────────────────────────────
// 周期边界, 与 CPU applyPeriodicBoundary 两段循环一致 (先 x 后 y, y 含角块)。
__global__ void bcXKernel(Real* U, int nx, int ny, int ghost) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nj = ny;                       // j in [ghost, ny+ghost)
  if (t >= nj * ghost) return;
  int g = t % ghost;
  int j = ghost + t / ghost;
  for (int k = 0; k < NV; k++) {
    U[(size_t)idx2(g, j, nx_total) * NV + k] =
        U[(size_t)idx2(nx + g, j, nx_total) * NV + k];
    U[(size_t)idx2(nx + ghost + g, j, nx_total) * NV + k] =
        U[(size_t)idx2(ghost + g, j, nx_total) * NV + k];
  }
}

__global__ void bcYKernel(Real* U, int nx, int ny, int ghost) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nx_total * ghost) return;
  int g = t % ghost;
  int i = t / ghost;
  for (int k = 0; k < NV; k++) {
    U[(size_t)idx2(i, g, nx_total) * NV + k] =
        U[(size_t)idx2(i, ny + g, nx_total) * NV + k];
    U[(size_t)idx2(i, ny + ghost + g, nx_total) * NV + k] =
        U[(size_t)idx2(i, ghost + g, nx_total) * NV + k];
  }
}

// 物理最大信号速度 max(|v|+cf) 的分块归约 (求 ch 用)。
__global__ void speedKernel(const Real* U, int nx, int ny, int ghost,
                            Real* part) {
  extern __shared__ unsigned char smem_raw[];
  Real* smem = reinterpret_cast<Real*>(smem_raw);
  const int nx_total = nx + 2 * ghost;
  int tid = threadIdx.x;
  int c = blockIdx.x * blockDim.x + tid;      // 0..nx*ny-1
  Real v = (Real)0.0;
  if (c < nx * ny) {
    int i = ghost + c % nx;
    int j = ghost + c / nx;
    Real P[NV];
    stateToPrimitiveDev(&U[(size_t)idx2(i, j, nx_total) * NV], P);
    Real sx = mhdAbs(P[MHD_PR_VX]) + mhdFastSpeedX(P);
    Real sy = mhdAbs(P[MHD_PR_VY]) + mhdFastSpeedY(P);
    v = mhdMax(sx, sy);
  }
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = mhdMax(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  if (tid == 0) part[blockIdx.x] = smem[0];
}

// 信号速率 max( max(sx,ch)/dx + max(sy,ch)/dy ) 的分块归约 (求 dt 用)。
__global__ void rateKernel(const Real* U, int nx, int ny, int ghost,
                           Real dx, Real dy, Real ch, Real* part) {
  extern __shared__ unsigned char smem_raw[];
  Real* smem = reinterpret_cast<Real*>(smem_raw);
  const int nx_total = nx + 2 * ghost;
  int tid = threadIdx.x;
  int c = blockIdx.x * blockDim.x + tid;
  Real v = (Real)0.0;
  if (c < nx * ny) {
    int i = ghost + c % nx;
    int j = ghost + c / nx;
    Real P[NV];
    stateToPrimitiveDev(&U[(size_t)idx2(i, j, nx_total) * NV], P);
    Real sx = mhdMax(mhdAbs(P[MHD_PR_VX]) + mhdFastSpeedX(P), ch);
    Real sy = mhdMax(mhdAbs(P[MHD_PR_VY]) + mhdFastSpeedY(P), ch);
    v = sx / dx + sy / dy;
  }
  smem[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] = mhdMax(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  if (tid == 0) part[blockIdx.x] = smem[0];
}

// x 界面通量: i in [ghost-1, nx+ghost), j in [ghost, ny+ghost)。
// counters = {x_evals, y_evals, degenerate, nonphysical}。
__global__ void fluxXKernel(const Real* U, Real* Fx, int nx, int ny, int ghost,
                            Real ch, int use_muscl, int use_hlld,
                            unsigned long long* counters) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = nx + 1;                    // interfaces per row
  if (t >= ni * ny) return;
  int i = (ghost - 1) + t % ni;
  int j = ghost + t / ni;
  Real UL[NV], UR[NV], F[NV];
  xFaceStatesDev(U, i, j, nx_total, use_muscl, UL, UR);
  int status = mhdGlmFluxX(UL, UR, ch, use_hlld != 0, F);
  if (use_hlld) {
    atomicAdd(&counters[0], 1ULL);
    if (status == MHD_HLLD_FALLBACK_DEGENERATE) atomicAdd(&counters[2], 1ULL);
    if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) atomicAdd(&counters[3], 1ULL);
  }
  for (int k = 0; k < NV; k++)
    Fx[(size_t)idx2(i, j, nx_total) * NV + k] = F[k];
}

// y 界面通量: i in [ghost, nx+ghost), j in [ghost-1, ny+ghost)。
__global__ void fluxYKernel(const Real* U, Real* Gy, int nx, int ny, int ghost,
                            Real ch, int use_muscl, int use_hlld,
                            unsigned long long* counters) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int nj = ny + 1;
  if (t >= nx * nj) return;
  int i = ghost + t % nx;
  int j = (ghost - 1) + t / nx;
  Real UL[NV], UR[NV], G[NV];
  yFaceStatesDev(U, i, j, nx_total, use_muscl, UL, UR);
  int status = mhdGlmFluxY(UL, UR, ch, use_hlld != 0, G);
  if (use_hlld) {
    atomicAdd(&counters[1], 1ULL);
    if (status == MHD_HLLD_FALLBACK_DEGENERATE) atomicAdd(&counters[2], 1ULL);
    if (status == MHD_HLLD_FALLBACK_NONPHYSICAL) atomicAdd(&counters[3], 1ULL);
  }
  for (int k = 0; k < NV; k++)
    Gy[(size_t)idx2(i, j, nx_total) * NV + k] = G[k];
}

// rhs = -dFx/dx - dGy/dy, psi 加阻尼项 (与 CPU computeRhs 第三段一致)。
__global__ void rhsKernel(const Real* U, const Real* Fx, const Real* Gy,
                          Real* rhs, int nx, int ny, int ghost,
                          Real dx, Real dy, Real damping_rate) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nx * ny) return;
  int i = ghost + t % nx;
  int j = ghost + t / nx;
  size_t id = (size_t)idx2(i, j, nx_total) * NV;
  size_t im = (size_t)idx2(i - 1, j, nx_total) * NV;
  size_t jm = (size_t)idx2(i, j - 1, nx_total) * NV;
  for (int k = 0; k < NV; k++) {
    rhs[id + k] = -(Fx[id + k] - Fx[im + k]) / dx
                  - (Gy[id + k] - Gy[jm + k]) / dy;
  }
  rhs[id + MHD_GLM_PSI] -= damping_rate * U[id + MHD_GLM_PSI];
}

// SSPRK2 第一级: U1 = U + dt*rhs; 正定性检查记录最小坏格点平铺下标。
__global__ void stage1Kernel(const Real* U, const Real* rhs, Real* U1,
                             Real dt, int nx, int ny, int ghost, int* bad) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nx * ny) return;
  int i = ghost + t % nx;
  int j = ghost + t / nx;
  size_t id = (size_t)idx2(i, j, nx_total) * NV;
  Real Ui[NV];
  for (int k = 0; k < NV; k++) {
    Ui[k] = U[id + k] + dt * rhs[id + k];
    U1[id + k] = Ui[k];
  }
  if (!stateIsUsableDev(Ui)) atomicMin(bad, t);
}

// SSPRK2 第二级: U = 0.5*U + 0.5*(U1 + dt*rhs2)。
__global__ void stage2Kernel(Real* U, const Real* U1, const Real* rhs2,
                             Real dt, int nx, int ny, int ghost, int* bad) {
  const int nx_total = nx + 2 * ghost;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nx * ny) return;
  int i = ghost + t % nx;
  int j = ghost + t / nx;
  size_t id = (size_t)idx2(i, j, nx_total) * NV;
  Real Ui[NV];
  for (int k = 0; k < NV; k++) {
    Ui[k] = (Real)0.5 * U[id + k]
          + (Real)0.5 * (U1[id + k] + dt * rhs2[id + k]);
    U[id + k] = Ui[k];
  }
  if (!stateIsUsableDev(Ui)) atomicMin(bad, t);
}

// ── host 侧小工具 ───────────────────────────────────────────────────────────
static void applyBC(Real* dU, int nx, int ny, int ghost) {
  const int nx_total = nx + 2 * ghost;
  int nX = ny * ghost, nY = nx_total * ghost;
  bcXKernel<<<(nX + BLK - 1) / BLK, BLK>>>(dU, nx, ny, ghost);
  bcYKernel<<<(nY + BLK - 1) / BLK, BLK>>>(dU, nx, ny, ghost);
}

static Real reducePart(const std::vector<Real>& part) {
  Real m = (Real)0.0;
  for (Real v : part) m = mhdMax(m, v);
  return m;
}

// 一次 rhs 评估 (等价于 CPU computeRhs): BC -> fluxX -> fluxY -> rhs。
static void computeRhsGpu(Real* dU, Real* dFx, Real* dGy, Real* dRhs,
                          int nx, int ny, int ghost, Real dx, Real dy,
                          int use_muscl, int use_hlld, Real ch, Real alpha,
                          unsigned long long* dCnt) {
  applyBC(dU, nx, ny, ghost);
  Real damping_rate = ch / mhdMax(alpha, (Real)1.0e-12);
  int nfx = (nx + 1) * ny, nfy = nx * (ny + 1), nc = nx * ny;
  fluxXKernel<<<(nfx + BLK - 1) / BLK, BLK>>>(dU, dFx, nx, ny, ghost, ch,
                                              use_muscl, use_hlld, dCnt);
  fluxYKernel<<<(nfy + BLK - 1) / BLK, BLK>>>(dU, dGy, nx, ny, ghost, ch,
                                              use_muscl, use_hlld, dCnt);
  rhsKernel<<<(nc + BLK - 1) / BLK, BLK>>>(dU, dFx, dGy, dRhs, nx, ny, ghost,
                                           dx, dy, damping_rate);
}

int main(int argc, char* argv[]) {
  int nx = 128, ny = 128;
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
  std::string problem = "ot";
  if (argc > 8) problem = argv[8];

  if (problem != "ot" && problem != "kh") {
    std::cerr << "Error: problem must be 'ot' or 'kh'." << std::endl;
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
  const int use_muscl = (order == "muscl") ? 1 : 0;
  const int use_hlld = (solver == "hlld") ? 1 : 0;

  const int ghost = 2;
  const int nx_total = nx + 2 * ghost;
  const int ny_total = ny + 2 * ghost;
  const int total = nx_total * ny_total;
  const Real x_min = (Real)0.0, x_max = (Real)1.0;
  const Real y_min = (Real)0.0, y_max = (Real)1.0;
  const Real dx = (x_max - x_min) / (Real)nx;
  const Real dy = (y_max - y_min) / (Real)ny;
  const Real pi = std::acos((Real)-1.0);
  const Real two_pi = (Real)2.0 * pi;
  const Real inv_sqrt_4pi = (Real)1.0 / std::sqrt((Real)4.0 * pi);

  std::cout << "===== 2D ideal MHD Orszag-Tang vortex with mixed GLM (GPU) ====="
            << std::endl;
  std::cout << "Solver: " << (use_hlld ? "HLLD" : "HLL") << std::endl;
  std::cout << "Reconstruction/time: "
            << (use_muscl ? "MUSCL + SSPRK2" : "first-order + SSPRK2")
            << std::endl;
  std::cout << "Precision: " << (int)(sizeof(Real) * 8) << " bit"
            << (sizeof(Real) == 8 ? " (fp64)" : " (fp32)") << std::endl;
  std::cout << "gamma = " << MHD_GAMMA << ", alpha = " << alpha << std::endl;
  std::cout << "Nx x Ny = " << nx << " x " << ny << ", CFL = " << cfl
            << ", t_end = " << t_end << std::endl;

  // OT 初始条件 (与 CPU 一致), host 端构造后拷上设备。
  std::vector<Real> hU((size_t)total * NV);
  for (int j = 0; j < ny_total; j++) {
    for (int i = 0; i < nx_total; i++) {
      Real x = x_min + ((Real)(i - ghost) + (Real)0.5) * dx;
      Real y = y_min + ((Real)(j - ghost) + (Real)0.5) * dy;
      Real P[NV], W[MHD_NVAR], U9[NV];
      if (problem == "kh") {
        // Kelvin-Helmholtz: 双剪切层 (双周期), tanh 过渡 + 确定性单模扰动。
        // S(y): 带内 (0.25<y<0.75) = 1, 带外 = 0, 过渡宽度 a。
        const Real a = (Real)0.025;
        const Real sigma = (Real)0.05;
        const Real delta = (Real)0.01;
        Real S = (Real)0.5 * (std::tanh((y - (Real)0.25) / a) -
                              std::tanh((y - (Real)0.75) / a));
        Real g1 = (y - (Real)0.25) / sigma;
        Real g2 = (y - (Real)0.75) / sigma;
        P[MHD_PR_RHO] = (Real)1.0 + S;                    // 1 外 / 2 内
        P[MHD_PR_VX] = -(Real)0.5 + S;                    // ∓0.5 剪切
        P[MHD_PR_VY] = delta * std::sin((Real)2.0 * two_pi * x) *
                       (std::exp(-(Real)0.5 * g1 * g1) +
                        std::exp(-(Real)0.5 * g2 * g2));
        P[MHD_PR_VZ] = (Real)0.0;
        P[MHD_PR_P] = (Real)2.5;
        P[MHD_PR_BX] = (Real)0.2;                         // 弱平行场, HLLD 非退化
        P[MHD_PR_BY] = (Real)0.0;
        P[MHD_PR_BZ] = (Real)0.0;
        P[MHD_GLM_PSI] = (Real)0.0;
      } else {
      P[MHD_PR_RHO] = (Real)25.0 / ((Real)36.0 * pi);
      P[MHD_PR_VX] = -std::sin(two_pi * y);
      P[MHD_PR_VY] = std::sin(two_pi * x);
      P[MHD_PR_VZ] = (Real)0.0;
      P[MHD_PR_P] = (Real)5.0 / ((Real)12.0 * pi);
      P[MHD_PR_BX] = -std::sin(two_pi * y) * inv_sqrt_4pi;
      P[MHD_PR_BY] = std::sin((Real)2.0 * two_pi * x) * inv_sqrt_4pi;
      P[MHD_PR_BZ] = (Real)0.0;
      P[MHD_GLM_PSI] = (Real)0.0;
      }
      for (int k = 0; k < MHD_NVAR; k++) W[k] = P[k];
      mhdGlmPrimToCons(W, P[MHD_GLM_PSI], U9);
      for (int k = 0; k < NV; k++) hU[(size_t)idx2(i, j, nx_total) * NV + k] = U9[k];
    }
  }

  const size_t nbytes = (size_t)total * NV * sizeof(Real);
  Real *dU, *dU1, *dRhs, *dFx, *dGy, *dPart;
  int* dBad;
  unsigned long long* dCnt;
  const int nblk_red = (nx * ny + BLK - 1) / BLK;
  CUDA_CHECK(cudaMalloc(&dU, nbytes));
  CUDA_CHECK(cudaMalloc(&dU1, nbytes));
  CUDA_CHECK(cudaMalloc(&dRhs, nbytes));
  CUDA_CHECK(cudaMalloc(&dFx, nbytes));
  CUDA_CHECK(cudaMalloc(&dGy, nbytes));
  CUDA_CHECK(cudaMalloc(&dPart, (size_t)nblk_red * sizeof(Real)));
  CUDA_CHECK(cudaMalloc(&dBad, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dCnt, 4 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(dCnt, 0, 4 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemcpy(dU, hU.data(), nbytes, cudaMemcpyHostToDevice));
  applyBC(dU, nx, ny, ghost);

  std::vector<Real> hPart(nblk_red);
  Real t = (Real)0.0, last_ch = (Real)0.0;
  int step = 0;
  const int INT_BIG = 0x7fffffff;
  const size_t shmem = BLK * sizeof(Real);

  auto t0 = std::chrono::steady_clock::now();
  while (t < t_end) {
    // ch: 物理最大信号速度 (与 CPU 相同的两段归约)。
    speedKernel<<<nblk_red, BLK, shmem>>>(dU, nx, ny, ghost, dPart);
    CUDA_CHECK(cudaMemcpy(hPart.data(), dPart, nblk_red * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    Real ch = mhdMax(reducePart(hPart), (Real)1.0e-12);
    last_ch = ch;

    rateKernel<<<nblk_red, BLK, shmem>>>(dU, nx, ny, ghost, dx, dy, ch, dPart);
    CUDA_CHECK(cudaMemcpy(hPart.data(), dPart, nblk_red * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    Real max_rate = reducePart(hPart);
    if (max_rate <= (Real)0.0) {
      std::cerr << "Error: non-positive maximum signal rate." << std::endl;
      return 2;
    }
    Real dt = cfl / max_rate;
    if (t + dt > t_end) dt = t_end - t;

    int nc = nx * ny;

    // 第一级
    computeRhsGpu(dU, dFx, dGy, dRhs, nx, ny, ghost, dx, dy, use_muscl,
                  use_hlld, ch, alpha, dCnt);
    CUDA_CHECK(cudaMemcpy(dBad, &INT_BIG, sizeof(int), cudaMemcpyHostToDevice));
    stage1Kernel<<<(nc + BLK - 1) / BLK, BLK>>>(dU, dRhs, dU1, dt, nx, ny,
                                                ghost, dBad);
    int bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, dBad, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad != INT_BIG) {
      std::cerr << "Error: non-physical RK stage at step " << step
                << ", cell (" << bad % nx << ", " << bad / nx << ")"
                << std::endl;
      return 3;
    }
    applyBC(dU1, nx, ny, ghost);

    // 第二级
    computeRhsGpu(dU1, dFx, dGy, dRhs, nx, ny, ghost, dx, dy, use_muscl,
                  use_hlld, ch, alpha, dCnt);
    CUDA_CHECK(cudaMemcpy(dBad, &INT_BIG, sizeof(int), cudaMemcpyHostToDevice));
    stage2Kernel<<<(nc + BLK - 1) / BLK, BLK>>>(dU, dU1, dRhs, dt, nx, ny,
                                                ghost, dBad);
    CUDA_CHECK(cudaMemcpy(&bad, dBad, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad != INT_BIG) {
      std::cerr << "Error: non-physical state at step " << step
                << ", cell (" << bad % nx << ", " << bad / nx << ")"
                << std::endl;
      return 4;
    }
    applyBC(dU, nx, ny, ghost);

    t += dt;
    step++;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double wall = std::chrono::duration<double>(t1 - t0).count();

  unsigned long long hCnt[4] = {0, 0, 0, 0};
  CUDA_CHECK(cudaMemcpy(hCnt, dCnt, 4 * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hU.data(), dU, nbytes, cudaMemcpyDeviceToHost));

  std::cout << "Done. Steps: " << step << ", final t = " << t << std::endl;
  std::cout << "c_h(last) = " << last_ch << ", alpha = " << alpha << std::endl;
  std::cout << "Time loop wall time = " << std::fixed << std::setprecision(3)
            << wall << " s ("
            << std::setprecision(2)
            << ((double)nx * ny * step / wall / 1.0e6)
            << " Mcell-steps/s)" << std::defaultfloat << std::endl;

  // divB / psi 诊断 (host 端, 与 CPU 相同的中心差分)。
  Real div_l1 = (Real)0.0, div_linf = (Real)0.0, psi_linf = (Real)0.0;
  auto Uat = [&](int i, int j, int k) -> Real {
    return hU[(size_t)idx2(i, j, nx_total) * NV + k];
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

  unsigned long long evals = hCnt[0] + hCnt[1];
  if (use_hlld) {
    unsigned long long fb = hCnt[2] + hCnt[3];
    double frac = evals > 0 ? (double)fb / (double)evals : 0.0;
    std::cout << "HLLD flux evaluations = " << evals
              << " (x=" << hCnt[0] << ", y=" << hCnt[1] << ")" << std::endl;
    std::cout << "HLLD relative fallback tolerance = "
              << std::scientific << (double)mhdHlldTolerance()
              << std::defaultfloat << std::endl;
    std::cout << "HLLD->HLL fallbacks = " << fb
              << " (degenerate=" << hCnt[2] << ", nonphysical=" << hCnt[3]
              << "), fraction = " << std::scientific << frac
              << std::defaultfloat << std::endl;
  }

  const char* out_env = std::getenv("OUTDIR");
  const std::string prob_base =
      (problem == "kh") ? std::string("kh") : std::string("orszag_tang");
  std::string out_dir = out_env ? std::string(out_env)
                                : std::string("results/mhd/gpu/") + prob_base;
  std::string mk = "mkdir -p " + out_dir;
  if (std::system(mk.c_str()) != 0) { /* ignore */ }

  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
  std::ostringstream fname;
  fname << out_dir << "/" << prob_base << "_glm_" << order << "_" << solver
        << "_" << prec_name << "_N" << nx << "x" << ny << ".dat";

  std::ofstream out(fname.str());
  if (!out) {
    std::cerr << "Error: failed to open output file: " << fname.str()
              << std::endl;
    return 5;
  }
  out << ((problem == "kh") ? "# Kelvin-Helmholtz ideal MHD with mixed GLM (GPU)\n"
                            : "# Orszag-Tang ideal MHD with mixed GLM (GPU)\n");
  out << "# nx " << nx << " ny " << ny << " t " << t << " gamma " << MHD_GAMMA
      << " cfl " << cfl << " order " << order << " solver " << solver
      << " alpha " << alpha << " ch_last " << last_ch << "\n";
  if (use_hlld) {
    unsigned long long fb = hCnt[2] + hCnt[3];
    double frac = evals > 0 ? (double)fb / (double)evals : 0.0;
    out << "# hlld_evaluations " << evals
        << " hlld_tolerance " << mhdHlldTolerance()
        << " hlld_x_evaluations " << hCnt[0]
        << " hlld_y_evaluations " << hCnt[1]
        << " hlld_fallbacks " << fb
        << " hlld_degenerate " << hCnt[2]
        << " hlld_nonphysical " << hCnt[3]
        << " hlld_fallback_fraction " << frac << "\n";
  }
  out << "# columns: x y rho vx vy vz p Bx By Bz E divB psi\n";
  out << std::scientific
      << std::setprecision(std::numeric_limits<Real>::max_digits10);
  for (int j = ghost; j < ny + ghost; j++) {
    for (int i = ghost; i < nx + ghost; i++) {
      Real U9[NV], W[MHD_NVAR], psi;
      for (int k = 0; k < NV; k++)
        U9[k] = hU[(size_t)idx2(i, j, nx_total) * NV + k];
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

  cudaFree(dU); cudaFree(dU1); cudaFree(dRhs); cudaFree(dFx); cudaFree(dGy);
  cudaFree(dPart); cudaFree(dBad); cudaFree(dCnt);
  return 0;
}