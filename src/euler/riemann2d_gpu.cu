/**
 * ============================================================================
 *   2D Riemann 问题 (Liska & Wendroff 2003) — CUDA 版本
 * ============================================================================
 *
 * 算法与 src/riemann2d_cpu.cpp 完全一致 (MUSCL + Hancock + HLLC + 方向分裂),
 * 共享数值函数来自 include/riemann.hpp (传递包含 euler_common.hpp / config.hpp)。
 *
 * 注意:
 *   config.hpp 中 NX/NY/DX/DY/T_END 是 shock-bubble 专用的宏, 这里一律不用,
 *   网格参数通过命令行 + 局部变量 (nx2/ny2/dx2/dy2) 传递给 kernel,
 *   避免与宏冲突 (跟 riemann1d_gpu.cu 同套做法).
 *
 * 用法:
 *   ./bin/riemann2d_gpu              -> Config 6, 400x400
 *   ./bin/riemann2d_gpu 400 3        -> Config 3, 400x400
 *   ./bin/riemann2d_gpu 800 12       -> Config 12, 800x800
 *
 * 输出: results/2d/{prec}/gpu/riemann2d_{cfgname}_hllc_{N}.dat
 *       (CPU 版本输出到 results/2d/{prec}/, 用 gpu/ 子目录避免覆盖)
 *
 * ============================================================================
 */

#include "euler/riemann.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define BLOCK_X 16
#define BLOCK_Y 16

// ============================================================================
//  Grid (SoA-on-flat-array; layout = [v][j][i])
// ============================================================================
enum Processor { HOST_CPU, DEVICE_GPU };

struct Grid {
  Real* data;
  int xCells, yCells, total;

  __host__ __device__ Real& operator()(int i, int j, int v) {
    return data[i + j * xCells + v * total];
  }
  __host__ __device__ const Real& operator()(int i, int j, int v) const {
    return data[i + j * xCells + v * total];
  }
};

Grid createGrid(int nx, int ny, Processor proc) {
  Grid g;
  g.xCells = nx; g.yCells = ny; g.total = nx * ny;
  if (proc == HOST_CPU) {
    g.data = new Real[4 * g.total]();
  } else {
    CUDA_CHECK(cudaMalloc(&g.data, 4 * g.total * sizeof(Real)));
    CUDA_CHECK(cudaMemset(g.data, 0, 4 * g.total * sizeof(Real)));
  }
  return g;
}
void copyGridToDevice(Grid& d, const Grid& h) {
  CUDA_CHECK(cudaMemcpy(d.data, h.data, 4*h.total*sizeof(Real), cudaMemcpyHostToDevice));
}
void copyGridToHost(Grid& h, const Grid& d) {
  CUDA_CHECK(cudaMemcpy(h.data, d.data, 4*d.total*sizeof(Real), cudaMemcpyDeviceToHost));
}
void freeGridHost(Grid& g)   { delete[] g.data; g.data = nullptr; }
void freeGridDevice(Grid& g) { CUDA_CHECK(cudaFree(g.data)); g.data = nullptr; }

// ============================================================================
//  四象限初始条件 (Liska & Wendroff 2003, Table 4.3)
//  与 riemann2d_cpu.cpp 完全一致
// ============================================================================
struct QuadState { Real rho, u, v, p; };

struct Config2D {
  const char* name;
  QuadState Q1, Q2, Q3, Q4;   // 右上, 左上, 左下, 右下
  Real Tend;                   // 避开 T_END 宏
};

static Config2D configs[] = {
  {"cfg3_4shocks",
   {(Real)0.5197, (Real)-0.6259, (Real)-0.3109, (Real)0.4},
   {(Real)1.0,    (Real)0.1,     (Real)-0.3109, (Real)1.0},
   {(Real)0.8,    (Real)0.1,     (Real)0.4276,  (Real)0.4},
   {(Real)0.5197, (Real)-0.6259, (Real)0.4276,  (Real)0.4},
   (Real)0.3},

  {"cfg6_4contacts",
   {(Real)1.0, (Real)0.75,  (Real)-0.5, (Real)1.0},
   {(Real)2.0, (Real)0.75,  (Real)0.5,  (Real)1.0},
   {(Real)1.0, (Real)-0.75, (Real)0.5,  (Real)1.0},
   {(Real)3.0, (Real)-0.75, (Real)-0.5, (Real)1.0},
   (Real)0.3},

  {"cfg12_2S2C",
   {(Real)0.5313, (Real)0.0,    (Real)0.0,    (Real)0.4},
   {(Real)1.0,    (Real)0.7276, (Real)0.0,    (Real)1.0},
   {(Real)0.8,    (Real)0.0,    (Real)0.0,    (Real)1.0},
   {(Real)1.0,    (Real)0.0,    (Real)0.7276, (Real)1.0},
   (Real)0.25},
};

static int configIndex(int cfg_id) {
  switch (cfg_id) {
    case 3:  return 0;
    case 6:  return 1;
    case 12: return 2;
    default: return -1;
  }
}

// ============================================================================
//  Kernels
// ============================================================================

// --- 透射边界 ---
__global__ void applyBCKernel(Grid u, int nx, int ny, int gh) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tot_x = nx + 2 * gh;
  int tot_y = ny + 2 * gh;
  if (tid < tot_y) {
    int j = tid;
    for (int g = 0; g < gh; g++)
      for (int v = 0; v < 4; v++) {
        u(g, j, v)             = u(gh, j, v);
        u(tot_x - 1 - g, j, v) = u(tot_x - 1 - gh, j, v);
      }
  }
  if (tid < tot_x) {
    int i = tid;
    for (int g = 0; g < gh; g++)
      for (int v = 0; v < 4; v++) {
        u(i, g, v)             = u(i, gh, v);
        u(i, tot_y - 1 - g, v) = u(i, tot_y - 1 - gh, v);
      }
  }
}

// --- CFL 归约 (输出每块的局部最大 (|u|+c)/dx, (|v|+c)/dy 中较大者) ---
__global__ void cflReductionKernel(Grid u, Real* d_block_max, int num_blocks_x,
                                    int nx, int ny, int gh, Real dx, Real dy) {
  extern __shared__ Real sdata[];
  int i   = blockIdx.x * blockDim.x + threadIdx.x + gh;
  int j   = blockIdx.y * blockDim.y + threadIdx.y + gh;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int block_size = blockDim.x * blockDim.y;

  Real local_max = (Real)0.0;
  if (i < nx + gh && j < ny + gh) {
    Real U[4];
    for (int v = 0; v < 4; v++) U[v] = u(i, j, v);
    Real W[4];
    consToPrim(U, W);
    Real c  = soundSpeed(W[3], W[0]);
    Real sx = (fabs(W[1]) + c) / dx;
    Real sy = (fabs(W[2]) + c) / dy;
    local_max = fmax(sx, sy);
  }
  sdata[tid] = local_max;
  __syncthreads();
  for (int s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0)
    d_block_max[blockIdx.y * num_blocks_x + blockIdx.x] = sdata[0];
}

// --- 通量 kernel (一个线程一条界面: MUSCL + Hancock + HLLC) ---
template <int coord>
__global__ void computeFluxKernel(Grid u_old, Grid fluxes, Real dt,
                                   int nx, int ny, int gh, Real dx, Real dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + gh;
  int j = blockIdx.y * blockDim.y + threadIdx.y + gh;
  int tot_x = nx + 2 * gh;
  int tot_y = ny + 2 * gh;

  int limit_i = (coord == 0) ? nx + gh + 1 : nx + gh;
  int limit_j = (coord == 1) ? ny + gh + 1 : ny + gh;
  if (i >= limit_i || j >= limit_j) return;

  int n_idx = (coord == 0) ? 1 : 2;
  int t_idx = (coord == 0) ? 2 : 1;
  Real ds   = (coord == 0) ? dx : dy;

  int di = (coord == 0) ? 1 : 0;
  int dj = (coord == 1) ? 1 : 0;

  int i_LL = i - 2 * di, j_LL = j - 2 * dj;
  int i_L  = i -     di, j_L  = j -     dj;
  int i_0  = i,          j_0  = j;
  int i_R  = i +     di, j_R  = j +     dj;

  if (i_LL < 0 || j_LL < 0 || i_R >= tot_x || j_R >= tot_y) return;

  Real U_LL[4], U_L[4], U_0[4], U_R[4];
  for (int k = 0; k < 4; k++) {
    U_LL[k] = u_old(i_LL, j_LL, k);
    U_L[k]  = u_old(i_L,  j_L,  k);
    U_0[k]  = u_old(i_0,  j_0,  k);
    U_R[k]  = u_old(i_R,  j_R,  k);
  }
  Real W_LL[4], W_L[4], W_0[4], W_R[4];
  consToPrim(U_LL, W_LL);
  consToPrim(U_L,  W_L);
  consToPrim(U_0,  W_0);
  consToPrim(U_R,  W_R);

  // MUSCL 重构
  Real WL_Lface[4], WL_Rface[4], W0_Lface[4], W0_Rface[4];
  for (int k = 0; k < 4; k++) {
    muscl_recon(W_LL[k], W_L[k], W_0[k], WL_Lface[k], WL_Rface[k]);
    muscl_recon(W_L[k],  W_0[k], W_R[k], W0_Lface[k], W0_Rface[k]);
  }

  // Hancock 半步
  Real factor = (Real)0.5 * dt / ds;
  Real WL_half[4], W0_half[4];
  hancock_evolve(WL_Rface, WL_Lface, WL_Rface, WL_half, factor, n_idx, t_idx);
  hancock_evolve(W0_Lface, W0_Lface, W0_Rface, W0_half, factor, n_idx, t_idx);

  // HLLC 通量
  Real f[4];
  hllc_flux(WL_half, W0_half, f, n_idx, t_idx);

  for (int k = 0; k < 4; k++) fluxes(i, j, k) = f[k];
}

// --- 守恒量更新 ---
template <int coord>
__global__ void updateConservativeKernel(Grid u_old, Grid fluxes, Grid u_new,
                                          Real dt, int nx, int ny, int gh,
                                          Real dx, Real dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + gh;
  int j = blockIdx.y * blockDim.y + threadIdx.y + gh;
  if (i >= nx + gh || j >= ny + gh) return;

  Real ds    = (coord == 0) ? dx : dy;
  Real ratio = dt / ds;
  int  i_R   = (coord == 0) ? i + 1 : i;
  int  j_R   = (coord == 1) ? j + 1 : j;

  for (int k = 0; k < 4; k++)
    u_new(i, j, k) = u_old(i, j, k) - ratio * (fluxes(i_R, j_R, k) - fluxes(i, j, k));
}

// ============================================================================
//  Host: 四象限初始条件 (与 CPU 完全一致)
// ============================================================================
void initialize4Quad(Grid& h_grid, const Config2D& cfg,
                     int nx, int ny, int gh, Real dx, Real dy,
                     Real x_min, Real y_min) {
  int tot_x = nx + 2 * gh;
  int tot_y = ny + 2 * gh;
  for (int j = 0; j < tot_y; j++) {
    Real y = y_min + (Real)(j - gh + 0.5) * dy;
    for (int i = 0; i < tot_x; i++) {
      Real x = x_min + (Real)(i - gh + 0.5) * dx;
      const QuadState* qs;
      if      (x >= (Real)0.5 && y >= (Real)0.5) qs = &cfg.Q1;  // 右上
      else if (x <  (Real)0.5 && y >= (Real)0.5) qs = &cfg.Q2;  // 左上
      else if (x <  (Real)0.5 && y <  (Real)0.5) qs = &cfg.Q3;  // 左下
      else                                        qs = &cfg.Q4;  // 右下
      Real W[4] = {qs->rho, qs->u, qs->v, qs->p};
      Real U[4];
      primToCons(W, U);
      for (int k = 0; k < 4; k++) h_grid(i, j, k) = U[k];
    }
  }
}

// ============================================================================
//  Host: 输出 (格式与 riemann2d_cpu.cpp 完全一致 — 外层 j, 内层 i, 无空行)
// ============================================================================
void writeOutput2D(const std::string& fname, Grid& h_grid,
                    int nx, int ny, int gh, Real dx, Real dy,
                    Real x_min, Real y_min) {
  std::ofstream out(fname);
  out << std::scientific;
  out.precision(std::numeric_limits<Real>::max_digits10);
  for (int j = gh; j < ny + gh; j++) {
    for (int i = gh; i < nx + gh; i++) {
      Real x = x_min + (Real)(i - gh + 0.5) * dx;
      Real y = y_min + (Real)(j - gh + 0.5) * dy;
      Real U[4];
      for (int v = 0; v < 4; v++) U[v] = h_grid(i, j, v);
      Real W[4];
      consToPrim(U, W);
      out << x << " " << y << " " << W[0] << " " << W[1] << " "
          << W[2] << " " << W[3] << "\n";
    }
  }
  out.close();
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char* argv[]) {
  int N = 400;
  int cfg_id = 6;
  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) cfg_id = std::atoi(argv[2]);

  int ci = configIndex(cfg_id);
  if (ci < 0) {
    std::cerr << "Error: config_id must be 3, 6, or 12\n";
    return 1;
  }
  const Config2D& cfg = configs[ci];

  // 局部网格参数 (避开 config.hpp 的宏)
  const int  nx2   = N, ny2 = N;
  const int  gh    = 2;
  const int  tot_x = nx2 + 2 * gh;
  const int  tot_y = ny2 + 2 * gh;
  const Real xmin = (Real)0.0, xmax = (Real)1.0;
  const Real ymin = (Real)0.0, ymax = (Real)1.0;
  const Real dx2  = (xmax - xmin) / (Real)nx2;
  const Real dy2  = (ymax - ymin) / (Real)ny2;
  const Real cfl  = (Real)0.8;
  const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";

  std::cout << "===== 2D Riemann GPU =====\n"
            << "Config:    " << cfg.name << "  (id=" << cfg_id << ")\n"
            << "Solver:    HLLC\n"
            << "Precision: " << prec_name << "\n"
            << "Grid:      " << nx2 << " x " << ny2
            << "  dx=" << dx2 << "\n"
            << "T_end:     " << cfg.Tend << "\n";

  // Host 分配 + 初始化
  Grid h_grid  = createGrid(tot_x, tot_y, HOST_CPU);
  Grid d_u_old = createGrid(tot_x, tot_y, DEVICE_GPU);
  Grid d_u_new = createGrid(tot_x, tot_y, DEVICE_GPU);
  Grid d_flux  = createGrid(tot_x, tot_y, DEVICE_GPU);

  initialize4Quad(h_grid, cfg, nx2, ny2, gh, dx2, dy2, xmin, ymin);
  copyGridToDevice(d_u_old, h_grid);

  // 启动配置
  int grid_rx = (nx2 + BLOCK_X - 1) / BLOCK_X;
  int grid_ry = (ny2 + BLOCK_Y - 1) / BLOCK_Y;
  int num_blocks = grid_rx * grid_ry;
  Real* d_block_max;
  CUDA_CHECK(cudaMalloc(&d_block_max, num_blocks * sizeof(Real)));
  std::vector<Real> h_block_max(num_blocks);

  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid_inner(grid_rx, grid_ry);
  size_t shared_bytes = BLOCK_X * BLOCK_Y * sizeof(Real);
  dim3 grid_flux_x(((nx2 + 1) + BLOCK_X - 1) / BLOCK_X,
                    (ny2       + BLOCK_Y - 1) / BLOCK_Y);
  dim3 grid_flux_y((nx2       + BLOCK_X - 1) / BLOCK_X,
                    ((ny2 + 1) + BLOCK_Y - 1) / BLOCK_Y);

  int bc_threads = std::max(tot_x, tot_y);
  int bc_blocks  = (bc_threads + 255) / 256;

  applyBCKernel<<<bc_blocks, 256>>>(d_u_old, nx2, ny2, gh);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 时间推进 (交替方向分裂, 与 CPU 完全一致)
  auto t_start = std::chrono::high_resolution_clock::now();
  Real t = (Real)0.0;
  int  step = 0;

  while (t < cfg.Tend) {
    // CFL
    cflReductionKernel<<<grid_inner, block, shared_bytes>>>(
        d_u_old, d_block_max, grid_rx, nx2, ny2, gh, dx2, dy2);
    CUDA_CHECK(cudaMemcpy(h_block_max.data(), d_block_max,
                          num_blocks * sizeof(Real), cudaMemcpyDeviceToHost));
    Real vmax = (Real)0.0;
    for (int b = 0; b < num_blocks; b++)
      vmax = std::max(vmax, h_block_max[b]);
    Real dt = cfl / vmax;
    if (t + dt > cfg.Tend) dt = cfg.Tend - t;
    if (dt <= (Real)0.0 || t + dt == t) break;

    if (step % 2 == 0) {
      // X -> Y
      computeFluxKernel<0><<<grid_flux_x, block>>>(d_u_old, d_flux, dt,
                                                    nx2, ny2, gh, dx2, dy2);
      CUDA_CHECK(cudaDeviceSynchronize());
      updateConservativeKernel<0><<<grid_inner, block>>>(d_u_old, d_flux, d_u_new,
                                                          dt, nx2, ny2, gh, dx2, dy2);
      applyBCKernel<<<bc_blocks, 256>>>(d_u_new, nx2, ny2, gh);
      CUDA_CHECK(cudaDeviceSynchronize());

      computeFluxKernel<1><<<grid_flux_y, block>>>(d_u_new, d_flux, dt,
                                                    nx2, ny2, gh, dx2, dy2);
      CUDA_CHECK(cudaDeviceSynchronize());
      updateConservativeKernel<1><<<grid_inner, block>>>(d_u_new, d_flux, d_u_old,
                                                          dt, nx2, ny2, gh, dx2, dy2);
      applyBCKernel<<<bc_blocks, 256>>>(d_u_old, nx2, ny2, gh);
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      // Y -> X
      computeFluxKernel<1><<<grid_flux_y, block>>>(d_u_old, d_flux, dt,
                                                    nx2, ny2, gh, dx2, dy2);
      CUDA_CHECK(cudaDeviceSynchronize());
      updateConservativeKernel<1><<<grid_inner, block>>>(d_u_old, d_flux, d_u_new,
                                                          dt, nx2, ny2, gh, dx2, dy2);
      applyBCKernel<<<bc_blocks, 256>>>(d_u_new, nx2, ny2, gh);
      CUDA_CHECK(cudaDeviceSynchronize());

      computeFluxKernel<0><<<grid_flux_x, block>>>(d_u_new, d_flux, dt,
                                                    nx2, ny2, gh, dx2, dy2);
      CUDA_CHECK(cudaDeviceSynchronize());
      updateConservativeKernel<0><<<grid_inner, block>>>(d_u_new, d_flux, d_u_old,
                                                          dt, nx2, ny2, gh, dx2, dy2);
      applyBCKernel<<<bc_blocks, 256>>>(d_u_old, nx2, ny2, gh);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    t += dt;
    step++;
    if (step % 200 == 0)
      std::cout << "Step " << step << ": t = " << t << "\n";
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = t_end - t_start;
  std::cout << "Done. Steps: " << step
            << "   Wall: " << elapsed.count() / 1000.0 << " s\n";

  // 拷回 + 输出
  copyGridToHost(h_grid, d_u_old);

  const char* out_env = std::getenv("OUTDIR");
  std::string out_dir = out_env ? std::string(out_env)
                                : (std::string("results/euler/2d/") + prec_name + "/gpu");
  std::system(("mkdir -p " + out_dir).c_str());
  std::ostringstream fname;
  fname << out_dir << "/riemann2d_" << cfg.name << "_hllc_" << N << ".dat";
  writeOutput2D(fname.str(), h_grid, nx2, ny2, gh, dx2, dy2, xmin, ymin);
  std::cout << "Saved: " << fname.str() << "\n";

  freeGridHost(h_grid);
  freeGridDevice(d_u_old);
  freeGridDevice(d_u_new);
  freeGridDevice(d_flux);
  CUDA_CHECK(cudaFree(d_block_max));
  return 0;
}
