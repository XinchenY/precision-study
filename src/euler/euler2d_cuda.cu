/**
 * ============================================================================
 *      2D 欧拉方程求解器 (MUSCL-Hancock + HLLC) — CUDA 版本
 * ============================================================================
 *
 * 核心数值函数来自 include/ 下的共享头文件 (与 CPU 完全相同):
 *   config.hpp       — 全局参数
 *   euler_common.hpp — consToPrim, primToCons, soundSpeed, minmod
 *   riemann.hpp      — hllc_flux, hancock_evolve, muscl_recon
 *
 * 本文件只包含: Grid 类, CUDA Kernels, Host 初始化/输出, main
 *
 * ============================================================================
 */

#include "euler/riemann.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

// ============================================================================
//  CUDA 工具
// ============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define BLOCK_X 16
#define BLOCK_Y 16

enum Processor { HOST_CPU, DEVICE_GPU };

struct Grid {
    Real* data;
    int xCells, yCells;
    int total;

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
//  CUDA Kernels
// ============================================================================

// --- CFL 归约 ---
__global__ void cflReductionKernel(Grid u, Real* d_block_max, int num_blocks_x) {
    extern __shared__ Real sdata[];

    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    Real local_max = (Real)0.0;
    if (i < NX + GHOST && j < NY + GHOST) {
        Real U[4];
        for (int v = 0; v < 4; v++) U[v] = u(i, j, v);
        Real W[4];
        consToPrim(U, W);
        Real c = soundSpeed(W[3], W[0]);
        Real sx = (fabs(W[1]) + c) / DX;
        Real sy = (fabs(W[2]) + c) / DY;
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

// --- 边界条件 ---
__global__ void applyBCKernel(Grid u) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < TOTAL_Y) {
        int j = tid;
        for (int g = 0; g < GHOST; g++)
            for (int v = 0; v < 4; v++) {
                u(g, j, v) = u(GHOST, j, v);
                u(TOTAL_X - 1 - g, j, v) = u(TOTAL_X - 1 - GHOST, j, v);
            }
    }
    if (tid < TOTAL_X) {
        int i = tid;
        for (int g = 0; g < GHOST; g++)
            for (int v = 0; v < 4; v++) {
                u(i, g, v) = u(i, GHOST, v);
                u(i, TOTAL_Y - 1 - g, v) = u(i, TOTAL_Y - 1 - GHOST, v);
            }
    }
}

// --- 通量计算 kernel ---
// 每个线程: 读4个cell → MUSCL → Hancock → HLLC → 写通量
// 使用共享函数: consToPrim, muscl_recon, hancock_evolve, hllc_flux
template<int coord>
__global__ void computeFluxKernel(Grid u_old, Grid fluxes, Real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;

    int limit_i = (coord == 0) ? NX + GHOST + 1 : NX + GHOST;
    int limit_j = (coord == 1) ? NY + GHOST + 1 : NY + GHOST;
    if (i >= limit_i || j >= limit_j) return;

    int n_idx = (coord == 0) ? 1 : 2;
    int t_idx = (coord == 0) ? 2 : 1;
    Real ds = (coord == 0) ? (Real)DX : (Real)DY;

    int di = (coord == 0) ? 1 : 0;
    int dj = (coord == 1) ? 1 : 0;

    int i_LL = i-2*di, j_LL = j-2*dj;
    int i_L  = i-1*di, j_L  = j-1*dj;
    int i_0  = i,      j_0  = j;
    int i_R  = i+1*di, j_R  = j+1*dj;

    if (i_LL < 0 || j_LL < 0 || i_R >= TOTAL_X || j_R >= TOTAL_Y) return;

    // 读取并转换
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

    for (int k = 0; k < 4; k++)
        fluxes(i, j, k) = f[k];
}

// --- 守恒量更新 kernel ---
template<int coord>
__global__ void updateConservativeKernel(Grid u_old, Grid fluxes, Grid u_new, Real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + GHOST;
    int j = blockIdx.y * blockDim.y + threadIdx.y + GHOST;
    if (i >= NX + GHOST || j >= NY + GHOST) return;

    Real ds = (coord == 0) ? (Real)DX : (Real)DY;
    Real ratio = dt / ds;

    int i_R = (coord == 0) ? i + 1 : i;
    int j_R = (coord == 1) ? j + 1 : j;

    for (int k = 0; k < 4; k++)
        u_new(i,j,k) = u_old(i,j,k) - ratio * (fluxes(i_R,j_R,k) - fluxes(i,j,k));
}

// ============================================================================
//  Host 辅助函数
// ============================================================================

void initializeShockBubble(Grid& h_grid) {
    Real g = GAMMA;
    Real rho1 = RHO_AIR, p1 = P_ATM;
    Real c1 = sqrt(g * p1 / rho1);

    Real rho2 = rho1 * ((g+1)*MACH_NUM*MACH_NUM) / ((g-1)*MACH_NUM*MACH_NUM + 2);
    Real p2   = p1 * (2*g*MACH_NUM*MACH_NUM - (g-1)) / (g+1);
    Real u2   = c1 * MACH_NUM * ((Real)1.0 - rho1/rho2);

    printf("===== Shock-Bubble Problem (CUDA) =====\n");
    printf("Pre-shock:  rho=%.4f, p=%.1f\n", (double)rho1, (double)p1);
    printf("Post-shock: rho=%.4f, p=%.1f, u=%.4f\n", (double)rho2, (double)p2, (double)u2);
    printf("=======================================\n");

    Real R2 = BUBBLE_R * BUBBLE_R;
    for (int j = 0; j < TOTAL_Y; j++) {
        Real y = Y_MIN + (j - GHOST + 0.5) * DY;
        for (int i = 0; i < TOTAL_X; i++) {
            Real x = X_MIN + (i - GHOST + 0.5) * DX;
            Real rho, u, v, p;
            Real dx_b = x - BUBBLE_X, dy_b = y - BUBBLE_Y;

            if (dx_b*dx_b + dy_b*dy_b < R2)     { rho=RHO_HELIUM; u=0; v=0; p=p1; }
            else if (x < SHOCK_X)                 { rho=rho2; u=u2; v=0; p=p2; }
            else                                   { rho=rho1; u=0; v=0; p=p1; }

            Real W[4] = {rho, u, v, p};
            Real U[4];
            primToCons(W, U);
            h_grid(i,j,0) = U[0]; h_grid(i,j,1) = U[1];
            h_grid(i,j,2) = U[2]; h_grid(i,j,3) = U[3];
        }
    }
}

void writeOutput(const char* filename, Grid& h_grid, Real t) {
    std::ofstream out(filename);
    out << "# Time = " << t << "\n";
    out << std::scientific;
    out.precision(std::numeric_limits<Real>::max_digits10);
    for (int i = GHOST; i < NX + GHOST; i++) {
        Real x = X_MIN + (i - GHOST + 0.5) * DX;
        for (int j = GHOST; j < NY + GHOST; j++) {
            Real y = Y_MIN + (j - GHOST + 0.5) * DY;
            Real U[4];
            for (int v = 0; v < 4; v++) U[v] = h_grid(i,j,v);
            Real W[4];
            consToPrim(U, W);
            out << x << " " << y << " "
                << W[0] << " " << W[1] << " " << W[2] << " " << W[3] << "\n";
        }
        out << "\n";
    }
    out.close();
    printf("Saved %s (t=%.6e)\n", filename, (double)t);
}

// ============================================================================
//  主程序
// ============================================================================

int main() {
    printf("Grid: %d x %d (with %d ghost -> %d x %d)\n", NX, NY, GHOST, TOTAL_X, TOTAL_Y);
    printf("Domain: x=[%.3f, %.3f], y=[%.3f, %.3f]\n",
           (double)X_MIN, (double)X_MAX, (double)Y_MIN, (double)Y_MAX);
    printf("T_end = %.7f s\n\n", T_END);

    Grid h_grid = createGrid(TOTAL_X, TOTAL_Y, HOST_CPU);
    Grid d_u_old = createGrid(TOTAL_X, TOTAL_Y, DEVICE_GPU);
    Grid d_u_new = createGrid(TOTAL_X, TOTAL_Y, DEVICE_GPU);
    Grid d_flux  = createGrid(TOTAL_X, TOTAL_Y, DEVICE_GPU);

    int grid_rx = (NX + BLOCK_X - 1) / BLOCK_X;
    int grid_ry = (NY + BLOCK_Y - 1) / BLOCK_Y;
    int num_blocks = grid_rx * grid_ry;
    Real* d_block_max;
    CUDA_CHECK(cudaMalloc(&d_block_max, num_blocks * sizeof(Real)));
    Real* h_block_max = new Real[num_blocks];

    initializeShockBubble(h_grid);
    copyGridToDevice(d_u_old, h_grid);

    int bc_threads = max(TOTAL_X, TOTAL_Y);
    int bc_blocks  = (bc_threads + 255) / 256;
    applyBCKernel<<<bc_blocks, 256>>>(d_u_old);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid_inner(grid_rx, grid_ry);
    size_t shared_bytes = BLOCK_X * BLOCK_Y * sizeof(Real);
    dim3 grid_flux_x(((NX+1)+BLOCK_X-1)/BLOCK_X, (NY+BLOCK_Y-1)/BLOCK_Y);
    dim3 grid_flux_y((NX+BLOCK_X-1)/BLOCK_X, ((NY+1)+BLOCK_Y-1)/BLOCK_Y);

    printf("Starting CUDA simulation...\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    Real t = (Real)0.0;
    int step = 0;

    while (t < T_END) {
        // CFL
        cflReductionKernel<<<grid_inner, block, shared_bytes>>>(d_u_old, d_block_max, grid_rx);
        CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max, num_blocks*sizeof(Real), cudaMemcpyDeviceToHost));
        Real max_speed = (Real)0.0;
        for (int b = 0; b < num_blocks; b++)
            max_speed = std::max(max_speed, h_block_max[b]);
        Real dt = CFL_NUM / max_speed;
        if (t + dt > T_END) dt = T_END - t;

        // fp32 安全: dt 太小无法推进 t 时直接退出
        if (dt <= (Real)0.0 || t + dt == t) break;

        // 交替方向分裂 (与 CPU 一致)
        if (step % 2 == 0) {
            // X → Y
            computeFluxKernel<0><<<grid_flux_x, block>>>(d_u_old, d_flux, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
            updateConservativeKernel<0><<<grid_inner, block>>>(d_u_old, d_flux, d_u_new, dt);
            applyBCKernel<<<bc_blocks, 256>>>(d_u_new);
            CUDA_CHECK(cudaDeviceSynchronize());

            computeFluxKernel<1><<<grid_flux_y, block>>>(d_u_new, d_flux, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
            updateConservativeKernel<1><<<grid_inner, block>>>(d_u_new, d_flux, d_u_old, dt);
            applyBCKernel<<<bc_blocks, 256>>>(d_u_old);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            // Y → X
            computeFluxKernel<1><<<grid_flux_y, block>>>(d_u_old, d_flux, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
            updateConservativeKernel<1><<<grid_inner, block>>>(d_u_old, d_flux, d_u_new, dt);
            applyBCKernel<<<bc_blocks, 256>>>(d_u_new);
            CUDA_CHECK(cudaDeviceSynchronize());

            computeFluxKernel<0><<<grid_flux_x, block>>>(d_u_new, d_flux, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
            updateConservativeKernel<0><<<grid_inner, block>>>(d_u_new, d_flux, d_u_old, dt);
            applyBCKernel<<<bc_blocks, 256>>>(d_u_old);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        t += dt;
        step++;
        if (step % 100 == 0) printf("Step %d: t = %.6e\n", step, t);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    printf("\nDone. Steps: %d\n", step);
    printf("Execution Time: %.2f ms (%.4f s)\n", duration.count(), duration.count()/1000.0);

    copyGridToHost(h_grid, d_u_old);
    const char* prec_name = (sizeof(Real) == 8) ? "fp64" : "fp32";
    const char* out_env = std::getenv("OUTDIR");
    std::string out_dir = out_env ? std::string(out_env)
                                  : (std::string("results/euler/2d/") + prec_name);
    std::system(("mkdir -p " + out_dir).c_str());
    std::ostringstream fname;
    fname << out_dir << "/gpu_shock_bubble.dat";
    writeOutput(fname.str().c_str(), h_grid, t);

    freeGridHost(h_grid);
    freeGridDevice(d_u_old);
    freeGridDevice(d_u_new);
    freeGridDevice(d_flux);
    CUDA_CHECK(cudaFree(d_block_max));
    delete[] h_block_max;

    return 0;
}
