/**
 * ============================================================================
 *  mhd_common.hpp — 1D ideal MHD basic operations for the CPU solver
 *  mhd_common.hpp — 1D 理想 MHD 的基础单点公式
 * ============================================================================
 *
 *  English:
 *  This header is intentionally separate from the Euler headers.  The Euler
 *  baseline stays unchanged.  The MHD solver reuses the same finite-volume
 *  programming pattern, but it does not reuse the Euler physical formulae:
 *  the variables, fluxes, wave speeds, and energy definition are different.
 *
 *  中文:
 *  这个头文件故意和 Euler 的头文件分开。Euler baseline 不改动。MHD 只是复用
 *  有限体积程序结构的思路，不复用 Euler 的物理公式；MHD 的变量、通量、波速和
 *  总能量定义都不同。
 *
 *  Primitive variables W / 原始变量 W:
 *    W = (rho, vx, vy, vz, p, Bx, By, Bz)
 *
 *  Conserved variables U / 守恒变量 U:
 *    U = (rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz)
 *
 *  Ideal-MHD total energy density / 理想 MHD 总能量密度:
 *
 *    E = p/(gamma - 1)
 *      + 0.5 * rho * (vx^2 + vy^2 + vz^2)
 *      + 0.5 * (Bx^2 + By^2 + Bz^2)
 *
 *  This is the formula in the thesis theory chapter: internal energy + kinetic
 *  energy + magnetic energy.
 *
 *  这就是理论章节里的公式：内能 + 动能 + 磁能。Brio-Wu shock tube 使用
 *  gamma = 2；在 1D 问题中 Bx 是常数，用来满足 div B = 0。
 * ============================================================================
 */

#ifndef MHD_COMMON_HPP
#define MHD_COMMON_HPP

#include "base.hpp"

#include <algorithm>
#include <cmath>

static const int MHD_NVAR = 8;
// MHD gamma is deliberately separate from the Euler GAMMA=1.4 in config.hpp.
// Brio-Wu uses the default gamma=2.  Other MHD tests can define
// MHD_GAMMA_VALUE before including this header, e.g. Orszag-Tang uses 5/3.
// MHD 的 gamma 故意不复用 Euler 的 GAMMA=1.4。默认值服务 Brio-Wu；
// 其他 MHD 算例可以在 include 前定义 MHD_GAMMA_VALUE。
#ifndef MHD_GAMMA_VALUE
#define MHD_GAMMA_VALUE 2.0
#endif
static const Real MHD_GAMMA = (Real)MHD_GAMMA_VALUE;
static const Real MHD_RHO_FLOOR = (Real)1.0e-12;
static const Real MHD_P_FLOOR = (Real)1.0e-12;

// ============================================================================
//  HLLD 退化判据容差 — 无量纲、按局部物理尺度归一化、且 fp32/fp64 统一,
//  使两种精度运行同一套格式策略(干净的精度对比实验的前提)。
//  HLLD degeneracy tolerance — dimensionless, scaled by local magnitudes,
//  and deliberately IDENTICAL for fp32 and fp64 so both precisions run the
//  same scheme policy (required for a clean precision comparison).
//
//  旧判据把 |分母| 与绝对常数 1e-14 比较: 无量纲错误且实测从不触发
//  (Brio-Wu N=800 与 OT 64^2 禁用后逐位一致), 近退化 star state 全部放行,
//  在电流片处可把恢复压力推成负值 (OT 高分辨率崩溃的根源)。
//  默认 1e-6 约为 8*eps_fp32: 对两种精度都能识别被舍入噪声主导的小分母。
//  sqrt(eps_fp64)=1.5e-8 低于 fp32 的机器精度(1.2e-7), 对 fp32 无保护作用
//  (实测 fp32 在 OT 256^2 仍崩溃); 故不取该值。
//  The default 1e-6 (~8*eps_fp32) can flag denominators dominated by rounding
//  noise in BOTH precisions; sqrt(eps_fp64) sits below fp32 rounding and
//  failed to protect fp32 at OT 256^2.  Override at compile time
//  (-DMHD_HLLD_TOLERANCE_VALUE=...) for the threshold-sensitivity check.
// ============================================================================
#ifndef MHD_HLLD_TOLERANCE_VALUE
#define MHD_HLLD_TOLERANCE_VALUE 1.0e-6
#endif

enum MHDIndex {
  MHD_RHO = 0,
  MHD_MX  = 1,
  MHD_MY  = 2,
  MHD_MZ  = 3,
  MHD_E   = 4,
  MHD_BX  = 5,
  MHD_BY  = 6,
  MHD_BZ  = 7
};

// Primitive arrays use the same 8-slot storage size, but indices 1--4 mean
// velocities and gas pressure rather than momentum and total energy.
// 原始变量也用 8 个位置，但第 1--4 位表示速度和压力，不是动量和总能量。
enum MHDPrimIndex {
  MHD_PR_RHO = 0,
  MHD_PR_VX  = 1,
  MHD_PR_VY  = 2,
  MHD_PR_VZ  = 3,
  MHD_PR_P   = 4,
  MHD_PR_BX  = 5,
  MHD_PR_BY  = 6,
  MHD_PR_BZ  = 7
};

__host__ __device__ inline Real mhdMax(Real a, Real b) {
#ifdef __CUDA_ARCH__
  return fmax(a, b);
#else
  return std::max(a, b);
#endif
}

__host__ __device__ inline Real mhdMin(Real a, Real b) {
#ifdef __CUDA_ARCH__
  return fmin(a, b);
#else
  return std::min(a, b);
#endif
}

__host__ __device__ inline Real mhdAbs(Real a) {
#ifdef __CUDA_ARCH__
  return fabs(a);
#else
  return std::fabs(a);
#endif
}

__host__ __device__ inline Real mhdSqrt(Real a) {
#ifdef __CUDA_ARCH__
  return sqrt(a);
#else
  return std::sqrt(a);
#endif
}

// Minmod slope limiter for MUSCL reconstruction.
// MUSCL 重构用的 minmod 限制器；间断处自动把斜率压小，避免非物理振荡。
__host__ __device__ inline Real mhdMinmod(Real a, Real b) {
  if (a * b <= (Real)0.0) return (Real)0.0;
  return (mhdAbs(a) < mhdAbs(b)) ? a : b;
}

// Primitive -> conserved / 原始变量 -> 守恒变量。
// This is where the MHD total energy differs from Euler: the final term is
// magnetic energy, 0.5*|B|^2.
// 这里是 MHD 和 Euler 最关键的差别：总能量 E 里多了磁能 0.5*|B|^2。
__host__ __device__ inline void mhdPrimToCons(const Real W[MHD_NVAR],
                                              Real U[MHD_NVAR]) {
  Real rho = W[MHD_PR_RHO];
  Real vx = W[MHD_PR_VX];
  Real vy = W[MHD_PR_VY];
  Real vz = W[MHD_PR_VZ];
  Real p = W[MHD_PR_P];
  Real bx = W[MHD_PR_BX];
  Real by = W[MHD_PR_BY];
  Real bz = W[MHD_PR_BZ];

  Real v2 = vx*vx + vy*vy + vz*vz;
  Real b2 = bx*bx + by*by + bz*bz;

  U[MHD_RHO] = rho;
  U[MHD_MX]  = rho * vx;
  U[MHD_MY]  = rho * vy;
  U[MHD_MZ]  = rho * vz;
  U[MHD_E]   = p / (MHD_GAMMA - (Real)1.0)
             + (Real)0.5 * rho * v2
             + (Real)0.5 * b2;
  U[MHD_BX]  = bx;
  U[MHD_BY]  = by;
  U[MHD_BZ]  = bz;
}

// Conserved -> primitive / 守恒变量 -> 原始变量。
// Pressure is recovered by subtracting kinetic and magnetic energy from E.
// 反算压力时，从总能量 E 中扣除动能和磁能，剩下的是内能。
__host__ __device__ inline void mhdConsToPrim(const Real U[MHD_NVAR],
                                              Real W[MHD_NVAR]) {
  Real rho = mhdMax(U[MHD_RHO], MHD_RHO_FLOOR);
  Real vx = U[MHD_MX] / rho;
  Real vy = U[MHD_MY] / rho;
  Real vz = U[MHD_MZ] / rho;
  Real bx = U[MHD_BX];
  Real by = U[MHD_BY];
  Real bz = U[MHD_BZ];

  Real v2 = vx*vx + vy*vy + vz*vz;
  Real b2 = bx*bx + by*by + bz*bz;
  Real p = (MHD_GAMMA - (Real)1.0)
         * (U[MHD_E] - (Real)0.5 * rho * v2 - (Real)0.5 * b2);

  W[MHD_PR_RHO] = rho;
  W[MHD_PR_VX]  = vx;
  W[MHD_PR_VY]  = vy;
  W[MHD_PR_VZ]  = vz;
  W[MHD_PR_P]   = p;
  W[MHD_PR_BX]  = bx;
  W[MHD_PR_BY]  = by;
  W[MHD_PR_BZ]  = bz;
}

// Fast magnetosonic speed in the x direction.
// HLL needs only the outermost signal speeds vx +/- c_f.
// x 方向快磁声速。HLL 只需要最外侧信号速度 vx +/- c_f。
__host__ __device__ inline Real mhdFastSpeedX(const Real W[MHD_NVAR]) {
  Real rho = mhdMax(W[MHD_PR_RHO], MHD_RHO_FLOOR);
  Real p = mhdMax(W[MHD_PR_P], MHD_P_FLOOR);
  Real bx = W[MHD_PR_BX];
  Real by = W[MHD_PR_BY];
  Real bz = W[MHD_PR_BZ];

  Real a2 = MHD_GAMMA * p / rho;
  Real b2 = (bx*bx + by*by + bz*bz) / rho;
  Real cax2 = bx*bx / rho;
  Real disc = (a2 + b2) * (a2 + b2) - (Real)4.0 * a2 * cax2;
  disc = mhdMax(disc, (Real)0.0);
  Real cf2 = (Real)0.5 * (a2 + b2 + mhdSqrt(disc));
  return mhdSqrt(mhdMax(cf2, (Real)0.0));
}

// Fast magnetosonic speed in the y direction.
// y 方向快磁声速，用于二维 CFL 和 y 方向 HLL 外侧波速。
__host__ __device__ inline Real mhdFastSpeedY(const Real W[MHD_NVAR]) {
  Real rho = mhdMax(W[MHD_PR_RHO], MHD_RHO_FLOOR);
  Real p = mhdMax(W[MHD_PR_P], MHD_P_FLOOR);
  Real bx = W[MHD_PR_BX];
  Real by = W[MHD_PR_BY];
  Real bz = W[MHD_PR_BZ];

  Real a2 = MHD_GAMMA * p / rho;
  Real b2 = (bx*bx + by*by + bz*bz) / rho;
  Real cay2 = by*by / rho;
  Real disc = (a2 + b2) * (a2 + b2) - (Real)4.0 * a2 * cay2;
  disc = mhdMax(disc, (Real)0.0);
  Real cf2 = (Real)0.5 * (a2 + b2 + mhdSqrt(disc));
  return mhdSqrt(mhdMax(cf2, (Real)0.0));
}

// Physical ideal-MHD flux in the x direction.
// x 方向理想 MHD 物理通量。它不是 Euler 通量，包含总压和磁张力项。
// Core x-flux from primitive variables W plus total energy e.
// Callers that already have W (HLL/HLLD) use this to avoid a redundant
// cons->prim conversion.
// 从原始变量 W 和总能量 e 直接算 x 通量。已经有 W 的调用者(HLL/HLLD)
// 用它避免重复的守恒->原始转换。
__host__ __device__ inline void mhdFluxXW(const Real W[MHD_NVAR], Real e,
                                          Real F[MHD_NVAR]) {
  Real rho = W[MHD_PR_RHO];
  Real vx = W[MHD_PR_VX];
  Real vy = W[MHD_PR_VY];
  Real vz = W[MHD_PR_VZ];
  Real p = W[MHD_PR_P];
  Real bx = W[MHD_PR_BX];
  Real by = W[MHD_PR_BY];
  Real bz = W[MHD_PR_BZ];

  Real b2 = bx*bx + by*by + bz*bz;
  Real ptot = p + (Real)0.5 * b2;
  Real vdotb = vx*bx + vy*by + vz*bz;

  F[MHD_RHO] = rho * vx;
  F[MHD_MX]  = rho * vx * vx + ptot - bx * bx;
  F[MHD_MY]  = rho * vx * vy - bx * by;
  F[MHD_MZ]  = rho * vx * vz - bx * bz;
  F[MHD_E]   = (e + ptot) * vx - bx * vdotb;
  F[MHD_BX]  = (Real)0.0;
  F[MHD_BY]  = vx * by - bx * vy;
  F[MHD_BZ]  = vx * bz - bx * vz;
}

// Conserved-input wrapper: recover primitives once, then reuse the core.
// 守恒量入口:反算一次原始量后复用上面的核心公式。
__host__ __device__ inline void mhdFluxX(const Real U[MHD_NVAR],
                                         Real F[MHD_NVAR]) {
  Real W[MHD_NVAR];
  mhdConsToPrim(U, W);
  mhdFluxXW(W, U[MHD_E], F);
}

// Physical ideal-MHD flux in the y direction.
// y 方向理想 MHD 通量；二维 Orszag-Tang 会用到它。
// Core y-flux from primitive variables W plus total energy e.
// 从原始变量 W 和总能量 e 直接算 y 通量,供已持有 W 的调用者复用。
__host__ __device__ inline void mhdFluxYW(const Real W[MHD_NVAR], Real e,
                                          Real G[MHD_NVAR]) {
  Real rho = W[MHD_PR_RHO];
  Real vx = W[MHD_PR_VX];
  Real vy = W[MHD_PR_VY];
  Real vz = W[MHD_PR_VZ];
  Real p = W[MHD_PR_P];
  Real bx = W[MHD_PR_BX];
  Real by = W[MHD_PR_BY];
  Real bz = W[MHD_PR_BZ];

  Real b2 = bx*bx + by*by + bz*bz;
  Real ptot = p + (Real)0.5 * b2;
  Real vdotb = vx*bx + vy*by + vz*bz;

  G[MHD_RHO] = rho * vy;
  G[MHD_MX]  = rho * vy * vx - by * bx;
  G[MHD_MY]  = rho * vy * vy + ptot - by * by;
  G[MHD_MZ]  = rho * vy * vz - by * bz;
  G[MHD_E]   = (e + ptot) * vy - by * vdotb;
  G[MHD_BX]  = vy * bx - by * vx;
  G[MHD_BY]  = (Real)0.0;
  G[MHD_BZ]  = vy * bz - by * vz;
}

// Conserved-input wrapper.
// 守恒量入口。
__host__ __device__ inline void mhdFluxY(const Real U[MHD_NVAR],
                                         Real G[MHD_NVAR]) {
  Real W[MHD_NVAR];
  mhdConsToPrim(U, W);
  mhdFluxYW(W, U[MHD_E], G);
}

// First-order HLL flux for 1D ideal MHD.
// This is deliberately conservative and robust for the first Brio-Wu baseline.
// 一阶 HLL 通量：第一版先求稳，用来建立 Brio-Wu 的 CPU FP64 baseline。
// 后续可以在这个基础上加 MUSCL-Hancock 或 HLLD。
__host__ __device__ inline void mhdHllFluxX(const Real UL[MHD_NVAR],
                                            const Real UR[MHD_NVAR],
                                            Real F[MHD_NVAR]) {
  Real WL[MHD_NVAR], WR[MHD_NVAR];
  Real FL[MHD_NVAR], FR[MHD_NVAR];
  mhdConsToPrim(UL, WL);
  mhdConsToPrim(UR, WR);
  mhdFluxXW(WL, UL[MHD_E], FL);   // reuse WL, no redundant cons->prim
  mhdFluxXW(WR, UR[MHD_E], FR);

  Real cfL = mhdFastSpeedX(WL);
  Real cfR = mhdFastSpeedX(WR);
  Real sL = mhdMin(WL[MHD_PR_VX] - cfL, WR[MHD_PR_VX] - cfR);
  Real sR = mhdMax(WL[MHD_PR_VX] + cfL, WR[MHD_PR_VX] + cfR);

  if (sL >= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) F[k] = FL[k];
  } else if (sR <= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) F[k] = FR[k];
  } else {
    Real denom = sR - sL;
    for (int k = 0; k < MHD_NVAR; k++) {
      F[k] = (sR * FL[k] - sL * FR[k] + sL * sR * (UR[k] - UL[k])) / denom;
    }
  }
}

// First-order HLL flux in the y direction.
// y 方向 HLL 通量，用于二维更新和 HLLD 退化回退。
__host__ __device__ inline void mhdHllFluxY(const Real UL[MHD_NVAR],
                                            const Real UR[MHD_NVAR],
                                            Real G[MHD_NVAR]) {
  Real WL[MHD_NVAR], WR[MHD_NVAR];
  Real GL[MHD_NVAR], GR[MHD_NVAR];
  mhdConsToPrim(UL, WL);
  mhdConsToPrim(UR, WR);
  mhdFluxYW(WL, UL[MHD_E], GL);   // reuse WL, no redundant cons->prim
  mhdFluxYW(WR, UR[MHD_E], GR);

  Real cfL = mhdFastSpeedY(WL);
  Real cfR = mhdFastSpeedY(WR);
  Real sL = mhdMin(WL[MHD_PR_VY] - cfL, WR[MHD_PR_VY] - cfR);
  Real sR = mhdMax(WL[MHD_PR_VY] + cfL, WR[MHD_PR_VY] + cfR);

  if (sL >= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) G[k] = GL[k];
  } else if (sR <= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) G[k] = GR[k];
  } else {
    Real denom = sR - sL;
    for (int k = 0; k < MHD_NVAR; k++) {
      G[k] = (sR * GL[k] - sL * GR[k] + sL * sR * (UR[k] - UL[k])) / denom;
    }
  }
}

// Rotate a y-normal Riemann problem into the x-normal ordering used by HLLD.
// 把 y 法向问题旋转成 x 法向变量顺序，从而复用 x 方向 HLLD。
__host__ __device__ inline void mhdRotateYToX(const Real Uy[MHD_NVAR],
                                              Real Ux[MHD_NVAR]) {
  Ux[MHD_RHO] = Uy[MHD_RHO];
  Ux[MHD_MX]  = Uy[MHD_MY];
  Ux[MHD_MY]  = Uy[MHD_MX];
  Ux[MHD_MZ]  = Uy[MHD_MZ];
  Ux[MHD_E]   = Uy[MHD_E];
  Ux[MHD_BX]  = Uy[MHD_BY];
  Ux[MHD_BY]  = Uy[MHD_BX];
  Ux[MHD_BZ]  = Uy[MHD_BZ];
}

// Rotate an x-normal flux back to y-normal conserved-variable ordering.
// 把旋转后得到的 x 通量映射回 y 方向通量。
__host__ __device__ inline void mhdRotateFluxXToY(const Real Fx[MHD_NVAR],
                                                  Real Gy[MHD_NVAR]) {
  Gy[MHD_RHO] = Fx[MHD_RHO];
  Gy[MHD_MX]  = Fx[MHD_MY];
  Gy[MHD_MY]  = Fx[MHD_MX];
  Gy[MHD_MZ]  = Fx[MHD_MZ];
  Gy[MHD_E]   = Fx[MHD_E];
  Gy[MHD_BX]  = Fx[MHD_BY];
  Gy[MHD_BY]  = Fx[MHD_BX];
  Gy[MHD_BZ]  = Fx[MHD_BZ];
}

__host__ __device__ inline Real mhdSignNonZero(Real a) {
  return (a < (Real)0.0) ? (Real)-1.0 : (Real)1.0;
}

__host__ __device__ inline Real mhdHlldTolerance() {
  return (Real)MHD_HLLD_TOLERANCE_VALUE;
}

// 统一的尺度化小量判据: |value| 相对局部尺度 scale 是否已退化。
// The one scale-aware smallness test used by every HLLD degeneracy guard.
__host__ __device__ inline bool mhdHlldIsSmall(Real value, Real scale) {
  return mhdAbs(value) <= mhdHlldTolerance() * scale;
}

// 物理一致性检查: 守恒态必须能恢复出正密度和正压力。无阈值调参,
// 且 !(x>y) 对 NaN 为真, 顺带拦截 NaN。
// Physical-consistency check: the conserved state must recover positive
// density and pressure.  Threshold-free; the negated comparison also
// catches NaN (all comparisons with NaN are false).
__host__ __device__ inline bool mhdConservedHasPositivePressure(const Real U[MHD_NVAR]) {
  if (!(U[MHD_RHO] > MHD_RHO_FLOOR)) return false;
  Real W[MHD_NVAR];
  mhdConsToPrim(U, W);
  return (W[MHD_PR_RHO] > MHD_RHO_FLOOR && W[MHD_PR_P] > MHD_P_FLOOR);
}

// HLLD 通量返回状态码, 由调用方统计 fallback 次数与原因; 共享头文件本身
// 不保存全局计数 (host/device 兼容, 无全局状态)。
// Status returned by an HLLD flux evaluation.  Returning the reason keeps
// fallback diagnostics in the caller and avoids global state in this
// host/device header.
static const int MHD_HLLD_OK = 0;
static const int MHD_HLLD_FALLBACK_DEGENERATE = 1;
static const int MHD_HLLD_FALLBACK_NONPHYSICAL = 2;

// Build one of the HLLD star states adjacent to the outer fast waves.
// 构造 HLLD 外侧 fast wave 后面的 star state。若分母退化或中间态非物理,
// 返回对应状态码; 上层会在该界面局部回退到 HLL。
__host__ __device__ inline int mhdHlldStarStateX(const Real U[MHD_NVAR],
                                                 const Real W[MHD_NVAR],
                                                 Real s,
                                                 Real sm,
                                                 Real pt_star,
                                                 Real bx,
                                                 Real Ustar[MHD_NVAR]) {
  Real rho = mhdMax(W[MHD_PR_RHO], MHD_RHO_FLOOR);
  Real vx = W[MHD_PR_VX];
  Real vy = W[MHD_PR_VY];
  Real vz = W[MHD_PR_VZ];
  Real p = mhdMax(W[MHD_PR_P], MHD_P_FLOOR);
  Real by = W[MHD_PR_BY];
  Real bz = W[MHD_PR_BZ];
  Real e = U[MHD_E];

  Real s_minus_vx = s - vx;
  Real s_minus_sm = s - sm;
  // 波速差相对局部波速尺度退化 → fallback。
  // Wave-speed gap degenerate relative to the local wave-speed scale.
  Real wave_scale = mhdMax(mhdAbs(s), mhdAbs(sm));
  if (mhdHlldIsSmall(s_minus_sm, wave_scale)) {
    return MHD_HLLD_FALLBACK_DEGENERATE;
  }

  Real rho_star = rho * s_minus_vx / s_minus_sm;
  if (rho_star <= MHD_RHO_FLOOR) {
    return MHD_HLLD_FALLBACK_NONPHYSICAL;
  }

  Real denom = rho * s_minus_vx * s_minus_sm - bx * bx;
  // Alfven 退化: |denom| 不能相对其两个组成项过小, 否则 star state 公式
  // 的放大因子失控 / |denom| must not be small RELATIVE to its two terms.
  Real denom_scale = mhdAbs(rho * s_minus_vx * s_minus_sm) + bx * bx;
  if (mhdHlldIsSmall(denom, denom_scale)) {
    return MHD_HLLD_FALLBACK_DEGENERATE;
  }

  Real common = sm - vx;
  Real vy_star = vy - bx * by * common / denom;
  Real vz_star = vz - bx * bz * common / denom;
  Real by_star = by * (rho * s_minus_vx * s_minus_vx - bx * bx) / denom;
  Real bz_star = bz * (rho * s_minus_vx * s_minus_vx - bx * bx) / denom;

  Real b2 = bx*bx + by*by + bz*bz;
  Real pt = p + (Real)0.5 * b2;
  Real vdotb = vx*bx + vy*by + vz*bz;
  Real vstar_dot_bstar = sm*bx + vy_star*by_star + vz_star*bz_star;
  Real e_star = (s_minus_vx * e - pt * vx + pt_star * sm
               + bx * (vdotb - vstar_dot_bstar)) / s_minus_sm;

  Ustar[MHD_RHO] = rho_star;
  Ustar[MHD_MX]  = rho_star * sm;
  Ustar[MHD_MY]  = rho_star * vy_star;
  Ustar[MHD_MZ]  = rho_star * vz_star;
  Ustar[MHD_E]   = e_star;
  Ustar[MHD_BX]  = bx;
  Ustar[MHD_BY]  = by_star;
  Ustar[MHD_BZ]  = bz_star;

  // 物理一致性检查: star state 必须能恢复出正密度/正压力, 否则报告非物理
  // 让上层回退 HLL。
  // Physical-consistency check: the star state must recover positive
  // density and pressure, else the caller falls back to HLL.
  return mhdConservedHasPositivePressure(Ustar)
       ? MHD_HLLD_OK : MHD_HLLD_FALLBACK_NONPHYSICAL;
}

// HLLD flux for 1D ideal MHD.
// HLLD resolves the two fast waves, two rotational/Alfven waves, and the
// contact wave.  It is less diffusive than HLL for Brio-Wu-type MHD profiles.
// HLLD 能分辨 fast waves、Alfven waves 和 contact wave，比 HLL 更适合
// Brio-Wu 这种 MHD 激波管。若中间状态退化，则自动回退到 HLL。
__host__ __device__ inline int mhdHlldFluxX(const Real UL[MHD_NVAR],
                                            const Real UR[MHD_NVAR],
                                            Real F[MHD_NVAR]) {
  Real WL[MHD_NVAR], WR[MHD_NVAR];
  Real FL[MHD_NVAR], FR[MHD_NVAR];
  mhdConsToPrim(UL, WL);
  mhdConsToPrim(UR, WR);
  mhdFluxXW(WL, UL[MHD_E], FL);   // reuse WL, no redundant cons->prim
  mhdFluxXW(WR, UR[MHD_E], FR);

  Real cfL = mhdFastSpeedX(WL);
  Real cfR = mhdFastSpeedX(WR);
  Real sL = mhdMin(WL[MHD_PR_VX] - cfL, WR[MHD_PR_VX] - cfR);
  Real sR = mhdMax(WL[MHD_PR_VX] + cfL, WR[MHD_PR_VX] + cfR);

  if (sL >= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) F[k] = FL[k];
    return MHD_HLLD_OK;
  }
  if (sR <= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) F[k] = FR[k];
    return MHD_HLLD_OK;
  }

  Real rhoL = mhdMax(WL[MHD_PR_RHO], MHD_RHO_FLOOR);
  Real rhoR = mhdMax(WR[MHD_PR_RHO], MHD_RHO_FLOOR);
  Real vxL = WL[MHD_PR_VX];
  Real vxR = WR[MHD_PR_VX];
  Real bx = (Real)0.5 * (WL[MHD_PR_BX] + WR[MHD_PR_BX]);

  Real b2L = bx*bx + WL[MHD_PR_BY]*WL[MHD_PR_BY] + WL[MHD_PR_BZ]*WL[MHD_PR_BZ];
  Real b2R = bx*bx + WR[MHD_PR_BY]*WR[MHD_PR_BY] + WR[MHD_PR_BZ]*WR[MHD_PR_BZ];
  Real ptL = mhdMax(WL[MHD_PR_P], MHD_P_FLOOR) + (Real)0.5 * b2L;
  Real ptR = mhdMax(WR[MHD_PR_P], MHD_P_FLOOR) + (Real)0.5 * b2R;

  Real denom_sm = rhoR * (sR - vxR) - rhoL * (sL - vxL);
  // Scale-aware 退化判据: denom_sm 相对其组成项、bx 相对总磁场强度。
  // 任一退化 → 整个界面回退 HLL (M&K 的标准处理)。
  // Scale-aware degeneracy tests: denom_sm vs its own terms, bx vs the
  // total field strength. Either degenerate -> fall back to HLL.
  Real denom_sm_scale = mhdAbs(rhoR * (sR - vxR)) + mhdAbs(rhoL * (sL - vxL));
  Real b_scale = mhdSqrt(mhdMax(b2L, b2R));
  if (mhdHlldIsSmall(denom_sm, denom_sm_scale) ||
      mhdHlldIsSmall(bx, b_scale)) {
    mhdHllFluxX(UL, UR, F);
    return MHD_HLLD_FALLBACK_DEGENERATE;
  }

  Real sm = (rhoR * vxR * (sR - vxR)
           - rhoL * vxL * (sL - vxL)
           + ptL - ptR) / denom_sm;
  Real pt_star_L = ptL + rhoL * (sL - vxL) * (sm - vxL);
  Real pt_star_R = ptR + rhoR * (sR - vxR) * (sm - vxR);
  Real pt_star = (Real)0.5 * (pt_star_L + pt_star_R);

  Real ULs[MHD_NVAR], URs[MHD_NVAR];
  int left_status = mhdHlldStarStateX(UL, WL, sL, sm, pt_star, bx, ULs);
  int right_status = mhdHlldStarStateX(UR, WR, sR, sm, pt_star, bx, URs);
  if (left_status != MHD_HLLD_OK || right_status != MHD_HLLD_OK) {
    mhdHllFluxX(UL, UR, F);
    return (left_status != MHD_HLLD_OK) ? left_status : right_status;
  }

  Real sqrt_rhoLs = mhdSqrt(mhdMax(ULs[MHD_RHO], MHD_RHO_FLOOR));
  Real sqrt_rhoRs = mhdSqrt(mhdMax(URs[MHD_RHO], MHD_RHO_FLOOR));
  Real sgn_bx = mhdSignNonZero(bx);
  Real sLs = sm - mhdAbs(bx) / sqrt_rhoLs;
  Real sRs = sm + mhdAbs(bx) / sqrt_rhoRs;
  Real denom = sqrt_rhoLs + sqrt_rhoRs;
  if (denom <= (Real)0.0) {
    mhdHllFluxX(UL, UR, F);
    return MHD_HLLD_FALLBACK_DEGENERATE;
  }

  Real vyLs = ULs[MHD_MY] / ULs[MHD_RHO];
  Real vzLs = ULs[MHD_MZ] / ULs[MHD_RHO];
  Real byLs = ULs[MHD_BY];
  Real bzLs = ULs[MHD_BZ];
  Real vyRs = URs[MHD_MY] / URs[MHD_RHO];
  Real vzRs = URs[MHD_MZ] / URs[MHD_RHO];
  Real byRs = URs[MHD_BY];
  Real bzRs = URs[MHD_BZ];

  Real vyD = (sqrt_rhoLs * vyLs + sqrt_rhoRs * vyRs
            + sgn_bx * (byRs - byLs)) / denom;
  Real vzD = (sqrt_rhoLs * vzLs + sqrt_rhoRs * vzRs
            + sgn_bx * (bzRs - bzLs)) / denom;
  Real byD = (sqrt_rhoLs * byRs + sqrt_rhoRs * byLs
            + sgn_bx * sqrt_rhoLs * sqrt_rhoRs * (vyRs - vyLs)) / denom;
  Real bzD = (sqrt_rhoLs * bzRs + sqrt_rhoRs * bzLs
            + sgn_bx * sqrt_rhoLs * sqrt_rhoRs * (vzRs - vzLs)) / denom;

  Real ULD[MHD_NVAR], URD[MHD_NVAR];
  for (int k = 0; k < MHD_NVAR; k++) {
    ULD[k] = ULs[k];
    URD[k] = URs[k];
  }

  ULD[MHD_MY] = ULD[MHD_RHO] * vyD;
  ULD[MHD_MZ] = ULD[MHD_RHO] * vzD;
  ULD[MHD_BY] = byD;
  ULD[MHD_BZ] = bzD;
  URD[MHD_MY] = URD[MHD_RHO] * vyD;
  URD[MHD_MZ] = URD[MHD_RHO] * vzD;
  URD[MHD_BY] = byD;
  URD[MHD_BZ] = bzD;

  Real vdotbLD = sm*bx + vyD*byD + vzD*bzD;
  Real vdotbLs = sm*bx + vyLs*byLs + vzLs*bzLs;
  Real vdotbRD = vdotbLD;
  Real vdotbRs = sm*bx + vyRs*byRs + vzRs*bzRs;
  ULD[MHD_E] = ULs[MHD_E] - sgn_bx * sqrt_rhoLs * (vdotbLs - vdotbLD);
  URD[MHD_E] = URs[MHD_E] + sgn_bx * sqrt_rhoRs * (vdotbRs - vdotbRD);

  // double-star 状态同样必须能恢复出正密度/正压力, 否则回退 HLL。
  // The double-star states must also recover positive density and pressure.
  if (!mhdConservedHasPositivePressure(ULD) ||
      !mhdConservedHasPositivePressure(URD)) {
    mhdHllFluxX(UL, UR, F);
    return MHD_HLLD_FALLBACK_NONPHYSICAL;
  }

  if (sLs >= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) F[k] = FL[k] + sL * (ULs[k] - UL[k]);
  } else if (sm >= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) {
      F[k] = FL[k] + sL * (ULs[k] - UL[k]) + sLs * (ULD[k] - ULs[k]);
    }
  } else if (sRs >= (Real)0.0) {
    for (int k = 0; k < MHD_NVAR; k++) {
      F[k] = FR[k] + sR * (URs[k] - UR[k]) + sRs * (URD[k] - URs[k]);
    }
  } else {
    for (int k = 0; k < MHD_NVAR; k++) F[k] = FR[k] + sR * (URs[k] - UR[k]);
  }
  return MHD_HLLD_OK;
}

// HLLD flux in the y direction, implemented by rotating the normal direction.
// y 方向 HLLD：通过变量旋转复用 x 方向 HLLD，避免写两套复杂公式。
__host__ __device__ inline int mhdHlldFluxY(const Real UL[MHD_NVAR],
                                            const Real UR[MHD_NVAR],
                                            Real G[MHD_NVAR]) {
  Real ULx[MHD_NVAR], URx[MHD_NVAR], Fx[MHD_NVAR];
  mhdRotateYToX(UL, ULx);
  mhdRotateYToX(UR, URx);
  int status = mhdHlldFluxX(ULx, URx, Fx);
  mhdRotateFluxXToY(Fx, G);
  return status;
}

#endif // MHD_COMMON_HPP
