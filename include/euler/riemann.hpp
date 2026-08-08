/**
 * ============================================================================
 *  riemann.hpp — HLLC 近似 Riemann 求解器 + MUSCL 重构 + Hancock 半步演化
 * ============================================================================
 *
 *  CPU 和 GPU 共用的核心数值函数。
 *  所有函数使用 Real 类型 (在 config.hpp 中定义为 double 或 float)。
 *  所有函数标记 __host__ __device__，可同时在主机和设备上调用。
 *
 *  波速估计: 全部使用 Davis bounded estimate (Toro 2009, Eq.10.48)
 *    S_L = min(u_L - c_L, u_R - c_R)
 *    S_R = max(u_L + c_L, u_R + c_R)
 *
 *  函数列表:
 *    muscl_recon()    — MUSCL 线性重构 (minmod 限制器)
 *    hancock_evolve() — Hancock 半时间步演化
 *    hllc_flux_x()    — x方向 HLLC 通量 (法向=u, 切向=v)
 *    hllc_flux_y()    — y方向 HLLC 通量 (法向=v, 切向=u)
 *    hllc_flux()      — 通用方向 HLLC (用于 CUDA template<coord> kernel)
 *    hllc_flux_1d()   — 1D HLLC (用于 Sod 等 1D 测试)
 *
 * ============================================================================
 */

#ifndef RIEMANN_HPP
#define RIEMANN_HPP

#include "euler_common.hpp"
#include <algorithm>

// ============================================================================
//  MUSCL 线性重构 (逐点, 单个变量)
//
//  输入: 三个相邻 cell 的某一原始变量值 qm(i-1), q0(i), qp(i+1)
//  输出: cell i 的左面值 q_left (i-1/2) 和右面值 q_right (i+1/2)
//  限制器: minmod, 防止在间断处产生虚假振荡
// ============================================================================
__host__ __device__ inline void muscl_recon(Real qm, Real q0, Real qp,
                                            Real &q_left, Real &q_right) {
  Real slope = minmod(q0 - qm, qp - q0);
  q_left  = q0 - (Real)0.5 * slope;
  q_right = q0 + (Real)0.5 * slope;
}

// ============================================================================
//  Hancock 半步演化 (逐点)
//
//  目的: 将 MUSCL 重构出的面值做半个时间步的演化,
//        使整体格式达到时间二阶精度 (predictor-corrector 思想)
//
//  参数:
//    W_face_evolve[4] — 要演化的面的原始变量 (ρ, u, v, p)
//    W_lf[4]          — 该 cell 左面的原始值 (用于计算通量差)
//    W_rf[4]          — 该 cell 右面的原始值
//    W_half[4]        — 输出: 演化后的原始变量
//    factor           — = 0.5 * dt / ds
//    n_idx            — 法向速度分量索引 (x方向=1, y方向=2)
//    t_idx            — 切向速度分量索引 (x方向=2, y方向=1)
// ============================================================================
__host__ __device__ inline void hancock_evolve(
    const Real W_face_evolve[4],
    const Real W_lf[4], const Real W_rf[4],
    Real W_half[4], Real factor, int n_idx, int t_idx)
{
  // 面值 → 守恒量
  Real U_face[4];
  primToCons(W_face_evolve, U_face);

  // 左右面通量差 dF = F_right - F_left
  Real rL = W_lf[0], uL = W_lf[n_idx], vL = W_lf[t_idx], pL = W_lf[3];
  Real rR = W_rf[0], uR = W_rf[n_idx], vR = W_rf[t_idx], pR = W_rf[3];

  Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rL * (uL*uL + vL*vL);
  Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rR * (uR*uR + vR*vR);

  Real dF[4];
  dF[0]     = rR*uR - rL*uL;
  dF[n_idx] = (rR*uR*uR + pR) - (rL*uL*uL + pL);
  dF[t_idx] = rR*uR*vR - rL*uL*vL;
  dF[3]     = (ER+pR)*uR - (EL+pL)*uL;

  // 半步演化 U_half = U_face - factor * dF
  Real U_half[4];
  for (int k = 0; k < 4; k++)
    U_half[k] = U_face[k] - factor * dF[k];

  consToPrim(U_half, W_half);
}

// ============================================================================
//  HLLC 通量 — x方向
//
//  法向速度 = u (分量 [1]), 切向速度 = v (分量 [2])
//  波速: Davis bounded estimate (Toro 2009, Eq.10.48)
//    S_L = min(u_L - c_L, u_R - c_R)
//    S_R = max(u_L + c_L, u_R + c_R)
// ============================================================================
__host__ __device__ inline void hllc_flux_x(const Real WL[4], const Real WR[4], Real f[4]) {
  Real rhoL = WL[0], uL = WL[1], vL = WL[2], pL = WL[3];
  Real rhoR = WR[0], uR = WR[1], vR = WR[2], pR = WR[3];

  Real cL = soundSpeed(pL, rhoL);
  Real cR = soundSpeed(pR, rhoR);

  // Davis 波速
#ifdef __CUDA_ARCH__
  Real S_L = fmin(uL - cL, uR - cR);
  Real S_R = fmax(uL + cL, uR + cR);
#else
  Real S_L = std::min(uL - cL, uR - cR);
  Real S_R = std::max(uL + cL, uR + cR);
#endif

  Real denom = rhoL * (S_L - uL) - rhoR * (S_R - uR);
  Real S_star = (pR - pL + rhoL*uL*(S_L - uL) - rhoR*uR*(S_R - uR)) / denom;

  Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rhoL * (uL*uL + vL*vL);
  Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rhoR * (uR*uR + vR*vR);

  if (S_L > (Real)0.0) {
    f[0] = rhoL * uL;
    f[1] = rhoL * uL * uL + pL;
    f[2] = rhoL * uL * vL;
    f[3] = (EL + pL) * uL;
  } else if (S_star > (Real)0.0) {
    Real coeff = rhoL * (S_L - uL) / (S_L - S_star);
    Real U0s = coeff;
    Real U1s = coeff * S_star;
    Real U2s = coeff * vL;
    Real U3s = coeff * (EL/rhoL + (S_star - uL)*(S_star + pL/(rhoL*(S_L - uL))));
    Real fL0 = rhoL*uL, fL1 = rhoL*uL*uL + pL;
    Real fL2 = rhoL*uL*vL, fL3 = (EL+pL)*uL;
    f[0] = fL0 + S_L * (U0s - rhoL);
    f[1] = fL1 + S_L * (U1s - rhoL*uL);
    f[2] = fL2 + S_L * (U2s - rhoL*vL);
    f[3] = fL3 + S_L * (U3s - EL);
  } else if (S_R > (Real)0.0) {
    Real coeff = rhoR * (S_R - uR) / (S_R - S_star);
    Real U0s = coeff;
    Real U1s = coeff * S_star;
    Real U2s = coeff * vR;
    Real U3s = coeff * (ER/rhoR + (S_star - uR)*(S_star + pR/(rhoR*(S_R - uR))));
    Real fR0 = rhoR*uR, fR1 = rhoR*uR*uR + pR;
    Real fR2 = rhoR*uR*vR, fR3 = (ER+pR)*uR;
    f[0] = fR0 + S_R * (U0s - rhoR);
    f[1] = fR1 + S_R * (U1s - rhoR*uR);
    f[2] = fR2 + S_R * (U2s - rhoR*vR);
    f[3] = fR3 + S_R * (U3s - ER);
  } else {
    f[0] = rhoR * uR;
    f[1] = rhoR * uR * uR + pR;
    f[2] = rhoR * uR * vR;
    f[3] = (ER + pR) * uR;
  }
}

// ============================================================================
//  HLLC 通量 — y方向
//
//  法向速度 = v (分量 [2]), 切向速度 = u (分量 [1])
//  波速: Davis bounded estimate
// ============================================================================
__host__ __device__ inline void hllc_flux_y(const Real WB[4], const Real WT[4], Real g[4]) {
  Real rhoB = WB[0], uB = WB[1], vB = WB[2], pB = WB[3];
  Real rhoT = WT[0], uT = WT[1], vT = WT[2], pT = WT[3];

  Real cB = soundSpeed(pB, rhoB);
  Real cT = soundSpeed(pT, rhoT);

#ifdef __CUDA_ARCH__
  Real S_B = fmin(vB - cB, vT - cT);
  Real S_T = fmax(vB + cB, vT + cT);
#else
  Real S_B = std::min(vB - cB, vT - cT);
  Real S_T = std::max(vB + cB, vT + cT);
#endif

  Real denom = rhoB * (S_B - vB) - rhoT * (S_T - vT);
  Real S_star = (pT - pB + rhoB*vB*(S_B - vB) - rhoT*vT*(S_T - vT)) / denom;

  Real EB = pB / (GAMMA - (Real)1.0) + (Real)0.5 * rhoB * (uB*uB + vB*vB);
  Real ET = pT / (GAMMA - (Real)1.0) + (Real)0.5 * rhoT * (uT*uT + vT*vT);

  if (S_B > (Real)0.0) {
    g[0] = rhoB * vB;
    g[1] = rhoB * uB * vB;
    g[2] = rhoB * vB * vB + pB;
    g[3] = (EB + pB) * vB;
  } else if (S_star > (Real)0.0) {
    Real coeff = rhoB * (S_B - vB) / (S_B - S_star);
    Real U0s = coeff;
    Real U1s = coeff * uB;
    Real U2s = coeff * S_star;
    Real U3s = coeff * (EB/rhoB + (S_star - vB)*(S_star + pB/(rhoB*(S_B - vB))));
    Real gB0 = rhoB*vB, gB1 = rhoB*uB*vB;
    Real gB2 = rhoB*vB*vB + pB, gB3 = (EB+pB)*vB;
    g[0] = gB0 + S_B * (U0s - rhoB);
    g[1] = gB1 + S_B * (U1s - rhoB*uB);
    g[2] = gB2 + S_B * (U2s - rhoB*vB);
    g[3] = gB3 + S_B * (U3s - EB);
  } else if (S_T > (Real)0.0) {
    Real coeff = rhoT * (S_T - vT) / (S_T - S_star);
    Real U0s = coeff;
    Real U1s = coeff * uT;
    Real U2s = coeff * S_star;
    Real U3s = coeff * (ET/rhoT + (S_star - vT)*(S_star + pT/(rhoT*(S_T - vT))));
    Real gT0 = rhoT*vT, gT1 = rhoT*uT*vT;
    Real gT2 = rhoT*vT*vT + pT, gT3 = (ET+pT)*vT;
    g[0] = gT0 + S_T * (U0s - rhoT);
    g[1] = gT1 + S_T * (U1s - rhoT*uT);
    g[2] = gT2 + S_T * (U2s - rhoT*vT);
    g[3] = gT3 + S_T * (U3s - ET);
  } else {
    g[0] = rhoT * vT;
    g[1] = rhoT * uT * vT;
    g[2] = rhoT * vT * vT + pT;
    g[3] = (ET + pT) * vT;
  }
}

// ============================================================================
//  HLLC 通量 — 通用方向 (CUDA template<coord> kernel 使用)
//
//  通过 n_idx/t_idx 参数区分方向:
//    x方向: n_idx=1 (法向=u), t_idx=2 (切向=v)
//    y方向: n_idx=2 (法向=v), t_idx=1 (切向=u)
//  波速: Davis bounded estimate (两个方向统一)
// ============================================================================
__host__ __device__ inline void hllc_flux(const Real WL_h[4], const Real WR_h[4],
                                          Real f[4], int n_idx, int t_idx) {
  Real rhoL = WL_h[0], uL = WL_h[n_idx], vL = WL_h[t_idx], pL = WL_h[3];
  Real rhoR = WR_h[0], uR = WR_h[n_idx], vR = WR_h[t_idx], pR = WR_h[3];

  Real cL = soundSpeed(pL, rhoL);
  Real cR = soundSpeed(pR, rhoR);

#ifdef __CUDA_ARCH__
  Real S_L = fmin(uL - cL, uR - cR);
  Real S_R = fmax(uL + cL, uR + cR);
#else
  Real S_L = std::min(uL - cL, uR - cR);
  Real S_R = std::max(uL + cL, uR + cR);
#endif

  Real denom = rhoL*(S_L - uL) - rhoR*(S_R - uR);
  Real S_star = (pR - pL + rhoL*uL*(S_L - uL) - rhoR*uR*(S_R - uR)) / denom;

  Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rhoL * (uL*uL + vL*vL);
  Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rhoR * (uR*uR + vR*vR);

  if (S_L > (Real)0.0) {
    f[0] = rhoL * uL;
    f[n_idx] = rhoL * uL * uL + pL;
    f[t_idx] = rhoL * uL * vL;
    f[3] = (EL + pL) * uL;
  } else if (S_star > (Real)0.0) {
    Real coeff = rhoL * (S_L - uL) / (S_L - S_star);
    Real U0s = coeff;
    Real U1s = coeff * S_star;
    Real U2s = coeff * vL;
    Real U3s = coeff * (EL/rhoL + (S_star - uL)*(S_star + pL/(rhoL*(S_L - uL))));
    Real fL0 = rhoL*uL, fL1 = rhoL*uL*uL + pL;
    Real fL2 = rhoL*uL*vL, fL3 = (EL+pL)*uL;
    f[0]     = fL0 + S_L * (U0s - rhoL);
    f[n_idx] = fL1 + S_L * (U1s - rhoL*uL);
    f[t_idx] = fL2 + S_L * (U2s - rhoL*vL);
    f[3]     = fL3 + S_L * (U3s - EL);
  } else if (S_R > (Real)0.0) {
    Real coeff = rhoR * (S_R - uR) / (S_R - S_star);
    Real U0s = coeff;
    Real U1s = coeff * S_star;
    Real U2s = coeff * vR;
    Real U3s = coeff * (ER/rhoR + (S_star - uR)*(S_star + pR/(rhoR*(S_R - uR))));
    Real fR0 = rhoR*uR, fR1 = rhoR*uR*uR + pR;
    Real fR2 = rhoR*uR*vR, fR3 = (ER+pR)*uR;
    f[0]     = fR0 + S_R * (U0s - rhoR);
    f[n_idx] = fR1 + S_R * (U1s - rhoR*uR);
    f[t_idx] = fR2 + S_R * (U2s - rhoR*vR);
    f[3]     = fR3 + S_R * (U3s - ER);
  } else {
    f[0] = rhoR * uR;
    f[n_idx] = rhoR * uR * uR + pR;
    f[t_idx] = rhoR * uR * vR;
    f[3] = (ER + pR) * uR;
  }
}

// ============================================================================
//  1D HLLC — 用于 Sod/Lax 等 1D 测试
//
//  输入: WL = (ρ, u, p), WR = (ρ, u, p)   [1D, 无 v 分量]
//  输出: f  = (F_ρ, F_ρu, F_E)
//  波速: Davis bounded estimate (与 2D 统一)
// ============================================================================
__host__ __device__ inline void hllc_flux_1d(const Real WL[3], const Real WR[3], Real f[3]) {
  Real rhoL = WL[0], uL = WL[1], pL = WL[2];
  Real rhoR = WR[0], uR = WR[1], pR = WR[2];

  Real cL = soundSpeed(pL, rhoL);
  Real cR = soundSpeed(pR, rhoR);

#ifdef __CUDA_ARCH__
  Real S_L = fmin(uL - cL, uR - cR);
  Real S_R = fmax(uL + cL, uR + cR);
#else
  Real S_L = std::min(uL - cL, uR - cR);
  Real S_R = std::max(uL + cL, uR + cR);
#endif

  Real denom = rhoL*(S_L - uL) - rhoR*(S_R - uR);
  Real S_star = (pR - pL + rhoL*uL*(S_L - uL) - rhoR*uR*(S_R - uR)) / denom;

  Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rhoL * uL * uL;
  Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rhoR * uR * uR;

  if (S_L > (Real)0.0) {
    f[0] = rhoL * uL;
    f[1] = rhoL * uL * uL + pL;
    f[2] = (EL + pL) * uL;
  } else if (S_star > (Real)0.0) {
    Real coeff = rhoL * (S_L - uL) / (S_L - S_star);
    Real U0s = coeff;
    Real U1s = coeff * S_star;
    Real U2s = coeff * (EL/rhoL + (S_star - uL)*(S_star + pL/(rhoL*(S_L - uL))));
    Real fL0 = rhoL*uL, fL1 = rhoL*uL*uL + pL, fL2 = (EL+pL)*uL;
    f[0] = fL0 + S_L * (U0s - rhoL);
    f[1] = fL1 + S_L * (U1s - rhoL*uL);
    f[2] = fL2 + S_L * (U2s - EL);
  } else if (S_R > (Real)0.0) {
    Real coeff = rhoR * (S_R - uR) / (S_R - S_star);
    Real U0s = coeff;
    Real U1s = coeff * S_star;
    Real U2s = coeff * (ER/rhoR + (S_star - uR)*(S_star + pR/(rhoR*(S_R - uR))));
    Real fR0 = rhoR*uR, fR1 = rhoR*uR*uR + pR, fR2 = (ER+pR)*uR;
    f[0] = fR0 + S_R * (U0s - rhoR);
    f[1] = fR1 + S_R * (U1s - rhoR*uR);
    f[2] = fR2 + S_R * (U2s - ER);
  } else {
    f[0] = rhoR * uR;
    f[1] = rhoR * uR * uR + pR;
    f[2] = (ER + pR) * uR;
  }
}

// ############################################################################
//  HLL 近似 Riemann 求解器
//  与 HLLC 相比: 没有接触波 S_*, 只用 S_L/S_R, 更耗散但运算更少
//  用于和 HLLC 的精度对比实验
// ############################################################################

// ============================================================================
//  HLL 通量 — 1D
// ============================================================================
__host__ __device__ inline void hll_flux_1d(const Real WL[3], const Real WR[3], Real f[3]) {
  Real rhoL = WL[0], uL = WL[1], pL = WL[2];
  Real rhoR = WR[0], uR = WR[1], pR = WR[2];

  Real cL = soundSpeed(pL, rhoL);
  Real cR = soundSpeed(pR, rhoR);

#ifdef __CUDA_ARCH__
  Real S_L = fmin(uL - cL, uR - cR);
  Real S_R = fmax(uL + cL, uR + cR);
#else
  Real S_L = std::min(uL - cL, uR - cR);
  Real S_R = std::max(uL + cL, uR + cR);
#endif

  Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rhoL * uL * uL;
  Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rhoR * uR * uR;

  Real FL[3] = { rhoL*uL, rhoL*uL*uL + pL, (EL+pL)*uL };
  Real FR[3] = { rhoR*uR, rhoR*uR*uR + pR, (ER+pR)*uR };
  Real UL[3] = { rhoL, rhoL*uL, EL };
  Real UR[3] = { rhoR, rhoR*uR, ER };

  if (S_L > (Real)0.0) {
    for (int k = 0; k < 3; k++) f[k] = FL[k];
  } else if (S_R > (Real)0.0) {
    // HLL 中间态通量: F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
    Real inv = (Real)1.0 / (S_R - S_L);
    for (int k = 0; k < 3; k++)
      f[k] = (S_R * FL[k] - S_L * FR[k] + S_L * S_R * (UR[k] - UL[k])) * inv;
  } else {
    for (int k = 0; k < 3; k++) f[k] = FR[k];
  }
}

// ============================================================================
//  HLL 通量 — 通用方向 (2D, CUDA template<coord> kernel 使用)
// ============================================================================
__host__ __device__ inline void hll_flux(const Real WL_h[4], const Real WR_h[4],
                                          Real f[4], int n_idx, int t_idx) {
  Real rhoL = WL_h[0], uL = WL_h[n_idx], vL = WL_h[t_idx], pL = WL_h[3];
  Real rhoR = WR_h[0], uR = WR_h[n_idx], vR = WR_h[t_idx], pR = WR_h[3];

  Real cL = soundSpeed(pL, rhoL);
  Real cR = soundSpeed(pR, rhoR);

#ifdef __CUDA_ARCH__
  Real S_L = fmin(uL - cL, uR - cR);
  Real S_R = fmax(uL + cL, uR + cR);
#else
  Real S_L = std::min(uL - cL, uR - cR);
  Real S_R = std::max(uL + cL, uR + cR);
#endif

  Real EL = pL / (GAMMA - (Real)1.0) + (Real)0.5 * rhoL * (uL*uL + vL*vL);
  Real ER = pR / (GAMMA - (Real)1.0) + (Real)0.5 * rhoR * (uR*uR + vR*vR);

  // 物理通量
  Real FL[4], FR[4], UL[4], UR[4];
  FL[0] = rhoL*uL;        FR[0] = rhoR*uR;
  FL[n_idx] = rhoL*uL*uL + pL; FR[n_idx] = rhoR*uR*uR + pR;
  FL[t_idx] = rhoL*uL*vL;      FR[t_idx] = rhoR*uR*vR;
  FL[3] = (EL+pL)*uL;     FR[3] = (ER+pR)*uR;

  UL[0] = rhoL;       UR[0] = rhoR;
  UL[n_idx] = rhoL*uL; UR[n_idx] = rhoR*uR;
  UL[t_idx] = rhoL*vL; UR[t_idx] = rhoR*vR;
  UL[3] = EL;          UR[3] = ER;

  if (S_L > (Real)0.0) {
    for (int k = 0; k < 4; k++) f[k] = FL[k];
  } else if (S_R > (Real)0.0) {
    Real inv = (Real)1.0 / (S_R - S_L);
    for (int k = 0; k < 4; k++)
      f[k] = (S_R * FL[k] - S_L * FR[k] + S_L * S_R * (UR[k] - UL[k])) * inv;
  } else {
    for (int k = 0; k < 4; k++) f[k] = FR[k];
  }
}

// ============================================================================
//  HLL 通量 — x方向
// ============================================================================
__host__ __device__ inline void hll_flux_x(const Real WL[4], const Real WR[4], Real f[4]) {
  hll_flux(WL, WR, f, 1, 2);
}

// ============================================================================
//  HLL 通量 — y方向
// ============================================================================
__host__ __device__ inline void hll_flux_y(const Real WB[4], const Real WT[4], Real g[4]) {
  hll_flux(WB, WT, g, 2, 1);
}

#endif // RIEMANN_HPP
