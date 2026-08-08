/**
 * ============================================================================
 *  mhd_glm_common.hpp — mixed-GLM extension for cell-centred ideal MHD
 * ============================================================================
 *
 *  This header extends the 8-variable ideal-MHD helpers in mhd_common.hpp with
 *  the Dedner-style mixed GLM cleaning variable psi:
 *
 *    U_glm = (rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz, psi).
 *
 *  The physical MHD flux is still computed by the existing HLL/HLLD routines.
 *  The normal magnetic field and psi are coupled through the linear GLM
 *  subsystem.  The damping source is applied separately as
 *
 *    d psi / dt = - (c_h / alpha) psi,
 *
 *  using the thesis convention c_p^2 / c_h = alpha.
 * ============================================================================
 */

#ifndef MHD_GLM_COMMON_HPP
#define MHD_GLM_COMMON_HPP

#include "mhd_common.hpp"

static const int MHD_GLM_NVAR = 9;
static const int MHD_GLM_PSI = 8;

__host__ __device__ inline void mhdGlmExtractMhdState(const Real U9[MHD_GLM_NVAR],
                                                      Real U8[MHD_NVAR]) {
  for (int k = 0; k < MHD_NVAR; k++) U8[k] = U9[k];
}

__host__ __device__ inline void mhdGlmPrimToCons(const Real W[MHD_NVAR],
                                                 Real psi,
                                                 Real U9[MHD_GLM_NVAR]) {
  Real U8[MHD_NVAR];
  mhdPrimToCons(W, U8);
  for (int k = 0; k < MHD_NVAR; k++) U9[k] = U8[k];
  U9[MHD_GLM_PSI] = psi;
}

__host__ __device__ inline void mhdGlmConsToPrim(const Real U9[MHD_GLM_NVAR],
                                                 Real W[MHD_NVAR],
                                                 Real& psi) {
  Real U8[MHD_NVAR];
  mhdGlmExtractMhdState(U9, U8);
  mhdConsToPrim(U8, W);
  psi = U9[MHD_GLM_PSI];
}

// GLM interface correction for the normal magnetic field and cleaning field.
// This is the exact upwind flux for the linear subsystem:
//   dB_n/dt + dpsi/dn = 0,   dpsi/dt + c_h^2 dB_n/dn = 0.
__host__ __device__ inline void mhdGlmNormalState(Real bnL,
                                                  Real bnR,
                                                  Real psiL,
                                                  Real psiR,
                                                  Real ch,
                                                  Real& bnStar,
                                                  Real& psiStar) {
  Real ch_safe = mhdMax(ch, (Real)1.0e-12);
  bnStar = (Real)0.5 * (bnL + bnR) - (Real)0.5 * (psiR - psiL) / ch_safe;
  psiStar = (Real)0.5 * (psiL + psiR) - (Real)0.5 * ch_safe * (bnR - bnL);
}

// x-normal mixed-GLM flux.  The MHD part uses HLLD or HLL; the Bx/psi pair is
// handled by the GLM subsystem.
__host__ __device__ inline int mhdGlmFluxX(const Real UL[MHD_GLM_NVAR],
                                           const Real UR[MHD_GLM_NVAR],
                                           Real ch,
                                           bool use_hlld,
                                           Real F[MHD_GLM_NVAR]) {
  Real bnStar, psiStar;
  mhdGlmNormalState(UL[MHD_BX], UR[MHD_BX],
                    UL[MHD_GLM_PSI], UR[MHD_GLM_PSI],
                    ch, bnStar, psiStar);

  Real UL8[MHD_NVAR], UR8[MHD_NVAR], F8[MHD_NVAR];
  mhdGlmExtractMhdState(UL, UL8);
  mhdGlmExtractMhdState(UR, UR8);
  UL8[MHD_BX] = bnStar;
  UR8[MHD_BX] = bnStar;

  int status = MHD_HLLD_OK;
  if (use_hlld) {
    status = mhdHlldFluxX(UL8, UR8, F8);
  } else {
    mhdHllFluxX(UL8, UR8, F8);
  }

  for (int k = 0; k < MHD_NVAR; k++) F[k] = F8[k];
  F[MHD_BX] = psiStar;
  F[MHD_GLM_PSI] = ch * ch * bnStar;
  return status;
}

// y-normal mixed-GLM flux.
__host__ __device__ inline int mhdGlmFluxY(const Real UL[MHD_GLM_NVAR],
                                           const Real UR[MHD_GLM_NVAR],
                                           Real ch,
                                           bool use_hlld,
                                           Real G[MHD_GLM_NVAR]) {
  Real bnStar, psiStar;
  mhdGlmNormalState(UL[MHD_BY], UR[MHD_BY],
                    UL[MHD_GLM_PSI], UR[MHD_GLM_PSI],
                    ch, bnStar, psiStar);

  Real UL8[MHD_NVAR], UR8[MHD_NVAR], G8[MHD_NVAR];
  mhdGlmExtractMhdState(UL, UL8);
  mhdGlmExtractMhdState(UR, UR8);
  UL8[MHD_BY] = bnStar;
  UR8[MHD_BY] = bnStar;

  int status = MHD_HLLD_OK;
  if (use_hlld) {
    status = mhdHlldFluxY(UL8, UR8, G8);
  } else {
    mhdHllFluxY(UL8, UR8, G8);
  }

  for (int k = 0; k < MHD_NVAR; k++) G[k] = G8[k];
  G[MHD_BY] = psiStar;
  G[MHD_GLM_PSI] = ch * ch * bnStar;
  return status;
}

#endif // MHD_GLM_COMMON_HPP
