/**
 * ============================================================================
 *  base.hpp — 跨模块共享的可移植性样板 (Euler 与 MHD 通用)
 * ============================================================================
 *
 *  只放两样两个 backend 都需要、且与具体物理无关的东西:
 *    1. HD 宏: 让 g++ 编译 .cpp 时忽略 __host__/__device__ 关键词
 *    2. Real 精度类型: -DUSE_FLOAT 切 float, 否则 double
 *
 *  这样 MHD 头文件可以 #include "base.hpp" 拿到 Real/HD,
 *  而不必 #include Euler 专属的 config.hpp (那里还有 GAMMA/NX/domain 等
 *  shock-bubble 物理宏, MHD 不需要)。Euler 的 config.hpp 保持原样不动。
 *
 * ============================================================================
 */

#ifndef BASE_HPP
#define BASE_HPP

// ── CPU/GPU 兼容: 让 g++ 忽略 CUDA 关键词 (nvcc 编 .cu 时自带, 不进此分支) ──
#ifndef __CUDACC__
  #define __host__
  #define __device__
#endif

// ── 精度开关: 编译时加 -DUSE_FLOAT 切 float, 不加默认 double ──
#ifdef USE_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

#endif // BASE_HPP
