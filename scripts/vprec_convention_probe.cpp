#include <cstdio>
int main(){
  volatile double one = 1.0;
  volatile double e52 = 0x1p-52, e23 = 0x1p-23;
  volatile double r52 = one + e52;   // 在 t 位下舍入
  volatile double r23 = one + e23;
  // 后处理运算全是 2 的幂次差/积, 在任意位宽下均精确
  printf("probe52 = %.1f   probe23 = %.1f\n",
         (double)((r52 - 1.0) * 0x1p52), (double)((r23 - 1.0) * 0x1p23));
  return 0;
}
/* 用法 / usage:
 *   /lsc/opt/verificarlo-2.4.0/bin/verificarlo-c++ -O0 scripts/vprec_convention_probe.cpp -o /tmp/probe
 *   for t in 53 52 51 24 23 22; do
 *     VFC_BACKENDS="libinterflop_vprec.so --precision-binary64=$t --mode=ob" /tmp/probe
 *   done
 * 判读: probe_t = 1 表示 1+2^-t 在精度 t 下恰可表示 (t = 显式尾数位数);
 *       t=53 被后端拒绝 (上限 52), 证明 VPREC 的 52 = 完整 binary64。
 *       实测 (Verificarlo 2.4.0, 2026-07-20): t=52->1, t=51->0, t=23->1, t=22->0。
 */
