# Chapter 9 — Precision Sensitivity and Reproducibility
*(draft of §9.1 and §9.2 — thesis prose, ready to adapt; numbers from `compare_precision.py`)*

---

## 9.1 Experimental Setup

### 9.1.1 Test cases
Two standard ideal-MHD benchmarks span the regimes of interest:

- **Brio–Wu shock tube (1D).** γ = 2, domain x ∈ [0, 1], initial discontinuity at
  x = 0.5, integrated to t = 0.1. Left state (ρ, p, Bₓ, B_y) = (1, 1, 0.75, 1),
  right state (0.125, 0.1, 0.75, −1), zero velocity. A mild test dominated by a
  handful of discontinuities (fast rarefactions, a slow compound wave, a contact
  and a slow shock). Because Bₓ is constant, ∇·B = 0 holds identically and no
  cleaning is required.
- **Orszag–Tang vortex (2D).** γ = 5/3, unit square with periodic boundaries,
  integrated to t = 0.5. The flow develops MHD turbulence — interacting current
  sheets and magnetic reconnection. Divergence errors are controlled with mixed
  hyperbolic–parabolic GLM cleaning (Dedner et al.), adding the scalar ψ field
  (9 conserved variables).

Both cases use the same finite-volume solver: MUSCL–Hancock reconstruction with
the minmod limiter and the HLL or HLLD approximate Riemann solver.

### 9.1.2 Grids
- Brio–Wu: N = 800 cells.
- Orszag–Tang: N × N with N ∈ {128, 192, 256, 384}. The finer grids serve the
  stability study of §9.2.3; the accuracy comparison uses the resolutions at
  which *both* precisions complete.

### 9.1.3 Precision and round-off conventions
The identical source is compiled in two precisions through a single typedef:
double precision (FP64, 53-bit significand, ≈ 15.95 decimal digits) and single
precision (FP32, 24-bit significand, ≈ 7.22 decimal digits). FP64 is the
reference against which FP32 is measured.

All binaries are built with `-O2 -std=c++14` and **without** `-march=native`.
Code generation is therefore restricted to the generic x86-64 baseline, so no
fused multiply–add (FMA) contraction is emitted and the arithmetic is identical,
bit-for-bit, across the two host architectures used in this work (an Ivy-Bridge
and an Ice-Lake Xeon). The solver is serial, so results are additionally
reproducible run-to-run. These properties make the deterministic comparisons of
§9.2–9.3 well defined; controlled random rounding is introduced separately
through MCA in §9.4.

### 9.1.4 Error norms
For a field q sampled at N cells, the difference dᵢ = qᵢ^FP64 − qᵢ^FP32 is
summarised by three discrete norms on the unit domain,

  L1 = (1/N) Σ |dᵢ|,  L2 = √((1/N) Σ dᵢ²),  L∞ = maxᵢ |dᵢ|,

together with the dimensionless relative error relL2 = L2 / RMS(q^FP64). The
three norms probe complementary aspects: L1 the bulk agreement, L2 the typical
(energy-weighted) error, and L∞ the single worst point. Their ratio quantifies
how localised the error is — L∞ is set by isolated sharp features, L1 by the
smooth bulk. The relative error maps onto significant digits: relL2 ≈ 10⁻ˢ
corresponds to ≈ s correct decimal digits, the natural bridge to the
significant-digit analysis of §9.4.

---

## 9.2 FP32 vs FP64

### 9.2.1 Brio–Wu (1D): single precision is sufficient
Table 9.1 gives the FP32-vs-FP64 relative error for every evolved variable at
N = 800. Both Riemann solvers retain **5–6 significant digits** across all
variables, and every configuration runs to completion (760 steps) without a
non-physical state. Single precision is entirely adequate for this mild problem.

**Table 9.1** — Brio–Wu, relL2 (fp32 vs fp64), N = 800.

| variable | HLL | HLLD |
|----------|---------|---------|
| ρ  | 2.2×10⁻⁷ | 9.3×10⁻⁷ |
| vₓ | 1.2×10⁻⁶ | 8.2×10⁻⁶ |
| v_y| 9.4×10⁻⁷ | 6.6×10⁻⁶ |
| p  | 3.1×10⁻⁷ | 2.3×10⁻⁶ |
| B_y| 2.3×10⁻⁷ | 1.5×10⁻⁶ |
| E  | 2.4×10⁻⁷ | 1.4×10⁻⁶ |

Two features stand out. First, **HLL is consistently less precision-sensitive
than HLLD**, by a factor of ≈ 4–10 in relL2 (density: 2.2×10⁻⁷ vs 9.3×10⁻⁷).
HLLD performs substantially more arithmetic — intermediate star states, extra
wave-speed estimates and branch selections — and every additional operation is
an opportunity for the FP32 and FP64 evaluations to diverge. Second, the error is
**highly localised**: for the HLLD velocity, L∞/L1 ≈ 90, so the smooth regions
agree to ≈ 7 digits and essentially all of the discrepancy is carried by the
discontinuities. (Full L1/L2/L∞ norms in Appendix A.)

### 9.2.2 Orszag–Tang (2D): accuracy degrades with resolution
The turbulent 2D case is far more demanding. Table 9.2 shows the FP32 relative
error growing sharply with resolution: for density, from relL2 ≈ 1.5×10⁻⁶
(≈ 6 digits) at 128² to ≈ 1.0×10⁻⁴ (≈ 4 digits) at 192² — a ≈ 60× increase.

**Table 9.2** — Orszag–Tang (HLLD + GLM), relL2 (fp32 vs fp64).

| variable | 128² | 192² |
|----------|---------|---------|
| ρ  | 1.5×10⁻⁶ | 1.0×10⁻⁴ |
| vₓ | 2.0×10⁻⁶ | 4.3×10⁻⁵ |
| v_y| 2.0×10⁻⁶ | 4.2×10⁻⁵ |
| p  | 1.7×10⁻⁶ | 3.6×10⁻⁵ |
| B_y| 1.1×10⁻⁶ | 2.9×10⁻⁵ |
| E  | 1.4×10⁻⁶ | 3.1×10⁻⁵ |

Figure 9.x maps |Δρ|, |Δp| and |ΔB_y| in the plane. The error is not uniform but
**concentrated, over three orders of magnitude, at the current sheets and — most
strongly — at the two magnetic reconnection sites.** The smooth bulk remains
accurate to ≈ 6 digits in FP32; it is the sharp, magnetically dominated
structures that consume precision. The GLM divergence control, by contrast, is
**precision-insensitive**: the ∇·B norms of the FP32 and FP64 runs agree to four
significant figures (mean |∇·B| ≈ 0.068 at 192²), so single precision does not
degrade divergence cleaning.

### 9.2.3 The 256² failure: a stability ceiling
Refining further exposes a hard limit. At 256² the FP32 run terminates at step
2340 with a non-physical state (negative pressure) at a current-sheet cell,
whereas the FP64 run at the same resolution completes (2595 steps); FP64 in turn
fails only at 384². The maximum *stable* resolution is therefore **192² for FP32
and 256² for FP64** with this scheme.

The mechanism is loss of **pressure positivity**. Pressure is recovered as
p = (γ−1)(E − ½ρ|v|² − ½|B|²); at a current sheet the magnetic energy approaches
the total energy, so p is the small difference of two large quantities. As the
grid is refined the sheets sharpen and this cancellation worsens; the reduced
significand of FP32 exhausts its margin one refinement level before FP64 does.
Single precision thus not only loses accuracy but **lowers the maximum stable
resolution**. (The scheme here uses no positivity-preserving limiter; adding one,
or a dual-energy formalism, would raise both ceilings at the cost of altering the
scheme.)

*This localisation of both the FP32 error and the eventual failure at the same
magnetically dominated features is the central observation of the chapter, and
the point of contact with the MHD-vs-Euler comparison of §9.5: Euler pressure
recovery subtracts only kinetic energy, one term rather than two, and has no
∇·B constraint — so it does not exhibit this coupled accuracy/positivity ceiling.*
