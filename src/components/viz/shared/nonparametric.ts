/**
 * Topic 29 — Order Statistics & Quantiles (Track 8 opener).
 *
 * New shared module for nonparametric machinery. Topics 30 (KDE),
 * 31 (Bootstrap), and 32 (Empirical Processes) extend this file rather
 * than adding sibling modules — the extend-don't-create convention that
 * Tracks 4–6 established within `testing.ts` and Track 7 within `bayes.ts`.
 *
 * Scope (see docs/formalstatistics-order-statistics-and-quantiles-handoff-brief.md
 * §6.1 for the full manifest):
 *
 *   • Order statistics: joint-density formulas, marginal i-th density with log-gamma
 *     stability, the Uniform→Beta specialization, and MC simulation helpers.
 *   • Empirical CDF with O(n log n) sort + O(log n) queries (closure form).
 *   • Sample quantile (Type 7, Hyndman-Fan 1996 default — R/NumPy/SciPy default).
 *   • Kolmogorov-Smirnov statistic (one- and two-sample), the Kolmogorov asymptotic
 *     CDF + inverse + p-value, computed via the primary series for x ≥ 0.3 and the
 *     Jacobi-theta alternate series for x < 0.3 (both converge rapidly in their
 *     respective ranges; see BIL1999 §14).
 *   • DKW inequality with Massart's (1990) tight constant, plus distribution-free
 *     quantile CI via binomial-exact order-statistic pairs.
 *   • Bahadur residual for Monte-Carlo validation of §29.6's featured theorem.
 *   • Rényi-spacings generator: Y_(k) = Σ E_i / (n - i + 1), distributionally
 *     identical to iid-Exp(1) order statistics by Rényi 1953.
 *
 * Imports only `quantileStdNormal` from distributions.ts (Acklam rational
 * approximation) for the §11 pointwise KDE confidence interval. Internal
 * helpers: Lanczos log-gamma, binomial PMF, and two Topic 30 utilities
 * (log-spaced / linear-spaced grids).
 */

import { cdfStdNormal, quantileStdNormal as normalQuantile } from './distributions';
import { type SeededRng } from './bayes';

// ═══════════════════════════════════════════════════════════════════════════
// 1. Order statistics
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Return the i-th order statistic X_{(i)} of a sample (1-indexed).
 * Throws if i is out of range. Sorts a copy of the input (does not mutate).
 * (DAV2003 §2.1)
 */
export function orderStatistic(sample: number[], i: number): number {
  const n = sample.length;
  if (i < 1 || i > n) throw new Error(`orderStatistic: i=${i} out of range [1, ${n}]`);
  const sorted = [...sample].sort((a, b) => a - b);
  return sorted[i - 1];
}

/**
 * Density of X_{(i)} for an iid sample of size n from a distribution with
 * CDF F and density f. Formula:
 *   f_{X_(i)}(x) = n! / ((i-1)!(n-i)!) F(x)^{i-1} (1-F(x))^{n-i} f(x).
 * Uses log-gamma for numerical stability at large n.
 * (DAV2003 §2.2)
 */
export function orderStatisticDensity(
  x: number,
  i: number,
  n: number,
  cdf: (x: number) => number,
  pdf: (x: number) => number
): number {
  const Fx = cdf(x);
  const fx = pdf(x);
  const leftPower = i - 1;
  const rightPower = n - i;
  // Boundary handling: treat 0 * log(0) as 0 (the mass-at-boundary limit). This
  // is what the density actually is there: for i=1 at Fx=0 we should get
  // n * (1 - 0)^{n-1} * f(x), not NaN. Without these special-cases, the
  // Uniform preset's i=1 or i=n endpoints would render as NaN.
  if (Fx <= 0 && leftPower > 0) return 0;
  if (Fx >= 1 && rightPower > 0) return 0;
  if (fx <= 0) return 0;
  const logCoef = lgamma(n + 1) - lgamma(i) - lgamma(n - i + 1);
  const leftLogTerm = leftPower === 0 ? 0 : leftPower * Math.log(Fx);
  const rightLogTerm = rightPower === 0 ? 0 : rightPower * Math.log(1 - Fx);
  return Math.exp(logCoef + leftLogTerm + rightLogTerm + Math.log(fx));
}

/**
 * Density of U_{(i)} for a Uniform(0,1) sample — the Beta(i, n-i+1) density.
 * Convenience wrapper for readability; identical to orderStatisticDensity
 * with F(u) = u, f(u) = 1, on (0,1).
 */
export function uniformOrderStatisticDensity(
  u: number,
  i: number,
  n: number
): number {
  if (u <= 0 || u >= 1) return 0;
  const logCoef = lgamma(n + 1) - lgamma(i) - lgamma(n - i + 1);
  return Math.exp(logCoef + (i - 1) * Math.log(u) + (n - i) * Math.log(1 - u));
}

/**
 * Monte-Carlo simulation of X_{(i)} for a custom sampler. Returns m replicates
 * of X_{(i)} for iid sample of size n.
 */
export function simulateOrderStatistic(
  sampleFn: () => number,
  i: number,
  n: number,
  m: number
): number[] {
  const out: number[] = new Array(m);
  for (let j = 0; j < m; j++) {
    const sample: number[] = new Array(n);
    for (let k = 0; k < n; k++) sample[k] = sampleFn();
    out[j] = orderStatistic(sample, i);
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Empirical CDF
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Empirical CDF F_n(x) = (1/n) Σ_i 1{X_i ≤ x}.
 * O(n) per query. For repeated queries at different x, use `ecdfFn` instead.
 */
export function ecdf(sample: number[], x: number): number {
  if (sample.length === 0) return 0;
  let count = 0;
  for (const xi of sample) if (xi <= x) count++;
  return count / sample.length;
}

/**
 * Returns a function x → F_n(x) with O(log n) per-query cost after O(n log n)
 * one-time sort. Binary-searches the sorted sample.
 */
export function ecdfFn(sample: number[]): (x: number) => number {
  const sorted = [...sample].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return () => 0; // mirror `ecdf` on empty input
  return (x: number): number => {
    let lo = 0;
    let hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo / n;
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Sample quantile (Type 7 — Hyndman-Fan 1996 default)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Sample p-quantile using Hyndman-Fan Type 7 (default in R, NumPy, SciPy).
 *   h = (n - 1) p,  q = X_{(⌊h⌋+1)} + (h - ⌊h⌋)(X_{(⌊h⌋+2)} - X_{(⌊h⌋+1)})
 * Throws for p ∉ [0, 1] or empty sample. This is the only quantile convention
 * exposed by the module — matching the reader's numpy/R environment is the
 * pedagogical priority (brief Appendix B locks Type 7).
 * (HYN1996)
 */
export function sampleQuantile(sample: number[], p: number): number {
  if (p < 0 || p > 1) throw new Error(`sampleQuantile: p=${p} out of [0,1]`);
  const n = sample.length;
  if (n === 0) throw new Error('sampleQuantile: empty sample');
  const sorted = [...sample].sort((a, b) => a - b);
  if (p === 0) return sorted[0];
  if (p === 1) return sorted[n - 1];
  const h = (n - 1) * p;
  const lo = Math.floor(h);
  const frac = h - lo;
  if (lo >= n - 1) return sorted[n - 1];
  return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
}

/**
 * Empirical density estimate f̂(ξ_p) at the sample p-quantile. Delegates to
 * Topic 30's Gaussian KDE (§9) with Silverman's rule-of-thumb bandwidth (§10)
 * evaluated at the Type-7 sample quantile. Consumed by
 * `QuantileAsymptoticsExplorer` for the Bahadur-limit variance readout.
 */
export function empiricalDensityAtQuantile(sample: number[], p: number): number {
  const n = sample.length;
  if (n < 2) return 0;
  const q = sampleQuantile(sample, p);
  const h = silvermanBandwidth(sample);
  if (h <= 0) return 0;
  return kdeEvaluate(sample, q, h, gaussianKernel);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Kolmogorov-Smirnov statistic + distribution
// ═══════════════════════════════════════════════════════════════════════════

/**
 * One-sample Kolmogorov-Smirnov statistic D_n = sup_x |F_n(x) - F(x)|.
 * The sup is attained at one of the sample points (pre- or post-jump), so we
 * evaluate only there:
 *   D_n = max_i max(|i/n - F(X_{(i)})|, |(i-1)/n - F(X_{(i)})|).
 * Correct for continuous F only (matches SciPy convention).
 */
export function ksStatistic(sample: number[], cdf: (x: number) => number): number {
  const sorted = [...sample].sort((a, b) => a - b);
  const n = sorted.length;
  let maxDiff = 0;
  for (let i = 0; i < n; i++) {
    const Fx = cdf(sorted[i]);
    const upper = Math.abs((i + 1) / n - Fx);
    const lower = Math.abs(i / n - Fx);
    if (upper > maxDiff) maxDiff = upper;
    if (lower > maxDiff) maxDiff = lower;
  }
  return maxDiff;
}

/**
 * Two-sample Kolmogorov-Smirnov statistic sup_x |F_n(x) - G_m(x)|.
 * O((n+m) log(n+m)) via merge of sorted samples. Used by `KSGoodnessExplorer`
 * for the two-sample extension sidebar (§29.8 Rem 15).
 */
export function ksTwoSample(sample1: number[], sample2: number[]): number {
  const s1 = [...sample1].sort((a, b) => a - b);
  const s2 = [...sample2].sort((a, b) => a - b);
  const n = s1.length;
  const m = s2.length;
  if (n === 0 || m === 0) return 0;
  let i = 0;
  let j = 0;
  let maxDiff = 0;
  // Walk the union of sample values, advancing past all ties in both samples
  // before measuring the ECDF gap — otherwise e.g. ksTwoSample([1], [1]) returns 1
  // instead of 0 because the two-sample step happens at the same x but the loop
  // only increments one pointer at a time.
  while (i < n || j < m) {
    const x = Math.min(i < n ? s1[i] : Infinity, j < m ? s2[j] : Infinity);
    while (i < n && s1[i] === x) i++;
    while (j < m && s2[j] === x) j++;
    const diff = Math.abs(i / n - j / m);
    if (diff > maxDiff) maxDiff = diff;
  }
  return maxDiff;
}

/**
 * Kolmogorov distribution CDF: K(x) = 1 - 2 Σ_{k≥1} (-1)^{k-1} exp(-2 k² x²).
 * Primary series for x ≥ 0.3; Jacobi-θ alternate for x < 0.3 (BIL1999 §14).
 * Truncates once the next term falls below 1e-15.
 */
export function kolmogorovCDF(x: number): number {
  if (x <= 0) return 0;
  if (x >= 4) return 1;
  if (x >= 0.3) {
    let sum = 0;
    for (let k = 1; k <= 100; k++) {
      const term = (k % 2 === 1 ? 1 : -1) * Math.exp(-2 * k * k * x * x);
      sum += term;
      if (Math.abs(term) < 1e-15) break;
    }
    return 1 - 2 * sum;
  }
  // Jacobi-θ identity for small x:
  //   K(x) = (√(2π)/x) Σ_{k≥0} exp(-(2k+1)² π² / (8 x²)).
  const coef = Math.sqrt(2 * Math.PI) / x;
  let sum = 0;
  for (let k = 0; k <= 100; k++) {
    const term = Math.exp(-((2 * k + 1) ** 2 * Math.PI * Math.PI) / (8 * x * x));
    sum += term;
    if (term < 1e-15) break;
  }
  return coef * sum;
}

/**
 * Inverse of `kolmogorovCDF`: returns x such that K(x) = p. Bisection to 1e-8.
 * Throws for p ∉ (0, 1).
 */
export function kolmogorovQuantile(p: number): number {
  if (p <= 0 || p >= 1)
    throw new Error(`kolmogorovQuantile: p=${p} out of (0,1)`);
  // Start lo at 0 (kolmogorovCDF(0) = 0) so the bracket always contains the
  // root, including for very small p. Cap iterations as a numerical-safety
  // belt; convergence normally lands in ~30 bisections.
  let lo = 0;
  let hi = 5;
  for (let iter = 0; iter < 64; iter++) {
    if (hi - lo <= 1e-8) break;
    const mid = 0.5 * (lo + hi);
    if (kolmogorovCDF(mid) < p) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * Asymptotic KS-test p-value for observed one-sample statistic d at sample size n:
 *   P_{H0}(√n D > √n d) = 1 - K(√n d).
 * For small n (< 40), SciPy's exact Smirnov formula is more accurate — this
 * module only ships the asymptotic form; the browser components use this for
 * all n (§29.8 Rem 16 discusses the small-n crossover).
 */
export function ksPValue(n: number, d: number): number {
  return 1 - kolmogorovCDF(Math.sqrt(n) * d);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. DKW inequality and distribution-free quantile CI
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Dvoretzky-Kiefer-Wolfowitz bound ε_n satisfying
 *   P(sup_x |F_n(x) - F(x)| > ε_n) ≤ α
 * with Massart's (1990) tight constant:
 *   ε_n = √(log(2/α) / (2n)).
 * (DVO1956 / MAS1990)
 */
export function dkwBand(n: number, alpha: number): number {
  if (!Number.isInteger(n) || n < 1)
    throw new Error(`dkwBand: n=${n} must be a positive integer`);
  if (alpha <= 0 || alpha >= 1)
    throw new Error(`dkwBand: alpha=${alpha} out of (0,1)`);
  return Math.sqrt(Math.log(2 / alpha) / (2 * n));
}

/**
 * Distribution-free binomial-exact (1-α)-CI for the population p-quantile ξ_p
 * via an order-statistic pair [X_{(r)}, X_{(s)}]. We pick r, s so that
 *   P(Bin(n, p) < r) ≤ α/2  AND  P(Bin(n, p) ≥ s) ≤ α/2.
 * Returns the chosen indices plus the actual coverage level (generally > 1-α
 * — coverage is a step function of n). Used by §29.7's pedagogy and
 * `QuantileAsymptoticsExplorer`'s CI readout.
 * (SER1980 §2.6.1)
 */
export function quantileCIOrderStatisticBounds(
  n: number,
  p: number,
  alpha: number
): { r: number; s: number; actualLevel: number } {
  // Hoist loop-invariant log-space terms (shared by both cumulative sweeps).
  const logNFactorial = lgamma(n + 1);
  const logP = Math.log(p);
  const log1mP = Math.log(1 - p);
  const pmf = (k: number): number => {
    if (k < 0 || k > n) return 0;
    if (p === 0) return k === 0 ? 1 : 0;
    if (p === 1) return k === n ? 1 : 0;
    const logCoef = logNFactorial - lgamma(k + 1) - lgamma(n - k + 1);
    return Math.exp(logCoef + k * logP + (n - k) * log1mP);
  };
  // Largest r with P(Bin(n,p) < r) ≤ α/2. After iteration k completes,
  // cumLower = P(B ≤ k), and r = k + 1 satisfies P(B < r) = P(B ≤ k) ≤ α/2.
  let r = 0;
  let cumLower = 0;
  for (let k = 0; k <= n; k++) {
    const next = cumLower + pmf(k);
    if (next > alpha / 2) break;
    cumLower = next;
    r = k + 1;
  }
  if (r < 1) r = 1; // degenerate guard (extreme α): fall back to the min.
  if (r > n) r = n;
  // Smallest s with P(Bin(n,p) ≥ s) ≤ α/2.
  let s = n;
  let cumUpper = 0;
  for (let k = n; k >= 0; k--) {
    const next = cumUpper + pmf(k);
    if (next > alpha / 2) break;
    cumUpper = next;
    s = k;
  }
  if (s < 1) s = 1;
  if (r > s) {
    // Degenerate — can happen at extreme p with small n. Return full range.
    return { r: 1, s: n, actualLevel: 1 };
  }
  const actualLevel = 1 - cumLower - cumUpper;
  return { r, s, actualLevel };
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Bahadur residual
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Bahadur residual
 *   R_n = (ξ̂_p - ξ_p) + (F_n(ξ_p) - p) / f(ξ_p).
 * Requires the true quantile ξ_p and the true density f(ξ_p) as inputs —
 * this function is for validating Thm 6 in Monte Carlo, not for production
 * inference. (BAH1966; KIE1967)
 */
export function bahadurResidual(
  sample: number[],
  p: number,
  trueQuantile: number,
  densityAtQuantile: number
): number {
  if (densityAtQuantile <= 0)
    throw new Error('bahadurResidual: density must be positive');
  const xiHat = sampleQuantile(sample, p);
  const Fn = ecdf(sample, trueQuantile);
  return xiHat - trueQuantile + (Fn - p) / densityAtQuantile;
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Rényi spacings
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Generate Y_(k) = Σ_{i=1}^k E_i / (n - i + 1) for k = 1, ..., n using iid Exp(1)
 * inputs drawn from the supplied RNG. By Rényi's representation (§29.3 Thm 3),
 * the returned sequence is distributionally identical to the order statistics
 * of an iid Exp(1) sample of size n. Consumed by `OrderStatisticDensityBrowser`
 * (Exp preset) as a sanity check and by §29.3 Fig 3's schematic.
 */
export function renyiSpacingsSimulate(n: number, rng: () => number): number[] {
  const out: number[] = new Array(n);
  let running = 0;
  for (let k = 1; k <= n; k++) {
    const u = rng();
    // Exp(1) via inverse-CDF: E = -log(1 - U).
    const e = -Math.log(1 - u);
    running += e / (n - k + 1);
    out[k - 1] = running;
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers (not exported)
// ═══════════════════════════════════════════════════════════════════════════

/** Lanczos log-gamma approximation. Accurate to ~1e-15 for x ≥ 0.5. */
function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula for x < 0.5.
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const z = x - 1;
  let a = c[0];
  const t = z + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Binomial PMF P(Bin(n, p) = k) via lgamma. */
function binomialPmf(n: number, p: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (p === 0) return k === 0 ? 1 : 0;
  if (p === 1) return k === n ? 1 : 0;
  const logCoef = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  return Math.exp(logCoef + k * Math.log(p) + (n - k) * Math.log(1 - p));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Kernel functions (Topic 30 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Kernel function signature. A symmetric, nonnegative function integrating to 1.
 * Pointwise evaluation: K(u) for u ∈ ℝ.
 */
export type KernelFn = (u: number) => number;

/** Named kernel identifier. Union of the five kernels §30.3 tabulates. */
export type KernelName =
  | 'gaussian'
  | 'epanechnikov'
  | 'biweight'
  | 'triangular'
  | 'uniform';

/**
 * Kernel properties used in AMISE and bandwidth calculations.
 *   R      = ∫ K(u)² du            (roughness)
 *   mu2    = ∫ u² K(u) du          (second moment)
 *   support = [a, b] where K(u) = 0 outside (Gaussian uses ±Infinity)
 *   efficiency = relative to Epanechnikov (1.0 = optimal)
 * (SIL1986 §3.3.2; WAN1995 §2.7)
 */
export interface KernelSpec {
  name: string;
  fn: KernelFn;
  R: number;
  mu2: number;
  support: [number, number];
  efficiency: number;
}

/**
 * Standard Gaussian kernel K(u) = φ(u) = (1/√(2π)) exp(-u²/2).
 * Infinite support; smooth but non-compact. (ROS1956)
 */
export function gaussianKernel(u: number): number {
  return Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
}

/**
 * Epanechnikov kernel K_E(u) = (3/4)(1 − u²) · 1{|u| ≤ 1}.
 * The MISE-optimal symmetric kernel subject to ∫K = 1 and μ₂(K) = 1/5.
 * Properties: R(K_E) = 3/5, μ₂(K_E) = 1/5. (EPA1969)
 */
export function epanechnikovKernel(u: number): number {
  return Math.abs(u) <= 1 ? 0.75 * (1 - u * u) : 0;
}

/**
 * Biweight (quartic) kernel K(u) = (15/16)(1 − u²)² · 1{|u| ≤ 1}.
 * Smoother than Epanechnikov near the support edges; efficiency ≈ 0.994.
 * Properties: R = 5/7, μ₂ = 1/7. (SIL1986 §3.3)
 */
export function biweightKernel(u: number): number {
  if (Math.abs(u) > 1) return 0;
  const m = 1 - u * u;
  return (15 / 16) * m * m;
}

/**
 * Triangular kernel K(u) = (1 − |u|) · 1{|u| ≤ 1}.
 * Piecewise-linear; efficiency ≈ 0.986. Properties: R = 2/3, μ₂ = 1/6.
 */
export function triangularKernel(u: number): number {
  const abs = Math.abs(u);
  return abs <= 1 ? 1 - abs : 0;
}

/**
 * Uniform (rectangular / boxcar) kernel K(u) = (1/2) · 1{|u| ≤ 1}.
 * The "moving-histogram" limit. Properties: R = 1/2, μ₂ = 1/3. Efficiency ≈ 0.930.
 */
export function uniformKernel(u: number): number {
  return Math.abs(u) <= 1 ? 0.5 : 0;
}

/**
 * Kernel property lookup by name. Returns the full KernelSpec.
 * Throws on unknown name. All constants exact in closed form.
 */
export function kernelProperties(name: KernelName): KernelSpec {
  switch (name) {
    case 'gaussian':
      return {
        name: 'Gaussian',
        fn: gaussianKernel,
        R: 1 / (2 * Math.sqrt(Math.PI)),   // ≈ 0.2820948
        mu2: 1,
        support: [-Infinity, Infinity],
        efficiency: 0.951,
      };
    case 'epanechnikov':
      return {
        name: 'Epanechnikov',
        fn: epanechnikovKernel,
        R: 3 / 5,
        mu2: 1 / 5,
        support: [-1, 1],
        efficiency: 1.0,
      };
    case 'biweight':
      return {
        name: 'Biweight',
        fn: biweightKernel,
        R: 5 / 7,
        mu2: 1 / 7,
        support: [-1, 1],
        efficiency: 0.994,
      };
    case 'triangular':
      return {
        name: 'Triangular',
        fn: triangularKernel,
        R: 2 / 3,
        mu2: 1 / 6,
        support: [-1, 1],
        efficiency: 0.986,
      };
    case 'uniform':
      return {
        name: 'Uniform',
        fn: uniformKernel,
        R: 1 / 2,
        mu2: 1 / 3,
        support: [-1, 1],
        efficiency: 0.930,
      };
    default: {
      // Exhaustiveness check — never reached if KernelName is complete.
      const _exhaustive: never = name;
      throw new Error(`kernelProperties: unknown kernel '${_exhaustive}'`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Kernel density estimator (Topic 30 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Evaluate the KDE at a single point x.
 *   f̂_h(x) = (1 / n h) · Σ K((x − X_i) / h)
 * Throws if h ≤ 0; returns 0 for empty samples. (PAR1962; SIL1986 §3.1)
 */
export function kdeEvaluate(
  sample: number[],
  x: number,
  h: number,
  kernel: KernelFn = gaussianKernel,
): number {
  if (h <= 0) throw new Error(`kdeEvaluate: bandwidth h=${h} must be positive`);
  const n = sample.length;
  if (n === 0) return 0;
  let sum = 0;
  for (const xi of sample) sum += kernel((x - xi) / h);
  return sum / (n * h);
}

/**
 * Evaluate the KDE on a grid. O(n · G) where G = grid.length.
 * Useful for plotting; typical G = 400.
 */
export function kdeEvaluateGrid(
  sample: number[],
  grid: number[],
  h: number,
  kernel: KernelFn = gaussianKernel,
): number[] {
  return grid.map(x => kdeEvaluate(sample, x, h, kernel));
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Bandwidth selectors (Topic 30 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Silverman's rule-of-thumb bandwidth, robust variant:
 *   ĥ_ROT = 1.06 · σ̂ · n^{-1/5},   σ̂ = min(SD, IQR / 1.34).
 * IQR via the Type-7 sample quantile (Topic 29 §29.4). Assumes Gaussian
 * kernel and underlying f ≈ Normal — conservative (over-smooths) for
 * bimodal or heavy-tailed densities. (SIL1986 §3.4.2)
 */
export function silvermanBandwidth(sample: number[]): number {
  const n = sample.length;
  if (n < 2) throw new Error(`silvermanBandwidth: n=${n} too small`);
  // Single-pass sum and sum-of-squares → mean and Bessel-corrected variance.
  let sum = 0;
  let sumSq = 0;
  for (const x of sample) {
    sum += x;
    sumSq += x * x;
  }
  const mean = sum / n;
  const variance = Math.max((sumSq - n * mean * mean) / (n - 1), 0);
  const sd = Math.sqrt(variance);
  // Sort once; read Q1 and Q3 from the sorted array (cheaper than two
  // independent `sampleQuantile` calls, which each sort internally).
  const sorted = [...sample].sort((a, b) => a - b);
  const q = (p: number): number => {
    const h = (n - 1) * p;
    const lo = Math.floor(h);
    const frac = h - lo;
    if (lo >= n - 1) return sorted[n - 1];
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  };
  const iqrScaled = (q(0.75) - q(0.25)) / 1.34;
  const sigmaHat = iqrScaled > 0 ? Math.min(sd, iqrScaled) : sd;
  return 1.06 * sigmaHat * Math.pow(n, -1 / 5);
}

/**
 * Scott's rule-of-thumb bandwidth:
 *   ĥ_Scott = 1.06 · SD · n^{-1/5}.
 * Uses only the sample SD (no IQR robustness). Numerically close to Silverman
 * on well-behaved samples; diverges under heavy tails or bimodality.
 * (SCO2015 §6.5)
 */
export function scottBandwidth(sample: number[]): number {
  const n = sample.length;
  if (n < 2) throw new Error(`scottBandwidth: n=${n} too small`);
  // Single-pass mean + variance (same as silvermanBandwidth).
  let sum = 0;
  let sumSq = 0;
  for (const x of sample) {
    sum += x;
    sumSq += x * x;
  }
  const mean = sum / n;
  const variance = Math.max((sumSq - n * mean * mean) / (n - 1), 0);
  const sd = Math.sqrt(variance);
  return 1.06 * sd * Math.pow(n, -1 / 5);
}

/**
 * Unbiased cross-validation (UCV) bandwidth via grid search. Minimises
 *   UCV(h) = ∫ f̂_h(x)² dx  −  (2 / n) Σ_i f̂_{h, −i}(X_i),
 * where f̂_{h, −i} is the leave-one-out KDE. O(n² · G) per call;
 * **capped at n ≤ 2000** for browser use. Default grid is 30 log-spaced
 * values in [0.1·hS, 2·hS] with hS = Silverman. (SIL1986 §3.4.3)
 */
export function ucvBandwidth(
  sample: number[],
  kernelName: KernelName = 'gaussian',
  hGrid?: number[],
): number {
  const n = sample.length;
  if (n < 10) throw new Error(`ucvBandwidth: n=${n} too small for CV`);
  if (n > 2000) throw new Error(`ucvBandwidth: n=${n} exceeds browser cap of 2000`);
  const spec = kernelProperties(kernelName);
  // Silverman's rule returns 0 when the sample has zero spread (all values
  // equal, or sd = IQR = 0 pathologies). In that case the default log-spaced
  // grid is undefined — require the caller to supply an explicit hGrid, or
  // fail fast with a clear message.
  if (hGrid === undefined) {
    const hS = silvermanBandwidth(sample);
    if (!Number.isFinite(hS) || hS <= 0) {
      throw new Error(
        `ucvBandwidth: Silverman bandwidth ${hS} is not positive — the ` +
          `sample may be degenerate (zero spread). Pass an explicit hGrid.`,
      );
    }
    hGrid = logspace(0.1 * hS, 2 * hS, 30);
  }
  let bestH = hGrid[0];
  let bestUCV = Infinity;
  for (const h of hGrid) {
    const ucv = ucvObjective(sample, h, spec);
    if (ucv < bestUCV) {
      bestUCV = ucv;
      bestH = h;
    }
  }
  return bestH;
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Pointwise CI for f(x) (Topic 30 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Pointwise (1 − α) Normal-calibrated CI for f(x) via plug-in variance.
 *   V̂(x) = f̂_h(x) · R(K) / (n · h),
 *   CI    = f̂_h(x) ± z_{1−α/2} · √V̂(x).
 * Centered at 𝔼[f̂_h(x)], not f(x); at finite n the bias O(h² μ₂(K) f''(x)/2)
 * shifts coverage below 1 − α. For simultaneous CIs use the Bickel–Rosenblatt
 * machinery (deferred to formalml). (vdV2000 §24.1; PAR1962)
 */
export function kdePointwiseCI(
  sample: number[],
  x: number,
  h: number,
  kernelName: KernelName,
  alpha: number = 0.05,
): { estimate: number; se: number; lower: number; upper: number } {
  if (h <= 0) throw new Error(`kdePointwiseCI: bandwidth h=${h} must be positive`);
  const spec = kernelProperties(kernelName);
  const estimate = kdeEvaluate(sample, x, h, spec.fn);
  const n = sample.length;
  const variance = (estimate * spec.R) / (n * h);
  const se = Math.sqrt(Math.max(variance, 0));
  const z = normalQuantile(1 - alpha / 2);
  return { estimate, se, lower: estimate - z * se, upper: estimate + z * se };
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. AMISE-optimal bandwidth (Topic 30 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * AMISE-optimal bandwidth (§30.6 Thm 4):
 *   h* = ( R(K) / [n · μ₂(K)² · R(f'')] )^{1/5}  = O(n^{-1/5}).
 * Requires the curvature integral R(f'') = ∫ f''(x)² dx — caller supplies
 * it either from a closed form (R(φ'') = 3/(8√π) for f = Normal(0,1)) or
 * a numerical Simpson's-rule integration of the known f''. Used by
 * `BandwidthExplorer` to draw the oracle h*. (WAN1995 §2.5)
 */
export function amiseOptimalBandwidth(
  R_f_double_prime: number,
  n: number,
  kernelName: KernelName,
): number {
  if (R_f_double_prime <= 0)
    throw new Error(`amiseOptimalBandwidth: R(f'') = ${R_f_double_prime} must be > 0`);
  if (n < 2) throw new Error(`amiseOptimalBandwidth: n = ${n} too small`);
  const { R, mu2 } = kernelProperties(kernelName);
  return Math.pow(R / (n * mu2 * mu2 * R_f_double_prime), 1 / 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Module-private helpers for §§10–12
// ═══════════════════════════════════════════════════════════════════════════

/**
 * UCV objective. Numerically integrates ∫ f̂_h² on a padded grid and
 * subtracts twice the leave-one-out cross-validation term. O(n² + n·G).
 */
function ucvObjective(sample: number[], h: number, spec: KernelSpec): number {
  const n = sample.length;
  const K = spec.fn;
  let minX = sample[0];
  let maxX = sample[0];
  for (const x of sample) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
  }
  const pad = 3 * h;
  const grid = linspace(minX - pad, maxX + pad, 200);
  const dx = grid[1] - grid[0];
  const fh = kdeEvaluateGrid(sample, grid, h, K);
  const integralFhSq = fh.reduce((s, v) => s + v * v, 0) * dx;
  // Leave-one-out double sum. Since K is symmetric, K(-u) = K(u), so
  //   Σ_i Σ_{j≠i} K((X_i-X_j)/h) = 2 · Σ_{i<j} K((X_i-X_j)/h)
  // and we compute only the upper triangle. Halves kernel evaluations.
  let sumK = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      sumK += K((sample[i] - sample[j]) / h);
    }
  }
  const sumLOO = (2 * sumK) / ((n - 1) * h);
  return integralFhSq - (2 * sumLOO) / n;
}

/** Log-spaced grid on (a, b] with n points. Requires a > 0, b > 0, n >= 2. */
function logspace(a: number, b: number, n: number): number[] {
  if (a <= 0 || b <= 0 || !Number.isFinite(a) || !Number.isFinite(b)) {
    throw new Error(`logspace: endpoints a=${a}, b=${b} must be positive and finite`);
  }
  if (n < 2 || !Number.isInteger(n)) {
    throw new Error(`logspace: n=${n} must be an integer ≥ 2`);
  }
  const la = Math.log(a);
  const lb = Math.log(b);
  const step = (lb - la) / (n - 1);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.exp(la + i * step);
  return out;
}

/** Linear-spaced grid on [a, b] with n points. Requires n >= 2 and a, b finite. */
function linspace(a: number, b: number, n: number): number[] {
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    throw new Error(`linspace: endpoints a=${a}, b=${b} must be finite`);
  }
  if (n < 2 || !Number.isInteger(n)) {
    throw new Error(`linspace: n=${n} must be an integer ≥ 2`);
  }
  const step = (b - a) / (n - 1);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. Bootstrap resamplers (Topic 31 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Draw a single nonparametric bootstrap resample of size n from the data.
 * Each X*_i is selected uniformly with replacement from x via rng.random().
 * Read-only input; returns a fresh array. (EFR1979)
 */
export function bootstrapResample(
  x: readonly number[],
  rng: SeededRng,
): number[] {
  const n = x.length;
  if (n === 0) throw new Error('bootstrapResample: empty sample');
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    // Floor(random * n) is uniform on {0, 1, ..., n-1}. random() returns
    // values in [0, 1), so Math.floor can never equal n.
    out[i] = x[Math.floor(rng.random() * n)];
  }
  return out;
}

/**
 * Draw B bootstrap replicates of a scalar statistic.
 * Returns replicates SORTED ascending — CI constructors assume sorted input.
 * (EFR1979; BIC-FRE1981)
 */
export function nonparametricBootstrap(
  x: readonly number[],
  stat: (sample: readonly number[]) => number,
  B: number,
  rng: SeededRng,
): number[] {
  if (B < 1 || !Number.isInteger(B)) {
    throw new Error(`nonparametricBootstrap: B=${B} must be a positive integer`);
  }
  const reps: number[] = new Array(B);
  for (let b = 0; b < B; b++) {
    reps[b] = stat(bootstrapResample(x, rng));
  }
  reps.sort((a, c) => a - c);
  return reps;
}

/**
 * Parametric bootstrap. `fit` returns a sampler (n, rng) => number[] tied to
 * the fitted parameters — typically an MLE fit. Resample from the fitted
 * distribution rather than from F_n. Returns sorted replicates.
 */
export function parametricBootstrap(
  x: readonly number[],
  fit: (sample: readonly number[]) => (n: number, rng: SeededRng) => number[],
  stat: (sample: readonly number[]) => number,
  B: number,
  rng: SeededRng,
): number[] {
  if (B < 1 || !Number.isInteger(B)) {
    throw new Error(`parametricBootstrap: B=${B} must be a positive integer`);
  }
  const sampler = fit(x);
  const n = x.length;
  const reps: number[] = new Array(B);
  for (let b = 0; b < B; b++) {
    reps[b] = stat(sampler(n, rng));
  }
  reps.sort((a, c) => a - c);
  return reps;
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Kolmogorov-distance helpers (Topic 31 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Kolmogorov distance sup_x |F_a(x) - F_b(x)| between two ECDFs.
 * Inputs need not be sorted; function sorts internally. Right-continuous
 * ECDF convention: F_a(x) = |{a_i ≤ x}| / n. Match to notebook
 * `kolmogorov_distance` (searchsorted on the pooled sort). O((m+n) log(m+n)).
 */
export function kolmogorovDistance(
  a: readonly number[],
  b: readonly number[],
): number {
  const sa = [...a].sort((p, q) => p - q);
  const sb = [...b].sort((p, q) => p - q);
  const n = sa.length;
  const m = sb.length;
  if (n === 0 || m === 0) return 1;
  // Walk through both sorted arrays in pooled order; at each transition, both
  // ECDFs are right-continuous step functions, so evaluate AFTER the step.
  let i = 0;
  let j = 0;
  let maxD = 0;
  while (i < n || j < m) {
    // Advance whichever pointer has the smaller head; break ties by moving both.
    let nextVal: number;
    if (i >= n) nextVal = sb[j];
    else if (j >= m) nextVal = sa[i];
    else nextVal = Math.min(sa[i], sb[j]);
    while (i < n && sa[i] <= nextVal) i++;
    while (j < m && sb[j] <= nextVal) j++;
    const d = Math.abs(i / n - j / m);
    if (d > maxD) maxD = d;
  }
  return maxD;
}

/**
 * KS distance between an empirical sample and an analytic CDF F.
 * Used in Fig 3 featured panel: for the Normal-mean case, H_n is exactly
 * N(0, σ²/n), so we compare against the analytic target rather than an MC
 * reference sample. Two-sided sup: max(D+, D-) where
 *   D+ = max_i [(i+1)/n - F(x_(i))],   D- = max_i [F(x_(i)) - i/n].
 * (BIL1999 §14)
 */
export function kolmogorovDistanceToCdf(
  sample: readonly number[],
  cdf: (x: number) => number,
): number {
  const s = [...sample].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 1;
  let maxD = 0;
  for (let i = 0; i < n; i++) {
    const Fxi = cdf(s[i]);
    const dPlus = (i + 1) / n - Fxi;
    const dMinus = Fxi - i / n;
    if (dPlus > maxD) maxD = dPlus;
    if (dMinus > maxD) maxD = dMinus;
  }
  return maxD;
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Bootstrap CI constructors (Topic 31 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Quantile from a PRE-SORTED array via Type-7 linear interpolation (NumPy /
 * R / SciPy default). Private to §§15–16 — the public `sampleQuantile` in §3
 * sorts defensively; these CI helpers receive already-sorted bootstrap
 * replicates and avoid the re-sort.
 */
function quantileSorted(sorted: readonly number[], p: number): number {
  const n = sorted.length;
  if (n === 0) throw new Error('quantileSorted: empty array');
  if (p <= 0) return sorted[0];
  if (p >= 1) return sorted[n - 1];
  const h = (n - 1) * p;
  const lo = Math.floor(h);
  const frac = h - lo;
  if (lo >= n - 1) return sorted[n - 1];
  return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
}

export interface BootstrapCI {
  readonly lower: number;
  readonly upper: number;
  readonly method: 'percentile' | 'basic' | 'bca' | 'studentized';
}

/**
 * Percentile CI (EFR1979): the α/2 and 1−α/2 empirical quantiles of the
 * bootstrap replicates. First-order accurate — O(n^{-1/2}) coverage error.
 * Expects replicates sorted ascending.
 */
export function percentileCI(
  replicates: readonly number[],
  alpha: number,
): BootstrapCI {
  if (replicates.length < 2) {
    throw new Error(`percentileCI: need ≥ 2 replicates (got ${replicates.length})`);
  }
  if (!(alpha > 0 && alpha < 1)) {
    throw new Error(`percentileCI: alpha=${alpha} must be in (0, 1)`);
  }
  return {
    lower: quantileSorted(replicates, alpha / 2),
    upper: quantileSorted(replicates, 1 - alpha / 2),
    method: 'percentile',
  };
}

/**
 * Basic (Hall) CI: reflects the percentile endpoints around θ̂.
 *   lower = 2 θ̂ − q_{1−α/2},   upper = 2 θ̂ − q_{α/2}.
 * First-order accurate; differs from percentileCI mainly when the bootstrap
 * distribution is skewed. (HAL1992 §3.3)
 */
export function basicCI(
  thetaHat: number,
  replicates: readonly number[],
  alpha: number,
): BootstrapCI {
  if (replicates.length < 2) {
    throw new Error(`basicCI: need ≥ 2 replicates (got ${replicates.length})`);
  }
  if (!(alpha > 0 && alpha < 1)) {
    throw new Error(`basicCI: alpha=${alpha} must be in (0, 1)`);
  }
  const qLo = quantileSorted(replicates, alpha / 2);
  const qHi = quantileSorted(replicates, 1 - alpha / 2);
  return {
    lower: 2 * thetaHat - qHi,
    upper: 2 * thetaHat - qLo,
    method: 'basic',
  };
}

/**
 * BCa (bias-corrected accelerated) CI of EFR1987. Second-order accurate —
 * O(n^{-1}) coverage error (HAL1992 Thm 3.2). Computes:
 *   • z₀ = Φ^{-1}(#{θ*_b < θ̂} / B)  — bias correction
 *   • â  = Σ(θ_{(·)} − θ_{(i)})³ / (6 · [Σ(θ_{(·)} − θ_{(i)})²]^{3/2})
 *          where θ_{(i)} is the jackknife leave-one-out estimate and
 *          θ_{(·)} is the jackknife mean.
 *   • adj_p = Φ(z₀ + (z₀ + z_p) / (1 − â(z₀ + z_p))) for p ∈ {α/2, 1−α/2}
 * Endpoints are the adj_p quantiles of the sorted replicates.
 *
 * Sign convention matches the notebook: num = Σ(jack_mean − jack_i)³.
 * For a right-skewed parent (e.g. Exponential), â > 0.
 *
 * Inner-loop cost: O(n · stat-cost) for the jackknife.
 */
export function bcaCI(
  x: readonly number[],
  stat: (sample: readonly number[]) => number,
  replicates: readonly number[],
  alpha: number,
): BootstrapCI {
  const n = x.length;
  const B = replicates.length;
  if (n < 2) throw new Error(`bcaCI: n=${n} too small`);
  if (B < 2) throw new Error(`bcaCI: B=${B} too small`);
  if (!(alpha > 0 && alpha < 1)) throw new Error(`bcaCI: alpha=${alpha} must be in (0, 1)`);

  const thetaHat = stat(x);

  // z0 from proportion of replicates strictly below thetaHat. Clip away from
  // {0, 1} to avoid ±∞ from Φ^{-1} at the boundary (EFR1993 §14.3 footnote).
  let below = 0;
  for (const r of replicates) if (r < thetaHat) below++;
  let propBelow = below / B;
  const eps = 1 / B;
  if (propBelow < eps) propBelow = eps;
  if (propBelow > 1 - eps) propBelow = 1 - eps;
  const z0 = normalQuantile(propBelow);

  // Jackknife acceleration. Leave-one-out for each i; theta_bar is the
  // jackknife mean.
  const jack: number[] = new Array(n);
  const loo: number[] = new Array(n - 1);
  for (let i = 0; i < n; i++) {
    let k = 0;
    for (let j = 0; j < n; j++) {
      if (j !== i) {
        loo[k++] = x[j];
      }
    }
    jack[i] = stat(loo);
  }
  let jackSum = 0;
  for (const v of jack) jackSum += v;
  const jackMean = jackSum / n;
  let num = 0;
  let den = 0;
  for (const v of jack) {
    const d = jackMean - v;
    num += d * d * d;
    den += d * d;
  }
  const denFull = 6 * Math.pow(den, 1.5);
  const aHat = denFull > 0 ? num / denFull : 0;

  const zLo = normalQuantile(alpha / 2);
  const zHi = normalQuantile(1 - alpha / 2);
  const adjLo = cdfStdNormal(z0 + (z0 + zLo) / (1 - aHat * (z0 + zLo)));
  const adjHi = cdfStdNormal(z0 + (z0 + zHi) / (1 - aHat * (z0 + zHi)));

  return {
    lower: quantileSorted(replicates, adjLo),
    upper: quantileSorted(replicates, adjHi),
    method: 'bca',
  };
}

/**
 * Internal BCa diagnostic — exposes z₀ and â alongside the endpoints. Used by
 * test-pin T31.4 internal sanity check (brief §6.2: z₀ ≈ 0.0221, â ≈ 0.0332).
 * Call signature matches `bcaCI`; returns the same CI plus the two constants.
 */
export function bcaCIWithDiagnostics(
  x: readonly number[],
  stat: (sample: readonly number[]) => number,
  replicates: readonly number[],
  alpha: number,
): BootstrapCI & { z0: number; aHat: number } {
  const n = x.length;
  const B = replicates.length;
  const thetaHat = stat(x);

  let below = 0;
  for (const r of replicates) if (r < thetaHat) below++;
  let propBelow = below / B;
  const eps = 1 / B;
  if (propBelow < eps) propBelow = eps;
  if (propBelow > 1 - eps) propBelow = 1 - eps;
  const z0 = normalQuantile(propBelow);

  const jack: number[] = new Array(n);
  const loo: number[] = new Array(n - 1);
  for (let i = 0; i < n; i++) {
    let k = 0;
    for (let j = 0; j < n; j++) if (j !== i) loo[k++] = x[j];
    jack[i] = stat(loo);
  }
  let jackSum = 0;
  for (const v of jack) jackSum += v;
  const jackMean = jackSum / n;
  let num = 0;
  let den = 0;
  for (const v of jack) {
    const d = jackMean - v;
    num += d * d * d;
    den += d * d;
  }
  const denFull = 6 * Math.pow(den, 1.5);
  const aHat = denFull > 0 ? num / denFull : 0;

  const ci = bcaCI(x, stat, replicates, alpha);
  return { ...ci, z0, aHat };
}

/**
 * Studentized (bootstrap-t) CI. For each outer-bootstrap replicate, runs an
 * INNER bootstrap to estimate the standard error of that replicate's
 * statistic, then builds the studentized pivot
 *   T*_b = (θ̂*_b - θ̂) / SE*_b
 * and returns the α/2 and 1-α/2 quantiles of T*_b scaled by the observed SE.
 * Second-order accurate — O(n^{-1}) coverage error (HAL1992 §3.4).
 *
 * Cost: O(B · BInner · n · stat-cost). Default BInner = 50 in brief.
 */
export function studentizedCI(
  x: readonly number[],
  stat: (sample: readonly number[]) => number,
  se: (sample: readonly number[]) => number,
  B: number,
  BInner: number,
  alpha: number,
  rng: SeededRng,
): BootstrapCI {
  if (B < 1 || !Number.isInteger(B)) {
    throw new Error(`studentizedCI: B=${B} must be a positive integer`);
  }
  if (BInner < 1 || !Number.isInteger(BInner)) {
    throw new Error(`studentizedCI: BInner=${BInner} must be a positive integer`);
  }
  if (!(alpha > 0 && alpha < 1)) {
    throw new Error(`studentizedCI: alpha=${alpha} must be in (0, 1)`);
  }
  const thetaHat = stat(x);
  const seObs = se(x);

  const tStats: number[] = new Array(B);
  for (let b = 0; b < B; b++) {
    const xBoot = bootstrapResample(x, rng);
    const thetaBoot = stat(xBoot);
    const inner: number[] = new Array(BInner);
    for (let bi = 0; bi < BInner; bi++) {
      inner[bi] = stat(bootstrapResample(xBoot, rng));
    }
    // SE of inner replicates (Bessel-corrected).
    let sum = 0;
    for (const v of inner) sum += v;
    const mean = sum / BInner;
    let ss = 0;
    for (const v of inner) {
      const d = v - mean;
      ss += d * d;
    }
    const seBoot = Math.sqrt(ss / (BInner - 1));
    tStats[b] = seBoot > 0 ? (thetaBoot - thetaHat) / seBoot : 0;
  }
  tStats.sort((a, c) => a - c);
  const tLo = quantileSorted(tStats, alpha / 2);
  const tHi = quantileSorted(tStats, 1 - alpha / 2);
  // Note the flip — endpoints use the OPPOSITE t-quantile by the pivot's
  // inversion: if T = (θ̂ - θ) / SE, then θ = θ̂ - T · SE, so large T → small θ.
  return {
    lower: thetaHat - tHi * seObs,
    upper: thetaHat - tLo * seObs,
    method: 'studentized',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Smooth bootstrap + bias correction (Topic 31 extension)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Smooth (kernel-smoothed) bootstrap (SIL-YOU1987). For each resample:
 *   X*_i = X_{J_i} + h · Z_i,   J_i ~ Uniform{1,...,n},  Z_i ~ N(0, 1).
 * Equivalent to resampling from the Gaussian KDE f̂_h. Useful when the
 * statistic of interest depends on local continuity (sample median, quantile
 * regression). Default bandwidth uses the "robust Silverman" / R bw.nrd0
 * variant 0.9 · min(SD, IQR/1.34) · n^{-1/5} — NOT the 1.06 factor used by
 * the public `silvermanBandwidth` in §9. The 0.9 variant undersmooths
 * slightly and is the de-facto default for smooth-bootstrap practice
 * (matches Cell 13's T31.8 pin).
 *
 * Gaussian kernel is hard-coded; if a non-Gaussian kernel is needed, pass an
 * explicit h and a custom sampler-based approach via `bootstrapResample`.
 *
 * Returns sorted replicates.
 */
export function smoothBootstrap(
  x: readonly number[],
  stat: (sample: readonly number[]) => number,
  B: number,
  rng: SeededRng,
  h?: number,
): number[] {
  if (B < 1 || !Number.isInteger(B)) {
    throw new Error(`smoothBootstrap: B=${B} must be a positive integer`);
  }
  const n = x.length;
  if (n < 2) throw new Error(`smoothBootstrap: n=${n} too small`);
  const bandwidth = h ?? smoothBootstrapBandwidth(x);
  if (bandwidth <= 0) {
    throw new Error(`smoothBootstrap: bandwidth h=${bandwidth} must be positive`);
  }
  const reps: number[] = new Array(B);
  const buf: number[] = new Array(n);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(rng.random() * n);
      buf[i] = x[idx] + bandwidth * rng.normal();
    }
    reps[b] = stat(buf);
  }
  reps.sort((a, c) => a - c);
  return reps;
}

/**
 * Robust Silverman bandwidth matching R's bw.nrd0 and the notebook convention:
 *   h = 0.9 · min(SD, IQR/1.34) · n^{-1/5}.
 * Distinct from the public `silvermanBandwidth` in §9, which uses 1.06.
 * Exposed so the smooth-bootstrap demo can display h when the user selects
 * "Silverman (auto)".
 */
export function smoothBootstrapBandwidth(sample: readonly number[]): number {
  const n = sample.length;
  if (n < 2) throw new Error(`smoothBootstrapBandwidth: n=${n} too small`);
  let sum = 0;
  let sumSq = 0;
  for (const v of sample) {
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / n;
  const variance = Math.max((sumSq - n * mean * mean) / (n - 1), 0);
  const sd = Math.sqrt(variance);
  const sorted = [...sample].sort((a, b) => a - b);
  const q = (p: number): number => {
    const h = (n - 1) * p;
    const lo = Math.floor(h);
    const frac = h - lo;
    if (lo >= n - 1) return sorted[n - 1];
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  };
  const iqrScaled = (q(0.75) - q(0.25)) / 1.34;
  const sigmaHat = iqrScaled > 0 ? Math.min(sd, iqrScaled) : sd;
  return 0.9 * sigmaHat * Math.pow(n, -1 / 5);
}

/**
 * Bootstrap bias estimator: mean(replicates) − θ̂. Positive when the
 * bootstrap-world distribution is centered above the observed statistic.
 */
export function bootstrapBias(
  thetaHat: number,
  replicates: readonly number[],
): number {
  if (replicates.length === 0) {
    throw new Error('bootstrapBias: empty replicates');
  }
  let sum = 0;
  for (const r of replicates) sum += r;
  return sum / replicates.length - thetaHat;
}

/**
 * Bias-corrected estimator: θ̃ = 2 θ̂ − mean(replicates). Efron 1990's basic
 * bias correction; removes the first-order bias estimate under the Taylor
 * linearization. More elaborate corrections (jackknife-after-bootstrap,
 * iterated bootstrap) are deferred to §31.10's forward-pointing remarks.
 */
export function biasCorrected(
  thetaHat: number,
  replicates: readonly number[],
): number {
  if (replicates.length === 0) {
    throw new Error('biasCorrected: empty replicates');
  }
  let sum = 0;
  for (const r of replicates) sum += r;
  return 2 * thetaHat - sum / replicates.length;
}
