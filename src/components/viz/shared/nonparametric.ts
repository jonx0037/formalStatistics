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
 * No imports from other shared modules — the module is self-contained except for
 * two internal helpers (Lanczos log-gamma and binomial PMF).
 */

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
 * Empirical density estimate f̂(ξ_p) at the sample p-quantile, using a
 * simple uniform-kernel bin-count. Silverman rule-of-thumb bandwidth
 * h = 1.06 · IQR · n^{-1/5}. Placeholder for Topic 30's KDE; consumed by
 * `QuantileAsymptoticsExplorer` for the Bahadur-limit variance readout.
 */
export function empiricalDensityAtQuantile(sample: number[], p: number): number {
  const n = sample.length;
  if (n === 0) return 0;
  const sorted = [...sample].sort((a, b) => a - b);
  const quantileOf = (prob: number): number => {
    if (prob === 0) return sorted[0];
    if (prob === 1) return sorted[n - 1];
    const h = (n - 1) * prob;
    const lo = Math.floor(h);
    const frac = h - lo;
    if (lo >= n - 1) return sorted[n - 1];
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  };
  const q = quantileOf(p);
  const iqr = quantileOf(0.75) - quantileOf(0.25);
  const h = 1.06 * iqr * Math.pow(n, -0.2);
  if (h === 0) return 0;
  let count = 0;
  for (const xi of sorted) if (Math.abs(xi - q) <= h) count++;
  return count / (2 * h * n);
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
