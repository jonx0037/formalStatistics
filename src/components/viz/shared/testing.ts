/**
 * testing.ts — Track 5 shared utilities for hypothesis testing.
 *
 * The Track-5 counterpart to estimation.ts (Track 4), probability.ts (Track 2),
 * convergence.ts (Track 3). Seeds Topic 17 and will be extended by Topics 18–20
 * (optimality, confidence intervals, multiple testing).
 *
 * Contents
 *   • Re-exports: standard-Normal PDF/CDF/InvCDF under the Track-5 naming
 *   • Student's t and Chi-squared densities, CDFs, inverse CDFs
 *     (via regularized incomplete beta and lower incomplete gamma)
 *   • Test statistics: z, two-sample z, two-proportion z, one-sample t,
 *     pooled t, Welch t, variance chi-squared, Pearson chi-squared
 *   • P-values (analytic) for the same families
 *   • Binomial exact test — p-value, rejection boundary, power
 *   • Size / power utilities, including the textbook sample-size calculator
 *   • Asymptotic trio — Wald, Score (Rao), LRT — for Bernoulli, Normal-mean
 *     (σ known and unknown), and Poisson
 *   • Generic Monte Carlo p-value for arbitrary test statistics
 */

import {
  pdfStdNormal,
  cdfStdNormal,
  quantileStdNormal,
  cdfBinomial,
  pmfBinomial,
  lnGamma,
} from './distributions';

// ── Re-exports under the Track-5 brief's canonical names ────────────────────

/** Standard Normal PDF φ(z). Thin alias for distributions.ts `pdfStdNormal`. */
export const standardNormalPDF = pdfStdNormal;
/** Standard Normal CDF Φ(z). Thin alias for distributions.ts `cdfStdNormal`. */
export const standardNormalCDF = cdfStdNormal;
/** Standard Normal inverse CDF Φ⁻¹(p) (Beasley-Springer-Moro). */
export const standardNormalInvCDF = quantileStdNormal;

// ── Numerical helpers: regularized incomplete gamma and beta ────────────────

/**
 * Regularized lower incomplete gamma P(a, x) = γ(a, x) / Γ(a).
 * Series for x < a + 1, continued fraction (Lentz) for x ≥ a + 1 via Q = 1 − P.
 * Numerical Recipes §6.2.
 */
function regGammaP(a: number, x: number): number {
  if (x <= 0) return 0;
  const logPref = -x + a * Math.log(x) - lnGamma(a);
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let n = 0; n < 500; n++) {
      term *= x / (a + n + 1);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    return sum * Math.exp(logPref);
  }
  // Continued fraction for Q(a, x) via modified Lentz.
  const TINY = 1e-300;
  let b = x + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= 500; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  const Q = h * Math.exp(logPref);
  return 1 - Q;
}

/**
 * Regularized incomplete beta I_x(a, b).
 * Uses the continued-fraction expansion with the symmetry
 *   I_x(a, b) = 1 − I_{1−x}(b, a)
 * to keep x within the faster-converging half-plane. Numerical Recipes §6.4.
 */
function regBetaI(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const logBt =
    lnGamma(a + b) - lnGamma(a) - lnGamma(b) +
    a * Math.log(x) + b * Math.log(1 - x);
  const bt = Math.exp(logBt);
  if (x < (a + 1) / (a + b + 2)) {
    return bt * betaCF(x, a, b) / a;
  }
  return 1 - bt * betaCF(1 - x, b, a) / b;
}

function betaCF(x: number, a: number, b: number): number {
  const MAXIT = 500;
  const EPS = 1e-15;
  const TINY = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

// ── Student's t distribution ────────────────────────────────────────────────

/** Student's t density f(t; ν). */
export function studentTPDF(t: number, df: number): number {
  if (df <= 0) return NaN;
  const logC =
    lnGamma((df + 1) / 2) - lnGamma(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(logC - ((df + 1) / 2) * Math.log(1 + (t * t) / df));
}

/** Student's t CDF F(t; ν) via regularized incomplete beta. */
export function studentTCDF(t: number, df: number): number {
  if (df <= 0) return NaN;
  const x = df / (df + t * t);
  const half = regBetaI(x, df / 2, 0.5) / 2;
  return t >= 0 ? 1 - half : half;
}

/** Student's t inverse CDF. Seeds Newton from standard-Normal quantile plus
 *  a Cornish-Fisher-style tail correction, then polishes. */
export function studentTInvCDF(p: number, df: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;
  let t = quantileStdNormal(p);
  const g1 = (t * t * t + t) / 4;
  const g2 = (5 * Math.pow(t, 5) + 16 * Math.pow(t, 3) + 3 * t) / 96;
  t += g1 / df + g2 / (df * df);
  for (let i = 0; i < 60; i++) {
    const F = studentTCDF(t, df);
    const f = studentTPDF(t, df);
    if (f <= 0) break;
    const dt = (F - p) / f;
    t -= dt;
    if (Math.abs(dt) < 1e-12) break;
  }
  return t;
}

// ── Chi-squared distribution ────────────────────────────────────────────────

/** Chi-squared density f(w; k). */
export function chiSquaredPDF(w: number, df: number): number {
  if (w <= 0 || df <= 0) return 0;
  const k = df / 2;
  return Math.exp((k - 1) * Math.log(w) - w / 2 - k * Math.log(2) - lnGamma(k));
}

/** Chi-squared CDF F(w; k) = P(k/2, w/2). */
export function chiSquaredCDF(w: number, df: number): number {
  if (w <= 0) return 0;
  return regGammaP(df / 2, w / 2);
}

/** Chi-squared inverse CDF via Wilson-Hilferty seed plus Newton polish. */
export function chiSquaredInvCDF(p: number, df: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  const z = quantileStdNormal(p);
  const c = 2 / (9 * df);
  let w = Math.max(df * Math.pow(1 - c + z * Math.sqrt(c), 3), 1e-10);
  for (let i = 0; i < 60; i++) {
    const F = chiSquaredCDF(w, df);
    const f = chiSquaredPDF(w, df);
    if (f <= 0) break;
    const dw = (F - p) / f;
    const next = w - dw;
    w = next > 0 ? next : w / 2;
    if (Math.abs(dw) < 1e-12) break;
  }
  return w;
}

// ── Sample-summary helpers ──────────────────────────────────────────────────

function sampleMean(data: number[]): number {
  let s = 0;
  for (let i = 0; i < data.length; i++) s += data[i];
  return s / data.length;
}

function sampleVarUnbiased(data: number[]): number {
  const n = data.length;
  if (n < 2) return NaN;
  const m = sampleMean(data);
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (data[i] - m) * (data[i] - m);
  return ss / (n - 1);
}

function sampleVarMLE(data: number[]): number {
  const n = data.length;
  if (n === 0) return NaN;
  const m = sampleMean(data);
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (data[i] - m) * (data[i] - m);
  return ss / n;
}

// ── Test statistics ─────────────────────────────────────────────────────────

/** One-sample z-statistic Z = √n (x̄ − μ₀) / σ. Requires σ known. */
export function zTestStatistic(
  data: number[],
  mu0: number,
  sigmaKnown: number,
): number {
  const n = data.length;
  return (Math.sqrt(n) * (sampleMean(data) - mu0)) / sigmaKnown;
}

/** Two-sample z-statistic for means with both σ's known. */
export function twoSampleZStatistic(
  data1: number[],
  data2: number[],
  sigma1: number,
  sigma2: number,
): number {
  const m1 = sampleMean(data1);
  const m2 = sampleMean(data2);
  const se = Math.sqrt(
    (sigma1 * sigma1) / data1.length + (sigma2 * sigma2) / data2.length,
  );
  return (m1 - m2) / se;
}

/** Two-sample proportion z-statistic with pooled variance under H₀:
 *  p̂_pool = (ΣX + ΣY) / (n₁ + n₂). The A/B-testing workhorse. */
export function twoProportionZStatistic(
  successes1: number,
  n1: number,
  successes2: number,
  n2: number,
): number {
  const p1 = successes1 / n1;
  const p2 = successes2 / n2;
  const pPool = (successes1 + successes2) / (n1 + n2);
  const se = Math.sqrt(pPool * (1 - pPool) * (1 / n1 + 1 / n2));
  return (p1 - p2) / se;
}

/** One-sample t-statistic T = √n (x̄ − μ₀) / S using the unbiased S². */
export function tTestStatistic(data: number[], mu0: number): number {
  const n = data.length;
  const s = Math.sqrt(sampleVarUnbiased(data));
  return (Math.sqrt(n) * (sampleMean(data) - mu0)) / s;
}

/** Two-sample pooled t-statistic (equal-variance assumption). */
export function pooledTStatistic(data1: number[], data2: number[]): number {
  const n1 = data1.length;
  const n2 = data2.length;
  const m1 = sampleMean(data1);
  const m2 = sampleMean(data2);
  const s1sq = sampleVarUnbiased(data1);
  const s2sq = sampleVarUnbiased(data2);
  const sp2 = ((n1 - 1) * s1sq + (n2 - 1) * s2sq) / (n1 + n2 - 2);
  return (m1 - m2) / Math.sqrt(sp2 * (1 / n1 + 1 / n2));
}

/** Welch t-statistic (unequal variances) with Satterthwaite degrees of freedom. */
export function welchTStatistic(
  data1: number[],
  data2: number[],
): { t: number; df: number } {
  const n1 = data1.length;
  const n2 = data2.length;
  const m1 = sampleMean(data1);
  const m2 = sampleMean(data2);
  const v1 = sampleVarUnbiased(data1) / n1;
  const v2 = sampleVarUnbiased(data2) / n2;
  const t = (m1 - m2) / Math.sqrt(v1 + v2);
  const df =
    (v1 + v2) * (v1 + v2) /
    ((v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1));
  return { t, df };
}

/** Chi-squared variance statistic W = (n−1) S² / σ₀². */
export function varianceChiSquaredStatistic(
  data: number[],
  sigma0Squared: number,
): number {
  const n = data.length;
  return ((n - 1) * sampleVarUnbiased(data)) / sigma0Squared;
}

/** Pearson chi-squared goodness-of-fit statistic Σ (O − E)² / E. */
export function pearsonChiSquaredStatistic(
  observed: number[],
  expected: number[],
): number {
  if (observed.length !== expected.length) {
    throw new Error('pearsonChiSquaredStatistic: length mismatch');
  }
  let s = 0;
  for (let k = 0; k < observed.length; k++) {
    const e = expected[k];
    if (e <= 0) continue;
    const d = observed[k] - e;
    s += (d * d) / e;
  }
  return s;
}

// ── P-values (analytic) ─────────────────────────────────────────────────────

/** Right-tailed p-value 1 − F(t) for an arbitrary continuous null CDF. */
export function pValueRightTail(
  t: number,
  nullDistCDF: (x: number) => number,
): number {
  return 1 - nullDistCDF(t);
}

/** Two-sided p-value 2 · min(F(t), 1 − F(t)) — symmetric convention. */
export function pValueTwoSided(
  t: number,
  nullDistCDF: (x: number) => number,
): number {
  const F = nullDistCDF(t);
  return 2 * Math.min(F, 1 - F);
}

export type TestSide = 'left' | 'right' | 'two';

/** Z-test p-value. */
export function zTestPValue(z: number, side: TestSide): number {
  const F = cdfStdNormal(z);
  if (side === 'left') return F;
  if (side === 'right') return 1 - F;
  return 2 * Math.min(F, 1 - F);
}

/** T-test p-value. */
export function tTestPValue(t: number, df: number, side: TestSide): number {
  const F = studentTCDF(t, df);
  if (side === 'left') return F;
  if (side === 'right') return 1 - F;
  return 2 * Math.min(F, 1 - F);
}

/** Chi-squared test p-value. Two-sided uses the equal-tailed convention
 *  2 · min(F, 1 − F), matching the variance-test rendering in §17.8's figure
 *  (2.5 % / 97.5 % quantiles on the non-symmetric χ² density). */
export function chiSquaredPValue(
  w: number,
  df: number,
  side: 'right' | 'two',
): number {
  const F = chiSquaredCDF(w, df);
  if (side === 'right') return 1 - F;
  return 2 * Math.min(F, 1 - F);
}

/** Two-sample proportion p-value built on `twoProportionZStatistic`. */
export function twoProportionPValue(
  successes1: number,
  n1: number,
  successes2: number,
  n2: number,
  side: TestSide,
): number {
  const z = twoProportionZStatistic(successes1, n1, successes2, n2);
  return zTestPValue(z, side);
}

// ── Binomial exact test ─────────────────────────────────────────────────────

/**
 * Binomial exact-test p-value for H₀: p = p₀.
 *   right: P_{p₀}(X ≥ xObs)
 *   left:  P_{p₀}(X ≤ xObs)
 *   two:   Σ_{k: P(X=k|p₀) ≤ P(X=xObs|p₀)} P(X=k|p₀)  (point-probability convention)
 *
 * The two-sided convention matches SciPy's `binomtest(..., alternative='two-sided')`
 * method='point-probability' and the default recommended by Agresti (2013).
 * An equal-tail ("symmetric") convention is available in the notebook; this module
 * uses point-probability throughout to stay consistent with §17.6 Example 11.
 */
export function binomialExactPValue(
  xObs: number,
  n: number,
  p0: number,
  side: TestSide,
): number {
  if (side === 'right') {
    if (xObs <= 0) return 1;
    return 1 - cdfBinomial(xObs - 1, n, p0);
  }
  if (side === 'left') return cdfBinomial(xObs, n, p0);
  const pObs = pmfBinomial(xObs, n, p0);
  // Floating-point cushion: include k whose pmf is within 1e-12 of pObs.
  const cushion = pObs * 1e-9 + 1e-12;
  let total = 0;
  for (let k = 0; k <= n; k++) {
    const pk = pmfBinomial(k, n, p0);
    if (pk <= pObs + cushion) total += pk;
  }
  return Math.min(total, 1);
}

/**
 * Binomial exact-test rejection boundary: the smallest (right) or largest (left)
 * integer x such that the rejection region {X ≥ x} (or {X ≤ x}) has exact size
 * no greater than α. Returns {boundary, exactSize}.
 *
 * The exactSize is ≤ α by construction; for discrete nulls it is usually strictly
 * less — the "conservative" size of §17.3 Thm 1 and §17.6 Example 11.
 */
export function binomialExactRejectionBoundary(
  n: number,
  p0: number,
  alpha: number,
  side: 'left' | 'right',
): { boundary: number; exactSize: number } {
  if (side === 'right') {
    for (let x = 0; x <= n; x++) {
      const size = x === 0 ? 1 : 1 - cdfBinomial(x - 1, n, p0);
      if (size <= alpha) return { boundary: x, exactSize: size };
    }
    // No rejection region achieves level α.
    return { boundary: n + 1, exactSize: 0 };
  }
  // Left-tailed: find the largest x with P(X ≤ x) ≤ α.
  let best = { boundary: -1, exactSize: 0 };
  for (let x = 0; x <= n; x++) {
    const size = cdfBinomial(x, n, p0);
    if (size <= alpha) best = { boundary: x, exactSize: size };
    else break;
  }
  return best;
}

/** Binomial exact-test power at a specified true success probability. */
export function binomialExactPower(
  n: number,
  p0: number,
  pTrue: number,
  alpha: number,
  side: 'left' | 'right',
): number {
  const { boundary } = binomialExactRejectionBoundary(n, p0, alpha, side);
  if (side === 'right') {
    if (boundary > n) return 0;
    if (boundary <= 0) return 1;
    return 1 - cdfBinomial(boundary - 1, n, pTrue);
  }
  if (boundary < 0) return 0;
  return cdfBinomial(boundary, n, pTrue);
}

// ── Size and power utilities ────────────────────────────────────────────────

/** MC estimate of Type I error rate. */
export function sizeOfTest(
  sampleUnderH0: () => number[],
  testStatistic: (x: number[]) => number,
  rejectionPredicate: (t: number) => boolean,
  M: number,
): number {
  let rejects = 0;
  for (let i = 0; i < M; i++) {
    if (rejectionPredicate(testStatistic(sampleUnderH0()))) rejects++;
  }
  return rejects / M;
}

/** MC estimate of power at a specific alternative. */
export function powerOfTest(
  sampleUnderHA: () => number[],
  testStatistic: (x: number[]) => number,
  rejectionPredicate: (t: number) => boolean,
  M: number,
): number {
  let rejects = 0;
  for (let i = 0; i < M; i++) {
    if (rejectionPredicate(testStatistic(sampleUnderHA()))) rejects++;
  }
  return rejects / M;
}

/** Closed-form power for the one-sample z-test under a Normal mean alternative.
 *  Formulas — one-sided right: β = 1 − Φ(z_α − δ); two-sided: add the opposite
 *  tail correction; where δ = √n (μ − μ₀) / σ is the standardized effect size. */
export function zTestPower(
  mu: number,
  mu0: number,
  sigma: number,
  n: number,
  alpha: number,
  side: TestSide,
): number {
  const delta = (Math.sqrt(n) * (mu - mu0)) / sigma;
  if (side === 'right') {
    const z = quantileStdNormal(1 - alpha);
    return 1 - cdfStdNormal(z - delta);
  }
  if (side === 'left') {
    const z = quantileStdNormal(alpha);
    return cdfStdNormal(z - delta);
  }
  const zHalf = quantileStdNormal(1 - alpha / 2);
  return 1 - cdfStdNormal(zHalf - delta) + cdfStdNormal(-zHalf - delta);
}

/**
 * Noncentral t CDF F_{T'}(x; ν, δ) via the Johnson-Kotz-Balakrishnan series
 * (Continuous Univariate Distributions, Vol. 2 §31.7). Used by `tTestPower`.
 */
function noncentralTCDF(x: number, nu: number, delta: number): number {
  if (x === 0) return cdfStdNormal(-delta);
  if (x < 0) return 1 - noncentralTCDF(-x, nu, -delta);
  const y = (x * x) / (x * x + nu);
  const half = (delta * delta) / 2;
  let pTerm = Math.exp(-half);
  let qLog =
    Math.log(Math.abs(delta)) - half - 0.5 * Math.log(2) - lnGamma(1.5);
  const sign = delta >= 0 ? 1 : -1;
  let sum = 0;
  for (let j = 0; j < 200; j++) {
    const I1 = regBetaI(y, j + 0.5, nu / 2);
    const I2 = regBetaI(y, j + 1, nu / 2);
    const qTerm = sign * Math.exp(qLog);
    const term = pTerm * I1 + qTerm * I2;
    sum += term;
    if (Math.abs(term) < 1e-16 && j > 2) break;
    pTerm *= half / (j + 1);
    qLog += Math.log(half) - Math.log(j + 1.5);
  }
  return cdfStdNormal(-delta) + 0.5 * sum;
}

/** Closed-form power for the one-sample t-test via the noncentral t CDF. */
export function tTestPower(
  mu: number,
  mu0: number,
  sigma: number,
  n: number,
  alpha: number,
  side: TestSide,
): number {
  const df = n - 1;
  const delta = (Math.sqrt(n) * (mu - mu0)) / sigma;
  if (side === 'right') {
    const tCrit = studentTInvCDF(1 - alpha, df);
    return 1 - noncentralTCDF(tCrit, df, delta);
  }
  if (side === 'left') {
    const tCrit = studentTInvCDF(alpha, df);
    return noncentralTCDF(tCrit, df, delta);
  }
  const tCrit = studentTInvCDF(1 - alpha / 2, df);
  return (
    1 - noncentralTCDF(tCrit, df, delta) + noncentralTCDF(-tCrit, df, delta)
  );
}

/**
 * Closed-form power for the two-sample proportion z-test.
 *
 * The test statistic is the pooled-SE z: Z = (p̂₁ − p̂₂)/SE_pool. The rejection
 * rule uses pooled variance under H₀, but the alternative distribution's
 * variance is unpooled (which is why both appear in the formula).
 *
 * The parameters are ordered `p1, p2` matching the statistic's sign:
 * `delta = p1 − p2`. Callers that frame things as an A/B "lift" should pass
 * the treatment rate as `p1` and the control rate as `p2` — e.g.,
 * `twoProportionPower(p_B, p_A, n_B, n_A, α, 'right')` for a one-sided test
 * of "B beats A". This was ambiguous in the original `pA, pB` signature;
 * renamed after Copilot review #3103512401.
 */
export function twoProportionPower(
  p1: number,
  p2: number,
  n1: number,
  n2: number,
  alpha: number,
  side: TestSide,
): number {
  const pPool = (n1 * p1 + n2 * p2) / (n1 + n2);
  const se0 = Math.sqrt(pPool * (1 - pPool) * (1 / n1 + 1 / n2));
  const seA = Math.sqrt(
    (p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2,
  );
  const delta = p1 - p2;
  if (side === 'right') {
    const z = quantileStdNormal(1 - alpha);
    return 1 - cdfStdNormal((z * se0 - delta) / seA);
  }
  if (side === 'left') {
    const z = quantileStdNormal(alpha);
    return cdfStdNormal((z * se0 - delta) / seA);
  }
  const zHalf = quantileStdNormal(1 - alpha / 2);
  return (
    1 - cdfStdNormal((zHalf * se0 - delta) / seA) +
    cdfStdNormal((-zHalf * se0 - delta) / seA)
  );
}

export type SampleSizeScenario =
  | 'z-one-sample'
  | 't-one-sample'
  | 'z-two-proportion';

export interface SampleSizeParams {
  delta: number;      // Detectable effect (shift for Normal; lift for proportion)
  sigma?: number;     // Required for z-one-sample / t-one-sample
  alpha: number;      // Level
  power: number;      // Target power 1 − β
  baselineP?: number; // Required for z-two-proportion
}

/** Required sample size for target power at a specified effect size.
 *  z-one-sample: closed form n = (z_α + z_β)² σ² / δ².
 *  t-one-sample: numerical search on `tTestPower` (handles small df).
 *  z-two-proportion: closed form using pooled SE under H₀ and unpooled under H_A. */
export function requiredSampleSize(
  scenario: SampleSizeScenario,
  params: SampleSizeParams,
  side: TestSide,
): number {
  const { delta, sigma, alpha, power, baselineP } = params;
  // Guard against zero effect size: any n gives power = α, so no n solves
  // β ≥ power except when target power ≤ α. Return −1 ("no finite answer").
  // Gemini review #3103520850.
  if (delta === 0) return -1;
  const zA =
    side === 'two'
      ? quantileStdNormal(1 - alpha / 2)
      : quantileStdNormal(1 - alpha);
  const zB = quantileStdNormal(power);

  if (scenario === 'z-one-sample') {
    if (sigma === undefined) {
      throw new Error('z-one-sample requires sigma');
    }
    return Math.ceil(Math.pow(((zA + zB) * sigma) / delta, 2));
  }

  if (scenario === 't-one-sample') {
    if (sigma === undefined) {
      throw new Error('t-one-sample requires sigma');
    }
    // Power is monotone-nondecreasing in n, so bisect from the z-based lower
    // bound. O(log n) vs. the previous O(n) linear scan — matters in the
    // interactive sample-size calculator. Gemini review #3103520864.
    let lo = Math.max(4, Math.ceil(Math.pow(((zA + zB) * sigma) / delta, 2)));
    let hi = 100000;
    if (tTestPower(delta, 0, sigma, lo, alpha, side) >= power) return lo;
    if (tTestPower(delta, 0, sigma, hi, alpha, side) < power) return -1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >>> 1;
      if (tTestPower(delta, 0, sigma, mid, alpha, side) >= power) hi = mid;
      else lo = mid;
    }
    return hi;
  }

  if (scenario === 'z-two-proportion') {
    if (baselineP === undefined) {
      throw new Error('z-two-proportion requires baselineP');
    }
    const p1 = baselineP;
    const p2 = baselineP + delta;
    const q1 = 1 - p1;
    const q2 = 1 - p2;
    const pBar = (p1 + p2) / 2;
    const qBar = 1 - pBar;
    const a = zA * Math.sqrt(2 * pBar * qBar);
    const b = zB * Math.sqrt(p1 * q1 + p2 * q2);
    return Math.ceil(Math.pow(a + b, 2) / (delta * delta));
  }

  throw new Error(`Unknown sample-size scenario: ${scenario}`);
}

// ── Asymptotic tests: Wald, Score (Rao), LRT ────────────────────────────────

export type AsymptoticTestFamily =
  | 'bernoulli'
  | 'normal-mean-known-sigma'
  | 'normal-mean'
  | 'poisson';

/** Safe x log(x/y) with the convention 0 log 0 = 0. */
function xLogQuot(x: number, y: number): number {
  if (x <= 0) return 0;
  return x * Math.log(x / y);
}

/** Wald statistic W_n = n (θ̂ − θ₀)² I(θ̂). Asymptotically χ²_1 under H₀. */
export function waldStatistic(
  family: AsymptoticTestFamily,
  data: number[],
  theta0: number,
  knownParam?: number,
): number {
  const n = data.length;
  const mean = sampleMean(data);
  if (family === 'bernoulli') {
    const p = mean;
    const v = p * (1 - p);
    if (v <= 0) return (p === theta0) ? 0 : Infinity;
    return (n * (p - theta0) * (p - theta0)) / v;
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    return (n * (mean - theta0) * (mean - theta0)) / (knownParam * knownParam);
  }
  if (family === 'normal-mean') {
    const s2 = sampleVarUnbiased(data);
    return (n * (mean - theta0) * (mean - theta0)) / s2;
  }
  if (family === 'poisson') {
    if (mean <= 0) return (theta0 === 0) ? 0 : Infinity;
    return (n * (mean - theta0) * (mean - theta0)) / mean;
  }
  throw new Error(`Unknown family: ${family}`);
}

/** Score (Rao) statistic S_n = U(θ₀)² / [n I(θ₀)]. Asymptotically χ²_1 under H₀. */
export function scoreStatistic(
  family: AsymptoticTestFamily,
  data: number[],
  theta0: number,
  knownParam?: number,
): number {
  const n = data.length;
  const mean = sampleMean(data);
  if (family === 'bernoulli') {
    const v = theta0 * (1 - theta0);
    if (v <= 0) return (mean === theta0) ? 0 : Infinity;
    return (n * (mean - theta0) * (mean - theta0)) / v;
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    return (n * (mean - theta0) * (mean - theta0)) / (knownParam * knownParam);
  }
  if (family === 'normal-mean') {
    // Score test is evaluated at θ₀: variance estimator is the null-restricted
    // MLE σ̂²₀ = n⁻¹ Σ(X_i − μ₀)², NOT the unrestricted MLE σ̂² = n⁻¹ Σ(X_i − x̄)².
    // Using σ̂² would require first fitting the full model, which defeats the
    // score test's practical advantage over Wald/LRT. Gemini review #3103520874.
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const d = data[i] - theta0;
      ss += d * d;
    }
    const s2null = ss / n;
    if (s2null <= 0) return (mean === theta0) ? 0 : Infinity;
    return (n * (mean - theta0) * (mean - theta0)) / s2null;
  }
  if (family === 'poisson') {
    if (theta0 <= 0) return Infinity;
    return (n * (mean - theta0) * (mean - theta0)) / theta0;
  }
  throw new Error(`Unknown family: ${family}`);
}

/** Likelihood-ratio statistic −2 log Λ = −2 [ℓ(θ₀) − ℓ(θ̂)]. */
export function lrtStatistic(
  family: AsymptoticTestFamily,
  data: number[],
  theta0: number,
  knownParam?: number,
): number {
  const n = data.length;
  const mean = sampleMean(data);
  if (family === 'bernoulli') {
    const p = mean;
    return 2 * n * (xLogQuot(p, theta0) + xLogQuot(1 - p, 1 - theta0));
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    return (n * (mean - theta0) * (mean - theta0)) / (knownParam * knownParam);
  }
  if (family === 'normal-mean') {
    const s2mle = sampleVarMLE(data);
    return n * Math.log(1 + ((mean - theta0) * (mean - theta0)) / s2mle);
  }
  if (family === 'poisson') {
    return 2 * n * (xLogQuot(mean, theta0) - (mean - theta0));
  }
  throw new Error(`Unknown family: ${family}`);
}

// ── Generic Monte Carlo p-value ─────────────────────────────────────────────

/**
 * Monte Carlo p-value for an arbitrary test statistic when its null
 * distribution is not available in closed form. Not as accurate as the
 * analytic p-value functions — mainly useful for non-standard statistics.
 *
 * The two-sided branch uses the equal-tailed convention
 *   p_two = min(1, 2 · min(P(T ≤ t_obs), P(T ≥ t_obs))),
 * which is correct for BOTH symmetric and non-symmetric nulls (χ², Poisson,
 * binomial). The previous `|t| ≥ |t_obs|` rule implicitly assumed a symmetric
 * null and produced incorrect p-values for e.g. a chi-squared statistic.
 * Gemini review #3103520877.
 */
export function monteCarloPValue(
  sampleUnderH0: () => number[],
  testStatistic: (x: number[]) => number,
  tObs: number,
  M: number,
  side: TestSide,
): number {
  let countLeft = 0;
  let countRight = 0;
  for (let i = 0; i < M; i++) {
    const t = testStatistic(sampleUnderH0());
    if (t <= tObs) countLeft++;
    if (t >= tObs) countRight++;
  }
  if (side === 'right') return countRight / M;
  if (side === 'left') return countLeft / M;
  return Math.min(1, (2 * Math.min(countLeft, countRight)) / M);
}
