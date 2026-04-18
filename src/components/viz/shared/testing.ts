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
  cdfPoisson,
  lnGamma,
} from './distributions';
import { seededRandom } from './probability';
import {
  normalSample,
  bernoulliSample,
  poissonSample,
  exponentialSample,
} from './convergence';

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
  | 'poisson'
  | 'exponential';

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
  if (family === 'exponential') {
    // Exponential(λ): ℓ(λ) = n log λ − λ Σx; MLE λ̂ = 1/x̄; I(λ) = 1/λ².
    // W = n (λ̂ − λ₀)² I(λ̂) = n (1 − λ₀ x̄)². Topic 18 Example 12 / §18.7.
    if (mean <= 0) return (theta0 === 0) ? 0 : Infinity;
    const u = theta0 * mean;
    return n * (1 - u) * (1 - u);
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
  if (family === 'exponential') {
    // For Exponential(λ), Wald and Score coincide exactly:
    // S_n = U(λ₀)² / [n I(λ₀)] = n² (1/λ₀ − x̄)² / (n/λ₀²) = n (1 − λ₀ x̄)². See §18.7 Ex 12.
    if (theta0 <= 0) return Infinity;
    const u = theta0 * mean;
    return n * (1 - u) * (1 - u);
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
  if (family === 'exponential') {
    // −2 log Λ = 2n [λ₀ x̄ − 1 − log(λ₀ x̄)]  =  2n · φ(u) with u = λ₀ x̄.
    // Near u = 1, φ(u) ≈ (u−1)²/2, so LRT → Wald = Score at first order. §18.7 Ex 12.
    if (theta0 <= 0 || mean <= 0) return Infinity;
    const u = theta0 * mean;
    return 2 * n * (u - 1 - Math.log(u));
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

// ═══════════════════════════════════════════════════════════════════════════
// TOPIC 18 EXTENSIONS — Neyman-Pearson, Karlin-Rubin, Wilks, non-central χ²
// ═══════════════════════════════════════════════════════════════════════════
//
// These functions extend the Track 5 module with the optimality-theory
// machinery of Topic 18: simple-vs-simple likelihood ratios (§18.2), UMP
// one-sided boundaries via MLR (§18.3–§18.4), Wilks' MC under H₀ (§18.6),
// and the non-central χ² engine for local-power analysis (§18.9).
//
// Architectural note: unlike the Topic 17 `sizeOfTest` / `powerOfTest` design
// (which takes a caller-provided sampler), `wilksSimulate` takes the family
// name directly and generates samples internally — matching the brief's
// locked signature. This imports `normalSample`, `bernoulliSample`,
// `poissonSample`, `exponentialSample` from `./convergence` and `seededRandom`
// from `./probability`, a small cross-module dependency acceptable for the
// topic's MC needs.

// ── Internal: Fisher information by family ──────────────────────────────────

function fisherInformation(
  family: AsymptoticTestFamily,
  theta: number,
  knownParam?: number,
): number {
  if (family === 'bernoulli') {
    if (theta <= 0 || theta >= 1) {
      throw new Error('Bernoulli I(θ) requires θ ∈ (0, 1)');
    }
    return 1 / (theta * (1 - theta));
  }
  if (family === 'normal-mean-known-sigma' || family === 'normal-mean') {
    const sigma = knownParam ?? 1;
    return 1 / (sigma * sigma);
  }
  if (family === 'poisson') {
    if (theta <= 0) throw new Error('Poisson I(λ) requires λ > 0');
    return 1 / theta;
  }
  if (family === 'exponential') {
    if (theta <= 0) throw new Error('Exponential I(λ) requires λ > 0');
    return 1 / (theta * theta);
  }
  throw new Error(`Unknown family: ${family}`);
}

// ── Internal: iid sample generator keyed by family ──────────────────────────

function generateSample(
  family: AsymptoticTestFamily,
  theta: number,
  n: number,
  rng: () => number,
  knownParam?: number,
): number[] {
  const out = new Array<number>(n);
  if (family === 'bernoulli') {
    for (let i = 0; i < n; i++) out[i] = bernoulliSample(theta, rng);
    return out;
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    for (let i = 0; i < n; i++) out[i] = normalSample(theta, knownParam, rng);
    return out;
  }
  if (family === 'normal-mean') {
    // σ unknown at the MDX level, but MC under H₀ needs a concrete scale;
    // the null distribution of −2 log Λ_n is scale-invariant, so any σ > 0 works.
    const sigma = knownParam ?? 1;
    for (let i = 0; i < n; i++) out[i] = normalSample(theta, sigma, rng);
    return out;
  }
  if (family === 'poisson') {
    for (let i = 0; i < n; i++) out[i] = poissonSample(theta, rng);
    return out;
  }
  if (family === 'exponential') {
    for (let i = 0; i < n; i++) out[i] = exponentialSample(theta, rng);
    return out;
  }
  throw new Error(`Unknown family: ${family}`);
}

// ── Neyman-Pearson & simple-vs-simple likelihood ratios ─────────────────────

/**
 * Log likelihood ratio log Λ(x) = Σᵢ [log f(xᵢ; θ₁) − log f(xᵢ; θ₀)] for an
 * iid sample. Returned in log-space for numerical stability (Λ itself can
 * overflow or underflow at modest sample sizes).
 *
 * `normal-mean` (σ unknown) intentionally throws — the simple-vs-simple
 * framing of §18.2 requires both densities fully specified, so σ must be
 * known (use `normal-mean-known-sigma` instead).
 */
export function logLikelihoodRatio(
  family: AsymptoticTestFamily,
  data: number[],
  theta0: number,
  theta1: number,
  knownParam?: number,
): number {
  const n = data.length;
  const mean = sampleMean(data);
  if (family === 'bernoulli') {
    const successes = Math.round(mean * n);
    return (
      successes * Math.log(theta1 / theta0) +
      (n - successes) * Math.log((1 - theta1) / (1 - theta0))
    );
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    // Σᵢ[(xᵢ−θ₀)² − (xᵢ−θ₁)²] = n (θ₁−θ₀)(2 x̄ − θ₀ − θ₁). Derivation in §18.2 Ex 2.
    const num = n * (theta1 - theta0) * (2 * mean - theta0 - theta1);
    return num / (2 * knownParam * knownParam);
  }
  if (family === 'poisson') {
    // log f = −λ + x log λ + const; summed: n(λ₀ − λ₁) + Σx log(λ₁/λ₀).
    return n * (theta0 - theta1) + n * mean * Math.log(theta1 / theta0);
  }
  if (family === 'exponential') {
    // log f = log λ − λ x; summed: n log(λ₁/λ₀) − n (λ₁ − λ₀) x̄.
    return n * Math.log(theta1 / theta0) - n * (theta1 - theta0) * mean;
  }
  if (family === 'normal-mean') {
    throw new Error(
      'logLikelihoodRatio is not defined for normal-mean (σ unknown); ' +
        'use normal-mean-known-sigma for the simple-vs-simple NP setting.',
    );
  }
  throw new Error(`Unknown family: ${family}`);
}

/**
 * Neyman-Pearson critical value: the threshold k such that the size of the
 * NP test {Λ ≥ k} under H₀ equals α. Every one of Topic 18's MLR families
 * admits a sufficient-statistic reformulation: the NP threshold k on Λ
 * is equivalent to a threshold c on T(x) (x̄, ΣXᵢ, etc.). The returned
 * `{threshold, onT, Tform, exactSize}` reports the equivalent T-threshold.
 *
 * `exactSize` equals α for continuous-null families; for discrete nulls
 * (Bernoulli, Poisson) it is ≤ α, matching the conservative size of §18.4.
 */
export function npCriticalValue(
  family: AsymptoticTestFamily,
  theta0: number,
  theta1: number,
  n: number,
  alpha: number,
  knownParam?: number,
): { threshold: number; onT: boolean; Tform: string; exactSize: number } {
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    const side: 'right' | 'left' = theta1 > theta0 ? 'right' : 'left';
    const zA = quantileStdNormal(1 - alpha);
    const halfWidth = (zA * knownParam) / Math.sqrt(n);
    const threshold = side === 'right' ? theta0 + halfWidth : theta0 - halfWidth;
    return { threshold, onT: true, Tform: 'xbar', exactSize: alpha };
  }
  if (family === 'bernoulli') {
    const side: 'right' | 'left' = theta1 > theta0 ? 'right' : 'left';
    const b = binomialExactRejectionBoundary(n, theta0, alpha, side);
    return {
      threshold: b.boundary,
      onT: true,
      Tform: 'ΣXᵢ',
      exactSize: b.exactSize,
    };
  }
  if (family === 'poisson') {
    // Σ Xᵢ ~ Poisson(n θ₀) under H₀.
    const side: 'right' | 'left' = theta1 > theta0 ? 'right' : 'left';
    const lambdaSum = n * theta0;
    const ceiling = Math.ceil(lambdaSum + 10 * Math.sqrt(lambdaSum + 1) + 20);
    if (side === 'right') {
      for (let x = 0; x <= ceiling; x++) {
        const tail = x === 0 ? 1 : 1 - cdfPoisson(x - 1, lambdaSum);
        if (tail <= alpha) {
          return { threshold: x, onT: true, Tform: 'ΣXᵢ', exactSize: tail };
        }
      }
      return { threshold: Infinity, onT: true, Tform: 'ΣXᵢ', exactSize: 0 };
    }
    let best = { threshold: -1, exactSize: 0 };
    for (let x = 0; x <= ceiling; x++) {
      const tail = cdfPoisson(x, lambdaSum);
      if (tail <= alpha) best = { threshold: x, exactSize: tail };
      else break;
    }
    return { threshold: best.threshold, onT: true, Tform: 'ΣXᵢ', exactSize: best.exactSize };
  }
  if (family === 'exponential') {
    // Σ Xᵢ ~ Gamma(n, θ₀): a right-tail threshold on Σ corresponds to rejecting
    // "long total wait", which signals θ < θ₀ (smaller rate = longer waits).
    // Convention: `side` follows the θ₁ direction via MLR monotonicity.
    // For θ₁ > θ₀ (faster rate), LR is decreasing in Σx, so NP rejects {Σx < c_lo}.
    // Use (2 θ₀ · Σx) ~ χ²_{2n} to compute the threshold.
    if (theta1 > theta0) {
      const q = chiSquaredInvCDF(alpha, 2 * n) / 2;
      return { threshold: q / theta0, onT: true, Tform: 'ΣXᵢ (left-tail reject)', exactSize: alpha };
    }
    const q = chiSquaredInvCDF(1 - alpha, 2 * n) / 2;
    return { threshold: q / theta0, onT: true, Tform: 'ΣXᵢ (right-tail reject)', exactSize: alpha };
  }
  if (family === 'normal-mean') {
    throw new Error(
      'npCriticalValue requires known σ; use normal-mean-known-sigma.',
    );
  }
  throw new Error(`Unknown family: ${family}`);
}

// ── Karlin-Rubin / MLR one-sided UMP ────────────────────────────────────────

/**
 * Uniformly most powerful (UMP) one-sided rejection boundary for an MLR
 * family. Returns the threshold on the sufficient statistic T and the
 * achieved exact size. For discrete families the exact size is ≤ α
 * (conservative, per §18.4 Remark 9).
 *
 * `side` refers to the rejection region in T-space ({T > c} = `'right'`).
 * The MLR monotonicity direction is absorbed into the family choice — for
 * Exponential rate with θ₁ > θ₀, the caller passes `side: 'left'`.
 */
export function umpOneSidedBoundary(
  family: AsymptoticTestFamily,
  theta0: number,
  n: number,
  alpha: number,
  side: 'left' | 'right',
  knownParam?: number,
): { boundary: number; Tform: string; exactSize: number } {
  if (family === 'bernoulli') {
    const b = binomialExactRejectionBoundary(n, theta0, alpha, side);
    return { boundary: b.boundary, Tform: 'ΣXᵢ', exactSize: b.exactSize };
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    const zA = quantileStdNormal(1 - alpha);
    const halfWidth = (zA * knownParam) / Math.sqrt(n);
    const boundary = side === 'right' ? theta0 + halfWidth : theta0 - halfWidth;
    return { boundary, Tform: 'xbar', exactSize: alpha };
  }
  if (family === 'poisson') {
    const lambdaSum = n * theta0;
    const ceiling = Math.ceil(lambdaSum + 10 * Math.sqrt(lambdaSum + 1) + 20);
    if (side === 'right') {
      for (let x = 0; x <= ceiling; x++) {
        const tail = x === 0 ? 1 : 1 - cdfPoisson(x - 1, lambdaSum);
        if (tail <= alpha) {
          return { boundary: x, Tform: 'ΣXᵢ', exactSize: tail };
        }
      }
      return { boundary: Infinity, Tform: 'ΣXᵢ', exactSize: 0 };
    }
    let best: { boundary: number; exactSize: number } = {
      boundary: -1,
      exactSize: 0,
    };
    for (let x = 0; x <= ceiling; x++) {
      const tail = cdfPoisson(x, lambdaSum);
      if (tail <= alpha) best = { boundary: x, exactSize: tail };
      else break;
    }
    return { boundary: best.boundary, Tform: 'ΣXᵢ', exactSize: best.exactSize };
  }
  if (family === 'exponential') {
    // Σ Xᵢ ~ Gamma(n, θ₀) ⇔ (2 θ₀ Σ Xᵢ) ~ χ²_{2n}.
    if (side === 'right') {
      const q = chiSquaredInvCDF(1 - alpha, 2 * n) / 2;
      return { boundary: q / theta0, Tform: 'ΣXᵢ', exactSize: alpha };
    }
    const q = chiSquaredInvCDF(alpha, 2 * n) / 2;
    return { boundary: q / theta0, Tform: 'ΣXᵢ', exactSize: alpha };
  }
  if (family === 'normal-mean') {
    throw new Error(
      'umpOneSidedBoundary requires known σ; use normal-mean-known-sigma.',
    );
  }
  throw new Error(`Unknown family: ${family}`);
}

// ── Wilks' theorem: Monte Carlo under H₀ ────────────────────────────────────

/**
 * Monte Carlo simulation of −2 log Λₙ under H₀ for visualizing Wilks'
 * convergence. Reuses the family-specific `lrtStatistic` branches (Topic 17).
 * Seeded for reproducibility; default seed = 42 when omitted.
 *
 * Used by `WilksConvergence` and Example 11 in §18.6 of Topic 18.
 */
export function wilksSimulate(
  family: AsymptoticTestFamily,
  theta0: number,
  n: number,
  M: number,
  knownParam?: number,
  seed?: number,
): number[] {
  const rng = seededRandom(seed ?? 42);
  const out = new Array<number>(M);
  for (let i = 0; i < M; i++) {
    const data = generateSample(family, theta0, n, rng, knownParam);
    out[i] = lrtStatistic(family, data, theta0, knownParam);
  }
  return out;
}

// ── Non-central χ² for local power ──────────────────────────────────────────

/**
 * Sum the Poisson-mixture series
 *   Σⱼ Poisson(j; λ/2) · term(k + 2j)
 * using a peak-first log-space iteration. For small λ we could start at j = 0
 * with w₀ = exp(−λ/2), but that underflows to 0 for λ ≳ 1400. Instead we
 * compute the log-weight at the series peak j* = ⌊λ/2⌋ via lnGamma, iterate
 * outward in both directions via the multiplicative recurrence
 *   w_{j+1}/w_j = (λ/2)/(j+1),     w_{j−1}/w_j = j/(λ/2),
 * and truncate each tail when the weight falls below 1e-14 of the peak.
 * See the Johnson-Kotz-Balakrishnan treatment (Continuous Univariate
 * Distributions, Vol. 2 §29) for the series itself; the peak-first trick
 * is the standard fix for the Gemini review flagged on PR #20.
 */
function poissonMixtureSum(
  k: number,
  lambda: number,
  term: (df: number) => number,
): number {
  const halfLambda = lambda / 2;
  const jStar = Math.floor(halfLambda);
  const logWPeak =
    -halfLambda + jStar * Math.log(halfLambda) - lnGamma(jStar + 1);
  const wPeak = Math.exp(logWPeak);
  const cutoff = 1e-14 * Math.max(wPeak, Number.MIN_VALUE);

  let sum = wPeak * term(k + 2 * jStar);

  // Iterate upward from the peak: w_{j+1} = w_j · (λ/2) / (j+1)
  let w = wPeak;
  for (let j = jStar + 1; j < jStar + 2000; j++) {
    w *= halfLambda / j;
    sum += w * term(k + 2 * j);
    if (w < cutoff) break;
  }

  // Iterate downward from the peak: w_{j−1} = w_j · j / (λ/2)
  w = wPeak;
  for (let j = jStar - 1; j >= 0; j--) {
    w *= (j + 1) / halfLambda;
    sum += w * term(k + 2 * j);
    if (w < cutoff) break;
  }

  return sum;
}

/**
 * Non-central χ²_k(λ) CDF via the Poisson-mixture expansion
 *   F(x; k, λ) = Σⱼ e^{−λ/2} (λ/2)^j / j! · F_{χ²_{k+2j}}(x).
 *
 * Summed via `poissonMixtureSum` (peak-first, log-space). Stable for λ well
 * beyond Topic 18's natural range (local-power h ≤ 4 gives λ ≤ 64) —
 * validated against `scipy.stats.ncx2.cdf` for λ ∈ [0, 1000].
 */
export function nonCentralChiSquaredCDF(
  x: number,
  k: number,
  lambda: number,
): number {
  if (x <= 0) return 0;
  if (lambda < 0) {
    throw new Error('nonCentralChiSquaredCDF requires λ ≥ 0');
  }
  if (lambda === 0) return chiSquaredCDF(x, k);
  return poissonMixtureSum(k, lambda, (df) => chiSquaredCDF(x, df));
}

/**
 * Non-central χ²_k(λ) density via the same Poisson-mixture expansion
 *   f(x; k, λ) = Σⱼ e^{−λ/2} (λ/2)^j / j! · f_{χ²_{k+2j}}(x).
 */
export function nonCentralChiSquaredPDF(
  x: number,
  k: number,
  lambda: number,
): number {
  if (x <= 0) return 0;
  if (lambda < 0) {
    throw new Error('nonCentralChiSquaredPDF requires λ ≥ 0');
  }
  if (lambda === 0) return chiSquaredPDF(x, k);
  return poissonMixtureSum(k, lambda, (df) => chiSquaredPDF(x, df));
}

// ── Local-power envelope ────────────────────────────────────────────────────

/**
 * Local power at a √n-scaled alternative θₙ = θ₀ + h/√n under the asymptotic
 * χ²_1(h² I(θ₀)) law (§18.9 Thm 7). This is the power of any of the three
 * asymptotic tests (Wald, Score, LRT) at the local alternative and level α,
 * ignoring the first-order divergence of §18.7.
 *
 *   local-power = 1 − F_{ncχ²_1}(χ²_{1, 1−α}; h² I(θ₀)).
 *
 * When h = 0 the local-power equals α (the test's size under H₀), a sanity
 * check that the returned value honors at the numerical tolerance of
 * `nonCentralChiSquaredCDF`.
 */
export function localPower(
  family: AsymptoticTestFamily,
  theta0: number,
  h: number,
  alpha: number,
  knownParam?: number,
): number {
  const info = fisherInformation(family, theta0, knownParam);
  const nonCentrality = h * h * info;
  const crit = chiSquaredInvCDF(1 - alpha, 1);
  return 1 - nonCentralChiSquaredCDF(crit, 1, nonCentrality);
}

// ═══════════════════════════════════════════════════════════════════════════
// TOPIC 19 EXTENSIONS — Confidence Intervals & Duality
// ═══════════════════════════════════════════════════════════════════════════
//
// CI construction layer: pivotal (z, t, χ², F), asymptotic (Wald, Score, LRT),
// binomial-specific (Wilson, Agresti-Coull, Clopper-Pearson), profile
// likelihood, TOST equivalence, and a coverage simulator. Every CI function
// either exploits test-CI duality directly — inverting one of the Topic 17–18
// test statistics — or constructs a pivotal quantity in closed form.

// ── F distribution ──────────────────────────────────────────────────────────

/** Snedecor's F density f(x; d1, d2). */
export function fPDF(x: number, df1: number, df2: number): number {
  if (x <= 0 || df1 <= 0 || df2 <= 0) return 0;
  const a = df1 / 2;
  const b = df2 / 2;
  const logC = lnGamma(a + b) - lnGamma(a) - lnGamma(b)
    + a * Math.log(df1) - a * Math.log(df2);
  return Math.exp(logC + (a - 1) * Math.log(x) - (a + b) * Math.log(1 + (df1 * x) / df2));
}

/** Snedecor's F CDF via I_{d1 x/(d1 x + d2)}(d1/2, d2/2). */
export function fCDF(x: number, df1: number, df2: number): number {
  if (x <= 0) return 0;
  const z = (df1 * x) / (df1 * x + df2);
  return regBetaI(z, df1 / 2, df2 / 2);
}

/**
 * F inverse CDF via bisection on fCDF. The F range is [0, ∞); we bracket by
 * doubling until fCDF exceeds p, then bisect. Tolerance 1e-10.
 */
export function fInvCDF(p: number, df1: number, df2: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  // Bracket: start at the mode area, double upward.
  let lo = 0;
  let hi = 2;
  while (fCDF(hi, df1, df2) < p) {
    lo = hi;
    hi *= 2;
    if (hi > 1e12) return hi;
  }
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (fCDF(mid, df1, df2) < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12 * Math.max(1, Math.abs(hi))) break;
  }
  return 0.5 * (lo + hi);
}

// ── Beta inverse CDF (powers Clopper–Pearson) ──────────────────────────────

/**
 * Inverse of regularized incomplete beta: returns x ∈ [0, 1] such that
 * I_x(a, b) = p. Uses bisection for numerical robustness — the Beta CDF is
 * monotone and well-behaved on (0, 1), and bisection converges to 1e-12 in
 * 50 iterations without the failure modes Newton can hit near the endpoints.
 *
 * Boundary behavior: returns 0 for p ≤ 0 and 1 for p ≥ 1. This is exact for
 * the degenerate cases and matches scipy's `beta.ppf` at the tolerance.
 */
export function betaInvCDF(p: number, a: number, b: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (regBetaI(mid, a, b) < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12) break;
  }
  return 0.5 * (lo + hi);
}

// ── Shared CI types ─────────────────────────────────────────────────────────

/** Standard CI result shape, used for all CI procedures. */
export interface ConfidenceInterval {
  lower: number;
  upper: number;
  alpha: number;
  procedure: string;
}

/** Minimal CI endpoint pair — the return type most functions use. */
export interface CIEndpoints {
  lower: number;
  upper: number;
}

// ── Pivotal confidence intervals (§19.3, exact small-sample) ────────────────

/** z-CI for Normal mean, σ known: x̄ ± z_{α/2} σ / √n. */
export function zCINormalMean(
  data: number[],
  sigmaKnown: number,
  alpha: number,
): CIEndpoints {
  const n = data.length;
  const mean = sampleMean(data);
  const z = standardNormalInvCDF(1 - alpha / 2);
  const half = (z * sigmaKnown) / Math.sqrt(n);
  return { lower: mean - half, upper: mean + half };
}

/** t-CI for Normal mean, σ unknown: x̄ ± t_{n-1, α/2} S / √n. */
export function tCINormalMean(data: number[], alpha: number): CIEndpoints {
  const n = data.length;
  const mean = sampleMean(data);
  const s = Math.sqrt(sampleVarUnbiased(data));
  const tq = studentTInvCDF(1 - alpha / 2, n - 1);
  const half = (tq * s) / Math.sqrt(n);
  return { lower: mean - half, upper: mean + half };
}

/**
 * χ²-CI for Normal variance σ² (μ unknown):
 *   [(n-1) S² / χ²_{n-1, 1-α/2}, (n-1) S² / χ²_{n-1, α/2}].
 * The pivot is W = (n-1) S² / σ² ∼ χ²_{n-1}; inverting the event
 * {χ²_{α/2} ≤ W ≤ χ²_{1-α/2}} gives the stated bounds.
 */
export function chiSquaredCINormalVariance(
  data: number[],
  alpha: number,
): CIEndpoints {
  const n = data.length;
  const s2 = sampleVarUnbiased(data);
  const qLo = chiSquaredInvCDF(alpha / 2, n - 1);
  const qHi = chiSquaredInvCDF(1 - alpha / 2, n - 1);
  return { lower: ((n - 1) * s2) / qHi, upper: ((n - 1) * s2) / qLo };
}

/**
 * F-CI for ratio of Normal variances σ²₁ / σ²₂ (two independent samples).
 * Pivot F = (S²₁/σ²₁) / (S²₂/σ²₂) ∼ F_{n1-1, n2-1}; inverting gives
 *   [(S²₁/S²₂) / F_{α/2 upper}, (S²₁/S²₂) · F_{α/2 upper reversed df}]
 * — stated here with the symmetric quantile convention F_{d1,d2,α/2} being
 * the upper-α/2 quantile.
 */
export function fCIVarianceRatio(
  data1: number[],
  data2: number[],
  alpha: number,
): CIEndpoints {
  const n1 = data1.length;
  const n2 = data2.length;
  const s1 = sampleVarUnbiased(data1);
  const s2 = sampleVarUnbiased(data2);
  const ratio = s1 / s2;
  const fLo = fInvCDF(alpha / 2, n1 - 1, n2 - 1);
  const fHi = fInvCDF(1 - alpha / 2, n1 - 1, n2 - 1);
  return { lower: ratio / fHi, upper: ratio / fLo };
}

// ── Asymptotic CIs: Wald, Score, LRT (§19.4) ────────────────────────────────
//
// All three are test-inversion CIs. Wald inverts the Wald statistic (evaluated
// at θ̂), Score inverts the Score statistic (evaluated at θ₀), and LRT inverts
// −2 log Λₙ(θ₀). For simple families all three have closed-form or tractable
// rootfind; complicated cases fall back to bracketed bisection.

/** Clamp a scalar to the parameter space of a family. */
function clampToParameterSpace(
  family: AsymptoticTestFamily | 'normal-variance',
  value: number,
): number {
  if (family === 'bernoulli') return Math.max(0, Math.min(1, value));
  if (family === 'poisson' || family === 'exponential') return Math.max(0, value);
  if (family === 'normal-variance') return Math.max(0, value);
  return value;
}

/**
 * Wald CI: θ̂ ± z_{α/2} / √(n I(θ̂)). Closed form for every supported family.
 *
 * Notes:
 *  - `bernoulli`: θ̂ ± z √(p̂(1-p̂)/n) — the symmetric interval that motivates
 *    Wilson's boundary fix (§19.5).
 *  - `normal-mean-known-sigma`: coincides with the z-CI of §19.3.
 *  - `normal-mean`: x̄ ± z S/√n — the asymptotic analog of the t-CI.
 *  - `normal-variance` (μ estimated by x̄): σ̂² ± z σ̂² √(2/n) using the
 *    variance-MLE standard error.
 *  - `poisson`: λ̂ ± z √(λ̂ / n).
 *  - `exponential`: λ̂ ± z λ̂ / √n, where λ̂ = 1 / x̄.
 *
 * Results are clamped to the parameter space; callers needing the raw
 * unclamped endpoints can compute them directly from the formulas above.
 */
export function waldCI(
  family: AsymptoticTestFamily | 'normal-variance',
  data: number[],
  alpha: number,
  knownParam?: number,
): CIEndpoints {
  const n = data.length;
  const mean = sampleMean(data);
  const z = standardNormalInvCDF(1 - alpha / 2);
  if (family === 'bernoulli') {
    const p = mean;
    const se = Math.sqrt((p * (1 - p)) / n);
    return {
      lower: clampToParameterSpace('bernoulli', p - z * se),
      upper: clampToParameterSpace('bernoulli', p + z * se),
    };
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    const half = (z * knownParam) / Math.sqrt(n);
    return { lower: mean - half, upper: mean + half };
  }
  if (family === 'normal-mean') {
    const s = Math.sqrt(sampleVarUnbiased(data));
    const half = (z * s) / Math.sqrt(n);
    return { lower: mean - half, upper: mean + half };
  }
  if (family === 'normal-variance') {
    const s2 = sampleVarMLE(data);
    const half = z * s2 * Math.sqrt(2 / n);
    return {
      lower: clampToParameterSpace('normal-variance', s2 - half),
      upper: s2 + half,
    };
  }
  if (family === 'poisson') {
    const se = Math.sqrt(mean / n);
    return {
      lower: clampToParameterSpace('poisson', mean - z * se),
      upper: mean + z * se,
    };
  }
  if (family === 'exponential') {
    if (mean <= 0) throw new Error('exponential Wald CI needs positive sample mean');
    const lamHat = 1 / mean;
    const se = lamHat / Math.sqrt(n);
    return {
      lower: clampToParameterSpace('exponential', lamHat - z * se),
      upper: lamHat + z * se,
    };
  }
  throw new Error(`Unknown family for waldCI: ${family}`);
}

/**
 * Score CI: the set {θ₀ : S_n(θ₀) ≤ z²_{α/2}}, i.e. the inversion of the
 * score (Rao) test. Closed form for Bernoulli (returns Wilson) and
 * Poisson (quadratic in θ₀); numerical bisection elsewhere.
 */
export function scoreCI(
  family: AsymptoticTestFamily,
  data: number[],
  alpha: number,
  knownParam?: number,
): CIEndpoints {
  const n = data.length;
  const mean = sampleMean(data);
  const z = standardNormalInvCDF(1 - alpha / 2);
  const z2 = z * z;
  const crit = z2; // χ²_{1, 1-α}; since S_n is χ²_1-distributed, rejection iff S_n > z².

  if (family === 'bernoulli') {
    // Wilson form — see wilsonInterval; repeat here rather than recurse so
    // the function can be imported standalone.
    const p = mean;
    const denom = 1 + z2 / n;
    const center = (p + z2 / (2 * n)) / denom;
    const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
    return {
      lower: Math.max(0, center - half),
      upper: Math.min(1, center + half),
    };
  }
  if (family === 'normal-mean-known-sigma') {
    if (knownParam === undefined) {
      throw new Error('normal-mean-known-sigma requires knownParam (σ)');
    }
    const half = (z * knownParam) / Math.sqrt(n);
    return { lower: mean - half, upper: mean + half };
  }
  if (family === 'poisson') {
    // S_n(λ) = n(x̄ − λ)² / λ ≤ z²  ⇔  λ² − (2x̄ + z²/n)λ + x̄² ≤ 0.
    // Roots of A λ² + B λ + C = 0 with A=1, B=−(2x̄ + z²/n), C=x̄².
    const B = -(2 * mean + z2 / n);
    const disc = B * B - 4 * mean * mean;
    if (disc < 0) return { lower: Math.max(0, mean), upper: mean };
    const rt = Math.sqrt(disc);
    return {
      lower: clampToParameterSpace('poisson', (-B - rt) / 2),
      upper: (-B + rt) / 2,
    };
  }
  // Numerical fallback: bracket and bisect on scoreStatistic.
  return inverseStatisticCI(family, data, knownParam, crit, scoreStatistic);
}

/**
 * LRT CI: the set {θ₀ : −2 log Λ_n(θ₀) ≤ χ²_{1, 1-α}}. Obtained by bracketed
 * bisection on the LRT statistic as a function of θ₀; asymmetric around θ̂,
 * honoring the log-likelihood curvature that Wald's quadratic approximation
 * erases.
 */
export function lrtCI(
  family: AsymptoticTestFamily,
  data: number[],
  alpha: number,
  knownParam?: number,
): CIEndpoints {
  const crit = chiSquaredInvCDF(1 - alpha, 1);
  return inverseStatisticCI(family, data, knownParam, crit, lrtStatistic);
}

/**
 * Generic test-inversion CI: given a statistic T(θ₀) that is 0 at θ̂ and
 * increases monotonically as θ₀ moves away from θ̂, find the two θ₀ values
 * where T(θ₀) = critical value. Bracket on each side of θ̂, then bisect.
 */
function inverseStatisticCI(
  family: AsymptoticTestFamily,
  data: number[],
  knownParam: number | undefined,
  critical: number,
  statistic: (
    fam: AsymptoticTestFamily,
    d: number[],
    theta0: number,
    k?: number,
  ) => number,
): CIEndpoints {
  const n = data.length;
  const mean = sampleMean(data);
  const thetaHat = familyMLE(family, data, knownParam);

  const f = (theta: number): number =>
    statistic(family, data, theta, knownParam) - critical;

  // Bracket left: search from θ̂ downward.
  const lowerLimit = familyLowerBound(family);
  let loLeft = thetaHat;
  let hiLeft = thetaHat;
  let stepLeft = Math.max(1e-4, Math.abs(thetaHat) * 0.01,
                          Math.sqrt(sampleVarUnbiased(data) / n) * 0.5);
  for (let i = 0; i < 80; i++) {
    loLeft = Math.max(lowerLimit + 1e-12, hiLeft - stepLeft);
    if (f(loLeft) > 0) break;
    hiLeft = loLeft;
    stepLeft *= 1.6;
    if (loLeft === lowerLimit + 1e-12) break;
  }

  // Bracket right: search from θ̂ upward.
  const upperLimit = familyUpperBound(family);
  let loRight = thetaHat;
  let hiRight = thetaHat;
  let stepRight = Math.max(1e-4, Math.abs(thetaHat) * 0.01,
                           Math.sqrt(sampleVarUnbiased(data) / n) * 0.5);
  for (let i = 0; i < 80; i++) {
    hiRight = Math.min(upperLimit - 1e-12, loRight + stepRight);
    if (f(hiRight) > 0) break;
    loRight = hiRight;
    stepRight *= 1.6;
    if (hiRight === upperLimit - 1e-12) break;
  }

  // `mean` and `n` are referenced above in the bracket-step initializers;
  // kept in scope via the enclosing function closure but otherwise unused here.
  void mean;
  void n;

  const lower = bisectRoot(f, loLeft, thetaHat, 60);
  const upper = bisectRoot(f, thetaHat, hiRight, 60);
  return {
    lower: clampToParameterSpace(family, lower),
    upper: clampToParameterSpace(family, upper),
  };
}

function familyMLE(
  family: AsymptoticTestFamily,
  data: number[],
  knownParam?: number,
): number {
  void knownParam;
  const mean = sampleMean(data);
  if (family === 'bernoulli') return mean;
  if (family === 'normal-mean' || family === 'normal-mean-known-sigma') return mean;
  if (family === 'poisson') return mean;
  if (family === 'exponential') return 1 / mean;
  throw new Error(`No MLE for family ${family}`);
}

function familyLowerBound(family: AsymptoticTestFamily): number {
  if (family === 'bernoulli') return 0;
  if (family === 'poisson' || family === 'exponential') return 0;
  return -Infinity;
}

function familyUpperBound(family: AsymptoticTestFamily): number {
  if (family === 'bernoulli') return 1;
  return Infinity;
}

function bisectRoot(
  f: (x: number) => number,
  lo: number,
  hi: number,
  iters: number,
): number {
  let a = lo;
  let b = hi;
  const fa = f(a);
  const fb = f(b);
  if (fa * fb > 0) {
    // Endpoints have same sign — no sign change. Return the endpoint with
    // smaller |f| as a graceful degradation.
    return Math.abs(fa) < Math.abs(fb) ? a : b;
  }
  for (let i = 0; i < iters; i++) {
    const mid = 0.5 * (a + b);
    const fm = f(mid);
    if (fm === 0 || b - a < 1e-12 * Math.max(1, Math.abs(mid))) return mid;
    if (fa * fm < 0) b = mid;
    else a = mid;
  }
  return 0.5 * (a + b);
}

// ── Binomial-specific CIs: Wilson, Agresti-Coull, Clopper-Pearson (§19.5–6) ─

/**
 * Wilson interval for binomial p — the closed-form score-test inversion.
 * Respects the parameter space [0, 1] by construction; at the boundary
 * p̂ = 0 returns [0, z²/(n + z²)], at p̂ = 1 returns [n/(n + z²), 1]
 * (both endpoints strictly interior).
 */
export function wilsonInterval(x: number, n: number, alpha: number): CIEndpoints {
  const z = standardNormalInvCDF(1 - alpha / 2);
  const z2 = z * z;
  const pHat = x / n;
  const denom = 1 + z2 / n;
  const center = (pHat + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt((pHat * (1 - pHat)) / n + z2 / (4 * n * n))) / denom;
  return {
    lower: Math.max(0, center - half),
    upper: Math.min(1, center + half),
  };
}

/**
 * Agresti-Coull "plus-4" approximation to Wilson (AGR1998). Adds z²/2 ≈ 2
 * successes and z²/2 failures to the observed counts, then applies Wald.
 * Usually matches Wilson to within 0.01 and is easier to remember.
 */
export function agrestiCoullInterval(
  x: number,
  n: number,
  alpha: number,
): CIEndpoints {
  const z = standardNormalInvCDF(1 - alpha / 2);
  const z2 = z * z;
  const nTilde = n + z2;
  const pTilde = (x + z2 / 2) / nTilde;
  const half = z * Math.sqrt((pTilde * (1 - pTilde)) / nTilde);
  return {
    lower: Math.max(0, pTilde - half),
    upper: Math.min(1, pTilde + half),
  };
}

/**
 * Clopper–Pearson exact binomial CI via beta quantiles (CLO1934):
 *   p_L = Beta^{-1}(α/2;     x,     n − x + 1),
 *   p_U = Beta^{-1}(1 − α/2; x + 1, n − x).
 * Boundary conventions (from the proof):
 *   x = 0 ⇒ p_L = 0 and p_U = Beta^{-1}(1 − α/2; 1, n).
 *   x = n ⇒ p_U = 1 and p_L = Beta^{-1}(α/2; n, 1).
 * Exact (size ≤ α) but conservative on average (actual coverage > 1 − α).
 */
export function clopperPearsonInterval(
  x: number,
  n: number,
  alpha: number,
): CIEndpoints {
  if (x === 0) {
    return { lower: 0, upper: betaInvCDF(1 - alpha / 2, 1, n) };
  }
  if (x === n) {
    return { lower: betaInvCDF(alpha / 2, n, 1), upper: 1 };
  }
  return {
    lower: betaInvCDF(alpha / 2, x, n - x + 1),
    upper: betaInvCDF(1 - alpha / 2, x + 1, n - x),
  };
}

/**
 * Generic exact CI for discrete one-parameter families via inversion of the
 * exact two-sided test. Binomial uses the beta-quantile closed form (same as
 * `clopperPearsonInterval`); Poisson uses the gamma-Poisson duality
 *   P_θ(X ≥ k) = F_Γ(θ; k, 1)  ⇒  θ_L = Γ^{-1}(α/2; k, 1) / n,
 * where Γ^{-1} is the inverse of the gamma CDF with shape k, rate 1.
 */
export function exactDiscreteCI(
  family: 'binomial' | 'poisson',
  observedStatistic: number,
  sampleSize: number,
  alpha: number,
): CIEndpoints {
  if (family === 'binomial') {
    return clopperPearsonInterval(observedStatistic, sampleSize, alpha);
  }
  // Poisson: θ = λ, total count is ΣX_i ~ Poisson(nλ). Exact CI endpoints
  // come from inverting the two-sided test via the gamma-Poisson duality.
  const k = observedStatistic;
  const n = sampleSize;
  const lower = k === 0 ? 0 : gammaInvCDF(alpha / 2, k, 1) / n;
  const upper = gammaInvCDF(1 - alpha / 2, k + 1, 1) / n;
  return { lower, upper };
}

/** Inverse gamma CDF (shape a, rate 1) via bisection on regGammaP. */
function gammaInvCDF(p: number, a: number, rate: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  let lo = 0;
  let hi = 2 * Math.max(1, a);
  while (regGammaP(a, hi) < p) {
    lo = hi;
    hi *= 2;
    if (hi > 1e14) return hi / rate;
  }
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (regGammaP(a, mid) < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12 * Math.max(1, hi)) break;
  }
  return (0.5 * (lo + hi)) / rate;
}

// ── Profile likelihood CI (§19.7) ───────────────────────────────────────────

/** Normal(μ, σ) log-likelihood. */
export function logLikelihoodNormal2D(
  mu: number,
  sigma: number,
  data: number[],
): number {
  if (sigma <= 0) return -Infinity;
  const n = data.length;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = data[i] - mu;
    ss += d * d;
  }
  return -0.5 * n * Math.log(2 * Math.PI) - n * Math.log(sigma) - ss / (2 * sigma * sigma);
}

/** Gamma(shape, rate) log-likelihood. */
export function logLikelihoodGamma2D(
  shape: number,
  rate: number,
  data: number[],
): number {
  if (shape <= 0 || rate <= 0) return -Infinity;
  const n = data.length;
  let sumLog = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (data[i] <= 0) return -Infinity;
    sumLog += Math.log(data[i]);
    sum += data[i];
  }
  return n * shape * Math.log(rate) - n * lnGamma(shape)
    + (shape - 1) * sumLog - rate * sum;
}

/**
 * Conditional MLE of σ given μ for a Normal sample:
 *   σ̂(μ) = √(n⁻¹ Σ(X_i − μ)²).
 */
export function profileNuisanceOptimizerNormal(
  mu: number,
  data: number[],
): number {
  const n = data.length;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = data[i] - mu;
    ss += d * d;
  }
  return Math.sqrt(ss / n);
}

/**
 * Conditional MLE of β given shape α for a Gamma sample:
 *   β̂(α) = α / x̄.
 * Derivation: ∂ℓ/∂β = nα/β − Σx_i = 0 ⇒ β̂ = nα / Σx_i = α / x̄.
 */
export function profileNuisanceOptimizerGamma(
  shape: number,
  data: number[],
): number {
  return shape / sampleMean(data);
}

/**
 * Profile-likelihood CI for a scalar θ. Given a 2D log-likelihood
 * `logLikelihood(θ, ψ)` and a nuisance optimizer `psiOptimize(θ)`, computes
 *   C_P = {θ : 2 [ℓ_P(θ̂) − ℓ_P(θ)] ≤ χ²_{1, 1-α}}
 * by evaluating the profile on a uniform θ-grid over `thetaSearchRange`, then
 * bisecting for the endpoints on each side of the profile maximum.
 *
 * Returns the profile curve as `profileCurve: [θ, ℓ_P(θ)][]` so UI components
 * can trace the threshold intersection directly.
 */
export function profileLikelihoodCI(
  logLikelihood: (theta: number, psi: number) => number,
  psiOptimize: (theta: number) => number,
  thetaHat: number,
  thetaSearchRange: [number, number],
  alpha: number,
  gridPoints = 200,
): CIEndpoints & { profileCurve: Array<[number, number]> } {
  const crit = chiSquaredInvCDF(1 - alpha, 1);
  const [tLo, tHi] = thetaSearchRange;
  const profileCurve: Array<[number, number]> = [];
  let maxLL = -Infinity;
  for (let i = 0; i < gridPoints; i++) {
    const theta = tLo + (i / (gridPoints - 1)) * (tHi - tLo);
    const psi = psiOptimize(theta);
    const ll = logLikelihood(theta, psi);
    profileCurve.push([theta, ll]);
    if (ll > maxLL) maxLL = ll;
  }
  const threshold = maxLL - crit / 2;
  const g = (theta: number): number => {
    const psi = psiOptimize(theta);
    return logLikelihood(theta, psi) - threshold;
  };
  // Bracket and bisect on each side of thetaHat.
  let leftLo = tLo;
  let leftHi = thetaHat;
  if (g(leftLo) > 0) {
    // Threshold crosses below grid minimum — return grid edge.
    // (Useful for small-sample profiles that stay above threshold on one side.)
  }
  const lower = bisectRoot(g, leftLo, leftHi, 60);
  const upper = bisectRoot(g, thetaHat, tHi, 60);
  return { lower, upper, profileCurve };
}

// ── TOST equivalence procedure (§19.9) ──────────────────────────────────────

/**
 * Two One-Sided Tests (TOST) for equivalence with margin δ. Rejects the
 * non-equivalence null H_0: |θ − θ_0| ≥ δ iff both one-sided z/t tests of
 *   H_0^L: θ ≤ θ_0 − δ   (against θ > θ_0 − δ)
 *   H_0^U: θ ≥ θ_0 + δ   (against θ < θ_0 + δ)
 * reject at level α. Equivalent to checking that the (1 − 2α) two-sided CI
 * is contained in [θ_0 − δ, θ_0 + δ].
 *
 * Implemented for `normal-mean` (t-distribution small-sample exact),
 * `bernoulli` (z approximation on θ̂), and `poisson` (z approximation on λ̂).
 */
export function tostTest(
  data: number[],
  theta0: number,
  delta: number,
  alpha: number,
  family: 'normal-mean' | 'bernoulli' | 'poisson',
): { rejectLow: boolean; rejectHigh: boolean; equivalence: boolean; pLow: number; pHigh: number } {
  const n = data.length;
  const mean = sampleMean(data);
  let tLow: number;
  let tHigh: number;
  let pLow: number;
  let pHigh: number;

  if (family === 'normal-mean') {
    const s = Math.sqrt(sampleVarUnbiased(data));
    const se = s / Math.sqrt(n);
    tLow = (mean - (theta0 - delta)) / se; // should reject if large positive
    tHigh = (mean - (theta0 + delta)) / se; // should reject if large negative
    pLow = 1 - studentTCDF(tLow, n - 1);
    pHigh = studentTCDF(tHigh, n - 1);
  } else if (family === 'bernoulli') {
    const p = mean;
    const se = Math.sqrt((p * (1 - p)) / n);
    tLow = (p - (theta0 - delta)) / se;
    tHigh = (p - (theta0 + delta)) / se;
    pLow = 1 - standardNormalCDF(tLow);
    pHigh = standardNormalCDF(tHigh);
  } else {
    // poisson
    const se = Math.sqrt(mean / n);
    tLow = (mean - (theta0 - delta)) / se;
    tHigh = (mean - (theta0 + delta)) / se;
    pLow = 1 - standardNormalCDF(tLow);
    pHigh = standardNormalCDF(tHigh);
  }

  const rejectLow = pLow <= alpha;
  const rejectHigh = pHigh <= alpha;
  return { rejectLow, rejectHigh, equivalence: rejectLow && rejectHigh, pLow, pHigh };
}

/**
 * The (1 − 2α) two-sided CI that is equivalent to the TOST procedure at
 * level α, computed as the standard asymptotic Wald-type CI for the relevant
 * family. Equivalence is concluded iff this interval ⊂ [θ_0 − δ, θ_0 + δ].
 */
export function tostConfidenceInterval(
  data: number[],
  alpha: number,
  family: 'normal-mean' | 'bernoulli' | 'poisson',
): CIEndpoints {
  if (family === 'normal-mean') return tCINormalMean(data, 2 * alpha);
  if (family === 'bernoulli') return waldCI('bernoulli', data, 2 * alpha);
  return waldCI('poisson', data, 2 * alpha);
}

// ── Coverage diagnostics (§19.8) ────────────────────────────────────────────

/**
 * Exact actual coverage of a binomial CI procedure: for each p in `pGrid`,
 * sum P_p(X = x) over all x ∈ {0, ..., n} whose CI contains p. No Monte
 * Carlo — the answer is exact, and the sawtooth pattern caused by the
 * discreteness of binomial X is visible (which MC averaging would smooth
 * away).
 */
export function actualCoverageBinomial(
  ciProcedure: (x: number, n: number, alpha: number) => CIEndpoints,
  n: number,
  alpha: number,
  pGrid: number[],
): number[] {
  const intervals: CIEndpoints[] = [];
  for (let x = 0; x <= n; x++) intervals.push(ciProcedure(x, n, alpha));
  const out: number[] = [];
  for (const p of pGrid) {
    let cov = 0;
    for (let x = 0; x <= n; x++) {
      const { lower, upper } = intervals[x];
      if (p >= lower && p <= upper) cov += pmfBinomial(x, n, p);
    }
    out.push(cov);
  }
  return out;
}

/**
 * Monte Carlo coverage simulator for a continuous-parameter CI procedure.
 * `ciProcedure` consumes a sample array and an α; `sampler(trueParam, n)`
 * returns a sample of length n. Seed defaults to 42 for reproducibility.
 */
export function coverageSimulator(
  ciProcedure: (data: number[], alpha: number) => CIEndpoints,
  sampler: (trueParam: number, n: number, rng: () => number) => number[],
  trueParam: number,
  n: number,
  alpha: number,
  M: number,
  seed = 42,
): { actualCoverage: number; meanWidth: number } {
  const rng = seededRandom(seed);
  let cov = 0;
  let widthSum = 0;
  for (let i = 0; i < M; i++) {
    const data = sampler(trueParam, n, rng);
    const { lower, upper } = ciProcedure(data, alpha);
    if (trueParam >= lower && trueParam <= upper) cov++;
    widthSum += upper - lower;
  }
  return { actualCoverage: cov / M, meanWidth: widthSum / M };
}
