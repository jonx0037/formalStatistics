/**
 * estimation.ts — Point estimation utilities.
 *
 * Track 4 shared module. Complements convergence.ts (sampling, running stats,
 * convergence diagnostics) with estimator-specific computations: bias, MSE,
 * score, Fisher information, Cramér–Rao bound, James–Stein, log-likelihood.
 *
 * Topic 13 — Point Estimation & Bias-Variance: full estimator framework.
 * Topics 14–16 will extend with MLE-, MoM-, and sufficiency-specific helpers.
 */
import { sampleMean as convergenceSampleMean } from './convergence';

// Re-export so consumers import the whole estimator toolkit from one module.
export const sampleMean = convergenceSampleMean;

// ── Types ────────────────────────────────────────────────────────────────────

/** Parametric family specification for Fisher-info / log-likelihood work. */
export interface ParametricFamily {
  name: string;
  params: Record<string, number>;
  pdf: (x: number, params: Record<string, number>) => number;
  logPdf: (x: number, params: Record<string, number>) => number;
  sample: (n: number, params: Record<string, number>) => number[];
  trueParam: (paramName: string, params: Record<string, number>) => number;
}

/** Estimator specification for Monte Carlo comparison. */
export interface EstimatorSpec {
  name: string;
  compute: (sample: number[]) => number;
  isUnbiased: boolean;
  theoreticalBias?: (n: number, params: Record<string, number>) => number;
  theoreticalVariance?: (n: number, params: Record<string, number>) => number;
}

/** Summary of a Monte-Carlo sampling-distribution simulation. */
export interface SamplingDistResult {
  estimates: number[];
  mean: number;
  bias: number;
  variance: number;
  mse: number;
  se: number;
}

// ── Estimator Functions ──────────────────────────────────────────────────────

/** Sample median. Copies the input so the caller's array is not mutated. */
export function sampleMedian(x: number[]): number {
  if (x.length === 0) return 0;
  const sorted = [...x].sort((a, b) => a - b);
  const n = sorted.length;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? 0.5 * (sorted[mid - 1] + sorted[mid]) : sorted[mid];
}

/** Trimmed mean: remove ⌊n·proportion⌋ from each tail, then average. */
export function trimmedMean(x: number[], proportion: number): number {
  if (x.length === 0) return 0;
  const p = Math.max(0, Math.min(0.49, proportion));
  const sorted = [...x].sort((a, b) => a - b);
  const k = Math.floor(sorted.length * p);
  const slice = sorted.slice(k, sorted.length - k);
  if (slice.length === 0) return sampleMean(sorted);
  return sampleMean(slice);
}

/**
 * Sample variance with configurable dof correction.
 * ddof=1 → Bessel's correction (unbiased). ddof=0 → MLE (biased by factor (n-1)/n).
 * Uses Welford's single-pass algorithm for numerical stability.
 */
export function sampleVariance(x: number[], ddof: 0 | 1 = 1): number {
  const n = x.length;
  if (n - ddof <= 0) return 0;
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < n; i++) {
    const delta = x[i] - mean;
    mean += delta / (i + 1);
    m2 += delta * (x[i] - mean);
  }
  return m2 / (n - ddof);
}

/** Standard error of the sample mean: S/√n, using the unbiased S (Bessel's correction). */
export function standardError(x: number[]): number {
  if (x.length === 0) return 0;
  return Math.sqrt(sampleVariance(x, 1) / x.length);
}

// ── Bias, Variance, MSE ──────────────────────────────────────────────────────

/** Monte-Carlo bias: mean of estimates minus the true parameter. */
export function computeBias(estimates: number[], trueParam: number): number {
  return sampleMean(estimates) - trueParam;
}

/** Monte-Carlo MSE: mean squared deviation from the true parameter. */
export function computeMSE(estimates: number[], trueParam: number): number {
  if (estimates.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < estimates.length; i++) {
    const d = estimates[i] - trueParam;
    s += d * d;
  }
  return s / estimates.length;
}

/** Monte-Carlo standard error: sqrt of the sample variance of estimates. */
export function computeSE(estimates: number[]): number {
  return Math.sqrt(sampleVariance(estimates, 1));
}

/**
 * MSE decomposition for the shrinkage estimator θ̂ = c·X̄ of the mean μ.
 * Bias² = (c−1)²μ²; Var = c²σ²/n; MSE = Bias² + Var.
 */
export function mseShrinkage(
  c: number,
  mu: number,
  sigma2: number,
  n: number,
): { bias2: number; variance: number; mse: number } {
  const bias2 = (c - 1) * (c - 1) * mu * mu;
  const variance = (c * c * sigma2) / n;
  return { bias2, variance, mse: bias2 + variance };
}

/** Shrinkage factor c* that minimises MSE of c·X̄: c* = μ² / (μ² + σ²/n). */
export function optimalShrinkage(mu: number, sigma2: number, n: number): number {
  const snr = mu * mu;
  return snr / (snr + sigma2 / n);
}

/**
 * MSE for ridge regression as an illustrative single-coefficient model:
 * β̂ = (n/(n+λ))·β̂_OLS, so Bias² = (λ/(n+λ))²β², Var = (n/(n+λ))²·σ²/n.
 */
export function mseRidge(
  lambda: number,
  betaTrue: number,
  sigma2: number,
  n: number,
): { bias2: number; variance: number; mse: number } {
  const shrink = n / (n + lambda);
  const bias2 = (1 - shrink) * (1 - shrink) * betaTrue * betaTrue;
  const variance = (shrink * shrink * sigma2) / n;
  return { bias2, variance, mse: bias2 + variance };
}

/**
 * Prediction-MSE for a polynomial of given `degree` approximating a target of
 * `trueDegree`. Simple illustrative model: Bias² vanishes once degree ≥ trueDegree,
 * else grows with the squared gap; Var grows linearly in degree and inversely in n.
 */
export function msePolynomial(
  degree: number,
  trueDegree: number,
  n: number,
  sigmaNoise: number,
): { bias2: number; variance: number; mse: number } {
  const gap = Math.max(0, trueDegree - degree);
  const bias2 = gap * gap * 0.1;
  const variance = ((degree + 1) * sigmaNoise * sigmaNoise) / n;
  return { bias2, variance, mse: bias2 + variance };
}

// ── Score & Fisher Information ───────────────────────────────────────────────

/** Fisher information for Normal(μ, σ²) with respect to μ. I(μ) = 1/σ². */
export function fisherNormalMu(sigma2: number): number {
  return 1 / sigma2;
}

/** Fisher information for Bernoulli(p). I(p) = 1 / (p(1−p)). */
export function fisherBernoulli(p: number): number {
  const q = 1 - p;
  if (p <= 0 || q <= 0) return Infinity;
  return 1 / (p * q);
}

/** Fisher information for Exponential(λ). I(λ) = 1/λ². */
export function fisherExponential(lambda: number): number {
  return 1 / (lambda * lambda);
}

/** Fisher information for Poisson(λ). I(λ) = 1/λ. */
export function fisherPoisson(lambda: number): number {
  return 1 / lambda;
}

/**
 * Score function ∂/∂θ log f(x; θ) for a given parametric family.
 * Returns a closure (x, theta, otherParams?) → score value.
 */
export function scoreFunction(
  family: string,
  paramName: string,
): (x: number, theta: number, otherParams?: Record<string, number>) => number {
  if (family === 'Normal' && paramName === 'mu') {
    return (x, mu, otherParams) => {
      const sigma2 = otherParams?.sigma2 ?? 1;
      return (x - mu) / sigma2;
    };
  }
  if (family === 'Bernoulli' && paramName === 'p') {
    return (x, p) => x / p - (1 - x) / (1 - p);
  }
  if (family === 'Exponential' && paramName === 'lambda') {
    return (x, lambda) => 1 / lambda - x;
  }
  if (family === 'Poisson' && paramName === 'lambda') {
    return (x, lambda) => x / lambda - 1;
  }
  throw new Error(`scoreFunction: unsupported family/param (${family}/${paramName})`);
}

/**
 * Fisher information I(θ) for a given parametric family.
 * Returns a closure (theta, otherParams?) → I(theta).
 */
export function fisherInformation(
  family: string,
  paramName: string,
): (theta: number, otherParams?: Record<string, number>) => number {
  if (family === 'Normal' && paramName === 'mu') {
    return (_mu, otherParams) => fisherNormalMu(otherParams?.sigma2 ?? 1);
  }
  if (family === 'Bernoulli' && paramName === 'p') {
    return (p) => fisherBernoulli(p);
  }
  if (family === 'Exponential' && paramName === 'lambda') {
    return (lambda) => fisherExponential(lambda);
  }
  if (family === 'Poisson' && paramName === 'lambda') {
    return (lambda) => fisherPoisson(lambda);
  }
  throw new Error(`fisherInformation: unsupported family/param (${family}/${paramName})`);
}

// ── Cramér-Rao bound and asymptotic variances ────────────────────────────────

/** Cramér–Rao lower bound for unbiased estimators based on n iid samples. */
export function cramerRaoBound(n: number, fisherInfo: number): number {
  if (n <= 0 || fisherInfo <= 0) return Infinity;
  return 1 / (n * fisherInfo);
}

/** Asymptotic variance of the sample mean: σ²/n. */
export function asymptVarMean(sigma2: number, n: number): number {
  return sigma2 / n;
}

/**
 * Asymptotic variance of the sample median for a symmetric density:
 * Var(median) ≈ 1 / (4 n f(θ)²) where f is the density at the median.
 */
export function asymptVarMedian(f_at_median: number, n: number): number {
  if (f_at_median <= 0 || n <= 0) return Infinity;
  return 1 / (4 * n * f_at_median * f_at_median);
}

/** Asymptotic relative efficiency: Var(θ̂₁) / Var(θ̂₂). */
export function are(var1: number, var2: number): number {
  if (var2 <= 0) return Infinity;
  return var1 / var2;
}

// ── Sampling-Distribution Simulation ─────────────────────────────────────────

/**
 * Monte-Carlo simulation of an estimator's sampling distribution.
 * Draws M independent samples of size n, applies the estimator to each,
 * and returns the summary statistics.
 */
export function simulateSamplingDist(
  sampler: (n: number) => number[],
  estimator: (sample: number[]) => number,
  trueParam: number,
  n: number,
  M: number,
): SamplingDistResult {
  const estimates = new Array<number>(M);
  for (let i = 0; i < M; i++) {
    estimates[i] = estimator(sampler(n));
  }
  const mean = sampleMean(estimates);
  const bias = mean - trueParam;
  const variance = sampleVariance(estimates, 1);
  const mse = computeMSE(estimates, trueParam);
  const se = Math.sqrt(variance);
  return { estimates, mean, bias, variance, mse, se };
}

/**
 * Running estimator: applies `estimator` to the first k observations for
 * k = 1, ..., n. Used by consistency trajectory plots. Note: O(n²) in the
 * general case because we don't assume the estimator is incremental.
 */
export function runningEstimator(
  data: number[],
  estimator: (sample: number[]) => number,
): number[] {
  const out = new Array<number>(data.length);
  for (let k = 1; k <= data.length; k++) {
    out[k - 1] = estimator(data.slice(0, k));
  }
  return out;
}

// ── James-Stein ──────────────────────────────────────────────────────────────

/**
 * Positive-part James–Stein estimator for the mean of N(θ, σ²·I_d) with σ² = 1.
 * Shrinkage factor c = max(0, 1 − (d−2)/||x||²). Returns x unchanged when d ≤ 2.
 */
export function jamesStein(x: number[]): number[] {
  const d = x.length;
  if (d <= 2) return [...x];
  let norm2 = 0;
  for (let i = 0; i < d; i++) norm2 += x[i] * x[i];
  if (norm2 === 0) return [...x];
  const c = Math.max(0, 1 - (d - 2) / norm2);
  return x.map((xi) => c * xi);
}

/** Total squared-error risk of the MLE in d dimensions: d·σ². */
export function riskMLE(d: number, sigma2: number): number {
  return d * sigma2;
}

/**
 * Approximate James–Stein risk:
 * R(JS, θ) = d·σ² − (d−2)² σ⁴ / (||θ||² + (d−2) σ²).
 * Dominates the MLE whenever d ≥ 3.
 */
export function riskJamesStein(d: number, sigma2: number, thetaNorm2: number): number {
  if (d <= 2) return d * sigma2;
  const correction = ((d - 2) * (d - 2) * sigma2 * sigma2) / (thetaNorm2 + (d - 2) * sigma2);
  return d * sigma2 - correction;
}

// ── Log-Likelihood ───────────────────────────────────────────────────────────

/** Log-likelihood ℓ(θ) = Σᵢ log f(xᵢ; θ) for iid data. */
export function logLikelihood(
  data: number[],
  logPdf: (x: number, theta: number) => number,
  theta: number,
): number {
  let s = 0;
  for (let i = 0; i < data.length; i++) s += logPdf(data[i], theta);
  return s;
}

/** Log-likelihood evaluated over a grid of θ values. */
export function logLikelihoodCurve(
  data: number[],
  logPdf: (x: number, theta: number) => number,
  thetaGrid: number[],
): { thetas: number[]; logLiks: number[] } {
  const logLiks = thetaGrid.map((theta) => logLikelihood(data, logPdf, theta));
  return { thetas: thetaGrid, logLiks };
}

// ── Dev tests (browser-side only, runs once per page load in dev) ────────────

if (import.meta.env.DEV && typeof window !== 'undefined') {
  const near = (a: number, b: number, eps = 1e-9) => Math.abs(a - b) < eps;
  const withinPct = (a: number, b: number, pct: number) =>
    Math.abs(a - b) <= pct * Math.abs(b);

  const results: boolean[] = [];

  // 1. sampleMean([1,2,3,4,5]) === 3
  results.push(near(sampleMean([1, 2, 3, 4, 5]), 3));

  // 2. sampleVariance with ddof=1 → 2.5; with ddof=0 → 2.0
  results.push(
    near(sampleVariance([1, 2, 3, 4, 5], 1), 2.5) &&
      near(sampleVariance([1, 2, 3, 4, 5], 0), 2.0),
  );

  // 3 & 4. Simulate 1000 sample means from N(5, 4) with n=25.
  //     Use a deterministic-ish Monte Carlo check: bias near 0, MSE near 4/25 = 0.16.
  {
    // inline normal sampler via Box–Muller so we don't import convergence.ts here
    const rng = (() => {
      let s = 42;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x1_0000_0000;
      };
    })();
    const normalSample = (mu: number, sigma: number) => {
      const u = Math.max(rng(), 1e-12);
      const v = rng();
      return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    const draws: number[] = [];
    for (let i = 0; i < 1000; i++) {
      const s: number[] = [];
      for (let j = 0; j < 25; j++) s.push(normalSample(5, 2));
      draws.push(sampleMean(s));
    }
    const bias = computeBias(draws, 5);
    const mse = computeMSE(draws, 5);
    results.push(Math.abs(bias) < 0.1);
    results.push(withinPct(mse, 4 / 25, 0.15));
  }

  // 5. fisherNormalMu(4) === 0.25
  results.push(near(fisherNormalMu(4), 0.25));

  // 6. fisherBernoulli(0.3) ≈ 4.762
  results.push(withinPct(fisherBernoulli(0.3), 1 / (0.3 * 0.7), 1e-6));

  // 7. cramerRaoBound(100, 0.25) === 0.04
  results.push(near(cramerRaoBound(100, 0.25), 0.04, 1e-9));

  // 8. mseShrinkage(1.0, 5, 4, 25) === {bias2: 0, variance: 0.16, mse: 0.16}
  {
    const r = mseShrinkage(1.0, 5, 4, 25);
    results.push(near(r.bias2, 0) && near(r.variance, 0.16) && near(r.mse, 0.16));
  }

  // 9. optimalShrinkage(5, 4, 25) ≈ 25 / (25 + 0.16) ≈ 0.993636…
  results.push(withinPct(optimalShrinkage(5, 4, 25), 25 / (25 + 4 / 25), 1e-6));

  // 10. jamesStein([3, -1, 2]) norm < input norm
  {
    const input = [3, -1, 2];
    const js = jamesStein(input);
    const normIn = Math.sqrt(9 + 1 + 4);
    const normOut = Math.sqrt(js[0] ** 2 + js[1] ** 2 + js[2] ** 2);
    results.push(normOut < normIn && normOut > 0);
  }

  const passed = results.filter(Boolean).length;
  if (passed === results.length) {
    // eslint-disable-next-line no-console
    console.log(`[estimation.ts] ${passed}/${results.length} dev tests passed ✓`);
  } else {
    // eslint-disable-next-line no-console
    console.warn(
      `[estimation.ts] ${passed}/${results.length} dev tests passed — failures: `,
      results.map((p, i) => (p ? null : i + 1)).filter((i) => i !== null),
    );
  }
}
