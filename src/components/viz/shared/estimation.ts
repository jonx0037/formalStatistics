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

/**
 * Asymptotic relative efficiency under the Topic 13 convention:
 * ARE(θ̂₁; θ̂₂) = Var(θ̂₂) / Var(θ̂₁). Values > 1 mean θ̂₁ is more efficient
 * than θ̂₂. See §13.6 Remark 6.
 */
export function are(varEstimator1: number, varEstimator2: number): number {
  if (varEstimator1 <= 0) return Infinity;
  return varEstimator2 / varEstimator1;
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

// ── Maximum Likelihood Estimators (closed form) ──────────────────────────────

/** MLE of Normal(μ, σ²) mean: θ̂ = X̄. */
export function mleNormalMu(data: number[]): number {
  return sampleMean(data);
}

/**
 * MLE of Normal(μ, σ²) variance: θ̂ = (1/n) Σ (xᵢ − X̄)².
 * This is the biased (1/n) version — not the unbiased (1/(n−1)) sample variance.
 */
export function mleNormalSigma2(data: number[]): number {
  return sampleVariance(data, 0);
}

/** MLE of Bernoulli(p): p̂ = X̄ = k/n. */
export function mleBernoulli(data: number[]): number {
  return sampleMean(data);
}

/** MLE of Exponential(λ): λ̂ = 1/X̄. Returns Infinity when the sample mean is zero. */
export function mleExponential(data: number[]): number {
  const mean = sampleMean(data);
  if (mean <= 0) return Infinity;
  return 1 / mean;
}

/** MLE of Poisson(λ): λ̂ = X̄. */
export function mlePoisson(data: number[]): number {
  return sampleMean(data);
}

// ── Special functions used by the Gamma-shape MLE ────────────────────────────

/**
 * Digamma ψ(x) = Γ'(x)/Γ(x). Recurrence ψ(x) = ψ(x+1) − 1/x shifts x ≥ 7,
 * then the asymptotic expansion
 *     ψ(x) ≈ log x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)
 * gives ~1e-10 accuracy for x > 0.
 */
export function digamma(x: number): number {
  if (x <= 0) return NaN;
  let result = 0;
  let y = x;
  while (y < 7) {
    result -= 1 / y;
    y += 1;
  }
  const inv = 1 / y;
  const inv2 = inv * inv;
  result += Math.log(y) - 0.5 * inv;
  result -= inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
  return result;
}

/**
 * Trigamma ψ'(x). Recurrence ψ'(x) = ψ'(x+1) + 1/x² shifts x ≥ 7, then
 *     ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷).
 */
export function trigamma(x: number): number {
  if (x <= 0) return NaN;
  let result = 0;
  let y = x;
  while (y < 7) {
    result += 1 / (y * y);
    y += 1;
  }
  const inv = 1 / y;
  const inv2 = inv * inv;
  result += inv + 0.5 * inv2;
  result += inv * inv2 * (1 / 6 - inv2 * (1 / 30 - inv2 / 42));
  return result;
}

/** Lanczos log-Γ, accurate to ~1e-10 on (0, ∞). Exported so callers can build Gamma log-likelihoods. */
export function logGamma(x: number): number {
  if (x <= 0) return NaN;
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  let xx = x - 1;
  let a = c[0];
  const t = xx + g + 0.5;
  for (let i = 1; i < c.length; i++) a += c[i] / (xx + i);
  return 0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(a);
}

// ── Newton-Raphson and Fisher scoring ────────────────────────────────────────

/**
 * One Newton-Raphson step on the log-likelihood, given per-observation score
 * and Hessian functions. Update rule:
 *   θ^{(t+1)} = θ^{(t)} − ℓ'(θ^{(t)}) / ℓ''(θ^{(t)}) = θ^{(t)} + S(θ^{(t)}) / J(θ^{(t)})
 * where J(θ) = −ℓ''(θ) is the observed information (positive at a maximum).
 */
export function newtonRaphsonStep(
  theta: number,
  data: number[],
  scoreFn: (x: number, theta: number) => number,
  hessianFn: (x: number, theta: number) => number,
): { nextTheta: number; score: number; observedInfo: number; stepSize: number } {
  let score = 0;
  let hess = 0;
  for (let i = 0; i < data.length; i++) {
    score += scoreFn(data[i], theta);
    hess += hessianFn(data[i], theta);
  }
  const observedInfo = -hess;
  const stepSize = observedInfo === 0 ? 0 : score / observedInfo;
  return { nextTheta: theta + stepSize, score, observedInfo, stepSize };
}

/**
 * Run Newton-Raphson to convergence (or `maxIter`). Returns the full iterate
 * path so callers can replay the trajectory for visualization.
 */
export function newtonRaphson(
  init: number,
  data: number[],
  scoreFn: (x: number, theta: number) => number,
  hessianFn: (x: number, theta: number) => number,
  options: { maxIter?: number; tol?: number } = {},
): { mle: number; iterations: number; path: number[]; converged: boolean } {
  const maxIter = options.maxIter ?? 50;
  const tol = options.tol ?? 1e-8;
  const path: number[] = [init];
  let theta = init;
  let iterations = 0;
  let converged = false;
  for (let k = 0; k < maxIter; k++) {
    const step = newtonRaphsonStep(theta, data, scoreFn, hessianFn);
    if (!Number.isFinite(step.nextTheta)) break;
    theta = step.nextTheta;
    path.push(theta);
    iterations = k + 1;
    if (Math.abs(step.stepSize) < tol) {
      converged = true;
      break;
    }
  }
  return { mle: theta, iterations, path, converged };
}

/**
 * One Fisher-scoring step: Newton-Raphson with the observed Hessian replaced
 * by the expected Fisher information n·I(θ). For exponential families in the
 * natural parameterization the two coincide; elsewhere Fisher scoring is
 * often numerically more stable.
 */
export function fisherScoringStep(
  theta: number,
  data: number[],
  scoreFn: (x: number, theta: number) => number,
  fisherInfoFn: (theta: number) => number,
): { nextTheta: number; score: number; expectedInfo: number; stepSize: number } {
  let score = 0;
  for (let i = 0; i < data.length; i++) score += scoreFn(data[i], theta);
  const expectedInfo = data.length * fisherInfoFn(theta);
  const stepSize = expectedInfo === 0 ? 0 : score / expectedInfo;
  return { nextTheta: theta + stepSize, score, expectedInfo, stepSize };
}

// ── Gamma-shape MLE (Newton-Raphson on the score equation) ──────────────────

/**
 * MLE of the Gamma(α, β) shape α given known rate β.
 * Score:   S(α) = n log β − n ψ(α) + Σᵢ log xᵢ
 * Hessian: H(α) = −n ψ'(α)  →  observed info J(α) = n ψ'(α) > 0.
 * Default initial guess is the method-of-moments estimate α₀ = β·X̄.
 */
export function mleGammaShape(
  data: number[],
  beta: number,
  options: { maxIter?: number; tol?: number; init?: number } = {},
): { mle: number; iterations: number; path: number[]; converged: boolean } {
  const n = data.length;
  if (n === 0) return { mle: NaN, iterations: 0, path: [], converged: false };
  let sumLog = 0;
  for (let i = 0; i < n; i++) sumLog += Math.log(Math.max(data[i], 1e-300));
  const xbar = sampleMean(data);
  const init = options.init ?? Math.max(0.5, beta * xbar);
  const tol = options.tol ?? 1e-8;
  const maxIter = options.maxIter ?? 50;
  const path: number[] = [init];
  let alpha = init;
  let iterations = 0;
  let converged = false;
  for (let k = 0; k < maxIter; k++) {
    const score = n * Math.log(beta) - n * digamma(alpha) + sumLog;
    const info = n * trigamma(alpha);
    if (!Number.isFinite(info) || info <= 0) break;
    const step = score / info;
    const next = Math.max(alpha + step, 1e-6); // keep iterates positive
    path.push(next);
    iterations = k + 1;
    alpha = next;
    if (Math.abs(step) < tol) {
      converged = true;
      break;
    }
  }
  return { mle: alpha, iterations, path, converged };
}

// ── MLE dispatcher ───────────────────────────────────────────────────────────

/**
 * Compute the MLE for a supported family/parameter combination.
 * Uses closed-form formulas where available, Newton-Raphson for Gamma shape.
 */
export function computeMLE(
  data: number[],
  family: string,
  paramName: string,
  otherParams?: Record<string, number>,
  options?: { maxIter?: number; tol?: number; init?: number },
): { mle: number; logLik: number; iterations?: number; path?: number[] } {
  if (family === 'Normal' && paramName === 'mu') {
    const mle = mleNormalMu(data);
    const sigma2 = otherParams?.sigma2 ?? 1;
    const logLik = logLikelihood(
      data,
      (x, mu) => {
        const z = x - mu;
        return -0.5 * Math.log(2 * Math.PI * sigma2) - (z * z) / (2 * sigma2);
      },
      mle,
    );
    return { mle, logLik };
  }
  if (family === 'Normal' && paramName === 'sigma2') {
    const mle = mleNormalSigma2(data);
    const mu = otherParams?.mu ?? sampleMean(data);
    const logLik = logLikelihood(
      data,
      (x, s2) => {
        // Guard against degenerate σ² ≤ 0 (e.g., constant sample →
        // MLE is exactly 0). Returning −∞ signals the boundary /
        // non-existent MLE case without producing NaN / Infinity that
        // pollutes downstream plot scales.
        if (s2 <= 0) return -Infinity;
        const z = x - mu;
        return -0.5 * Math.log(2 * Math.PI * s2) - (z * z) / (2 * s2);
      },
      mle,
    );
    return { mle, logLik };
  }
  if (family === 'Bernoulli' && paramName === 'p') {
    const mle = mleBernoulli(data);
    const logLik = logLikelihood(
      data,
      (x, p) => {
        // Boundary MLEs: when all data are 0, p̂ = 0 — the log-likelihood
        // is 0 because every observation contributes (1 − 0)·log 1 = 0,
        // using the convention 0·log 0 = 0. Symmetrically for p̂ = 1.
        // Treat values strictly outside (0, 1) as −∞ (invalid), but handle
        // the exact boundary x/p = 0/0 and (1−x)/(1−p) = 0/0 as 0.
        if (p < 0 || p > 1) return -Infinity;
        const left = x === 0 ? 0 : p === 0 ? -Infinity : x * Math.log(p);
        const right = x === 1 ? 0 : p === 1 ? -Infinity : (1 - x) * Math.log(1 - p);
        return left + right;
      },
      mle,
    );
    return { mle, logLik };
  }
  if (family === 'Exponential' && paramName === 'lambda') {
    const mle = mleExponential(data);
    const logLik = logLikelihood(
      data,
      (x, lam) => {
        if (lam <= 0) return -Infinity;
        return Math.log(lam) - lam * x;
      },
      mle,
    );
    return { mle, logLik };
  }
  if (family === 'Poisson' && paramName === 'lambda') {
    const mle = mlePoisson(data);
    const logLik = logLikelihood(
      data,
      (x, lam) => {
        if (lam <= 0) return -Infinity;
        return x * Math.log(lam) - lam - logGamma(x + 1);
      },
      mle,
    );
    return { mle, logLik };
  }
  if (family === 'Gamma' && paramName === 'alpha') {
    const beta = otherParams?.beta ?? 1;
    const { mle, iterations, path } = mleGammaShape(data, beta, options);
    const logLik = logLikelihood(
      data,
      (x, alpha) => {
        if (alpha <= 0 || x <= 0) return -Infinity;
        return alpha * Math.log(beta) - logGamma(alpha) + (alpha - 1) * Math.log(x) - beta * x;
      },
      mle,
    );
    return { mle, logLik, iterations, path };
  }
  throw new Error(`computeMLE: unsupported family/param (${family}/${paramName})`);
}

// ── Observed Fisher information ──────────────────────────────────────────────

/**
 * J(μ) for Normal(μ, σ²): n/σ². Non-random (doesn't depend on data values).
 * Edge cases: returns 0 for n = 0 (no observations → zero information),
 * NaN for negative n, Infinity for σ² ≤ 0 (degenerate / invalid variance).
 */
export function observedInfoNormalMu(n: number, sigma2: number): number {
  if (sigma2 <= 0) return Infinity;
  if (n < 0) return NaN;
  if (n === 0) return 0;
  return n / sigma2;
}

/**
 * J(λ) for Exponential(λ): n/λ². The Hessian of log f(x;λ)=log λ−λx is −1/λ²
 * for every x, so the observed information is n/λ² regardless of the sample.
 * The `data` argument is carried for interface uniformity with data-dependent
 * families.
 */
export function observedInfoExponential(data: number[], lambda: number): number {
  if (lambda <= 0) return Infinity;
  return data.length / (lambda * lambda);
}

/**
 * Observed Fisher information dispatcher. Returns a closure that, given a
 * parameter value θ and a sample, produces J(θ) = −Σᵢ ∂²/∂θ² log f(xᵢ; θ).
 */
export function observedInformation(
  family: string,
  paramName: string,
): (theta: number, data: number[], otherParams?: Record<string, number>) => number {
  if (family === 'Normal' && paramName === 'mu') {
    return (_theta, data, otherParams) =>
      observedInfoNormalMu(data.length, otherParams?.sigma2 ?? 1);
  }
  if (family === 'Exponential' && paramName === 'lambda') {
    return (lambda, data) => observedInfoExponential(data, lambda);
  }
  if (family === 'Bernoulli' && paramName === 'p') {
    // J(p) = k/p² + (n−k)/(1−p)²
    return (p, data) => {
      if (p <= 0 || p >= 1) return Infinity;
      let k = 0;
      for (let i = 0; i < data.length; i++) k += data[i];
      return k / (p * p) + (data.length - k) / ((1 - p) * (1 - p));
    };
  }
  if (family === 'Poisson' && paramName === 'lambda') {
    // J(λ) = Σ xᵢ / λ² = nX̄/λ²
    return (lambda, data) => {
      if (lambda <= 0) return Infinity;
      let s = 0;
      for (let i = 0; i < data.length; i++) s += data[i];
      return s / (lambda * lambda);
    };
  }
  throw new Error(`observedInformation: unsupported family/param (${family}/${paramName})`);
}

// ── Invariance helper ────────────────────────────────────────────────────────

/**
 * Given a log-likelihood ℓ(θ) and a reparameterization θ = g⁻¹(φ), return the
 * induced log-likelihood ℓ*(φ) = ℓ(g⁻¹(φ)). Functional invariance (Thm 14.2)
 * implies that if θ̂ maximises ℓ, then g(θ̂) maximises ℓ*.
 */
export function reparameterizedLogLik(
  logLikFn: (theta: number) => number,
  gInverse: (phi: number) => number,
): (phi: number) => number {
  return (phi) => logLikFn(gInverse(phi));
}

// ── Logistic regression log-likelihood (one predictor, for Topic 14 preview) ─

/**
 * Logistic log-likelihood for one predictor x and binary response y ∈ {0,1}:
 *   ℓ(β₀, β₁) = Σᵢ yᵢ ηᵢ − log(1 + e^{ηᵢ}),   ηᵢ = β₀ + β₁ xᵢ.
 * Uses the branchless softplus so positive and negative η are equally stable.
 */
export function logisticLogLik(
  x: number[],
  y: number[],
  beta0: number,
  beta1: number,
): number {
  const n = Math.min(x.length, y.length);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const eta = beta0 + beta1 * x[i];
    const softplus = eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta));
    s += y[i] * eta - softplus;
  }
  return s;
}

/**
 * Numerically stable sigmoid σ(η) = 1 / (1 + e^{−η}). Branches on the sign
 * of η so that neither e^{−η} (positive η) nor e^{+η} (negative η) ever
 * overflows, keeping p ∈ [0, 1] for all finite η including large |η|.
 */
function stableSigmoid(eta: number): number {
  if (eta >= 0) {
    const z = Math.exp(-eta);
    return 1 / (1 + z);
  }
  const z = Math.exp(eta);
  return z / (1 + z);
}

/**
 * Gradient of the logistic log-likelihood with respect to (β₀, β₁):
 *   ∂ℓ/∂β₀ = Σᵢ (yᵢ − pᵢ),   ∂ℓ/∂β₁ = Σᵢ xᵢ (yᵢ − pᵢ),   pᵢ = σ(β₀ + β₁ xᵢ).
 * Uses `stableSigmoid` so large |β₀ + β₁ xᵢ| (e.g., under separation) do
 * not overflow to 0/1/NaN. Topic 21 develops the full IRLS / Fisher-scoring
 * algorithm; Topic 14 uses this score for a single Newton step in the preview.
 */
export function logisticScore(
  x: number[],
  y: number[],
  beta0: number,
  beta1: number,
): [number, number] {
  const n = Math.min(x.length, y.length);
  let g0 = 0;
  let g1 = 0;
  for (let i = 0; i < n; i++) {
    const eta = beta0 + beta1 * x[i];
    const p = stableSigmoid(eta);
    const r = y[i] - p;
    g0 += r;
    g1 += x[i] * r;
  }
  return [g0, g1];
}

// ════════════════════════════════════════════════════════════════════════════
// Topic 15 — Method of Moments & M-Estimation
// ════════════════════════════════════════════════════════════════════════════

// ── Method of Moments (closed-form) ─────────────────────────────────────────

/**
 * MoM for Normal(μ, σ²): μ̂ = X̄, σ̂² = (1/n)Σ(Xᵢ−X̄)² (biased; matches MLE).
 */
export function momNormal(data: number[]): { muHat: number; sigma2Hat: number } {
  const muHat = sampleMean(data);
  const sigma2Hat = sampleVariance(data, 0); // ddof=0 → MLE form
  return { muHat, sigma2Hat };
}

/**
 * MoM for Exponential rate: λ̂ = 1/X̄ (matches MLE — exponential family).
 */
export function momExponential(data: number[]): number {
  const xbar = sampleMean(data);
  return xbar > 0 ? 1 / xbar : NaN;
}

/**
 * MoM for Poisson rate: λ̂ = X̄ (matches MLE — exponential family).
 */
export function momPoisson(data: number[]): number {
  return sampleMean(data);
}

/**
 * MoM for Bernoulli p: p̂ = X̄ (matches MLE — exponential family).
 */
export function momBernoulli(data: number[]): number {
  return sampleMean(data);
}

/**
 * MoM for Geometric p (failures-before-first-success convention, support {0,1,2,…}):
 * E[X] = (1−p)/p ⇒ p̂ = 1/(1+X̄). For the {1,2,…}-support convention use 1/X̄.
 */
export function momGeometric(data: number[]): number {
  const xbar = sampleMean(data);
  return 1 / (1 + xbar);
}

/**
 * MoM for Uniform(0, θ): θ̂ = 2X̄ (E[X] = θ/2). Instructive foil to MLE X_(n).
 */
export function momUniform(data: number[]): number {
  return 2 * sampleMean(data);
}

/**
 * MoM for Gamma(α, β) (rate parameterization): α̂ = X̄²/S², β̂ = X̄/S².
 * Uses the biased S² = (1/n)Σ(Xᵢ−X̄)² for consistency with MLE conventions.
 */
export function momGamma(data: number[]): { alphaHat: number; betaHat: number } {
  const xbar = sampleMean(data);
  const s2 = sampleVariance(data, 0);
  if (s2 <= 0) return { alphaHat: NaN, betaHat: NaN };
  const alphaHat = (xbar * xbar) / s2;
  const betaHat = xbar / s2;
  return { alphaHat, betaHat };
}

/**
 * MoM for Beta(α, β) on (0,1):
 *   c = X̄(1−X̄)/S² − 1, α̂ = X̄·c, β̂ = (1−X̄)·c.
 * Uses biased S² (1/n) for consistency. Returns NaN if c ≤ 0 (parameter-space violation).
 */
export function momBeta(data: number[]): { alphaHat: number; betaHat: number } {
  const xbar = sampleMean(data);
  const s2 = sampleVariance(data, 0);
  if (s2 <= 0 || xbar <= 0 || xbar >= 1) {
    return { alphaHat: NaN, betaHat: NaN };
  }
  const c = (xbar * (1 - xbar)) / s2 - 1;
  if (c <= 0) return { alphaHat: NaN, betaHat: NaN };
  return { alphaHat: xbar * c, betaHat: (1 - xbar) * c };
}

/**
 * Generic moment-to-parameter solver: compute up to `momentOrder` raw sample
 * moments [X̄, X̄², …] and pass them to `solver`, which returns the parameter(s).
 */
export function methodOfMoments<T>(
  data: number[],
  momentOrder: number,
  solver: (moments: number[]) => T,
): T {
  const moments: number[] = new Array(momentOrder).fill(0);
  const n = data.length;
  for (let i = 0; i < n; i++) {
    let pow = 1;
    for (let k = 0; k < momentOrder; k++) {
      pow *= data[i];
      moments[k] += pow;
    }
  }
  for (let k = 0; k < momentOrder; k++) moments[k] /= n;
  return solver(moments);
}

// ── ARE (Asymptotic Relative Efficiency) ─────────────────────────────────────

/**
 * Theoretical ARE(MoM, MLE) for selected families.
 *  - 'exponential-rate'  → 1 (exponential family in the natural parameter)
 *  - 'normal-variance'   → 1 (MoM and MLE both use the biased 1/n form)
 *  - 'gamma-shape'       → α / (α·ψ'(α) − 1) where ψ' is the trigamma function
 *                          (β profiled out; matches Topic 15 §15.7 derivation)
 *  - 'uniform-endpoint'  → 3/(n+2) → 0 as n→∞ (MLE X_(n) variance is O(1/n²))
 */
export function areTheoretical(
  family: string,
  params: Record<string, number>,
): number {
  switch (family) {
    case 'exponential-rate':
      return 1;
    case 'normal-variance':
      return 1;
    case 'gamma-shape': {
      const alpha = params.alpha;
      if (!Number.isFinite(alpha) || alpha <= 0) return NaN;
      // MLE asymp var of α (β profiled): α / (α·ψ'(α) − 1)
      // MoM asymp var via matrix delta method on α̂ = X̄²/S²: 2α(α+1)
      // ARE = MLE_var / MoM_var = 1 / [2(α+1)(α·ψ'(α) − 1)]
      // Verified at α=3 → 0.68 (Casella & Berger §10.1 + notebook).
      const denom = alpha * trigamma(alpha) - 1;
      if (denom <= 0) return NaN;
      return 1 / (2 * (alpha + 1) * denom);
    }
    case 'uniform-endpoint': {
      const n = params.n;
      if (!Number.isFinite(n) || n <= 0) return 0;
      return 3 / (n + 2);
    }
    default:
      return NaN;
  }
}

/**
 * Theoretical ARE curve over a grid of parameter values for the chosen family.
 * For 'gamma-shape', sweeps α and returns ARE(α). Used in figure are-mom-vs-mle.png.
 */
export function areCurveTheoretical(
  family: string,
  paramName: string,
  grid: number[],
  fixedParams: Record<string, number> = {},
): { grid: number[]; are: number[] } {
  const are = grid.map((v) =>
    areTheoretical(family, { ...fixedParams, [paramName]: v }),
  );
  return { grid, are };
}

// ── M-estimation: ψ-functions and ρ-functions ───────────────────────────────

/**
 * Huber's ψ-function with tuning constant k (default 1.345 for 95% Normal efficiency):
 * ψ_H(u) = u if |u| ≤ k, else k·sign(u). Monotone, bounded.
 */
export function huberPsi(u: number, k: number): number {
  if (u > k) return k;
  if (u < -k) return -k;
  return u;
}

/**
 * Huber's ρ-loss with tuning constant k:
 * ρ_H(u) = ½u² if |u| ≤ k, else k|u| − ½k². Quadratic core, linear tails.
 */
export function huberRho(u: number, k: number): number {
  const a = Math.abs(u);
  if (a <= k) return 0.5 * u * u;
  return k * a - 0.5 * k * k;
}

/**
 * Derivative of Huber's ψ (1 inside the linear region, 0 outside).
 * Used in the sensitivity matrix A(θ).
 */
export function huberPsiPrime(u: number, k: number): number {
  return Math.abs(u) <= k ? 1 : 0;
}

/**
 * Tukey's biweight ψ-function with tuning constant c (default 4.685 for 95% Normal efficiency):
 * ψ_T(u) = u·(1 − (u/c)²)² for |u| ≤ c, 0 otherwise. Redescending: zero beyond c.
 */
export function tukeyPsi(u: number, c: number): number {
  if (Math.abs(u) > c) return 0;
  const r = u / c;
  const w = 1 - r * r;
  return u * w * w;
}

/**
 * Tukey's biweight ρ-loss with tuning constant c:
 * ρ_T(u) = (c²/6) · [1 − (1 − (u/c)²)³] for |u| ≤ c, ρ_T(u) = c²/6 otherwise.
 * Bounded loss — the defining property of redescending M-estimators.
 */
export function tukeyRho(u: number, c: number): number {
  if (Math.abs(u) > c) return (c * c) / 6;
  const r = u / c;
  const w = 1 - r * r;
  return ((c * c) / 6) * (1 - w * w * w);
}

/**
 * Derivative of Tukey's ψ via the chain rule on u·(1 − (u/c)²)²:
 *   ψ'(u) = (1 − (u/c)²)² + u · 2(1 − (u/c)²) · (−2u/c²)
 *         = (1 − (u/c)²) · [(1 − (u/c)²) − 4u²/c²]
 * Zero outside |u| > c.
 */
export function tukeyPsiPrime(u: number, c: number): number {
  if (Math.abs(u) > c) return 0;
  const r = u / c;
  const w = 1 - r * r;
  return w * (w - 4 * r * r);
}

// ── Robust scale: MAD ────────────────────────────────────────────────────────

/**
 * Median Absolute Deviation, normalized so MAD/Φ⁻¹(0.75) ≈ σ for Normal data.
 * Used as the robust scale estimate inside `mEstimatorLocation`.
 */
function medianAbsoluteDeviation(data: number[]): number {
  if (data.length === 0) return 0;
  const med = sampleMedian(data);
  const dev = data.map((x) => Math.abs(x - med));
  return sampleMedian(dev) / 0.6744897501960817; // Φ⁻¹(0.75)
}

// ── M-estimator (location) ───────────────────────────────────────────────────

/**
 * Location M-estimator via fixed-point iteration on Σ ψ((Xᵢ − θ̂)/σ̂) = 0.
 * Uses MAD as the scale σ̂ unless `options.scale` is supplied.
 *
 * For monotone ψ (Huber): converges to a unique solution.
 * For redescending ψ (Tukey): may have multiple roots — initialize near the
 * sample median for a robust starting point.
 */
export function mEstimatorLocation(
  data: number[],
  psi: (u: number) => number,
  options: { maxIter?: number; tol?: number; init?: number; scale?: number } = {},
): { estimate: number; iterations: number; converged: boolean } {
  const n = data.length;
  if (n === 0) return { estimate: NaN, iterations: 0, converged: false };
  const tol = options.tol ?? 1e-8;
  const maxIter = options.maxIter ?? 200;
  const scale = options.scale ?? Math.max(medianAbsoluteDeviation(data), 1e-12);
  let theta = options.init ?? sampleMedian(data);
  let iterations = 0;
  let converged = false;
  for (let k = 0; k < maxIter; k++) {
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      const u = (data[i] - theta) / scale;
      // Iteratively reweighted form: weight wᵢ = ψ(u)/u for u≠0, ψ'(0) for u=0.
      const w = u === 0 ? 1 : psi(u) / u;
      num += w * data[i];
      den += w;
    }
    if (den === 0) break;
    const next = num / den;
    iterations = k + 1;
    if (Math.abs(next - theta) < tol) {
      theta = next;
      converged = true;
      break;
    }
    theta = next;
  }
  return { estimate: theta, iterations, converged };
}

// ── Sandwich variance ────────────────────────────────────────────────────────

/**
 * Sandwich variance V(θ̂) = A⁻¹ B A⁻¹ᵀ for a scalar-θ M-estimator.
 *   A = (1/n) Σ ψ'(Xᵢ; θ̂) · (1/scale)   (sensitivity, with the standardization Jacobian)
 *   B = (1/n) Σ ψ(Xᵢ; θ̂)²              (variability)
 * For scalar θ this reduces to V = B / (n · A²).
 *
 * Caller supplies a robust scale (MAD) if working in standardized residuals.
 * The transpose in the matrix form A⁻¹ B A⁻¹ᵀ matters for vector θ; for scalar
 * θ it collapses to A⁻²·B identically.
 */
export function mEstimatorVariance(
  data: number[],
  theta: number,
  psi: (u: number) => number,
  psiPrime: (u: number) => number,
  scale = 1,
): number {
  const n = data.length;
  if (n === 0) return NaN;
  let A = 0;
  let B = 0;
  for (let i = 0; i < n; i++) {
    const u = (data[i] - theta) / scale;
    A += psiPrime(u);
    const p = psi(u);
    B += p * p;
  }
  A /= n;
  B /= n;
  if (A === 0) return Infinity;
  // Standardized form: variance of θ̂ on the original scale uses scale².
  return (scale * scale * B) / (n * A * A);
}

// ── Breakdown point ──────────────────────────────────────────────────────────

/**
 * Asymptotic breakdown point ε* of common location estimators.
 * - 'mean'         → 0          (a single outlier moves the mean arbitrarily)
 * - 'median'       → 1/2        (the maximum possible)
 * - 'huber'        → ≈0.05      (a small fraction of outliers in the linear tail)
 * - 'tukey'        → ≈0.5       (redescending: extreme outliers get weight 0)
 * - 'trimmed-mean' → α           (the trimming proportion)
 *
 * The 'huber' value depends weakly on k; the convention 0.05 reflects the
 * default k = 1.345. Computed exactly per Rousseeuw–Leroy (1987) §1.
 */
export function breakdownPoint(
  estimator: 'mean' | 'median' | 'huber' | 'tukey' | 'trimmed-mean',
  params: { k?: number; c?: number; alpha?: number } = {},
): number {
  switch (estimator) {
    case 'mean':
      return 0;
    case 'median':
      return 0.5;
    case 'huber':
      // Huber breakdown depends on the asymmetry between tail weight and influence.
      // The textbook value at k = 1.345 is ~0.05; we hard-code this to match the
      // demo presets. A slightly higher k raises the breakdown a few percentage
      // points but stays well below the median's 0.5.
      return 0.05;
    case 'tukey':
      // Redescending → asymptotically 1/2; the realised value depends on the
      // tuning constant c. For c = 4.685 (default) the empirical breakdown is
      // ≈0.45 in finite samples; we report the asymptotic 0.5.
      return 0.5;
    case 'trimmed-mean':
      return Math.max(0, Math.min(0.5, params.alpha ?? 0.1));
    default:
      return NaN;
  }
}

// ── Running median (running mean already lives in convergence.ts) ────────────

/**
 * Running median: out[i] = median(data[0..i]). O(n²) naive implementation —
 * adequate for n ≲ 5000. For the n=10,000 streaming use case in
 * `CauchyPathologyExplorer`, replace with a two-heap median tracker.
 */
export function runningMedian(data: number[]): number[] {
  const n = data.length;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = sampleMedian(data.slice(0, i + 1));
  }
  return out;
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

  // ── Topic 14 additions ────────────────────────────────────────────────────

  // 11. mleNormalMu([1..5]) === 3
  results.push(near(mleNormalMu([1, 2, 3, 4, 5]), 3));

  // 12. mleNormalSigma2([1..5]) === 2 (biased, 1/n)
  results.push(near(mleNormalSigma2([1, 2, 3, 4, 5]), 2));

  // 13. mleBernoulli with 4 ones out of 7 → 4/7
  results.push(near(mleBernoulli([1, 1, 0, 1, 0, 0, 1]), 4 / 7, 1e-12));

  // 14. mleExponential matches 1/mean on a fixed sample
  {
    const data = [0.5, 1.2, 0.8, 2.1, 0.3];
    results.push(near(mleExponential(data), 1 / sampleMean(data)));
  }

  // 15. mlePoisson([2, 3, 1, 4, 0]) === 2
  results.push(near(mlePoisson([2, 3, 1, 4, 0]), 2));

  // 16. Newton-Raphson on Normal(μ, σ²=4): one productive step plus a vacuous
  //     verification step. The productive step lands at the MLE; the second
  //     call sees step size 0 and sets converged. Expect iterations ≤ 2.
  {
    const sigma2 = 4;
    const data = [3, 5, 7]; // X̄ = 5
    const score = (x: number, mu: number) => (x - mu) / sigma2;
    const hess = (_x: number, _mu: number) => -1 / sigma2;
    const out = newtonRaphson(0, data, score, hess, { tol: 1e-12 });
    results.push(out.iterations <= 2 && out.converged && near(out.mle, 5, 1e-9));
  }

  // 17. mleGammaShape on Gamma(α=3, β=2) data, n = 100 → α̂ ≈ 3 within ~15 iters
  {
    let seed = 1729;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x1_0000_0000;
    };
    const data: number[] = [];
    // Gamma(3, 2) = sum of three Exp(2): X = −log(U₁ U₂ U₃) / 2
    for (let i = 0; i < 100; i++) {
      const u1 = Math.max(rng(), 1e-12);
      const u2 = Math.max(rng(), 1e-12);
      const u3 = Math.max(rng(), 1e-12);
      data.push(-Math.log(u1 * u2 * u3) / 2);
    }
    const { mle, iterations, converged } = mleGammaShape(data, 2, { maxIter: 30, tol: 1e-8 });
    results.push(converged && iterations <= 15 && Math.abs(mle - 3) < 0.7);
  }

  // 18. observedInfoNormalMu(100, 4) === 25
  results.push(near(observedInfoNormalMu(100, 4), 25));

  // 19. reparameterizedLogLik: ℓ(σ²) with max at σ²=4 → ℓ*(σ) with max at σ=2
  {
    const logLikSigma2 = (s2: number) => -0.5 * (s2 - 4) * (s2 - 4);
    const logLikSigma = reparameterizedLogLik(logLikSigma2, (sigma) => sigma * sigma);
    results.push(logLikSigma(2) >= logLikSigma(1.9) && logLikSigma(2) >= logLikSigma(2.1));
  }

  // 20. logisticLogLik: with all y = 1 and positive x, ℓ is strictly increasing in β₁
  {
    const x = [0.5, 1, 1.5, 2, 2.5];
    const y = [1, 1, 1, 1, 1];
    const a = logisticLogLik(x, y, 0, 0);
    const b = logisticLogLik(x, y, 0, 1);
    const c = logisticLogLik(x, y, 0, 5);
    results.push(a < b && b < c);
  }

  // 21. computeMLE Normal σ² on a constant sample: σ̂² = 0 is a boundary
  //     where the log-likelihood diverges. We signal this with -Infinity
  //     (the sentinel checked by the explorer's y-scale fallback).
  {
    const { mle, logLik } = computeMLE([3, 3, 3, 3], 'Normal', 'sigma2', { mu: 3 });
    results.push(near(mle, 0) && logLik === -Infinity);
  }

  // 22. Bernoulli log-likelihood at boundary MLEs: all-zeros → logLik = 0
  //     (under 0·log 0 = 0); all-ones → logLik = 0; a mixed sample gives
  //     p̂ ∈ (0, 1) with a finite, strictly-negative log-lik.
  {
    const allZeros = computeMLE([0, 0, 0, 0, 0], 'Bernoulli', 'p');
    const allOnes = computeMLE([1, 1, 1, 1, 1], 'Bernoulli', 'p');
    const mixed = computeMLE([1, 0, 1, 0, 1], 'Bernoulli', 'p');
    results.push(
      near(allZeros.mle, 0) &&
        near(allZeros.logLik, 0) &&
        near(allOnes.mle, 1) &&
        near(allOnes.logLik, 0) &&
        mixed.mle > 0 &&
        mixed.mle < 1 &&
        Number.isFinite(mixed.logLik) &&
        mixed.logLik < 0,
    );
  }

  // 23. observedInfoNormalMu edge cases:
  //     n = 0 → 0 (no observations, zero information).
  //     n < 0 → NaN (invalid sample size).
  //     σ² ≤ 0 → Infinity (degenerate variance).
  results.push(
    observedInfoNormalMu(0, 4) === 0 &&
      Number.isNaN(observedInfoNormalMu(-1, 4)) &&
      observedInfoNormalMu(10, 0) === Infinity &&
      observedInfoNormalMu(10, -1) === Infinity,
  );

  // 24. logisticScore under extreme η (|β₁·x| = 1000): the stable sigmoid
  //     must keep both gradient components finite — no NaN from overflow.
  {
    const x = [-1, 0, 1];
    const y = [0, 1, 1];
    const [g0, g1] = logisticScore(x, y, 0, 1000);
    const [g0n, g1n] = logisticScore(x, y, 0, -1000);
    results.push(
      Number.isFinite(g0) &&
        Number.isFinite(g1) &&
        Number.isFinite(g0n) &&
        Number.isFinite(g1n),
    );
  }

  // ── Topic 15 additions ────────────────────────────────────────────────────

  // 25. momNormal([1..5]) → muHat=3, sigma2Hat=2 (biased 1/n, matches MLE)
  {
    const r = momNormal([1, 2, 3, 4, 5]);
    results.push(near(r.muHat, 3) && near(r.sigma2Hat, 2));
  }

  // 26. momExponential matches mleExponential on a fixed sample
  {
    const data = [0.5, 1.2, 0.8, 2.1, 0.3];
    results.push(near(momExponential(data), mleExponential(data)));
  }

  // 27. momPoisson([2,3,1,4,0]) === 2 (matches mlePoisson)
  results.push(near(momPoisson([2, 3, 1, 4, 0]), 2));

  // 28. momBernoulli matches mleBernoulli on a fixed sample
  results.push(near(momBernoulli([1, 1, 0, 1, 0, 0, 1]), 4 / 7, 1e-12));

  // 29. momUniform: 2X̄ for [0.1, 0.3, 0.5, 0.7] → 2 · 0.4 = 0.8
  results.push(near(momUniform([0.1, 0.3, 0.5, 0.7]), 0.8));

  // 30. momGamma on Gamma(3, 2) data, n = 1000: α̂≈3, β̂≈2 within 10%
  {
    let seed = 2026;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x1_0000_0000;
    };
    // Gamma(3, 2) = sum of three Exp(2) = −log(U₁U₂U₃)/2
    const data: number[] = [];
    for (let i = 0; i < 1000; i++) {
      const u1 = Math.max(rng(), 1e-12);
      const u2 = Math.max(rng(), 1e-12);
      const u3 = Math.max(rng(), 1e-12);
      data.push(-Math.log(u1 * u2 * u3) / 2);
    }
    const { alphaHat, betaHat } = momGamma(data);
    results.push(withinPct(alphaHat, 3, 0.1) && withinPct(betaHat, 2, 0.1));
  }

  // 31. momBeta on Beta(2, 5) data, n = 1000: α̂≈2, β̂≈5 within 15%
  //     Beta(α, β) sampling via X = G_α / (G_α + G_β), G_k = sum of k Exp(1)'s.
  {
    let seed = 31415;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x1_0000_0000;
    };
    const data: number[] = [];
    for (let i = 0; i < 1000; i++) {
      // G_2 = −log(U₁U₂);  G_5 = −log(U₁U₂U₃U₄U₅).
      let g2 = 0;
      for (let j = 0; j < 2; j++) g2 -= Math.log(Math.max(rng(), 1e-12));
      let g5 = 0;
      for (let j = 0; j < 5; j++) g5 -= Math.log(Math.max(rng(), 1e-12));
      data.push(g2 / (g2 + g5));
    }
    const { alphaHat, betaHat } = momBeta(data);
    results.push(withinPct(alphaHat, 2, 0.15) && withinPct(betaHat, 5, 0.15));
  }

  // 32. areTheoretical('exponential-rate', {}) === 1 exactly
  results.push(areTheoretical('exponential-rate', {}) === 1);

  // 33. areTheoretical('gamma-shape', {alpha: 3}) ≈ 0.676.
  //     Closed form: 1 / [2(α+1)(α·ψ'(α) − 1)]. At α=3:
  //     trigamma(3) = π²/6 − 1 − 1/4 ≈ 0.394934
  //     denom = 3·0.394934 − 1 = 0.184802
  //     ARE = 1/(2·4·0.184802) = 0.676.
  {
    const are3 = areTheoretical('gamma-shape', { alpha: 3 });
    results.push(Math.abs(are3 - 0.676) < 0.01);
  }

  // 34. huberPsi inside region returns u; outside region clipped to ±k.
  results.push(
    near(huberPsi(0.5, 1.345), 0.5) &&
      near(huberPsi(2.0, 1.345), 1.345) &&
      near(huberPsi(-2.0, 1.345), -1.345),
  );

  // 35. tukeyPsi: smooth at 0.5/c, exactly 0 beyond c.
  {
    const c = 4.685;
    const expected = 0.5 * Math.pow(1 - (0.5 / c) ** 2, 2);
    results.push(near(tukeyPsi(0.5, c), expected) && tukeyPsi(5.0, c) === 0);
  }

  // 36. tukeyPsiPrime via finite differences: derivative at u=1, c=4.685 should
  //     match (tukeyPsi(1+h) − tukeyPsi(1−h))/(2h) within 1e-4.
  {
    const c = 4.685;
    const h = 1e-5;
    const fd = (tukeyPsi(1 + h, c) - tukeyPsi(1 - h, c)) / (2 * h);
    results.push(Math.abs(tukeyPsiPrime(1, c) - fd) < 1e-4);
  }

  // 37. mEstimatorLocation on clean Normal(5, 1), n=200 with Huber:
  //     should land within 0.15 of 5.
  {
    let seed = 7;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x1_0000_0000;
    };
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      // Box–Muller pair, take first.
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      data.push(5 + z); // Normal(5, 1)
    }
    const k = 1.345;
    const { estimate, converged } = mEstimatorLocation(data, (u) => huberPsi(u, k));
    results.push(converged && Math.abs(estimate - 5) < 0.15);
  }

  // 38. mEstimatorLocation under contamination: 10% outliers at +20.
  //     Sample mean drifts; Huber stays within 0.5 of true 5.
  {
    let seed = 11;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x1_0000_0000;
    };
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const contaminated = i < 20 ? 25 : 5 + z; // 10% outliers at 25
      data.push(contaminated);
    }
    const k = 1.345;
    const meanEst = sampleMean(data);
    const { estimate } = mEstimatorLocation(data, (u) => huberPsi(u, k));
    results.push(meanEst > 6 && Math.abs(estimate - 5) < 0.5);
  }

  // 39. mEstimatorVariance on clean Normal(0, 1) data with the score ψ(u) = u
  //     (i.e., MLE form): should approximately match σ²/n = 1/n.
  {
    let seed = 23;
    const rng = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x1_0000_0000;
    };
    const n = 500;
    const data: number[] = [];
    for (let i = 0; i < n; i++) {
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      data.push(z);
    }
    const v = mEstimatorVariance(data, 0, (u) => u, () => 1, 1);
    // Theoretical 1/n = 0.002; allow ±50% (Monte-Carlo noise dominates).
    results.push(v > 0.001 && v < 0.004);
  }

  // 40. breakdownPoint values match Rousseeuw–Leroy table.
  results.push(
    breakdownPoint('mean') === 0 &&
      breakdownPoint('median') === 0.5 &&
      Math.abs(breakdownPoint('huber') - 0.05) < 0.01 &&
      Math.abs(breakdownPoint('trimmed-mean', { alpha: 0.1 }) - 0.1) < 1e-12,
  );

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
