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
function digamma(x: number): number {
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
function trigamma(x: number): number {
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

/** Lanczos log-Γ, accurate to ~1e-10 on (0, ∞). Internal helper for Gamma log-lik. */
function logGamma(x: number): number {
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
        if (p <= 0 || p >= 1) return -Infinity;
        return x * Math.log(p) + (1 - x) * Math.log(1 - p);
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
        let logFact = 0;
        for (let j = 2; j <= x; j++) logFact += Math.log(j);
        return x * Math.log(lam) - lam - logFact;
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

/** J(μ) for Normal(μ, σ²): n/σ². Non-random (doesn't depend on data values). */
export function observedInfoNormalMu(n: number, sigma2: number): number {
  if (sigma2 <= 0 || n <= 0) return Infinity;
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
 * Gradient of the logistic log-likelihood with respect to (β₀, β₁):
 *   ∂ℓ/∂β₀ = Σᵢ (yᵢ − pᵢ),   ∂ℓ/∂β₁ = Σᵢ xᵢ (yᵢ − pᵢ),   pᵢ = σ(β₀ + β₁ xᵢ).
 * Topic 21 develops the full IRLS / Fisher-scoring algorithm; Topic 14 uses
 * this score for a single Newton step in the preview example.
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
    const p = 1 / (1 + Math.exp(-eta));
    const r = y[i] - p;
    g0 += r;
    g1 += x[i] * r;
  }
  return [g0, g1];
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

  // 16. Newton-Raphson on Normal(μ, σ²=4): one step converges (quadratic log-lik)
  {
    const sigma2 = 4;
    const data = [3, 5, 7]; // X̄ = 5
    const score = (x: number, mu: number) => (x - mu) / sigma2;
    const hess = (_x: number, _mu: number) => -1 / sigma2;
    const out = newtonRaphson(0, data, score, hess, { tol: 1e-12 });
    results.push(out.iterations <= 1 && near(out.mle, 5, 1e-9));
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
