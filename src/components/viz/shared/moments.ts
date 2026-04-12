/**
 * moments.ts — Moment computation module.
 *
 * Pure, deterministic functions for expectation, variance, covariance,
 * correlation, moment-generating functions, probability bounds, and
 * conditional expectation. Separate from distributions.ts (which handles
 * PMF/PDF/CDF evaluation) — this module takes distributions and returns
 * summary statistics.
 *
 * Topic 4: Expectation, Variance & Moments (Track 1 capstone).
 * Used by ExpectationBalanceExplorer, VarianceDecompositionExplorer,
 * InequalityExplorer, LawOfTotalExpectationExplorer, and MGFExplorer.
 */

import { trapezoidalIntegral } from './distributions';

// ── Types ───────────────────────────────────────────────────────────────────

/** A discrete distribution as arrays of values and probabilities. */
export interface DiscreteDistribution {
  values: number[];
  probabilities: number[];
}

/** A continuous distribution defined by its PDF on a support. */
export interface ContinuousDistribution {
  pdf: (x: number) => number;
  support: [number, number]; // [lower, upper], use large finite bounds for ±∞
}

// ── Expectation ─────────────────────────────────────────────────────────────

/**
 * E[X] for a discrete distribution.
 * Σ xᵢ · P(X = xᵢ)
 */
export function expectationDiscrete(dist: DiscreteDistribution): number {
  let sum = 0;
  for (let i = 0; i < dist.values.length; i++) {
    sum += dist.values[i] * dist.probabilities[i];
  }
  return sum;
}

/**
 * E[g(X)] for a discrete distribution (LOTUS).
 * Σ g(xᵢ) · P(X = xᵢ)
 */
export function expectationDiscreteG(
  dist: DiscreteDistribution,
  g: (x: number) => number,
): number {
  let sum = 0;
  for (let i = 0; i < dist.values.length; i++) {
    sum += g(dist.values[i]) * dist.probabilities[i];
  }
  return sum;
}

/**
 * E[X] for a continuous distribution (numerical integration).
 * ∫ x · f(x) dx over the support.
 */
export function expectationContinuous(
  dist: ContinuousDistribution,
  n: number = 1000,
): number {
  return trapezoidalIntegral(
    (x) => x * dist.pdf(x),
    dist.support[0],
    dist.support[1],
    n,
  );
}

/**
 * E[g(X)] for a continuous distribution (LOTUS, numerical).
 * ∫ g(x) · f(x) dx over the support.
 */
export function expectationContinuousG(
  dist: ContinuousDistribution,
  g: (x: number) => number,
  n: number = 1000,
): number {
  return trapezoidalIntegral(
    (x) => g(x) * dist.pdf(x),
    dist.support[0],
    dist.support[1],
    n,
  );
}

// ── Variance ────────────────────────────────────────────────────────────────

/**
 * Var(X) for a discrete distribution.
 * E[X²] − (E[X])²
 */
export function varianceDiscrete(dist: DiscreteDistribution): number {
  const mean = expectationDiscrete(dist);
  const meanSq = expectationDiscreteG(dist, (x) => x * x);
  return meanSq - mean * mean;
}

/**
 * Var(X) for a continuous distribution (numerical).
 * E[X²] − (E[X])²
 */
export function varianceContinuous(
  dist: ContinuousDistribution,
  n: number = 1000,
): number {
  const mean = expectationContinuous(dist, n);
  const meanSq = expectationContinuousG(dist, (x) => x * x, n);
  return meanSq - mean * mean;
}

/** Standard deviation: σ = √Var(X). */
export function stdDev(variance: number): number {
  return Math.sqrt(Math.max(0, variance));
}

// ── Covariance & Correlation ────────────────────────────────────────────────

/**
 * Cov(X, Y) from arrays of paired observations.
 * (1/n) Σ (xᵢ − x̄)(yᵢ − ȳ)
 */
export function covariance(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  let cov = 0;
  for (let i = 0; i < n; i++) {
    cov += (x[i] - meanX) * (y[i] - meanY);
  }
  return cov / n;
}

/**
 * Pearson correlation coefficient ρ(X, Y) from paired observations.
 * Cov(X,Y) / (σ_X · σ_Y)
 */
export function correlation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;

  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }

  const denom = Math.sqrt(varX * varY);
  if (denom === 0) return 0;
  return cov / denom;
}

// ── Moment-Generating Functions ─────────────────────────────────────────────

/** MGF for Bernoulli(p): M(t) = (1−p) + p·eᵗ */
export function mgfBernoulli(t: number, p: number): number {
  return (1 - p) + p * Math.exp(t);
}

/** MGF for Binomial(n,p): M(t) = ((1−p) + p·eᵗ)ⁿ */
export function mgfBinomial(t: number, n: number, p: number): number {
  return Math.pow((1 - p) + p * Math.exp(t), n);
}

/** MGF for Normal(μ, σ²): M(t) = exp(μt + σ²t²/2) */
export function mgfNormal(t: number, mu: number, sigma2: number): number {
  return Math.exp(mu * t + sigma2 * t * t / 2);
}

/** MGF for Exponential(λ): M(t) = λ/(λ−t) for t < λ. Returns null if t ≥ λ. */
export function mgfExponential(t: number, lambda: number): number | null {
  if (t >= lambda) return null;
  return lambda / (lambda - t);
}

/** MGF for Poisson(λ): M(t) = exp(λ(eᵗ − 1)) */
export function mgfPoisson(t: number, lambda: number): number {
  return Math.exp(lambda * (Math.exp(t) - 1));
}

/**
 * Numerical nth derivative of an MGF at t=0.
 * Returns E[Xⁿ] = M⁽ⁿ⁾(0).
 *
 * Uses central finite differences for accuracy.
 */
export function mgfMoment(
  mgf: (t: number) => number,
  n: number,
  h: number = 1e-5,
): number {
  if (n === 0) return mgf(0);
  if (n === 1) {
    // Central difference: (M(h) − M(−h)) / (2h)
    return (mgf(h) - mgf(-h)) / (2 * h);
  }
  if (n === 2) {
    // Central difference: (M(h) − 2M(0) + M(−h)) / h²
    return (mgf(h) - 2 * mgf(0) + mgf(-h)) / (h * h);
  }
  // General case: recursive central difference
  const halfStep = h / 2;
  const fPlus = (t: number) => mgf(t + halfStep);
  const fMinus = (t: number) => mgf(t - halfStep);
  return (mgfMoment(fPlus, n - 1, h) - mgfMoment(fMinus, n - 1, h)) / h;
}

// ── Inequalities ────────────────────────────────────────────────────────────

/**
 * Markov bound: P(X ≥ a) ≤ E[X]/a for X ≥ 0, a > 0.
 * Returns the upper bound (clamped to [0, 1]).
 */
export function markovBound(expectation: number, a: number): number {
  if (a <= 0 || expectation < 0) return 1;
  return Math.min(1, expectation / a);
}

/**
 * Chebyshev bound: P(|X − μ| ≥ ε) ≤ Var(X)/ε² for ε > 0.
 * Returns the upper bound (clamped to [0, 1]).
 */
export function chebyshevBound(variance: number, epsilon: number): number {
  if (epsilon <= 0) return 1;
  return Math.min(1, variance / (epsilon * epsilon));
}

// ── Conditional Expectation ─────────────────────────────────────────────────

/**
 * Bivariate normal conditional expectation: E[X|Y=y].
 * E[X|Y=y] = μ_X + ρ(σ_X/σ_Y)(y − μ_Y)
 */
export function conditionalExpectationBVN(
  muX: number,
  muY: number,
  sigmaX: number,
  sigmaY: number,
  rho: number,
  y: number,
): number {
  return muX + rho * (sigmaX / sigmaY) * (y - muY);
}

/**
 * Bivariate normal conditional variance: Var(X|Y=y).
 * Var(X|Y=y) = σ_X²(1 − ρ²)
 * (Does not depend on y — homoscedastic.)
 */
export function conditionalVarianceBVN(
  sigmaX: number,
  rho: number,
): number {
  return sigmaX * sigmaX * (1 - rho * rho);
}

// ── Re-export numerical integration ─────────────────────────────────────────

export { trapezoidalIntegral } from './distributions';
