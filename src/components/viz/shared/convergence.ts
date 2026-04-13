/**
 * convergence.ts — Sampling, running statistics, and convergence diagnostics.
 *
 * A new shared module (not an extension of distributions.ts) providing
 * utilities for convergence visualization and analysis. Distributions.ts
 * handles static PMF/PDF/CDF evaluation; this module handles dynamic
 * sequences and empirical statistics.
 *
 * Topic 9:  Sample path generation, running mean/variance, empirical CDF,
 *           KS statistic, total variation distance, typewriter sequence,
 *           Box–Muller normal sampling.
 * Topics 10–12 will extend with: bootstrapSample, cltNormalization,
 *           confidenceIntervalCoverage, concentrationBounds.
 */

// ── Sampling Utilities ────────────────────────────────────────────────────────

/**
 * Generate n iid samples from a distribution.
 * @param sampler — function that returns a single sample
 * @param n — number of samples
 */
export function sampleSequence(sampler: () => number, n: number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = sampler();
  return out;
}

/**
 * Compute the sample mean of an array.
 */
export function sampleMean(samples: number[]): number {
  if (samples.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < samples.length; i++) s += samples[i];
  return s / samples.length;
}

/**
 * Compute the running mean: X̄₁, X̄₂, ..., X̄ₙ.
 * runningMean[k] = (1/(k+1)) Σᵢ₌₀ᵏ samples[i].
 * Uses an incremental formula to avoid catastrophic cancellation.
 */
export function runningMean(samples: number[]): number[] {
  const n = samples.length;
  const out = new Array<number>(n);
  let mean = 0;
  for (let i = 0; i < n; i++) {
    mean += (samples[i] - mean) / (i + 1);
    out[i] = mean;
  }
  return out;
}

/**
 * Compute the running variance: S²₁, S²₂, ..., S²ₙ.
 * Uses Welford's online algorithm for numerical stability.
 * Returns the population variance (divided by n, not n-1).
 */
export function runningVariance(samples: number[]): number[] {
  const n = samples.length;
  const out = new Array<number>(n);
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < n; i++) {
    const delta = samples[i] - mean;
    mean += delta / (i + 1);
    const delta2 = samples[i] - mean;
    m2 += delta * delta2;
    out[i] = i === 0 ? 0 : m2 / (i + 1);
  }
  return out;
}

// ── Empirical CDF ─────────────────────────────────────────────────────────────

/**
 * Evaluate the empirical CDF: F̂ₙ(x) = (1/n) Σ 1(Xᵢ ≤ x).
 */
export function empiricalCDF(samples: number[], x: number): number {
  let count = 0;
  for (let i = 0; i < samples.length; i++) {
    if (samples[i] <= x) count++;
  }
  return count / samples.length;
}

/**
 * Compute the empirical CDF as a step function.
 * Returns sorted {x, Fn} pairs suitable for D3 curveStepAfter plotting.
 * Includes a leading point at (min - offset, 0) for the left tail.
 */
export function empiricalCDFPoints(
  samples: number[]
): Array<{ x: number; Fn: number }> {
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const points: Array<{ x: number; Fn: number }> = [];

  // Leading zero point
  const range = sorted[n - 1] - sorted[0];
  points.push({ x: sorted[0] - Math.max(0.5, range * 0.05), Fn: 0 });

  for (let i = 0; i < n; i++) {
    points.push({ x: sorted[i], Fn: (i + 1) / n });
  }

  return points;
}

// ── Convergence Diagnostics ──────────────────────────────────────────────────

/**
 * Total variation distance between two discrete PMFs.
 * TV(P, Q) = (1/2) Σ |P(k) − Q(k)|.
 */
export function totalVariationDistance(
  pmf1: Map<number, number>,
  pmf2: Map<number, number>
): number {
  const allKeys = new Set([...pmf1.keys(), ...pmf2.keys()]);
  let sum = 0;
  for (const k of allKeys) {
    sum += Math.abs((pmf1.get(k) ?? 0) - (pmf2.get(k) ?? 0));
  }
  return sum / 2;
}

/**
 * Kolmogorov–Smirnov statistic: sup|F̂ₙ(x) − F(x)|.
 * Evaluates at each sample point (both left and right limits of the ECDF).
 */
export function kolmogorovSmirnovStat(
  samples: number[],
  cdf: (x: number) => number
): number {
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  let maxDiff = 0;

  for (let i = 0; i < n; i++) {
    const Fx = cdf(sorted[i]);
    // ECDF jumps from i/n to (i+1)/n at sorted[i]
    const ecdfRight = (i + 1) / n;
    const ecdfLeft = i / n;
    maxDiff = Math.max(maxDiff, Math.abs(ecdfRight - Fx), Math.abs(ecdfLeft - Fx));
  }

  return maxDiff;
}

/**
 * Estimate P(|Xₙ − X| > ε) from M Monte Carlo replications.
 * @param generateXn — function returning one realization of Xₙ
 * @param target — the limit value X (constant for most examples)
 * @param epsilon — tolerance
 * @param M — number of replications
 */
export function empiricalDeviationProb(
  generateXn: () => number,
  target: number,
  epsilon: number,
  M: number
): number {
  let count = 0;
  for (let i = 0; i < M; i++) {
    if (Math.abs(generateXn() - target) > epsilon) count++;
  }
  return count / M;
}

// ── Counterexample Sequence Generators ───────────────────────────────────────

/**
 * Typewriter interval endpoints.
 *
 * The typewriter (or "sliding indicator") sequence partitions [0,1] into
 * rows of increasing granularity:
 *   Row 1: 1 interval  of length 1   — [0,1]
 *   Row 2: 2 intervals of length 1/2 — [0,1/2], [1/2,1]
 *   Row 3: 3 intervals of length 1/3 — [0,1/3], [1/3,2/3], [2/3,1]
 *   ...
 *
 * Row k starts at 1-based index: 1 + k(k-1)/2.
 *
 * @param n — 1-based term index
 * @returns {a, b} endpoints of the interval for term n
 */
export function typewriterInterval(n: number): { a: number; b: number } {
  // Find the row k: solve 1 + k(k-1)/2 <= n, i.e. k(k-1)/2 < n
  // k = ceil((-1 + sqrt(1 + 8*(n-1))) / 2) but we clamp to avoid float issues
  const k = Math.ceil((-1 + Math.sqrt(1 + 8 * (n - 1))) / 2);
  // Offset within row k (0-based)
  const rowStart = 1 + (k * (k - 1)) / 2;
  const j = n - rowStart;
  return { a: j / k, b: (j + 1) / k };

  // Verification (first 10 terms):
  // n=1: k=1, j=0 → [0, 1]
  // n=2: k=2, j=0 → [0, 0.5]
  // n=3: k=2, j=1 → [0.5, 1]
  // n=4: k=3, j=0 → [0, 1/3]
  // n=5: k=3, j=1 → [1/3, 2/3]
  // n=6: k=3, j=2 → [2/3, 1]
  // n=7: k=4, j=0 → [0, 0.25]
  // n=8: k=4, j=1 → [0.25, 0.5]
  // n=9: k=4, j=2 → [0.5, 0.75]
  // n=10: k=4, j=3 → [0.75, 1]
}

/**
 * Generate the typewriter sequence for a fixed u ∈ [0,1].
 * Returns array of 0s and 1s: Xₙ(u) = 1 iff u ∈ [aₙ, bₙ].
 */
export function typewriterSequence(u: number, N: number): number[] {
  const out = new Array<number>(N);
  for (let n = 1; n <= N; n++) {
    const { a, b } = typewriterInterval(n);
    out[n - 1] = u >= a && u < b ? 1 : 0;
  }
  return out;
}

// ── Random Number Generation ────────────────────────────────────────────────

/**
 * Generate a Normal(μ, σ²) sample using the Box–Muller transform.
 * @param mu — mean (default 0)
 * @param sigma — standard deviation (default 1)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function normalSample(
  mu: number = 0,
  sigma: number = 1,
  rng: () => number = Math.random
): number {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
}

/**
 * Generate an Exponential(λ) sample via inverse CDF.
 * @param lambda — rate parameter (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function exponentialSample(
  lambda: number,
  rng: () => number = Math.random
): number {
  return -Math.log(1 - rng()) / lambda;
}

/**
 * Generate a Poisson(λ) sample using the Knuth algorithm.
 * Efficient for λ ≤ 30; for larger λ a normal approximation could be used.
 * @param lambda — rate parameter (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function poissonSample(
  lambda: number,
  rng: () => number = Math.random
): number {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rng();
  } while (p > L);
  return k - 1;
}

/**
 * Generate a Uniform(a, b) sample.
 */
export function uniformSample(
  a: number,
  b: number,
  rng: () => number = Math.random
): number {
  return a + (b - a) * rng();
}

/**
 * Generate a Bernoulli(p) sample.
 */
export function bernoulliSample(
  p: number,
  rng: () => number = Math.random
): number {
  return rng() < p ? 1 : 0;
}
