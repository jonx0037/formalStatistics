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
 * Topic 10: Gamma/Student-t/Pareto sampling, bootstrapSample,
 *           Chebyshev/Hoeffding bounds, DKW bound, running KS,
 *           law of iterated logarithm envelope.
 * Topic 11: CLT normalization and replications, empirical CF,
 *           Berry–Esseen bound, chi-squared and beta samplers.
 * Topic 12: Bernstein, sub-Gaussian, Chernoff, and McDiarmid bounds;
 *           Cramér rate function; KL divergence (generic + Bernoulli);
 *           Johnson–Lindenstrauss target dimension; sample-size
 *           requirements under each bound family.
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
  // k = ceil((-1 + sqrt(1 + 8*n)) / 2), with max(1, ...) to handle n=1
  const k = Math.max(1, Math.ceil((-1 + Math.sqrt(1 + 8 * n)) / 2));
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
  const u1 = Math.max(rng(), Number.EPSILON); // clamp to avoid log(0)
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
 * Generate a Poisson(λ) sample.
 * Uses Knuth's algorithm for λ ≤ 30 and a Normal approximation for λ > 30
 * (where Knuth underflows because Math.exp(-λ) → 0 for λ > ~745).
 * @param lambda — rate parameter (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function poissonSample(
  lambda: number,
  rng: () => number = Math.random
): number {
  if (lambda > 30) {
    // Normal approximation: Poisson(λ) ≈ round(N(λ, λ))
    return Math.max(0, Math.round(normalSample(lambda, Math.sqrt(lambda), rng)));
  }
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

// ── Topic 10: Law of Large Numbers Extensions ──────────────────────────────

// ── Additional Sampling Functions ──────────────────────────────────────────

/**
 * Generate a Gamma(α, β) sample using the Marsaglia–Tsang method.
 * β is the rate parameter: mean = α/β, variance = α/β².
 * Works for any α > 0 (uses rejection for α < 1).
 * @param alpha — shape parameter (> 0)
 * @param beta — rate parameter (> 0, default 1)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function gammaSample(
  alpha: number,
  beta: number = 1,
  rng: () => number = Math.random
): number {
  // For α < 1, use the identity: Gamma(α, 1) = Gamma(α+1, 1) · U^(1/α); then scale by 1/β.
  if (alpha < 1) {
    const u = Math.max(rng(), Number.EPSILON);
    return gammaSample(alpha + 1, 1, rng) * Math.pow(u, 1 / alpha) / beta;
  }
  // Marsaglia–Tsang method for α ≥ 1 (returns Gamma(α, 1), then scale by 1/β).
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number;
    let v: number;
    do {
      x = normalSample(0, 1, rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return (d * v) / beta;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return (d * v) / beta;
  }
}

/**
 * Generate a Student-t(ν) sample using the ratio method.
 * t = Z / √(V/ν) where Z ~ N(0,1) and V ~ χ²(ν) ~ Gamma(ν/2, 2).
 * Supports non-integer ν (e.g., ν = 1.5) via Gamma sampling.
 * @param nu — degrees of freedom (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function tSample(
  nu: number,
  rng: () => number = Math.random
): number {
  const z = normalSample(0, 1, rng);
  // χ²(ν) = Gamma(ν/2, 1/2), so 2 · Gamma(ν/2, 1) works too.
  const v = 2 * gammaSample(nu / 2, 1, rng);
  return z / Math.sqrt(v / nu);
}

/**
 * Generate a Pareto(α, xₘ=1) sample via inverse CDF.
 * X = 1 / U^{1/α} where U ~ Uniform(0,1).
 * @param alpha — shape parameter (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function paretoSample(
  alpha: number,
  rng: () => number = Math.random
): number {
  const u = Math.max(rng(), Number.EPSILON);
  return 1 / Math.pow(u, 1 / alpha);
}

/**
 * Generate a bootstrap sample: n draws with replacement from data.
 * @param data — original sample array
 * @param rng — uniform [0,1) generator (default Math.random)
 * @returns array of length data.length, drawn with replacement
 */
export function bootstrapSample(
  data: number[],
  rng: () => number = Math.random
): number[] {
  const n = data.length;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = data[Math.floor(rng() * n)];
  }
  return out;
}

// ── Concentration Bounds ───────────────────────────────────────────────────

/**
 * Chebyshev bound: P(|X̄ₙ − μ| > ε) ≤ σ²/(nε²).
 * Clamped to [0, 1].
 */
export function chebyshevBound(
  n: number,
  sigmaSquared: number,
  epsilon: number
): number {
  return Math.min(sigmaSquared / (n * epsilon * epsilon), 1);
}

/**
 * Hoeffding bound: P(|X̄ₙ − μ| > ε) ≤ 2 exp(−2nε²/(b−a)²).
 * For bounded random variables X ∈ [a, b].
 */
export function hoeffdingBound(
  n: number,
  a: number,
  b: number,
  epsilon: number
): number {
  const range = b - a;
  return Math.min(2 * Math.exp(-2 * n * epsilon * epsilon / (range * range)), 1);
}

// ── Glivenko–Cantelli Utilities ────────────────────────────────────────────

/**
 * DKW confidence band half-width: √(ln(2/α) / (2n)).
 * The Dvoretzky–Kiefer–Wolfowitz inequality gives:
 *   P(supₓ |Fₙ(x) − F(x)| > ε) ≤ 2 exp(−2nε²).
 * Inverting: the band width at level α is √(ln(2/α)/(2n)).
 */
export function dkwBound(n: number, alpha: number): number {
  return Math.sqrt(Math.log(2 / alpha) / (2 * n));
}

/**
 * Running KS statistic: Dₙ for n = 1, 2, ..., N.
 * Returns array where entry k is the KS statistic using
 * samples[0..k] (i.e., the first k+1 observations).
 * @param samples — all N samples
 * @param cdf — theoretical CDF function
 */
export function runningKSStatistic(
  samples: number[],
  cdf: (x: number) => number
): number[] {
  const N = samples.length;
  const result = new Array<number>(N);
  const running: number[] = [];

  for (let i = 0; i < N; i++) {
    running.push(samples[i]);
    // Sort the running sample to compute KS stat
    const sorted = [...running].sort((a, b) => a - b);
    const n = sorted.length;
    let maxDiff = 0;
    for (let j = 0; j < n; j++) {
      const Fx = cdf(sorted[j]);
      const ecdfRight = (j + 1) / n;
      const ecdfLeft = j / n;
      maxDiff = Math.max(
        maxDiff,
        Math.abs(ecdfRight - Fx),
        Math.abs(ecdfLeft - Fx)
      );
    }
    result[i] = maxDiff;
  }

  return result;
}

/**
 * Law of the iterated logarithm envelope: σ√(2 ln(ln(n)) / n).
 * Gives the precise a.s. oscillation rate of X̄ₙ around μ.
 * For n < 3, ln(ln(n)) is undefined or negative; returns Infinity.
 */
export function lilEnvelope(n: number, sigma: number): number {
  if (n < 3) return Infinity;
  return sigma * Math.sqrt(2 * Math.log(Math.log(n)) / n);
}

// ── Topic 11: Central Limit Theorem Extensions ─────────────────────────────

// ── CLT Normalization ──────────────────────────────────────────────────────

/**
 * Compute the CLT-standardized sample mean: √n(X̄ₙ − μ) / σ.
 * @param samples — array of iid observations
 * @param mu — true population mean
 * @param sigma — true population standard deviation (σ, not σ²)
 * @returns the standardized value √n(X̄ₙ − μ)/σ, or 0 if samples is empty
 *          or sigma is zero (guard against division by zero)
 */
export function cltNormalization(
  samples: number[],
  mu: number,
  sigma: number
): number {
  const n = samples.length;
  if (n === 0 || sigma === 0) return 0;
  const mean = sampleMean(samples);
  return (Math.sqrt(n) * (mean - mu)) / sigma;
}

/**
 * Generate M replications of the CLT-standardized sample mean.
 * For each replication: draw n samples, compute √n(X̄ − μ)/σ.
 * The workhorse for CLTExplorer and BerryEsseenExplorer —
 * avoids intermediate array allocation inside the hot loop so that
 * M = 10,000 at n = 200 stays under ~500 ms in modern browsers.
 * Returns an empty array when n or M is non-positive, or when σ = 0
 * (mirrors the guards in cltNormalization to avoid NaN/Infinity).
 * @param sampler — function that generates one sample per call
 * @param n — sample size per replication
 * @param M — number of replications
 * @param mu — true mean
 * @param sigma — true standard deviation
 */
export function cltReplications(
  sampler: () => number,
  n: number,
  M: number,
  mu: number,
  sigma: number
): number[] {
  if (n <= 0 || M <= 0 || sigma === 0) return [];
  const out = new Array<number>(M);
  const sqrtN = Math.sqrt(n);
  for (let rep = 0; rep < M; rep++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += sampler();
    const mean = sum / n;
    out[rep] = (sqrtN * (mean - mu)) / sigma;
  }
  return out;
}

// ── Characteristic Function Helpers ────────────────────────────────────────

/**
 * Evaluate the empirical characteristic function at point t:
 *   φ̂ₙ(t) = (1/n) Σ exp(i·t·Xⱼ) = (1/n) Σ [cos(t·Xⱼ) + i·sin(t·Xⱼ)].
 * @param samples — array of observations
 * @param t — evaluation point
 * @returns [real part, imaginary part]
 */
export function empiricalCF(
  samples: number[],
  t: number
): [number, number] {
  const n = samples.length;
  if (n === 0) return [0, 0];
  let re = 0;
  let im = 0;
  for (let j = 0; j < n; j++) {
    re += Math.cos(t * samples[j]);
    im += Math.sin(t * samples[j]);
  }
  return [re / n, im / n];
}

/**
 * Berry–Esseen bound: C · ρ / √n, giving an upper bound on
 *   sup_x |Fₙ(x) − Φ(x)|
 * for the standardized sample mean of an iid sequence with finite
 * absolute third moment.
 * @param rho — E[|X − μ|³] / σ³ (absolute third-moment ratio)
 * @param n — sample size
 * @param C — Berry–Esseen constant (default 0.4748, Shevtsova 2011)
 */
export function berryEsseenBound(
  rho: number,
  n: number,
  C: number = 0.4748
): number {
  if (n <= 0) return Infinity;
  return (C * rho) / Math.sqrt(n);
}

// ── Additional Samplers ────────────────────────────────────────────────────

/**
 * Generate a Chi-squared(k) sample as Gamma(k/2, 1/2).
 * Equivalent to the sum of k squared iid standard Normals, but much
 * faster for moderate-to-large k via the Gamma route.
 * @param k — degrees of freedom (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function chiSquaredSample(
  k: number,
  rng: () => number = Math.random
): number {
  return gammaSample(k / 2, 0.5, rng);
}

/**
 * Generate a Beta(a, b) sample via the Gamma ratio:
 *   X / (X + Y), with X ~ Gamma(a, 1), Y ~ Gamma(b, 1).
 * @param a — first shape parameter (> 0)
 * @param b — second shape parameter (> 0)
 * @param rng — uniform [0,1) generator (default Math.random)
 */
export function betaSample(
  a: number,
  b: number,
  rng: () => number = Math.random
): number {
  const x = gammaSample(a, 1, rng);
  const y = gammaSample(b, 1, rng);
  return x / (x + y);
}

// ── Topic 12: Large Deviations & Tail Bounds ───────────────────────────────

/**
 * Bernstein upper bound: P(|X̄ₙ − μ| ≥ ε) ≤ 2·exp(−nε²/(2σ² + 2Mε/3)).
 * Variance-aware refinement of Hoeffding: when σ² ≪ (b−a)²/4 the variance
 * term dominates, giving a much tighter bound than Hoeffding.
 * Two-regime behaviour: sub-Gaussian for small ε (2σ² dominates the
 * denominator), sub-exponential for large ε (2Mε/3 dominates).
 * Clamped to [0, 1].
 * @param n — sample size
 * @param sigmaSquared — per-variable variance σ² = Var(Xᵢ)
 * @param M — a.s. bound on |Xᵢ − μ|
 * @param epsilon — deviation tolerance ε
 */
export function bernsteinBound(
  n: number,
  sigmaSquared: number,
  M: number,
  epsilon: number
): number {
  const denom = 2 * sigmaSquared + (2 * M * epsilon) / 3;
  if (denom <= 0) return 1;
  return Math.min(2 * Math.exp(-(n * epsilon * epsilon) / denom), 1);
}

/**
 * Sub-Gaussian tail bound: P(|X̄ₙ − μ| ≥ ε) ≤ 2·exp(−nε²/(2σ²)).
 * Assumes each Xᵢ is sub-Gaussian with parameter σ — i.e.
 *   E[exp(t(Xᵢ − μ))] ≤ exp(σ²t²/2) for all t ∈ ℝ.
 * For Uniform(0, 1), the sub-Gaussian parameter is (b−a)/2 = 0.5, so
 * this recovers Hoeffding exactly. Clamped to [0, 1].
 * @param n — sample size
 * @param subGaussianParam — sub-Gaussian parameter σ
 * @param epsilon — deviation tolerance ε
 */
export function subGaussianBound(
  n: number,
  subGaussianParam: number,
  epsilon: number
): number {
  if (subGaussianParam <= 0) return 1;
  const denom = 2 * subGaussianParam * subGaussianParam;
  return Math.min(2 * Math.exp(-(n * epsilon * epsilon) / denom), 1);
}

/**
 * Generic Chernoff bound for iid sums: minimize exp(−tnε) · Mₓ(t)ⁿ
 * over t > 0. Returns { bound, optimalT }. Uses golden-section search
 * on the log objective φ(t) = n(−tε + log Mₓ(t)), which is convex in t.
 * @param mgf — per-sample MGF Mₓ(t) as a callable
 * @param epsilon — deviation threshold
 * @param n — sample size (the iid sum)
 * @param tRange — search interval [tMin, tMax] (default [0.001, 10])
 */
export function chernoffBound(
  mgf: (t: number) => number,
  epsilon: number,
  n: number,
  tRange: [number, number] = [0.001, 10]
): { bound: number; optimalT: number } {
  const objective = (t: number): number => {
    const m = mgf(t);
    if (!isFinite(m) || m <= 0) return Infinity;
    return n * (-t * epsilon + Math.log(m));
  };
  const phi = (Math.sqrt(5) - 1) / 2; // golden ratio reciprocal
  let [a, b] = tRange;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  for (let i = 0; i < 100; i++) {
    if (objective(c) < objective(d)) b = d;
    else a = c;
    c = b - phi * (b - a);
    d = a + phi * (b - a);
    if (Math.abs(b - a) < 1e-8) break;
  }
  const optimalT = (a + b) / 2;
  const bound = Math.min(Math.exp(objective(optimalT)), 1);
  return { bound, optimalT };
}

/**
 * Cramér rate function: I(x) = sup_t (t·x − Λ(t)), where Λ(t) = log Mₓ(t)
 * is the cumulant-generating function. Computed numerically via
 * golden-section search over t. For x above the mean, the supremum is
 * attained at t > 0; for x below, at t < 0. The function is convex and
 * non-negative with I(μ) = 0.
 * @param logMgf — Λ(t) = log Mₓ(t) as a callable
 * @param x — point at which to evaluate the rate function
 * @param tRange — search interval (default [−10, 10])
 */
export function cramerRateFunction(
  logMgf: (t: number) => number,
  x: number,
  tRange: [number, number] = [-10, 10]
): { rateValue: number; optimalT: number } {
  // We maximize g(t) = t·x − Λ(t) via golden-section search.
  // Search for the minimum of −g(t).
  const neg = (t: number): number => {
    const lambda = logMgf(t);
    if (!isFinite(lambda)) return Infinity;
    return -(t * x - lambda);
  };
  const phi = (Math.sqrt(5) - 1) / 2;
  let [a, b] = tRange;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  for (let i = 0; i < 100; i++) {
    if (neg(c) < neg(d)) b = d;
    else a = c;
    c = b - phi * (b - a);
    d = a + phi * (b - a);
    if (Math.abs(b - a) < 1e-8) break;
  }
  const optimalT = (a + b) / 2;
  const rateValue = Math.max(0, -neg(optimalT));
  return { rateValue, optimalT };
}

/**
 * McDiarmid bound: for f satisfying the bounded-differences property with
 * constants cᵢ, P(|f(X₁,…,Xₙ) − E[f]| ≥ t) ≤ 2·exp(−2t²/Σcᵢ²).
 * Recovers Hoeffding when f is the empirical mean with cᵢ = (bᵢ − aᵢ)/n.
 * Clamped to [0, 1].
 * @param boundedDifferences — array of cᵢ constants
 * @param t — deviation tolerance
 */
export function mcDiarmidBound(
  boundedDifferences: number[],
  t: number
): number {
  let sumCi2 = 0;
  for (let i = 0; i < boundedDifferences.length; i++) {
    sumCi2 += boundedDifferences[i] * boundedDifferences[i];
  }
  if (sumCi2 <= 0) return 1;
  return Math.min(2 * Math.exp(-(2 * t * t) / sumCi2), 1);
}

/**
 * KL divergence D_KL(P ‖ Q) for discrete distributions on a shared
 * support. Convention: 0·log(0/q) = 0; if pᵢ > 0 and qᵢ = 0, the
 * divergence is +∞. Returns NaN if the two arrays have different lengths.
 * Uses natural logarithm (nats), consistent with the Cramér rate function.
 * @param p — probability array for P
 * @param q — probability array for Q (same length as p)
 */
export function klDivergence(p: number[], q: number[]): number {
  if (p.length !== q.length) return NaN;
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] === 0) continue;
    if (q[i] <= 0) return Infinity;
    kl += p[i] * Math.log(p[i] / q[i]);
  }
  return kl;
}

/**
 * KL divergence between two Bernoulli distributions:
 *   D_KL(Ber(p) ‖ Ber(q)) = p·log(p/q) + (1−p)·log((1−p)/(1−q)).
 * This is the Cramér rate function for Bernoulli(q) evaluated at x = p.
 * Edge cases: 0·log(0/q) = 0; if the numerator is positive while the
 * denominator is 0, returns +∞.
 * @param p — parameter of the first Bernoulli (0 ≤ p ≤ 1)
 * @param q — parameter of the second Bernoulli (0 < q < 1 for finiteness)
 */
export function bernoulliKL(p: number, q: number): number {
  const term1 = p === 0 ? 0 : q <= 0 ? Infinity : p * Math.log(p / q);
  const term2 =
    p === 1 ? 0 : q >= 1 ? Infinity : (1 - p) * Math.log((1 - p) / (1 - q));
  return term1 + term2;
}

/**
 * Johnson–Lindenstrauss target dimension: minimum k for which a random
 * projection from ℝᵈ to ℝᵏ approximately preserves pairwise distances
 * among n points within factor (1 ± ε) with probability ≥ 1 − δ.
 * Formula (sub-Gaussian concentration + union bound over (n choose 2)
 * pairs): k ≥ 8·ln(n / δ̃) / ε², with δ̃ = δ/(n choose 2). We use the
 * conservative textbook bound k ≥ 8·ln(n²/δ) / ε² ≈ 8·(2 ln n − ln δ)/ε².
 * @param n — number of points to embed
 * @param epsilon — distortion tolerance
 * @param delta — failure probability
 */
export function johnsonLindenstraussDim(
  n: number,
  epsilon: number,
  delta: number
): number {
  if (n <= 1 || epsilon <= 0 || delta <= 0) return Infinity;
  return Math.ceil((8 * (2 * Math.log(n) - Math.log(delta))) / (epsilon * epsilon));
}

/**
 * Sample-size requirements under each bound family for achieving
 *   P(|X̄ₙ − μ| ≥ ε) ≤ δ.
 * Returns the minimum n implied by each applicable bound; fields are
 * undefined when the required parameter is not supplied.
 *   • markov:     n ≥ μ/ε       (requires mu; often loose)
 *   • chebyshev:  n ≥ σ²/(ε²δ)
 *   • hoeffding:  n ≥ (b−a)²·ln(2/δ) / (2ε²)
 *   • bernstein:  n ≥ (2σ² + 2Mε/3)·ln(2/δ) / ε²
 *   • subGauss:   n ≥ 2σ_sg²·ln(2/δ) / ε²
 *   • clt:        n ≥ (z_{δ/2}·σ/ε)²  (approximate, using z = Φ⁻¹(1−δ/2))
 * @param epsilon — accuracy tolerance
 * @param delta — failure probability
 * @param params — distribution parameters (any subset)
 */
export function sampleSizeRequirements(
  epsilon: number,
  delta: number,
  params: {
    mu?: number;
    sigma2?: number;
    range?: [number, number];
    M?: number;
    subGaussianParam?: number;
    zScore?: number;
  }
): {
  markov?: number;
  chebyshev?: number;
  hoeffding?: number;
  bernstein?: number;
  subGaussian?: number;
  clt?: number;
} {
  const out: Record<string, number | undefined> = {};
  const lnFactor = Math.log(2 / delta);
  const eps2 = epsilon * epsilon;

  if (params.mu !== undefined && epsilon > 0) {
    out.markov = Math.ceil(params.mu / epsilon);
  }
  if (params.sigma2 !== undefined && eps2 > 0) {
    out.chebyshev = Math.ceil(params.sigma2 / (eps2 * delta));
  }
  if (params.range !== undefined) {
    const r = params.range[1] - params.range[0];
    out.hoeffding = Math.ceil((r * r * lnFactor) / (2 * eps2));
  }
  if (
    params.sigma2 !== undefined &&
    params.M !== undefined &&
    eps2 > 0
  ) {
    out.bernstein = Math.ceil(
      ((2 * params.sigma2 + (2 * params.M * epsilon) / 3) * lnFactor) / eps2
    );
  }
  if (params.subGaussianParam !== undefined && eps2 > 0) {
    const sg = params.subGaussianParam;
    out.subGaussian = Math.ceil((2 * sg * sg * lnFactor) / eps2);
  }
  if (params.sigma2 !== undefined && params.zScore !== undefined) {
    const z = params.zScore;
    out.clt = Math.ceil((z * z * params.sigma2) / eps2);
  }
  return out;
}
