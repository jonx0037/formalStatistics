/**
 * distributions.ts — Distribution function evaluation module.
 *
 * Pure, deterministic functions for PMF/PDF/CDF/quantile evaluation
 * used across interactive components in Topics 3+. Separate from
 * probability.ts (which handles set operations, conditional probability,
 * and independence from Topics 1–2).
 *
 * Topic 3: Bernoulli, Binomial, Geometric, Poisson PMFs;
 *          Uniform, Normal, Exponential PDFs; all CDFs and quantiles;
 *          bivariate normal; conditional normal; numerical integration.
 * Topic 5: Negative Binomial, Hypergeometric, Discrete Uniform PMFs/CDFs;
 *          MGFs for Geometric/NegBin/DiscreteUniform; PGFs for all six;
 *          closed-form moment functions for all seven discrete distributions.
 * Topic 6: Gamma, Beta, Chi-squared, Student's t, F PDFs/CDFs;
 *          Beta quantile via bisection; continuous MGFs (Uniform, Normal,
 *          Exponential, Gamma, Chi-squared); closed-form moment functions
 *          for all eight continuous distributions.
 */

// ── Types ───────────────────────────────────────────────────────────────────

/** Parameters for a distribution. */
export interface DistributionParams {
  [key: string]: number;
}

/** A distribution descriptor. */
export interface Distribution {
  name: string;
  type: 'discrete' | 'continuous';
  params: DistributionParams;
  support: [number, number]; // [min, max] (Infinity allowed)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Error function approximation (Abramowitz & Stegun, formula 7.1.26).
 * Maximum error: |ε(x)| ≤ 1.5 × 10⁻⁷.
 */
export function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  const a = Math.abs(x);

  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;

  const t = 1 / (1 + p * a);
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t3 * t;
  const t5 = t4 * t;

  const y = 1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.exp(-a * a);
  return sign * y;
}

/**
 * Log-gamma function via Lanczos approximation.
 * Accurate for x > 0.
 */
export function lnGamma(x: number): number {
  if (x <= 0) return Infinity;
  if (x < 0.5) {
    // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  }

  x -= 1;
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

  let sum = c[0];
  for (let i = 1; i < g + 2; i++) {
    sum += c[i] / (x + i);
  }

  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * Binomial coefficient C(n, k) via log-gamma.
 * Numerically stable for large n.
 */
export function binomialCoeff(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  return Math.round(Math.exp(lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1)));
}

/**
 * Log of binomial coefficient: ln C(n, k) = lnΓ(n+1) − lnΓ(k+1) − lnΓ(n−k+1).
 * Returns the raw log value for use in log-space PMF computations
 * (Hypergeometric, Negative Binomial) where exponentiating intermediate
 * binomial coefficients would overflow.
 */
export function logComb(n: number, k: number): number {
  if (k < 0 || k > n) return -Infinity;
  if (k === 0 || k === n) return 0;
  return lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
}

// ── Discrete PMFs ───────────────────────────────────────────────────────────

/**
 * Bernoulli PMF: P(X = k) = p^k (1-p)^(1-k), k ∈ {0, 1}.
 */
export function pmfBernoulli(k: number, p: number): number {
  if (k === 0) return 1 - p;
  if (k === 1) return p;
  return 0;
}

/**
 * Binomial PMF: P(X = k) = C(n,k) p^k (1-p)^(n-k), k ∈ {0, 1, ..., n}.
 */
export function pmfBinomial(k: number, n: number, p: number): number {
  if (!Number.isInteger(k) || k < 0 || k > n) return 0;
  if (p === 0) return k === 0 ? 1 : 0;
  if (p === 1) return k === n ? 1 : 0;
  // Use log-space for numerical stability
  const logP = lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1)
    + k * Math.log(p) + (n - k) * Math.log(1 - p);
  return Math.exp(logP);
}

/**
 * Geometric PMF: P(X = k) = (1-p)^(k-1) p, k = 1, 2, 3, ...
 * (Number of trials until first success.)
 */
export function pmfGeometric(k: number, p: number): number {
  if (!Number.isInteger(k) || k < 1) return 0;
  return Math.pow(1 - p, k - 1) * p;
}

/**
 * Poisson PMF: P(X = k) = e^(-λ) λ^k / k!
 */
export function pmfPoisson(k: number, lambda: number): number {
  if (!Number.isInteger(k) || k < 0) return 0;
  if (lambda === 0) return k === 0 ? 1 : 0;
  // Use log-space for numerical stability
  const logP = -lambda + k * Math.log(lambda) - lnGamma(k + 1);
  return Math.exp(logP);
}

/**
 * Negative Binomial PMF: P(X = k) = C(k-1, r-1) p^r (1-p)^(k-r).
 * k = number of trials until r-th success, k = r, r+1, r+2, ...
 * Uses log-space for numerical stability with large r.
 */
export function pmfNegativeBinomial(k: number, r: number, p: number): number {
  if (!Number.isInteger(k) || k < r) return 0;
  if (p === 0) return 0;
  if (p === 1) return k === r ? 1 : 0;
  const logP = logComb(k - 1, r - 1) + r * Math.log(p) + (k - r) * Math.log(1 - p);
  return Math.exp(logP);
}

/**
 * Hypergeometric PMF: P(X = k) = C(K,k) C(N-K, n-k) / C(N, n).
 * N = population, K = success states, n = draws, k = observed successes.
 * Uses log-space for numerical stability with large N.
 */
export function pmfHypergeometric(k: number, N: number, K: number, n: number): number {
  if (!Number.isInteger(k)) return 0;
  if (N < 0 || K < 0 || K > N || n < 0 || n > N) return 0;
  const lo = Math.max(0, n - (N - K));
  const hi = Math.min(n, K);
  if (k < lo || k > hi) return 0;
  const logP = logComb(K, k) + logComb(N - K, n - k) - logComb(N, n);
  return Math.exp(logP);
}

/**
 * Discrete Uniform PMF: P(X = k) = 1/(b - a + 1) for k ∈ {a, a+1, ..., b}.
 */
export function pmfDiscreteUniform(k: number, a: number, b: number): number {
  if (b < a) return 0;
  if (!Number.isInteger(k) || k < a || k > b) return 0;
  return 1 / (b - a + 1);
}

// ── Continuous PDFs ───────────────────────────────────────────────────────────

/**
 * Uniform PDF on [a, b].
 */
export function pdfUniform(x: number, a: number, b: number): number {
  if (b <= a) return 0;
  if (x < a || x > b) return 0;
  return 1 / (b - a);
}

/**
 * Normal PDF with mean μ and variance σ².
 */
export function pdfNormal(x: number, mu: number, sigma2: number): number {
  if (sigma2 <= 0) return 0;
  const sigma = Math.sqrt(sigma2);
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/**
 * Standard normal PDF φ(x) = N(0,1).
 */
export function pdfStdNormal(x: number): number {
  return pdfNormal(x, 0, 1);
}

/**
 * Exponential PDF with rate λ: f(x) = λ e^(-λx), x ≥ 0.
 */
export function pdfExponential(x: number, lambda: number): number {
  if (lambda <= 0 || x < 0) return 0;
  return lambda * Math.exp(-lambda * x);
}

/**
 * Bivariate normal PDF with means (μX, μY), std devs (σX, σY), correlation ρ.
 *
 * f(x,y) = (1 / (2π σX σY √(1-ρ²))) exp(-Q/2)
 * where Q = (1/(1-ρ²)) [(x-μX)²/σX² - 2ρ(x-μX)(y-μY)/(σXσY) + (y-μY)²/σY²]
 */
export function pdfBivariateNormal(
  x: number,
  y: number,
  muX: number,
  muY: number,
  sigmaX: number,
  sigmaY: number,
  rho: number,
): number {
  if (sigmaX <= 0 || sigmaY <= 0 || Math.abs(rho) >= 1) return 0;

  const zx = (x - muX) / sigmaX;
  const zy = (y - muY) / sigmaY;
  const rho2 = rho * rho;
  const oneMinusRho2 = 1 - rho2;

  const Q = (zx * zx - 2 * rho * zx * zy + zy * zy) / oneMinusRho2;
  const normalization = 2 * Math.PI * sigmaX * sigmaY * Math.sqrt(oneMinusRho2);

  return Math.exp(-0.5 * Q) / normalization;
}

// ── CDFs ────────────────────────────────────────────────────────────────────

/**
 * Standard normal CDF Φ(x) via the error function.
 */
export function cdfStdNormal(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/**
 * Normal CDF: P(X ≤ x) for X ~ N(μ, σ²).
 */
export function cdfNormal(x: number, mu: number, sigma2: number): number {
  if (sigma2 <= 0) return x >= mu ? 1 : 0;
  return cdfStdNormal((x - mu) / Math.sqrt(sigma2));
}

/**
 * Uniform CDF on [a, b].
 */
export function cdfUniform(x: number, a: number, b: number): number {
  if (b <= a) return 0;
  if (x < a) return 0;
  if (x > b) return 1;
  return (x - a) / (b - a);
}

/**
 * Exponential CDF: P(X ≤ x) = 1 - e^(-λx), x ≥ 0.
 */
export function cdfExponential(x: number, lambda: number): number {
  if (lambda <= 0 || x < 0) return 0;
  return 1 - Math.exp(-lambda * x);
}

/**
 * Binomial CDF: P(X ≤ k) = Σ_{i=0}^{⌊k⌋} C(n,i) p^i (1-p)^(n-i).
 */
export function cdfBinomial(k: number, n: number, p: number): number {
  if (k < 0) return 0;
  if (k >= n) return 1;
  const kFloor = Math.floor(k);
  let sum = 0;
  for (let i = 0; i <= kFloor; i++) {
    sum += pmfBinomial(i, n, p);
  }
  return Math.min(sum, 1); // Clamp for floating-point safety
}

/**
 * Geometric CDF: P(X ≤ k) = 1 - (1-p)^k, k = 1, 2, ...
 */
export function cdfGeometric(k: number, p: number): number {
  if (k < 1) return 0;
  return 1 - Math.pow(1 - p, Math.floor(k));
}

/**
 * Poisson CDF: P(X ≤ k) = Σ_{i=0}^{⌊k⌋} e^(-λ) λ^i / i!
 */
export function cdfPoisson(k: number, lambda: number): number {
  if (k < 0) return 0;
  const kFloor = Math.floor(k);
  let sum = 0;
  for (let i = 0; i <= kFloor; i++) {
    sum += pmfPoisson(i, lambda);
  }
  return Math.min(sum, 1);
}

/**
 * Negative Binomial CDF: P(X ≤ k) = Σ_{i=r}^{⌊k⌋} pmfNegBin(i, r, p).
 */
export function cdfNegativeBinomial(k: number, r: number, p: number): number {
  if (k < r) return 0;
  const kFloor = Math.floor(k);
  let sum = 0;
  for (let i = r; i <= kFloor; i++) {
    sum += pmfNegativeBinomial(i, r, p);
  }
  return Math.min(sum, 1);
}

/**
 * Hypergeometric CDF: P(X ≤ k) = Σ pmfHypergeometric over valid range.
 */
export function cdfHypergeometric(k: number, N: number, K: number, n: number): number {
  const lo = Math.max(0, n - (N - K));
  if (k < lo) return 0;
  const hi = Math.min(n, K);
  if (k >= hi) return 1;
  const kFloor = Math.floor(k);
  let sum = 0;
  for (let i = lo; i <= kFloor; i++) {
    sum += pmfHypergeometric(i, N, K, n);
  }
  return Math.min(sum, 1);
}

/**
 * Discrete Uniform CDF: P(X ≤ k) = (⌊k⌋ - a + 1) / (b - a + 1).
 */
export function cdfDiscreteUniform(k: number, a: number, b: number): number {
  if (b < a) return 0;
  if (k < a) return 0;
  if (k >= b) return 1;
  return (Math.floor(k) - a + 1) / (b - a + 1);
}

// ── Quantile (Inverse CDF) Functions ─────────────────────────────────────────

/**
 * Standard normal quantile Φ⁻¹(p).
 * Uses Peter Acklam's rational approximation.
 * Accurate to about 1.15 × 10⁻⁹ in the full range.
 */
export function quantileStdNormal(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  // Coefficients for rational approximation
  const a = [
    -3.969683028665376e+01,
     2.209460984245205e+02,
    -2.759285104469687e+02,
     1.383577518672690e+02,
    -3.066479806614716e+01,
     2.506628277459239e+00,
  ];
  const b = [
    -5.447609879822406e+01,
     1.615858368580409e+02,
    -1.556989798598866e+02,
     6.680131188771972e+01,
    -1.328068155288572e+01,
  ];
  const c = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
     4.374664141464968e+00,
     2.938163982698783e+00,
  ];
  const d = [
     7.784695709041462e-03,
     3.224671290700398e-01,
     2.445134137142996e+00,
     3.754408661907416e+00,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number;
  let r: number;

  if (p < pLow) {
    // Rational approximation for lower region
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
         / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    // Rational approximation for central region
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
         / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    // Rational approximation for upper region
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
          / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
}

/**
 * Normal quantile: F⁻¹(p) = μ + σ · Φ⁻¹(p).
 */
export function quantileNormal(p: number, mu: number, sigma2: number): number {
  if (sigma2 <= 0) return mu;
  return mu + Math.sqrt(sigma2) * quantileStdNormal(p);
}

/**
 * Exponential quantile: F⁻¹(p) = -ln(1-p) / λ.
 */
export function quantileExponential(p: number, lambda: number): number {
  if (lambda <= 0 || p <= 0) return 0;
  if (p >= 1) return Infinity;
  return -Math.log(1 - p) / lambda;
}

/**
 * Uniform quantile on [a, b]: F⁻¹(p) = a + p(b - a).
 */
export function quantileUniform(p: number, a: number, b: number): number {
  if (p <= 0) return a;
  if (p >= 1) return b;
  return a + p * (b - a);
}

// ── Conditional Normal ──────────────────────────────────────────────────────

/**
 * Conditional parameters of X|Y=y for bivariate normal.
 *
 * If (X,Y) ~ BVN(μX, μY, σX, σY, ρ), then:
 *   X | Y=y ~ N(condMean, condVar)
 *   condMean = μX + ρ(σX/σY)(y - μY)
 *   condVar  = σX²(1 - ρ²)
 */
export function conditionalNormalParams(
  muX: number,
  muY: number,
  sigmaX: number,
  sigmaY: number,
  rho: number,
  yGiven: number,
): { condMean: number; condVar: number } {
  const condMean = muX + rho * (sigmaX / sigmaY) * (yGiven - muY);
  const condVar = sigmaX * sigmaX * (1 - rho * rho);
  return { condMean, condVar };
}

// ── Numerical Integration ───────────────────────────────────────────────────

/**
 * Trapezoidal rule integration of f over [a, b] with n steps.
 * Default n = 1000 provides good accuracy for smooth functions.
 */
export function trapezoidalIntegral(
  f: (x: number) => number,
  a: number,
  b: number,
  n: number = 1000,
): number {
  if (a >= b || n < 1) return 0;
  const h = (b - a) / n;
  let sum = 0.5 * (f(a) + f(b));
  for (let i = 1; i < n; i++) {
    sum += f(a + i * h);
  }
  return sum * h;
}

// ── MGFs (Topic 5) ─────────────────────────────────────────────────────────

/**
 * Geometric MGF: M(t) = pe^t / (1 − qe^t), defined for t < −ln(1−p).
 */
export function mgfGeometric(t: number, p: number): number {
  const q = 1 - p;
  if (t >= -Math.log(q)) return Infinity;
  return (p * Math.exp(t)) / (1 - q * Math.exp(t));
}

/**
 * Negative Binomial MGF: M(t) = (pe^t / (1 − qe^t))^r, t < −ln(1−p).
 */
export function mgfNegativeBinomial(t: number, r: number, p: number): number {
  const q = 1 - p;
  if (t >= -Math.log(q)) return Infinity;
  return Math.pow((p * Math.exp(t)) / (1 - q * Math.exp(t)), r);
}

/**
 * Discrete Uniform MGF: M(t) = e^{ta}(1 − e^{nt}) / (n(1 − e^t)).
 * At t = 0, uses L'Hôpital: M(0) = 1.
 * n = b − a + 1 is the support size.
 */
export function mgfDiscreteUniform(t: number, a: number, b: number): number {
  const n = b - a + 1;
  if (Math.abs(t) < 1e-12) return 1;
  return (Math.exp(t * a) * (1 - Math.exp(n * t))) / (n * (1 - Math.exp(t)));
}

// ── PGFs (Topic 5) ─────────────────────────────────────────────────────────

/**
 * Bernoulli PGF: G(s) = (1−p) + ps.
 */
export function pgfBernoulli(s: number, p: number): number {
  if (s === 1) return 1;
  return (1 - p) + p * s;
}

/**
 * Binomial PGF: G(s) = ((1−p) + ps)^n.
 */
export function pgfBinomial(s: number, n: number, p: number): number {
  if (s === 1) return 1;
  return Math.pow((1 - p) + p * s, n);
}

/**
 * Geometric PGF: G(s) = ps / (1 − (1−p)s), |s| < 1/q.
 * For the {1,2,...} convention, G(0) = 0.
 */
export function pgfGeometric(s: number, p: number): number {
  if (s === 1) return 1;
  const q = 1 - p;
  const denom = 1 - q * s;
  if (denom <= 0) return Infinity;
  return (p * s) / denom;
}

/**
 * Negative Binomial PGF: G(s) = (ps / (1 − qs))^r, |s| < 1/q.
 */
export function pgfNegativeBinomial(s: number, r: number, p: number): number {
  if (s === 1) return 1;
  const q = 1 - p;
  const denom = 1 - q * s;
  if (denom <= 0) return Infinity;
  return Math.pow((p * s) / denom, r);
}

/**
 * Poisson PGF: G(s) = e^{λ(s−1)}.
 */
export function pgfPoisson(s: number, lambda: number): number {
  if (s === 1) return 1;
  return Math.exp(lambda * (s - 1));
}

/**
 * Discrete Uniform PGF: G(s) = s^a (1 − s^n) / (n(1 − s)) for s ≠ 1.
 * At s = 1, returns 1 (L'Hôpital). n = b − a + 1.
 */
export function pgfDiscreteUniform(s: number, a: number, b: number): number {
  if (s === 1) return 1;
  const n = b - a + 1;
  return (Math.pow(s, a) * (1 - Math.pow(s, n))) / (n * (1 - s));
}

// ── Closed-Form Moment Functions (Topic 5) ─────────────────────────────────

/** Bernoulli E[X] = p. */
export function expectationBernoulli(p: number): number { return p; }
/** Bernoulli Var(X) = p(1−p). */
export function varianceBernoulli(p: number): number { return p * (1 - p); }

/** Binomial E[X] = np. */
export function expectationBinomial(n: number, p: number): number { return n * p; }
/** Binomial Var(X) = np(1−p). */
export function varianceBinomial(n: number, p: number): number { return n * p * (1 - p); }

/** Geometric E[X] = 1/p (trials until first success). */
export function expectationGeometric(p: number): number { return 1 / p; }
/** Geometric Var(X) = (1−p)/p². */
export function varianceGeometric(p: number): number { return (1 - p) / (p * p); }

/** Negative Binomial E[X] = r/p. */
export function expectationNegBin(r: number, p: number): number { return r / p; }
/** Negative Binomial Var(X) = r(1−p)/p². */
export function varianceNegBin(r: number, p: number): number { return r * (1 - p) / (p * p); }

/** Poisson E[X] = λ. */
export function expectationPoisson(lambda: number): number { return lambda; }
/** Poisson Var(X) = λ (equidispersion). */
export function variancePoisson(lambda: number): number { return lambda; }

/** Hypergeometric E[X] = nK/N. */
export function expectationHypergeometric(N: number, K: number, n: number): number {
  return n * K / N;
}
/** Hypergeometric Var(X) = n·K(N−K)(N−n) / (N²(N−1)). */
export function varianceHypergeometric(N: number, K: number, n: number): number {
  return (n * K * (N - K) * (N - n)) / (N * N * (N - 1));
}

/** Discrete Uniform E[X] = (a+b)/2. */
export function expectationDiscreteUniform(a: number, b: number): number {
  return (a + b) / 2;
}
/** Discrete Uniform Var(X) = ((b−a+1)² − 1) / 12. */
export function varianceDiscreteUniform(a: number, b: number): number {
  const n = b - a + 1;
  return (n * n - 1) / 12;
}

// ── Continuous PDFs (Topic 6) ──────────────────────────────────────────────

/**
 * Gamma(α, β) PDF using shape-rate parameterization.
 * f(x) = β^α / Γ(α) · x^{α−1} e^{−βx}, x > 0.
 * Computed in log space to avoid overflow for large α.
 */
export function pdfGamma(x: number, alpha: number, beta: number): number {
  if (x <= 0 || alpha <= 0 || beta <= 0) return 0;
  const logPdf = alpha * Math.log(beta) - lnGamma(alpha)
    + (alpha - 1) * Math.log(x) - beta * x;
  return Math.exp(logPdf);
}

/**
 * Beta(α, β) PDF on (0, 1).
 * f(x) = x^{α−1}(1−x)^{β−1} / B(α,β), where B(α,β) = Γ(α)Γ(β)/Γ(α+β).
 * Computed in log space.
 */
export function pdfBeta(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1 || a <= 0 || b <= 0) return 0;
  const logPdf = lnGamma(a + b) - lnGamma(a) - lnGamma(b)
    + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x);
  return Math.exp(logPdf);
}

/**
 * Chi-squared(k) PDF = Gamma(k/2, 1/2).
 */
export function pdfChi2(x: number, k: number): number {
  return pdfGamma(x, k / 2, 0.5);
}

/**
 * Student's t(ν) PDF.
 * f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^{−(ν+1)/2}.
 * Computed in log space.
 */
export function pdfStudentT(x: number, nu: number): number {
  if (nu <= 0) return 0;
  const logPdf = lnGamma((nu + 1) / 2) - lnGamma(nu / 2)
    - 0.5 * Math.log(nu * Math.PI)
    - ((nu + 1) / 2) * Math.log(1 + (x * x) / nu);
  return Math.exp(logPdf);
}

/**
 * F(d₁, d₂) PDF. x > 0.
 * f(x) = √((d₁x)^d₁ · d₂^d₂ / (d₁x+d₂)^{d₁+d₂}) / (x · B(d₁/2, d₂/2)).
 * Computed entirely in log space.
 */
export function pdfF(x: number, d1: number, d2: number): number {
  if (x <= 0 || d1 <= 0 || d2 <= 0) return 0;
  const logPdf = lnGamma((d1 + d2) / 2) - lnGamma(d1 / 2) - lnGamma(d2 / 2)
    + (d1 / 2) * Math.log(d1 / d2)
    + (d1 / 2 - 1) * Math.log(x)
    - ((d1 + d2) / 2) * Math.log(1 + (d1 / d2) * x);
  return Math.exp(logPdf);
}

// ── Continuous CDFs (Topic 6) ──────────────────────────────────────────────

/**
 * Gamma CDF via numerical integration: P(X ≤ x) = ∫₀ˣ pdfGamma(t, α, β) dt.
 * For α < 1, starts at ε = 1e-10 to avoid the PDF singularity at 0.
 */
export function cdfGamma(x: number, alpha: number, beta: number): number {
  if (x <= 0 || alpha <= 0 || beta <= 0) return 0;
  const eps = alpha < 1 ? 1e-10 : 0;
  return Math.min(trapezoidalIntegral(t => pdfGamma(t, alpha, beta), eps, x, 200), 1);
}

/**
 * Beta CDF via numerical integration: P(X ≤ x) = ∫₀ˣ pdfBeta(t, α, β) dt.
 */
export function cdfBeta(x: number, a: number, b: number): number {
  if (x <= 0 || a <= 0 || b <= 0) return 0;
  if (x >= 1) return 1;
  const lo = 1e-10;
  const hi = Math.min(x, 1 - 1e-10);
  if (hi <= lo) return 0;
  return Math.min(trapezoidalIntegral(t => pdfBeta(t, a, b), lo, hi, 200), 1);
}

/**
 * Chi-squared CDF = cdfGamma(x, k/2, 1/2).
 */
export function cdfChi2(x: number, k: number): number {
  return cdfGamma(x, k / 2, 0.5);
}

// ── Continuous Quantiles (Topic 6) ─────────────────────────────────────────

/**
 * Beta quantile via bisection on cdfBeta.
 * Returns x such that cdfBeta(x, a, b) ≈ p.
 * 50 iterations provides ~15 digits of precision.
 */
export function quantileBeta(p: number, a: number, b: number): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2;
    if (cdfBeta(mid, a, b) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

// ── Continuous MGFs (Topic 6) ──────────────────────────────────────────────

/**
 * Uniform(a, b) MGF: M(t) = (e^{tb} − e^{ta}) / (t(b−a)).
 * At t ≈ 0, returns 1 (L'Hôpital).
 */
export function mgfUniform(t: number, a: number, b: number): number {
  if (Math.abs(t) < 1e-12) return 1;
  return (Math.exp(t * b) - Math.exp(t * a)) / (t * (b - a));
}

/**
 * Normal(μ, σ²) MGF: M(t) = exp(μt + σ²t²/2).
 */
export function mgfNormal(t: number, mu: number, sigma2: number): number {
  return Math.exp(mu * t + sigma2 * t * t / 2);
}

/**
 * Exponential(λ) MGF: M(t) = λ/(λ−t), defined for t < λ.
 */
export function mgfExponential(t: number, lambda: number): number {
  if (t >= lambda) return Infinity;
  return lambda / (lambda - t);
}

/**
 * Gamma(α, β) MGF: M(t) = (β/(β−t))^α, defined for t < β.
 */
export function mgfGamma(t: number, alpha: number, beta: number): number {
  if (t >= beta) return Infinity;
  return Math.pow(beta / (beta - t), alpha);
}

/**
 * Chi-squared(k) MGF: M(t) = (1−2t)^{−k/2}, defined for t < 1/2.
 */
export function mgfChi2(t: number, k: number): number {
  if (t >= 0.5) return Infinity;
  return Math.pow(1 - 2 * t, -k / 2);
}

// ── Closed-Form Moment Functions (Topic 6) ─────────────────────────────────

/** Uniform E[X] = (a+b)/2. */
export function expectationUniform(a: number, b: number): number { return (a + b) / 2; }
/** Uniform Var(X) = (b−a)²/12. */
export function varianceUniform(a: number, b: number): number { return (b - a) * (b - a) / 12; }

/** Normal E[X] = μ. */
export function expectationNormal(mu: number): number { return mu; }
/** Normal Var(X) = σ². */
export function varianceNormal(sigma2: number): number { return sigma2; }

/** Exponential E[X] = 1/λ. */
export function expectationExponential(lambda: number): number { return 1 / lambda; }
/** Exponential Var(X) = 1/λ². */
export function varianceExponential(lambda: number): number { return 1 / (lambda * lambda); }

/** Gamma E[X] = α/β. */
export function expectationGamma(alpha: number, beta: number): number { return alpha / beta; }
/** Gamma Var(X) = α/β². */
export function varianceGamma(alpha: number, beta: number): number { return alpha / (beta * beta); }

/** Beta E[X] = α/(α+β). */
export function expectationBeta(a: number, b: number): number { return a / (a + b); }
/** Beta Var(X) = αβ/((α+β)²(α+β+1)). */
export function varianceBeta(a: number, b: number): number {
  const s = a + b;
  return (a * b) / (s * s * (s + 1));
}

/** Chi-squared E[X] = k. */
export function expectationChi2(k: number): number { return k; }
/** Chi-squared Var(X) = 2k. */
export function varianceChi2(k: number): number { return 2 * k; }

/** Student's t E[X] = 0 (defined for ν > 1). */
export function expectationStudentT(nu: number): number {
  return nu > 1 ? 0 : NaN;
}
/** Student's t Var(X) = ν/(ν−2) (defined for ν > 2). */
export function varianceStudentT(nu: number): number {
  if (nu <= 2) return NaN;
  return nu / (nu - 2);
}

/** F E[X] = d₂/(d₂−2) (defined for d₂ > 2). */
export function expectationF(d1: number, d2: number): number {
  if (d2 <= 2) return NaN;
  return d2 / (d2 - 2);
}
/** F Var(X) = 2d₂²(d₁+d₂−2) / (d₁(d₂−2)²(d₂−4)) (defined for d₂ > 4). */
export function varianceF(d1: number, d2: number): number {
  if (d2 <= 4) return NaN;
  return (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) * (d2 - 2) * (d2 - 4));
}
