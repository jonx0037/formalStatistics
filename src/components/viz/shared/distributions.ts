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

// ── Continuous PDFs ─────────────────────────────────────────────────────────

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

// ── Quantile (Inverse CDF) Functions ────────────────────────────────────────

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
