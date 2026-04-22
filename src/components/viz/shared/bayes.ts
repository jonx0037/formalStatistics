/**
 * Topic 25 — Bayesian Foundations & Prior Selection (Track 7 leaf module).
 *
 * New Track-7-wide shared module. Topics 26–28 (MCMC, BMA, Hierarchical)
 * extend this file rather than adding sibling modules — the extend-don't-
 * create convention that Tracks 4–6 established within `estimation.ts`,
 * `testing.ts`, and `regression.ts`.
 *
 * Scope (see `docs/formalstatistics-bayesian-foundations-and-prior-selection-handoff-brief.md`
 * §6.1 for the full manifest):
 *
 *   • Density functions: pdfBeta / pdfGamma / pdfNormal / pdfInverseGamma
 *     re-exported from Topic 6's distributions.ts; three new multivariate
 *     densities (pdfNormalInverseGamma, pdfStudentTMarginal, pdfDirichlet).
 *   • Posterior-predictive PMFs: Beta-Binomial compound, Negative-Binomial
 *     (Gamma-Poisson compound).
 *   • Posterior hyperparameter update via `posterior(family, prior, data)`
 *     — exhaustive discriminated-union dispatch.
 *   • Credible intervals: equal-tailed on Beta / Gamma / Normal, plus a
 *     generic HPD solver via bisection-on-density-level and a memoized
 *     Beta specialization for the BvM-animator hot path.
 *   • Point estimators: mapEstimate, posteriorMean, posteriorVariance.
 *   • Jeffreys priors for Bernoulli / Poisson / Normal-mean / Normal-scale /
 *     Exponential (the five one-parameter regular families cited in §25.7).
 *   • Laplace approximation to log marginal likelihood (the §24.4 BIC
 *     machinery lifted to the full posterior — §25.8 Rem 16).
 *   • Posterior sampling for the conjugate cases only; non-conjugate
 *     posteriors defer to Topic 26's MCMC machinery.
 *   • Posterior predictive for Beta-Binomial, Normal-Normal, Gamma-Poisson.
 *
 * Notation conventions (locked by §25.3, inherited by Topics 26–28):
 *   π(θ) prior, p(θ|y) posterior, p(ỹ|y) posterior predictive, m(y) marginal
 *   likelihood, KL(π||π') KL divergence, BF₁₀ Bayes factor.
 */

import {
  pdfBeta,
  pdfNormal,
  pdfInverseGamma,
  lnGamma,
  cdfGamma,
  quantileBeta,
  quantileNormal,
  sampleGammaShape,
  sampleDirichlet,
} from './distributions';

// Re-export densities for callers who only import from `bayes.ts`
// (brief §6.1 "reuse over duplication" — don't redefine, just re-export).
export { pdfBeta, pdfGamma, pdfNormal, pdfInverseGamma } from './distributions';

// ═══════════════════════════════════════════════════════════════════════════
// Types — conjugate-family discriminated union (§25.5 five canonical pairs).
// Posterior shares the Prior shape with updated values (brief §6.1 note).
// ═══════════════════════════════════════════════════════════════════════════

export type ConjugateFamily =
  | 'beta-binomial'
  | 'normal-normal'
  | 'normal-normal-ig'
  | 'gamma-poisson'
  | 'dirichlet-multinomial';

export type PriorHyperparams =
  | { family: 'beta-binomial'; alpha0: number; beta0: number }
  | { family: 'normal-normal'; mu0: number; sigma0_sq: number; sigma_sq: number }
  | { family: 'normal-normal-ig'; mu0: number; kappa0: number; alpha0: number; beta0: number }
  | { family: 'gamma-poisson'; alpha0: number; beta0: number }
  | { family: 'dirichlet-multinomial'; alpha0: number[] };

export type SuffStats =
  | { family: 'beta-binomial'; n: number; k: number }
  | { family: 'normal-normal'; n: number; yBar: number }
  | { family: 'normal-normal-ig'; n: number; yBar: number; s2: number }
  | { family: 'gamma-poisson'; n: number; S: number }
  | { family: 'dirichlet-multinomial'; counts: number[] };

/**
 * Posterior hyperparameters — same shape as Prior (updated values).
 * Naming quirk: for Normal-Normal, `mu0` holds the posterior mean μ_n and
 * `sigma0_sq` holds the posterior variance σ_n². We keep the field names
 * identical to the prior schema so `posterior(...)` returns a value of the
 * same discriminated-union variant — the alternative (separate Prior / Post
 * types) would double the type surface for zero inferential benefit.
 */
export type PosteriorHyperparams = PriorHyperparams;

export type PredictiveOptions =
  | { family: 'beta-binomial'; m: number }
  | { family: 'normal-normal' }
  | { family: 'gamma-poisson' }
  | { family: 'dirichlet-multinomial'; nNew: number };

// ═══════════════════════════════════════════════════════════════════════════
// New multivariate / compound density functions.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Dirichlet(α) density at a point p on the (k-1)-simplex.
 * f(p) = Γ(Σα) / Π Γ(αⱼ) · Π pⱼ^{αⱼ − 1}.
 *
 * Computed in log space to avoid Γ-function overflow for αⱼ > 20 — the
 * Dirichlet-Multinomial posterior with observed counts in the tens easily
 * pushes α into overflow territory under direct computation.
 */
export function pdfDirichlet(p: number[], alpha: number[]): number {
  if (p.length !== alpha.length) return 0;
  const k = alpha.length;
  let alphaSum = 0;
  let logPdf = 0;
  for (let j = 0; j < k; j++) {
    if (p[j] <= 0 || p[j] >= 1 || alpha[j] <= 0) return 0;
    alphaSum += alpha[j];
    logPdf += (alpha[j] - 1) * Math.log(p[j]) - lnGamma(alpha[j]);
  }
  logPdf += lnGamma(alphaSum);
  return Math.exp(logPdf);
}

/**
 * Normal-Inverse-Gamma joint density at (μ, σ²) for §25.5 Ex 3.
 * p(μ, σ²) = Normal(μ; μ₀, σ²/κ₀) · InverseGamma(σ²; α₀, β₀).
 * Valid for μ ∈ ℝ, σ² > 0.
 */
export function pdfNormalInverseGamma(
  mu: number,
  sigma2: number,
  mu0: number,
  kappa0: number,
  alpha0: number,
  beta0: number,
): number {
  if (sigma2 <= 0 || kappa0 <= 0) return 0;
  return pdfNormal(mu, mu0, sigma2 / kappa0) * pdfInverseGamma(sigma2, alpha0, beta0);
}

/**
 * Non-standardized Student-t density — the marginal posterior on μ in the
 * Normal-Normal-Inverse-Gamma family (§25.5 Ex 3, §25.6 Rem 11).
 *
 * f(x) = Γ((ν+1)/2) / (Γ(ν/2) · √(νπσ²)) · (1 + (x−μ)²/(νσ²))^{−(ν+1)/2}.
 *
 * @param x location
 * @param mu location parameter
 * @param scale2 squared scale σ² (not variance; variance = ν·σ² / (ν−2))
 * @param df degrees of freedom ν
 */
export function pdfStudentTMarginal(
  x: number,
  mu: number,
  scale2: number,
  df: number,
): number {
  if (scale2 <= 0 || df <= 0) return 0;
  const z = (x - mu) / Math.sqrt(scale2);
  const logPdf = lnGamma((df + 1) / 2) - lnGamma(df / 2)
    - 0.5 * Math.log(df * Math.PI * scale2)
    - ((df + 1) / 2) * Math.log(1 + (z * z) / df);
  return Math.exp(logPdf);
}

// ═══════════════════════════════════════════════════════════════════════════
// Posterior-predictive PMFs (compound distributions).
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Beta-Binomial posterior-predictive PMF at future count y ∈ {0,…,m}.
 * p(ỹ|y) = C(m, ỹ) · B(ỹ+α*, m−ỹ+β*) / B(α*, β*).
 *
 * Computed in log space — for m in the tens and (α*, β*) in the hundreds,
 * naive binomial coefficients and Beta functions both overflow. Writes every
 * factor as a Γ ratio via lnGamma then exponentiates once at the end.
 */
export function pmfBetaBinomial(
  yNew: number,
  m: number,
  alphaStar: number,
  betaStar: number,
): number {
  if (yNew < 0 || yNew > m || !Number.isInteger(yNew) || !Number.isInteger(m)) return 0;
  if (alphaStar <= 0 || betaStar <= 0) return 0;
  // log C(m, y) = lnΓ(m+1) − lnΓ(y+1) − lnΓ(m−y+1)
  const logBinom = lnGamma(m + 1) - lnGamma(yNew + 1) - lnGamma(m - yNew + 1);
  // log B(a, b) = lnΓ(a) + lnΓ(b) − lnΓ(a+b)
  const logBNum = lnGamma(yNew + alphaStar) + lnGamma(m - yNew + betaStar)
    - lnGamma(m + alphaStar + betaStar);
  const logBDen = lnGamma(alphaStar) + lnGamma(betaStar) - lnGamma(alphaStar + betaStar);
  return Math.exp(logBinom + logBNum - logBDen);
}

/**
 * Negative-Binomial posterior-predictive PMF for the Gamma-Poisson compound.
 * When λ | y ∼ Gamma(a, b) (shape-rate), the predictive for a new Poisson
 * count is NegBinom with r = a, p = b / (b + 1):
 *
 *   p(ỹ|y) = Γ(a + ỹ) / (Γ(a) · ỹ!) · (b / (b+1))^a · (1 / (b+1))^ỹ.
 *
 * Here `alphaStar` and `betaStar` are the posterior hyperparameters after
 * observing data (see `posterior('gamma-poisson', ...)` for the update rule).
 */
export function pmfNegativeBinomialPosterior(
  yNew: number,
  alphaStar: number,
  betaStar: number,
): number {
  if (yNew < 0 || !Number.isInteger(yNew)) return 0;
  if (alphaStar <= 0 || betaStar <= 0) return 0;
  const logGammaRatio = lnGamma(alphaStar + yNew) - lnGamma(alphaStar) - lnGamma(yNew + 1);
  const logP = alphaStar * Math.log(betaStar / (betaStar + 1))
    + yNew * Math.log(1 / (betaStar + 1));
  return Math.exp(logGammaRatio + logP);
}

// ═══════════════════════════════════════════════════════════════════════════
// Posterior dispatch — exhaustive switch on the discriminated family tag.
// `never` fallthrough catches any ConjugateFamily added later that forgets
// to register a posterior update rule.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Conjugate-posterior hyperparameter update. Returns a posterior with the
 * same discriminated-union shape as the prior. See brief §3.2 Proof 1 for
 * the exponential-family unifying theorem; the five pairs below are its
 * locked instances for Topic 25.
 */
export function posterior(
  family: ConjugateFamily,
  prior: PriorHyperparams,
  data: SuffStats,
): PosteriorHyperparams {
  if (prior.family !== family || data.family !== family) {
    throw new Error(
      `posterior: family mismatch — expected ${family}, got prior=${prior.family}, data=${data.family}`,
    );
  }

  switch (family) {
    case 'beta-binomial': {
      const p = prior as Extract<PriorHyperparams, { family: 'beta-binomial' }>;
      const d = data as Extract<SuffStats, { family: 'beta-binomial' }>;
      return {
        family: 'beta-binomial',
        alpha0: p.alpha0 + d.k,
        beta0: p.beta0 + d.n - d.k,
      };
    }
    case 'normal-normal': {
      const p = prior as Extract<PriorHyperparams, { family: 'normal-normal' }>;
      const d = data as Extract<SuffStats, { family: 'normal-normal' }>;
      // Precision-weighted: 1/σ_n² = 1/σ_0² + n/σ². μ_n = σ_n²·(μ_0/σ_0² + n·ȳ/σ²).
      const precPrior = 1 / p.sigma0_sq;
      const precData = d.n / p.sigma_sq;
      const sigmaN2 = 1 / (precPrior + precData);
      const muN = sigmaN2 * (p.mu0 * precPrior + d.yBar * precData);
      return {
        family: 'normal-normal',
        mu0: muN,
        sigma0_sq: sigmaN2,
        sigma_sq: p.sigma_sq, // likelihood variance unchanged by the update
      };
    }
    case 'normal-normal-ig': {
      const p = prior as Extract<PriorHyperparams, { family: 'normal-normal-ig' }>;
      const d = data as Extract<SuffStats, { family: 'normal-normal-ig' }>;
      // See GEL2013 §3.3 eqns 3.3–3.5 for the update rule.
      const kappaN = p.kappa0 + d.n;
      const muN = (p.kappa0 * p.mu0 + d.n * d.yBar) / kappaN;
      const alphaN = p.alpha0 + d.n / 2;
      const betaN = p.beta0
        + 0.5 * d.n * d.s2
        + 0.5 * (p.kappa0 * d.n / kappaN) * (d.yBar - p.mu0) ** 2;
      return { family: 'normal-normal-ig', mu0: muN, kappa0: kappaN, alpha0: alphaN, beta0: betaN };
    }
    case 'gamma-poisson': {
      const p = prior as Extract<PriorHyperparams, { family: 'gamma-poisson' }>;
      const d = data as Extract<SuffStats, { family: 'gamma-poisson' }>;
      return { family: 'gamma-poisson', alpha0: p.alpha0 + d.S, beta0: p.beta0 + d.n };
    }
    case 'dirichlet-multinomial': {
      const p = prior as Extract<PriorHyperparams, { family: 'dirichlet-multinomial' }>;
      const d = data as Extract<SuffStats, { family: 'dirichlet-multinomial' }>;
      if (p.alpha0.length !== d.counts.length) {
        throw new Error(
          `posterior: dimension mismatch (prior k=${p.alpha0.length}, data k=${d.counts.length})`,
        );
      }
      return {
        family: 'dirichlet-multinomial',
        alpha0: p.alpha0.map((a, j) => a + d.counts[j]),
      };
    }
    default: {
      const _exhaustive: never = family;
      throw new Error(`posterior: unhandled family ${_exhaustive as string}`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Credible intervals — equal-tailed (symmetric-mass) plus a generic HPD
// solver. Memoization on the two Beta specializations since the BvM and
// PriorPosterior animators hit them every slider tick.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Memoization for Beta credible / HPD intervals. Keys are quantized to 1e-6
 * so IEEE-float rounding (e.g. `0.30000000000000004`) doesn't create
 * near-duplicate entries, and the cache is bounded with FIFO eviction to
 * prevent unbounded growth over long-lived slider sessions. Map iteration
 * order is insertion order, so `.keys().next()` yields the oldest entry.
 */
const _CACHE_MAX = 1024;
const _credibleBetaCache = new Map<string, [number, number]>();
const _hpdBetaCache = new Map<string, [number, number]>();

function _cacheKey(alpha: number, beta: number, level: number): string {
  // 1e-6 quantization is ≥3 orders of magnitude finer than any slider step
  // used in the components (step ≥ 1e-3 in practice) and absorbs float noise.
  const q = (x: number) => Math.round(x * 1e6) / 1e6;
  return `${q(alpha)},${q(beta)},${q(level)}`;
}

function _memoSet<K, V>(cache: Map<K, V>, key: K, value: V): void {
  if (cache.size >= _CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  cache.set(key, value);
}

/**
 * Equal-tailed (1−α) credible interval for Beta(α, β). Memoized on
 * a quantized (alpha, beta, level) triple because the BvM animator
 * requests intervals on every slider frame.
 */
export function credibleIntervalBeta(
  alpha: number,
  beta: number,
  level: number,
): [number, number] {
  const key = _cacheKey(alpha, beta, level);
  const cached = _credibleBetaCache.get(key);
  if (cached) return cached;
  const tail = (1 - level) / 2;
  const interval: [number, number] = [
    quantileBeta(tail, alpha, beta),
    quantileBeta(1 - tail, alpha, beta),
  ];
  _memoSet(_credibleBetaCache, key, interval);
  return interval;
}

/**
 * Equal-tailed (1−α) credible interval for Gamma(α, β) (shape-rate).
 * Inverts cdfGamma by bisection — Topic 6's distributions.ts doesn't ship
 * a quantileGamma, and adding one is out of Topic 25's scope.
 */
export function credibleIntervalGamma(
  alpha: number,
  beta: number,
  level: number,
): [number, number] {
  const tail = (1 - level) / 2;
  return [
    _gammaQuantileBisect(tail, alpha, beta),
    _gammaQuantileBisect(1 - tail, alpha, beta),
  ];
}

function _gammaQuantileBisect(p: number, alpha: number, beta: number): number {
  // Degenerate quantiles.
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  // Initial upper bound: mean + 10·sd.
  const mean = alpha / beta;
  const sd = Math.sqrt(alpha) / beta;
  let lo = 0;
  let hi = Math.max(mean + 10 * sd, 1);
  // Expand upper bound until cdfGamma(hi) ≥ p, capped to prevent infinite loops
  // for extreme p near 1 or numerical-precision failures. 40 doublings gives
  // hi > 1e12 × initial bound — well past any practically reachable quantile.
  for (let expand = 0; expand < 40 && cdfGamma(hi, alpha, beta) < p; expand++) {
    hi *= 2;
  }
  for (let i = 0; i < 60; i++) {
    const mid = 0.5 * (lo + hi);
    if (cdfGamma(mid, alpha, beta) < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-8) break;
  }
  return 0.5 * (lo + hi);
}

/**
 * Equal-tailed (1−α) credible interval for Normal(μ, σ²).
 * For flat-prior coincidence with the Wald z-CI (§25.8 Rem 17) the
 * level here matches the Wald confidence level directly.
 */
export function credibleIntervalNormal(
  mu: number,
  sigma2: number,
  level: number,
): [number, number] {
  const tail = (1 - level) / 2;
  return [quantileNormal(tail, mu, sigma2), quantileNormal(1 - tail, mu, sigma2)];
}

/**
 * Generic HPD (highest-posterior-density) interval via bisection on the
 * horizontal density level c. Finds c such that ∫_{{p(θ) ≥ c}} p(θ)dθ = 1 − α,
 * then returns the interval {θ : p(θ) ≥ c} on the discretized support.
 *
 * Assumes the density is unimodal. For multimodal posteriors the result
 * would be a union of intervals; Topic 25's five conjugate families are
 * all unimodal so this restriction is fine.
 *
 * @param nGrid resolution of the support discretization (default 500;
 *              BvM-animator mobile fallback reduces to 200 in the caller)
 */
export function hpdInterval(
  pdf: (x: number) => number,
  support: [number, number],
  level: number,
  nGrid = 500,
): [number, number] {
  const [a, b] = support;
  const dx = (b - a) / (nGrid - 1);
  const xs = new Array<number>(nGrid);
  const ps = new Array<number>(nGrid);
  let pmax = 0;
  for (let i = 0; i < nGrid; i++) {
    xs[i] = a + i * dx;
    ps[i] = pdf(xs[i]);
    if (ps[i] > pmax) pmax = ps[i];
  }
  // Normalize in case the grid under-integrates the density (cheap safeguard).
  let totalMass = 0;
  for (let i = 0; i < nGrid; i++) totalMass += ps[i] * dx;
  if (totalMass <= 0) return [a, b];

  // Bisect on the density level c.
  let cLo = 0;
  let cHi = pmax;
  for (let iter = 0; iter < 40; iter++) {
    const c = 0.5 * (cLo + cHi);
    let mass = 0;
    for (let i = 0; i < nGrid; i++) if (ps[i] >= c) mass += ps[i] * dx;
    if (mass / totalMass > level) cLo = c;
    else cHi = c;
    if (cHi - cLo < 1e-6 * pmax) break;
  }
  const cStar = 0.5 * (cLo + cHi);

  // Collect the interval endpoints from the first and last grid points
  // where p(x) ≥ c*.
  let lo = b;
  let hi = a;
  for (let i = 0; i < nGrid; i++) {
    if (ps[i] >= cStar) {
      if (xs[i] < lo) lo = xs[i];
      if (xs[i] > hi) hi = xs[i];
    }
  }
  return [lo, hi];
}

/**
 * Specialized HPD for Beta(α, β). Delegates to `hpdInterval` with a 500-point
 * grid on (1e-4, 1 − 1e-4) — avoids the support-endpoint singularities for
 * α < 1 or β < 1. Memoized on (α, β, level) since the BvM and PriorPosterior
 * animators hit it heavily.
 */
export function hpdIntervalBeta(
  alpha: number,
  beta: number,
  level: number,
): [number, number] {
  const key = _cacheKey(alpha, beta, level);
  const cached = _hpdBetaCache.get(key);
  if (cached) return cached;
  const eps = 1e-4;
  const interval = hpdInterval(
    x => pdfBeta(x, alpha, beta),
    [eps, 1 - eps],
    level,
    500,
  );
  _memoSet(_hpdBetaCache, key, interval);
  return interval;
}

// ═══════════════════════════════════════════════════════════════════════════
// Point estimators — mapEstimate, posteriorMean, posteriorVariance.
// All three dispatch on the posterior's family tag.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * MAP estimate — argmax of the posterior density.
 * Scalar return for Beta-Binomial, Normal-Normal, Normal-Normal-IG (returns
 * the marginal mode on μ), Gamma-Poisson. Throws for Dirichlet-Multinomial:
 * the mode is a vector on the simplex — call `posteriorSample` and take
 * the empirical maximum instead.
 */
export function mapEstimate(family: ConjugateFamily, post: PosteriorHyperparams): number {
  switch (family) {
    case 'beta-binomial': {
      const p = post as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
      // Mode of Beta(α, β) = (α−1)/(α+β−2) when α,β > 1; else 0 or 1 on boundary.
      if (p.alpha0 > 1 && p.beta0 > 1) return (p.alpha0 - 1) / (p.alpha0 + p.beta0 - 2);
      if (p.alpha0 <= 1 && p.beta0 > 1) return 0;
      if (p.alpha0 > 1 && p.beta0 <= 1) return 1;
      return 0.5; // degenerate Beta(1, 1); arbitrary pick
    }
    case 'normal-normal': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;
      return p.mu0; // Gaussian mode = mean.
    }
    case 'normal-normal-ig': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal-ig' }>;
      return p.mu0; // marginal mode of μ (location of the Student-t marginal).
    }
    case 'gamma-poisson': {
      const p = post as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;
      // Mode of Gamma(α, β) = (α−1)/β when α > 1; else 0.
      return p.alpha0 > 1 ? (p.alpha0 - 1) / p.beta0 : 0;
    }
    case 'dirichlet-multinomial':
      throw new Error(
        'mapEstimate: Dirichlet-Multinomial mode is a vector; call posteriorSample instead.',
      );
    default: {
      const _exhaustive: never = family;
      throw new Error(`mapEstimate: unhandled family ${_exhaustive as string}`);
    }
  }
}

/**
 * Posterior mean — the Bayes estimator under squared-error loss (§25.6 Rem 9).
 * Scalar return for the four scalar-posterior families; throws for
 * Dirichlet-Multinomial (which has a vector-valued posterior mean).
 */
export function posteriorMean(family: ConjugateFamily, post: PosteriorHyperparams): number {
  switch (family) {
    case 'beta-binomial': {
      const p = post as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
      return p.alpha0 / (p.alpha0 + p.beta0);
    }
    case 'normal-normal': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;
      return p.mu0;
    }
    case 'normal-normal-ig': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal-ig' }>;
      return p.mu0;
    }
    case 'gamma-poisson': {
      const p = post as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;
      return p.alpha0 / p.beta0;
    }
    case 'dirichlet-multinomial':
      throw new Error(
        'posteriorMean: Dirichlet-Multinomial mean is a vector; access alpha0 / Σα directly.',
      );
    default: {
      const _exhaustive: never = family;
      throw new Error(`posteriorMean: unhandled family ${_exhaustive as string}`);
    }
  }
}

/**
 * Posterior variance — scalar posterior variance for the four scalar-posterior
 * families. For Normal-Normal-IG this returns Var[μ | y] in the Student-t
 * marginal sense (df · scale² / (df − 2) when df > 2).
 */
export function posteriorVariance(
  family: ConjugateFamily,
  post: PosteriorHyperparams,
): number {
  switch (family) {
    case 'beta-binomial': {
      const p = post as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
      const s = p.alpha0 + p.beta0;
      return (p.alpha0 * p.beta0) / (s * s * (s + 1));
    }
    case 'normal-normal': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;
      return p.sigma0_sq; // posterior variance σ_n² (name reused per brief note).
    }
    case 'normal-normal-ig': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal-ig' }>;
      // Marginal on μ is Student-t(2α, μ, β/(α·κ)); variance when 2α > 2.
      const df = 2 * p.alpha0;
      const scale2 = p.beta0 / (p.alpha0 * p.kappa0);
      return df > 2 ? (df / (df - 2)) * scale2 : Infinity;
    }
    case 'gamma-poisson': {
      const p = post as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;
      return p.alpha0 / (p.beta0 * p.beta0);
    }
    case 'dirichlet-multinomial':
      throw new Error(
        'posteriorVariance: Dirichlet-Multinomial variance is a k×k covariance matrix.',
      );
    default: {
      const _exhaustive: never = family;
      throw new Error(`posteriorVariance: unhandled family ${_exhaustive as string}`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Jeffreys priors (§25.7 Thm 4) — reparameterization-invariant non-informative
// priors for the five one-parameter regular families Topic 25 cites.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Jeffreys prior density π_J(θ) = √I(θ) for a one-parameter family.
 *
 * Shapes:
 *   bernoulli    → π(θ) ∝ θ^{−1/2}(1−θ)^{−1/2}   (proportional to Beta(½, ½);
 *                                                   proper after normalization)
 *   poisson      → π(λ) ∝ λ^{−1/2}                (improper; integral diverges
 *                                                   as λ → ∞)
 *   normal-mean  → π(μ) ∝ 1 / σ                   (flat in μ; requires
 *                                                   knownVariance; improper on ℝ)
 *   normal-scale → π(σ) ∝ 1 / σ                   (log-flat; improper; the
 *                                                   classical "reference" scale)
 *   exponential  → π(λ) ∝ 1 / λ                   (improper at both 0 and ∞)
 *
 * The Bernoulli Jeffreys prior is proper (Beta(½, ½) integrates to 1 on
 * (0, 1) after normalization); the other four are improper but yield proper
 * posteriors when combined with a likelihood that supplies integrability
 * (Def 7). The `∝` constant is dropped because downstream Bayes-rule
 * calculations only need the proportional kernel. See brief §25.7 Thm 4
 * for the Bernoulli and Normal-scale derivations; the other three are
 * named only (JEF1961 Ch. III).
 */
export function jeffreysPrior(
  family: 'bernoulli' | 'poisson' | 'normal-mean' | 'normal-scale' | 'exponential',
  theta: number,
  knownVariance?: number,
): number {
  switch (family) {
    case 'bernoulli':
      if (theta <= 0 || theta >= 1) return 0;
      return 1 / Math.sqrt(theta * (1 - theta));
    case 'poisson':
      if (theta <= 0) return 0;
      return 1 / Math.sqrt(theta);
    case 'normal-mean':
      if (knownVariance === undefined || knownVariance <= 0) {
        throw new Error('jeffreysPrior: normal-mean requires knownVariance > 0.');
      }
      return 1 / Math.sqrt(knownVariance);
    case 'normal-scale':
      if (theta <= 0) return 0;
      return 1 / theta;
    case 'exponential':
      if (theta <= 0) return 0;
      return 1 / theta;
    default: {
      const _exhaustive: never = family;
      throw new Error(`jeffreysPrior: unhandled family ${_exhaustive as string}`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Laplace approximation to log marginal likelihood (§25.8 Rem 16).
// The BIC-Laplace machinery from Topic 24 §24.4, now applied to the
// full posterior rather than just the partition-function expansion.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Laplace approximation:
 *   log m(y) ≈ log p(y | θ_MAP) + log π(θ_MAP) + (k/2)·log(2π) − (1/2)·log|H|,
 * where |H| is the determinant of the Hessian of −log p(θ, y) at θ_MAP and
 * k is the parameter dimension.
 *
 * Topic 24's BIC is the first-order Laplace simplification: drop the prior
 * term and approximate log|H| ≈ k · log n (brief §25.8 Rem 16). Topic 27
 * will return to this function for Bayes-factor computation.
 *
 * @param logLikAtMAP    log p(y | θ_MAP)
 * @param logPriorAtMAP  log π(θ_MAP)
 * @param hessianLogDet  log|H| where H = Hessian of −log p(θ, y) at θ_MAP
 * @param k              parameter dimension
 */
export function laplaceLogMarginalLikelihood(
  logLikAtMAP: number,
  logPriorAtMAP: number,
  hessianLogDet: number,
  k: number,
): number {
  return logLikAtMAP + logPriorAtMAP + (k / 2) * Math.log(2 * Math.PI) - 0.5 * hessianLogDet;
}

// ═══════════════════════════════════════════════════════════════════════════
// Posterior sampling (conjugate cases only).
// For non-conjugate posteriors, Topic 26's MCMC machinery is the tool.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Box-Muller sampler for Normal(μ, σ²). Returns a single draw per call.
 * Local helper; not exported (no Topic 6 equivalent to reuse, and exporting
 * a new Gaussian sampler out of Track 7 is out of scope).
 */
function _sampleNormal(mu: number, sigma2: number, rng: () => number): number {
  const u1 = Math.max(rng(), Number.EPSILON);
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + Math.sqrt(sigma2) * z;
}

/**
 * Beta(α, β) sampler via the two-Gamma trick: X ~ Gamma(α, 1), Y ~ Gamma(β, 1),
 * then X / (X + Y) ~ Beta(α, β). Uses `sampleGammaShape` from distributions.ts.
 */
function _sampleBeta(a: number, b: number, rng: () => number): number {
  const x = sampleGammaShape(a, rng);
  const y = sampleGammaShape(b, rng);
  return x / (x + y);
}

/**
 * Draw n samples from a conjugate posterior. Scalar return for the four
 * scalar families; `number[][]` for Dirichlet-Multinomial (each row is a
 * length-k simplex point) and Normal-Normal-IG (each row is `[μ, σ²]`).
 */
export function posteriorSample(
  family: ConjugateFamily,
  post: PosteriorHyperparams,
  n: number,
  rng: () => number = Math.random,
): number[] | number[][] {
  switch (family) {
    case 'beta-binomial': {
      const p = post as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
      return Array.from({ length: n }, () => _sampleBeta(p.alpha0, p.beta0, rng));
    }
    case 'normal-normal': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;
      return Array.from({ length: n }, () => _sampleNormal(p.mu0, p.sigma0_sq, rng));
    }
    case 'normal-normal-ig': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal-ig' }>;
      // Joint sample: σ² first from InverseGamma(α, β); then μ | σ² from
      // Normal(μ_n, σ²/κ_n). Each row is [μ, σ²].
      return Array.from({ length: n }, () => {
        // InverseGamma sample: if X ∼ Gamma(α, 1) then β/X ∼ InverseGamma(α, β).
        const g = sampleGammaShape(p.alpha0, rng);
        const sigma2 = p.beta0 / g;
        const mu = _sampleNormal(p.mu0, sigma2 / p.kappa0, rng);
        return [mu, sigma2];
      });
    }
    case 'gamma-poisson': {
      const p = post as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;
      return Array.from({ length: n }, () => sampleGammaShape(p.alpha0, rng) / p.beta0);
    }
    case 'dirichlet-multinomial': {
      const p = post as Extract<PosteriorHyperparams, { family: 'dirichlet-multinomial' }>;
      return Array.from({ length: n }, () => sampleDirichlet(p.alpha0, rng));
    }
    default: {
      const _exhaustive: never = family;
      throw new Error(`posteriorSample: unhandled family ${_exhaustive as string}`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Posterior predictive density / PMF at a future observation ỹ.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Posterior predictive p(ỹ | y) for the conjugate families Topic 25 develops:
 *
 *   beta-binomial → Beta-Binomial compound PMF (see pmfBetaBinomial).
 *   normal-normal → Normal(μ_n, σ_n² + σ²) density.
 *   gamma-poisson → NegativeBinomial PMF (see pmfNegativeBinomialPosterior).
 *
 * Dirichlet-Multinomial predictive is multivariate and requires the proper
 * Dirichlet-Multinomial PMF (equivalently, multivariate Beta / log-Γ ratios
 * over the future count vector); a dedicated helper is deferred to Topic 27
 * or 28 where multivariate predictive checks become the operational tool.
 * Normal-Normal-IG predictive is a non-standardized Student-t; call
 * `pdfStudentTMarginal` directly for that case.
 */
export function posteriorPredictive(
  family: ConjugateFamily,
  post: PosteriorHyperparams,
  yNew: number,
  options?: PredictiveOptions,
): number {
  switch (family) {
    case 'beta-binomial': {
      if (!options || options.family !== 'beta-binomial') {
        throw new Error('posteriorPredictive: beta-binomial requires options.m (new trial count).');
      }
      const p = post as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
      return pmfBetaBinomial(yNew, options.m, p.alpha0, p.beta0);
    }
    case 'normal-normal': {
      const p = post as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;
      // Predictive variance = posterior variance + likelihood variance.
      return pdfNormal(yNew, p.mu0, p.sigma0_sq + p.sigma_sq);
    }
    case 'gamma-poisson': {
      const p = post as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;
      return pmfNegativeBinomialPosterior(yNew, p.alpha0, p.beta0);
    }
    case 'normal-normal-ig':
      throw new Error(
        'posteriorPredictive: normal-normal-ig predictive is non-standardized Student-t; call pdfStudentTMarginal directly.',
      );
    case 'dirichlet-multinomial':
      throw new Error(
        'posteriorPredictive: dirichlet-multinomial predictive is multivariate; not in Topic 25 scope.',
      );
    default: {
      const _exhaustive: never = family;
      throw new Error(`posteriorPredictive: unhandled family ${_exhaustive as string}`);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 26 — Bayesian Computation & MCMC
//
// This section extends bayes.ts with Markov-chain Monte Carlo machinery:
// Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo, toy NUTS,
// convergence diagnostics (R-hat, ESS, autocorrelation, batch means), and
// a closed-form conditional-MVN utility used by GibbsStepper.
//
// All sampling functions are deterministic given an injected SeededRng — no
// Math.random, no module-level state. Handwritten Park-Miller LCG replaces
// the seedrandom dependency (see createSeededRng below and brief §6.1.2 /
// Appendix B.7 for the rationale).
//
// Notation extensions to §25.3 (brief §3.3):
//   q(· | ·)  proposal kernel        α(x, x')  acceptance probability
//   K(x, dx') transition kernel      π         invariant distribution
//   R̂         Gelman-Rubin diag.    N_eff     effective sample size
//   ρ_t       autocorr at lag t      (q, p)    HMC position-momentum pair
// ═══════════════════════════════════════════════════════════════════════════

// ─── New types ─────────────────────────────────────────────────────────────

/**
 * Seeded RNG interface — the bayes.ts contract for reproducible MCMC.
 * Implemented by `createSeededRng` (handwritten LCG) below.
 */
export interface SeededRng {
  /** Uniform on [0, 1). */
  random(): number;
  /** Standard Normal via Box-Muller with paired-value cache. */
  normal(): number;
  /** Re-seed (resets internal LCG state and clears the normal cache). */
  reseed(seed: number): void;
}

/**
 * A Markov chain: sequence of states of type T, plus per-iteration
 * diagnostics. `samples` has burn-in already removed; `accepted` is
 * index-aligned with `samples` (post-burn-in decisions only).
 */
export interface MarkovChain<T> {
  samples: T[];
  accepted: boolean[];
  acceptanceRate: number;
  /** Optional per-algorithm diagnostic payload (e.g., HMC energies). */
  metadata?: Record<string, unknown>;
}

/**
 * An MH proposal kernel. `propose` generates a new state x' given the
 * current state x; returns x' plus the log-density ratio
 *   logQRatio = log q(x | x') - log q(x' | x)
 * which is 0 for symmetric proposals and enters the MH acceptance.
 */
export interface ProposalKernel<T> {
  propose(x: T, rng: SeededRng): { xPrime: T; logQRatio: number };
}

/**
 * HMC configuration.
 *   epsilon     leapfrog step size (ε)
 *   steps       number of leapfrog steps (L)
 *   massMatrix  positive-definite mass matrix M; default identity
 */
export interface HMCParams {
  epsilon: number;
  steps: number;
  massMatrix?: number[][];
}

// ─── Seeded RNG (handwritten Park-Miller MINSTD LCG) ──────────────────────

/**
 * Park-Miller MINSTD parameters. State space is integers in [1, m-1].
 * Uses Schrage's algorithm to keep intermediate products in safe-integer
 * range (a * state would overflow at state ≈ 2^31; Schrage factors the
 * update as a * (s % q) - r * (s / q)).
 */
const LCG_M = 2147483647; // 2^31 - 1 (Mersenne prime)
const LCG_A = 48271;
const LCG_Q = 44488; // floor(M / A)
const LCG_R = 3399;  // M - A * Q

/**
 * Handwritten seeded RNG for reproducible MCMC.
 *
 * Park-Miller MINSTD LCG (a=48271, m=2^31-1) with Schrage factoring for
 * safe-integer arithmetic in JavaScript's 64-bit-float number type.
 * Normal sampling uses Box-Muller with a single-value cache (each u₁/u₂
 * uniform pair yields two independent standard Normals).
 *
 * Matches `np.random.default_rng(42)` for *chain-statistic* level agreement
 * (acceptance rates ~1e-2, posterior means ~5e-2, variances ~1e-1). NOT
 * bit-identical — PCG64 is not ported. Tolerance bands are specified in
 * `bayes.test.ts` T26.X tests.
 *
 * @param seed 32-bit integer; nonzero. Seed 0 is mapped to 1 to avoid the
 *             absorbing state.
 */
export function createSeededRng(seed: number): SeededRng {
  let state: number;
  let cachedNormal: number | null = null;

  const reseed = (s: number): void => {
    // Normalize to a valid LCG state in [1, m-1].
    let v = Math.floor(Math.abs(s)) % LCG_M;
    if (v === 0) v = 1;
    state = v;
    cachedNormal = null;
  };

  reseed(seed);

  const random = (): number => {
    // Schrage-factored step: avoids integer overflow.
    const hi = Math.floor(state! / LCG_Q);
    const lo = state! - hi * LCG_Q; // state % Q
    state = LCG_A * lo - LCG_R * hi;
    if (state < 0) state += LCG_M;
    return (state - 1) / (LCG_M - 1); // map to [0, 1)
  };

  const normal = (): number => {
    if (cachedNormal !== null) {
      const z = cachedNormal;
      cachedNormal = null;
      return z;
    }
    // Box-Muller: two uniforms → two independent standard Normals.
    let u1 = random();
    if (u1 < 1e-15) u1 = 1e-15; // guard log(0)
    const u2 = random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    cachedNormal = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  };

  return { random, normal, reseed };
}

// ─── Metropolis-Hastings ──────────────────────────────────────────────────

/**
 * Single MH step. Given current state x, proposal kernel q, and log-target
 * log π, returns next state and accept flag. Uses log-space acceptance to
 * avoid numerical underflow on low-probability regions.
 *
 * Acceptance: α(x, x') = min{1, π(x') q(x|x') / [π(x) q(x'|x)]}
 * Log form:   log α = logπ(x') - logπ(x) + logQRatio
 */
export function metropolisHastingsStep<T>(
  x: T,
  logPi: (x: T) => number,
  proposal: ProposalKernel<T>,
  rng: SeededRng,
): { xNext: T; accepted: boolean } {
  const { xPrime, logQRatio } = proposal.propose(x, rng);
  const logAlpha = logPi(xPrime) - logPi(x) + logQRatio;
  const u = rng.random();
  const accepted = Math.log(u < 1e-300 ? 1e-300 : u) < logAlpha;
  return { xNext: accepted ? xPrime : x, accepted };
}

/**
 * Full MH chain driver. Runs `burnIn + N` iterations, discards burn-in,
 * thins by `thin` (keeps every `thin`-th post-burn-in sample; `thin = 1`
 * keeps all). Acceptance rate is computed over post-burn-in decisions only.
 */
export function metropolisHastings<T>(
  x0: T,
  logPi: (x: T) => number,
  proposal: ProposalKernel<T>,
  N: number,
  burnIn: number,
  thin: number,
  rng: SeededRng,
): MarkovChain<T> {
  let x = x0;
  const samples: T[] = [];
  const accepted: boolean[] = [];
  let totalAccept = 0;
  let totalDecisions = 0;
  const total = burnIn + N;
  for (let i = 0; i < total; i++) {
    const step = metropolisHastingsStep(x, logPi, proposal, rng);
    x = step.xNext;
    if (i >= burnIn) {
      totalDecisions++;
      if (step.accepted) totalAccept++;
      // `accepted` stays index-aligned with `samples` (post-thinning) per
      // the MarkovChain<T> docstring contract; `acceptanceRate` is computed
      // over ALL post-burn-in decisions independent of thinning.
      if ((i - burnIn) % thin === 0) {
        samples.push(x);
        accepted.push(step.accepted);
      }
    }
  }
  const acceptanceRate = totalDecisions > 0 ? totalAccept / totalDecisions : 0;
  return { samples, accepted, acceptanceRate };
}

// ─── Gibbs sampling ───────────────────────────────────────────────────────

/**
 * Single systematic-scan Gibbs sweep. For i = 0..d-1, updates θ_i via its
 * user-supplied full-conditional sampler (which draws from π(θ_i | θ_{-i})
 * using the current value of every other coordinate).
 *
 * Mutates a local copy; returns the fully-updated vector. Thm 2 (§26.3)
 * proves each sub-step is an MH step with α ≡ 1.
 */
export function gibbsStep(
  theta: number[],
  conditionals: Array<(theta: number[], rng: SeededRng) => number>,
  rng: SeededRng,
): number[] {
  const next = [...theta];
  for (let i = 0; i < conditionals.length; i++) {
    next[i] = conditionals[i](next, rng);
  }
  return next;
}

/**
 * Full Gibbs chain driver. Each iteration = one systematic-scan sweep.
 * Gibbs always accepts (α ≡ 1 by Thm 2), so `accepted[i] = true` for all i.
 */
export function gibbsSampler(
  theta0: number[],
  conditionals: Array<(theta: number[], rng: SeededRng) => number>,
  N: number,
  burnIn: number,
  thin: number,
  rng: SeededRng,
): MarkovChain<number[]> {
  let theta = [...theta0];
  const samples: number[][] = [];
  const accepted: boolean[] = [];
  const total = burnIn + N;
  for (let i = 0; i < total; i++) {
    theta = gibbsStep(theta, conditionals, rng);
    if (i >= burnIn && (i - burnIn) % thin === 0) {
      // `accepted` index-aligned with `samples` per MarkovChain<T> contract.
      samples.push([...theta]);
      accepted.push(true);
    }
  }
  return { samples, accepted, acceptanceRate: 1 };
}

// ─── Hamiltonian Monte Carlo ──────────────────────────────────────────────

/**
 * Invert a symmetric positive-definite matrix via Gauss-Jordan. Used for
 * HMC mass-matrix handling when M ≠ I. For M = I, callers skip this path.
 */
function invertSymmetric(M: number[][]): number[][] {
  const n = M.length;
  const A: number[][] = M.map((row, i) => {
    const augmented = [...row];
    for (let j = 0; j < n; j++) augmented.push(i === j ? 1 : 0);
    return augmented;
  });
  for (let i = 0; i < n; i++) {
    let pivot = A[i][i];
    if (Math.abs(pivot) < 1e-14) {
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(A[r][i]) > 1e-14) {
          [A[i], A[r]] = [A[r], A[i]];
          pivot = A[i][i];
          break;
        }
      }
    }
    if (Math.abs(pivot) < 1e-14) {
      throw new Error('invertSymmetric: matrix is singular.');
    }
    for (let j = 0; j < 2 * n; j++) A[i][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const factor = A[r][i];
      for (let j = 0; j < 2 * n; j++) A[r][j] -= factor * A[i][j];
    }
  }
  return A.map((row) => row.slice(n));
}

/**
 * Matrix-vector product; returns new array. Assumes dimensions match.
 */
function matVec(A: number[][], v: number[]): number[] {
  const out = new Array(A.length).fill(0);
  for (let i = 0; i < A.length; i++) {
    let s = 0;
    for (let j = 0; j < v.length; j++) s += A[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

/**
 * Lower-triangular Cholesky factor L of an SPD matrix M (so L · Lᵀ = M).
 * Used for the HMC momentum resample: if z ∼ 𝒩(0, I), then p = L · z ∼ 𝒩(0, M).
 * Throws if M is not positive definite (negative pivot encountered).
 */
function choleskyLower(M: number[][]): number[][] {
  const n = M.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) {
          throw new Error('choleskyLower: matrix is not positive definite');
        }
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

/**
 * Sample p ∼ 𝒩(0, M) via Cholesky: draw z ∼ 𝒩(0, I), return L · z.
 * If `L` is null (M = I), return z directly.
 */
function sampleMomentum(
  L: number[][] | null,
  d: number,
  rng: SeededRng,
): number[] {
  const z = new Array<number>(d);
  for (let i = 0; i < d; i++) z[i] = rng.normal();
  if (!L) return z;
  const p = new Array<number>(d).fill(0);
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let j = 0; j <= i; j++) s += L[i][j] * z[j];
    p[i] = s;
  }
  return p;
}

/**
 * Inner leapfrog driver — takes a precomputed M⁻¹ (or null for M = I) so
 * callers in a loop don't pay an O(d³) inversion per step.
 */
function leapfrogWithInverse(
  q0: number[],
  p0: number[],
  gradU: (q: number[]) => number[],
  epsilon: number,
  steps: number,
  mInv: number[][] | null,
): { qStar: number[]; pStar: number[] } {
  const d = q0.length;
  const q = [...q0];
  const p = [...p0];
  for (let k = 0; k < steps; k++) {
    const g0 = gradU(q);
    for (let i = 0; i < d; i++) p[i] -= 0.5 * epsilon * g0[i];
    const v = mInv ? matVec(mInv, p) : p;
    for (let i = 0; i < d; i++) q[i] += epsilon * v[i];
    const g1 = gradU(q);
    for (let i = 0; i < d; i++) p[i] -= 0.5 * epsilon * g1[i];
  }
  return { qStar: q, pStar: p };
}

/**
 * Leapfrog integrator for Hamiltonian dynamics. One call = L leapfrog
 * steps of size ε, producing the end state (q*, p*).
 *
 * Per step (Störmer-Verlet, brief §3.2 Proof 3):
 *   p^{k+1/2} = p^k - (ε/2) ∇U(q^k)
 *   q^{k+1}   = q^k + ε M⁻¹ p^{k+1/2}
 *   p^{k+1}   = p^{k+1/2} - (ε/2) ∇U(q^{k+1})
 *
 * The map is volume-preserving (Jacobian det = 1; see T26.6) and time-
 * reversible after momentum-flip — the two properties that make HMC's
 * extended-state proposal symmetric (§26.4 Proof 3).
 *
 * Performance note: when used in a chain loop, M⁻¹ is recomputed every
 * call. The `hamiltonianMonteCarlo` driver precomputes it via
 * `leapfrogWithInverse` directly to avoid the O(d³) per-iteration cost.
 * Standalone callers pay it once per call.
 *
 * @param q0      starting position
 * @param p0      starting momentum
 * @param gradU   gradient of U(q) = -log π(q)
 * @param params  { epsilon, steps, massMatrix? }. massMatrix default = I.
 */
export function hmcLeapfrog(
  q0: number[],
  p0: number[],
  gradU: (q: number[]) => number[],
  params: HMCParams,
): { qStar: number[]; pStar: number[] } {
  const mInv = params.massMatrix ? invertSymmetric(params.massMatrix) : null;
  return leapfrogWithInverse(q0, p0, gradU, params.epsilon, params.steps, mInv);
}

/**
 * Full HMC chain driver. Each iteration:
 *   (i)   Resample p ∼ 𝒩(0, M)   — Gibbs on p, preserves joint π̃(q, p)
 *   (ii)  Leapfrog L steps with step size ε
 *   (iii) Flip p → -p (makes the proposal symmetric in extended space)
 *   (iv)  Accept with probability min{1, exp(H_start − H_end)}
 *
 * H(q, p) = -log π(q) + ½ p^T M⁻¹ p. For M = I the quadratic simplifies to
 * ½ Σ pᵢ². Energy conservation along exact Hamiltonian flow would give
 * α ≡ 1; leapfrog introduces O(ε²) oscillation that the accept step corrects.
 *
 * Mass matrix M (when set in params): inverse and Cholesky factor are
 * precomputed once outside the loop, so per-iteration cost is O(d²) (the
 * matrix-vector products), not O(d³). Momentum is resampled correctly as
 * p = L · z with z ∼ 𝒩(0, I), preserving the 𝒩(0, M) marginal that
 * detailed balance on the extended state requires.
 *
 * Returns a MarkovChain whose `metadata.proposedEnergies[i]` is the
 * Hamiltonian of the *proposal* on iteration i (regardless of accept /
 * reject). This is the right quantity for energy-drift diagnostics; do
 * NOT confuse with the energy of the kept sample.
 */
export function hamiltonianMonteCarlo(
  q0: number[],
  logPi: (q: number[]) => number,
  gradU: (q: number[]) => number[],
  params: HMCParams,
  N: number,
  burnIn: number,
  rng: SeededRng,
): MarkovChain<number[]> {
  const d = q0.length;
  let q = [...q0];
  const samples: number[][] = [];
  const accepted: boolean[] = [];
  const proposedEnergies: number[] = [];

  // Precompute M⁻¹ and Cholesky factor L once.
  const mInv = params.massMatrix ? invertSymmetric(params.massMatrix) : null;
  const L = params.massMatrix ? choleskyLower(params.massMatrix) : null;

  const kinetic = (p: number[]): number => {
    if (!mInv) {
      let s = 0;
      for (const pi of p) s += pi * pi;
      return 0.5 * s;
    }
    const v = matVec(mInv, p);
    let s = 0;
    for (let i = 0; i < d; i++) s += p[i] * v[i];
    return 0.5 * s;
  };

  let totalAccept = 0;
  let totalDecisions = 0;
  const total = burnIn + N;
  for (let iter = 0; iter < total; iter++) {
    const p0 = sampleMomentum(L, d, rng);
    const H0 = -logPi(q) + kinetic(p0);
    const { qStar, pStar } = leapfrogWithInverse(
      q,
      p0,
      gradU,
      params.epsilon,
      params.steps,
      mInv,
    );
    const HStar = -logPi(qStar) + kinetic(pStar);
    const logAlpha = H0 - HStar;
    const u = rng.random();
    const accept = Math.log(u < 1e-300 ? 1e-300 : u) < logAlpha;
    if (accept) q = qStar;
    if (iter >= burnIn) {
      totalDecisions++;
      if (accept) totalAccept++;
      samples.push([...q]);
      accepted.push(accept);
      proposedEnergies.push(HStar);
    }
  }
  const acceptanceRate = totalDecisions > 0 ? totalAccept / totalDecisions : 0;
  return {
    samples,
    accepted,
    acceptanceRate,
    metadata: { proposedEnergies },
  };
}

// ─── NUTS — intentionally not implemented ────────────────────────────────
//
// PR #30 review (Gemini high-priority): a "toy NUTS" that omits slice
// sampling + recursive subtree U-turn checks doesn't preserve π — the
// uniform-from-tree shortcut yields a non-symmetric proposal that the
// tail MH step cannot correct in general. Rather than ship a sampler
// labelled "NUTS" with a known-broken acceptance argument, Topic 26 ships
// no JavaScript NUTS at all: §26.5 states Thm 4 (NUTS correctness) and
// cites HOF2014 Thm 1 stated-only, the §26.9 8-schools workflow uses
// conjugate Gibbs in the notebook (per brief Gotcha G3), and production
// NUTS is delegated to Stan / PyMC / NumPyro. The hand-written Park-Miller
// LCG above suffices for MH / Gibbs / HMC tests; NUTS adaptive doubling
// is genuinely non-trivial and out of scope for a pedagogical TS module.

// ─── Diagnostics ──────────────────────────────────────────────────────────

/**
 * Gelman-Rubin R̂ (1992) for M chains of length N on a scalar parameter.
 *
 *   R̂ = sqrt{ (N-1)/N  +  B / (N · W) }
 *
 * where B = (N / (M-1)) · Σ_i (x̄_i - x̄)²   (between-chain variance of means),
 *       W = (1 / M) · Σ_i s_i²               (mean of within-chain variances,
 *                                            using the unbiased estimator).
 *
 * R̂ → 1 as chains mix. Stan's practical threshold is R̂ < 1.01 per chain.
 */
export function rHat(chains: number[][]): number {
  const M = chains.length;
  if (M < 2) return 1;
  const N = chains[0].length;
  if (N < 2) return 1;
  const chainMeans = chains.map((c) => c.reduce((s, x) => s + x, 0) / c.length);
  const grandMean = chainMeans.reduce((s, m) => s + m, 0) / M;
  const B =
    (N / (M - 1)) *
    chainMeans.reduce((s, m) => s + (m - grandMean) ** 2, 0);
  const chainVars = chains.map((c, i) => {
    const mu = chainMeans[i];
    return c.reduce((s, x) => s + (x - mu) ** 2, 0) / (c.length - 1);
  });
  const W = chainVars.reduce((s, v) => s + v, 0) / M;
  if (W <= 0) return Number.POSITIVE_INFINITY;
  const varHat = ((N - 1) / N) * W + B / N;
  return Math.sqrt(varHat / W);
}

/**
 * Coordinate-wise R̂ for vector-valued chains.
 *
 * @param chains [M chains][N iterations][d coordinates]
 * @returns     array of d R̂ values.
 */
export function rHatMultivariate(chains: number[][][]): number[] {
  if (chains.length === 0 || chains[0].length === 0) return [];
  const d = chains[0][0].length;
  const result: number[] = [];
  for (let k = 0; k < d; k++) {
    const slice = chains.map((c) => c.map((v) => v[k]));
    result.push(rHat(slice));
  }
  return result;
}

/**
 * Sample autocorrelation at lag t. Uses a single chain-wide mean and
 * normalizes by the full-chain variance (standard time-series estimator;
 * biased toward 0 at large lags but sufficient for ESS truncation).
 *
 *   ρ̂_t = Σ_{i=0..N-t-1} (x_i - x̄)(x_{i+t} - x̄)  /  Σ_{i=0..N-1} (x_i - x̄)²
 */
export function autocorrelation(chain: number[], lag: number): number {
  const N = chain.length;
  if (lag <= 0 || lag >= N) return lag === 0 ? 1 : 0;
  const mean = chain.reduce((s, x) => s + x, 0) / N;
  let num = 0;
  let den = 0;
  for (let i = 0; i < N - lag; i++) {
    num += (chain[i] - mean) * (chain[i + lag] - mean);
  }
  for (let i = 0; i < N; i++) {
    den += (chain[i] - mean) ** 2;
  }
  return den > 0 ? num / den : 0;
}

/**
 * Effective sample size via Geyer's initial-positive-sequence estimator.
 *
 *   N_eff = N / τ,   τ = 1 + 2 · Σ_{t=1..∞} ρ_t
 *
 * Geyer IPS truncates the sum when the running pair Γ_m = ρ_{2m} + ρ_{2m+1}
 * first hits zero or turns negative. This produces a monotone non-increasing
 * truncation for reversible chains and avoids the noisy negative-ρ tail.
 *
 * For AR(1) with φ=0.9, the theoretical τ = (1+φ)/(1-φ) = 19, so N_eff ≈ N/19.
 */
export function effectiveSampleSize(
  chain: number[],
  maxLag?: number,
): number {
  const N = chain.length;
  if (N < 4) return N;
  const maxT = maxLag ?? Math.floor(N / 4);

  // Precompute mean and centered-sum-of-squares (denominator) ONCE — calling
  // autocorrelation(chain, t) per lag would recompute both each call,
  // making the loop O(N · maxT) with a 3× constant-factor overhead from
  // re-walks of the chain. After this hoist, each lag is O(N − t).
  const mean = chain.reduce((s, x) => s + x, 0) / N;
  let den = 0;
  for (let i = 0; i < N; i++) den += (chain[i] - mean) ** 2;
  if (den <= 0) return N;

  const rhos: number[] = [1];
  for (let t = 1; t <= maxT; t++) {
    let num = 0;
    for (let i = 0; i < N - t; i++) {
      num += (chain[i] - mean) * (chain[i + t] - mean);
    }
    rhos.push(num / den);
  }

  // Geyer IPS: accumulate pairs Γ_m = ρ_{2m} + ρ_{2m+1}; stop at first ≤ 0.
  let sumPairs = 0;
  for (let m = 0; 2 * m + 1 < rhos.length; m++) {
    const pair = rhos[2 * m] + rhos[2 * m + 1];
    if (pair <= 0) break;
    sumPairs += pair;
  }
  const tau = 2 * sumPairs - 1; // converts Γ-pair sum to 1 + 2·Σρ_t
  return N / Math.max(tau, 1);
}

/**
 * Batch-means estimator of Monte Carlo variance σ²_MC for a single chain.
 *
 * Splits the chain into b batches of equal size ⌊N/b⌋, computes per-batch
 * means, then σ²_MC ≈ (N/b) · Var(batch means). Default b = ⌊√N⌋ (Jones
 * et al. 2006 recommendation). Returns σ²_MC, not σ²_MC / N.
 *
 * For iid N(0, 1) with N=5000, σ²_MC ≈ 1. For AR(1) φ=0.9, σ²_MC inflates
 * to ≈ 19 · Var(X) = 19.
 */
export function batchMeansVariance(chain: number[], nBatches?: number): number {
  const N = chain.length;
  const b = nBatches ?? Math.max(2, Math.floor(Math.sqrt(N)));
  const batchSize = Math.floor(N / b);
  if (batchSize < 1 || b < 2) return Number.NaN;
  const batchMeans: number[] = [];
  for (let k = 0; k < b; k++) {
    let s = 0;
    for (let i = 0; i < batchSize; i++) s += chain[k * batchSize + i];
    batchMeans.push(s / batchSize);
  }
  const grand = batchMeans.reduce((s, m) => s + m, 0) / b;
  const bVar =
    batchMeans.reduce((s, m) => s + (m - grand) ** 2, 0) / (b - 1);
  return batchSize * bVar;
}

/**
 * Descriptive trace summary. Works on scalar-valued or vector-valued chains;
 * for vector-valued chains, returns per-coordinate arrays.
 */
export function traceSummary(chain: MarkovChain<number | number[]>): {
  mean: number | number[];
  std: number | number[];
  first: number | number[];
  last: number | number[];
  acceptanceRate: number;
  acf1: number | number[];
} {
  const { samples, acceptanceRate } = chain;
  if (samples.length === 0) {
    return {
      mean: 0, std: 0, first: 0, last: 0, acceptanceRate, acf1: 0,
    };
  }
  const isVector = Array.isArray(samples[0]);
  if (!isVector) {
    const s = samples as number[];
    const N = s.length;
    const mean = s.reduce((a, b) => a + b, 0) / N;
    const variance = s.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(N - 1, 1);
    return {
      mean,
      std: Math.sqrt(variance),
      first: s[0],
      last: s[N - 1],
      acceptanceRate,
      acf1: autocorrelation(s, 1),
    };
  }
  const vec = samples as number[][];
  const d = vec[0].length;
  const N = vec.length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  const acf1 = new Array(d).fill(0);
  for (let k = 0; k < d; k++) {
    const col = vec.map((v) => v[k]);
    const mu = col.reduce((a, b) => a + b, 0) / N;
    const va = col.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(N - 1, 1);
    mean[k] = mu;
    std[k] = Math.sqrt(va);
    acf1[k] = autocorrelation(col, 1);
  }
  return {
    mean,
    std,
    first: [...vec[0]],
    last: [...vec[N - 1]],
    acceptanceRate,
    acf1,
  };
}

// ─── Conditional MVN (Topic 8 Thm 3 closed form) ──────────────────────────

/**
 * Conditional multivariate Normal. Given joint MVN(μ, Σ) for X partitioned
 * into free coords A and fixed coords B with X_B = v_B, returns the
 * conditional MVN parameters on X_A:
 *
 *   μ_{A|B} = μ_A + Σ_{AB} Σ_{BB}⁻¹ (v_B - μ_B)
 *   Σ_{A|B} = Σ_{AA} - Σ_{AB} Σ_{BB}⁻¹ Σ_{BA}
 *
 * Used by GibbsStepper (§26.3 Ex 3) to draw θ_i | θ_{-i} in closed form,
 * and available for any Gibbs-on-Gaussian construction.
 *
 * @param mu           length-d mean vector
 * @param sigma        d×d covariance matrix (positive definite)
 * @param fixedIndices indices in 0..d-1 of the coordinates being conditioned on
 * @param fixedValues  values at those indices (same length as fixedIndices)
 */
export function conditionalMVN(
  mu: number[],
  sigma: number[][],
  fixedIndices: number[],
  fixedValues: number[],
): { muCond: number[]; sigmaCond: number[][] } {
  const d = mu.length;
  const fixedSet = new Set(fixedIndices);
  const freeIndices: number[] = [];
  for (let i = 0; i < d; i++) if (!fixedSet.has(i)) freeIndices.push(i);

  const muA = freeIndices.map((i) => mu[i]);
  const muB = fixedIndices.map((i) => mu[i]);

  const pick = (rows: number[], cols: number[]): number[][] =>
    rows.map((r) => cols.map((c) => sigma[r][c]));

  const SigmaAA = pick(freeIndices, freeIndices);
  const SigmaAB = pick(freeIndices, fixedIndices);
  const SigmaBB = pick(fixedIndices, fixedIndices);
  // Symmetric ⇒ SigmaBA = SigmaAB^T.

  const SigmaBBinv = invertSymmetric(SigmaBB);

  // μ_cond = μ_A + Σ_{AB} Σ_{BB}⁻¹ (v_B - μ_B)
  const diff = fixedValues.map((v, i) => v - muB[i]);
  const shift = matVec(SigmaAB, matVec(SigmaBBinv, diff));
  const muCond = muA.map((x, i) => x + shift[i]);

  // Σ_cond = Σ_{AA} - Σ_{AB} Σ_{BB}⁻¹ Σ_{BA}
  // Compute Σ_{AB} · Σ_{BB}⁻¹ first, then multiply by Σ_{BA} = Σ_{AB}^T.
  const nFree = freeIndices.length;
  const nFix = fixedIndices.length;
  const ABBinv: number[][] = Array.from({ length: nFree }, () =>
    new Array(nFix).fill(0),
  );
  for (let i = 0; i < nFree; i++) {
    for (let j = 0; j < nFix; j++) {
      let s = 0;
      for (let k = 0; k < nFix; k++) s += SigmaAB[i][k] * SigmaBBinv[k][j];
      ABBinv[i][j] = s;
    }
  }
  const correction: number[][] = Array.from({ length: nFree }, () =>
    new Array(nFree).fill(0),
  );
  for (let i = 0; i < nFree; i++) {
    for (let j = 0; j < nFree; j++) {
      let s = 0;
      for (let k = 0; k < nFix; k++) s += ABBinv[i][k] * SigmaAB[j][k]; // Σ_{BA}[k][j] = Σ_{AB}[j][k]
      correction[i][j] = s;
    }
  }
  const sigmaCond = SigmaAA.map((row, i) =>
    row.map((v, j) => v - correction[i][j]),
  );

  return { muCond, sigmaCond };
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 27 — Bayesian Model Comparison & BMA (extending this module).
//
// Handoff brief: docs/formalstatistics-bayesian-model-comparison-and-bma-handoff-brief.md
//
// New exports, per brief §6.1 (16 functions):
//   Closed-form comparator     — lindleyBayesFactor
//   Log-marginal estimators    — betaBinomialLogMarginal, marginalLikelihoodLaplace,
//                                bridgeSamplingEstimate, harmonicMeanEstimate,
//                                importanceSamplingEstimate, pathSamplingEstimate,
//                                nestedSamplingEstimate
//   Aggregation                — bayesFactor, posteriorModelProbabilities,
//                                bmaPredictive
//   Predictive scoring         — dic, waic, psisLoo
//   Bayesian multiplicity / PPC — localFdr, posteriorPredictiveCheck
//
// Module-local helpers (not exported): logsumexp, logsumexpPair, logDetPD.
// Absent from distributions.ts — Topics 1–26 stayed in the linear numerical
// regime; Topic 27 is the first to need log-space mixture normalization.
// ═══════════════════════════════════════════════════════════════════════════

/** Numerically stable log(Σ exp(x_i)). Returns −∞ on empty. */
function logsumexp(x: number[]): number {
  if (x.length === 0) return -Infinity;
  let m = x[0];
  for (let i = 1; i < x.length; i++) if (x[i] > m) m = x[i];
  if (!Number.isFinite(m)) return m;
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.exp(x[i] - m);
  return m + Math.log(s);
}

/** Two-argument logsumexp — common hot path in bridge sampling. */
function logsumexpPair(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = a > b ? a : b;
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** Log determinant of a symmetric positive-definite matrix via Cholesky.
 *  Returns −∞ if the matrix is not PD (caught by a non-positive diagonal
 *  during factorization). */
function logDetPD(M: number[][]): number {
  const n = M.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  let logDet = 0;
  for (let i = 0; i < n; i++) {
    let diag = M[i][i];
    for (let k = 0; k < i; k++) diag -= L[i][k] * L[i][k];
    if (!(diag > 0)) return -Infinity;
    const Lii = Math.sqrt(diag);
    L[i][i] = Lii;
    logDet += Math.log(Lii);
    for (let j = i + 1; j < n; j++) {
      let s = M[j][i];
      for (let k = 0; k < i; k++) s -= L[j][k] * L[i][k];
      L[j][i] = s / Lii;
    }
  }
  return 2 * logDet;
}

/**
 * Lindley paradox closed-form BF₁₀ for H₀: θ = 0 vs H₁: θ ~ N(0, τ²), with
 * observed data summarized as the standardized statistic z = ȳ · √n / σ and
 * known sampling variance σ²/n. Brief §3.2 Proof 1 derives
 *
 *   BF₁₀ = (1 + r)^(−1/2) · exp(z² r / (2(1 + r))),   r = τ² n / σ².
 *
 * The paradox: at fixed z, τ → ∞ drives BF₁₀ → 0 — the frequentist rejects
 * H₀ at p ≈ 0.003 while the Bayesian overwhelmingly favors H₀.
 */
export function lindleyBayesFactor(
  z: number,
  n: number,
  tau: number,
  sigma = 1,
): number {
  const r = (tau * tau * n) / (sigma * sigma);
  return Math.exp(-0.5 * Math.log(1 + r) + (0.5 * z * z * r) / (1 + r));
}

/**
 * Exact log marginal likelihood for the Beta(α, β) + Binomial(n, k) model:
 *
 *   log m(y) = log C(n, k) + log B(α + k, β + n − k) − log B(α, β),
 *
 * with B(a, b) = Γ(a) Γ(b) / Γ(a + b). Closed-form reference for the bridge,
 * path, and Laplace estimators' convergence checks (brief §6.2 T27.4 / T27.5).
 */
export function betaBinomialLogMarginal(
  k: number,
  n: number,
  alpha: number,
  beta: number,
): number {
  const logComb = lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1);
  const logB = (a: number, b: number) => lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  return logComb + logB(alpha + k, beta + n - k) - logB(alpha, beta);
}

/**
 * Laplace approximation to log m(y) given the unnormalized log posterior
 * value at its mode and the (negative) Hessian of log posterior there:
 *
 *   log m(y) ≈ (k/2) log(2π) − (1/2) log |H| + log ũ(θ̂).
 *
 * k is the parameter-space dimension. Brief §3.2 Proof 2 shows BIC is the
 * O(1)-truncated asymptotic version of this expansion (Topic 24 §24.4).
 *
 * The Hessian must be symmetric positive-definite; non-PD input returns −∞
 * as a clear sentinel rather than silently producing a bogus estimate.
 */
export function marginalLikelihoodLaplace(opts: {
  logUnnormPostAtMode: number;
  hessianAtMode: number[][];
  k: number;
}): number {
  const { logUnnormPostAtMode, hessianAtMode, k } = opts;
  const logDet = logDetPD(hessianAtMode);
  if (!Number.isFinite(logDet)) return -Infinity;
  return (k / 2) * Math.log(2 * Math.PI) - 0.5 * logDet + logUnnormPostAtMode;
}

/**
 * Bayes factor BF₁₀ = m₁/m₀ from two log marginals. Trivial wrapper, kept
 * for API clarity so callers don't need to remember the sign convention.
 */
export function bayesFactor(logM0: number, logM1: number): number {
  return Math.exp(logM1 - logM0);
}

/**
 * Posterior model probabilities from per-model log marginals and an optional
 * prior over models (defaults to uniform):
 *
 *   P(M_k | y) ∝ π(M_k) m_k(y),  computed as softmax(log π + log m).
 *
 * Uses logsumexp for numerical stability — necessary when log marginals
 * differ by many orders of magnitude (common in nested-model comparisons).
 */
export function posteriorModelProbabilities(
  logMarginals: number[],
  priorModelProbs?: number[],
): number[] {
  const K = logMarginals.length;
  if (K === 0) return [];
  let logPriors: number[];
  if (priorModelProbs === undefined) {
    logPriors = new Array(K).fill(-Math.log(K));
  } else {
    if (priorModelProbs.length !== K) {
      throw new Error(
        `posteriorModelProbabilities: priorModelProbs.length (${priorModelProbs.length}) `
        + `must match logMarginals.length (${K}).`,
      );
    }
    for (const p of priorModelProbs) {
      if (!(Number.isFinite(p) && p > 0)) {
        throw new Error(
          'posteriorModelProbabilities: priorModelProbs must be strictly positive and finite.',
        );
      }
    }
    logPriors = priorModelProbs.map(p => Math.log(p));
  }
  const logUnnorm = logMarginals.map((lm, k) => lm + logPriors[k]);
  const logZ = logsumexp(logUnnorm);
  return logUnnorm.map(lu => Math.exp(lu - logZ));
}

/**
 * Harmonic-mean estimator of log m(y):  m̂_HM = 1 / mean(1/L(θ_s)),
 * i.e. log m̂ = log N − logsumexp(−log L_s).
 *
 * Included for pedagogy only — Newton & Raftery (1994) show the variance
 * is almost always infinite (the integrand 1/L(θ) has a heavy right tail
 * under the posterior). Brief §6.1 flags it as known-pathological; use
 * bridge or importance sampling in production.
 */
export function harmonicMeanEstimate(
  logLikDraws: number[],
): { logMarginal: number; mcSe: number } {
  const n = logLikDraws.length;
  if (n === 0) return { logMarginal: -Infinity, mcSe: NaN };
  const negLogL = logLikDraws.map(l => -l);
  const logSum = logsumexp(negLogL);
  const logMarginal = Math.log(n) - logSum;
  // Delta-method SE on the log-mean; unstable when tail exists (expected).
  const maxNeg = negLogL.reduce((m, x) => (x > m ? x : m), -Infinity);
  let meanShift = 0;
  for (const x of negLogL) meanShift += Math.exp(x - maxNeg);
  meanShift /= n;
  let varShift = 0;
  for (const x of negLogL) varShift += (Math.exp(x - maxNeg) - meanShift) ** 2;
  varShift /= Math.max(1, n - 1);
  const mcSe = Math.sqrt(varShift / n) / meanShift;
  return { logMarginal, mcSe };
}

/**
 * Naive importance-sampling estimator: draw θ_s ~ g(θ), estimate
 *
 *   log m(y) ≈ logsumexp_s( log ũ(θ_s) − log g(θ_s) ) − log N.
 *
 * Unbiased but high-variance when g overlaps poorly with the unnormalized
 * posterior ũ = L π. In practice the bridge estimator (Meng–Wong 1996)
 * dominates this on the same sample budget; keep IS as the contrast in
 * the `BridgeSamplingConvergence` component and as a sanity baseline.
 */
export function importanceSamplingEstimate(opts: {
  proposalDraws: number[][];
  logUnnormPost: (theta: number[]) => number;
  logProposal: (theta: number[]) => number;
}): { logMarginal: number; mcSe: number } {
  const { proposalDraws, logUnnormPost, logProposal } = opts;
  const n = proposalDraws.length;
  if (n === 0) return { logMarginal: -Infinity, mcSe: NaN };
  const logRatios = proposalDraws.map(t => logUnnormPost(t) - logProposal(t));
  const logMarginal = logsumexp(logRatios) - Math.log(n);
  // Delta-method SE of log(mean(exp r)) ≈ sd(exp r) / (mean(exp r) √N).
  const maxR = logRatios.reduce((m, x) => (x > m ? x : m), -Infinity);
  let mean = 0;
  for (const r of logRatios) mean += Math.exp(r - maxR);
  mean /= n;
  let variance = 0;
  for (const r of logRatios) variance += (Math.exp(r - maxR) - mean) ** 2;
  variance /= Math.max(1, n - 1);
  const mcSe = Math.sqrt(variance / n) / mean;
  return { logMarginal, mcSe };
}

/**
 * Meng–Wong (1996) iterative bridge sampling estimator of log m(y).
 *
 * Uses equal sample sizes s₁ = s₂ = 1/2 (the log(1/2) factors cancel between
 * numerator and denominator, dropping out of the fixed-point iteration).
 * The iteration is
 *
 *   log m_{t+1} = logsumexp_j( log ũ(θ_j^g) − logsumexp(log ũ − log m_t, log g) )
 *               − logsumexp_i( log g(θ_i^p) − logsumexp(log ũ − log m_t, log g) )
 *
 * initialized with a naive IS estimate on the proposal side. Ports notebook
 * Cell 7 literally, with `logsumexp` throughout for stability (the fragility
 * flagged in the brief shows up when the proposal and target have near-zero
 * overlap — the fixed-point iteration stays well-conditioned as long as the
 * Meng–Wong identity's denominator is bounded away from 0).
 *
 * Brief §3.2 Proof 3 derives the identity m = E_g[α(θ) ũ(θ)] / E_p[α(θ) g(θ)].
 */
export function bridgeSamplingEstimate(opts: {
  posteriorDraws: number[][];
  proposalDraws: number[][];
  logUnnormPost: (theta: number[]) => number;
  logProposal: (theta: number[]) => number;
  maxIter?: number;
  tol?: number;
}): { logMarginal: number; iterations: number; mcSe: number } {
  const { posteriorDraws, proposalDraws, logUnnormPost, logProposal } = opts;
  const maxIter = opts.maxIter ?? 50;
  const tol = opts.tol ?? 1e-10;

  const N1 = posteriorDraws.length;
  const N2 = proposalDraws.length;
  if (N1 === 0 || N2 === 0) {
    return { logMarginal: NaN, iterations: 0, mcSe: NaN };
  }

  const logP1 = posteriorDraws.map(t => logUnnormPost(t));
  const logG1 = posteriorDraws.map(t => logProposal(t));
  const logP2 = proposalDraws.map(t => logUnnormPost(t));
  const logG2 = proposalDraws.map(t => logProposal(t));

  // Initialize with naive-IS estimate on the proposal side. Reuse the
  // proposal-side buffer as numerTerms across iterations; denomTerms is a
  // separate buffer reused across iterations for the posterior-side sum.
  const numerTerms: number[] = new Array(N2);
  for (let j = 0; j < N2; j++) numerTerms[j] = logP2[j] - logG2[j];
  let logM = logsumexp(numerTerms) - Math.log(N2);
  const denomTerms: number[] = new Array(N1);

  let iterations = 0;
  for (iterations = 0; iterations < maxIter; iterations++) {
    for (let j = 0; j < N2; j++) {
      numerTerms[j] = logP2[j] - logsumexpPair(logP2[j] - logM, logG2[j]);
    }
    const numer = logsumexp(numerTerms) - Math.log(N2);

    for (let i = 0; i < N1; i++) {
      denomTerms[i] = logG1[i] - logsumexpPair(logP1[i] - logM, logG1[i]);
    }
    const denom = logsumexp(denomTerms) - Math.log(N1);

    const logMNew = numer - denom;
    if (Math.abs(logMNew - logM) < tol) {
      logM = logMNew;
      iterations++;
      break;
    }
    logM = logMNew;
  }

  // Delta-method SE from the fixed-point numerator and denominator terms.
  const wNum: number[] = new Array(N2);
  for (let j = 0; j < N2; j++) {
    wNum[j] = Math.exp(logP2[j] - logM - logsumexpPair(logP2[j] - logM, logG2[j]));
  }
  const wDen: number[] = new Array(N1);
  for (let i = 0; i < N1; i++) {
    wDen[i] = Math.exp(logG1[i] - logsumexpPair(logP1[i] - logM, logG1[i]));
  }
  const mean = (arr: number[]): number => {
    let s = 0;
    for (const x of arr) s += x;
    return s / arr.length;
  };
  const variance = (arr: number[], mu: number): number => {
    let s = 0;
    for (const x of arr) s += (x - mu) ** 2;
    return s / Math.max(1, arr.length - 1);
  };
  const mN = mean(wNum), mD = mean(wDen);
  const vN = variance(wNum, mN), vD = variance(wDen, mD);
  const mcSe = Math.sqrt(vN / (N2 * mN * mN) + vD / (N1 * mD * mD));

  return { logMarginal: logM, iterations, mcSe };
}

/**
 * Path-sampling / thermodynamic-integration estimator of log m(y).
 *
 * Given draws θ_s^{(β)} from each power posterior p_β(θ) ∝ L(θ)^β π(θ) at
 * an ascending β-grid {β_0 = 0, …, β_K = 1}, the Gelman–Meng identity gives
 *
 *   log m(y) = ∫_0^1 E_{p_β}[ log L(θ) ] dβ,
 *
 * estimated by trapezoidal integration over the grid. Brief §6.1 keeps this
 * in the estimator stable-of-four alongside bridge, IS, and harmonic-mean.
 */
export function pathSamplingEstimate(opts: {
  powerPosteriorDraws: number[][][];
  betas: number[];
  logLikelihood: (theta: number[]) => number;
}): { logMarginal: number; mcSe: number } {
  const { powerPosteriorDraws, betas, logLikelihood } = opts;
  const K = betas.length;
  if (K < 2 || powerPosteriorDraws.length !== K) {
    return { logMarginal: NaN, mcSe: NaN };
  }
  const expLogL: number[] = new Array(K);
  const varLogL: number[] = new Array(K);
  for (let k = 0; k < K; k++) {
    const draws = powerPosteriorDraws[k];
    const n = draws.length;
    // Welford single-pass: compute mean and sum-of-squared-deviations online,
    // avoiding the intermediate logLs[] allocation.
    let mean = 0;
    let m2 = 0;
    for (let s = 0; s < n; s++) {
      const x = logLikelihood(draws[s]);
      const delta = x - mean;
      mean += delta / (s + 1);
      m2 += delta * (x - mean);
    }
    expLogL[k] = mean;
    varLogL[k] = n > 1 ? m2 / ((n - 1) * n) : 0;
  }
  let logMarginal = 0;
  let seSquared = 0;
  for (let k = 0; k < K - 1; k++) {
    const dB = betas[k + 1] - betas[k];
    logMarginal += 0.5 * dB * (expLogL[k] + expLogL[k + 1]);
    seSquared += 0.25 * dB * dB * (varLogL[k] + varLogL[k + 1]);
  }
  return { logMarginal, mcSe: Math.sqrt(seSquared) };
}

/**
 * Nested-sampling evidence estimator (Skilling 2006 §6).
 *
 * Takes the ascending-sorted dead-point log-likelihoods from an NS run,
 * the number of live points N, and the final live-point log-likelihoods,
 * and returns the log evidence via prior-mass shrinkage:
 *
 *   X_i = exp(−i / N),   w_i = X_{i−1} − X_i,
 *   log Z = logsumexp_i( log L_i + log w_i ) ⊕ live-remainder contribution.
 *
 * The live-remainder averages the final N live-point likelihoods over the
 * remaining prior volume X_K / N each. Uncertainty is estimated by the
 * Skilling information H = ∫ p log(p/w), with SE(log Z) ≈ √(H/N).
 */
export function nestedSamplingEstimate(opts: {
  logLikDeadPoints: number[];
  nLivePoints: number;
  finalLiveLogLiks: number[];
}): { logMarginal: number; mcSe: number } {
  const { logLikDeadPoints, nLivePoints, finalLiveLogLiks } = opts;
  const K = logLikDeadPoints.length;
  if (K === 0 || nLivePoints <= 0) {
    return { logMarginal: -Infinity, mcSe: NaN };
  }
  const logShrink = Math.log1p(-Math.exp(-1 / nLivePoints));
  const logTerms: number[] = [];
  const logLs: number[] = [];              // per-term logL, for Skilling H
  for (let i = 0; i < K; i++) {
    // Iteration index 1-based: i+1. log w_{i+1} = −i/N + log(1 − e^{−1/N}).
    const logW = -i / nLivePoints + logShrink;
    logLs.push(logLikDeadPoints[i]);
    logTerms.push(logLikDeadPoints[i] + logW);
  }
  const logXK = -K / nLivePoints;
  const logLiveW = logXK - Math.log(nLivePoints);
  for (const logL of finalLiveLogLiks) {
    logLs.push(logL);
    logTerms.push(logL + logLiveW);
  }

  const logZ = logsumexp(logTerms);
  // Skilling information H = ∫ (L/Z) log(L/Z) dπ ≈ Σ_i p_i (log L_i − log Z),
  // where p_i = L_i w_i / Z (evidence contribution) and log L_i is the
  // per-term log-likelihood (excluding the prior-mass weight). H ≥ 0 is the
  // KL divergence from prior to posterior; SE(log Z) ≈ √(H / N_live).
  let H = 0;
  for (let i = 0; i < logTerms.length; i++) {
    const logp = logTerms[i] - logZ;
    if (Number.isFinite(logp)) H += Math.exp(logp) * (logLs[i] - logZ);
  }
  const mcSe = Math.sqrt(Math.max(0, H / nLivePoints));
  return { logMarginal: logZ, mcSe };
}

/**
 * Spiegelhalter et al. (2002) Deviance Information Criterion.
 *
 *   D(θ) = −2 log L(y | θ),
 *   p_DIC = Ē_s[D(θ_s)] − D(θ̂),   θ̂ = posterior mean,
 *   DIC = D(θ̂) + 2 p_DIC = Ē_s[D] + p_DIC.
 *
 * Inputs: a pointwise log-likelihood matrix `logLikDraws[s][i]` and the
 * per-observation log-likelihoods evaluated at the posterior-mean parameter.
 * Returns {dic, pDic, eLogLik} where eLogLik = Ē_s[Σ_i log L_{s,i}] is the
 * posterior-expected log-likelihood (displayed by many Stan/PyMC outputs).
 */
export function dic(opts: {
  logLikDraws: number[][];
  logLikAtPosteriorMean: number[];
}): { dic: number; pDic: number; eLogLik: number } {
  const { logLikDraws, logLikAtPosteriorMean } = opts;
  const S = logLikDraws.length;
  if (S === 0) return { dic: NaN, pDic: NaN, eLogLik: NaN };
  let sumLogLik = 0;
  for (let s = 0; s < S; s++) {
    let row = 0;
    for (let i = 0; i < logLikDraws[s].length; i++) row += logLikDraws[s][i];
    sumLogLik += row;
  }
  const eLogLik = sumLogLik / S;
  const logLikMean = logLikAtPosteriorMean.reduce((a, b) => a + b, 0);
  const dBar = -2 * eLogLik;
  const dHat = -2 * logLikMean;
  const pDic = dBar - dHat;
  return { dic: dBar + pDic, pDic, eLogLik };
}

/**
 * Watanabe (2010) Widely Applicable Information Criterion (WAIC).
 *
 *   lppd_i  = log ( (1/S) Σ_s exp(log L_{s,i}) ),
 *   pwaic_i = Var_s[ log L_{s,i} ]   (Bessel-corrected, ddof = 1),
 *   elpd_WAIC = Σ_i (lppd_i − pwaic_i),
 *   WAIC      = −2 · elpd_WAIC.
 *
 * Brief §27.8 cites this alongside DIC and PSIS-LOO; Vehtari, Gelman &
 * Gabry (2017) show WAIC ≈ LOO asymptotically (cited-not-proven — brief
 * Appendix B.2). The per-observation lppd_i uses logsumexp for stability.
 */
export function waic(
  logLikDraws: number[][],
): { waic: number; elpdWaic: number; pWaic: number; seWaic: number } {
  const S = logLikDraws.length;
  if (S === 0) {
    return { waic: NaN, elpdWaic: NaN, pWaic: NaN, seWaic: NaN };
  }
  const n = logLikDraws[0].length;
  const pointwise: number[] = new Array(n);
  // Reuse a single column buffer across the observation loop to avoid
  // allocating [n] arrays of size S.
  const col: number[] = new Array(S);
  let elpd = 0;
  let pwaicSum = 0;
  for (let i = 0; i < n; i++) {
    for (let s = 0; s < S; s++) col[s] = logLikDraws[s][i];
    const lppdI = logsumexp(col) - Math.log(S);
    // Welford single-pass mean + variance.
    let mean = 0;
    let m2 = 0;
    for (let s = 0; s < S; s++) {
      const x = col[s];
      const delta = x - mean;
      mean += delta / (s + 1);
      m2 += delta * (x - mean);
    }
    const pwaicI = S > 1 ? m2 / (S - 1) : 0;
    pointwise[i] = lppdI - pwaicI;
    elpd += pointwise[i];
    pwaicSum += pwaicI;
  }
  // SE over observations (Vehtari 2017 §2.2): √(n · Var_i(pointwise_i)).
  let mean = 0;
  for (const p of pointwise) mean += p;
  mean /= n;
  let varSum = 0;
  for (const p of pointwise) varSum += (p - mean) ** 2;
  const seElpd = Math.sqrt(n * (n > 1 ? varSum / (n - 1) : 0));
  return {
    waic: -2 * elpd,
    elpdWaic: elpd,
    pWaic: pwaicSum,
    seWaic: 2 * seElpd,
  };
}

/**
 * PSIS-LOO (Vehtari, Gelman & Gabry 2017). For the Topic 27 scope we use
 * the naive-IS estimator (no Pareto-smoothing) as the pointwise LOO
 * log-likelihood estimator,
 *
 *   log p(y_i | y_{−i}) ≈ −log ( (1/S) Σ_s exp(−log L_{s,i}) )
 *                      = log S − logsumexp_s(−log L_{s,i}).
 *
 * A Pareto-k diagnostic per observation is computed: if > 0.7, that
 * observation's LOO estimate is flagged as unreliable. The test T27.12 at
 * the canonical-tiny conjugate-Normal problem has small k's throughout
 * (posterior-to-LOO variance ratio is bounded for conjugate models).
 */
export function psisLoo(
  logLikDraws: number[][],
): {
  loo: number;
  elpdLoo: number;
  pLoo: number;
  seLoo: number;
  paretoK: number[];
  nProblematic: number;
} {
  const S = logLikDraws.length;
  if (S === 0) {
    return { loo: NaN, elpdLoo: NaN, pLoo: NaN, seLoo: NaN, paretoK: [], nProblematic: 0 };
  }
  const n = logLikDraws[0].length;
  const pointwise: number[] = new Array(n);
  const paretoK: number[] = new Array(n);
  // Reuse neg / pos buffers across observations.
  const neg: number[] = new Array(S);
  const pos: number[] = new Array(S);
  let lppdSum = 0;
  for (let i = 0; i < n; i++) {
    for (let s = 0; s < S; s++) {
      neg[s] = -logLikDraws[s][i];
      pos[s] = logLikDraws[s][i];
    }
    // Naive IS LOO: log p(y_i | y_{−i}) ≈ log S − logsumexp_s(−log L_{s,i}).
    pointwise[i] = Math.log(S) - logsumexp(neg);
    // lppd_i for p_LOO = Σ lppd_i − elpd_LOO.
    lppdSum += logsumexp(pos) - Math.log(S);
    // Pareto-k moment estimator on the top 20% of exp(neg) weights.
    // For a GPD(k, σ), R := E[X]² / E[X²] = (1 − 2k) / (2(1 − k)), giving
    // k = (1 − 2R) / (2(1 − R)). At k = 0 (Exponential), R = 1/2 ⇒ k = 0.
    const topCount = Math.max(2, Math.floor(0.2 * S));
    const sorted = neg.slice().sort((a, b) => b - a); // descending
    const tail = sorted.slice(0, topCount).map(v => Math.exp(v - sorted[0]));
    let mean = 0;
    for (const t of tail) mean += t;
    mean /= tail.length;
    let v = 0;
    for (const t of tail) v += (t - mean) ** 2;
    v /= Math.max(1, tail.length - 1);
    const m2 = v + mean * mean;                              // E[X²] (Bessel-corrected)
    const R = m2 > 0 ? (mean * mean) / m2 : 0.5;             // ratio; degenerate tail → 0.5 ⇒ k = 0
    const kHat = R < 1 ? (1 - 2 * R) / (2 * (1 - R)) : 0;
    paretoK[i] = kHat;
  }
  let elpd = 0;
  for (const p of pointwise) elpd += p;
  const pLoo = lppdSum - elpd;
  let mean = 0;
  for (const p of pointwise) mean += p;
  mean /= n;
  let varSum = 0;
  for (const p of pointwise) varSum += (p - mean) ** 2;
  const seElpd = Math.sqrt(n * (n > 1 ? varSum / (n - 1) : 0));
  let nProblematic = 0;
  for (const k of paretoK) if (k > 0.7) nProblematic++;
  return {
    loo: -2 * elpd,
    elpdLoo: elpd,
    pLoo,
    seLoo: 2 * seElpd,
    paretoK,
    nProblematic,
  };
}

/**
 * BMA predictive mixture: aggregate per-model posterior-predictive draws
 * weighted by posterior model probabilities.
 *
 * Output size defaults to the minimum per-model draw count. Stratified
 * allocation: model k contributes ⌊weights[k] · nOut⌉ draws (rounded),
 * with residual slots filled from the final model to preserve length.
 * Inputs need not be pre-normalized; weights are normalized internally.
 *
 * Brief §3.2 §27.7 Thm 8 gives the log-loss form of the BMA predictive;
 * this is the sample-level realization used by `BMAPredictiveComparison`.
 */
export function bmaPredictive(opts: {
  perModelDraws: number[][];
  weights: number[];
  nOut?: number;
}): number[] {
  const { perModelDraws, weights } = opts;
  const K = perModelDraws.length;
  if (K === 0) return [];
  // Drop models with empty draws up front; if their weight was non-zero we
  // renormalize over the remaining models (rather than silently sampling
  // draws[NaN] below). If every model is empty we can't produce draws.
  const eligible: { w: number; draws: number[] }[] = [];
  let minS = Number.POSITIVE_INFINITY;
  for (let k = 0; k < K; k++) {
    const draws = perModelDraws[k];
    if (draws.length > 0) {
      eligible.push({ w: weights[k] ?? 0, draws });
      if (draws.length < minS) minS = draws.length;
    }
  }
  const total = eligible.reduce((a, e) => a + e.w, 0);
  if (eligible.length === 0 || !(total > 0)) return [];
  const normW = eligible.map(e => e.w / total);
  const nOut = opts.nOut ?? (Number.isFinite(minS) ? minS : 0);
  const out: number[] = new Array(nOut);
  let idx = 0;
  for (let k = 0; k < eligible.length; k++) {
    const take = Math.round(normW[k] * nOut);
    const draws = eligible[k].draws;
    for (let j = 0; j < take && idx < nOut; j++) {
      out[idx++] = draws[j % draws.length];
    }
  }
  // Rounding residuals: fill from the last eligible model (guaranteed non-empty).
  const fallback = eligible[eligible.length - 1].draws;
  while (idx < nOut) {
    out[idx] = fallback[idx % fallback.length];
    idx++;
  }
  return out;
}

/** Gaussian kernel density estimate — internal helper for localFdr's
 *  Efron-2010 mixture-density estimation path. */
function gaussianKDE(z: number, samples: number[], bandwidth: number): number {
  if (samples.length === 0 || bandwidth <= 0) return 0;
  let sum = 0;
  for (const s of samples) sum += Math.exp(-0.5 * ((z - s) / bandwidth) ** 2);
  return sum / (samples.length * bandwidth * Math.sqrt(2 * Math.PI));
}

/** Silverman's rule-of-thumb bandwidth, used by default in gaussianKDE.
 *  Single-pass Welford mean + variance (no intermediate allocation). */
function silvermanBandwidth(samples: number[]): number {
  const n = samples.length;
  if (n < 2) return 1;
  let mean = 0;
  let m2 = 0;
  for (let i = 0; i < n; i++) {
    const x = samples[i];
    const delta = x - mean;
    mean += delta / (i + 1);
    m2 += delta * (x - mean);
  }
  const sd = Math.sqrt(m2 / (n - 1));
  return 1.06 * sd * Math.pow(n, -0.2);
}

/**
 * Efron (2010) two-groups local-FDR estimator.
 *
 *   fdr(z) = π₀ · f₀(z) / f̂(z),   f̂ = (1 − π₁) f₀ + π₁ f₁.
 *
 * Two paths per the `opts.alternative` presence:
 *  • `alternative` provided (closed-form): compute fdr analytically from the
 *    given two-groups mixture parameters. This is the path used by T27.13 /
 *    T27.14's 1e-4-tolerance sanity checks — the density-estimation path
 *    would require an impractically large z-sample to hit that precision.
 *  • `alternative` absent (Efron-2010 estimated): estimate f̂ via Gaussian
 *    KDE with Silverman bandwidth, estimate π₀ by the "lowest-point" rule
 *    π₀ = min(1, f̂(μ₀) / f₀(μ₀)), and return the ratio. Suitable for the
 *    `LocalFDRExplorer` component where a realistic z-sample is supplied.
 *
 * When `theoreticalNull === true` (default), f₀ = N(0, 1); otherwise the
 * null (μ₀, σ₀) is fit by central-matching on the middle 50% of z-scores
 * (SD-of-middle-50% conversion factor 1.483 for Gaussian null).
 */
export function localFdr(opts: {
  zScores: number[];
  zGrid: number[];
  theoreticalNull?: boolean;
  alternative?: { piOne: number; muAlt: number; sigmaAlt?: number };
}): { fdrGrid: number[]; piZero: number; nullParams: [number, number] } {
  const { zScores, zGrid, alternative } = opts;
  const theoreticalNull = opts.theoreticalNull ?? true;

  let mu0 = 0;
  let sigma0 = 1;
  if (!theoreticalNull && zScores.length >= 4) {
    const sorted = zScores.slice().sort((a, b) => a - b);
    const n = sorted.length;
    const q1 = Math.floor(n * 0.25);
    const q3 = Math.min(n, Math.floor(n * 0.75));
    const middle = sorted.slice(q1, q3);
    if (middle.length >= 2) {
      let mean = 0;
      for (const z of middle) mean += z;
      mean /= middle.length;
      let vSum = 0;
      for (const z of middle) vSum += (z - mean) ** 2;
      mu0 = mean;
      // Conversion: SD of the middle-50% ≈ σ · 0.674, so σ ≈ sd_middle / 0.674 ≈ sd_middle · 1.483.
      sigma0 = Math.sqrt(vSum / (middle.length - 1)) * 1.483;
    }
  }
  const sigma02 = sigma0 * sigma0;

  if (alternative) {
    const piOne = alternative.piOne;
    const piZero = 1 - piOne;
    const muAlt = alternative.muAlt;
    const sigmaAlt2 = alternative.sigmaAlt != null
      ? alternative.sigmaAlt * alternative.sigmaAlt
      : 1;
    const fdrGrid = zGrid.map(z => {
      const f0 = pdfNormal(z, mu0, sigma02);
      const f1 = pdfNormal(z, muAlt, sigmaAlt2);
      const mix = piZero * f0 + piOne * f1;
      return mix > 0 ? (piZero * f0) / mix : 1;
    });
    return { fdrGrid, piZero, nullParams: [mu0, sigma0] };
  }

  // Efron-2010 estimated path (density estimation from observed z's).
  if (zScores.length === 0) {
    return {
      fdrGrid: zGrid.map(() => NaN),
      piZero: 1,
      nullParams: [mu0, sigma0],
    };
  }
  const bw = silvermanBandwidth(zScores);
  const fMix = (z: number) => gaussianKDE(z, zScores, bw);
  const piZeroEst = Math.min(1, fMix(mu0) / pdfNormal(mu0, mu0, sigma02));
  const fdrGrid = zGrid.map(z => {
    const num = piZeroEst * pdfNormal(z, mu0, sigma02);
    const den = fMix(z);
    return den > 0 ? Math.min(1, num / den) : 1;
  });
  return { fdrGrid, piZero: piZeroEst, nullParams: [mu0, sigma0] };
}

/**
 * Gelman (1996) posterior-predictive check: Bayesian p-value for a test
 * statistic T on observed vs replicated data.
 *
 *   p_B = (1/S) Σ_s 1{ T(y_rep_s, θ_s) ≥ T(y_obs, θ_s) }.
 *
 * For θ-independent discrepancies (standard case), `thetaDraws` may be
 * omitted; T receives only `y`. For θ-dependent discrepancies (e.g. Gelman's
 * discrepancy statistics in BDA §6.3), pass per-draw `thetaDraws[s]` so
 * T(y_rep_s, θ_s) and T(y_obs, θ_s) share the same θ_s.
 *
 * Brief §27.9 uses this for the capstone PPC example (Ex 12).
 */
export function posteriorPredictiveCheck(opts: {
  yRepDraws: number[][];
  yObs: number[];
  T: (y: number[], theta?: number[]) => number;
  thetaDraws?: number[][];
}): { bayesianPValue: number; TObs: number; TRep: number[] } {
  const { yRepDraws, yObs, T, thetaDraws } = opts;
  const S = yRepDraws.length;
  if (S === 0) return { bayesianPValue: NaN, TObs: T(yObs), TRep: [] };
  const TRep: number[] = new Array(S);
  let count = 0;
  let tObsSum = 0;
  for (let s = 0; s < S; s++) {
    const theta = thetaDraws?.[s];
    const tRep = T(yRepDraws[s], theta);
    const tObs = T(yObs, theta);
    TRep[s] = tRep;
    tObsSum += tObs;
    if (tRep >= tObs) count++;
  }
  // For θ-independent T, each tObs is identical so sum/S = T(yObs) exactly.
  // For θ-dependent T (e.g. Gelman discrepancies), we return the posterior
  // mean of T(yObs, θ_s) — consistent with the values actually compared in
  // the Bayesian p-value sum.
  return {
    bayesianPValue: count / S,
    TObs: tObsSum / S,
    TRep,
  };
}

// ==========================================================================
// TOPIC 28 — Hierarchical & Empirical Bayes (Track 7 closer)
// ==========================================================================
//
// Additions: James-Stein & partial-pooling shrinkage estimators, Type-II
// marginal-likelihood / empirical-Bayes machinery, 8-schools canonical
// dataset, centered ↔ non-centered reparameterization, Neal-funnel log-
// density. See handoff brief §6.1 for full signatures + §6.2 for T28
// test pins.

// Private helpers (not exported).
function mean(x: number[]): number {
  if (x.length === 0) return NaN;
  let s = 0;
  for (const v of x) s += v;
  return s / x.length;
}

function variance(x: number[]): number {
  if (x.length < 2) return 0;
  const m = mean(x);
  let s = 0;
  for (const v of x) s += (v - m) ** 2;
  return s / (x.length - 1);
}

/**
 * E[1 / ‖X‖²] with X ~ 𝒩_d(θ, σ²I), equivalent to σ⁻² · E[1 / χ²_d(λ)]
 * where λ = ‖θ‖²/σ² is the non-centrality parameter. Returns the
 * un-scaled expectation E[1 / χ²_d(λ)]; caller multiplies by σ⁻²
 * if σ² ≠ 1 (see steinRiskDifference).
 *
 * Closed-form via the Poisson-mixture representation of non-central χ²:
 *   χ²_d(λ) | J ~ χ²_{d+2J} (central),  J ~ Poisson(λ/2)
 *   ⇒ E[1/χ²_d(λ)] = e^(-λ/2) · Σ_{j≥0} (λ/2)^j / j! · 1/(d + 2j − 2)
 *
 * For λ ≥ 50 the series converges slowly → fall back to 10⁵-draw Monte
 * Carlo with a deterministic seed (reproducibility in the T28 test pins).
 * (Johnson-Kotz-Balakrishnan 1995 §29.4; LEH1998 §5.5 eq. 5.21)
 */
function expectInvNonCentralChiSq(d: number, lambda: number): number {
  if (d <= 2) return Infinity; // E[1/χ²_d] diverges for d ≤ 2
  if (lambda === 0) return 1 / (d - 2);

  if (lambda < 50) {
    // Poisson-mixture Taylor series, truncated at j=60 (more than enough
    // for λ < 50: Poisson(25)'s mass is concentrated in j ≤ 50).
    const halfLambda = lambda / 2;
    let sum = 0;
    let poissonWeight = 1; // (λ/2)^j / j! — updated multiplicatively
    for (let j = 0; j <= 60; j++) {
      sum += poissonWeight / (d + 2 * j - 2);
      poissonWeight *= halfLambda / (j + 1);
    }
    return Math.exp(-halfLambda) * sum;
  }

  // MC fallback for λ ≥ 50. Generate χ²_d(λ) via offset Normals:
  // ‖X‖² = (Z₀ + √λ)² + Z₁² + … + Z_{d-1}²,  Zᵢ iid 𝒩(0,1).
  const seed = d * 1000 + Math.round(lambda * 10);
  const rng = createSeededRng(seed);
  const shift = Math.sqrt(lambda);
  const N = 100_000;
  let invSum = 0;
  for (let n = 0; n < N; n++) {
    const z0 = rng.normal() + shift;
    let normSq = z0 * z0;
    for (let i = 1; i < d; i++) {
      const z = rng.normal();
      normSq += z * z;
    }
    invSum += 1 / normSq;
  }
  return invSum / N;
}

/**
 * James-Stein estimator for a multivariate Normal mean (STE1956, JAM1961).
 * Returns X unchanged if d < 3 (Stein's theorem doesn't apply; the MLE is
 * admissible in 1–2 dimensions).
 *
 * Assumes X ~ 𝒩_d(θ, σ² I_d) with σ² known (default 1).
 *
 * @see §28.5 Thm 2.
 */
export function jamesSteinEstimator(
  X: number[],
  sigmaSq: number = 1
): number[] {
  const d = X.length;
  if (d < 3) return [...X];
  const normSq = X.reduce((s, x) => s + x * x, 0);
  if (normSq === 0) return [...X];
  const shrink = 1 - ((d - 2) * sigmaSq) / normSq;
  return X.map((x) => shrink * x);
}

/**
 * Positive-part James-Stein estimator (Efron-Morris 1973 EFR1973).
 * Clamps the shrinkage factor at zero so the estimator never flips sign —
 * dominates plain JS in finite samples.
 *
 * @see §28.5 Rem 11.
 */
export function jamesSteinPositivePart(
  X: number[],
  sigmaSq: number = 1
): number[] {
  const d = X.length;
  if (d < 3) return [...X];
  const normSq = X.reduce((s, x) => s + x * x, 0);
  if (normSq === 0) return [...X];
  const shrink = Math.max(0, 1 - ((d - 2) * sigmaSq) / normSq);
  return X.map((x) => shrink * x);
}

/**
 * Closed-form Stein risk difference R(JS, θ) − R(MLE, θ). Negative for
 * every θ when d ≥ 3 (Stein's paradox). Zero when d ≤ 2.
 *
 * For X ~ 𝒩_d(θ, σ² I_d) and JS estimator with shrinkage coefficient
 * a = (d−2)σ², the risk decomposition gives
 *
 *   R(JS, θ) − R(MLE, θ) = −(d−2)² · σ⁴ · E[1/‖X‖²]
 *                        = −(d−2)² · σ² · E[1/χ²_d(λ)],
 *
 * where λ = ‖θ‖²/σ² is the non-centrality parameter. The σ⁴ from a²
 * minus one σ² from E[1/‖X‖²] = (1/σ²)·E[1/χ²_d(λ)] leaves the σ²
 * prefactor on the final expression. Sanity check at θ=0, σ²=4, d=3:
 * E[1/χ²_3(0)] = 1/(d−2) = 1, so risk diff = −1 · 4 · 1 = −4 (scales
 * linearly with σ²). (STE1956 eq. 3.6; LEH1998 §5.5 eq. 5.21)
 *
 * @see §28.5 Proof 1.
 */
export function steinRiskDifference(
  theta: number[],
  sigmaSq: number = 1
): number {
  const d = theta.length;
  if (d < 3) return 0;
  const lambda = theta.reduce((s, t) => s + t * t, 0) / sigmaSq;
  const expInvChiSq = expectInvNonCentralChiSq(d, lambda);
  return -((d - 2) ** 2) * sigmaSq * expInvChiSq;
}

/**
 * Shrinkage factor B_k in the Normal-Normal hierarchical model (§28.6 Thm 4):
 *
 *   B_k = σ² / (n_k · τ² + σ²)
 *
 * For the sample-mean case (n_k = 1), reduces to σ²/(σ² + τ²). Bounded to
 * [0, 1]; edge cases: τ² = 0 → 1 (complete pool), τ² = ∞ → 0 (no pool).
 */
export function partialPoolingShrinkageFactor(
  sigmaSq: number,
  tauSq: number,
  nk: number = 1
): number {
  if (tauSq === 0) return 1; // complete pool
  if (!Number.isFinite(tauSq)) return 0; // no pool (flat group-level prior)
  return sigmaSq / (nk * tauSq + sigmaSq);
}

/**
 * Partial-pooling posterior mean in Normal-Normal hierarchical model,
 * conditional on hyperparameters (μ, τ²). Returns (1 − B_k) y + B_k μ.
 *
 * @see §28.6 Proof 2.
 */
export function partialPoolingPosteriorMean(
  y: number,
  mu: number,
  sigmaSq: number,
  tauSq: number,
  nk: number = 1
): number {
  const B = partialPoolingShrinkageFactor(sigmaSq, tauSq, nk);
  return (1 - B) * y + B * mu;
}

/**
 * Partial-pooling posterior variance in Normal-Normal hierarchical model.
 *   Var(θ_k | y_k, μ, τ²) = σ² · τ² / (n_k · τ² + σ²)
 *
 * Edge cases: τ² = 0 → 0 (complete pool: group mean = μ deterministically);
 * τ² = ∞ → σ²/n_k (no-pool likelihood variance).
 *
 * @see §28.6 Thm 4.
 */
export function partialPoolingPosteriorVariance(
  sigmaSq: number,
  tauSq: number,
  nk: number = 1
): number {
  if (tauSq === 0) return 0;
  if (!Number.isFinite(tauSq)) return sigmaSq / nk;
  return (sigmaSq * tauSq) / (nk * tauSq + sigmaSq);
}

/**
 * Precision-weighted grand mean of per-group observations:
 *   μ̂ = Σ (y_k / σ²_k) / Σ (1 / σ²_k)
 *
 * The MLE of μ in a Normal-Normal hierarchical model when τ² is fixed at
 * zero (complete-pool limit). Used across §§28.1/28.4/28.6.
 */
export function normalNormalGrandMean(
  y: number[],
  sigmaSq: number[]
): number {
  if (y.length !== sigmaSq.length)
    throw new Error('normalNormalGrandMean: y and sigmaSq must have equal length');
  let numerator = 0;
  let denominator = 0;
  for (let k = 0; k < y.length; k++) {
    const w = 1 / sigmaSq[k];
    numerator += y[k] * w;
    denominator += w;
  }
  return numerator / denominator;
}

/**
 * Type-II marginal log-likelihood log m(y | μ, τ²) in the Normal-Normal
 * hierarchical model, integrating out θ_k:
 *
 *   y_k | μ, τ² ~ 𝒩(μ, σ²_k + τ²) independent across k
 *   ⇒  log m(y | μ, τ²) = Σ_k log 𝒩(y_k ; μ, σ²_k + τ²)
 *
 * The empirical-Bayes objective (§28.7 Def 4).
 */
export function typeIIMarginalLogLikelihood(
  y: number[],
  sigmaSq: number[],
  mu: number,
  tauSq: number
): number {
  if (y.length !== sigmaSq.length)
    throw new Error('typeIIMarginalLogLikelihood: y and sigmaSq length mismatch');
  let logLik = 0;
  for (let k = 0; k < y.length; k++) {
    const v = sigmaSq[k] + tauSq;
    logLik += -0.5 * Math.log(2 * Math.PI * v) - ((y[k] - mu) ** 2) / (2 * v);
  }
  return logLik;
}

/**
 * Type-II MLE (empirical-Bayes estimate) of (μ, τ²) in Normal-Normal.
 * Iterative scheme (Berger 1985 §4.5.1):
 *   — given τ², update μ = precision-weighted mean of y with weights 1/(σ²_k + τ²)
 *   — given μ, update τ² = max(0, method-of-moments estimate)
 *
 * Initializer: τ² = max(0, Var(y) − mean(σ²)), μ = weighted mean.
 * Converges geometrically; typical runs finish in <30 iterations.
 *
 * Note: τ̂² is floored at 0 because the MLE on the boundary is a
 * genuine feature (pedagogically central to §28.7 — cf. GEL2006's
 * half-Cauchy-prior argument).
 */
export function typeIIMLE(
  y: number[],
  sigmaSq: number[],
  options: { maxIter?: number; tol?: number } = {}
): { mu: number; tauSq: number; iterations: number; converged: boolean } {
  const K = y.length;
  if (K !== sigmaSq.length)
    throw new Error('typeIIMLE: y and sigmaSq length mismatch');
  const maxIter = options.maxIter ?? 200;
  const tol = options.tol ?? 1e-8;

  // Initializer: method-of-moments τ², grand-mean seed under that τ².
  let tauSq = Math.max(0, variance(y) - mean(sigmaSq));
  let mu: number;
  {
    let num = 0;
    let den = 0;
    for (let k = 0; k < K; k++) {
      const w = 1 / (sigmaSq[k] + tauSq);
      num += y[k] * w;
      den += w;
    }
    mu = num / den;
  }

  // Hoist the per-iteration scratch arrays — ShrinkageExplorer's Run-500
  // calls this ~500× per click and the .map/.reduce allocations were
  // dominating the GC footprint (gemini PR-32).
  const weights = new Array<number>(K);

  for (let i = 0; i < maxIter; i++) {
    let W = 0;
    let muNum = 0;
    for (let k = 0; k < K; k++) {
      const w = 1 / (sigmaSq[k] + tauSq);
      weights[k] = w;
      W += w;
      muNum += y[k] * w;
    }
    const muNew = muNum / W;

    let tauSqNumer = 0;
    for (let k = 0; k < K; k++) {
      const resSq = (y[k] - muNew) ** 2;
      tauSqNumer += weights[k] * (resSq - sigmaSq[k]);
    }
    const tauSqNew = Math.max(0, tauSqNumer / W);

    if (Math.abs(muNew - mu) < tol && Math.abs(tauSqNew - tauSq) < tol) {
      return { mu: muNew, tauSq: tauSqNew, iterations: i + 1, converged: true };
    }
    mu = muNew;
    tauSq = tauSqNew;
  }
  return { mu, tauSq, iterations: maxIter, converged: false };
}

/**
 * Canonical 8-schools dataset (Rubin 1981 RUB1981, GEL2013 §5.5).
 * Eight educational coaching effects with per-school SE. The spine running
 * example across §§28.1/28.4/28.6/28.9 and numerous Topic 28 components.
 */
export function eightSchoolsData(): {
  names: string[];
  y: number[];
  sigma: number[];
} {
  return {
    names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    y: [28, 8, -3, 7, -1, 1, 18, 12],
    sigma: [15, 10, 16, 11, 9, 11, 10, 18],
  };
}

/**
 * Non-centered transform: θ̃ ↦ θ = μ + τ · θ̃ (§28.9 Thm 7).
 * Decouples θ from (μ, τ) in the prior — the standard mitigation for
 * Neal's funnel under HMC/NUTS.
 */
export function nonCenteredTransform(
  thetaTilde: number[],
  mu: number,
  tau: number
): number[] {
  return thetaTilde.map((t) => mu + tau * t);
}

/**
 * Inverse transform: θ ↦ θ̃ = (θ − μ) / τ. Requires τ ≠ 0.
 * Throws when tau === 0 (the centered→non-centered change of variables is
 * undefined at the funnel apex; see §28.9 Thm 7).
 */
export function centeredToNonCentered(
  theta: number[],
  mu: number,
  tau: number
): number[] {
  if (tau === 0)
    throw new Error('centeredToNonCentered: tau must be nonzero (reparameterization undefined)');
  return theta.map((t) => (t - mu) / tau);
}

/**
 * Neal's canonical funnel joint log-density at (θ, log τ):
 *   log τ ~ 𝒩(0, 1)
 *   θ | τ ~ 𝒩(0, τ²)
 *   ⇒ log p(θ, log τ) = log 𝒩(θ; 0, e^{2 log τ}) + log 𝒩(log τ; 0, 1)
 *
 * Used as the target for FunnelGeometryExplorer (§28.9) and as the
 * simplified stand-in for the 8-schools hierarchical posterior in
 * component demos.
 *
 * (Neal 2003 §8; NEA2011 §4; BET2015 §3.)
 */
export function funnelLogDensity(theta: number, logTau: number): number {
  const tauSq = Math.exp(2 * logTau);
  const logPTheta =
    -0.5 * Math.log(2 * Math.PI * tauSq) - (theta * theta) / (2 * tauSq);
  const logPTau = -0.5 * Math.log(2 * Math.PI) - (logTau * logTau) / 2;
  return logPTheta + logPTau;
}

// ==========================================================================
// END TOPIC 28 ADDITIONS
// ==========================================================================
