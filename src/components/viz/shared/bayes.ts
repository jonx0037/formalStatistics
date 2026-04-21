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

/** Memoization cache for (α, β, level) → [lo, hi]. Cleared by the GC; no eviction. */
const _credibleBetaCache = new Map<string, [number, number]>();
const _hpdBetaCache = new Map<string, [number, number]>();

/**
 * Equal-tailed (1−α) credible interval for Beta(α, β). Memoized on
 * the (alpha, beta, level) triple because the BvM animator requests
 * intervals on every slider frame.
 */
export function credibleIntervalBeta(
  alpha: number,
  beta: number,
  level: number,
): [number, number] {
  const key = `${alpha},${beta},${level}`;
  const cached = _credibleBetaCache.get(key);
  if (cached) return cached;
  const tail = (1 - level) / 2;
  const interval: [number, number] = [
    quantileBeta(tail, alpha, beta),
    quantileBeta(1 - tail, alpha, beta),
  ];
  _credibleBetaCache.set(key, interval);
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
  // Initial upper bound: mean + 10·sd.
  const mean = alpha / beta;
  const sd = Math.sqrt(alpha) / beta;
  let lo = 0;
  let hi = Math.max(mean + 10 * sd, 1);
  while (cdfGamma(hi, alpha, beta) < p) hi *= 2;
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
  const key = `${alpha},${beta},${level}`;
  const cached = _hpdBetaCache.get(key);
  if (cached) return cached;
  const eps = 1e-4;
  const interval = hpdInterval(
    x => pdfBeta(x, alpha, beta),
    [eps, 1 - eps],
    level,
    500,
  );
  _hpdBetaCache.set(key, interval);
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
 *   bernoulli    → π(θ) ∝ θ^{−1/2}(1−θ)^{−1/2}   (Beta(½, ½) up to a const)
 *   poisson      → π(λ) ∝ λ^{−1/2}
 *   normal-mean  → π(μ) ∝ 1 / σ                   (flat; requires knownVariance)
 *   normal-scale → π(σ) ∝ 1 / σ                   (log-flat; the classical
 *                                                   "reference" scale prior)
 *   exponential  → π(λ) ∝ 1 / λ
 *
 * All five are improper; the `∝` constant is dropped because downstream
 * Bayes-rule calculations only need the proportional kernel. See brief
 * §25.7 Thm 4 derivation for Bernoulli and Normal-scale; the other three
 * are named only (JEF1961 Ch. III).
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
 * Dirichlet-Multinomial predictive is a Dirichlet-Multinomial compound, itself
 * multivariate; Normal-Normal-IG predictive is a non-standardized Student-t.
 * Both are deferred — call `pdfStudentTMarginal` directly for the latter and
 * sum `pmfBetaBinomial` over nNew trials for the former.
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
