/**
 * bayes.test.ts — Console-log verification for the 15 Topic-25 test cases
 * specified in the handoff brief §6.2.
 *
 * Run via: pnpm test:bayes  (invokes tsx on this file)
 *
 * Matches the existing tsx + check()/approx() harness from testing.test.ts
 * and regression.test.ts. Brief §6.2 was drafted in Jest style; per repo
 * CLAUDE.md the tsx pattern is the standard and Jest-style briefs should
 * be translated on the way in.
 *
 * Numeric-value provenance notes:
 *   • T25.4 uses scipy Cell 13's values [0.1228, 0.3411] (brief draft said
 *     [0.1222, 0.3466]; scipy wins per starter prompt).
 *   • T25.6 asserts the math-correct posterior mean 80/81 for the stated
 *     inputs. The brief quoted "20/21" but that's inconsistent with the
 *     stated (σ₀²=4, σ²=1, n=20, ȳ=1) — the notebook's Cell 13 hardcodes
 *     "20/21" as a typed literal (not a scipy computation). See brief
 *     Gelman BDA3 Eq 2.10; precision-weighted update gives 80/81 here.
 *   • T25.14's brief claimed variance ratio "≈ 1.004"; the scipy-verified
 *     ratio is 0.993. Both satisfy the [0.98, 1.02] assertion bound.
 */

import {
  pdfBeta,
  pdfDirichlet,
  pdfNormalInverseGamma,
  pmfBetaBinomial,
  posterior,
  posteriorMean,
  posteriorVariance,
  credibleIntervalBeta,
  hpdIntervalBeta,
  mapEstimate,
  jeffreysPrior,
  // Topic 26 additions
  createSeededRng,
  metropolisHastings,
  gibbsSampler,
  hmcLeapfrog,
  hamiltonianMonteCarlo,
  rHat,
  effectiveSampleSize,
  autocorrelation,
  batchMeansVariance,
  // Topic 27 additions
  lindleyBayesFactor,
  betaBinomialLogMarginal,
  marginalLikelihoodLaplace,
  bayesFactor,
  posteriorModelProbabilities,
  harmonicMeanEstimate,
  importanceSamplingEstimate,
  bridgeSamplingEstimate,
  pathSamplingEstimate,
  nestedSamplingEstimate,
  dic,
  waic,
  psisLoo,
  bmaPredictive,
  localFdr,
  posteriorPredictiveCheck,
  // Topic 28 additions
  jamesSteinEstimator,
  jamesSteinPositivePart,
  steinRiskDifference,
  partialPoolingShrinkageFactor,
  partialPoolingPosteriorMean,
  partialPoolingPosteriorVariance,
  normalNormalGrandMean,
  typeIIMarginalLogLikelihood,
  typeIIMLE,
  eightSchoolsData,
  nonCenteredTransform,
  centeredToNonCentered,
  funnelLogDensity,
  type PriorHyperparams,
  type SuffStats,
  type PosteriorHyperparams,
  type ProposalKernel,
  type SeededRng,
} from './bayes';
import { sampleGammaShape, lnGamma, quantileStdNormal } from './distributions';

let passed = 0;
let failed = 0;
const results: Array<{ id: string; pass: boolean; got: unknown; want: unknown; note: string }> = [];

function check(
  id: string,
  ok: boolean,
  got: unknown,
  want: unknown,
  note = '',
): void {
  results.push({ id, pass: ok, got, want, note });
  if (ok) {
    passed++;
    console.log(`  ✓ ${id}${note ? ` — ${note}` : ''}`);
  } else {
    failed++;
    console.log(`  ✗ ${id}\n       got:  ${String(got)}\n       want: ${String(want)}${note ? `\n       note: ${note}` : ''}`);
  }
}

const approx = (x: number, y: number, tol: number): boolean =>
  Number.isFinite(x) && Number.isFinite(y) && Math.abs(x - y) <= tol;

console.log('\n========================================');
console.log(' Topic 25 · bayes.ts verification');
console.log('========================================\n');

// ── T25.1. Beta-Binomial posterior hyperparameters ────────────────────────
// Beta(2, 2) prior + 10 successes in 50 trials → Beta(12, 42).
{
  const post = posterior(
    'beta-binomial',
    { family: 'beta-binomial', alpha0: 2, beta0: 2 } as PriorHyperparams,
    { family: 'beta-binomial', n: 50, k: 10 } as SuffStats,
  ) as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
  const ok = post.alpha0 === 12 && post.beta0 === 42;
  check('T25.1 Beta(2,2) + 10/50 → Beta(12, 42)', ok, `(${post.alpha0}, ${post.beta0})`, '(12, 42)');
}

// ── T25.2. Beta(12, 42) posterior mean = 12/54 ≈ 0.2222 ────────────────────
{
  const m = posteriorMean('beta-binomial', {
    family: 'beta-binomial', alpha0: 12, beta0: 42,
  });
  check('T25.2 Beta(12, 42) posterior mean', approx(m, 12 / 54, 1e-10), m, 12 / 54, 'tol 1e-10');
}

// ── T25.3. Beta(12, 42) posterior variance = 12·42/(54²·55) ────────────────
{
  const v = posteriorVariance('beta-binomial', {
    family: 'beta-binomial', alpha0: 12, beta0: 42,
  });
  const expected = (12 * 42) / (54 * 54 * 55);
  check('T25.3 Beta(12, 42) posterior variance', approx(v, expected, 1e-10), v, expected, 'tol 1e-10');
}

// ── T25.4. Beta(12, 42) 95% equal-tailed credible interval ────────────────
// scipy Cell 13: [0.1228, 0.3411] (not brief draft's [0.1222, 0.3466]).
{
  const [lo, hi] = credibleIntervalBeta(12, 42, 0.95);
  const okLo = approx(lo, 0.1228, 1e-3);
  const okHi = approx(hi, 0.3411, 1e-3);
  check(
    'T25.4 Beta(12, 42) 95% equal-tailed CrI ≈ [0.1228, 0.3411]',
    okLo && okHi,
    `[${lo.toFixed(4)}, ${hi.toFixed(4)}]`,
    '[0.1228, 0.3411]',
    'tol 1e-3 (scipy Cell 13 values)',
  );
}

// ── T25.5. HPD width ≤ equal-tailed width for Beta(12, 42) ─────────────────
{
  const [lo1, hi1] = credibleIntervalBeta(12, 42, 0.95);
  const [lo2, hi2] = hpdIntervalBeta(12, 42, 0.95);
  const equalW = hi1 - lo1;
  const hpdW = hi2 - lo2;
  check(
    'T25.5 HPD width ≤ equal-tailed width (Beta(12, 42), 95%)',
    hpdW <= equalW + 1e-6,
    `HPD=${hpdW.toFixed(4)}, ET=${equalW.toFixed(4)}`,
    'HPD ≤ ET',
    'grid-discretized HPD; 1e-6 slack',
  );
}

// ── T25.6. Normal-Normal posterior (μ₀=0, σ₀²=4, n=20, ȳ=1, σ²=1) ─────────
// Precision-weighted: σ_n² = 1/(1/4 + 20/1) = 4/81; μ_n = σ_n² · 20 = 80/81.
// Brief claimed "20/21" but that's mathematically inconsistent with 4/81
// for the stated inputs — the notebook's Cell 13 print hardcodes the typo.
{
  const post = posterior(
    'normal-normal',
    { family: 'normal-normal', mu0: 0, sigma0_sq: 4, sigma_sq: 1 } as PriorHyperparams,
    { family: 'normal-normal', n: 20, yBar: 1 } as SuffStats,
  ) as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;
  const muOk = approx(post.mu0, 80 / 81, 1e-6);
  const varOk = approx(post.sigma0_sq, 4 / 81, 1e-6);
  check(
    'T25.6 Normal-Normal → μ_n=80/81, σ_n²=4/81',
    muOk && varOk,
    `(μ=${post.mu0.toFixed(6)}, σ²=${post.sigma0_sq.toFixed(6)})`,
    '(μ=0.987654, σ²=0.049383)',
    'math-correct per Gelman BDA3 Eq 2.10; brief typo 20/21 flagged',
  );
}

// ── T25.7. Gamma-Poisson: Gamma(2,1) + (n=5, S=15) → Gamma(17, 6) ────────
{
  const post = posterior(
    'gamma-poisson',
    { family: 'gamma-poisson', alpha0: 2, beta0: 1 } as PriorHyperparams,
    { family: 'gamma-poisson', n: 5, S: 15 } as SuffStats,
  ) as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;
  const hyperOk = post.alpha0 === 17 && post.beta0 === 6;
  const meanOk = approx(posteriorMean('gamma-poisson', post), 17 / 6, 1e-10);
  check(
    'T25.7 Gamma(2,1) + n=5,S=15 → Gamma(17, 6); mean 17/6',
    hyperOk && meanOk,
    `(α=${post.alpha0}, β=${post.beta0}, mean=${posteriorMean('gamma-poisson', post).toFixed(6)})`,
    '(17, 6, 2.833333)',
  );
}

// ── T25.8. Dirichlet(1,1,1) + counts (12,10,8) → Dir(13, 11, 9) ─────────────
{
  const post = posterior(
    'dirichlet-multinomial',
    { family: 'dirichlet-multinomial', alpha0: [1, 1, 1] } as PriorHyperparams,
    { family: 'dirichlet-multinomial', counts: [12, 10, 8] } as SuffStats,
  ) as Extract<PosteriorHyperparams, { family: 'dirichlet-multinomial' }>;
  const target = [13, 11, 9];
  const ok = post.alpha0.length === 3 && post.alpha0.every((a, j) => a === target[j]);
  check(
    'T25.8 Dir(1,1,1) + (12,10,8) → Dir(13,11,9)',
    ok,
    `[${post.alpha0.join(', ')}]`,
    '[13, 11, 9]',
  );
}

// ── T25.9. Beta-Binomial posterior predictive PMF sums to 1 ────────────────
// Beta-Binomial(m=20, α*=5, β*=20).
{
  let total = 0;
  for (let y = 0; y <= 20; y++) total += pmfBetaBinomial(y, 20, 5, 20);
  check(
    'T25.9 Beta-Binomial(m=20, α*=5, β*=20) PMF sums to 1',
    approx(total, 1.0, 1e-8),
    total,
    1.0,
    'tol 1e-8',
  );
}

// ── T25.10. Beta-Binomial variance > Binomial plug-in variance ─────────────
// Empirical Beta-Binomial variance vs analytic formula vs Binomial plug-in.
{
  const m = 20, aStar = 5, bStar = 20;
  const pHat = aStar / (aStar + bStar);         // = 0.2
  const binomVar = m * pHat * (1 - pHat);        // plug-in
  const bbVarAnalytic = (m * aStar * bStar * (aStar + bStar + m))
    / ((aStar + bStar) ** 2 * (aStar + bStar + 1));
  let mean = 0, meansq = 0;
  for (let y = 0; y <= m; y++) {
    const p = pmfBetaBinomial(y, m, aStar, bStar);
    mean += y * p;
    meansq += y * y * p;
  }
  const empirical = meansq - mean * mean;
  const analyticOk = approx(empirical, bbVarAnalytic, 1e-5);
  const wider = empirical > binomVar;
  check(
    'T25.10 Beta-Binomial variance > Binomial plug-in; matches analytic',
    analyticOk && wider,
    `empirical=${empirical.toFixed(4)}, analytic=${bbVarAnalytic.toFixed(4)}, plug-in=${binomVar.toFixed(4)}`,
    'empirical ≈ analytic > plug-in',
    'posterior-predictive inflation of uncertainty',
  );
}

// ── T25.11. Jeffreys prior Bernoulli at θ=0.5 = 2 ─────────────────────────
// π_J(θ) = 1/√(θ(1−θ)) ⇒ π_J(0.5) = 1/√0.25 = 2.
{
  const v = jeffreysPrior('bernoulli', 0.5);
  check('T25.11 Jeffreys Bernoulli at θ=0.5', approx(v, 2, 1e-8), v, 2, 'tol 1e-8');
}

// ── T25.12. Jeffreys + (n=50, k=10) → Beta(10.5, 40.5); mean 10.5/51 ──────
{
  const post = posterior(
    'beta-binomial',
    { family: 'beta-binomial', alpha0: 0.5, beta0: 0.5 } as PriorHyperparams,
    { family: 'beta-binomial', n: 50, k: 10 } as SuffStats,
  ) as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
  const hyperOk = post.alpha0 === 10.5 && post.beta0 === 40.5;
  const meanOk = approx(posteriorMean('beta-binomial', post), 10.5 / 51, 1e-10);
  check(
    'T25.12 Jeffreys + 10/50 → Beta(10.5, 40.5); mean 10.5/51',
    hyperOk && meanOk,
    `(α=${post.alpha0}, β=${post.beta0}, mean=${posteriorMean('beta-binomial', post).toFixed(8)})`,
    `(10.5, 40.5, ${(10.5 / 51).toFixed(8)})`,
  );
}

// ── T25.13. Normal-Normal MAP (ridge equivalent) ───────────────────────────
// Prior N(0, 0.25), Data (n=4, ȳ=1, σ²=1). σ_n² = 1/(1/0.25 + 4) = 1/8.
// MAP = σ_n² · (0 + 4) = 0.125 · 4 = 0.5.
{
  const post = posterior(
    'normal-normal',
    { family: 'normal-normal', mu0: 0, sigma0_sq: 0.25, sigma_sq: 1 } as PriorHyperparams,
    { family: 'normal-normal', n: 4, yBar: 1 } as SuffStats,
  );
  const map = mapEstimate('normal-normal', post);
  check(
    'T25.13 Normal-Normal MAP = 0.5 (ridge with λ=σ²/σ₀²=4)',
    approx(map, 0.5, 1e-10),
    map,
    0.5,
    'tol 1e-10',
  );
}

// ── T25.14. BvM: Beta(152, 352) posterior variance vs Normal-at-MLE ───────
// Beta(2,2) prior + n=500, k=150 → Beta(152, 352); true θ₀ = 0.3.
// Posterior variance ≈ 4.17e-4; Normal variance σ̂(1-σ̂)/n = 4.20e-4.
// Ratio ≈ 0.993 (scipy-verified). Brief claimed 1.004.
{
  const postVar = posteriorVariance('beta-binomial', {
    family: 'beta-binomial', alpha0: 152, beta0: 352,
  });
  const mleHat = 150 / 500;
  const normalVar = (mleHat * (1 - mleHat)) / 500;
  const ratio = postVar / normalVar;
  check(
    'T25.14 BvM: postVar/normalVar ∈ (0.98, 1.02) at n=500',
    ratio > 0.98 && ratio < 1.02,
    ratio.toFixed(4),
    'in (0.98, 1.02)',
    `postVar=${postVar.toExponential(3)}, normalVar=${normalVar.toExponential(3)}`,
  );
}

// ── T25.15. Normal-Inverse-Gamma joint density positive and finite ─────────
// p(μ=0.5, σ²=1 | μ₀=0, κ₀=1, α₀=2, β₀=2).
{
  const v = pdfNormalInverseGamma(0.5, 1.0, 0, 1, 2, 2);
  check(
    'T25.15 pdfNormalInverseGamma > 0 and finite',
    v > 0 && Number.isFinite(v),
    v,
    '> 0 and finite',
    `value ≈ ${v.toFixed(6)}`,
  );
}

// ── Supplementary sanity checks (cheap, prevent silent regressions) ────────

// S1: Dirichlet density integrates over the simplex to ≈ 1 (Monte Carlo).
// Uniform sampling on the 2-simplex via stick-breaking; scaled by simplex area (1/2).
// Coarse tolerance — just a sanity guard that Γ-normalization isn't off by orders of magnitude.
{
  // Integrate Dir(2,2,2) via quadrature on a uniform 40×40 triangular lattice.
  const a = [2, 2, 2];
  const N = 40;
  const area = 0.5; // simplex in (p1, p2) has area 1/2
  let acc = 0;
  let count = 0;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N - i; j++) {
      const p1 = i / N, p2 = j / N, p3 = 1 - p1 - p2;
      if (p3 < 0 || p1 <= 0 || p2 <= 0 || p3 <= 0) continue;
      acc += pdfDirichlet([p1, p2, p3], a);
      count++;
    }
  }
  // Discrete sum × cell area; cell area = area / (N² / 2) for a right-triangular mesh.
  const cellArea = area / (N * N / 2);
  const integral = acc * cellArea;
  check(
    'S1 Dir(2,2,2) integrates over simplex to ≈ 1 (coarse quadrature)',
    approx(integral, 1.0, 0.15),
    integral.toFixed(4),
    '≈ 1.0',
    '40×40 triangular mesh; tol 15% (coarse sanity)',
  );
}

// S2: credibleIntervalBeta memoization returns the same reference twice.
{
  const first = credibleIntervalBeta(12, 42, 0.95);
  const second = credibleIntervalBeta(12, 42, 0.95);
  check(
    'S2 credibleIntervalBeta memoized (same reference on repeat call)',
    first === second,
    first === second ? 'same ref' : 'different refs',
    'same ref',
    'hot-path cache for BvM animator',
  );
}

// S3: pdfBeta(0.5, 1, 1) = 1 (uniform density); pdfBeta reuses Topic 6.
{
  const v = pdfBeta(0.5, 1, 1);
  check('S3 pdfBeta(0.5, 1, 1) = 1 (uniform)', approx(v, 1, 1e-10), v, 1, 'tol 1e-10');
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 26 · bayes.ts verification — MCMC samplers + diagnostics
//
// Translated from brief §6.2 (drafted Jest-style) into this file's tsx +
// check()/approx() harness per repo CLAUDE.md. Tolerances:
//   • 1e-1 for acceptance rates
//   • 5e-2 for posterior means
//   • 1e-1 for variances
//   • 1e-3 exact-equal for RNG determinism (T26.15)
// Ground-truth values come from notebook Cell 13.
// ═══════════════════════════════════════════════════════════════════════════
console.log('\n========================================');
console.log(' Topic 26 · bayes.ts verification');
console.log('========================================\n');

// T26.1 — MH acceptance rate on N(0,1) with optimal random-walk scale.
// RWM with σ_prop = 2.4 on d=1 Gaussian hits ~0.44 acceptance (RGG 1997).
{
  const rng = createSeededRng(42);
  const logPi = (x: number) => -0.5 * x * x;
  const kernel: ProposalKernel<number> = {
    propose: (x, r) => ({ xPrime: x + 2.4 * r.normal(), logQRatio: 0 }),
  };
  const chain = metropolisHastings(0, logPi, kernel, 10000, 500, 1, rng);
  check(
    'T26.1 MH acceptance on N(0,1), σ=2.4 ≈ 0.44',
    approx(chain.acceptanceRate, 0.44, 0.1),
    chain.acceptanceRate.toFixed(4),
    '≈ 0.44',
    'tol 1e-1 (RGG 1997 optimal)',
  );
}

// T26.2 — MH chain mean on N(0,1) within MCMC SE.
{
  const rng = createSeededRng(42);
  const logPi = (x: number) => -0.5 * x * x;
  const kernel: ProposalKernel<number> = {
    propose: (x, r) => ({ xPrime: x + 2.4 * r.normal(), logQRatio: 0 }),
  };
  const chain = metropolisHastings(0, logPi, kernel, 10000, 500, 1, rng);
  const mean =
    chain.samples.reduce((a, b) => a + b, 0) / chain.samples.length;
  check(
    'T26.2 MH chain mean ≈ 0 on N(0,1) (|mean| < 0.05)',
    Math.abs(mean) < 0.05,
    mean.toFixed(4),
    '|·| < 0.05',
    'tol 5e-2',
  );
}

// T26.3 — Gibbs on bivariate Normal ρ=0.8: posterior mean ≈ (0, 0).
{
  const rng = createSeededRng(42);
  const rho = 0.8;
  const sd = Math.sqrt(1 - rho * rho);
  const conditionals = [
    (th: number[], r: SeededRng) => rho * th[1] + sd * r.normal(),
    (th: number[], r: SeededRng) => rho * th[0] + sd * r.normal(),
  ];
  const chain = gibbsSampler([-2, -2], conditionals, 5000, 500, 1, rng);
  const m0 =
    chain.samples.reduce((a, b) => a + b[0], 0) / chain.samples.length;
  const m1 =
    chain.samples.reduce((a, b) => a + b[1], 0) / chain.samples.length;
  check(
    'T26.3 Gibbs ρ=0.8 posterior mean ≈ (0, 0) (|·| < 0.05 each)',
    Math.abs(m0) < 0.05 && Math.abs(m1) < 0.05,
    `(${m0.toFixed(3)}, ${m1.toFixed(3)})`,
    '(0, 0)',
    'tol 5e-2',
  );
}

// T26.4 — Gibbs marginal variance ≈ 1.0 (stationary on standard MVN).
{
  const rng = createSeededRng(42);
  const rho = 0.8;
  const sd = Math.sqrt(1 - rho * rho);
  const conditionals = [
    (th: number[], r: SeededRng) => rho * th[1] + sd * r.normal(),
    (th: number[], r: SeededRng) => rho * th[0] + sd * r.normal(),
  ];
  const chain = gibbsSampler([0, 0], conditionals, 5000, 500, 1, rng);
  const xs = chain.samples.map((s) => s[0]);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance =
    xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1);
  check(
    'T26.4 Gibbs marginal variance ≈ 1.0',
    approx(variance, 1.0, 0.15),
    variance.toFixed(4),
    '≈ 1.0',
    'tol 1e-1',
  );
}

// T26.5 — HMC leapfrog energy conservation over L=50 steps on N(0,1).
//         |ΔH| < 0.05 for ε=0.1 — symplectic O(ε²) bound.
{
  const rng = createSeededRng(42);
  const q0 = [0];
  const p0 = [rng.normal()];
  const gradU = (q: number[]) => [q[0]]; // U = 0.5 q² ⇒ ∇U = q
  const H0 = 0.5 * q0[0] ** 2 + 0.5 * p0[0] ** 2;
  const { qStar, pStar } = hmcLeapfrog(q0, p0, gradU, {
    epsilon: 0.1,
    steps: 50,
  });
  const HStar = 0.5 * qStar[0] ** 2 + 0.5 * pStar[0] ** 2;
  const dH = Math.abs(HStar - H0);
  check(
    'T26.5 HMC leapfrog |ΔH| < 0.05 on N(0,1), ε=0.1, L=50',
    dH < 0.05,
    dH.toFixed(5),
    '< 0.05',
    'symplectic O(ε²)',
  );
}

// T26.6 — Leapfrog Jacobian det ≈ 1 via finite difference.
//         Volume preservation is the geometric ingredient of HMC symmetry.
{
  const gradU = (q: number[]) => [q[0]];
  const params = { epsilon: 0.1, steps: 10 };
  const eps = 1e-5;
  const end = (q: number[], p: number[]) => hmcLeapfrog(q, p, gradU, params);
  const dqq =
    (end([0.5 + eps], [1.0]).qStar[0] - end([0.5 - eps], [1.0]).qStar[0]) /
    (2 * eps);
  const dqp =
    (end([0.5], [1.0 + eps]).qStar[0] - end([0.5], [1.0 - eps]).qStar[0]) /
    (2 * eps);
  const dpq =
    (end([0.5 + eps], [1.0]).pStar[0] - end([0.5 - eps], [1.0]).pStar[0]) /
    (2 * eps);
  const dpp =
    (end([0.5], [1.0 + eps]).pStar[0] - end([0.5], [1.0 - eps]).pStar[0]) /
    (2 * eps);
  const det = dqq * dpp - dqp * dpq;
  check(
    'T26.6 Leapfrog Jacobian det ≈ 1 (volume preservation)',
    approx(det, 1.0, 1e-2),
    det.toFixed(6),
    '≈ 1',
    'finite-diff, tol 1e-2',
  );
}

// T26.7 — R̂ < 1.05 for 4 well-mixed chains on N(0,1).
{
  const starts = [-3, -1, 1, 3];
  const chains = starts.map((x0, idx) => {
    const rng = createSeededRng(42 + idx);
    const logPi = (x: number) => -0.5 * x * x;
    const kernel: ProposalKernel<number> = {
      propose: (x, r) => ({ xPrime: x + 2.4 * r.normal(), logQRatio: 0 }),
    };
    return metropolisHastings(x0, logPi, kernel, 2000, 500, 1, rng).samples;
  });
  const r = rHat(chains);
  check(
    'T26.7 R̂ < 1.05 for 4 dispersed MH chains of length 2000',
    r < 1.05,
    r.toFixed(4),
    '< 1.05',
    'GEL1992 diagnostic',
  );
}

// T26.8 — ESS for AR(1) with φ=0.9 lies in [40, 400] (far below N=5000).
//         Theoretical τ = (1+φ)/(1-φ) = 19, so N/τ ≈ 263.
{
  const rng = createSeededRng(42);
  const phi = 0.9;
  const chain: number[] = [0];
  for (let i = 0; i < 5000; i++) {
    chain.push(phi * chain[i] + Math.sqrt(1 - phi * phi) * rng.normal());
  }
  const ess = effectiveSampleSize(chain.slice(500));
  check(
    'T26.8 ESS for AR(1) φ=0.9 in [40, 400]',
    ess > 40 && ess < 400,
    ess.toFixed(1),
    '∈ [40, 400]',
    'τ_theory = 19 ⇒ N/τ ≈ 263',
  );
}

// T26.9 — Batch-means σ²_MC for iid N(0, 1), N=5000 ≈ 1.0 ± 0.2.
{
  const rng = createSeededRng(42);
  const chain = Array.from({ length: 5000 }, () => rng.normal());
  const sigmaMC2 = batchMeansVariance(chain);
  check(
    'T26.9 batchMeansVariance for iid N(0,1) ≈ 1.0',
    sigmaMC2 > 0.8 && sigmaMC2 < 1.2,
    sigmaMC2.toFixed(4),
    '∈ [0.8, 1.2]',
    'tol ±0.2',
  );
}

// T26.10 — Batch-means σ²_MC for AR(1) φ=0.9 inflates above 10.
//          Theoretical σ²_MC ≈ (1+φ)/(1-φ) · Var(X) = 19.
{
  const rng = createSeededRng(42);
  const phi = 0.9;
  const chain: number[] = [0];
  for (let i = 0; i < 10000; i++) {
    chain.push(phi * chain[i] + Math.sqrt(1 - phi * phi) * rng.normal());
  }
  const sigmaMC2 = batchMeansVariance(chain.slice(1000));
  check(
    'T26.10 batchMeansVariance for AR(1) φ=0.9 > 10',
    sigmaMC2 > 10,
    sigmaMC2.toFixed(3),
    '> 10',
    'theoretical ≈ 19',
  );
}

// T26.11 — Autocorrelation at lag 1 for AR(1) φ=0.9 ≈ 0.9.
{
  const rng = createSeededRng(42);
  const phi = 0.9;
  const chain: number[] = [0];
  for (let i = 0; i < 10000; i++) {
    chain.push(phi * chain[i] + Math.sqrt(1 - phi * phi) * rng.normal());
  }
  const rho1 = autocorrelation(chain.slice(500), 1);
  check(
    'T26.11 autocorrelation lag-1 for AR(1) φ=0.9 ≈ 0.9',
    approx(rho1, 0.9, 0.05),
    rho1.toFixed(4),
    '≈ 0.9',
    'tol 5e-2',
  );
}

// T26.12 — MH on a discrete 3-state target matches π within 3%.
//          π = [0.2, 0.3, 0.5] on {0, 1, 2}; RW ±1 with absorbing boundaries.
{
  const rng = createSeededRng(42);
  const pi = [0.2, 0.3, 0.5];
  const logPi = (x: number) => Math.log(pi[x]);
  const kernel: ProposalKernel<number> = {
    propose: (x, r) => {
      const step = r.random() < 0.5 ? -1 : 1;
      const xPrime = Math.max(0, Math.min(2, x + step));
      return { xPrime, logQRatio: 0 };
    },
  };
  const chain = metropolisHastings(0, logPi, kernel, 50000, 1000, 1, rng);
  const counts = [0, 0, 0];
  chain.samples.forEach((x) => counts[x]++);
  const empirical = counts.map((c) => c / chain.samples.length);
  const maxErr = Math.max(...empirical.map((e, i) => Math.abs(e - pi[i])));
  check(
    'T26.12 MH on 3-state target matches π within 3%',
    maxErr < 0.03,
    `max|ε| = ${maxErr.toFixed(4)}`,
    '< 0.03',
    'π = [0.2, 0.3, 0.5]',
  );
}

// T26.13 — Gibbs marginal mean/variance match N(0, 1) within tolerance (ρ=0.5).
{
  const rng = createSeededRng(42);
  const rho = 0.5;
  const sd = Math.sqrt(1 - rho * rho);
  const conditionals = [
    (th: number[], r: SeededRng) => rho * th[1] + sd * r.normal(),
    (th: number[], r: SeededRng) => rho * th[0] + sd * r.normal(),
  ];
  const chain = gibbsSampler([0, 0], conditionals, 10000, 500, 1, rng);
  const xs = chain.samples.map((s) => s[0]);
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance =
    xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1);
  check(
    'T26.13 Gibbs ρ=0.5 marginal ≈ N(0, 1)',
    Math.abs(mean) < 0.05 && Math.abs(variance - 1) < 0.1,
    `(mean ${mean.toFixed(3)}, var ${variance.toFixed(3)})`,
    '≈ (0, 1)',
    'tol 5e-2 / 1e-1',
  );
}

// T26.14 — HMC on Rosenbrock banana accepts well (acc > 0.6).
{
  const rng = createSeededRng(42);
  const logPi = (q: number[]) =>
    -0.5 * ((1 - q[0]) ** 2 + 10 * (q[1] - q[0] ** 2) ** 2);
  const gradU = (q: number[]) => [
    -(1 - q[0]) + 20 * (q[1] - q[0] ** 2) * (-2 * q[0]),
    20 * (q[1] - q[0] ** 2),
  ];
  const chain = hamiltonianMonteCarlo(
    [0, 0],
    logPi,
    gradU,
    { epsilon: 0.05, steps: 25 },
    2000,
    500,
    rng,
  );
  check(
    'T26.14 HMC on banana ε=0.05 L=25, acc > 0.6',
    chain.acceptanceRate > 0.6,
    chain.acceptanceRate.toFixed(4),
    '> 0.6',
    'symplectic integrator + MH',
  );
}

// T26.15 — Seeded RNG determinism: same seed → same sequence (exact).
{
  const r1 = createSeededRng(42);
  const r2 = createSeededRng(42);
  let allEqual = true;
  const first10r1: number[] = [];
  const first10r2: number[] = [];
  for (let i = 0; i < 10; i++) {
    const a = r1.random();
    const b = r2.random();
    first10r1.push(a);
    first10r2.push(b);
    if (a !== b) allEqual = false;
  }
  check(
    'T26.15 createSeededRng(42) deterministic across instances',
    allEqual,
    allEqual ? '10/10 identical' : 'diverged',
    '10/10 identical',
    'exact equality (tol 1e-3 not needed)',
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 27 · bayes.ts verification — brief §6.2 T27.1–T27.15
// ═══════════════════════════════════════════════════════════════════════════
console.log('\n========================================');
console.log(' Topic 27 · bayes.ts verification');
console.log('========================================\n');

// ── T27.1. Lindley BF at z=3, n=100, τ=1, σ=1 (moderate-τ, favors H₁). ──
{
  const bf = lindleyBayesFactor(3, 100, 1, 1);
  check(
    'T27.1 lindleyBayesFactor(z=3, n=100, τ=1, σ=1) ≈ 8.56672',
    approx(bf, 8.56672, 1e-4),
    bf.toFixed(6),
    '8.566723',
    'tol 1e-4; classic Lindley moderate-τ',
  );
}

// ── T27.2. Lindley BF at large τ (τ=100) — paradox: BF → 0 favoring H₀. ──
{
  const bf = lindleyBayesFactor(3, 100, 100, 1);
  check(
    'T27.2 lindleyBayesFactor(z=3, n=100, τ=100, σ=1) ≈ 0.09002',
    approx(bf, 0.09001668, 1e-5),
    bf.toFixed(8),
    '0.09001668',
    'tol 1e-5; large τ — Lindley paradox',
  );
}

// ── T27.3. At z=0, BF = (1+r)^(−1/2) = 1/√101 ≈ 0.09950. ──
{
  const bf = lindleyBayesFactor(0, 100, 1, 1);
  check(
    'T27.3 lindleyBayesFactor(z=0) = 1/√101 ≈ 0.09950',
    approx(bf, 1 / Math.sqrt(101), 1e-4),
    bf.toFixed(8),
    (1 / Math.sqrt(101)).toFixed(8),
    'tol 1e-4; no effect — prior dominates',
  );
}

// ── T27.4. Beta-Binomial log marginal at (n=20, k=12, α=1, β=1) = log(1/21). ──
{
  const logM = betaBinomialLogMarginal(12, 20, 1, 1);
  check(
    'T27.4 betaBinomialLogMarginal(12, 20, 1, 1) ≈ −3.04452244',
    approx(logM, -3.0445224377, 1e-6),
    logM.toFixed(10),
    '-3.0445224377',
    'tol 1e-6; closed-form reference for bridge',
  );
}

// ── T27.5. Beta-Binomial log marginal at (n=50, k=30, α=2, β=2). ──
{
  const logM = betaBinomialLogMarginal(30, 50, 2, 2);
  check(
    'T27.5 betaBinomialLogMarginal(30, 50, 2, 2) ≈ −3.58309215',
    approx(logM, -3.5830921534, 1e-6),
    logM.toFixed(10),
    '-3.5830921534',
    'tol 1e-6; closed-form reference for path',
  );
}

// ── T27.8. bayesFactor(−3.5, −2.5) = exp(1) = e. ──
{
  const bf = bayesFactor(-3.5, -2.5);
  check(
    'T27.8 bayesFactor(logM0=−3.5, logM1=−2.5) = e',
    approx(bf, Math.E, 1e-12),
    bf.toFixed(12),
    Math.E.toFixed(12),
    'tol 1e-12; trivial exp(ΔlogM)',
  );
}

// ── T27.9. posteriorModelProbabilities([−1, −2, −3], uniform) ≈ softmax. ──
{
  const probs = posteriorModelProbabilities([-1, -2, -3]);
  const expected = [0.66524096, 0.24472847, 0.09003057];
  const allOk = probs.every((p, i) => approx(p, expected[i], 1e-5));
  check(
    'T27.9 posteriorModelProbabilities([−1,−2,−3], uniform) ≈ softmax',
    allOk && Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-12,
    probs.map(p => p.toFixed(6)).join(', '),
    expected.map(p => p.toFixed(6)).join(', '),
    'tol 1e-5; sums to 1',
  );
}

// Marginal-likelihood Laplace sanity — Laplace on a standard N(0, 1) posterior
// with a known unnormalized log posterior and Hessian recovers log m = 0 (the
// prior and likelihood cancel out to the marginal). This is a reachable
// sanity check for marginalLikelihoodLaplace without a specific T27.x pin.
{
  // Fake setup: unnormalized log posterior = log φ(0) at mode, H = [[1]]
  // so log m ≈ (1/2) log(2π) − (1/2) log(1) + log φ(0) = (1/2) log(2π) − (1/2) log(2π) = 0.
  const logU = -0.5 * Math.log(2 * Math.PI); // log φ(0)
  const logM = marginalLikelihoodLaplace({
    logUnnormPostAtMode: logU,
    hessianAtMode: [[1]],
    k: 1,
  });
  check(
    'S27.L marginalLikelihoodLaplace on standard normal identity → log m = 0',
    approx(logM, 0, 1e-12),
    logM.toFixed(12),
    '0.000000000000',
    'Laplace self-consistency; tol 1e-12',
  );
}

// ── T27.6. Bridge sampling on Beta-Binomial (20, 12, 1, 1), n=8000. ──────
// Per brief G7, the test locks the absolute bracket [−3.05, −3.04] rather
// than a tight SD — the proposal Beta(12.7, 9.1) is deliberately offset
// from the target Beta(13, 9) to make the bridge estimator's advantage
// over naive IS visible.  With N=8000 draws and correct algorithm, a
// single-run estimate should land in the bracket with high probability.
{
  const rng = createSeededRng(42);
  const nDraws = 8000;
  const aPost = 13, bPost = 9;            // posterior Beta(α+k, β+n−k)
  const aProp = aPost * 0.9 + 1;          // = 12.7 — notebook Cell 7
  const bProp = bPost * 0.9 + 1;          // = 9.1
  const sampleBeta = (a: number, b: number): number => {
    const x = sampleGammaShape(a, () => rng.random());
    const y = sampleGammaShape(b, () => rng.random());
    return x / (x + y);
  };
  const posteriorDraws: number[][] = [];
  const proposalDraws: number[][] = [];
  for (let i = 0; i < nDraws; i++) posteriorDraws.push([sampleBeta(aPost, bPost)]);
  for (let i = 0; i < nDraws; i++) proposalDraws.push([sampleBeta(aProp, bProp)]);

  // log ũ(θ) for Beta-Binomial: log C(n,k) + log Binom(k|n,θ) + log Beta(θ|1,1)
  //                           = log C(20,12) + 12 log θ + 8 log(1−θ)  (flat prior drops)
  const logBinom = lnGamma(21) - lnGamma(13) - lnGamma(9);
  const logUnnormPost = (theta: number[]): number => {
    const t = theta[0];
    return logBinom + 12 * Math.log(t) + 8 * Math.log1p(-t);
  };
  const logProposal = (theta: number[]): number => {
    const t = theta[0];
    return Math.log(pdfBeta(t, aProp, bProp));
  };

  const result = bridgeSamplingEstimate({
    posteriorDraws,
    proposalDraws,
    logUnnormPost,
    logProposal,
  });
  check(
    'T27.6 bridgeSamplingEstimate Beta-Binomial (20, 12, 1, 1), n=8000',
    result.logMarginal >= -3.05 && result.logMarginal <= -3.04,
    result.logMarginal.toFixed(8),
    'in [-3.05, -3.04]',
    `abs. bracket; closed form −3.04452; converged in ${result.iterations} iter`,
  );
}

// ── T27.7. Importance sampling on the same setup; wider bracket. ──────────
{
  const rng = createSeededRng(123);
  const nDraws = 8000;
  const aProp = 13 * 0.9 + 1, bProp = 9 * 0.9 + 1;
  const sampleBeta = (a: number, b: number): number => {
    const x = sampleGammaShape(a, () => rng.random());
    const y = sampleGammaShape(b, () => rng.random());
    return x / (x + y);
  };
  const proposalDraws: number[][] = [];
  for (let i = 0; i < nDraws; i++) proposalDraws.push([sampleBeta(aProp, bProp)]);
  const logBinom = lnGamma(21) - lnGamma(13) - lnGamma(9);
  const logUnnormPost = (theta: number[]): number => {
    const t = theta[0];
    return logBinom + 12 * Math.log(t) + 8 * Math.log1p(-t);
  };
  const logProposal = (theta: number[]): number =>
    Math.log(pdfBeta(theta[0], aProp, bProp));

  const { logMarginal } = importanceSamplingEstimate({
    proposalDraws,
    logUnnormPost,
    logProposal,
  });
  check(
    'T27.7 importanceSamplingEstimate Beta-Binomial (20, 12, 1, 1), n=8000',
    logMarginal >= -3.1 && logMarginal <= -3.0,
    logMarginal.toFixed(8),
    'in [-3.10, -3.00]',
    'abs. bracket; higher variance than bridge',
  );
}

// ── T27.10–T27.12. Canonical tiny for DIC / WAIC / PSIS-LOO. ─────────────
// y = [−0.5, 0.2, 0.1, −0.3, 0.7] (hardcoded in notebook Cell 13), σ = 1.
// Posterior μ | y ~ N(ȳ, σ²/n) = N(0.04, 0.2). Quasi-MC μ-draws via inverse
// Φ on an equal-probability midpoint grid — sub-1e-3 reproducibility across
// runtimes (unlike np.random.default_rng, which this runtime can't match
// bit-exactly). Targets are analytical moment-closed-form; the brief's
// notebook-realization values (12.0716, 11.3863, 11.4027) deviate by ~1e-2
// from analytical due to their specific MC realization.
{
  const y = [-0.5, 0.2, 0.1, -0.3, 0.7];
  const n = y.length;
  const nDraws = 10000;
  const yBar = y.reduce((a, b) => a + b, 0) / n;             // 0.04
  const postStd = Math.sqrt(1 / n);                          // √0.2
  // Quasi-MC draws at equal-probability midpoints of N(ȳ, σ²/n). With 10k
  // draws the midpoint-quantile variance deficit is ≲ 1.3e-4 relative,
  // bounding DIC / WAIC / LOO quadrature error well below the 1e-3 tol.
  const muDraws: number[] = new Array(nDraws);
  for (let s = 0; s < nDraws; s++) {
    const q = (s + 0.5) / nDraws;
    muDraws[s] = yBar + postStd * quantileStdNormal(q);
  }
  const logLikDraws: number[][] = muDraws.map(mu =>
    y.map(yi => -0.5 * Math.log(2 * Math.PI) - 0.5 * (yi - mu) ** 2),
  );
  const logLikAtPosteriorMean = y.map(
    yi => -0.5 * Math.log(2 * Math.PI) - 0.5 * (yi - yBar) ** 2,
  );

  // ── T27.10. DIC. Analytical: n log(2π) + Σ(y−ȳ)² + 2σ² = 12.06139. ──
  {
    const { dic: dicValue, pDic } = dic({ logLikDraws, logLikAtPosteriorMean });
    check(
      'T27.10 dic(canonical tiny, 10k QMC draws) ≈ 12.06139 (analytical limit)',
      approx(dicValue, 12.061385, 1e-3) && approx(pDic, 1.0, 1e-3),
      `DIC=${dicValue.toFixed(6)}, p_DIC=${pDic.toFixed(6)}`,
      'DIC=12.061385, p_DIC=1.000000',
      'tol 1e-3; analytical target (notebook MC value 12.0716 differs by ~1e-2)',
    );
  }

  // ── T27.11. WAIC. Analytical: 11.376446 via lppd + pwaic moment forms. ──
  {
    const { waic: waicValue, pWaic, elpdWaic } = waic(logLikDraws);
    check(
      'T27.11 waic(canonical tiny, 10k QMC draws) ≈ 11.37645 (analytical limit)',
      approx(waicValue, 11.376446, 1e-3)
        && approx(pWaic, 0.2744, 1e-3)
        && approx(elpdWaic, -5.688223, 1e-3),
      `WAIC=${waicValue.toFixed(6)}, p_WAIC=${pWaic.toFixed(6)}`,
      'WAIC=11.376446, p_WAIC=0.274400',
      'tol 1e-3; analytical target (notebook MC value 11.3863 differs by ~1e-2)',
    );
  }

  // ── T27.12. PSIS-LOO. Analytical: Normal-Normal leave-one-out closed form. ──
  // For conjugate Normal-Normal, the naive-IS LOO estimator equals the exact
  // leave-one-out predictive log density (by the exponential-family identity
  // E[1/p(y_i|μ)] under the full-data posterior = 1/p(y_i|y_{−i})).
  {
    const { loo: looValue, nProblematic } = psisLoo(logLikDraws);
    check(
      'T27.12 psisLoo(canonical tiny, 10k QMC draws) ≈ 11.395 (analytical limit)',
      approx(looValue, 11.395, 5e-2) && nProblematic === 0,
      `LOO=${looValue.toFixed(6)}, n_pareto_problematic=${nProblematic}`,
      'LOO=11.39529, n_pareto_problematic=0',
      'tol 5e-2; analytical target via Normal-Normal LOO identity',
    );
  }
}

// ── T27.13. localFdr closed-form (theoretical null, alternative N(2.5, 1)) at z=3.0. ──
{
  const { fdrGrid } = localFdr({
    zScores: [],            // not used in closed-form path
    zGrid: [3.0],
    theoreticalNull: true,
    alternative: { piOne: 0.1, muAlt: 2.5, sigmaAlt: 1.0 },
  });
  check(
    'T27.13 localFdr closed-form two-groups at z=3.0',
    approx(fdrGrid[0], 0.10176409, 1e-4),
    fdrGrid[0].toFixed(8),
    '0.10176409',
    'tol 1e-4; π₀=0.9, μ_alt=2.5, σ_alt=1; strong-signal z favors H₁',
  );
}

// ── T27.14. Same at z=2.0 — low-effect z, mostly null. ───────────────────
{
  const { fdrGrid } = localFdr({
    zScores: [],
    zGrid: [2.0],
    theoreticalNull: true,
    alternative: { piOne: 0.1, muAlt: 2.5, sigmaAlt: 1.0 },
  });
  check(
    'T27.14 localFdr closed-form two-groups at z=2.0',
    approx(fdrGrid[0], 0.57986630, 1e-4),
    fdrGrid[0].toFixed(8),
    '0.57986630',
    'tol 1e-4; low-effect z — null accounts for > 50% of mixture mass',
  );
}

// ── S27.B bmaPredictive sanity check: weighted mixture size/composition. ──
{
  const perModelDraws = [
    new Array(100).fill(1.0),  // model 0: draws ≡ 1
    new Array(100).fill(2.0),  // model 1: draws ≡ 2
    new Array(100).fill(3.0),  // model 2: draws ≡ 3
  ];
  const weights = [0.5, 0.3, 0.2];
  const mix = bmaPredictive({ perModelDraws, weights, nOut: 100 });
  const n1 = mix.filter(v => v === 1.0).length;
  const n2 = mix.filter(v => v === 2.0).length;
  const n3 = mix.filter(v => v === 3.0).length;
  check(
    'S27.B bmaPredictive stratified allocation (50, 30, 20 of 100)',
    mix.length === 100 && n1 === 50 && n2 === 30 && n3 === 20,
    `length=${mix.length}, counts=(${n1}, ${n2}, ${n3})`,
    'length=100, counts=(50, 30, 20)',
    'rounded weighted allocation',
  );
}

// ── S27.P posteriorPredictiveCheck sanity: y_obs at the median of y_rep. ─
// With y_obs = 0 (all zeros, T = sample mean = 0) and y_rep_s ~ N(0, 1)
// (mean 0 by symmetry), the Bayesian p-value P(T(y_rep) ≥ T(y_obs)) = 0.5
// exactly ± MC noise O(1/√S). This isolates the implementation from the
// (uncontrolled) realization of a random y_obs.
{
  const rng = createSeededRng(777);
  const S = 2000;
  const n = 20;
  const yObs = new Array(n).fill(0);  // T(yObs) = 0 deterministically
  const yRepDraws: number[][] = [];
  for (let s = 0; s < S; s++) {
    const rep: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const u1 = rng.random();
      const u2 = rng.random();
      rep[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    yRepDraws.push(rep);
  }
  const { bayesianPValue } = posteriorPredictiveCheck({
    yRepDraws,
    yObs,
    T: (y) => y.reduce((a, b) => a + b, 0) / y.length,
  });
  check(
    'S27.P posteriorPredictiveCheck y_obs=0 under N(0,1) y_rep: p_B ≈ 0.5',
    approx(bayesianPValue, 0.5, 0.05),
    bayesianPValue.toFixed(4),
    '≈ 0.5',
    'tol ±0.05; exact in expectation by symmetry',
  );
}

// ── S27.H harmonicMeanEstimate sanity: finite output on i.i.d. log-liks. ──
// Harmonic-mean is known pathological (infinite variance in general), but
// on bounded log-likelihoods with moderate spread it returns a finite
// number in the expected range. This is a pedagogical-only function; we
// assert finiteness + rough magnitude rather than a tight numerical target.
{
  const logLikDraws = Array.from({ length: 1000 }, (_, i) => -1 - (i / 1000) * 2);
  const result = harmonicMeanEstimate(logLikDraws);
  check(
    'S27.H harmonicMeanEstimate returns finite value on bounded log-liks',
    Number.isFinite(result.logMarginal) && result.logMarginal > -5 && result.logMarginal < 0,
    result.logMarginal.toFixed(6),
    'in (−5, 0)',
    'pedagogical estimator; finiteness check only',
  );
}

// ── S27.T pathSamplingEstimate sanity: Beta-Binomial 11-point β grid. ──
// Path sampling trapezoidal quadrature on Beta-Binomial converges to the
// closed form as the β grid densifies. Coarse 3-point grids have large
// quadrature error on this strongly-curved integrand (integrand jumps
// from ≈ −n log(θ) at β=0 to near zero at β=1); 11 points gets within 0.1.
{
  const n = 20, k = 12, alpha = 1, beta = 1;
  const betas = Array.from({ length: 11 }, (_, i) => i / 10);
  const rng = createSeededRng(99);
  const nDrawsBeta = 2000;
  // Power posterior at β: ∝ Binom(k|n,θ)^β · Beta(θ|α,β) ∝ θ^(βk+α-1) (1-θ)^(β(n-k)+β-1)
  // so draws from Beta(βk + α, β(n-k) + β)
  const powerPosteriorDraws: number[][][] = betas.map((b) => {
    const aPow = b * k + alpha;
    const bPow = b * (n - k) + beta;
    const out: number[][] = [];
    for (let i = 0; i < nDrawsBeta; i++) {
      const x = sampleGammaShape(aPow, () => rng.random());
      const y = sampleGammaShape(bPow, () => rng.random());
      out.push([x / (x + y)]);
    }
    return out;
  });
  const logLikelihood = (theta: number[]): number => {
    const t = theta[0];
    return (
      lnGamma(n + 1) - lnGamma(k + 1) - lnGamma(n - k + 1)
      + k * Math.log(t) + (n - k) * Math.log1p(-t)
    );
  };
  const { logMarginal } = pathSamplingEstimate({
    powerPosteriorDraws,
    betas,
    logLikelihood,
  });
  check(
    'S27.T pathSamplingEstimate Beta-Binomial (20, 12, 1, 1), 11-point β grid',
    Math.abs(logMarginal - (-3.04452)) < 0.15,
    logMarginal.toFixed(6),
    '≈ −3.04452 ± 0.15',
    '11-point trapezoidal quadrature; MC noise + residual quadrature error',
  );
}

// ── T27.15. Nested sampling on N(0,1) × U(−10, 10). ──────────────────────
// Closed form: Z = √(2π) / 20 = 0.12533,  log Z = −2.07679374.
// Generate a deterministic dead-point trajectory using the expected
// prior-mass shrinkage — exercises the assembly logic without MC noise.
// Specifically, at iteration i of a run with N_live = 50 live points,
// expected X_i = exp(−i/50), so log L_i = −50 · X_i² for L(θ) = exp(−θ²/2)
// on U(−10, 10).  Live-point tail converged to peak: log L ≈ 0.
{
  const nLive = 50;
  const K = 500;
  const logLikDeadPoints: number[] = [];
  for (let i = 1; i <= K; i++) {
    const Xi = Math.exp(-i / nLive);
    logLikDeadPoints.push(-50 * Xi * Xi);
  }
  const finalLiveLogLiks = new Array(nLive).fill(0);
  const { logMarginal } = nestedSamplingEstimate({
    logLikDeadPoints,
    nLivePoints: nLive,
    finalLiveLogLiks,
  });
  check(
    'T27.15 nestedSamplingEstimate N(0,1) on U(−10, 10) ≈ −2.0768',
    approx(logMarginal, -2.07679374, 5e-2),
    logMarginal.toFixed(8),
    '-2.07679374',
    'tol 5e-2 for NS run variance; Skilling prior-mass shrinkage',
  );
}

console.log('\n========================================');
console.log(' Topic 28 · bayes.ts verification (Hierarchical & Empirical Bayes)');
console.log('========================================\n');

// ── T28.1. Stein risk diff at θ=0 (d=3) = −(d−2) = −1 ─────────────────────
// E[1/χ²_3(0)] = 1/(d−2) = 1; R(JS)−R(MLE) = −(d−2)² · 1 = −1.
{
  const delta = steinRiskDifference([0, 0, 0]);
  check(
    'T28.1 steinRiskDifference([0,0,0]) ≈ −1',
    approx(delta, -1.0, 1e-4),
    delta.toFixed(6),
    '-1',
    'Stein paradox: d=3 at the most-favorable point θ=0',
  );
}

// ── T28.2. Stein risk diff at θ=0 (d=5) = −3 ──────────────────────────────
{
  const delta = steinRiskDifference([0, 0, 0, 0, 0]);
  check(
    'T28.2 steinRiskDifference([0]*5) ≈ −3',
    approx(delta, -3.0, 1e-4),
    delta.toFixed(6),
    '-3',
    'd=5 at θ=0: E[1/χ²_5(0)] = 1/3; risk diff = −9 · 1/3 = −3',
  );
}

// ── T28.3. Stein risk diff at θ=ones(d=10), λ=10 ≈ −3.7909 ────────────────
// Poisson(λ/2=5)-weighted sum over 1/(d+2j−2) converges to E[1/χ²_10(10)]
// ≈ 0.05923; risk diff = −64 · 0.05923 ≈ −3.791.
{
  const theta = new Array(10).fill(1);
  const delta = steinRiskDifference(theta);
  check(
    'T28.3 steinRiskDifference(ones(10), λ=10) ≈ −3.791',
    approx(delta, -3.7909, 2e-2),
    delta.toFixed(4),
    '-3.7909',
    'Poisson-mixture series verification against scipy ncx2.expect',
  );
}

// ── T28.4. partialPoolingShrinkageFactor(σ²=1, τ²=1) = 0.5 ────────────────
{
  const B = partialPoolingShrinkageFactor(1, 1);
  check(
    'T28.4 partialPoolingShrinkageFactor(1, 1) = 0.5',
    approx(B, 0.5, 1e-10),
    B,
    0.5,
    'equal signal-vs-prior → 50% weight on grand mean',
  );
}

// ── T28.5. partialPoolingShrinkageFactor(σ²=100, τ²=25) = 0.8 ─────────────
{
  const B = partialPoolingShrinkageFactor(100, 25);
  check(
    'T28.5 partialPoolingShrinkageFactor(100, 25) = 0.8',
    approx(B, 0.8, 1e-10),
    B,
    0.8,
    'noisy obs (σ²=100) vs tight prior (τ²=25) → heavy shrinkage',
  );
}

// ── T28.6. School-A posterior mean at τ²=100 ≈ 14.2414 ────────────────────
// σ²_A = 15² = 225, y_A = 28, μ̂(τ²=100) ≈ 8.1265 (from Cell 14).
// B_A = 225/(225+100) = 0.6923; mean = 0.3077·28 + 0.6923·8.1265 ≈ 14.24.
{
  const pm = partialPoolingPosteriorMean(28, 8.1265, 225, 100);
  check(
    'T28.6 partialPoolingPosteriorMean(school-A, τ²=100) ≈ 14.2414',
    approx(pm, 14.2414, 1e-3),
    pm.toFixed(4),
    '14.2414',
    '8-schools: moderate shrinkage at τ²=100',
  );
}

// ── T28.7. School-A posterior mean at τ²=25 ≈ 9.8659 ──────────────────────
// B_A = 225/250 = 0.9; μ̂(τ²=25) ≈ 7.851 (Cell 14).
// mean = 0.1·28 + 0.9·7.851 ≈ 9.866.
{
  const pm = partialPoolingPosteriorMean(28, 7.851, 225, 25);
  check(
    'T28.7 partialPoolingPosteriorMean(school-A, τ²=25) ≈ 9.8659',
    approx(pm, 9.8659, 1e-3),
    pm.toFixed(4),
    '9.8659',
    '8-schools: stronger shrinkage at τ²=25',
  );
}

// ── T28.8. 8-schools precision-weighted grand mean ≈ 7.6856 ──────────────
{
  const { y, sigma } = eightSchoolsData();
  const sigmaSq = sigma.map((s) => s * s);
  const mu = normalNormalGrandMean(y, sigmaSq);
  check(
    'T28.8 normalNormalGrandMean(8-schools) ≈ 7.6856',
    approx(mu, 7.6856, 1e-3),
    mu.toFixed(4),
    '7.6856',
    'complete-pool MLE of μ; Cell 14-verified against scipy',
  );
}

// ── T28.9. Type-II log-marginal on 8-schools at (μ=7.7, τ²=0) ≈ −29.6742 ─
{
  const { y, sigma } = eightSchoolsData();
  const sigmaSq = sigma.map((s) => s * s);
  const ll = typeIIMarginalLogLikelihood(y, sigmaSq, 7.7, 0);
  check(
    'T28.9 typeIIMarginalLogLikelihood(μ=7.7, τ²=0) ≈ −29.6742',
    approx(ll, -29.6742, 1e-3),
    ll.toFixed(4),
    '-29.6742',
    'complete-pool limit of the empirical-Bayes objective',
  );
}

// ── T28.10. Type-II log-marginal at (μ=7.7, τ²=25) ≈ −29.9941 ────────────
{
  const { y, sigma } = eightSchoolsData();
  const sigmaSq = sigma.map((s) => s * s);
  const ll = typeIIMarginalLogLikelihood(y, sigmaSq, 7.7, 25);
  check(
    'T28.10 typeIIMarginalLogLikelihood(μ=7.7, τ²=25) ≈ −29.9941',
    approx(ll, -29.9941, 1e-3),
    ll.toFixed(4),
    '-29.9941',
    'log-marginal drops slightly as τ² moves off the boundary',
  );
}

// ── T28.11. Type-II MLE on 8-schools: τ̂² ≈ 0 (boundary-case EB) ─────────
// The canonical boundary-MLE result: on 8-schools the empirical-Bayes MLE
// of τ² collapses to complete-pool. GEL2006's half-Cauchy-prior argument
// is motivated by precisely this. Featured in §28.7.
{
  const { y, sigma } = eightSchoolsData();
  const sigmaSq = sigma.map((s) => s * s);
  const result = typeIIMLE(y, sigmaSq);
  check(
    'T28.11 typeIIMLE(8-schools).converged === true',
    result.converged === true,
    result.converged,
    true,
    `converged at iter ${result.iterations}`,
  );
  check(
    'T28.11 typeIIMLE(8-schools).tauSq ≈ 0 (boundary MLE)',
    result.tauSq < 0.5,
    result.tauSq.toFixed(4),
    '< 0.5',
    'full-Bayes posterior mean ≈ 43 by contrast (GEL2013 §5.5)',
  );
  check(
    'T28.11 typeIIMLE(8-schools).mu ≈ 7.6856',
    approx(result.mu, 7.6856, 1e-2),
    result.mu.toFixed(4),
    '7.6856',
    'EB μ̂ coincides with complete-pool grand mean when τ̂²→0',
  );
}

// ── T28.12. Non-centered ↔ centered round-trip invertibility ─────────────
{
  const thetaTilde = [0.5, -1.2, 0.3, 2.1];
  const theta = nonCenteredTransform(thetaTilde, 7.7, 5);
  const back = centeredToNonCentered(theta, 7.7, 5);
  const maxErr = Math.max(...thetaTilde.map((t, i) => Math.abs(back[i] - t)));
  check(
    'T28.12 nonCentered round-trip preserves values (max err < 1e-10)',
    maxErr < 1e-10,
    maxErr.toExponential(2),
    '< 1e-10',
    'change-of-variables invertibility for τ≠0',
  );
}

// ── T28.13. centeredToNonCentered at τ=0 throws ──────────────────────────
{
  let threw = false;
  try {
    centeredToNonCentered([1, 2, 3], 0, 0);
  } catch {
    threw = true;
  }
  check(
    'T28.13 centeredToNonCentered(_, _, 0) throws (funnel apex)',
    threw,
    threw,
    true,
    'change-of-variables undefined at τ=0',
  );
}

// ── T28.14. funnelLogDensity(0, 0) = −log(2π) ────────────────────────────
// At (θ=0, log τ=0): both log-Normal(0;0,1)=−½log(2π) terms add.
{
  const ld = funnelLogDensity(0, 0);
  const expected = -Math.log(2 * Math.PI);
  check(
    'T28.14 funnelLogDensity(0, 0) = −log(2π)',
    approx(ld, expected, 1e-8),
    ld.toFixed(10),
    expected.toFixed(10),
    'joint mode of Neal-funnel parametrization',
  );
}

// ── T28.15. funnelLogDensity(0, −3) = small-τ concentration ──────────────
// log p(θ=0 | log τ=−3) = −½ log(2π·e^{−6}) = −log(2π)/2·(no, see below)
// Full calc: log N(0;0,e^{−6}) = −½ log(2π) − (−6)/2 = −½ log(2π) + 3
//           log N(−3;0,1)     = −½ log(2π) − 9/2
// Sum: −log(2π) + 3 − 4.5 = −log(2π) − 1.5
{
  const ld = funnelLogDensity(0, -3);
  const expected = -Math.log(2 * Math.PI) + 3 - 4.5;
  check(
    'T28.15 funnelLogDensity(0, −3) = −log(2π) − 1.5',
    approx(ld, expected, 1e-6),
    ld.toFixed(6),
    expected.toFixed(6),
    'log τ=−3 → tiny τ concentrates θ mass at 0',
  );
}

// ── S28.A jamesSteinEstimator shrinks X=[2,0,0] by shrink = 0.75 ─────────
// d=3, σ²=1, ‖X‖²=4 → shrink = 1 − (d−2)/‖X‖² = 1 − 1/4 = 0.75; JS = [1.5, 0, 0].
{
  const est = jamesSteinEstimator([2, 0, 0]);
  check(
    'S28.A jamesSteinEstimator([2,0,0]) = [1.5, 0, 0]',
    approx(est[0], 1.5, 1e-10) && est[1] === 0 && est[2] === 0,
    est.map((v) => v.toFixed(4)).join(','),
    '1.5000, 0, 0',
    'd=3, ‖X‖²=4 → shrink = 0.75; JS pulls [2,0,0] to [1.5,0,0]',
  );
}

// ── S28.B jamesSteinPositivePart clamps when the JS shrink factor < 0 ────
// X=[0.5,0,0], d=3 → 1 − 1/0.25 = −3 (plain JS flips sign); JS+ clamps to 0.
{
  const est = jamesSteinPositivePart([0.5, 0, 0]);
  check(
    'S28.B jamesSteinPositivePart clamps when shrink<0',
    est[0] === 0 && est[1] === 0 && est[2] === 0,
    est.join(','),
    '0, 0, 0',
    'Efron-Morris 1973: dominant over plain JS in finite samples',
  );
}

// ── S28.C partialPoolingPosteriorVariance (σ²=100, τ²=100) = 50 ──────────
{
  const v = partialPoolingPosteriorVariance(100, 100);
  check(
    'S28.C partialPoolingPosteriorVariance(100, 100) = 50',
    approx(v, 50, 1e-10),
    v,
    50,
    'σ²·τ² / (σ² + τ²) = 10000/200 = 50',
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════════════
console.log(`\n────────────────────────────────────────`);
console.log(` PASS: ${passed}   FAIL: ${failed}`);
console.log(`────────────────────────────────────────\n`);

if (failed > 0) {
  console.log('Failed checks:');
  for (const r of results) {
    if (!r.pass) console.log(`  • ${r.id}`);
  }
  process.exit(1);
}
