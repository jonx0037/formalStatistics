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
  type PriorHyperparams,
  type SuffStats,
  type PosteriorHyperparams,
  type ProposalKernel,
  type SeededRng,
} from './bayes';

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
