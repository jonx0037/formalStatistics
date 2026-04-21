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
  type PriorHyperparams,
  type SuffStats,
  type PosteriorHyperparams,
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
