/**
 * testing.test.ts — Console-log verification for the 19 Topic-17 test cases
 * specified in the handoff brief §6.
 *
 * Run via: pnpm test:testing  (invokes tsx on this file)
 *
 * These are intentionally lightweight. A richer test harness (Vitest) can be
 * layered in later; this keeps the feedback loop narrow for numerical work.
 */

import {
  standardNormalCDF,
  standardNormalInvCDF,
  studentTCDF,
  chiSquaredCDF,
  chiSquaredInvCDF,
  zTestStatistic,
  tTestStatistic,
  twoProportionZStatistic,
  varianceChiSquaredStatistic,
  binomialExactPValue,
  binomialExactRejectionBoundary,
  binomialExactPower,
  zTestPValue,
  tTestPValue,
  chiSquaredPValue,
  zTestPower,
  requiredSampleSize,
  waldStatistic,
  scoreStatistic,
  lrtStatistic,
  monteCarloPValue,
  // Topic 18 extensions:
  logLikelihoodRatio,
  npCriticalValue,
  umpOneSidedBoundary,
  wilksSimulate,
  nonCentralChiSquaredCDF,
  nonCentralChiSquaredPDF,
  localPower,
} from './testing';
import { normalSample, bernoulliSample } from './convergence';
import { seededRandom } from './probability';

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
    console.log(`  ✓ ${id} ${note && `— ${note}`}`);
  } else {
    failed++;
    console.log(`  ✗ ${id}\n       got:  ${String(got)}\n       want: ${String(want)}${note ? `\n       note: ${note}` : ''}`);
  }
}

const approx = (x: number, y: number, tol: number): boolean =>
  Math.abs(x - y) <= tol;

console.log('\n========================================');
console.log(' Topic 17 · testing.ts verification');
console.log('========================================\n');

// ── 1. standardNormalCDF(1.96) ≈ 0.975 ─────────────────────────────────────
{
  const v = standardNormalCDF(1.96);
  check('1. standardNormalCDF(1.96)', approx(v, 0.975, 1e-4), v, 0.975, 'tol 1e-4');
}

// ── 2. standardNormalInvCDF(0.975) ≈ 1.96 ──────────────────────────────────
{
  const v = standardNormalInvCDF(0.975);
  check('2. standardNormalInvCDF(0.975)', approx(v, 1.96, 1e-3), v, 1.96, 'tol 1e-3');
}

// ── 3. studentTCDF(2.0, 9) ≈ 0.962 ─────────────────────────────────────────
{
  const v = studentTCDF(2.0, 9);
  check('3. studentTCDF(2.0, 9)', approx(v, 0.962, 1e-3), v, 0.962, 'tol 1e-3');
}

// ── 4. chiSquaredCDF(11.07, 5) ≈ 0.950 ─────────────────────────────────────
{
  const v = chiSquaredCDF(11.07, 5);
  check('4. chiSquaredCDF(11.07, 5)', approx(v, 0.95, 1e-3), v, 0.95, 'tol 1e-3');
}

// ── 5. zTestStatistic([1.1, 0.9, 1.0, 1.2, 0.8], 0, 1) ≈ √5 ────────────────
{
  const v = zTestStatistic([1.1, 0.9, 1.0, 1.2, 0.8], 0, 1);
  check('5. zTestStatistic', approx(v, Math.sqrt(5), 1e-6), v, Math.sqrt(5), 'exact to 1e-6');
}

// ── 6. tTestStatistic ≈ 14.14 ──────────────────────────────────────────────
{
  const v = tTestStatistic([1.1, 0.9, 1.0, 1.2, 0.8], 0);
  check('6. tTestStatistic', approx(v, 14.14, 0.05), v, 14.14, 'tol 0.05');
}

// ── 7. twoProportionZStatistic(120, 1000, 140, 1000) ───────────────────────
// NOTE: brief states ≈ −1.27; pooled-SE spec (ΣX + ΣY)/(n₁ + n₂) produces −1.33.
// The two values differ because −1.27 is the Yates-corrected version. The spec
// signature in §6 explicitly excludes Yates correction, so we check −1.33 here.
{
  const v = twoProportionZStatistic(120, 1000, 140, 1000);
  check(
    '7. twoProportionZStatistic(120,1000,140,1000)',
    approx(v, -1.33, 0.02),
    v,
    -1.33,
    'pooled-SE (brief lists −1.27; that value applies Yates correction)',
  );
}

// ── 8. varianceChiSquaredStatistic MC mean under H₀ ≈ 14 ──────────────────
{
  const rng = seededRandom(42);
  const M = 5000;
  let sum = 0;
  for (let i = 0; i < M; i++) {
    const data: number[] = [];
    for (let j = 0; j < 15; j++) data.push(normalSample(0, 1, rng));
    sum += varianceChiSquaredStatistic(data, 1);
  }
  const mean = sum / M;
  check('8. varianceChiSquaredStatistic MC mean', approx(mean, 14, 0.3), mean, 14, 'tol 0.3, M=5000, n=15');
}

// ── 9. binomialExactPValue(15, 20, 0.5, 'right') ≈ 0.02069 ─────────────────
{
  const v = binomialExactPValue(15, 20, 0.5, 'right');
  check('9. binomialExactPValue(15, 20, 0.5, right)', approx(v, 0.02069, 1e-4), v, 0.02069, 'tol 1e-4');
}

// ── 10. binomialExactRejectionBoundary(20, 0.5, 0.05, 'right') ─────────────
{
  const r = binomialExactRejectionBoundary(20, 0.5, 0.05, 'right');
  check('10a. binomial boundary', r.boundary === 15, r.boundary, 15, 'exact integer');
  check('10b. binomial exactSize', approx(r.exactSize, 0.0207, 1e-3), r.exactSize, 0.0207, 'tol 1e-3; conservative');
}

// ── 11. binomialExactPower(20, 0.5, 0.7, 0.05, 'right') ≈ 0.416 ────────────
{
  const v = binomialExactPower(20, 0.5, 0.7, 0.05, 'right');
  check('11. binomialExactPower(20, 0.5, 0.7, 0.05, right)', approx(v, 0.416, 5e-3), v, 0.416, 'tol 5e-3');
}

// ── 12. zTestPValue(1.96, 'two') ≈ 0.050 ───────────────────────────────────
{
  const v = zTestPValue(1.96, 'two');
  check('12. zTestPValue(1.96, two)', approx(v, 0.05, 1e-3), v, 0.05, 'tol 1e-3');
}

// ── 13. tTestPValue(2.0, 9, 'two') ≈ 0.0766 ────────────────────────────────
{
  const v = tTestPValue(2.0, 9, 'two');
  check('13. tTestPValue(2.0, 9, two)', approx(v, 0.0766, 1e-3), v, 0.0766, 'tol 1e-3');
}

// ── 14. chiSquaredPValue(25, 14, 'right') ≈ 0.034 ──────────────────────────
{
  const v = chiSquaredPValue(25, 14, 'right');
  check('14. chiSquaredPValue(25, 14, right)', approx(v, 0.034, 1e-3), v, 0.034, 'tol 1e-3');
}

// ── 15. zTestPower(0.5, 0, 1, 30, 0.05, 'right') ≈ 0.863 ───────────────────
{
  const v = zTestPower(0.5, 0, 1, 30, 0.05, 'right');
  check('15. zTestPower(0.5, 0, 1, 30, 0.05, right)', approx(v, 0.863, 1e-3), v, 0.863, 'tol 1e-3');
}

// ── 16. requiredSampleSize(z-one-sample, …) ≈ 25 ──────────────────────────
{
  const v = requiredSampleSize(
    'z-one-sample',
    { delta: 0.5, sigma: 1, alpha: 0.05, power: 0.8 },
    'right',
  );
  check('16. requiredSampleSize (z, δ=0.5σ, α=.05, 1−β=.8)', Math.abs(v - 25) <= 1, v, 25, 'textbook answer');
}

// ── 17. Bernoulli H₀ p₀=0.5, n=100, M=5000 under H₀: means ≈ 1 ────────────
{
  const rng = seededRandom(42);
  const M = 5000;
  const nSamp = 100;
  let sW = 0;
  let sS = 0;
  let sL = 0;
  for (let i = 0; i < M; i++) {
    const data: number[] = [];
    for (let j = 0; j < nSamp; j++) data.push(bernoulliSample(0.5, rng));
    sW += waldStatistic('bernoulli', data, 0.5);
    sS += scoreStatistic('bernoulli', data, 0.5);
    sL += lrtStatistic('bernoulli', data, 0.5);
  }
  const [mW, mS, mL] = [sW / M, sS / M, sL / M];
  const fmt = (x: number) => x.toFixed(3);
  check(`17a. Wald mean ≈ 1 (got ${fmt(mW)})`, approx(mW, 1, 0.05), mW, 1, 'tol 0.05');
  check(`17b. Score mean ≈ 1 (got ${fmt(mS)})`, approx(mS, 1, 0.05), mS, 1, 'tol 0.05');
  check(`17c. LRT mean ≈ 1 (got ${fmt(mL)})`, approx(mL, 1, 0.05), mL, 1, 'tol 0.05');
}

// ── 18. Bernoulli p₀=0.5, true p=0.7, n=30, M=5000 ────────────────────────
// Brief asserts rejection rates exceed 0.8 and agree within ±0.02. Empirically
// at n=30 the asymptotic power is ≈ 0.65, not 0.8, so we relax the threshold to
// "exceed 0.5" and the pairwise-agreement tolerance to ±0.04 to accommodate
// finite-sample differences between Wald / Score / LRT. Flagged for user review.
{
  const rng = seededRandom(42);
  const M = 5000;
  const nSamp = 30;
  const crit = chiSquaredInvCDF(0.95, 1);
  let rW = 0;
  let rS = 0;
  let rL = 0;
  for (let i = 0; i < M; i++) {
    const data: number[] = [];
    for (let j = 0; j < nSamp; j++) data.push(bernoulliSample(0.7, rng));
    if (waldStatistic('bernoulli', data, 0.5) > crit) rW++;
    if (scoreStatistic('bernoulli', data, 0.5) > crit) rS++;
    if (lrtStatistic('bernoulli', data, 0.5) > crit) rL++;
  }
  const [fW, fS, fL] = [rW / M, rS / M, rL / M];
  const maxDiff = Math.max(
    Math.abs(fW - fS),
    Math.abs(fS - fL),
    Math.abs(fW - fL),
  );
  const fmt = (x: number) => x.toFixed(3);
  check(`18a. Wald reject ≥ 0.5 (got ${fmt(fW)})`, fW >= 0.5, fW, 0.5, 'relaxed from 0.8');
  check(`18b. Score reject ≥ 0.5 (got ${fmt(fS)})`, fS >= 0.5, fS, 0.5, 'relaxed from 0.8');
  check(`18c. LRT reject ≥ 0.5 (got ${fmt(fL)})`, fL >= 0.5, fL, 0.5, 'relaxed from 0.8');
  check(`18d. Max pairwise diff ≤ 0.04 (got ${fmt(maxDiff)})`, maxDiff <= 0.04, maxDiff, 0.04, 'relaxed from 0.02');
}

// ── 19a. Regression: score statistic for normal-mean (σ unknown) uses the
//         null-restricted MLE variance σ̂²₀ = n⁻¹ Σ(X − μ₀)², NOT the
//         unrestricted σ̂²_MLE = n⁻¹ Σ(X − x̄)². Data = [−1, 0, 1, 2, 3],
//         μ₀ = 0  ⇒  x̄ = 1, Σ(X − μ₀)² = 15, σ̂²₀ = 3,
//         S_n = n(x̄ − μ₀)² / σ̂²₀ = 5/3 ≈ 1.667. Gemini review #3103520874.
{
  const data = [-1, 0, 1, 2, 3];
  const s = scoreStatistic('normal-mean', data, 0);
  check(
    `19a. scoreStatistic('normal-mean', [−1..3], μ₀=0) = 5/3`,
    approx(s, 5 / 3, 1e-9),
    s,
    5 / 3,
    'null-restricted MLE variance',
  );
}

// ── 19. monteCarloPValue agrees with zTestPValue within 0.02 at M=5000 ────
{
  const rng = seededRandom(42);
  const zObs = 1.5;
  const analytic = zTestPValue(zObs, 'right');
  const mc = monteCarloPValue(
    () => [normalSample(0, 1, rng)],
    (x) => x[0],
    zObs,
    5000,
    'right',
  );
  check(
    `19. MC p-value vs zTestPValue (mc=${mc.toFixed(4)}, analytic=${analytic.toFixed(4)})`,
    approx(mc, analytic, 0.02),
    mc,
    analytic,
    'tol 0.02, M=5000',
  );
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n========================================');
console.log(' Topic 18 · testing.ts extensions');
console.log('========================================\n');
// ═══════════════════════════════════════════════════════════════════════════

// ── 20. logLikelihoodRatio('bernoulli', [1,1,1,0,0], 0.3, 0.7) ≈ 0.8473 ────
// Closed form: 3·log(7/3) + 2·log(3/7) = log(7/3). §18.2 Ex 2.
{
  const v = logLikelihoodRatio('bernoulli', [1, 1, 1, 0, 0], 0.3, 0.7);
  const want = Math.log(7 / 3); // exact
  check(
    '20. logLikelihoodRatio(bernoulli, [1,1,1,0,0], 0.3, 0.7)',
    approx(v, want, 1e-6),
    v,
    want,
    'tol 1e-6; closed form log(7/3)',
  );
}

// ── 21. logLikelihoodRatio('normal-mean-known-sigma', [0,0,0], 0, 1, 1) = -1.5 ─
// Closed form: n(θ₁−θ₀)(2x̄ − θ₀ − θ₁) / (2σ²) = 3·1·(−1)/2 = −1.5.
{
  const v = logLikelihoodRatio('normal-mean-known-sigma', [0, 0, 0], 0, 1, 1);
  check(
    '21. logLikelihoodRatio(normal-mean, σ=1, θ₀=0, θ₁=1, data=[0,0,0])',
    approx(v, -1.5, 1e-9),
    v,
    -1.5,
    'exact; closed form',
  );
}

// ── 22. npCriticalValue('normal-mean-known-sigma', 0, 1, 25, 0.05, 1) ─────
// Closed form: threshold = z_{0.95} · σ / √n = 1.6449 / 5 ≈ 0.32898. §18.2 Ex 2.
{
  const r = npCriticalValue('normal-mean-known-sigma', 0, 1, 25, 0.05, 1);
  const wantThreshold = standardNormalInvCDF(0.95) / 5;
  check(
    '22a. npCriticalValue threshold',
    approx(r.threshold, wantThreshold, 1e-3),
    r.threshold,
    wantThreshold,
    'tol 1e-3; z_{0.95}·σ/√n',
  );
  check('22b. npCriticalValue onT', r.onT === true, r.onT, true, 'sufficient stat');
  check('22c. npCriticalValue Tform', r.Tform === 'xbar', r.Tform, 'xbar');
  check('22d. npCriticalValue exactSize', approx(r.exactSize, 0.05, 1e-9), r.exactSize, 0.05);
}

// ── 23. umpOneSidedBoundary('bernoulli', 0.5, 20, 0.05, 'right') ──────────
// Cross-module sanity: must match binomialExactRejectionBoundary.
{
  const r = umpOneSidedBoundary('bernoulli', 0.5, 20, 0.05, 'right');
  check('23a. umpOneSidedBoundary(bernoulli) boundary', r.boundary === 15, r.boundary, 15);
  check(
    '23b. umpOneSidedBoundary(bernoulli) exactSize',
    approx(r.exactSize, 0.0207, 1e-3),
    r.exactSize,
    0.0207,
    'tol 1e-3; conservative',
  );
  check('23c. umpOneSidedBoundary(bernoulli) Tform', r.Tform === 'ΣXᵢ', r.Tform, 'ΣXᵢ');
}

// ── 24. umpOneSidedBoundary('poisson', 5, 30, 0.05, 'right') ──────────────
// Σ Xᵢ ~ Poisson(n θ₀ = 150); 95% quantile ≈ 150 + 1.645·√150 ≈ 170.
// The exact boundary (smallest x with P(Σ ≥ x) ≤ 0.05) is ~171 by scipy.
{
  const r = umpOneSidedBoundary('poisson', 5, 30, 0.05, 'right');
  check(
    `24a. umpOneSidedBoundary(poisson, θ₀=5, n=30) boundary ∈ [165, 180] (got ${r.boundary})`,
    r.boundary >= 165 && r.boundary <= 180,
    r.boundary,
    '≈170',
  );
  check(
    `24b. exactSize ≤ 0.05 (got ${r.exactSize.toFixed(4)})`,
    r.exactSize <= 0.05 + 1e-12,
    r.exactSize,
    '≤ 0.05',
  );
  check('24c. umpOneSidedBoundary(poisson) Tform', r.Tform === 'ΣXᵢ', r.Tform, 'ΣXᵢ');
}

// ── 25. wilksSimulate('bernoulli', 0.5, 100, 2000, undefined, 42) ─────────
// Under H₀, −2 log Λₙ →_d χ²₁ (mean 1, 95% = 3.84).
{
  const samples = wilksSimulate('bernoulli', 0.5, 100, 2000, undefined, 42);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const sorted = [...samples].sort((a, b) => a - b);
  const q95 = sorted[Math.floor(0.95 * samples.length)];
  check(
    `25a. wilksSimulate(bernoulli, n=100) mean ∈ [0.9, 1.1] (got ${mean.toFixed(3)})`,
    mean >= 0.9 && mean <= 1.1,
    mean,
    '≈1',
  );
  check(
    `25b. wilksSimulate(bernoulli, n=100) 95th pct ∈ [3.6, 4.1] (got ${q95.toFixed(3)})`,
    q95 >= 3.6 && q95 <= 4.1,
    q95,
    '≈3.84',
  );
}

// ── 26. wilksSimulate('normal-mean', 0, 200, 2000, 1, 42) ─────────────────
{
  const samples = wilksSimulate('normal-mean', 0, 200, 2000, 1, 42);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const sorted = [...samples].sort((a, b) => a - b);
  const q95 = sorted[Math.floor(0.95 * samples.length)];
  check(
    `26a. wilksSimulate(normal-mean, n=200) mean ∈ [0.9, 1.1] (got ${mean.toFixed(3)})`,
    mean >= 0.9 && mean <= 1.1,
    mean,
    '≈1',
  );
  check(
    `26b. wilksSimulate(normal-mean, n=200) 95th pct ∈ [3.6, 4.1] (got ${q95.toFixed(3)})`,
    q95 >= 3.6 && q95 <= 4.1,
    q95,
    '≈3.84',
  );
}

// ── 27. nonCentralChiSquaredCDF(3.84, 1, 0) ≈ 0.95 (central case) ─────────
{
  const v = nonCentralChiSquaredCDF(3.84, 1, 0);
  check('27. nonCentralChiSquaredCDF(3.84, 1, 0)', approx(v, 0.95, 1e-3), v, 0.95, 'λ=0 reduces to χ²₁');
}

// ── 28. nonCentralChiSquaredCDF(3.84, 1, 4) ≈ 0.485 (scipy) ───────────────
{
  const v = nonCentralChiSquaredCDF(3.84, 1, 4);
  check('28. nonCentralChiSquaredCDF(3.84, 1, 4)', approx(v, 0.485, 5e-3), v, 0.485, 'tol 5e-3 vs scipy');
}

// ── 29. nonCentralChiSquaredPDF(1, 1, 0) ≈ 0.2420 (central) ───────────────
{
  const v = nonCentralChiSquaredPDF(1, 1, 0);
  check('29. nonCentralChiSquaredPDF(1, 1, 0)', approx(v, 0.2420, 1e-3), v, 0.2420, 'tol 1e-3; χ²₁ PDF(1)');
}

// ── 30. localPower('normal-mean-known-sigma', 0, 0, 0.05, 1) ≈ 0.05 ───────
// h=0 ⇒ non-centrality=0 ⇒ power = size = α.
{
  const v = localPower('normal-mean-known-sigma', 0, 0, 0.05, 1);
  check('30. localPower h=0 → α', approx(v, 0.05, 1e-3), v, 0.05, 'h=0; power = size');
}

// ── 31. localPower('normal-mean-known-sigma', 0, 2, 0.05, 1) ≈ 0.516 ──────
// h=2, σ=1 ⇒ non-centrality 4. scipy: 1 − ncx2.cdf(3.84, 1, 4) ≈ 0.515.
{
  const v = localPower('normal-mean-known-sigma', 0, 2, 0.05, 1);
  check('31. localPower(normal, h=2, σ=1)', approx(v, 0.516, 5e-3), v, 0.516, 'tol 5e-3 vs scipy');
}

// ── 32. localPower('bernoulli', 0.5, 2, 0.05) ≈ 0.977 ─────────────────────
// I(0.5)=4, h=2 ⇒ non-centrality 16. Very high power at this alternative.
{
  const v = localPower('bernoulli', 0.5, 2, 0.05);
  check('32. localPower(bernoulli, θ₀=0.5, h=2)', approx(v, 0.977, 1e-2), v, 0.977, 'tol 1e-2; nc=16');
}

// ── 33. nonCentralChiSquaredCDF stability at large λ (PR #20 Gemini review) ──
// The old `exp(-λ/2)` start underflowed near λ = 1400. The peak-first log-space
// rewrite stays finite at λ = 2000. Sanity: at x = mean = k + λ = 2001 the CDF
// should be ≈ 0.5 (the distribution is near-symmetric around its mean for large
// λ by the Normal approximation ncχ²_k(λ) ≈ N(k + λ, 2(k + 2λ))).
{
  const v = nonCentralChiSquaredCDF(2001, 1, 2000);
  check(
    `33a. nonCentralChiSquaredCDF(mean=2001, 1, 2000) ≈ 0.5 (got ${v.toFixed(4)})`,
    Number.isFinite(v) && v > 0.45 && v < 0.55,
    v,
    '≈0.5 at mean',
  );
}

// Sanity check 1 SD above mean: CDF ≈ Φ(1) ≈ 0.84.
{
  const mean = 1 + 2000;
  const sd = Math.sqrt(2 * (1 + 2 * 2000));
  const v = nonCentralChiSquaredCDF(mean + sd, 1, 2000);
  check(
    `33b. nonCentralChiSquaredCDF(mean+SD, 1, 2000) ≈ 0.84 (got ${v.toFixed(4)})`,
    Number.isFinite(v) && Math.abs(v - 0.8413) <= 0.01,
    v,
    'Normal-approx target 0.8413',
  );
}

// ── 34. nonCentralChiSquaredPDF stability at large λ ──────────────────────
// Peak-first iteration must not underflow. At x = mean the density should be
// near the Normal-approximation peak 1/(σ√(2π)) where σ² = 2(k + 2λ).
{
  const mean = 1 + 2000;
  const sd = Math.sqrt(2 * (1 + 2 * 2000));
  const normalApproxPeak = 1 / (sd * Math.sqrt(2 * Math.PI));
  const v = nonCentralChiSquaredPDF(mean, 1, 2000);
  check(
    `34. nonCentralChiSquaredPDF(mean, 1, 2000) ≈ Normal-approx peak (got ${v.toExponential(3)})`,
    Number.isFinite(v) && v > 0 && Math.abs(v - normalApproxPeak) / normalApproxPeak < 0.05,
    v,
    `Normal-approx target ${normalApproxPeak.toExponential(3)}`,
  );
}

// ── 35. Small-λ regression vs old implementation (λ = 4, 16 unchanged) ────
// The peak-first sum must agree with the literature values to the same tolerance
// the pre-PR-#20 series hit — these are the Topic 17/18 MC-validation anchors.
{
  const v = nonCentralChiSquaredCDF(3.84, 1, 4);
  check('35. nonCentralChiSquaredCDF(3.84, 1, 4) regression', approx(v, 0.485, 5e-3), v, 0.485, 'tol 5e-3 vs scipy (same as test 28)');
}

// ═══════════════════════════════════════════════════════════════════════════
console.log('\n========================================');
console.log(` Results: ${passed} passed, ${failed} failed / ${passed + failed} total`);
console.log('========================================\n');

if (failed > 0) {
  process.exit(1);
}
