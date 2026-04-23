/**
 * nonparametric.test.ts — Console-log verification for the 15 Topic-29 test
 * pins T29.1–T29.15 specified in the handoff brief §6.2.
 *
 * Run via: pnpm test:nonparametric  (invokes tsx on this file)
 *
 * Matches the existing tsx + check()/approx() harness from testing.test.ts
 * and bayes.test.ts. Brief §6.2 was drafted in Jest style; per repo CLAUDE.md
 * the tsx pattern is the standard — ignore the "vitest/jest" remark in §6.2.
 *
 * Authoritative expected values supersede brief §6.2 for T29.5 and T29.11:
 *   • T29.5 — brief said ~1.2395; notebook Cell 13 (authoritative) = 0.981772.
 *     The correct hand-computation: 1260 · 0.5⁴ · 0.5⁵ · φ(0) = 2.4609375 · (1/√(2π))
 *     ≈ 0.981772. Brief draft double-counted a factor; notebook wins.
 *   • T29.11 — brief said ~0.1667; notebook Cell 13 (authoritative) = 0.233333.
 *     The sup |F_n − F| on sorted [0.1, 0.5, 0.9] against F(x)=x is attained
 *     at x=0.9 where (i/n − F) = (3/3 − 0.9) = 0.1 and ((i−1)/n − F) = (2/3 − 0.9)
 *     = −0.233…; |·| = 0.233. Similarly at x=0.1 where |0 − 0.1| = 0.1 and
 *     |1/3 − 0.1| = 0.233. Max is 0.233, not 0.1667.
 *   • T29.15 — notebook Cell 13 confirms r=6, s=15, actualLevel≈0.958611.
 */

import {
  orderStatistic,
  orderStatisticDensity,
  uniformOrderStatisticDensity,
  ecdf,
  ecdfFn,
  sampleQuantile,
  ksStatistic,
  ksTwoSample,
  kolmogorovCDF,
  kolmogorovQuantile,
  dkwBand,
  quantileCIOrderStatisticBounds,
} from './nonparametric';

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

// Standard-normal CDF Φ and density φ for T29.5.
const SQRT_2PI = Math.sqrt(2 * Math.PI);
function normPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}
// erf via Abramowitz & Stegun 7.1.26 — max error 1.5e-7.
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y =
    1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}
function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

console.log('\n========================================');
console.log(' Topic 29 · nonparametric.ts verification');
console.log('========================================\n');

// ── T29.1. orderStatistic([5,2,8,1,9,3], i=3) = 3 ──────────────────────────
{
  const v = orderStatistic([5, 2, 8, 1, 9, 3], 3);
  check('T29.1.  orderStatistic([5,2,8,1,9,3], i=3)', v === 3, v, 3, 'exact');
}

// ── T29.2. orderStatistic([5,2,8,1,9,3], i=6) = 9 ──────────────────────────
{
  const v = orderStatistic([5, 2, 8, 1, 9, 3], 6);
  check('T29.2.  orderStatistic([5,2,8,1,9,3], i=6)', v === 9, v, 9, 'exact');
}

// ── T29.3. uniformOrderStatisticDensity(0.5, 3, 5) = 1.875 ─────────────────
// Beta(3, 3) density at 0.5 = (5!/(2!2!)) · 0.5² · 0.5² = 30/16 = 1.875.
{
  const v = uniformOrderStatisticDensity(0.5, 3, 5);
  check('T29.3.  uniformOrderStatDensity(u=0.5, i=3, n=5)', approx(v, 1.875, 1e-8), v, 1.875, 'tol 1e-8');
}

// ── T29.4. uniformOrderStatisticDensity(0.5, 5, 10) = 2.4609375 ────────────
// Beta(5, 6) density at 0.5 = C(10,4) · 10 · 0.5⁹ = 252 · 10/512 = 2.4609375.
{
  const v = uniformOrderStatisticDensity(0.5, 5, 10);
  check('T29.4.  uniformOrderStatDensity(u=0.5, i=5, n=10)', approx(v, 2.4609375, 1e-8), v, 2.4609375, 'tol 1e-8');
}

// ── T29.5. orderStatisticDensity(0, 5, 10, Φ, φ) ≈ 0.981772 ────────────────
// AUTHORITATIVE (notebook Cell 13); brief §6.2 provisional 1.2395 is wrong.
{
  const v = orderStatisticDensity(0, 5, 10, normCdf, normPdf);
  check('T29.5.  orderStatDensity(x=0, i=5, n=10, N(0,1))', approx(v, 0.981772, 1e-4), v, 0.981772, 'AUTHORITATIVE per notebook; tol 1e-4');
}

// ── T29.6. ecdf([1,2,3,4,5], x=3) = 0.6 ────────────────────────────────────
{
  const v = ecdf([1, 2, 3, 4, 5], 3);
  check('T29.6.  ecdf([1,2,3,4,5], x=3)', v === 0.6, v, 0.6, 'exact');
}

// ── T29.7. ecdfFn([1,2,3,4,5])(2.5) = 0.4 ─────────────────────────────────
{
  const F = ecdfFn([1, 2, 3, 4, 5]);
  const v = F(2.5);
  check('T29.7.  ecdfFn([1,2,3,4,5])(x=2.5)', v === 0.4, v, 0.4, 'exact');
}

// ── T29.8. sampleQuantile([1,2,3,4,5], p=0.5) = 3 ─────────────────────────
{
  const v = sampleQuantile([1, 2, 3, 4, 5], 0.5);
  check('T29.8.  sampleQuantile([1,2,3,4,5], p=0.5)', approx(v, 3, 1e-12), v, 3, 'tol 1e-12');
}

// ── T29.9. sampleQuantile([1,2,3,4,5], p=0.25) = 2 ────────────────────────
{
  const v = sampleQuantile([1, 2, 3, 4, 5], 0.25);
  check('T29.9.  sampleQuantile([1,2,3,4,5], p=0.25)', approx(v, 2, 1e-12), v, 2, 'tol 1e-12');
}

// ── T29.10. sampleQuantile([1,2,3,4,5], p=0.9) = 4.6 ──────────────────────
// Type 7: h = 4·0.9 = 3.6; sorted[3] + 0.6·(sorted[4] − sorted[3]) = 4 + 0.6·1 = 4.6.
{
  const v = sampleQuantile([1, 2, 3, 4, 5], 0.9);
  check('T29.10. sampleQuantile([1,2,3,4,5], p=0.9)', approx(v, 4.6, 1e-12), v, 4.6, 'tol 1e-12');
}

// ── T29.11. ksStatistic([0.1, 0.5, 0.9], F(x)=x) ≈ 0.233333 ──────────────
// AUTHORITATIVE (notebook Cell 13); brief §6.2 provisional 0.1667 is wrong.
{
  const v = ksStatistic([0.1, 0.5, 0.9], (x: number) => x);
  check('T29.11. ksStatistic([0.1,0.5,0.9], F(x)=x)', approx(v, 0.233333, 1e-6), v, 0.233333, 'AUTHORITATIVE per notebook; tol 1e-6');
}

// ── T29.12. kolmogorovCDF(1.0) ≈ 0.730 ────────────────────────────────────
{
  const v = kolmogorovCDF(1.0);
  check('T29.12. kolmogorovCDF(x=1.0)', approx(v, 0.7300, 1e-4), v, 0.7300, 'tol 1e-4');
}

// ── T29.13. kolmogorovQuantile(0.95) ≈ 1.358099 ──────────────────────────
{
  const v = kolmogorovQuantile(0.95);
  check('T29.13. kolmogorovQuantile(p=0.95)', approx(v, 1.358099, 1e-4), v, 1.358099, 'classical KS 5% crit; tol 1e-4');
}

// ── T29.14. dkwBand(n=100, alpha=0.05) ≈ 0.135810 ────────────────────────
// ε = √(log(40)/200) = √(3.68888.../200) ≈ 0.135810.
{
  const v = dkwBand(100, 0.05);
  check('T29.14. dkwBand(n=100, alpha=0.05)', approx(v, 0.135810, 1e-6), v, 0.135810, 'tol 1e-6');
}

// ── T29.15. quantileCIOrderStatisticBounds(20, 0.5, 0.05) = {r:6, s:15, ~0.9586} ──
{
  const v = quantileCIOrderStatisticBounds(20, 0.5, 0.05);
  const okR = v.r === 6;
  const okS = v.s === 15;
  const okLevel = approx(v.actualLevel, 0.958611, 1e-4);
  check(
    'T29.15. quantileCIOrderStatisticBounds(20, 0.5, 0.05)',
    okR && okS && okLevel,
    `{r=${v.r}, s=${v.s}, level=${v.actualLevel.toFixed(6)}}`,
    `{r=6, s=15, level=0.958611}`,
    'tol 1e-4 on level',
  );
}

// ── T29.16+ (regressions from PR #33 review) ─────────────────────────────
// ksTwoSample([1],[1]) should be 0, not 1 — tie handling regression from PR 33.
{
  const v = ksTwoSample([1], [1]);
  check('T29.16. ksTwoSample([1],[1]) (ties)', v === 0, v, 0, 'regression for PR 33 gemini');
}
{
  const v = ksTwoSample([1, 2, 3], [1, 2, 3]);
  check('T29.17. ksTwoSample identical samples', v === 0, v, 0, 'regression for PR 33 gemini');
}
// orderStatisticDensity at Uniform boundaries — Copilot CP1.
// For i=1, n=10, Uniform: at x=0, density = n * (1-0)^{n-1} * 1 = 10.
{
  const uCdf = (u: number) => Math.max(0, Math.min(1, u));
  const uPdf = (u: number) => (u >= 0 && u <= 1 ? 1 : 0);
  const v = orderStatisticDensity(0, 1, 10, uCdf, uPdf);
  check('T29.18. orderStatisticDensity Uniform i=1 at x=0', approx(v, 10, 1e-8), v, 10, 'regression for PR 33 copilot');
}
{
  const uCdf = (u: number) => Math.max(0, Math.min(1, u));
  const uPdf = (u: number) => (u >= 0 && u <= 1 ? 1 : 0);
  const v = orderStatisticDensity(1, 10, 10, uCdf, uPdf);
  check('T29.19. orderStatisticDensity Uniform i=n at x=1', approx(v, 10, 1e-8), v, 10, 'regression for PR 33 copilot');
}

// ── Summary ───────────────────────────────────────────────────────────────
console.log('\n========================================');
console.log(` Result: ${passed} passed, ${failed} failed of ${passed + failed}`);
console.log('========================================\n');

if (failed > 0) {
  console.error('Some tests failed.');
  process.exit(1);
}
