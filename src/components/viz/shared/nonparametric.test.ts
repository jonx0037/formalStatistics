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
  // Topic 30 (KDE) — §§8–12.
  gaussianKernel,
  epanechnikovKernel,
  kernelProperties,
  kdeEvaluate,
  silvermanBandwidth,
  ucvBandwidth,
  kdePointwiseCI,
  amiseOptimalBandwidth,
  // Topic 31 (Bootstrap) — §§13–16.
  bootstrapResample,
  nonparametricBootstrap,
  parametricBootstrap,
  kolmogorovDistance,
  kolmogorovDistanceToCdf,
  percentileCI,
  basicCI,
  bcaCI,
  bcaCIWithDiagnostics,
  studentizedCI,
  smoothBootstrap,
  smoothBootstrapBandwidth,
  bootstrapBias,
  biasCorrected,
} from './nonparametric';
import { createSeededRng } from './bayes';
import { cdfStdNormal } from './distributions';

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

// ═══════════════════════════════════════════════════════════════════════════
// Fixtures — NumPy seed=42 Normal(0,1) samples used by T30.9 / T30.10 / T30.11.
//
// Generated via `np.random.seed(42); np.random.standard_normal(500)`.
// NumPy's Mersenne Twister is not portable to JS RNGs, so baking the samples
// is the deterministic way to cross-verify TS vs Python reference values.
// The n=100 test-case reuses .slice(0, 100): np.random.seed(42) starts from
// the same stream head so the first 100 elements of the n=500 draw match a
// fresh n=100 draw at the same seed.
// ═══════════════════════════════════════════════════════════════════════════

/* prettier-ignore */
const SAMPLE_N500_SEED42: number[] = [
  +0.496714153011233, -0.138264301171185, +0.647688538100692, +1.523029856408025, -0.234153374723336, -0.234136956949181,
  +1.579212815507391, +0.767434729152909, -0.469474385934952, +0.542560043585965, -0.463417692812462, -0.465729753570257,
  +0.241962271566034, -1.913280244657798, -1.724917832513033, -0.562287529240973, -1.012831120334424, +0.314247332595274,
  -0.908024075521211, -1.412303701335292, +1.465648768921554, -0.225776300486536, +0.067528204687924, -1.424748186213457,
  -0.544382724525183, +0.110922589709866, -1.150993577422303, +0.375698018345672, -0.600638689918805, -0.291693749793277,
  -0.601706612229397, +1.852278184508938, -0.013497224737934, -1.057710928955900, +0.822544912103189, -1.220843649971022,
  +0.208863595004755, -1.959670123879776, -1.328186048898431, +0.196861235869124, +0.738466579995410, +0.171368281189970,
  -0.115648282388241, -0.301103695589289, -1.478521990367427, -0.719844208394709, -0.460638770959788, +1.057122226218916,
  +0.343618289568461, -1.763040155362734, +0.324083969394795, -0.385082280416317, -0.676922000305959, +0.611676288840868,
  +1.030999522495951, +0.931280119116199, -0.839217523222639, -0.309212375851215, +0.331263431403564, +0.975545127122359,
  -0.479174237845290, -0.185658976663817, -1.106334974006028, -1.196206624080671, +0.812525822394198, +1.356240028570823,
  -0.072010121580334, +1.003532897892024, +0.361636025047634, -0.645119754605124, +0.361395605508414, +1.538036566465969,
  -0.035826039109952, +1.564643655814006, -2.619745104089744, +0.821902504375224, +0.087047068238171, -0.299007350465868,
  +0.091760776535502, -1.987568914600893, -0.219671887837512, +0.357112571511746, +1.477894044741516, -0.518270218273647,
  -0.808493602893188, -0.501757043584537, +0.915402117702074, +0.328751109659684, -0.529760203767039, +0.513267433113356,
  +0.097077549348040, +0.968644990532889, -0.702053093877352, -0.327662146597768, -0.392108153132158, -1.463514948132119,
  +0.296120277064576, +0.261055272179889, +0.005113456642461, -0.234587133375147, -1.415370742050414, -0.420645322765359,
  -0.342714516526769, -0.802277269221619, -0.161285711666009, +0.404050856814538, +1.886185901210530, +0.174577812831839,
  +0.257550390722764, -0.074445915766167, -1.918771215299041, -0.026513875449217, +0.060230209941026, +2.463242112485286,
  -0.192360964781123, +0.301547342333612, -0.034711769705243, -1.168678037619532, +1.142822814515021, +0.751933032686774,
  +0.791031947043047, -0.909387454794739, +1.402794310936099, -1.401851062792281, +0.586857093800270, +2.190455625809979,
  -0.990536325130688, -0.566297729602772, +0.099651365087641, -0.503475654116199, -1.550663431066133, +0.068562974806027,
  -1.062303713726105, +0.473592430635182, -0.919424234233803, +1.549934405017540, -0.783253292336237, -0.322061516205676,
  +0.813517217369670, -1.230864316433955, +0.227459934604129, +1.307142754282428, -1.607483234561228, +0.184633858532304,
  +0.259882794248423, +0.781822871777310, -1.236950710878082, -1.320456613084276, +0.521941565616898, +0.296984673233186,
  +0.250492850345877, +0.346448209496976, -0.680024721578491, +0.232253697161004, +0.293072473298681, -0.714351418026368,
  +1.865774511144757, +0.473832920911788, -1.191303497202649, +0.656553608633830, -0.974681670227321, +0.787084603742452,
  +1.158595579007404, -0.820682318351710, +0.963376129244322, +0.412780926936498, +0.822060159994490, +1.896792982653947,
  -0.245388116002870, -0.753736164357490, -0.889514429625523, -0.815810284965438, -0.077101709414104, +0.341151974816643,
  +0.276690799330019, +0.827183249036024, +0.013001891877907, +1.453534077157317, -0.264656833237956, +2.720169166589619,
  +0.625667347765006, -0.857157556416283, -1.070892498061112, +0.482472415243185, -0.223462785325851, +0.714000494092092,
  +0.473237624573545, -0.072828912656873, -0.846793718068405, -1.514847224685865, -0.446514952067021, +0.856398794323472,
  +0.214093744130204, -1.245738778711988, +0.173180925851182, +0.385317379728837, -0.883857436201133, +0.153725105945528,
  +0.058208718446000, -1.142970297830623, +0.357787360348283, +0.560784526368234, +1.083051243175277, +1.053802052034903,
  -1.377669367957091, -0.937825039915123, +0.515035267208660, +0.513785950912209, +0.515047686306048, +3.852731490654721,
  +0.570890510693167, +1.135565640180599, +0.954001763493202, +0.651391251305798, -0.315269244640346, +0.758969220493267,
  -0.772825214537572, -0.236818606740009, -0.485363547829103, +0.081874139386323, +2.314658566673509, -1.867265192591748,
  +0.686260190374514, -1.612715871189652, -0.471931865789434, +1.088950596967366, +0.064280019095463, -1.077744777929306,
  -0.715303709259968, +0.679597748934676, -0.730366631717137, +0.216458589581975, +0.045571839903814, -0.651600347605817,
  +2.143944089325326, +0.633919022318011, -2.025142586657607, +0.186454314769428, -0.661786464768388, +0.852433334796224,
  -0.792520738432701, -0.114736441466899, +0.504987278980457, +0.865755194170121, -1.200296407055776, -0.334501235840948,
  -0.474945311160956, -0.653329232573712, +1.765454240281097, +0.404981710960956, -1.260883954335045, +0.917861947054776,
  +2.122156197012633, +1.032465260551147, -1.519369965954013, -0.484234072866251, +1.266911149186623, -0.707669465618781,
  +0.443819428146228, +0.774634053429337, -0.926930471578083, -0.059525356061800, -3.241267340069073, -1.024387641334290,
  -0.252568151393160, -1.247783181964849, +1.632411303931635, -1.430141377960633, -0.440044486696984, +0.130740577286091,
  +1.441273289066116, -1.435862151179439, +1.163163752154960, +0.010233061019587, -0.981508651047951, +0.462103474263271,
  +0.199059695573470, -0.600216877158795, +0.069802084990019, -0.385313596861760, +0.113517345251248, +0.662130674521046,
  +1.586016816145352, -1.237815498826849, +2.133033374656267, -1.952087799522502, -0.151785095035583, +0.588317206484576,
  +0.280991867735033, -0.622699519820594, -0.208122250357275, -0.493000934658833, -0.589364756944212, +0.849602097021025,
  +0.357015485965047, -0.692909595260654, +0.899599875433251, +0.307299520876609, +0.812862118838960, +0.629628841923612,
  -0.828995010922073, -0.560181040196970, +0.747293605123262, +0.610370265433465, -0.020901593964148, +0.117327383308782,
  +1.277664895788425, -0.591571388835830, +0.547097381170038, -0.202192652433894, -0.217681203227220, +1.098776851987190,
  +0.825416348988030, +0.813509636000639, +1.305478807154329, +0.021003841632759, +0.681952971294964, -0.310266756593456,
  +0.324166352488442, -0.130143054367685, +0.096995964992718, +0.595157025436914, -0.818220683233473, +2.092387275685460,
  -1.006017381499702, -1.214188612787732, +1.158110873500068, +0.791662693962936, +0.624119817052155, +0.628345509264280,
  -0.012246772846915, -0.897254371485832, +0.075804558193726, -0.677161711512112, +0.975119733417751, -0.147057381502139,
  -0.825497196792512, -0.321385841652993, +0.412931454275624, -0.563724552803975, -0.822220395566431, +0.243687211491912,
  +0.244966571108723, -0.506943175371130, -0.471038305618323, +0.232049937357636, -1.448084341497324, -1.407463774376555,
  -0.718444221252436, -0.213447151711847, +0.310907565598005, +1.475356216949552, +0.857659623202019, -0.159938529963427,
  -0.019016207902689, -1.002529364637809, -0.018513135992390, -0.288658638920138, +0.322718560338089, -0.827230943552323,
  +0.519346514241172, +1.532738913002578, -0.108760148456858, +0.401711722098941, +0.690143991711113, -0.401220471885836,
  +0.224092481810417, +0.012592400781795, +0.097676098548832, -0.773009783855466, +0.024510174258943, +0.497998291245449,
  +1.451143607795042, +0.959270826085207, +2.153182457511556, -0.767347562888050, +0.872320636720678, +0.183342005738352,
  +2.189802933217672, -0.808298285355151, -0.839721842180776, -0.599392645444022, -2.123895724309807, -0.525755021680761,
  -0.759132661553698, +0.150393786476208, +0.341755975777159, +1.876170839215886, +0.950423838186050, -0.576903655662403,
  -0.898414671348358, +0.491919171506506, -1.320233207020642, +1.831458765854354, +1.179440120721287, -0.469175652104705,
  -1.713134529090878, +1.353872374165413, -0.114539845252618, +1.237816311973462, -1.594427658794367, -0.599375022953773,
  +0.005243699718183, +0.046980593764742, -0.450065471479244, +0.622849932347499, -1.067620429382594, -0.142379485021293,
  +0.120295631711899, +0.514438834058749, +0.711614878088890, -1.124642091837869, -1.534114170735622, +1.277676821898509,
  +0.332314011979592, -0.748486536556554, +1.551151975522523, +0.115674634292859, +1.179297184063826, +0.067518481410109,
  +2.060747924881987, +1.755340842443204, -0.248964148479073, +0.971570950954355, +0.645375949585148, +1.368631557532349,
  -0.964923460580104, +0.686051459998439, +1.058424486849588, -1.758739486423114, -1.183258512665775, -2.039232177760101,
  -0.269406834444558, +0.717542255795962, +1.502357052096028, +0.074094780419775, +1.628615545571292, -1.380101458214891,
  -1.703382439355155, -0.055547698896619, +0.384065448939307, -0.032694748094093, -2.067442100039877, -0.089120039512788,
  -1.304469500504853, +0.669672548830038, +0.366598246096848, -0.939879786327355, -0.513866917336694, -1.059213521888952,
  -0.062679097273172, +0.955142320501238, -0.985726046335544, +0.504046515517844, -0.530257618372441, -0.792872832262344,
  -0.107030359954558, -1.035242322419374, -0.553649305347182, -1.197877892588848, +1.964725132916389, +0.035263551971729,
  -0.699725507992585, +0.213979910734222, -0.112328049690830, -0.220969599533223, +0.614166700043425, +0.757507710047305,
  -0.530501147610527, -0.575818240644680, -0.275051697151644, -2.301921164735585, -1.515191062198552, +1.366874267444524,
  +1.644967713501284, -0.249036039556378, +0.576556963055766, +0.311250154543536, +3.078880808455238, +1.119574911434577,
  -0.127917591480767, -0.955540440600426, -1.606446320257573, +0.203463635867223, -0.756350745284303, -1.422253709597674,
  -0.646572884242527, -1.081548003614395, +1.687141635072565, +0.881639756949451, -0.007972641316617, +1.479944138890026,
  +0.077368307647618, -0.861284201328264, +1.523124077269657, +0.538910043684659, -1.037246154326456, -0.190338678083608,
  -0.875618253384757, -1.382799730964336,
];

const SAMPLE_N100_SEED42: number[] = SAMPLE_N500_SEED42.slice(0, 100);

// ═══════════════════════════════════════════════════════════════════════════
// Topic 30 — KDE verification (T30.1–T30.14)
// ═══════════════════════════════════════════════════════════════════════════
//
// Expected values from notebook Cell 14 (authoritative).

// Group A: kernel functions and properties.
check(
  'T30.1.  gaussianKernel(0) = 1/√(2π)',
  approx(gaussianKernel(0), 0.398942280401433, 1e-10),
  gaussianKernel(0),
  0.398942280401433,
  'tol 1e-10',
);

check(
  'T30.2.  epanechnikovKernel(0) = 3/4',
  epanechnikovKernel(0) === 0.75,
  epanechnikovKernel(0),
  0.75,
  'exact',
);

check(
  'T30.3.  epanechnikovKernel(0.5) = 0.5625',
  epanechnikovKernel(0.5) === 0.5625,
  epanechnikovKernel(0.5),
  0.5625,
  'exact — 0.75·(1−0.25)',
);

check(
  'T30.4.  epanechnikovKernel(1.5) = 0',
  epanechnikovKernel(1.5) === 0,
  epanechnikovKernel(1.5),
  0,
  'exact — outside compact support',
);

{
  const v = kernelProperties('gaussian').R;
  check(
    'T30.5.  kernelProperties("gaussian").R = 1/(2√π)',
    approx(v, 0.282094791773878, 1e-8),
    v,
    0.282094791773878,
    'tol 1e-8',
  );
}

{
  const spec = kernelProperties('epanechnikov');
  const ok =
    approx(spec.R, 0.6, 1e-10) &&
    approx(spec.mu2, 0.2, 1e-10) &&
    approx(spec.efficiency, 1.0, 1e-10);
  check(
    'T30.6.  kernelProperties("epanechnikov") R/μ₂/eff',
    ok,
    { R: spec.R, mu2: spec.mu2, efficiency: spec.efficiency },
    { R: 0.6, mu2: 0.2, efficiency: 1.0 },
    'R=3/5, μ₂=1/5, eff=1 — tol 1e-10',
  );
}

// Group B: KDE evaluations.
{
  const v = kdeEvaluate([0], 0, 1, gaussianKernel);
  check(
    'T30.7.  kdeEvaluate([0], 0, 1, gaussian) = φ(0)',
    approx(v, 0.398942280401433, 1e-8),
    v,
    0.398942280401433,
    'closed form, tol 1e-8',
  );
}

{
  const v = kdeEvaluate([0, 1], 0.5, 0.5, gaussianKernel);
  check(
    'T30.8.  kdeEvaluate([0, 1], 0.5, 0.5, gaussian) = 2φ(1)',
    approx(v, 0.483941449038287, 1e-8),
    v,
    0.483941449038287,
    'closed form [φ(1)+φ(−1)]/(2·0.5), tol 1e-8',
  );
}

// Group C: bandwidth selectors (use seed-42 NumPy samples baked below).
{
  const h = silvermanBandwidth(SAMPLE_N100_SEED42);
  check(
    'T30.9.  silvermanBandwidth(N(0,1), n=100, seed=42)',
    approx(h, 0.317080, 1e-4),
    h,
    0.317080,
    'tol 1e-4 — 1.06·min(SD=0.908, IQR/1.34=0.751)·100^(−1/5)',
  );
}

{
  const h = ucvBandwidth(SAMPLE_N100_SEED42, 'gaussian');
  check(
    'T30.10. ucvBandwidth(N(0,1), n=100, seed=42, gaussian)',
    approx(h, 0.465168, 5e-2),
    h,
    0.465168,
    'tol 5e-2 — grid-search discretization + integration-method drift vs Python reference',
  );
}

// Group D: pointwise CI + AMISE-rate.
{
  const h = silvermanBandwidth(SAMPLE_N500_SEED42);
  const ci = kdePointwiseCI(SAMPLE_N500_SEED42, 0, h, 'gaussian', 0.05);
  const ok =
    approx(ci.estimate, 0.388205, 1e-4) &&
    approx(ci.se, 0.027014, 1e-4) &&
    approx(ci.lower, 0.335258, 1e-4) &&
    approx(ci.upper, 0.441153, 1e-4);
  check(
    'T30.11. kdePointwiseCI(N(0,1), n=500, seed=42, x=0, h=Silverman, α=0.05)',
    ok,
    {
      estimate: ci.estimate,
      se: ci.se,
      lower: ci.lower,
      upper: ci.upper,
    },
    { estimate: 0.388205, se: 0.027014, lower: 0.335258, upper: 0.441153 },
    'tol 1e-4 — true f(0)=0.39894 lies inside the CI',
  );
}

{
  const R_F_DD = 3 / (8 * Math.sqrt(Math.PI));  // R(f") for f = Normal(0,1)
  const h = amiseOptimalBandwidth(R_F_DD, 100, 'gaussian');
  check(
    'T30.12. amiseOptimalBandwidth(R(f")=3/(8√π), n=100, gaussian)',
    approx(h, 0.421685, 1e-6),
    h,
    0.421685,
    'closed form (R(K)/(n·μ₂²·R(f")))^(1/5), tol 1e-6',
  );
}

{
  const R_F_DD = 3 / (8 * Math.sqrt(Math.PI));
  const h100 = amiseOptimalBandwidth(R_F_DD, 100, 'gaussian');
  const h1000 = amiseOptimalBandwidth(R_F_DD, 1000, 'gaussian');
  const ratio = h1000 / h100;
  check(
    'T30.13. amiseOptimalBandwidth rate h*(1000)/h*(100) = 10^(−1/5)',
    approx(ratio, 0.630957344480193, 1e-6),
    ratio,
    0.630957344480193,
    'closed form, tol 1e-6',
  );
}

{
  // AMISE* = (5/4) · { R(K)^4 · μ₂(K)² · R(f") }^{1/5} · n^{−4/5}.
  // The R(K)/μ₂/R(f") factors cancel in the ratio — it reduces to n^{−4/5}.
  const R_F_DD = 3 / (8 * Math.sqrt(Math.PI));
  const { R, mu2 } = kernelProperties('gaussian');
  const amiseStar = (n: number): number =>
    (5 / 4) *
    Math.pow(Math.pow(R, 4) * mu2 * mu2 * R_F_DD, 1 / 5) *
    Math.pow(n, -4 / 5);
  const ratio = amiseStar(1000) / amiseStar(100);
  check(
    'T30.14. AMISE* rate AMISE*(1000)/AMISE*(100) = 10^(−4/5)',
    approx(ratio, 0.158489319246111, 1e-6),
    ratio,
    0.158489319246111,
    'closed form, tol 1e-6',
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 31 — Bootstrap verification (T31.1–T31.10)
// ═══════════════════════════════════════════════════════════════════════════
//
// RNG-provenance note. The handoff-brief §6.2 table and notebook Cell 13
// print pin values computed with NumPy's PCG64 generator (via
// np.random.default_rng(42)). The TypeScript harness uses the Park-Miller
// MINSTD LCG from bayes.ts — a DIFFERENT byte stream. Every T31.x pin below
// is therefore recomputed against the LCG-seeded sample and pinned to the
// first captured TS-side value. The notebook numbers remain authoritative
// for Python reproduction and for MDX prose; these pins guard against
// regressions in the TS port.
//
// Tolerance 1e-6 on deterministic helpers (kolmogorovDistance,
// kolmogorovDistanceToCdf, percentileCI/basicCI from a fixed replicates
// array) and on the BCa internal z0/a_hat; 1e-3 on stochastic SEs with
// B ≥ 10000; 5e-3 on the studentizedCI nested-inner variant.

function meanArr(a: readonly number[]): number {
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

function stdArr(a: readonly number[]): number {
  const m = meanArr(a);
  const n = a.length;
  let ss = 0;
  for (const v of a) {
    const d = v - m;
    ss += d * d;
  }
  return Math.sqrt(ss / (n - 1));
}

function medianSorted(a: readonly number[]): number {
  const n = a.length;
  const s = [...a].sort((p, q) => p - q);
  return n % 2 === 0 ? (s[n / 2 - 1] + s[n / 2]) / 2 : s[(n - 1) / 2];
}

function makeNormalSample(seed: number, n: number): number[] {
  const rng = createSeededRng(seed);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = rng.normal();
  return out;
}

function makeExpSample(seed: number, n: number, rate = 1): number[] {
  // Inverse-CDF sampling: X = -log(1-U)/rate, U ~ U[0,1).
  const rng = createSeededRng(seed);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = -Math.log(1 - rng.random()) / rate;
  return out;
}

console.log('\n========================================');
console.log(' Topic 31 · bootstrap verification');
console.log('========================================\n');

// ── T31.1 — nonparametricBootstrap SE (Normal mean, n=100, B=10000) ───────
{
  const x = makeNormalSample(42, 100);
  const rngB = createSeededRng(42);
  const reps = nonparametricBootstrap(x, meanArr, 10000, rngB);
  const se = stdArr(reps);
  check(
    'T31.1 nonparametricBootstrap SE (N(0,1) mean, n=100, B=10000)',
    approx(se, 0.10657132, 1e-6),
    se,
    0.10657132,
    'TS-LCG pin; notebook NumPy-PCG64 value 0.07776935 — RNG divergence expected',
  );
}

// ── T31.2 — percentileCI (same fixture, alpha=0.05) ───────────────────────
{
  const x = makeNormalSample(42, 100);
  const rngB = createSeededRng(42);
  const reps = nonparametricBootstrap(x, meanArr, 10000, rngB);
  const ci = percentileCI(reps, 0.05);
  const okLo = approx(ci.lower, -0.25123898, 1e-6);
  const okHi = approx(ci.upper, 0.16688824, 1e-6);
  check(
    'T31.2 percentileCI ≈ (−0.251, 0.167) [TS-LCG pin]',
    okLo && okHi,
    `(${ci.lower.toFixed(6)}, ${ci.upper.toFixed(6)})`,
    '(-0.25123898, 0.16688824)',
    'tol 1e-6 each; deterministic given TS-LCG replicates',
  );
}

// ── T31.3 — basicCI (same replicates) ─────────────────────────────────────
{
  const x = makeNormalSample(42, 100);
  const rngB = createSeededRng(42);
  const reps = nonparametricBootstrap(x, meanArr, 10000, rngB);
  const thetaHat = meanArr(x);
  const ci = basicCI(thetaHat, reps, 0.05);
  const okLo = approx(ci.lower, -0.24868503, 1e-6);
  const okHi = approx(ci.upper, 0.16944219, 1e-6);
  check(
    'T31.3 basicCI ≈ (−0.249, 0.169) [TS-LCG pin]',
    okLo && okHi,
    `(${ci.lower.toFixed(6)}, ${ci.upper.toFixed(6)})`,
    '(-0.24868503, 0.16944219)',
    'tol 1e-6 each; deterministic given TS-LCG replicates',
  );
}

// ── T31.4 — bcaCI + internal diagnostics (Exp(1) mean, n=100, B=10000) ────
// Internal sanity: z0 and a_hat. Notebook: z0=0.022060, a_hat=0.033213 for
// NumPy PCG64. TS-LCG values differ quantitatively; the important invariant
// is that a_hat > 0 (Exp(1) has positive skew → positive acceleration).
{
  const x = makeExpSample(42, 100);
  const rngB = createSeededRng(42);
  const reps = nonparametricBootstrap(x, meanArr, 10000, rngB);
  const diag = bcaCIWithDiagnostics(x, meanArr, reps, 0.05);
  const okLo = approx(diag.lower, 0.78465171, 1e-6);
  const okHi = approx(diag.upper, 1.11113124, 1e-6);
  const okSign = diag.aHat > 0;
  check(
    'T31.4 bcaCI ≈ (0.785, 1.111) [TS-LCG pin]',
    okLo && okHi,
    `(${diag.lower.toFixed(6)}, ${diag.upper.toFixed(6)})`,
    '(0.78465171, 1.11113124)',
    'tol 1e-6 each; deterministic given TS-LCG replicates and jackknife',
  );
  check(
    'T31.4 internal: a_hat > 0 (Exp(1) right-skew → positive acceleration)',
    okSign,
    `a_hat=${diag.aHat.toFixed(6)}, z0=${diag.z0.toFixed(6)}`,
    'a_hat > 0',
    'skew sign invariant; notebook a_hat=0.0332, z0=0.0221 for NumPy PCG64',
  );
}

// ── T31.5 — studentizedCI (Normal mean, n=100, B=1000, BInner=50) ─────────
{
  const x = makeNormalSample(42, 100);
  const se = (s: readonly number[]): number => stdArr(s) / Math.sqrt(s.length);
  const rngS = createSeededRng(42);
  const ci = studentizedCI(x, meanArr, se, 1000, 50, 0.05, rngS);
  const okLo = approx(ci.lower, -0.24281817, 1e-6);
  const okHi = approx(ci.upper, 0.16494354, 1e-6);
  check(
    'T31.5 studentizedCI ≈ (−0.243, 0.165) [TS-LCG pin]',
    okLo && okHi,
    `(${ci.lower.toFixed(6)}, ${ci.upper.toFixed(6)})`,
    '(-0.24281817, 0.16494354)',
    'tol 1e-6 each; deterministic given TS-LCG outer + inner bootstraps',
  );
}

// ── T31.6 — kolmogorovDistance (two Normal samples, n=500, seeds 42/43) ───
// Fully deterministic given the two LCG-seeded fixtures; tight tol.
{
  const a = makeNormalSample(42, 500);
  const b = makeNormalSample(43, 500);
  const d = kolmogorovDistance(a, b);
  check(
    'T31.6 kolmogorovDistance(seed42, seed43, n=500) = 0.072',
    approx(d, 0.072, 1e-6),
    d,
    0.072,
    'deterministic given TS-LCG fixtures; tol 1e-6',
  );
}

// ── T31.7 — kolmogorovDistanceToCdf (bootstrap scaled means vs N(0,1)) ────
{
  const x = makeNormalSample(42, 100);
  const thetaHat = meanArr(x);
  const rngB = createSeededRng(42);
  const reps = nonparametricBootstrap(x, meanArr, 10000, rngB);
  const scaled = reps.map(r => Math.sqrt(100) * (r - thetaHat));
  const d = kolmogorovDistanceToCdf(scaled, cdfStdNormal);
  check(
    'T31.7 kolmogorovDistanceToCdf(scaled bootstrap means, Φ)',
    approx(d, 0.01750458, 1e-6),
    d,
    0.01750458,
    'TS-LCG pin; tol 1e-6 deterministic',
  );
}

// ── T31.8 — smoothBootstrap SE (Normal median, n=50, Silverman h) ─────────
{
  const x = makeNormalSample(42, 50);
  const hSilver = smoothBootstrapBandwidth(x);
  const rngS = createSeededRng(42);
  const reps = smoothBootstrap(x, medianSorted, 10000, rngS, hSilver);
  const se = stdArr(reps);
  check(
    'T31.8 smoothBootstrap SE (N(0,1) median, n=50, Silverman h)',
    approx(se, 0.16392328, 1e-6),
    se,
    0.16392328,
    `TS-LCG pin; h=${hSilver.toFixed(5)}; notebook 0.14744 (NumPy PCG64); tol 1e-6`,
  );
}

// ── T31.9 — bootstrapBias (ratio of means on two Exp(1) samples) ──────────
{
  const xs = makeExpSample(42, 100);
  const ys = makeExpSample(43, 100);
  const thetaHat = meanArr(xs) / meanArr(ys);
  const rngB = createSeededRng(42);
  const B = 10000;
  const reps: number[] = new Array(B);
  for (let b = 0; b < B; b++) {
    const xb = bootstrapResample(xs, rngB);
    const yb = bootstrapResample(ys, rngB);
    reps[b] = meanArr(xb) / meanArr(yb);
  }
  reps.sort((p, q) => p - q);
  const bias = bootstrapBias(thetaHat, reps);
  check(
    'T31.9 bootstrapBias(ratio of means, n=100, B=10000)',
    approx(bias, 0.01105252, 5e-3),
    bias,
    0.01105252,
    'TS-LCG pin; tol 5e-3',
  );
}

// ── T31.10 — parametricBootstrap SE (Normal MLE, n=100) ───────────────────
{
  const x = makeNormalSample(42, 100);
  // Normal-MLE fit: sampler returns N(μ̂, σ̂²) draws via Box-Muller.
  const fit = (sample: readonly number[]) => {
    const mu = meanArr(sample);
    const sigma = stdArr(sample);
    return (n: number, rng: { normal(): number }): number[] => {
      const s: number[] = new Array(n);
      for (let i = 0; i < n; i++) s[i] = mu + sigma * rng.normal();
      return s;
    };
  };
  const rngP = createSeededRng(42);
  const reps = parametricBootstrap(x, fit, meanArr, 10000, rngP);
  const se = stdArr(reps);
  check(
    'T31.10 parametricBootstrap SE (Normal MLE, n=100, B=10000)',
    approx(se, 0.10756527, 1e-6),
    se,
    0.10756527,
    'TS-LCG pin; tol 1e-6; notebook 0.07716 (NumPy PCG64)',
  );
}

// ── Summary ───────────────────────────────────────────────────────────────
console.log('\n========================================');
console.log(` Result: ${passed} passed, ${failed} failed of ${passed + failed}`);
console.log('========================================\n');

if (failed > 0) {
  console.error('Some tests failed.');
  process.exit(1);
}

// Fixtures live above (before the T30 block) to avoid a temporal-dead-zone
// reference during module evaluation.
