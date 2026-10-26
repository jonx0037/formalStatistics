/**
 * timeseries.test.ts — Console-log verification for the Topic-34 test pins
 * T34.1–T34.20 and S34.A–S34.C specified in the handoff brief §6.2.
 *
 * Topic 35 appends T35.1–T35.24 and S35.A–S35.C (stationary-time-series-
 * and-arma brief §6.2; values from notebook Cell 15 of
 * notebooks/stationary-time-series-and-arma/35_stationary_time_series.ipynb).
 * Data-dependent T35 pins read __fixtures__/arma_fixture.json (Cell 4), the
 * market fixture, or src/data/sunspots.ts.
 *
 * Topic 36 appends T36.1–T36.22 and S36.A–S36.C (state-space-models-and-
 * kalman-smoothing brief §6.2; values from notebook Cell 15 of
 * notebooks/state-space-models-and-kalman-smoothing/36_state_space_models.ipynb,
 * each cross-checked there against statsmodels / scipy / dense Gaussian
 * conditioning). Data-dependent pins read __fixtures__/ssm_fixture.json,
 * __fixtures__/arma_fixture.json, and src/data/{nile,co2,sunspots}.ts.
 *
 * Run via: pnpm test:timeseries  (invokes tsx on this file)
 *
 * Matches the tsx + check()/approx() harness of bayes.test.ts and
 * nonparametric.test.ts. Expected values are printed by notebook Cell 16
 * (notebooks/hidden-markov-models/34_hidden_markov_models.ipynb).
 *
 * Pins that need the notebook's data read the 6-decimal market fixture
 * (__fixtures__/market_fixture_1000.json, written by notebook Cell 4); the
 * notebook computes those pins from the same rounded values, so they
 * reproduce exactly. Pins that compare against brute-force enumeration
 * (T34.11–T34.12) use this file's own seeds — no cross-language RNG match
 * is needed or attempted.
 */

import { readFileSync } from 'node:fs';
import { createSeededRng } from './bayes';
import {
  stationaryDistribution,
  matPow,
  sojournLengths,
  sampleHMM,
  forwardBackward,
  viterbi,
  posteriorDecode,
  pathLogProb,
  hmmLogLikelihood,
  baumWelch,
  baumWelchStep,
  initKMeans,
  initRandom,
  relabelByMean,
  hmmFreeParams,
  hmmBIC,
  emissionLikelihoods,
  bruteForceLogLik,
  bruteForceMAP,
  bruteForceMarginals,
  type HMMParams,
  arPoly,
  polyRoots,
  rootsToAR,
  isCausal,
  isInvertible,
  psiWeights,
  piWeights,
  armaACVF,
  armaACF,
  durbinLevinson,
  armaPACF,
  armaSpectrum,
  armaLongRunVariance,
  hacLongRunVariance,
  fft,
  periodogram,
  smoothPeriodogram,
  sampleACVF,
  sampleACF,
  bartlettSE,
  yuleWalker,
  armaCSS,
  whittleFit,
  armaIC,
  simulateARMA,
  garchQMLE,
  garchSqACF,
  garchFourthMomentFinite,
  type Complex,
  matMul,
  matVec,
  transpose,
  cholesky,
  choleskySolve,
  logDetChol,
  solveDiscreteLyapunov,
  KAPPA_DIFFUSE,
  localLevel,
  lltSeasonal,
  armaToStateSpace,
  constantVelocity,
  matern32StateSpace,
  kalmanFilter,
  rtsSmoother,
  ffbs,
  forecast,
  simulateSSM,
  armaExactLogLik,
  armaProfileLogLik,
  armaMLE,
  kalmanInnovations,
  localLevelMLE,
  ssmEMStep,
  ssmEM,
  localLevelSteady,
  steadyStateGain,
  gpSSMPosterior,
} from './timeseries';
import { sunspotValues } from '../../../data/sunspots';
import { nileValues } from '../../../data/nile';
import { co2Values } from '../../../data/co2';
import {
  casinoParams,
  marketParams,
  forbiddenTransitionExample,
} from '../../../data/time-series-data';

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
const maxAbsDiff = (a: number[][], b: number[][]): number =>
  Math.max(...a.map((row, i) => Math.max(...row.map((v, j) => Math.abs(v - b[i][j])))));
const allApprox = (a: number[], b: number[], tol: number): boolean =>
  a.length === b.length && a.every((v, i) => approx(v, b[i], tol));

const fixture: { obs: number[]; states: number[] } = JSON.parse(
  readFileSync(new URL('./__fixtures__/market_fixture_1000.json', import.meta.url), 'utf8'),
);
const xFix = fixture.obs;

// ═══════════════════════════════════════════════════════════════════════════
// Topic 34 — Hidden Markov Models
// ═══════════════════════════════════════════════════════════════════════════
console.log('\nTopic 34 — Hidden Markov Models');
{
  const A = marketParams.transition;

  // T34.1 — stationary distribution of the market chain: (5/7, 2/7).
  const pi = stationaryDistribution(A);
  check('T34.1 stationaryDistribution(marketA) = (5/7, 2/7)',
    allApprox(pi, [5 / 7, 2 / 7], 1e-9), pi.map((v) => v.toFixed(12)).join(', '), '0.714285714286, 0.285714285714');

  // T34.2 — fifty-step transition matrix.
  const A50 = matPow(A, 50);
  check('T34.2 matPow(marketA, 50)',
    maxAbsDiff(A50, [[0.721872876743, 0.278127123257], [0.695317808142, 0.304682191858]]) <= 1e-10,
    JSON.stringify(A50.map((r) => r.map((v) => +v.toFixed(12)))), '[[0.721872876743, 0.278127123257], [0.695317808142, 0.304682191858]]');

  // T34.3 — Ex 3 closed form: (e1 Aⁿ)₁ − π₁ = π₂ λⁿ with λ = 1 − p − q = 0.93.
  const lam = 0.93;
  const okCF = [1, 5, 20].every((n) => approx(matPow(A, n)[0][0] - pi[0], pi[1] * lam ** n, 1e-12));
  check('T34.3 two-state closed form (n = 1, 5, 20)', okCF,
    [1, 5, 20].map((n) => (matPow(A, n)[0][0] - pi[0]).toFixed(15)).join(', '),
    '0.265714285714286, 0.198768105514286, 0.066925392475026');

  // T34.4 — sojourn lengths; the final (right-censored) run is excluded.
  const runs = sojournLengths([0, 0, 1, 1, 1, 0], 2);
  check('T34.4 sojournLengths([0,0,1,1,1,0]) = [[2],[3]]',
    JSON.stringify(runs) === '[[2],[3]]', JSON.stringify(runs), '[[2],[3]]', 'final run censored');

  // T34.5 — Ex 4 joint probability: (1/3)(0.5)(0.9)(0.5)(0.1)(1/6) = 0.00125.
  const lpEx4 = pathLogProb(casinoParams, [5, 5, 0], [1, 1, 0]);
  check('T34.5 pathLogProb(casino, 6 6 1, L L F) = log 0.00125',
    approx(lpEx4, -6.684611727667927, 1e-12), lpEx4, -6.684611727667927);

  // T34.6 — Ex 4 likelihood: forward = brute force.
  const llEx4 = hmmLogLikelihood(casinoParams, [5, 5, 0]);
  const bfEx4 = bruteForceLogLik(casinoParams, [5, 5, 0]);
  check('T34.6 hmmLogLikelihood(casino, 6 6 1) = brute force',
    approx(llEx4, -4.454730085557369, 1e-12) && approx(bfEx4, llEx4, 1e-12), `${llEx4}, ${bfEx4}`, -4.454730085557369);

  // T34.7 — Ex 6: first ten casino rolls, forward = brute force over 2^10 paths.
  const x10 = [0, 1, 4, 5, 3, 3, 1, 1, 0, 3];
  const ll10 = hmmLogLikelihood(casinoParams, x10);
  const bf10 = bruteForceLogLik(casinoParams, x10);
  check('T34.7 casino T = 10: forward = brute force',
    approx(ll10, -18.488737947800267, 1e-10) && approx(bf10, ll10, 1e-10), `${ll10}, ${bf10}`, -18.488737947800267);

  // T34.8–T34.10 — normalization identities on the fixture.
  const fb = forwardBackward(marketParams, xFix);
  const rowDev = (M: number[][]) => Math.max(...M.map((r) => Math.abs(r.reduce((s, v) => s + v, 0) - 1)));
  check('T34.8 scaled alpha rows sum to 1', rowDev(fb.alphaHat) < 1e-12, rowDev(fb.alphaHat).toExponential(1), '< 1e-12');
  const xiTotal = fb.xiSum.flat().reduce((s, v) => s + v, 0);
  check('T34.9 gamma rows sum to 1; xiSum totals T − 1 = 999',
    rowDev(fb.gamma) < 1e-9 && approx(xiTotal, 999, 1e-9), `${rowDev(fb.gamma).toExponential(1)}, ${xiTotal}`, '< 1e-9, 999');
  const prodDev = Math.max(...fb.alphaHat.map((r, t) => Math.abs(r.reduce((s, v, k) => s + v * fb.betaHat[t][k], 0) - 1)));
  check('T34.10 Σ_k alphaHat·betaHat = 1 at every t (Thm 7(i), scaled)', prodDev < 1e-9, prodDev.toExponential(1), '< 1e-9');

  // T34.11 — smoothing marginals match enumeration.
  const x8 = [0, 1, 4, 5, 3, 3, 1, 1];
  const dg = maxAbsDiff(forwardBackward(casinoParams, x8).gamma, bruteForceMarginals(casinoParams, x8));
  check('T34.11 gamma = brute-force marginals (casino, T = 8)', dg < 1e-10, dg.toExponential(1), '< 1e-10');

  // T34.12 — Viterbi = brute-force MAP on 20 random K = 3 categorical problems.
  let okV = 0;
  for (let s = 0; s < 20; s++) {
    const rng = createSeededRng(3416 + s);
    const p = initRandom([], 3, rng, 'categorical', 3);
    const xx = Array.from({ length: 8 }, () => Math.floor(rng.random() * 3));
    if (JSON.stringify(viterbi(p, xx).path) === JSON.stringify(bruteForceMAP(p, xx))) okV++;
  }
  check('T34.12 viterbi = bruteForceMAP on 20 random problems', okV === 20, `${okV}/20`, '20/20');

  // T34.13 — Viterbi's reported log-probability is the log-probability of its path.
  const { obs: xCas } = sampleHMM(casinoParams, 300, createSeededRng(3402));
  const v = viterbi(casinoParams, xCas);
  const lpPath = pathLogProb(casinoParams, xCas, v.path);
  check('T34.13 viterbi.logProb = pathLogProb(viterbi.path)', approx(v.logProb, lpPath, 1e-10), `${v.logProb}, ${lpPath}`, 'equal');

  // T34.14 — Ex 10: posterior decoding uses the forbidden edge 3 → 1; Viterbi does not.
  const { params: pEx10, obs: xEx10 } = forbiddenTransitionExample;
  const zHat = posteriorDecode(forwardBackward(pEx10, xEx10).gamma);
  const vEx10 = viterbi(pEx10, xEx10);
  check('T34.14 Ex 10: posterior path impossible, Viterbi path (1,1,1,1,1) finite',
    pathLogProb(pEx10, xEx10, zHat) === -Infinity && JSON.stringify(zHat) === '[2,0,0,0,0]'
      && JSON.stringify(vEx10.path) === '[0,0,0,0,0]' && approx(vEx10.logProb, -8.638530661597, 1e-10),
    `${JSON.stringify(zHat)} → ${pathLogProb(pEx10, xEx10, zHat)}; ${JSON.stringify(vEx10.path)} → ${vEx10.logProb}`,
    '[2,0,0,0,0] → -Infinity; [0,0,0,0,0] → -8.638530661597');

  // T34.15–T34.16 — Ex 12: Baum–Welch from the deterministic k-means init.
  const init = initKMeans(xFix, 2);
  const initEm = init.emission as { means: number[]; sds: number[] };
  check('T34.16b initKMeans(fixture, 2) means and sds',
    allApprox(initEm.means, [-1.049430193182, 0.842649266071], 1e-9) && allApprox(initEm.sds, [0.995605055259, 0.911364059523], 1e-9),
    `${initEm.means} / ${initEm.sds}`, '[-1.049430193182, 0.842649266071] / [0.995605055259, 0.911364059523]');
  const fit = baumWelch(init, xFix, { maxIter: 500, tol: 1e-8 });
  const diffs = fit.logLikTrace.slice(1).map((v2, i) => v2 - fit.logLikTrace[i]);
  check('T34.15 Baum–Welch trace never decreases', Math.min(...diffs) >= -1e-9, Math.min(...diffs).toExponential(3), '>= -1e-9', 'notebook: 2.718e-09');
  const pf = relabelByMean(fit.params);
  const pfEm = pf.emission as { means: number[]; sds: number[] };
  check('T34.16 fitted parameters match the notebook (25 iterations)',
    fit.iterations === 25
      && allApprox(pf.initial, [1, 0], 1e-6)
      && maxAbsDiff(pf.transition, [[0.9414889479, 0.0585110521], [0.0282614237, 0.9717385763]]) <= 1e-6
      && allApprox(pfEm.means, [-0.040240323, 0.0356893493], 1e-6)
      && allApprox(pfEm.sds, [2.073306647, 0.7109540577], 1e-6),
    JSON.stringify({ it: fit.iterations, A: pf.transition, m: pfEm.means, s: pfEm.sds }),
    'it 25; A [[0.94149, 0.05851], [0.02826, 0.97174]]; m [-0.04024, 0.03569]; s [2.07331, 0.71095]');

  // T34.17 — one categorical M-step keeps every row stochastic.
  const { params: pc } = baumWelchStep(casinoParams, xCas);
  const pcEm = pc.emission as { probs: number[][] };
  check('T34.17 baumWelchStep(casino): rows of A and B sum to 1',
    rowDev(pc.transition) < 1e-12 && rowDev(pcEm.probs) < 1e-12, `${rowDev(pc.transition)}, ${rowDev(pcEm.probs)}`, '< 1e-12');

  // T34.18 — relabeling sorts by mean and leaves the likelihood unchanged.
  const swapped: HMMParams = {
    initial: marketParams.initial.slice().reverse(),
    transition: [[0.95, 0.05], [0.02, 0.98]],
    emission: { kind: 'gaussian', means: [-0.1, 0.05], sds: [2.0, 0.7] },
  };
  const pr = relabelByMean(swapped);
  const dll = Math.abs(hmmLogLikelihood(pr, xFix) - hmmLogLikelihood(swapped, xFix));
  check('T34.18 relabelByMean: ascending means, same likelihood',
    JSON.stringify((pr.emission as { means: number[] }).means) === '[-0.1,0.05]' && dll <= 1e-9,
    `${(pr.emission as { means: number[] }).means}; Δℓ = ${dll.toExponential(1)}`, '[-0.1, 0.05]; 0');

  // T34.19 — parameter count and BIC.
  check('T34.19 hmmFreeParams(3) = 14; hmmBIC(−1000, 3, 2000)',
    hmmFreeParams(3, 'gaussian') === 14 && approx(hmmBIC(-1000, 3, 2000, 'gaussian'), 2106.4126344336, 1e-9),
    `${hmmFreeParams(3, 'gaussian')}, ${hmmBIC(-1000, 3, 2000, 'gaussian')}`, '14, 2106.4126344336');

  // T34.20 — log-likelihood of the true market parameters on the fixture.
  const llTrue = hmmLogLikelihood(marketParams, xFix);
  check('T34.20 hmmLogLikelihood(market, fixture)', approx(llTrue, -1522.120882315, 1e-8), llTrue, -1522.120882315);

  // S34.A — no underflow at T = 5000.
  const { obs: x5000 } = sampleHMM(marketParams, 5000, createSeededRng(3417));
  const ll5000 = hmmLogLikelihood(marketParams, x5000);
  check('S34.A logLik finite at T = 5000', Number.isFinite(ll5000), ll5000, 'finite');

  // S34.B — k-means init is sorted and stochastic.
  const pk = initKMeans(xFix, 3);
  const pkm = (pk.emission as { means: number[] }).means;
  check('S34.B initKMeans(fixture, 3): sorted means, stochastic rows',
    pkm[0] < pkm[1] && pkm[1] < pkm[2] && rowDev(pk.transition) < 1e-12, `${pkm}`, 'ascending');

  // S34.C — with identical rows the HMM is a mixture: γ = responsibilities (Rem 5).
  const pnm: HMMParams = {
    initial: [0.7, 0.3],
    transition: [[0.7, 0.3], [0.7, 0.3]],
    emission: { kind: 'gaussian', means: [0, 2], sds: [1, 1] },
  };
  const xs = [-1, 0.5, 2.5, 1];
  const Bn = emissionLikelihoods(pnm, xs);
  const resp = Bn.map((r) => {
    const w = [0.7 * r[0], 0.3 * r[1]];
    const s = w[0] + w[1];
    return [w[0] / s, w[1] / s];
  });
  const dnm = maxAbsDiff(forwardBackward(pnm, xs).gamma, resp);
  check('S34.C no-memory HMM: gamma = mixture responsibilities', dnm < 1e-12, dnm.toExponential(1), '< 1e-12');
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 35 — Stationary Time Series & ARMA (T35.1–T35.24, S35.A–S35.C)
// ═══════════════════════════════════════════════════════════════════════════
{
  console.log('\nTopic 35 — Stationary Time Series & ARMA');
  const arma: { ar1: number[]; ar2: number[]; arma11: number[]; ar2peak: number[] } = JSON.parse(
    readFileSync(new URL('./__fixtures__/arma_fixture.json', import.meta.url), 'utf8'),
  );
  const cplx = (r: Complex) => `${r.re.toFixed(6)}${r.im >= 0 ? '+' : '-'}${Math.abs(r.im).toFixed(6)}i`;
  const polar = (mod: number, ang: number): Complex => ({ re: mod * Math.cos(ang), im: mod * Math.sin(ang) });

  // T35.1 — Durand–Kerner roots of 1 − z + 0.5z²
  const r1 = polyRoots([1, -1, 0.5]);
  check(
    'T35.1 polyRoots([1, -1, 0.5]) = {1 - i, 1 + i}',
    r1.length === 2 && approx(r1[0].re, 1, 1e-10) && approx(r1[0].im, -1, 1e-10) &&
      approx(r1[1].re, 1, 1e-10) && approx(r1[1].im, 1, 1e-10),
    r1.map(cplx).join(', '),
    '1-1i, 1+1i',
  );

  // T35.2 — roots (1/0.9)e^{±iπ/5} → φ = (1.8 cos π/5, −0.81)
  const phiPeak = rootsToAR([polar(1 / 0.9, Math.PI / 5), polar(1 / 0.9, -Math.PI / 5)]);
  check('T35.2 rootsToAR peak roots', allApprox(phiPeak, [1.456230589875, -0.81], 1e-12), phiPeak, [1.456230589875, -0.81]);

  // T35.3 — causality / invertibility
  const t3 = [isCausal([0.5, 0.3]), isCausal([1.2, -0.2]), isCausal([1.03]), isInvertible([0.8]), isInvertible([2])];
  check('T35.3 isCausal / isInvertible', t3.join() === [true, false, false, true, false].join(), t3, [true, false, false, true, false], 'unit-root case has roots 1 and 5');

  // T35.4–T35.5 — ψ and π weights of ARMA(1,1)
  const psi = psiWeights([0.7], [0.4], 6);
  check('T35.4 psiWeights([0.7],[0.4],6)', allApprox(psi, [1, 1.1, 0.77, 0.539, 0.3773, 0.26411], 1e-12), psi, '[1, 1.1, 0.77, 0.539, 0.3773, 0.26411]');
  const piw = piWeights([0.7], [0.4], 6);
  check('T35.5 piWeights([0.7],[0.4],6)', allApprox(piw, [1, -1.1, 0.44, -0.176, 0.0704, -0.02816], 1e-12), piw, '[1, -1.1, 0.44, -0.176, 0.0704, -0.02816]');

  // T35.6 — exact AR(1) autocovariance
  const g6 = armaACVF({ phi: [0.9], theta: [], sigma2: 1 }, 5);
  check(
    'T35.6 armaACVF AR(1) 0.9',
    allApprox(g6, [5.263157895, 4.736842105, 4.263157895, 3.836842105, 3.453157895, 3.107842105], 1e-9) && approx(g6[0], 1 / 0.19, 1e-12),
    g6,
    'γ(0) = 1/0.19',
  );

  // T35.7–T35.8 — ACF of ARMA(2,2) and MA(2)
  const a7 = armaACF([0.5, 0.2], [0.4, -0.3], 10);
  check(
    'T35.7 armaACF([0.5,0.2],[0.4,-0.3],10)',
    allApprox(a7, [1, 0.698549, 0.413491, 0.346456, 0.255926, 0.197254, 0.149812, 0.114357, 0.087141, 0.066442, 0.050649], 1e-6),
    a7.map((v) => v.toFixed(6)),
    'statsmodels arma_acf (6 dp)',
  );
  const a8 = armaACF([], [0.6, 0.4], 5);
  check(
    'T35.8 armaACF MA(2) cuts off after lag 2',
    approx(a8[1], 0.84 / 1.52, 1e-12) && approx(a8[2], 0.4 / 1.52, 1e-12) && a8.slice(3).every((v) => Math.abs(v) <= 1e-15),
    a8,
    'ρ1 = 0.84/1.52, ρ2 = 0.4/1.52, ρ≥3 = 0',
  );

  // T35.9 — Durbin–Levinson on AR(2) roots 1 ± i (Thms 3, 7)
  const dl9 = durbinLevinson(armaACVF({ phi: [1, -0.5], theta: [], sigma2: 1 }, 6));
  check(
    'T35.9 durbinLevinson AR(2): PACF cutoff, φ row, v6',
    allApprox(dl9.pacf, [2 / 3, -0.5, 0, 0, 0, 0], 1e-12) && allApprox(dl9.phi, [1, -0.5, 0, 0, 0, 0], 1e-12) && approx(dl9.v[6], 1, 1e-12),
    `pacf ${dl9.pacf.map((v) => v.toFixed(6))}; v6 ${dl9.v[6]}`,
    'pacf [2/3, −0.5, 0…]; v6 = 1',
  );

  // T35.10 — PACF of ARMA(1,1)
  const p10 = armaPACF([0.7], [0.4], 10);
  check(
    'T35.10 armaPACF([0.7],[0.4],10)',
    allApprox(p10, [0.818605, -0.294314, 0.11585, -0.046222, 0.018481, -0.007392, 0.002957, -0.001183, 0.000473, -0.000189], 1e-6),
    p10.map((v) => v.toFixed(6)),
    'statsmodels arma_pacf',
  );

  // T35.11 — AR(1) spectrum at 0 and ∫f = γ(0)
  const ar09 = { phi: [0.9], theta: [], sigma2: 1 };
  const f0 = armaSpectrum(ar09, [0])[0];
  const NG = 200001;
  const grid = Array.from({ length: NG }, (_, i) => -Math.PI + (2 * Math.PI * i) / (NG - 1));
  const fg = armaSpectrum(ar09, grid);
  const dw = (2 * Math.PI) / (NG - 1);
  const integral = fg.reduce((s, v, i) => s + (i === 0 || i === NG - 1 ? v / 2 : v), 0) * dw;
  check(
    'T35.11 armaSpectrum AR(1) 0.9: f(0) and ∫f = γ(0)',
    approx(f0, 15.915494309, 1e-9) && approx(integral, 5.263157895, 1e-6),
    `${f0}, ${integral}`,
    '15.915494309, 5.263157895',
  );

  // T35.12 — long-run variance σ²θ(1)²/φ(1)²
  const lr1 = armaLongRunVariance({ phi: [0.9], theta: [], sigma2: 0.19 });
  const lr2 = armaLongRunVariance({ phi: [0.7], theta: [0.4], sigma2: 1 });
  check('T35.12 armaLongRunVariance', approx(lr1, 19, 1e-12) && approx(lr2, 21.777777777778, 1e-12), `${lr1}, ${lr2}`, '19, 21.777777777778');

  // T35.13 — periodogram on fixture.ar2 (direct-DFT path, n = 1000)
  const pg = periodogram(arma.ar2);
  check('T35.13 periodogram(fixture.ar2).f[1..3]', allApprox(pg.f.slice(1, 4), [0.622576, 1.124598, 0.102019], 1e-6), pg.f.slice(1, 4), '[0.622576, 1.124598, 0.102019]');

  // T35.14 — Parseval: (2π/n) Σ_{j=0}^{n−1} I(ω_j) = γ̂(0)
  {
    const n = arma.ar2.length;
    const mean = arma.ar2.reduce((s, v) => s + v, 0) / n;
    const X = fft(arma.ar2.map((v) => v - mean));
    let tot = 0;
    for (let j = 0; j < n; j++) tot += (X.re[j] ** 2 + X.im[j] ** 2) / (2 * Math.PI * n);
    const lhs = ((2 * Math.PI) / n) * tot;
    const g0 = sampleACVF(arma.ar2, 0)[0];
    check('T35.14 Parseval on fixture.ar2', approx(lhs, 2.254405348, 1e-9) && approx(g0, 2.254405348, 1e-9), `${lhs}, ${g0}`, '2.254405348');
  }

  // T35.15 — smoothers on fixture.ar2peak (n = 512, M = 110 for the lag window)
  {
    const x = arma.ar2peak;
    const at = (f: number[]) => [f[0], f[25], f[50]];
    const d = at(smoothPeriodogram(x, 3, 'daniell').f);
    const md = at(smoothPeriodogram(x, 3, 'modifiedDaniell').f);
    const bl = at(smoothPeriodogram(x, 3, 'bartlettLag').f);
    const raw = periodogram(x).f.slice(1);
    const m0 = smoothPeriodogram(x, 0).f;
    check(
      'T35.15 smoothPeriodogram daniell / modifiedDaniell / bartlettLag (m = 3)',
      allApprox(d, [0.266389, 1.537472, 6.566088], 1e-6) &&
        allApprox(md, [0.233187, 1.469154, 7.148838], 1e-6) &&
        allApprox(bl, [0.299809, 1.689696, 6.423934], 1e-6) &&
        m0.length === raw.length && m0.every((v, i) => v === raw[i]),
      `D ${d.map((v) => v.toFixed(6))}; MD ${md.map((v) => v.toFixed(6))}; BL ${bl.map((v) => v.toFixed(6))}`,
      'D [0.266389, 1.537472, 6.566088]; MD [0.233187, 1.469154, 7.148838]; BL [0.299809, 1.689696, 6.423934]; m=0 ≡ raw',
    );
  }

  // T35.16 — sample ACVF on fixture.ar1
  const g16 = sampleACVF(arma.ar1, 5);
  check('T35.16 sampleACVF(fixture.ar1, 5)', allApprox(g16, [4.468839, 3.962818, 3.540452, 3.155558, 2.796726, 2.514322], 1e-6), g16.map((v) => v.toFixed(6)), 'statsmodels acovf');

  // T35.17 — Bartlett SEs on fixture.arma11
  const rho17 = sampleACF(arma.arma11, 3);
  const se1 = bartlettSE(rho17, 1000, 1);
  const se3 = bartlettSE(rho17, 1000, 3);
  check('T35.17 bartlettSE h = 1, 3', approx(se1, 0.031622777, 1e-8) && approx(se3, 0.052335077, 1e-8), `${se1}, ${se3}`, '0.031622777, 0.052335077');

  // T35.18 — Yule–Walker on fixture.ar2
  const yw = yuleWalker(arma.ar2, 2);
  check('T35.18 yuleWalker(fixture.ar2, 2)', allApprox(yw.phi, [0.983465, -0.480983], 1e-6) && approx(yw.sigma2, 0.968706, 1e-6), `${yw.phi}, ${yw.sigma2}`, '[0.983465, −0.480983], 0.968706');

  // T35.19 — CSS on fixture.arma11
  const css = armaCSS(arma.arma11, 1, 1);
  check(
    'T35.19 armaCSS(fixture.arma11, 1, 1)',
    approx(css.phi[0], 0.638968, 1e-4) && approx(css.theta[0], 0.402674, 1e-4) && approx(css.sigma2, 1.001488, 1e-4),
    `${css.phi}, ${css.theta}, ${css.sigma2}`,
    '0.638968, 0.402674, 1.001488',
  );

  // T35.20 — Whittle on fixture.ar2
  const wh = whittleFit(arma.ar2, 2, 0);
  check('T35.20 whittleFit(fixture.ar2, 2, 0)', allApprox(wh.phi, [0.983377, -0.480924], 1e-4) && approx(wh.sigma2, 0.970452, 1e-4), `${wh.phi}, ${wh.sigma2}`, '[0.983377, −0.480924], 0.970452');

  // T35.21 — GARCH(1,1) QMLE on Topic 34's market fixture
  const gq = garchQMLE(fixture.obs);
  check(
    'T35.21 garchQMLE(marketFixture.obs)',
    approx(gq.omega, 0.080256, 1e-3) && approx(gq.alpha, 0.28207, 1e-3) && approx(gq.beta, 0.702774, 1e-3) && approx(gq.loglik, -1557.0093, 1e-3),
    `ω ${gq.omega.toFixed(6)}, α ${gq.alpha.toFixed(6)}, β ${gq.beta.toFixed(6)}, ℓ ${gq.loglik.toFixed(4)}`,
    'ω 0.080256, α 0.282070, β 0.702774, ℓ −1557.0093',
  );

  // T35.22 — squared-return ACF and fourth-moment condition (Ex 18)
  const sq = garchSqACF(0.1, 0.85, 5);
  const uv = 0.05 / (1 - 0.1 - 0.85);
  check(
    'T35.22 garchSqACF / unconditional variance / fourth moment',
    approx(sq[1], 0.17907, 1e-6) && approx(sq[5], 0.145853, 1e-6) && approx(uv, 1, 1e-12) &&
      garchFourthMomentFinite(0.1, 0.85) && !garchFourthMomentFinite(0.28207, 0.702774),
    `ρ1 ${sq[1]}, ρ5 ${sq[5]}, Var ${uv}`,
    'ρ1 0.17907, ρ5 0.145853, Var 1',
  );

  // T35.23 — Newey–West on fixture.ar1
  const hac = hacLongRunVariance(arma.ar1, 20);
  check('T35.23 hacLongRunVariance(fixture.ar1, 20)', approx(hac, 51.737313, 1e-5), hac, 51.737313);

  // T35.24 — information criteria for the sunspot AR(2)
  const ic = armaIC(-1307.318547, 4, sunspotValues.length);
  check(
    'T35.24 armaIC(−1307.318547, 4, 309)',
    sunspotValues.length === 309 && approx(ic.aic, 2622.637093, 1e-5) && approx(ic.bic, 2637.570458, 1e-5) && approx(ic.hqic, 2628.607481, 1e-5),
    `${ic.aic}, ${ic.bic}, ${ic.hqic}`,
    '2622.637093, 2637.570458, 2628.607481',
  );

  // S35.A — simulated AR(2) lag-1 autocorrelation near 2/3 (TS RNG)
  const sim = simulateARMA({ phi: [1, -0.5], theta: [], sigma2: 1 }, 20000, createSeededRng(3530));
  const rhoSim = sampleACF(sim, 1)[1];
  check('S35.A simulateARMA AR(2): ρ̂(1) ≈ 2/3', Math.abs(rhoSim - 2 / 3) < 0.02, rhoSim.toFixed(4), '2/3 ± 0.02');

  // S35.B — direct-DFT path vs a naive DFT
  {
    const x = arma.ar1.slice(0, 97);
    const n = x.length;
    const X = fft(x);
    let err = 0;
    for (let j = 0; j < n; j++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < n; t++) {
        re += x[t] * Math.cos((2 * Math.PI * j * t) / n);
        im -= x[t] * Math.sin((2 * Math.PI * j * t) / n);
      }
      err = Math.max(err, Math.abs(X.re[j] - re), Math.abs(X.im[j] - im));
    }
    check('S35.B fft direct path (n = 97) vs naive DFT', err < 1e-9, err.toExponential(1), '< 1e-9');
  }

  // S35.C — roots → AR → roots round trip
  {
    const r = [polar(1.3, 0.7), polar(1.3, -0.7), polar(2.1, 2.2), polar(2.1, -2.2)];
    const back = polyRoots(arPoly(rootsToAR(r)));
    const err = Math.max(...r.map((z) => Math.min(...back.map((w) => Math.hypot(w.re - z.re, w.im - z.im)))));
    check('S35.C polyRoots(arPoly(rootsToAR(r))) round trip', back.length === 4 && err < 1e-9, err.toExponential(1), '< 1e-9');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Topic 36 — State-Space Models & Kalman Smoothing (T36.1–T36.22, S36.A–S36.C)
// ═══════════════════════════════════════════════════════════════════════════
{
  console.log('\nTopic 36 — State-Space Models & Kalman Smoothing');
  const ssm: {
    tracking: { positions: number[]; truth: number[][]; dt: number; qa: number; r: number };
    gp: { t: number[]; y: number[]; s2: number; ell: number; noise: number };
    localLevel: { x: number[]; q: number; r: number };
  } = JSON.parse(readFileSync(new URL('./__fixtures__/ssm_fixture.json', import.meta.url), 'utf8'));
  const arma: { arma11: number[] } = JSON.parse(
    readFileSync(new URL('./__fixtures__/arma_fixture.json', import.meta.url), 'utf8'),
  );
  const Q_DK = 1469.1;
  const R_DK = 15099;
  const nileGap = nileValues.map((v, t) => ((t >= 20 && t < 40) || (t >= 60 && t < 80) ? Number.NaN : v));
  const f9 = (v: number[]) => v.map((u) => u.toFixed(9)).join(', ');

  // T36.1 — matrix block: Cholesky solve and log-determinant
  {
    const M1 = [[4, 1.2, -0.6], [1.2, 3, 0.4], [-0.6, 0.4, 2]];
    const L = cholesky(M1);
    const xs = choleskySolve(L, [[1], [-2], [0.5]]).map((r) => r[0]);
    const ld = logDetChol(L);
    check(
      'T36.1 choleskySolve(cholesky(M1), b), logDetChol',
      allApprox(xs, [0.651296218, -1.01359966, 0.648108797], 1e-9) && approx(ld, 2.935132651425, 1e-12),
      `[${f9(xs)}], ${ld.toFixed(12)}`,
      '[0.651296218, -1.01359966, 0.648108797], 2.935132651425',
    );
  }

  // T36.2 — discrete Lyapunov: AR(1) and the ARMA(1,1) Harvey P1
  {
    const l1 = solveDiscreteLyapunov([[0.9]], [[1]])[0][0];
    const P1 = armaToStateSpace({ phi: [0.7], theta: [0.4], sigma2: 1 }).P1;
    check(
      'T36.2 solveDiscreteLyapunov; armaToStateSpace(ARMA(1,1)).P1',
      approx(l1, 5.263157894737, 1e-12) && maxAbsDiff(P1, [[3.37254902, 0.4], [0.4, 0.16]]) < 1e-8,
      `${l1.toFixed(12)}; ${JSON.stringify(P1)}`,
      '5.263157894737; [[3.37254902, 0.4], [0.4, 0.16]]',
    );
  }

  // T36.3 — Thm 2: H F^h P1 Hᵀ = γ(h); H F^j g = ψ_j
  {
    const m = armaToStateSpace({ phi: [0.5, 0.2], theta: [0.4, -0.3], sigma2: 1 });
    const gss: number[] = [];
    for (let h = 0; h <= 7; h++) gss.push(matMul(matMul(m.H, matPow(m.F, h)), matMul(m.P1, transpose(m.H)))[0][0]);
    const psi: number[] = [];
    for (let j = 0; j <= 5; j++) psi.push(matVec(matMul(m.H, matPow(m.F, j)), m.g)[0]);
    const want = [2.209401709, 1.543376068, 0.913568376, 0.765459402, 0.565443376, 0.435813568, 0.330995459, 0.252660443];
    check(
      'T36.3 Harvey form reproduces armaACVF and psiWeights (ARMA(2,2))',
      allApprox(gss, want, 1e-9) &&
        allApprox(gss, armaACVF({ phi: [0.5, 0.2], theta: [0.4, -0.3], sigma2: 1 }, 7), 1e-12) &&
        allApprox(psi, psiWeights([0.5, 0.2], [0.4, -0.3], 6), 1e-12) &&
        allApprox(psi, [1, 0.9, 0.35, 0.355, 0.2475, 0.19475], 1e-12),
      `[${f9(gss)}]; [${f9(psi)}]`,
      `[${want.join(', ')}]; [1, 0.9, 0.35, 0.355, 0.2475, 0.19475]`,
    );
  }

  // T36.4 — Topic 34 Ex 15 regression: q = 0.1, r = 1
  {
    const ss = steadyStateGain([[1]], [[1]], [[0.1]], [[1]]);
    const ll = localLevelSteady(0.1, 1);
    check(
      'T36.4 steadyStateGain / localLevelSteady(0.1, 1) (Topic 34 Ex 15)',
      approx(ss.K[0][0], 0.270156211871, 1e-9) && approx(ss.Pminus[0][0], 0.370156211871, 1e-9) &&
        approx(ll.Pminus, 0.370156211872, 1e-12) && approx(ll.K, 0.270156211872, 1e-12) &&
        approx(ll.rate, 0.53267195507, 1e-12) && ll.rate === (1 - ll.K) * (1 - ll.K),
      `${ss.K[0][0].toFixed(12)}, ${ss.Pminus[0][0].toFixed(12)}; ${ll.Pminus.toFixed(12)}, ${ll.K.toFixed(12)}, ${ll.rate.toFixed(12)}`,
      '0.270156211871, 0.370156211871; 0.370156211872, 0.270156211872, 0.532671955070',
    );
  }

  // T36.5–T36.7 — Nile at Durbin–Koopman's parameters
  const kfDK = kalmanFilter(nileValues, localLevel(Q_DK, R_DK));
  const smDK = rtsSmoother(kfDK, localLevel(Q_DK, R_DK));
  check(
    'T36.5 kalmanFilter(nile, localLevel(1469.1, 15099)).loglik',
    approx(kfDK.loglik, -632.537695048, 1e-8),
    kfDK.loglik.toFixed(9),
    '-632.537695048 (statsmodels UC)',
  );
  {
    const idx = [0, 27, 99];
    const mf = idx.map((t) => kfDK.mFilt[t][0]);
    const pf = idx.map((t) => kfDK.PFilt[t][0][0]);
    const ms = idx.map((t) => smDK.mS[t][0]);
    const ps = idx.map((t) => smDK.PS[t][0][0]);
    check(
      'T36.6 filtered/smoothed moments at t = 0, 27, 99',
      allApprox(mf, [1103.340659384, 1133.124530842, 798.370292608], 1e-6) &&
        allApprox(pf, [14874.41126432, 4032.158204433, 4032.157941808], 1e-6) &&
        allApprox(ms, [1107.203898136, 999.584202914, 798.370292608], 1e-6) &&
        allApprox(ps, [4015.964936894, 2326.756957264, 4032.157941808], 1e-6),
      `mF [${f9(mf)}] PF [${f9(pf)}] mS [${f9(ms)}] PS [${f9(ps)}]`,
      'statsmodels UC filtered/smoothed state',
    );
  }
  check(
    'T36.7 lag-one smoothed covariance C[27]',
    approx(smDK.C[27][0][0], 1705.401136091, 1e-6) && approx(smDK.PS[27][0][0], 2326.756957264, 1e-6),
    smDK.C[27][0][0].toFixed(9),
    '1705.401136091 (dense conditioning)',
  );

  // T36.8 — Thm 4: Durbin–Koopman gaps
  {
    const llGap = kalmanFilter(nileGap, localLevel(Q_DK, R_DK)).loglik;
    const llBig = kalmanFilter(nileGap, { ...localLevel(Q_DK, R_DK), P1: [[1e7]], burn: 0 }).loglik;
    check(
      'T36.8 gapped Nile: diffuse ℓ; κ = 1e7, burn 0 (= 60-dim MVN)',
      approx(llGap, -380.578748152, 1e-8) && approx(llBig, -389.626977526, 1e-6),
      `${llGap.toFixed(9)}, ${llBig.toFixed(9)}`,
      '-380.578748152, -389.626977526',
    );
  }

  // T36.9 — Thm 3: Kalman innovations = Durbin–Levinson innovations
  {
    const x = arma.arma11.slice(0, 200);
    const p = { phi: [0.7], theta: [0.4], sigma2: 1 };
    const ki = kalmanInnovations(x, p);
    const g = armaACVF(p, 200);
    const n = x.length;
    // Durbin–Levinson rows φ_{t,1..t} for each t (the O(n²) predictor of Topic 35 Thm 3)
    const rows: number[][] = [[]];
    const v = [g[0]];
    let prev: number[] = [];
    for (let k = 1; k < n; k++) {
      let num = g[k];
      for (let j = 0; j < prev.length; j++) num -= prev[j] * g[k - 1 - j];
      const a = num / v[k - 1];
      const cur = prev.map((val, j) => val - a * prev[prev.length - 1 - j]);
      cur.push(a);
      v.push(v[k - 1] * (1 - a * a));
      rows.push(cur);
      prev = cur;
    }
    const dlV = durbinLevinson(g.slice(0, n)).v;
    let de = 0;
    let dS = 0;
    for (let t = 0; t < n; t++) {
      let pred = 0;
      for (let j = 0; j < t; j++) pred += rows[t][j] * x[t - 1 - j];
      de = Math.max(de, Math.abs(ki.e[t] - (x[t] - pred)));
      dS = Math.max(dS, Math.abs(ki.S[t] - dlV[t]));
    }
    check(
      'T36.9 Kalman e, S = Durbin–Levinson innovations, v (Thm 3)',
      de < 1e-10 && dS < 1e-10 && allApprox(ki.S.slice(0, 3), [3.37254902, 1.11255814, 1.016187291], 1e-8) &&
        approx(ki.S[0], (1 + 2 * 0.28 + 0.16) / 0.51, 1e-12),
      `max|Δe| ${de.toExponential(1)}, max|ΔS| ${dS.toExponential(1)}, S[0..2] [${f9(ki.S.slice(0, 3))}]`,
      '< 1e-10, < 1e-10, [3.37254902, 1.11255814, 1.016187291]',
    );
  }

  // T36.10 — exact ARMA log-likelihood on sunspots at the SARIMAX MLE
  {
    const l20 = armaExactLogLik(sunspotValues, { phi: [1.390659344, -0.688568029], theta: [], sigma2: 274.76110375 }, 49.65933445);
    const l21 = armaExactLogLik(
      sunspotValues,
      { phi: [1.470748417, -0.755124595], theta: [-0.153706339], sigma2: 270.878240447 },
      49.749150064,
    );
    check(
      'T36.10 armaExactLogLik(sunspots) at SARIMAX AR(2), ARMA(2,1)',
      approx(l20, -1307.318169071, 1e-6) && approx(l21, -1305.138595828, 1e-6),
      `${l20.toFixed(9)}, ${l21.toFixed(9)}`,
      '-1307.318169071, -1305.138595828',
    );
  }

  // T36.11 — exact MLE, sunspots AR(2)
  {
    const f = armaMLE(sunspotValues, 2, 0);
    check(
      'T36.11 armaMLE(sunspots, 2, 0)',
      approx(f.mu, 49.659396, 1e-4) && allApprox(f.phi, [1.390656, -0.688571], 1e-4) &&
        approx(f.sigma2, 274.7604, 1e-2) && approx(f.loglik, -1307.318169, 1e-6),
      `mu ${f.mu.toFixed(6)} phi [${f.phi.map((v) => v.toFixed(6))}] s2 ${f.sigma2.toFixed(4)} ll ${f.loglik.toFixed(6)}`,
      'mu 49.659396 phi [1.390656, -0.688571] s2 274.7604 ll -1307.318169',
    );
  }

  // T36.12 — exact MLE on the demeaned ARMA(1,1) fixture beats CSS and Whittle
  {
    const x = arma.arma11;
    const xbar = x.reduce((s, v) => s + v, 0) / x.length;
    const xd = x.map((v) => v - xbar);
    const f = armaMLE(xd, 1, 1, { estimateMean: false });
    const lCSS = armaProfileLogLik(xd, [0.638968], [0.402674]).loglik;
    const lWh = armaProfileLogLik(xd, [0.641025], [0.38991]).loglik;
    check(
      'T36.12 armaMLE(arma11 − x̄, 1, 1) > CSS, Whittle in exact ℓ',
      approx(f.phi[0], 0.637421, 1e-4) && approx(f.theta[0], 0.40419, 1e-4) && approx(f.sigma2, 1.000176, 1e-5) &&
        approx(f.loglik, -1419.605827, 1e-5) && f.loglik > lCSS && f.loglik > lWh &&
        approx(lCSS, -1419.607146, 1e-5) && approx(lWh, -1419.678996, 1e-5),
      `phi ${f.phi[0].toFixed(6)} theta ${f.theta[0].toFixed(6)} s2 ${f.sigma2.toFixed(6)} ll ${f.loglik.toFixed(6)}; CSS ${lCSS.toFixed(6)}, Whittle ${lWh.toFixed(6)}`,
      'phi 0.637421 theta 0.404190 s2 1.000176 ll -1419.605827; CSS -1419.607146, Whittle -1419.678996',
    );
  }

  // T36.13 — direct local-level MLE on the Nile
  {
    const f = localLevelMLE(nileValues);
    check(
      'T36.13 localLevelMLE(nile)',
      Math.abs(f.q / 1463.5474 - 1) < 1e-3 && Math.abs(f.r / 15108.3145 - 1) < 1e-3 && approx(f.loglik, -632.537685587, 1e-6) &&
        f.loglik >= -632.537760576,
      `q ${f.q.toFixed(4)} r ${f.r.toFixed(4)} ll ${f.loglik.toFixed(9)} (${f.iterations} it.)`,
      'q 1463.5474 r 15108.3145 ll -632.537685587',
    );
  }

  // T36.14 — one EM step (Thm 7 closed form)
  {
    const s = ssmEMStep(nileValues, localLevel(10000, 5000));
    check(
      'T36.14 ssmEMStep(nile, localLevel(10000, 5000))',
      approx(s.model.Q[0][0], 10501.609718822, 1e-6) && approx(s.model.R[0][0], 6033.823301433, 1e-6) &&
        approx(s.loglik, -640.124650741, 1e-8),
      `Q ${s.model.Q[0][0].toFixed(9)} R ${s.model.R[0][0].toFixed(9)} ll ${s.loglik.toFixed(9)}`,
      'Q 10501.609718822 R 6033.823301433 ll -640.124650741',
    );
  }

  // T36.15 — full EM run: 172 iterations, ascent at every step
  {
    const em = ssmEM(nileValues, localLevel(10000, 5000));
    let minInc = Infinity;
    for (let i = 1; i < em.logliks.length; i++) minInc = Math.min(minInc, em.logliks[i] - em.logliks[i - 1]);
    check(
      'T36.15 ssmEM(nile) from (10000, 5000)',
      Math.abs(em.iterations - 172) <= 1 && Math.abs(em.model.Q[0][0] / 1482.569272 - 1) < 1e-5 &&
        Math.abs(em.model.R[0][0] / 15079.48814 - 1) < 1e-5 && approx(em.loglik, -632.537794924, 1e-6) && minInc > 0,
      `${em.iterations} it., Q ${em.model.Q[0][0].toFixed(6)} R ${em.model.R[0][0].toFixed(5)} ll ${em.loglik.toFixed(9)}, min Δℓ ${minInc.toExponential(2)}`,
      '172 ± 1, Q 1482.569272 R 15079.48814 ll -632.537794924, min Δℓ > 0',
    );
  }

  // T36.16 — CO2: local linear trend + seasonal(12), 13-dim state, burn 13
  {
    const m = lltSeasonal(0.050844309968, 3.450003e-6, 1.0336517e-5, 0.024022091168);
    const ll = kalmanFilter(co2Values, m).loglik;
    check(
      'T36.16 kalmanFilter(co2, lltSeasonal(UC MLE)).loglik',
      approx(ll, -157.914578, 1e-5) && m.F.length === 13 && m.burn === 13,
      `${ll.toFixed(6)}, k ${m.F.length}, burn ${m.burn}`,
      '-157.914578, k 13, burn 13',
    );
  }

  // T36.17 — Matérn-3/2 GP as a two-state SSM vs the dense O(n³) posterior
  {
    const lam = Math.sqrt(3) / 1.5;
    const F1 = matern32StateSpace(1, 1, lam).F;
    const { t, y } = ssm.gp;
    const post = gpSSMPosterior(t, y, 1, lam, 0.09);
    const n = t.length;
    const kern = (a: number, b: number) => {
      const u = lam * Math.abs(a - b);
      return (1 + u) * Math.exp(-u);
    };
    const Kxx = t.map((a) => t.map((b) => kern(a, b)));
    const L = cholesky(Kxx.map((row, i) => row.map((v, j) => v + (i === j ? 0.09 : 0))));
    const alpha = choleskySolve(L, y.map((v) => [v])).map((r) => r[0]);
    const meanD = matVec(Kxx, alpha);
    const V = choleskySolve(L, Kxx);
    const varD = t.map((_, i) => Kxx[i][i] - Kxx[i].reduce((s, kv, j) => s + kv * V[j][i], 0));
    let dm = 0;
    let dv = 0;
    for (let i = 0; i < n; i++) {
      dm = Math.max(dm, Math.abs(post.mean[i] - meanD[i]));
      dv = Math.max(dv, Math.abs(post.variance[i] - varD[i]));
    }
    const pick = [0, 29, 59].map((i) => post.mean[i]);
    check(
      'T36.17 matern32StateSpace F; gpSSMPosterior = dense GP',
      maxAbsDiff(F1, [[0.679057966, 0.315151899], [-0.420202532, -0.048754168]]) < 1e-9 &&
        allApprox(pick, [0.519234279, -0.949827573, -0.413826005], 1e-8) && dm < 1e-9 && dv < 1e-9,
      `F ${JSON.stringify(F1.map((r) => r.map((v) => +v.toFixed(9))))}; mean[0,29,59] [${f9(pick)}]; Δmean ${dm.toExponential(1)}, Δvar ${dv.toExponential(1)}`,
      'F [[0.679057966, 0.315151899], [-0.420202532, -0.048754168]]; [0.519234279, -0.949827573, -0.413826005]',
    );
  }

  // T36.18 — forecasting = NaN padding = closed form P_T + h q
  {
    const model = localLevel(Q_DK, R_DK);
    const kf = kalmanFilter([...nileValues, ...new Array<number>(10).fill(Number.NaN)], model);
    const fc = forecast(kfDK, model, 10);
    const a = kf.PPred[100][0][0] + R_DK;
    const b = kf.PPred[109][0][0] + R_DK;
    check(
      'T36.18 NaN-padded filter forecast = closed form = forecast()',
      approx(a, 20600.257942, 1e-6) && approx(b, 33822.157942, 1e-6) && approx(kf.mPred[109][0], kfDK.mFilt[99][0], 1e-9) &&
        approx(fc.obsCov[0][0][0], a, 1e-6) && approx(fc.obsCov[9][0][0], b, 1e-6) && approx(fc.obsMean[9][0], kfDK.mFilt[99][0], 1e-9),
      `${a.toFixed(6)}, ${b.toFixed(6)}`,
      '20600.257942, 33822.157942',
    );
  }

  // T36.19 — Ex 2: θ = K∞ − 1, θ/(1+θ²) = −r/(q+2r)
  {
    const { K } = localLevelSteady(Q_DK, R_DK);
    const th = K - 1;
    const rho = -R_DK / (Q_DK + 2 * R_DK);
    check(
      'T36.19 localLevelSteady(DK): θ = K − 1, θ/(1+θ²) = −r/(q+2r)',
      approx(K, 0.267048012571, 1e-12) && approx(th, -0.732951987429, 1e-12) &&
        approx(th / (1 + th * th), rho, 1e-12) && approx(rho, -0.476804001629, 1e-12),
      `${K.toFixed(12)}, ${th.toFixed(12)}, ${(th / (1 + th * th)).toFixed(12)}`,
      '0.267048012571, -0.732951987429, -0.476804001629',
    );
  }

  // T36.20 — Thm 10: constant-velocity DARE
  {
    const cv = constantVelocity(1, 0.3, 0.4);
    const ss = steadyStateGain(cv.F, cv.H, cv.Q, cv.R);
    const K = ss.K.map((r) => r[0]);
    check(
      'T36.20 steadyStateGain(constant velocity) = DARE',
      allApprox(K, [0.731704571, 0.448577275], 1e-9) &&
        maxAbsDiff(ss.Pminus, [[1.090893831, 0.668781092], [0.668781092, 0.639350182]]) < 1e-9 &&
        Math.abs(ss.iterations - 22) <= 1,
      `K [${f9(K)}], P ${JSON.stringify(ss.Pminus.map((r) => r.map((v) => +v.toFixed(9))))}, ${ss.iterations} it.`,
      'K [0.731704571, 0.448577275], P [[1.090893831, 0.668781092], [0.668781092, 0.639350182]], 22 ± 1',
    );
  }

  // T36.21 — Joseph vs standard covariance update (Rem 4)
  {
    let P = KAPPA_DIFFUSE;
    let err = 0;
    nileValues.forEach((_, t) => {
      if (t > 0) P += Q_DK;
      const K = P / (P + R_DK);
      P = (1 - K) * P;
      err = Math.max(err, Math.abs(P - kfDK.PFilt[t][0][0]));
    });
    check('T36.21 Joseph form = (1 − K)P⁻ on the Nile', err < 1e-6, err.toExponential(1), '< 1e-6 (oracle 4.5e-11)');
  }

  // T36.22 — tracking fixture
  {
    const model = constantVelocity(1, 0.3, 0.4);
    const kf = kalmanFilter(ssm.tracking.positions, model);
    const ms = rtsSmoother(kf, model).mS[59];
    check(
      'T36.22 kalmanFilter(tracking).loglik; rtsSmoother mS[59]',
      approx(kf.loglik, -97.184941392, 1e-8) && allApprox(ms, [69.59029242, 3.41918993], 1e-7),
      `${kf.loglik.toFixed(9)}; [${f9(ms)}]`,
      '-97.184941392; [69.59029242, 3.41918993]',
    );
  }

  // S36.A — FFBS draws centre on the smoother
  {
    const mle = localLevelMLE(nileValues);
    const model = localLevel(mle.q, mle.r);
    const kf = kalmanFilter(nileValues, model);
    const sm = rtsSmoother(kf, model);
    const rng = createSeededRng(3606);
    const N = 2000;
    const sum = new Array<number>(100).fill(0);
    for (let i = 0; i < N; i++) ffbs(kf, model, rng).forEach((z, t) => (sum[t] += z[0]));
    const dev = Math.max(...sum.map((s, t) => Math.abs(s / N - sm.mS[t][0]) / Math.sqrt(sm.PS[t][0][0])));
    check('S36.A 2000 FFBS draws: max |mean − mS|/sd', dev < 0.15, dev.toFixed(3), '< 0.15 (oracle 0.053)');
  }

  // S36.B — EM recovers (q, r) on the simulated local level
  {
    const em = ssmEM(ssm.localLevel.x, localLevel(1, 1));
    const q = em.model.Q[0][0];
    const r = em.model.R[0][0];
    check(
      'S36.B ssmEM(localLevel fixture, (1, 1))',
      Math.abs(q - 0.1) < 0.06 && Math.abs(r - 1) < 0.2 && approx(q, 0.121901, 1e-4) && approx(r, 0.995751, 1e-4) &&
        Math.abs(em.iterations - 87) <= 1,
      `q ${q.toFixed(6)} r ${r.toFixed(6)} (${em.iterations} it.)`,
      'q 0.121901 r 0.995751 (87 ± 1 it.)',
    );
  }

  // S36.C — simulateSSM moments of Δx
  {
    const { x } = simulateSSM({ ...localLevel(0.1, 1), P1: [[1]], burn: 0 }, 20000, createSeededRng(3630));
    const dx = x.slice(1).map((v, t) => v[0] - x[t][0]);
    const m = dx.reduce((s, v) => s + v, 0) / dx.length;
    const v0 = dx.reduce((s, v) => s + (v - m) * (v - m), 0) / dx.length;
    let c1 = 0;
    for (let t = 1; t < dx.length; t++) c1 += (dx[t] - m) * (dx[t - 1] - m);
    const rho1 = c1 / dx.length / v0;
    check(
      'S36.C simulateSSM: var(Δx) ≈ q + 2r, ρ(1) ≈ −r/(q+2r)',
      Math.abs(v0 - 2.1) < 0.05 && Math.abs(rho1 + 0.47619) < 0.02,
      `var ${v0.toFixed(4)}, ρ(1) ${rho1.toFixed(4)}`,
      'var 2.1 ± 0.05, ρ(1) −0.476190 ± 0.02',
    );
  }
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
