/**
 * timeseries.test.ts — Console-log verification for the Topic-34 test pins
 * T34.1–T34.20 and S34.A–S34.C specified in the handoff brief §6.2.
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
} from './timeseries';
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
