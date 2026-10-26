/**
 * Topic 34 — Hidden Markov Models (Track 9 opener).
 *
 * New shared module for time-series and state-space machinery. Future Track 9
 * topics extend this file rather than adding sibling modules — the
 * extend-don't-create convention of `testing.ts` (Tracks 4–6), `bayes.ts`
 * (Track 7), and `nonparametric.ts` (Track 8).
 *
 * Scope (see docs/formalstatistics-hidden-markov-models-handoff-brief.md §6.1):
 *
 *   • Finite Markov chains: stationary distribution (linear solve), matrix
 *     powers, simulation, and sojourn-length extraction.
 *   • HMM core: simulation, emission likelihoods, the scaled forward pass
 *     (= the filter, Thm 6), the scaled backward pass, forward–backward
 *     smoothing (Thm 7), Viterbi decoding in log space (Thm 8), pointwise
 *     posterior decoding, and path log-probabilities (−∞ allowed).
 *   • Learning: one Baum–Welch step (Thm 10) and the full EM loop (Thm 9),
 *     deterministic k-means and random initializations, label-switching
 *     fix, free-parameter counts and BIC.
 *   • Brute-force reference implementations (K^T enumeration) used by the
 *     test harness as the independent check.
 *
 * Every function in sections 1–4 mirrors the Python toolkit in notebook Cell 3
 * (notebooks/hidden-markov-models/34_hidden_markov_models.ipynb) step for
 * step, so the T34 pins reproduce the notebook to the stated tolerances.
 * States are 0-indexed here and 1-indexed in the prose.
 *
 * Topic 35 — Stationary Time Series & ARMA extends the module (sections 5–10;
 * see docs/formalstatistics-stationary-time-series-and-arma-handoff-brief.md
 * §6.1, mirroring Cell 3 of
 * notebooks/stationary-time-series-and-arma/35_stationary_time_series.ipynb):
 *
 *   • Polynomials and roots: AR/MA polynomials, Durand–Kerner roots,
 *     roots → coefficients, causality and invertibility checks (Thms 5–6).
 *   • ψ and π weights, the exact ARMA autocovariance (Brockwell–Davis §3.3
 *     method 2), ACF, Durbin–Levinson (Thm 3) and the PACF (Thm 7), sample
 *     ACVF/ACF and Bartlett standard errors (Thm 9).
 *   • Spectral: ARMA spectral density (Thm 11), long-run variance (Thm 2),
 *     the Newey–West HAC estimator, FFT, periodogram (Thm 12) and Daniell /
 *     modified-Daniell / Bartlett-lag-window smoothing (Thm 13).
 *   • Estimation: Yule–Walker, Nelder–Mead, conditional least squares,
 *     Whittle (Thm 14), information criteria.
 *   • Simulation of ARMA and GARCH(1,1) paths.
 *   • GARCH(1,1): conditional variances, Gaussian QMLE with an
 *     arch-compatible backcast, squared-return ACF (Thm 15, Ex 18).
 *
 * Imports `SeededRng` from bayes.ts; bayes.ts must not import this module.
 */

import { type SeededRng } from './bayes';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export type Matrix = number[][];

/** Categorical emissions: probs[k][v] = P(X_t = v | Z_t = k), a K × V matrix. */
export interface CategoricalEmission {
  kind: 'categorical';
  probs: Matrix;
}

/** Gaussian emissions: X_t | Z_t = k ~ N(means[k], sds[k]²). */
export interface GaussianEmission {
  kind: 'gaussian';
  means: number[];
  sds: number[];
}

export type Emission = CategoricalEmission | GaussianEmission;

/** θ = (ν, A, {b_k}): initial distribution, transition matrix, emissions. */
export interface HMMParams {
  initial: number[];
  transition: Matrix;
  emission: Emission;
}

/** Scaled forward pass: alphaHat[t][k] = P(Z_t = k | x_{1:t}); c[t] = p(x_t | x_{1:t-1}). */
export interface ForwardResult {
  alphaHat: Matrix;
  c: number[];
  logLik: number;
}

export interface ForwardBackwardResult extends ForwardResult {
  betaHat: Matrix;
  /** gamma[t][k] = P(Z_t = k | x_{1:T}). */
  gamma: Matrix;
  /** xiSum[j][k] = Σ_t P(Z_t = j, Z_{t+1} = k | x_{1:T}). */
  xiSum: Matrix;
  /** Emission likelihoods b_k(x_t), T × K. */
  B: Matrix;
}

export interface ViterbiResult {
  path: number[];
  /** log max_z p(z_{1:T}, x_{1:T}). */
  logProb: number;
  logDelta: Matrix;
}

export interface BaumWelchResult {
  params: HMMParams;
  /** trace[s] = ℓ(θ^(s)); the last entry is the log-likelihood of `params`. */
  logLikTrace: number[];
  iterations: number;
  converged: boolean;
}

const SQRT_2PI = Math.sqrt(2 * Math.PI);

// ═══════════════════════════════════════════════════════════════════════════
// 1. Finite Markov chains
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Stationary distribution π with πA = π, Σπ = 1. Solves (A − I)ᵀπ = 0 with the
 * last equation replaced by the normalization, by Gaussian elimination with
 * partial pivoting. Throws if the system is singular (reducible chain).
 */
export function stationaryDistribution(A: Matrix): number[] {
  const K = A.length;
  // Augmented system M | rhs, M = (A − I)ᵀ with last row = 1.
  const M: number[][] = [];
  for (let i = 0; i < K; i++) {
    const row: number[] = [];
    for (let j = 0; j < K; j++) row.push(i === K - 1 ? 1 : A[j][i] - (i === j ? 1 : 0));
    row.push(i === K - 1 ? 1 : 0);
    M.push(row);
  }
  for (let col = 0; col < K; col++) {
    let piv = col;
    for (let r = col + 1; r < K; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-14) throw new Error('stationaryDistribution: singular system');
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < K; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= K; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => row[K] / row[i]);
}

function matMul(X: Matrix, Y: Matrix): Matrix {
  const n = X.length;
  const m = Y[0].length;
  const p = Y.length;
  const out: Matrix = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(m).fill(0);
    for (let l = 0; l < p; l++) {
      const x = X[i][l];
      for (let j = 0; j < m; j++) row[j] += x * Y[l][j];
    }
    out.push(row);
  }
  return out;
}

/** A^n by repeated squaring (n ≥ 0). */
export function matPow(A: Matrix, n: number): Matrix {
  const K = A.length;
  let result: Matrix = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
  let base = A.map((r) => r.slice());
  let e = n;
  while (e > 0) {
    if (e & 1) result = matMul(result, base);
    base = matMul(base, base);
    e >>= 1;
  }
  return K ? result : [];
}

function sampleCategorical(p: number[], u: number): number {
  let cum = 0;
  for (let k = 0; k < p.length; k++) {
    cum += p[k];
    if (u < cum) return k;
  }
  return p.length - 1;
}

/** Simulate Z_1..Z_T from (ν, A). */
export function sampleMarkovChain(nu: number[], A: Matrix, T: number, rng: SeededRng): number[] {
  const z = new Array<number>(T);
  z[0] = sampleCategorical(nu, rng.random());
  for (let t = 1; t < T; t++) z[t] = sampleCategorical(A[z[t - 1]], rng.random());
  return z;
}

/**
 * Completed sojourn lengths per state (Thm 3). The final run is right-censored
 * and excluded; the first run counts, since Thm 3 treats s = 1 as an entry.
 */
export function sojournLengths(states: number[], K: number): number[][] {
  const runs: number[][] = Array.from({ length: K }, () => []);
  let start = 0;
  for (let t = 1; t < states.length; t++) {
    if (states[t] !== states[t - 1]) {
      runs[states[t - 1]].push(t - start);
      start = t;
    }
  }
  return runs;
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. HMM core
// ═══════════════════════════════════════════════════════════════════════════

/** Simulate (Z_{1:T}, X_{1:T}). Categorical observations are symbol indices. */
export function sampleHMM(p: HMMParams, T: number, rng: SeededRng): { states: number[]; obs: number[] } {
  const states = sampleMarkovChain(p.initial, p.transition, T, rng);
  const em = p.emission;
  const obs =
    em.kind === 'categorical'
      ? states.map((k) => sampleCategorical(em.probs[k], rng.random()))
      : states.map((k) => em.means[k] + em.sds[k] * rng.normal());
  return { states, obs };
}

/** B[t][k] = b_k(x_t), a T × K matrix. */
export function emissionLikelihoods(p: HMMParams, obs: number[]): Matrix {
  const em = p.emission;
  if (em.kind === 'categorical') return obs.map((x) => em.probs.map((row) => row[x]));
  return obs.map((x) =>
    em.means.map((m, k) => {
      const s = em.sds[k];
      const z = (x - m) / s;
      return Math.exp(-0.5 * z * z) / (s * SQRT_2PI);
    }),
  );
}

/** Scaled forward recursion (Thm 5 + Thm 6). Pass B to reuse emission likelihoods. */
export function forwardScaled(p: HMMParams, obs: number[], B?: Matrix): ForwardResult {
  const Bm = B ?? emissionLikelihoods(p, obs);
  const A = p.transition;
  const T = Bm.length;
  const K = p.initial.length;
  const alphaHat: Matrix = new Array(T);
  const c = new Array<number>(T);
  let a = p.initial.map((v, k) => v * Bm[0][k]);
  c[0] = a.reduce((s, v) => s + v, 0);
  alphaHat[0] = a.map((v) => v / c[0]);
  for (let t = 1; t < T; t++) {
    const prev = alphaHat[t - 1];
    a = new Array<number>(K);
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let j = 0; j < K; j++) s += prev[j] * A[j][k];
      a[k] = Bm[t][k] * s;
    }
    c[t] = a.reduce((s, v) => s + v, 0);
    alphaHat[t] = a.map((v) => v / c[t]);
  }
  const logLik = c.reduce((s, v) => s + Math.log(v), 0);
  return { alphaHat, c, logLik };
}

/** Scaled backward recursion: betaHat_t(j) = Σ_k a_jk b_k(x_{t+1}) betaHat_{t+1}(k) / c_{t+1}. */
export function backwardScaled(p: HMMParams, B: Matrix, c: number[]): Matrix {
  const A = p.transition;
  const T = B.length;
  const K = A.length;
  const betaHat: Matrix = new Array(T);
  betaHat[T - 1] = new Array<number>(K).fill(1);
  for (let t = T - 2; t >= 0; t--) {
    const next = betaHat[t + 1];
    const row = new Array<number>(K);
    for (let j = 0; j < K; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += A[j][k] * B[t + 1][k] * next[k];
      row[j] = s / c[t + 1];
    }
    betaHat[t] = row;
  }
  return betaHat;
}

/** Forward–backward smoothing (Thm 7, scaled form). */
export function forwardBackward(p: HMMParams, obs: number[]): ForwardBackwardResult {
  const B = emissionLikelihoods(p, obs);
  const { alphaHat, c, logLik } = forwardScaled(p, obs, B);
  const betaHat = backwardScaled(p, B, c);
  const T = B.length;
  const K = p.initial.length;
  const gamma = alphaHat.map((row, t) => row.map((v, k) => v * betaHat[t][k]));
  const xiSum: Matrix = Array.from({ length: K }, () => new Array<number>(K).fill(0));
  for (let t = 0; t < T - 1; t++) {
    for (let j = 0; j < K; j++) {
      const aj = alphaHat[t][j];
      for (let k = 0; k < K; k++) {
        xiSum[j][k] += aj * p.transition[j][k] * B[t + 1][k] * betaHat[t + 1][k] / c[t + 1];
      }
    }
  }
  return { alphaHat, c, logLik, betaHat, gamma, xiSum, B };
}

export function hmmLogLikelihood(p: HMMParams, obs: number[]): number {
  return forwardScaled(p, obs).logLik;
}

/** Viterbi max-product decoding in log space (Thm 8). Ties go to the lower index. */
export function viterbi(p: HMMParams, obs: number[]): ViterbiResult {
  const logB = emissionLikelihoods(p, obs).map((r) => r.map(Math.log));
  const logA = p.transition.map((r) => r.map(Math.log));
  const T = logB.length;
  const K = p.initial.length;
  const logDelta: Matrix = new Array(T);
  const psi: number[][] = new Array(T);
  logDelta[0] = p.initial.map((v, k) => Math.log(v) + logB[0][k]);
  psi[0] = new Array<number>(K).fill(0);
  for (let t = 1; t < T; t++) {
    const row = new Array<number>(K);
    const back = new Array<number>(K);
    for (let k = 0; k < K; k++) {
      let best = -Infinity;
      let bestJ = 0;
      for (let j = 0; j < K; j++) {
        const s = logDelta[t - 1][j] + logA[j][k];
        if (s > best) {
          best = s;
          bestJ = j;
        }
      }
      back[k] = bestJ;
      row[k] = best + logB[t][k];
    }
    logDelta[t] = row;
    psi[t] = back;
  }
  const path = new Array<number>(T);
  let last = 0;
  for (let k = 1; k < K; k++) if (logDelta[T - 1][k] > logDelta[T - 1][last]) last = k;
  path[T - 1] = last;
  for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];
  return { path, logProb: logDelta[T - 1][last], logDelta };
}

/** Pointwise posterior decoding ẑ_t = argmax_k γ_t(k). Ties go to the lower index. */
export function posteriorDecode(gamma: Matrix): number[] {
  return gamma.map((row) => {
    let best = 0;
    for (let k = 1; k < row.length; k++) if (row[k] > row[best]) best = k;
    return best;
  });
}

/** log p(z_{1:T}, x_{1:T}); −Infinity for a path that uses a zero-probability edge. */
export function pathLogProb(p: HMMParams, obs: number[], path: number[]): number {
  const B = emissionLikelihoods(p, obs);
  let lp = Math.log(p.initial[path[0]]) + Math.log(B[0][path[0]]);
  for (let t = 1; t < path.length; t++) {
    lp += Math.log(p.transition[path[t - 1]][path[t]]) + Math.log(B[t][path[t]]);
  }
  return lp;
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Learning: Baum–Welch
// ═══════════════════════════════════════════════════════════════════════════

function populationVariance(x: number[]): number {
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  return x.reduce((s, v) => s + (v - m) * (v - m), 0) / x.length;
}

/**
 * One EM step (Thm 10). Returns the updated parameters and the log-likelihood
 * of the INPUT parameters (computed in the E-step), so a trace needs no extra
 * forward pass. Gaussian variances are floored at varianceFloor × Var(x).
 */
export function baumWelchStep(
  p: HMMParams,
  obs: number[],
  opts: { varianceFloor?: number } = {},
): { params: HMMParams; logLik: number } {
  const floorFactor = opts.varianceFloor ?? 1e-6;
  const fb = forwardBackward(p, obs);
  const { gamma, xiSum } = fb;
  const T = obs.length;
  const K = p.initial.length;
  const gammaSumHead = new Array<number>(K).fill(0); // Σ_{t<T} γ_t
  const gammaSum = new Array<number>(K).fill(0);     // Σ_t γ_t
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < K; k++) {
      gammaSum[k] += gamma[t][k];
      if (t < T - 1) gammaSumHead[k] += gamma[t][k];
    }
  }
  const transition = xiSum.map((row, j) => row.map((v) => v / gammaSumHead[j]));
  const em = p.emission;
  let emission: Emission;
  if (em.kind === 'categorical') {
    const V = em.probs[0].length;
    const probs: Matrix = Array.from({ length: K }, () => new Array<number>(V).fill(0));
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) probs[k][obs[t]] += gamma[t][k];
    emission = { kind: 'categorical', probs: probs.map((row, k) => row.map((v) => v / gammaSum[k])) };
  } else {
    const means = new Array<number>(K).fill(0);
    for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) means[k] += gamma[t][k] * obs[t];
    for (let k = 0; k < K; k++) means[k] /= gammaSum[k];
    const vars = new Array<number>(K).fill(0);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < K; k++) {
        const d = obs[t] - means[k];
        vars[k] += gamma[t][k] * d * d;
      }
    }
    const floor = floorFactor * populationVariance(obs);
    const sds = vars.map((v, k) => Math.sqrt(Math.max(v / gammaSum[k], floor)));
    emission = { kind: 'gaussian', means, sds };
  }
  return { params: { initial: gamma[0].slice(), transition, emission }, logLik: fb.logLik };
}

/**
 * Baum–Welch (EM) to convergence. trace[s] is ℓ of the parameters entering
 * step s; stops when |ℓ_s − ℓ_{s−1}| < tol after taking the step, then appends
 * ℓ of the returned parameters, so iterations = trace.length − 1.
 */
export function baumWelch(
  init: HMMParams,
  obs: number[],
  opts: { maxIter?: number; tol?: number; varianceFloor?: number } = {},
): BaumWelchResult {
  const maxIter = opts.maxIter ?? 500;
  const tol = opts.tol ?? 1e-8;
  let p = init;
  const trace: number[] = [];
  let converged = false;
  for (let it = 0; it < maxIter; it++) {
    const step = baumWelchStep(p, obs, { varianceFloor: opts.varianceFloor });
    trace.push(step.logLik);
    p = step.params;
    const n = trace.length;
    if (n >= 2 && Math.abs(trace[n - 1] - trace[n - 2]) < tol) {
      converged = true;
      break;
    }
  }
  trace.push(hmmLogLikelihood(p, obs));
  return { params: p, logLikTrace: trace, iterations: trace.length - 1, converged };
}

function quantileLinear(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

function nearestCenter(x: number, centers: number[]): number {
  let best = 0;
  for (let k = 1; k < centers.length; k++) {
    if (Math.abs(x - centers[k]) < Math.abs(x - centers[best])) best = k;
  }
  return best;
}

/**
 * Deterministic 1-D k-means initialization for Gaussian emissions. Centers start
 * at linear-interpolated quantiles (k + 0.5)/K, then at most `nIter` Lloyd
 * iterations (stopping when the centers repeat exactly). States sorted by mean;
 * ν uniform; A = 0.9 on the diagonal, 0.1/(K − 1) off it.
 */
export function initKMeans(obs: number[], K: number, nIter = 50): HMMParams {
  const sorted = obs.slice().sort((a, b) => a - b);
  let centers = Array.from({ length: K }, (_, k) => quantileLinear(sorted, (k + 0.5) / K));
  const sdAll = Math.sqrt(populationVariance(obs));
  const assign = (cs: number[]) => obs.map((x) => nearestCenter(x, cs));
  for (let it = 0; it < nIter; it++) {
    const lab = assign(centers);
    const sums = new Array<number>(K).fill(0);
    const counts = new Array<number>(K).fill(0);
    lab.forEach((k, i) => {
      sums[k] += obs[i];
      counts[k]++;
    });
    const next = centers.map((c, k) => (counts[k] > 0 ? sums[k] / counts[k] : c));
    if (next.every((v, k) => v === centers[k])) break;
    centers = next;
  }
  const lab = assign(centers);
  const sds = centers.map((_, k) => {
    const xs = obs.filter((__, i) => lab[i] === k);
    const sd = xs.length > 1 ? Math.sqrt(populationVariance(xs)) : sdAll;
    return Math.max(sd, 1e-3 * sdAll);
  });
  const order = centers.map((_, k) => k).sort((a, b) => centers[a] - centers[b] || a - b);
  const transition: Matrix =
    K === 1
      ? [[1]]
      : Array.from({ length: K }, (_, i) =>
          Array.from({ length: K }, (__, j) => (i === j ? 0.9 : 0.1 / (K - 1))),
        );
  return {
    initial: new Array<number>(K).fill(1 / K),
    transition,
    emission: { kind: 'gaussian', means: order.map((k) => centers[k]), sds: order.map((k) => sds[k]) },
  };
}

function dirichlet1(rng: SeededRng, K: number): number[] {
  const e = Array.from({ length: K }, () => -Math.log(1 - rng.random()));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Random initialization: Dirichlet(1) rows for ν and A; Gaussian means drawn as
 * data values at random indices, SDs = sample SD × (0.5 + U); categorical rows
 * Dirichlet(1) over V symbols.
 */
export function initRandom(
  obs: number[],
  K: number,
  rng: SeededRng,
  kind: Emission['kind'] = 'gaussian',
  V?: number,
): HMMParams {
  const transition = Array.from({ length: K }, () => dirichlet1(rng, K));
  const initial = dirichlet1(rng, K);
  if (kind === 'categorical') {
    if (!V) throw new Error('initRandom: categorical emissions need V');
    return { initial, transition, emission: { kind, probs: Array.from({ length: K }, () => dirichlet1(rng, V)) } };
  }
  const sd = Math.sqrt(populationVariance(obs));
  const means = Array.from({ length: K }, () => obs[Math.floor(rng.random() * obs.length)]);
  const sds = Array.from({ length: K }, () => sd * (0.5 + rng.random()));
  return { initial, transition, emission: { kind: 'gaussian', means, sds } };
}

/** Permute Gaussian states into ascending-mean order (label-switching fix, Rem 13). */
export function relabelByMean(p: HMMParams): HMMParams {
  const em = p.emission;
  if (em.kind !== 'gaussian') return p;
  const o = em.means.map((_, k) => k).sort((a, b) => em.means[a] - em.means[b] || a - b);
  return {
    initial: o.map((k) => p.initial[k]),
    transition: o.map((i) => o.map((j) => p.transition[i][j])),
    emission: { kind: 'gaussian', means: o.map((k) => em.means[k]), sds: o.map((k) => em.sds[k]) },
  };
}

/** Free parameters: (K − 1) + K(K − 1) + emissions (2K Gaussian, K(V − 1) categorical). */
export function hmmFreeParams(K: number, kind: Emission['kind'] = 'gaussian', V?: number): number {
  const emis = kind === 'gaussian' ? 2 * K : K * ((V ?? 0) - 1);
  return K - 1 + K * (K - 1) + emis;
}

/** BIC = −2ℓ̂ + d_K log T. */
export function hmmBIC(
  logLik: number,
  K: number,
  T: number,
  kind: Emission['kind'] = 'gaussian',
  V?: number,
): number {
  return -2 * logLik + hmmFreeParams(K, kind, V) * Math.log(T);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Brute-force reference implementations (test harness; T ≤ 12)
// ═══════════════════════════════════════════════════════════════════════════

function allPaths(K: number, T: number): number[][] {
  const out: number[][] = [];
  const cur = new Array<number>(T).fill(0);
  const rec = (t: number) => {
    if (t === T) {
      out.push(cur.slice());
      return;
    }
    for (let k = 0; k < K; k++) {
      cur[t] = k;
      rec(t + 1);
    }
  };
  rec(0);
  return out;
}

/** log p(x_{1:T}) by summing all K^T paths. */
export function bruteForceLogLik(p: HMMParams, obs: number[]): number {
  const lps = allPaths(p.initial.length, obs.length).map((z) => pathLogProb(p, obs, z));
  const m = Math.max(...lps);
  return m + Math.log(lps.reduce((s, v) => s + Math.exp(v - m), 0));
}

/** argmax over all K^T paths (first maximizer in lexicographic order). */
export function bruteForceMAP(p: HMMParams, obs: number[]): number[] {
  let best: number[] = [];
  let bestLp = -Infinity;
  for (const z of allPaths(p.initial.length, obs.length)) {
    const lp = pathLogProb(p, obs, z);
    if (lp > bestLp) {
      bestLp = lp;
      best = z;
    }
  }
  return best;
}

/** Smoothing marginals P(Z_t = k | x) by enumeration. */
export function bruteForceMarginals(p: HMMParams, obs: number[]): Matrix {
  const K = p.initial.length;
  const paths = allPaths(K, obs.length);
  const lps = paths.map((z) => pathLogProb(p, obs, z));
  const m = Math.max(...lps);
  const w = lps.map((v) => Math.exp(v - m));
  const tot = w.reduce((a, b) => a + b, 0);
  const out: Matrix = obs.map(() => new Array<number>(K).fill(0));
  paths.forEach((z, i) => z.forEach((k, t) => (out[t][k] += w[i] / tot)));
  return out;
}

// ═══ Topic 35 — Stationary Time Series & ARMA ═══

// ═══════════════════════════════════════════════════════════════════════════
// 5. Types and complex arithmetic
// ═══════════════════════════════════════════════════════════════════════════

export interface Complex {
  re: number;
  im: number;
}

/** φ(z) = 1 − Σ φ_k z^k, θ(z) = 1 + Σ θ_k z^k, ε ~ WN(0, σ²). */
export interface ARMAParams {
  phi: number[];
  theta: number[];
  sigma2: number;
}

/** pacf = α(1..n); phi = the last row φ_{n,1..n}; v = v_0..v_n. */
export interface DurbinLevinsonResult {
  pacf: number[];
  phi: number[];
  v: number[];
}

export interface SpectrumEstimate {
  omega: number[];
  f: number[];
}

export type SmootherKind = 'daniell' | 'modifiedDaniell' | 'bartlettLag';

export interface GarchParams {
  omega: number;
  alpha: number;
  beta: number;
}

export interface GarchFit extends GarchParams {
  loglik: number;
  backcast: number;
  iterations: number;
}

export interface NelderMeadResult {
  x: number[];
  fun: number;
  iterations: number;
}

const cAdd = (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im });
const cSub = (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im });
const cMul = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});
const cDiv = (a: Complex, b: Complex): Complex => {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};
const cAbs = (a: Complex): number => Math.hypot(a.re, a.im);

/** Python's round(): halves go to the even neighbour. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Polynomials and roots
// ═══════════════════════════════════════════════════════════════════════════

/** Coefficients of φ(z) = 1 − φ1 z − … − φp z^p in increasing powers. */
export function arPoly(phi: number[]): number[] {
  return [1, ...phi.map((v) => -v)];
}

/** Coefficients of θ(z) = 1 + θ1 z + … + θq z^q in increasing powers. */
export function maPoly(theta: number[]): number[] {
  return [1, ...theta];
}

/**
 * Roots of Σ c_k z^k (increasing powers) by Durand–Kerner with a Gauss–Seidel
 * sweep. Trailing zero coefficients are trimmed and the polynomial made monic;
 * starting points z_k = (0.4 + 0.9i)^k; stops when the largest step < tol.
 * Imaginary parts below 1e-12 are zeroed; sorted by (|z|, Im z).
 */
export function polyRoots(c: number[], tol = 1e-14, maxIter = 1000): Complex[] {
  let coef = c.slice();
  while (coef.length > 1 && coef[coef.length - 1] === 0) coef = coef.slice(0, -1);
  const n = coef.length - 1;
  if (n < 1) return [];
  const lead = coef[n];
  const a = coef.map((v) => v / lead);
  const evalP = (z: Complex): Complex => {
    let v: Complex = { re: 0, im: 0 };
    for (let k = n; k >= 0; k--) v = cAdd(cMul(v, z), { re: a[k], im: 0 });
    return v;
  };
  const seed: Complex = { re: 0.4, im: 0.9 };
  const z: Complex[] = [];
  let pw: Complex = { re: 1, im: 0 };
  for (let k = 0; k < n; k++) {
    z.push(pw);
    pw = cMul(pw, seed);
  }
  for (let it = 0; it < maxIter; it++) {
    let delta = 0;
    for (let i = 0; i < n; i++) {
      let den: Complex = { re: 1, im: 0 };
      for (let j = 0; j < n; j++) if (j !== i) den = cMul(den, cSub(z[i], z[j]));
      const step = cDiv(evalP(z[i]), den);
      z[i] = cSub(z[i], step);
      delta = Math.max(delta, cAbs(step));
    }
    if (delta < tol) break;
  }
  return z
    .map((r) => (Math.abs(r.im) < 1e-12 ? { re: r.re, im: 0 } : r))
    .sort((u, v) => cAbs(u) - cAbs(v) || u.im - v.im);
}

/** Increasing-power coefficients of Π_k (1 − z/r_k); real when roots come in conjugate pairs. */
export function rootsToPoly(roots: Complex[]): number[] {
  let c: Complex[] = [{ re: 1, im: 0 }];
  for (const r of roots) {
    const b = cDiv({ re: -1, im: 0 }, r);
    const next: Complex[] = Array.from({ length: c.length + 1 }, () => ({ re: 0, im: 0 }));
    c.forEach((v, k) => {
      next[k] = cAdd(next[k], v);
      next[k + 1] = cAdd(next[k + 1], cMul(v, b));
    });
    c = next;
  }
  return c.map((v) => v.re);
}

/** AR coefficients (φ1..φp) whose φ(z) has the given roots. */
export function rootsToAR(roots: Complex[]): number[] {
  return rootsToPoly(roots).slice(1).map((v) => -v);
}

/** MA coefficients (θ1..θq) whose θ(z) has the given roots. */
export function rootsToMA(roots: Complex[]): number[] {
  return rootsToPoly(roots).slice(1);
}

/** Thm 5: every root of φ(z) strictly outside the unit circle. p = 0 is causal. */
export function isCausal(phi: number[], tol = 1e-10): boolean {
  return polyRoots(arPoly(phi)).every((r) => cAbs(r) > 1 + tol);
}

/** Thm 6: every root of θ(z) strictly outside the unit circle. */
export function isInvertible(theta: number[], tol = 1e-10): boolean {
  return polyRoots(maPoly(theta)).every((r) => cAbs(r) > 1 + tol);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Weights, autocovariance, PACF
// ═══════════════════════════════════════════════════════════════════════════

/** ψ_0..ψ_{n−1} of the causal MA(∞): ψ_j = θ_j + Σ_{k ≤ min(j, p)} φ_k ψ_{j−k}. */
export function psiWeights(phi: number[], theta: number[], n: number): number[] {
  const psi = new Array<number>(n).fill(0);
  for (let j = 0; j < n; j++) {
    let v = j === 0 ? 1 : j - 1 < theta.length ? theta[j - 1] : 0;
    for (let k = 1; k <= Math.min(j, phi.length); k++) v += phi[k - 1] * psi[j - k];
    psi[j] = v;
  }
  return psi;
}

/** π_0..π_{n−1} of the AR(∞) ε_t = Σ π_j X_{t−j}: θ(z)π(z) = φ(z). */
export function piWeights(phi: number[], theta: number[], n: number): number[] {
  const a = arPoly(phi);
  const pi = new Array<number>(n).fill(0);
  for (let j = 0; j < n; j++) {
    let v = j < a.length ? a[j] : 0;
    for (let k = 1; k <= Math.min(j, theta.length); k++) v -= theta[k - 1] * pi[j - k];
    pi[j] = v;
  }
  return pi;
}

/** Solve M x = b by Gaussian elimination with partial pivoting (M is copied). */
function solveLinear(M: number[][], b: number[]): number[] {
  const n = b.length;
  const A = M.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    for (let r = col + 1; r < n; r++) {
      const f = A[r][col] / A[col][col];
      for (let c = col; c <= n; c++) A[r][c] -= f * A[col][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = A[i][n];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}

/**
 * Exact γ(0..nlags) by Brockwell–Davis §3.3 method 2: solve the (p+1)×(p+1)
 * system γ(k) − Σ_j φ_j γ(|k−j|) = c_k for γ(0..p), with
 * c_k = σ² Σ_{j=k}^{q} θ_j ψ_{j−k} (θ_0 = 1), then recurse for k > p.
 */
export function armaACVF(p: ARMAParams, nlags: number): number[] {
  const { phi, theta, sigma2 } = p;
  const P = phi.length;
  const Q = theta.length;
  const th = maPoly(theta);
  const psi = psiWeights(phi, theta, Q + 1);
  const c = new Array<number>(Math.max(nlags, P, Q) + 1).fill(0);
  for (let k = 0; k <= Q; k++) {
    let s = 0;
    for (let j = k; j <= Q; j++) s += th[j] * psi[j - k];
    c[k] = sigma2 * s;
  }
  const A: number[][] = Array.from({ length: P + 1 }, () => new Array<number>(P + 1).fill(0));
  for (let k = 0; k <= P; k++) {
    A[k][k] += 1;
    for (let j = 1; j <= P; j++) A[k][Math.abs(k - j)] -= phi[j - 1];
  }
  const g = new Array<number>(Math.max(nlags, P) + 1).fill(0);
  solveLinear(A, c.slice(0, P + 1)).forEach((v, k) => (g[k] = v));
  for (let k = P + 1; k < g.length; k++) {
    let s = k <= Q ? c[k] : 0;
    for (let j = 1; j <= P; j++) s += phi[j - 1] * g[k - j];
    g[k] = s;
  }
  return g.slice(0, nlags + 1);
}

/** ρ(0..nlags). */
export function armaACF(phi: number[], theta: number[], nlags: number): number[] {
  const g = armaACVF({ phi, theta, sigma2: 1 }, nlags);
  return g.map((v) => v / g[0]);
}

/** Durbin–Levinson (Thm 3(ii)) on γ(0..n). */
export function durbinLevinson(gamma: number[]): DurbinLevinsonResult {
  const n = gamma.length - 1;
  const v = new Array<number>(n + 1).fill(0);
  v[0] = gamma[0];
  const pacf = new Array<number>(n).fill(0);
  let prev: number[] = [];
  for (let k = 1; k <= n; k++) {
    let num = gamma[k];
    for (let j = 0; j < prev.length; j++) num -= prev[j] * gamma[k - 1 - j];
    const akk = num / v[k - 1];
    const cur = prev.map((val, j) => val - akk * prev[prev.length - 1 - j]);
    cur.push(akk);
    v[k] = v[k - 1] * (1 - akk * akk);
    pacf[k - 1] = akk;
    prev = cur;
  }
  return { pacf, phi: prev, v };
}

/** α(1..nlags) = durbinLevinson(armaACVF).pacf. */
export function armaPACF(phi: number[], theta: number[], nlags: number): number[] {
  return durbinLevinson(armaACVF({ phi, theta, sigma2: 1 }, nlags)).pacf;
}

/** γ̂(h) = (1/n) Σ_t (x_t − x̄)(x_{t+h} − x̄), h = 0..nlags; 0 for h ≥ n. */
export function sampleACVF(x: number[], nlags: number): number[] {
  const n = x.length;
  const m = x.reduce((s, v) => s + v, 0) / n;
  const d = x.map((v) => v - m);
  const out = new Array<number>(nlags + 1).fill(0);
  for (let h = 0; h <= Math.min(nlags, n - 1); h++) {
    let s = 0;
    for (let t = 0; t < n - h; t++) s += d[t] * d[t + h];
    out[h] = s / n;
  }
  return out;
}

export function sampleACF(x: number[], nlags: number): number[] {
  const g = sampleACVF(x, nlags);
  return g.map((v) => v / g[0]);
}

/** Bartlett SE of ρ̂(h) under an MA(h−1) null (Thm 9): sqrt((1 + 2 Σ_{j<h} ρ_j²)/n). */
export function bartlettSE(rho: number[], n: number, h: number): number {
  let s = 0;
  for (let j = 1; j < h; j++) s += rho[j] * rho[j];
  return Math.sqrt((1 + 2 * s) / n);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Spectral
// ═══════════════════════════════════════════════════════════════════════════

/** |Σ_k c_k e^{−ikω}|². */
function polyAbs2AtUnit(c: number[], omega: number): number {
  let re = 0;
  let im = 0;
  c.forEach((v, k) => {
    re += v * Math.cos(k * omega);
    im -= v * Math.sin(k * omega);
  });
  return re * re + im * im;
}

/** f(ω) = (σ²/2π) |θ(e^{−iω})|² / |φ(e^{−iω})|² (Thm 11). */
export function armaSpectrum(p: ARMAParams, omegas: number[]): number[] {
  const a = arPoly(p.phi);
  const b = maPoly(p.theta);
  return omegas.map(
    (w) => ((p.sigma2 / (2 * Math.PI)) * polyAbs2AtUnit(b, w)) / polyAbs2AtUnit(a, w),
  );
}

/** Σ_h γ(h) = 2π f(0) = σ² θ(1)² / φ(1)². */
export function armaLongRunVariance(p: ARMAParams): number {
  const th = maPoly(p.theta).reduce((s, v) => s + v, 0);
  const ph = arPoly(p.phi).reduce((s, v) => s + v, 0);
  return (p.sigma2 * th * th) / (ph * ph);
}

/** Newey–West (Bartlett) estimator γ̂(0) + 2 Σ_{h=1}^{M} (1 − h/(M+1)) γ̂(h). */
export function hacLongRunVariance(x: number[], M: number): number {
  const g = sampleACVF(x, M);
  let s = g[0];
  for (let h = 1; h <= M; h++) s += 2 * (1 - h / (M + 1)) * g[h];
  return s;
}

/**
 * Discrete Fourier transform X_j = Σ_t x_t e^{−2πijt/n}. Iterative radix-2 for
 * powers of two; the direct O(n²) sum otherwise (sunspots n = 309).
 */
export function fft(re: number[], im?: number[]): { re: number[]; im: number[] } {
  const n = re.length;
  const xi = im ?? new Array<number>(n).fill(0);
  if (n === 0) return { re: [], im: [] };
  if ((n & (n - 1)) !== 0) {
    // Direct DFT; exponent reduced mod n so the twiddles stay accurate.
    const cs = Array.from({ length: n }, (_, k) => Math.cos((2 * Math.PI * k) / n));
    const sn = Array.from({ length: n }, (_, k) => Math.sin((2 * Math.PI * k) / n));
    const outRe = new Array<number>(n).fill(0);
    const outIm = new Array<number>(n).fill(0);
    for (let j = 0; j < n; j++) {
      let sr = 0;
      let si = 0;
      for (let t = 0; t < n; t++) {
        const k = (j * t) % n;
        sr += re[t] * cs[k] + xi[t] * sn[k];
        si += xi[t] * cs[k] - re[t] * sn[k];
      }
      outRe[j] = sr;
      outIm[j] = si;
    }
    return { re: outRe, im: outIm };
  }
  const R = re.slice();
  const I = xi.slice();
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [R[i], R[j]] = [R[j], R[i]];
      [I[i], I[j]] = [I[j], I[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const half = len >> 1;
    const wr = Array.from({ length: half }, (_, k) => Math.cos(ang * k));
    const wi = Array.from({ length: half }, (_, k) => Math.sin(ang * k));
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const tr = R[b] * wr[k] - I[b] * wi[k];
        const ti = R[b] * wi[k] + I[b] * wr[k];
        R[b] = R[a] - tr;
        I[b] = I[a] - ti;
        R[a] += tr;
        I[a] += ti;
      }
    }
  }
  return { re: R, im: I };
}

/** I_n(ω_j) = |Σ_t x_t e^{−itω_j}|²/(2πn) on the demeaned series, j = 0..⌊n/2⌋, with I(0) = 0. */
export function periodogram(x: number[]): SpectrumEstimate {
  const n = x.length;
  const m = x.reduce((s, v) => s + v, 0) / n;
  const X = fft(x.map((v) => v - m));
  const J = Math.floor(n / 2);
  const omega: number[] = [];
  const f: number[] = [];
  for (let j = 0; j <= J; j++) {
    omega.push((2 * Math.PI * j) / n);
    f.push(j === 0 ? 0 : (X.re[j] * X.re[j] + X.im[j] * X.im[j]) / (2 * Math.PI * n));
  }
  return { omega, f };
}

/** Frequency index mapped into 0..⌊n/2⌋ by periodicity and I(−ω) = I(ω); 0 → 1. */
function reflectIndex(i: number, n: number): number {
  let k = ((i % n) + n) % n;
  if (k > Math.floor(n / 2)) k = n - k;
  return k === 0 ? 1 : k;
}

function smootherWeights(m: number, kind: 'daniell' | 'modifiedDaniell'): number[] {
  if (kind === 'daniell') return new Array<number>(2 * m + 1).fill(1 / (2 * m + 1));
  if (m === 0) return [1];
  const w = new Array<number>(2 * m + 1).fill(1 / (2 * m));
  w[0] = w[2 * m] = 1 / (4 * m);
  return w;
}

/**
 * Bartlett lag-window estimate at every Fourier frequency ω_j, j = 0..n−1, for
 * power-of-two n in O(n log n): γ̂ by zero-padded FFT, then the weighted lags
 * folded mod n (e^{−ihω_j} has period n in h, so the fold is exact) and one
 * more FFT. Same values as the direct O(nM) sum in smoothPeriodogram.
 */
function lagWindowFFT(x: number[], M: number): number[] {
  const n = x.length;
  const mean = x.reduce((s, v) => s + v, 0) / n;
  const N2 = 2 * n;
  const X = fft([...x.map((v) => v - mean), ...new Array<number>(n).fill(0)]);
  const pow = X.re.map((r, k) => r * r + X.im[k] * X.im[k]);
  // Inverse FFT of a real, even sequence = forward FFT / N2.
  const ac = fft(pow).re.map((v) => v / N2 / n);
  const c = new Array<number>(n).fill(0);
  c[0] = ac[0];
  for (let h = 1; h < Math.min(M, n); h++) {
    const v = (1 - h / M) * ac[h];
    c[h % n] += v;
    c[(n - (h % n)) % n] += v;
  }
  return fft(c).re.map((v) => v / (2 * Math.PI));
}

/**
 * Smoothed spectral estimate at ω_j, j = 1..⌊n/2⌋ (Def 10, Thm 13).
 * 'daniell' / 'modifiedDaniell': weighted average of I over j−m..j+m with
 * circular reflection. 'bartlettLag': (1/2π)[γ̂(0) + 2Σ_{h<M}(1 − h/M)γ̂(h)cos hω]
 * with the variance-matched truncation M = max(1, round(3n/(2(2m+1)))).
 */
export function smoothPeriodogram(
  x: number[],
  m: number,
  kind: SmootherKind = 'daniell',
): SpectrumEstimate {
  const n = x.length;
  const { omega, f: I } = periodogram(x);
  const J = Math.floor(n / 2);
  const om = omega.slice(1, J + 1);
  if (kind === 'bartlettLag') {
    const M = Math.max(1, roundHalfEven((3 * n) / (2 * (2 * m + 1))));
    if ((n & (n - 1)) === 0) return { omega: om, f: lagWindowFFT(x, M).slice(1, J + 1) };
    const g = sampleACVF(x, M - 1);
    const f = om.map((w) => {
      let s = g[0];
      for (let h = 1; h < M; h++) s += 2 * (1 - h / M) * g[h] * Math.cos(h * w);
      return s / (2 * Math.PI);
    });
    return { omega: om, f };
  }
  const w = smootherWeights(m, kind);
  const f = om.map((_, idx) => {
    const j = idx + 1;
    let s = 0;
    for (let k = -m; k <= m; k++) s += w[k + m] * I[reflectIndex(j + k, n)];
    return s;
  });
  return { omega: om, f };
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Estimation
// ═══════════════════════════════════════════════════════════════════════════

/** Yule–Walker via Durbin–Levinson on γ̂(0..p): returns (φ̂, σ̂² = v_p). */
export function yuleWalker(x: number[], p: number): { phi: number[]; sigma2: number } {
  const dl = durbinLevinson(sampleACVF(x, p));
  return { phi: dl.phi, sigma2: dl.v[p] };
}

/**
 * Nelder–Mead: reflection 1, expansion 2, contraction 0.5 (outside if
 * f_r < f_worst, else inside), shrink 0.5. Initial simplex x0 + step·e_i.
 * Stops when the f-spread < ftol and max ‖x_i − x_best‖∞ < xtol.
 */
export function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  opts: { step?: number; xtol?: number; ftol?: number; maxIter?: number } = {},
): NelderMeadResult {
  const step = opts.step ?? 0.1;
  const xtol = opts.xtol ?? 1e-8;
  const ftol = opts.ftol ?? 1e-12;
  const maxIter = opts.maxIter ?? 10000;
  const d = x0.length;
  let S: number[][] = [x0.slice(), ...x0.map((_, i) => x0.map((v, k) => (k === i ? v + step : v)))];
  let F = S.map(f);
  const lerp = (a: number[], b: number[], t: number) => a.map((v, k) => v + t * (b[k] - v));
  let it = 0;
  for (it = 0; it < maxIter; it++) {
    const order = F.map((_, i) => i).sort((a, b) => F[a] - F[b]); // stable
    S = order.map((i) => S[i]);
    F = order.map((i) => F[i]);
    const spread = Math.max(
      ...S.slice(1).map((s) => Math.max(...s.map((v, k) => Math.abs(v - S[0][k])))),
    );
    if (Math.abs(F[d] - F[0]) < ftol && spread < xtol) break;
    const cen = new Array<number>(d).fill(0);
    for (let i = 0; i < d; i++) for (let k = 0; k < d; k++) cen[k] += S[i][k] / d;
    const xr = lerp(cen, S[d], -1);
    const fr = f(xr);
    if (fr < F[0]) {
      const xe = lerp(cen, S[d], -2);
      const fe = f(xe);
      if (fe < fr) {
        S[d] = xe;
        F[d] = fe;
      } else {
        S[d] = xr;
        F[d] = fr;
      }
    } else if (fr < F[d - 1]) {
      S[d] = xr;
      F[d] = fr;
    } else {
      let xc: number[];
      let fc: number;
      let ok: boolean;
      if (fr < F[d]) {
        xc = lerp(cen, xr, 0.5);
        fc = f(xc);
        ok = fc <= fr;
      } else {
        xc = lerp(cen, S[d], 0.5);
        fc = f(xc);
        ok = fc < F[d];
      }
      if (ok) {
        S[d] = xc;
        F[d] = fc;
      } else {
        S = [S[0], ...S.slice(1).map((s) => lerp(S[0], s, 0.5))];
        F = [F[0], ...S.slice(1).map(f)];
      }
    }
  }
  let best = 0;
  for (let i = 1; i <= d; i++) if (F[i] < F[best]) best = i;
  return { x: S[best], fun: F[best], iterations: it };
}

/** Conditional residuals e_t = x_t − Σφ_j x_{t−j} − Σθ_j e_{t−j}, returned for t ≥ p (e_t = 0 before). */
export function cssResiduals(x: number[], phi: number[], theta: number[]): number[] {
  const n = x.length;
  const p = phi.length;
  const q = theta.length;
  const e = new Array<number>(n).fill(0);
  for (let t = p; t < n; t++) {
    let v = x[t];
    for (let j = 1; j <= p; j++) v -= phi[j - 1] * x[t - j];
    for (let j = 1; j <= q; j++) if (t - j >= 0) v -= theta[j - 1] * e[t - j];
    e[t] = v;
  }
  return e.slice(p);
}

/** Conditional least squares on the demeaned series; Nelder–Mead from (Yule–Walker AR, zero MA). */
export function armaCSS(x: number[], p: number, q: number): ARMAParams & { sse: number } {
  const m = x.reduce((s, v) => s + v, 0) / x.length;
  const xd = x.map((v) => v - m);
  const start = [...(p ? yuleWalker(xd, p).phi : []), ...new Array<number>(q).fill(0)];
  const obj = (b: number[]) =>
    cssResiduals(xd, b.slice(0, p), b.slice(p)).reduce((s, v) => s + v * v, 0);
  const res = nelderMead(obj, start);
  const e = cssResiduals(xd, res.x.slice(0, p), res.x.slice(p));
  return {
    phi: res.x.slice(0, p),
    theta: res.x.slice(p),
    sigma2: e.reduce((s, v) => s + v * v, 0) / e.length,
    sse: res.fun,
  };
}

/** Profile Whittle objective N log σ̂² + Σ log g_j, σ̂² = 2π·mean(I/g), g = |θ|²/|φ|² at e^{−iω_j} (Thm 14). */
export function whittleObjective(
  b: number[],
  omega: number[],
  I: number[],
  p: number,
  q: number,
): number {
  const g = armaSpectrum({ phi: b.slice(0, p), theta: b.slice(p, p + q), sigma2: 2 * Math.PI }, omega);
  let ratio = 0;
  let logg = 0;
  g.forEach((v, j) => {
    ratio += I[j] / v;
    logg += Math.log(v);
  });
  const s2 = (2 * Math.PI * ratio) / omega.length;
  return omega.length * Math.log(s2) + logg;
}

/** Whittle estimate over Fourier frequencies j = 1..⌊(n−1)/2⌋; Nelder–Mead from (YW AR, zero MA). */
export function whittleFit(x: number[], p: number, q: number): ARMAParams & { objective: number } {
  const n = x.length;
  const pg = periodogram(x);
  const K = Math.floor((n - 1) / 2);
  const om = pg.omega.slice(1, K + 1);
  const I = pg.f.slice(1, K + 1);
  const start = [...(p ? yuleWalker(x, p).phi : []), ...new Array<number>(q).fill(0)];
  const res = nelderMead((b) => whittleObjective(b, om, I, p, q), start);
  const b = res.x;
  const g = armaSpectrum({ phi: b.slice(0, p), theta: b.slice(p), sigma2: 2 * Math.PI }, om);
  const sigma2 = (2 * Math.PI * I.reduce((s, v, j) => s + v / g[j], 0)) / om.length;
  return { phi: b.slice(0, p), theta: b.slice(p), sigma2, objective: res.fun };
}

/** AIC = −2ℓ + 2k, BIC = −2ℓ + k log n, HQIC = −2ℓ + 2k log log n. */
export function armaIC(loglik: number, k: number, n: number): { aic: number; bic: number; hqic: number } {
  return {
    aic: -2 * loglik + 2 * k,
    bic: -2 * loglik + k * Math.log(n),
    hqic: -2 * loglik + 2 * k * Math.log(Math.log(n)),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Simulation and GARCH(1,1)
// ═══════════════════════════════════════════════════════════════════════════

/** φ(B)X_t = θ(B)ε_t from zero initial conditions, ε_t = σ·N(0,1); the first `burn` values are dropped. */
export function simulateARMA(p: ARMAParams, n: number, rng: SeededRng, burn = 500): number[] {
  const N = n + burn;
  const sd = Math.sqrt(p.sigma2);
  const e = Array.from({ length: N }, () => rng.normal() * sd);
  const x = new Array<number>(N).fill(0);
  for (let t = 0; t < N; t++) {
    let v = e[t];
    for (let k = 1; k <= p.theta.length && k <= t; k++) v += p.theta[k - 1] * e[t - k];
    for (let k = 1; k <= p.phi.length && k <= t; k++) v += p.phi[k - 1] * x[t - k];
    x[t] = v;
  }
  return x.slice(burn);
}

/** GARCH(1,1): σ²_t = ω + αx²_{t−1} + βσ²_{t−1}, x_t = σ_t z_t, σ²_0 = ω/(1−α−β). Throws if α + β ≥ 1. */
export function simulateGARCH(
  g: GarchParams,
  n: number,
  rng: SeededRng,
  burn = 500,
): { x: number[]; sigma2: number[] } {
  if (g.alpha + g.beta >= 1) throw new Error('simulateGARCH: needs α + β < 1');
  const N = n + burn;
  const x = new Array<number>(N).fill(0);
  const s2 = new Array<number>(N).fill(0);
  s2[0] = g.omega / (1 - g.alpha - g.beta);
  x[0] = Math.sqrt(s2[0]) * rng.normal();
  for (let t = 1; t < N; t++) {
    s2[t] = g.omega + g.alpha * x[t - 1] * x[t - 1] + g.beta * s2[t - 1];
    x[t] = Math.sqrt(s2[t]) * rng.normal();
  }
  return { x: x.slice(burn), sigma2: s2.slice(burn) };
}

/** σ²_t = ω + αx²_{t−1} + βσ²_{t−1}, with x²_{−1} = σ²_{−1} = backcast. */
export function garchSigma2(x: number[], g: GarchParams, backcast: number): number[] {
  const s2 = new Array<number>(x.length);
  let prevX2 = backcast;
  let prevS2 = backcast;
  for (let t = 0; t < x.length; t++) {
    s2[t] = g.omega + g.alpha * prevX2 + g.beta * prevS2;
    prevX2 = x[t] * x[t];
    prevS2 = s2[t];
  }
  return s2;
}

/** ½ Σ [log 2π + log σ²_t + x²_t/σ²_t]. */
export function garchNegLogLik(x: number[], g: GarchParams, backcast: number): number {
  const s2 = garchSigma2(x, g, backcast);
  const L2PI = Math.log(2 * Math.PI);
  let s = 0;
  for (let t = 0; t < x.length; t++) s += L2PI + Math.log(s2[t]) + (x[t] * x[t]) / s2[t];
  return 0.5 * s;
}

/** ω = e^{u0}, (α, β) = (e^{u1}, e^{u2})/(1 + e^{u1} + e^{u2}): ω > 0, α, β > 0, α + β < 1. */
function garchUnpack(u: number[]): GarchParams {
  const ea = Math.exp(u[1]);
  const eb = Math.exp(u[2]);
  return { omega: Math.exp(u[0]), alpha: ea / (1 + ea + eb), beta: eb / (1 + ea + eb) };
}

/**
 * Gaussian QMLE of GARCH(1,1). Backcast b = mean(x²) seeds both x²_{−1} and
 * σ²_{−1}; Nelder–Mead (step 0.5) on the unconstrained (u0, u1, u2) from
 * (log 0.1b, log(0.1/0.1), log(0.8/0.1)).
 */
export function garchQMLE(x: number[]): GarchFit {
  const b = x.reduce((s, v) => s + v * v, 0) / x.length;
  const u0 = [Math.log(0.1 * b), Math.log(0.1 / 0.1), Math.log(0.8 / 0.1)];
  const res = nelderMead((u) => garchNegLogLik(x, garchUnpack(u), b), u0, { step: 0.5 });
  return { ...garchUnpack(res.x), loglik: -res.fun, backcast: b, iterations: res.iterations };
}

/** ACF of x²_t for Gaussian GARCH(1,1) (Ex 18): [1, ρ1, ρ1(α+β), …, ρ1(α+β)^{nlags−1}]. */
export function garchSqACF(alpha: number, beta: number, nlags: number): number[] {
  const r1 = (alpha * (1 - alpha * beta - beta * beta)) / (1 - 2 * alpha * beta - beta * beta);
  return [1, ...Array.from({ length: nlags }, (_, h) => r1 * (alpha + beta) ** h)];
}

/** E x⁴ < ∞ for Gaussian z: 3α² + 2αβ + β² < 1. */
export function garchFourthMomentFinite(alpha: number, beta: number): boolean {
  return 3 * alpha * alpha + 2 * alpha * beta + beta * beta < 1;
}
