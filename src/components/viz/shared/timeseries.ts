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
 * Every function mirrors the Python toolkit in notebook Cell 3
 * (notebooks/hidden-markov-models/34_hidden_markov_models.ipynb) step for
 * step, so the T34 pins reproduce the notebook to the stated tolerances.
 * States are 0-indexed here and 1-indexed in the prose.
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
