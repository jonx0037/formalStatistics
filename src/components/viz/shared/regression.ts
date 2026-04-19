/**
 * regression.ts — Track 6 shared module, seeded by Topic 21
 * (Simple & Multiple Linear Regression).
 *
 * Surface:
 *   §6.1.A  Linear-algebra primitives (QR, X^T X, Cholesky inverse, matvec/matmul)
 *   §6.1.B  OLS fitting (olsFit, rSquared, adjustedRSquared)
 *   §6.1.C  Hat matrix, leverage, diagnostics (hatMatrix, leverage, studentized residuals, Cook's, Q-Q)
 *   §6.1.D  F-distribution primitives (central F re-exports + non-central F density/power)
 *   §6.1.E  Coefficient inference (SE, t-stats, Wald-t / profile / Bonferroni CIs, Working–Hotelling)
 *   §6.1.F  Nested-model F-test (fTestNested, oneWayANOVA)
 *   §6.1.G  Simulation harness (simulateLinearModel, coverageSimulator)
 *
 * Central F, t, chi-squared, and non-central chi-squared primitives live in
 * testing.ts (Topic 17 seed); the seeded uniform RNG lives in probability.ts;
 * Box–Muller normal sampling lives in convergence.ts. We import, never duplicate.
 *
 * Numerical stability: qrSolve uses Householder reflections (not direct
 * (X^T X)^-1 inversion). hatMatrix and leverage derive from the thin Q
 * factor, not from forming X (X^T X)^-1 X^T. noncentralFDensity uses the
 * Poisson-mixture series representation truncated at Poisson weight < 1e-12.
 */

import {
  fPDF,
  fCDF,
  fInvCDF,
  studentTInvCDF,
  standardNormalCDF,
  standardNormalInvCDF,
  chiSquaredCDF,
  chiSquaredInvCDF,
} from './testing';
import { seededRandom } from './probability';
import {
  normalSample,
  bernoulliSample,
  poissonSample,
  gammaSample,
} from './convergence';

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.A — Linear-algebra primitives
// ─────────────────────────────────────────────────────────────────────────────

/** Copy a 2D matrix (for non-destructive algorithms). */
function copyMatrix(A: number[][]): number[][] {
  return A.map((row) => row.slice());
}

/**
 * Thin Householder QR decomposition. Given X of shape n × m (n ≥ m, full column
 * rank), returns Q1 (n × m orthonormal) and R (m × m upper triangular) such
 * that X = Q1 R. Throws when the matrix is rank-deficient (any pivot norm
 * below 1e-12) — caller should catch and surface a friendlier error.
 *
 * Algorithm: forward pass builds R = H_{m-1} ⋯ H_0 X (upper triangular) and
 * saves each Householder vector v_k. Then Q1 is reconstructed by applying the
 * saved reflections **in reverse** to the first m columns of I_n — which
 * yields Q · E = H_0 H_1 ⋯ H_{m-1} · E_{n×m}, the thin Q we need.
 */
function thinQR(X: number[][]): { Q1: number[][]; R: number[][] } {
  const n = X.length;
  if (n === 0) throw new Error('thinQR: empty design matrix');
  const m = X[0].length;
  if (m === 0) throw new Error('thinQR: zero-column design matrix');
  if (n < m) {
    throw new Error(`thinQR: underdetermined system (n=${n} < p+1=${m})`);
  }

  // Working copy of X — becomes R (upper m×m block) after reflections.
  const R = copyMatrix(X);
  // Saved normalized Householder vectors, one per column reduction step.
  const householders: number[][] = [];

  for (let k = 0; k < m; k++) {
    let normSq = 0;
    for (let i = k; i < n; i++) normSq += R[i][k] * R[i][k];
    const norm = Math.sqrt(normSq);
    if (norm < 1e-12) {
      throw new Error(
        `thinQR: rank-deficient design matrix (column ${k} has near-zero norm)`,
      );
    }

    // Householder vector: v = x + sign(x[0]) * ‖x‖ * e_1, normalized.
    const sign = R[k][k] >= 0 ? 1 : -1;
    const v: number[] = new Array(n - k).fill(0);
    v[0] = R[k][k] + sign * norm;
    for (let i = k + 1; i < n; i++) v[i - k] = R[i][k];
    let vNorm = 0;
    for (const vi of v) vNorm += vi * vi;
    vNorm = Math.sqrt(vNorm);
    if (vNorm < 1e-14) {
      // Column k was already triangularized (zero below diagonal); save a
      // zero-reflection placeholder so the reverse-pass indexing still aligns.
      householders.push(new Array(n - k).fill(0));
      continue;
    }
    for (let i = 0; i < v.length; i++) v[i] /= vNorm;
    householders.push(v);

    // Apply H_k = I - 2 v v^T to remaining columns of R (columns k..m-1).
    // Only rows k..n-1 change (v has zeros above row k).
    for (let j = k; j < m; j++) {
      let dot = 0;
      for (let i = k; i < n; i++) dot += v[i - k] * R[i][j];
      const scale = 2 * dot;
      for (let i = k; i < n; i++) R[i][j] -= scale * v[i - k];
    }
  }

  // Extract the m×m upper-triangular R.
  const Rsq: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = i; j < m; j++) Rsq[i][j] = R[i][j];
  }

  // Build Q1 = H_0 H_1 ⋯ H_{m-1} · E_{n×m} by applying reflections in reverse
  // (left-multiplying each H_k onto Q1). Start Q1 as the first m columns of I_n.
  const Q1: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? 1 : 0)),
  );
  for (let k = m - 1; k >= 0; k--) {
    const v = householders[k];
    // Zero-reflection sentinel (‖v‖ was ≈ 0) — nothing to apply.
    let vNonZero = false;
    for (const vi of v) {
      if (vi !== 0) {
        vNonZero = true;
        break;
      }
    }
    if (!vNonZero) continue;

    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let i = k; i < n; i++) dot += v[i - k] * Q1[i][j];
      const scale = 2 * dot;
      for (let i = k; i < n; i++) Q1[i][j] -= scale * v[i - k];
    }
  }

  return { Q1, R: Rsq };
}

/**
 * Solve the OLS normal equations X^T X β = X^T y via thin Householder QR.
 * Numerically stable: avoids forming (X^T X)^-1. Throws on rank-deficient X.
 */
export function qrSolve(X: number[][], y: number[]): number[] {
  const n = X.length;
  const m = X[0].length;
  if (y.length !== n) {
    throw new Error(`qrSolve: dimension mismatch (n=${n}, y.length=${y.length})`);
  }

  const { Q1, R } = thinQR(X);

  // c = Q1^T y, an m-vector.
  const c: number[] = new Array(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Q1[i][j] * y[i];
    c[j] = s;
  }

  // Back-substitute R β = c.
  const beta: number[] = new Array(m).fill(0);
  for (let i = m - 1; i >= 0; i--) {
    let s = c[i];
    for (let j = i + 1; j < m; j++) s -= R[i][j] * beta[j];
    if (Math.abs(R[i][i]) < 1e-12) {
      throw new Error(`qrSolve: rank-deficient (R[${i}][${i}] ≈ 0)`);
    }
    beta[i] = s / R[i][i];
  }

  return beta;
}

/** Compute X^T X — returned as a fresh symmetric m × m matrix. */
export function xtx(X: number[][]): number[][] {
  const n = X.length;
  const m = X[0].length;
  const out: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let j = 0; j < m; j++) {
    for (let k = j; k < m; k++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X[i][j] * X[i][k];
      out[j][k] = s;
      out[k][j] = s;
    }
  }
  return out;
}

/** Compute X^T y — returned as a fresh m-vector. */
export function xty(X: number[][], y: number[]): number[] {
  const n = X.length;
  const m = X[0].length;
  const out: number[] = new Array(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j] * y[i];
    out[j] = s;
  }
  return out;
}

/**
 * Invert a positive-definite symmetric matrix via Cholesky. Throws on non-PD.
 * Used for (X^T X)^-1 once thinQR has established X is full column rank.
 */
export function choleskyInverse(A: number[][]): number[][] {
  const n = A.length;
  // Cholesky factor L (lower triangular): A = L L^T.
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('choleskyInverse: matrix is not positive-definite');
        L[i][i] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }

  // Invert L via forward substitution: L X = I → X = L^-1 (stored in Linv).
  const Linv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let col = 0; col < n; col++) {
    for (let i = 0; i < n; i++) {
      const rhs = i === col ? 1 : 0;
      let s = rhs;
      for (let k = 0; k < i; k++) s -= L[i][k] * Linv[k][col];
      Linv[i][col] = s / L[i][i];
    }
  }

  // A^-1 = L^-T L^-1 = (Linv)^T Linv.
  const inv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Linv[k][i] * Linv[k][j];
      inv[i][j] = s;
      inv[j][i] = s;
    }
  }
  return inv;
}

/** Matrix × vector. */
export function matVec(A: number[][], v: number[]): number[] {
  const n = A.length;
  const m = A[0].length;
  if (v.length !== m) throw new Error(`matVec: dimension mismatch (m=${m}, v.length=${v.length})`);
  const out: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += A[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

/** Matrix × matrix. */
export function matMul(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  const k = A[0].length;
  const m = B[0].length;
  if (B.length !== k) throw new Error(`matMul: dimension mismatch (A is n×k with k=${k}, B has ${B.length} rows)`);
  const out: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
      out[i][j] = s;
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.B — OLS fitting
// ─────────────────────────────────────────────────────────────────────────────

export interface OLSFit {
  beta: number[];
  residuals: number[];
  fitted: number[];
  sigmaSquared: number;
  sse: number;
  sst: number;
  ssr: number;
  xtxInv: number[][];
  rSquared: number;
  adjustedRSquared: number;
  n: number;
  p: number; // non-intercept predictors (p+1 = columns of X, assuming intercept)
}

/**
 * Fit OLS via QR. X is n × (p+1) — include a leading column of ones if an
 * intercept is desired. `p` in the returned OLSFit is (columns of X) - 1,
 * following the brief's convention that `p` counts non-intercept predictors.
 */
export function olsFit(X: number[][], y: number[]): OLSFit {
  const n = X.length;
  const m = X[0].length;
  const pNonIntercept = m - 1;

  const beta = qrSolve(X, y);
  const fitted = matVec(X, beta);
  const residuals = y.map((yi, i) => yi - fitted[i]);

  let sse = 0;
  for (const r of residuals) sse += r * r;
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  let sst = 0;
  for (const yi of y) sst += (yi - yMean) * (yi - yMean);
  const ssr = sst - sse;

  const residualDf = n - m; // n - (p + 1)
  if (residualDf <= 0) {
    throw new Error(`olsFit: residual df = ${residualDf} ≤ 0 (n=${n}, columns of X=${m})`);
  }
  const sigmaSquared = sse / residualDf;

  // (X^T X)^-1 via Cholesky on X^T X.
  const xtxInv = choleskyInverse(xtx(X));

  const rSq = sst > 0 ? 1 - sse / sst : 1;
  const adjRSq =
    sst > 0 && residualDf > 0
      ? 1 - ((1 - rSq) * (n - 1)) / residualDf
      : rSq;

  return {
    beta,
    residuals,
    fitted,
    sigmaSquared,
    sse,
    sst,
    ssr,
    xtxInv,
    rSquared: rSq,
    adjustedRSquared: adjRSq,
    n,
    p: pNonIntercept,
  };
}

export function rSquared(fit: OLSFit): number {
  return fit.rSquared;
}

export function adjustedRSquared(fit: OLSFit): number {
  return fit.adjustedRSquared;
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.C — Hat matrix, leverage, diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hat matrix H = X (X^T X)^-1 X^T, computed as Q1 Q1^T from the thin QR of X.
 * O(n² m) in time, O(n²) in storage. Use `leverage` if only diag(H) is needed.
 */
export function hatMatrix(X: number[][]): number[][] {
  const n = X.length;
  const { Q1 } = thinQR(X);
  const m = Q1[0].length;
  const H: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += Q1[i][k] * Q1[j][k];
      H[i][j] = s;
      H[j][i] = s;
    }
  }
  return H;
}

/**
 * Leverage values h_ii = ‖Q1[i, :]‖² from the thin QR of X. O(n m²) in time,
 * O(n m) in storage — cheaper than forming H when only the diagonal is wanted.
 */
export function leverage(X: number[][]): number[] {
  const n = X.length;
  const { Q1 } = thinQR(X);
  const m = Q1[0].length;
  const h: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < m; k++) s += Q1[i][k] * Q1[i][k];
    h[i] = s;
  }
  return h;
}

/**
 * Internally studentized residuals: e_i / (σ̂ √(1 - h_ii)). When the model
 * fits perfectly (σ̂² = 0), numerator and denominator are both zero — return
 * 0 (the limit of the ratio along the residuals-are-zero line) rather than
 * propagating NaN into the UI. Also guards h_ii → 1 (leverage-one observations).
 */
export function studentizedResiduals(fit: OLSFit, X: number[][]): number[] {
  return studentizedResidualsFromLeverage(fit, leverage(X));
}

/**
 * Lower-level variant that accepts a precomputed leverage vector — lets
 * Cook's distance avoid recomputing the thin-QR. Shape matches
 * `studentizedResiduals(fit, X)` for the same `(fit, X)` pair.
 */
function studentizedResidualsFromLeverage(fit: OLSFit, h: number[]): number[] {
  const sigma = Math.sqrt(fit.sigmaSquared);
  if (sigma === 0) return fit.residuals.map(() => 0);
  return fit.residuals.map((ei, i) => {
    const denom = sigma * Math.sqrt(Math.max(1 - h[i], 1e-12));
    return ei / denom;
  });
}

/**
 * Cook's distance: D_i = (r_i² / (p+1)) * (h_ii / (1 - h_ii)), where r_i is
 * the internally studentized residual. Flags observations whose removal would
 * substantially shift the fit. Computes the thin-QR exactly once by sharing
 * the leverage vector with the studentized-residual helper.
 */
export function cooksDistance(fit: OLSFit, X: number[][]): number[] {
  const h = leverage(X);
  const r = studentizedResidualsFromLeverage(fit, h);
  const pPlusOne = fit.p + 1;
  return r.map((ri, i) => {
    const denom = Math.max(1 - h[i], 1e-12);
    return ((ri * ri) / pPlusOne) * (h[i] / denom);
  });
}

/**
 * Q-Q plot data — studentized residuals vs theoretical t_{n-p-1} quantiles.
 * Returns sorted {theoretical, sample} pairs at the (i - 0.5)/n plotting positions.
 */
export function qqPlotData(
  fit: OLSFit,
  X: number[][],
): { theoretical: number[]; sample: number[] } {
  const n = fit.n;
  const df = n - fit.p - 1;
  const sample = studentizedResiduals(fit, X).slice().sort((a, b) => a - b);
  const theoretical: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const q = (i + 0.5) / n;
    theoretical[i] = studentTInvCDF(q, df);
  }
  return { theoretical, sample };
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.D — F-distribution primitives
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Central F_{df1, df2} density. Re-exports testing.ts's fPDF under the
 * brief's naming convention.
 */
export function fDensity(x: number, df1: number, df2: number): number {
  return fPDF(x, df1, df2);
}

/** Central F_{df1, df2} CDF — re-export of testing.ts's fCDF. */
export { fCDF };

/**
 * Central F_{df1, df2} quantile. Re-exports testing.ts's fInvCDF under the
 * brief's naming convention.
 */
export function fQuantile(p: number, df1: number, df2: number): number {
  return fInvCDF(p, df1, df2);
}

/**
 * Shared Poisson(λ/2)-mixture summer for the non-central F representations.
 *
 *   result(x; df1, df2, λ) = Σ_j P(j; λ/2) · term(j, df1 + 2j, x·df1/(df1+2j))
 *
 * Mode-centered summation: the naive forward sweep starting at
 * weight₀ = exp(−λ/2) underflows to 0 for λ ≳ 1490 (since exp(−745) ≈ 5e−324,
 * the subnormal floor), after which the whole series silently returns zero.
 * Starting at the Poisson mode j* = ⌊λ/2⌋ with unnormalized weight 1 and
 * iterating outward — upward via wₖ₊₁ = wₖ · (λ/2)/(k+1), downward via
 * wₖ₋₁ = wₖ · k/(λ/2) — keeps every intermediate weight bounded near 1 in
 * the region where terms matter. Normalizing by the sum of visited weights
 * recovers the true Poisson probabilities without ever materializing exp(−λ/2).
 */
function poissonFMixtureSum(
  lambda: number,
  df1: number,
  x: number,
  term: (df1Eff: number, xEff: number) => number,
): number {
  const halfLambda = lambda / 2;
  const mode = Math.floor(halfLambda);
  const maxJ = 10000;
  const eps = 1e-12;

  const addTerm = (j: number, w: number): number => {
    const df1Eff = df1 + 2 * j;
    const xEff = (x * df1) / df1Eff;
    return w * term(df1Eff, xEff);
  };

  let numer = addTerm(mode, 1);
  let denom = 1;

  // Upward: j = mode+1, mode+2, …
  let w = 1;
  for (let j = mode; j < mode + maxJ; j++) {
    w *= halfLambda / (j + 1);
    if (w < eps) break;
    numer += addTerm(j + 1, w);
    denom += w;
  }

  // Downward: j = mode−1, mode−2, …, 0
  w = 1;
  for (let j = mode; j > 0; j--) {
    w *= j / halfLambda;
    if (w < eps) break;
    numer += addTerm(j - 1, w);
    denom += w;
  }

  return numer / denom;
}

/**
 * Non-central F_{df1, df2}(λ) density at x via the Poisson-mixture series
 *   f(x; df1, df2, λ) = Σ_j P(j; λ/2) · (df1 / (df1+2j)) · f_F(x·df1/(df1+2j); df1+2j, df2).
 * See `poissonFMixtureSum` for the mode-centered summation rationale.
 */
export function noncentralFDensity(
  x: number,
  df1: number,
  df2: number,
  lambda: number,
): number {
  if (x <= 0 || df1 <= 0 || df2 <= 0 || lambda < 0) return 0;
  if (lambda === 0) return fPDF(x, df1, df2);
  return poissonFMixtureSum(
    lambda,
    df1,
    x,
    (df1Eff, xEff) => (df1 / df1Eff) * fPDF(xEff, df1Eff, df2),
  );
}

/**
 * Non-central F CDF at x via the Poisson-mixture representation
 *   F(x; df1, df2, λ) = Σ_j P(j; λ/2) · F_F(x·df1/(df1+2j); df1+2j, df2).
 */
function noncentralFCDF(
  x: number,
  df1: number,
  df2: number,
  lambda: number,
): number {
  if (x <= 0 || df1 <= 0 || df2 <= 0 || lambda < 0) return 0;
  if (lambda === 0) return fCDF(x, df1, df2);
  return poissonFMixtureSum(
    lambda,
    df1,
    x,
    (df1Eff, xEff) => fCDF(xEff, df1Eff, df2),
  );
}

/**
 * Power of the level-α F-test at non-centrality λ:
 *   Power(λ) = 1 - F_{NC-F}(F_crit; df1, df2, λ)
 * where F_crit = F_{df1, df2}^-1(1 - α) is the level-α critical value under H_0.
 */
export function noncentralFPower(
  df1: number,
  df2: number,
  lambda: number,
  alpha: number,
): number {
  const fCrit = fInvCDF(1 - alpha, df1, df2);
  return 1 - noncentralFCDF(fCrit, df1, df2, lambda);
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.E — Coefficient inference
// ─────────────────────────────────────────────────────────────────────────────

/** Per-coefficient standard errors: σ̂ √((X^T X)^-1_{jj}). */
export function coefSE(fit: OLSFit): number[] {
  const sigma = Math.sqrt(fit.sigmaSquared);
  return fit.xtxInv.map((row, j) => sigma * Math.sqrt(Math.max(row[j], 0)));
}

/**
 * Per-coefficient t-statistics: (β̂_j - β_j^0) / SE(β̂_j). Default null value
 * β_j^0 = 0 for each coefficient (the "is this coefficient present?" test).
 */
export function coefTStatistic(
  fit: OLSFit,
  nullValues?: number[],
): number[] {
  const se = coefSE(fit);
  const nulls = nullValues ?? new Array(fit.beta.length).fill(0);
  return fit.beta.map((b, j) => (b - nulls[j]) / se[j]);
}

export interface CIEntry {
  lower: number;
  upper: number;
  center: number;
}

/**
 * Per-coefficient Wald-t CI at individual level 1 - α:
 *   β̂_j ± t_{n-p-1, 1-α/2} · SE(β̂_j)
 */
export function coefCIWald(fit: OLSFit, alpha: number): CIEntry[] {
  const se = coefSE(fit);
  const df = fit.n - fit.p - 1;
  const tCrit = studentTInvCDF(1 - alpha / 2, df);
  return fit.beta.map((b, j) => ({
    lower: b - tCrit * se[j],
    upper: b + tCrit * se[j],
    center: b,
  }));
}

/**
 * Per-coefficient profile-likelihood CI. Under the Normal linear model, the
 * profile likelihood for β_j is quadratic and the profile CI coincides with
 * the Wald-t CI. The function exists for API parity with Topic 19 and for
 * forward compatibility with Topic 22's GLM profile CIs (where they diverge).
 */
export function coefCIProfile(
  fit: OLSFit,
  _X: number[][],
  _y: number[],
  alpha: number,
): CIEntry[] {
  // Under Normal errors, profile CI ≡ Wald-t CI. Retain the signature for
  // Topic-19 parity; Topic 22 will overload this with IRLS-based profile CIs.
  return coefCIWald(fit, alpha);
}

/**
 * Bonferroni-adjusted Wald-t CIs at family-wise level 1 - α. Each individual
 * CI is widened to level 1 - α/(p+1), guaranteeing FWER ≤ α by the union
 * bound. See Topic 20 §20.9 Thm 8.
 */
export function coefCIBonferroni(fit: OLSFit, alpha: number): CIEntry[] {
  const pPlusOne = fit.beta.length;
  return coefCIWald(fit, alpha / pPlusOne);
}

export interface BandEntry {
  x: number[];
  lower: number;
  upper: number;
  center: number;
}

/**
 * Working–Hotelling simultaneous confidence band for the regression function
 * μ(x) = x^T β at a grid of x values. Critical value is √((p+1) · F_{p+1,
 * n-p-1, 1-α}) — the same bound for every grid point, so the FWER ≤ α
 * guarantee holds uniformly across the continuum of x values, not merely at
 * the evaluation grid.
 */
export function workingHotellingBand(
  fit: OLSFit,
  _X: number[][],
  xGrid: number[][],
  alpha: number,
): BandEntry[] {
  const pPlusOne = fit.beta.length;
  const df2 = fit.n - pPlusOne;
  const fCrit = fInvCDF(1 - alpha, pPlusOne, df2);
  const w = Math.sqrt(pPlusOne * fCrit);
  const sigma = Math.sqrt(fit.sigmaSquared);

  return xGrid.map((x) => {
    const center = x.reduce((s, xj, j) => s + xj * fit.beta[j], 0);
    // SE(x^T β̂) = σ̂ √(x^T (X^T X)^-1 x).
    let q = 0;
    for (let i = 0; i < pPlusOne; i++) {
      for (let j = 0; j < pPlusOne; j++) {
        q += x[i] * fit.xtxInv[i][j] * x[j];
      }
    }
    const se = sigma * Math.sqrt(Math.max(q, 0));
    return { x, lower: center - w * se, upper: center + w * se, center };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.F — Nested-model F-test
// ─────────────────────────────────────────────────────────────────────────────

export interface FTestResult {
  F: number;
  pValue: number;
  df1: number;
  df2: number;
  sseFull: number;
  sseReduced: number;
  reject: boolean;
}

/**
 * Nested-model F-test. X_full and X_reduced must have the same n, share y,
 * and the columns of X_reduced must span a subspace of the columns of X_full
 * (the test is only valid under nesting). Returns F, p-value, and a reject
 * flag at level α.
 */
export function fTestNested(
  Xfull: number[][],
  Xreduced: number[][],
  y: number[],
  alpha: number,
): FTestResult {
  const fullFit = olsFit(Xfull, y);
  const redFit = olsFit(Xreduced, y);
  const k = Xfull[0].length - Xreduced[0].length;
  if (k <= 0) {
    throw new Error('fTestNested: X_full must have more columns than X_reduced');
  }
  const df2 = fullFit.n - Xfull[0].length;
  const F = (redFit.sse - fullFit.sse) / k / (fullFit.sse / df2);
  const pValue = 1 - fCDF(F, k, df2);
  return {
    F,
    pValue,
    df1: k,
    df2,
    sseFull: fullFit.sse,
    sseReduced: redFit.sse,
    reject: pValue < alpha,
  };
}

/**
 * One-way ANOVA as a nested F-test. `groups` is an integer label per
 * observation (0-indexed); `y` the response. The full model has a separate
 * intercept per group (dummy-coded, no overall intercept); the reduced model
 * is the common-mean model (single intercept column of ones). This is the
 * "ANOVA is regression" framing of §21.8 Example 9.
 */
export function oneWayANOVA(
  groups: number[],
  y: number[],
  alpha: number,
): FTestResult {
  if (groups.length !== y.length) {
    throw new Error('oneWayANOVA: groups and y must have equal length');
  }
  const n = y.length;
  const uniqueGroups = Array.from(new Set(groups)).sort((a, b) => a - b);
  const g = uniqueGroups.length;
  if (g < 2) {
    throw new Error('oneWayANOVA: at least 2 groups required');
  }
  const Xfull: number[][] = Array.from({ length: n }, () => new Array(g).fill(0));
  for (let i = 0; i < n; i++) {
    const idx = uniqueGroups.indexOf(groups[i]);
    Xfull[i][idx] = 1;
  }
  const Xreduced: number[][] = Array.from({ length: n }, () => [1]);
  return fTestNested(Xfull, Xreduced, y, alpha);
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.G — Simulation harness
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Simulate y from the Normal linear model y = X β + ε, ε_i ~ iid N(0, σ²).
 * Uses the Topic-17 seeded RNG (`seededRandom` in shared/probability.ts, a
 * 32-bit linear-congruential generator) composed with Box–Muller normal
 * sampling (`normalSample`) for reproducibility. The LCG has known
 * low-dimensional correlation between nearby seeds — use well-separated
 * seeds (differ by ≥ 10⁵) when testing stream independence.
 */
export function simulateLinearModel(
  X: number[][],
  beta: number[],
  sigma: number,
  seed: number,
): number[] {
  const n = X.length;
  const rng = seededRandom(seed);
  const mean = matVec(X, beta);
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    y[i] = mean[i] + normalSample(0, sigma, rng);
  }
  return y;
}

/**
 * MC coverage simulator for a CI constructor. Runs `iterations` simulations
 * of y from (X, β_true, σ) with the given seed stream, fits OLS, constructs
 * CIs via `ciConstructor`, and returns empirical per-coefficient coverage.
 */
export function coverageSimulator(
  X: number[][],
  betaTrue: number[],
  sigma: number,
  iterations: number,
  ciConstructor: (fit: OLSFit, alpha: number) => CIEntry[],
  alpha: number,
  seed: number,
): number[] {
  const pPlusOne = betaTrue.length;
  const hits: number[] = new Array(pPlusOne).fill(0);
  for (let iter = 0; iter < iterations; iter++) {
    const y = simulateLinearModel(X, betaTrue, sigma, seed + iter);
    const fit = olsFit(X, y);
    const cis = ciConstructor(fit, alpha);
    for (let j = 0; j < pPlusOne; j++) {
      if (cis[j].lower <= betaTrue[j] && betaTrue[j] <= cis[j].upper) hits[j]++;
    }
  }
  return hits.map((h) => h / iterations);
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.H — Link function / variance function / GLM family catalog (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A link function bundles g, g^{-1}, and g'(μ) — all three are needed for IRLS
 * (g for evaluation, g^{-1} for the inverse-link μ = g^{-1}(η), g'(μ) for the
 * adjusted-response weighting).
 */
export interface LinkFunction {
  name: 'logit' | 'probit' | 'cloglog' | 'log' | 'identity' | 'inverse' | 'sqrt';
  g: (mu: number) => number;
  gInv: (eta: number) => number;
  gPrime: (mu: number) => number;
}

/**
 * Exponential-family variance function V(μ). Bernoulli: μ(1-μ); Poisson: μ;
 * Gamma: μ²; Normal: 1.
 */
export interface VarianceFunction {
  name: 'bernoulli' | 'poisson' | 'gamma' | 'normal';
  V: (mu: number) => number;
}

/**
 * A GLM family bundles the canonical link, the variance function, the
 * log-likelihood contribution, the saturated-model log-likelihood contribution
 * (for deviance), an initial-μ heuristic for IRLS, and a flag for whether the
 * dispersion φ is fixed at 1 (Bernoulli, Poisson) or estimated (Gamma, Normal).
 */
export interface GLMFamily {
  name: 'bernoulli' | 'poisson' | 'gamma' | 'normal';
  canonicalLink: LinkFunction;
  variance: VarianceFunction;
  /** ℓ_i(μ; y, φ) — pointwise log-likelihood contribution. */
  logLik: (y: number, mu: number, phi: number) => number;
  /** ℓ_i(y; y, φ) — saturated-model contribution (used in deviance). */
  logLikSaturated: (y: number, phi: number) => number;
  /** Per-observation deviance contribution d_i = 2[ℓ_sat - ℓ_i] / φ. */
  devianceContribution: (y: number, mu: number) => number;
  /** Family-specific initial μ for IRLS (smoothed-y for Bernoulli/Poisson). */
  initMu: (y: number[]) => number[];
  /** φ ≡ 1 for Bernoulli/Poisson; estimated for Gamma/Normal. */
  fixedDispersion: boolean;
}

// ── Numerical clip constants (per §22 brief Topic-22 gotcha G5) ─────────────
const PROB_CLIP_LO = 1e-12;
const PROB_CLIP_HI = 1 - 1e-12;
const POSITIVE_CLIP = 1e-12;
const HC3_LEVERAGE_CAP = 0.9999;
const WEIGHT_CLIP_LO = 1e-12;

const sigmoid = (x: number): number => {
  // Stable variant: avoid overflow on very negative x.
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
};

const clip = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

// ── Link functions ──────────────────────────────────────────────────────────

const logitLink: LinkFunction = {
  name: 'logit',
  g: (mu) => {
    const m = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    return Math.log(m / (1 - m));
  },
  gInv: (eta) => sigmoid(eta),
  gPrime: (mu) => {
    const m = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    return 1 / (m * (1 - m));
  },
};

const probitLink: LinkFunction = {
  name: 'probit',
  g: (mu) => standardNormalInvCDF(clip(mu, PROB_CLIP_LO, PROB_CLIP_HI)),
  gInv: (eta) => standardNormalCDF(eta),
  gPrime: (mu) => {
    const m = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    const eta = standardNormalInvCDF(m);
    const phiEta = Math.exp(-0.5 * eta * eta) / Math.sqrt(2 * Math.PI);
    return 1 / Math.max(phiEta, PROB_CLIP_LO);
  },
};

const cloglogLink: LinkFunction = {
  name: 'cloglog',
  g: (mu) => {
    const m = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    return Math.log(-Math.log(1 - m));
  },
  gInv: (eta) => 1 - Math.exp(-Math.exp(eta)),
  gPrime: (mu) => {
    const m = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    return -1 / ((1 - m) * Math.log(1 - m));
  },
};

const logLink: LinkFunction = {
  name: 'log',
  g: (mu) => Math.log(Math.max(mu, POSITIVE_CLIP)),
  gInv: (eta) => Math.exp(eta),
  gPrime: (mu) => 1 / Math.max(mu, POSITIVE_CLIP),
};

const identityLink: LinkFunction = {
  name: 'identity',
  g: (mu) => mu,
  gInv: (eta) => eta,
  gPrime: () => 1,
};

const inverseLink: LinkFunction = {
  name: 'inverse',
  g: (mu) => 1 / Math.max(mu, POSITIVE_CLIP),
  gInv: (eta) => 1 / Math.max(eta, POSITIVE_CLIP),
  gPrime: (mu) => -1 / (mu * mu),
};

const sqrtLink: LinkFunction = {
  name: 'sqrt',
  g: (mu) => Math.sqrt(Math.max(mu, POSITIVE_CLIP)),
  gInv: (eta) => eta * eta,
  gPrime: (mu) => 0.5 / Math.sqrt(Math.max(mu, POSITIVE_CLIP)),
};

export const LINKS: Record<string, LinkFunction> = {
  logit: logitLink,
  probit: probitLink,
  cloglog: cloglogLink,
  log: logLink,
  identity: identityLink,
  inverse: inverseLink,
  sqrt: sqrtLink,
};

// ── Variance functions ──────────────────────────────────────────────────────

const bernoulliVar: VarianceFunction = {
  name: 'bernoulli',
  V: (mu) => {
    const m = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    return m * (1 - m);
  },
};
const poissonVar: VarianceFunction = {
  name: 'poisson',
  V: (mu) => Math.max(mu, POSITIVE_CLIP),
};
const gammaVar: VarianceFunction = {
  name: 'gamma',
  V: (mu) => {
    const m = Math.max(mu, POSITIVE_CLIP);
    return m * m;
  },
};
const normalVar: VarianceFunction = { name: 'normal', V: () => 1 };

export const VARIANCES: Record<string, VarianceFunction> = {
  bernoulli: bernoulliVar,
  poisson: poissonVar,
  gamma: gammaVar,
  normal: normalVar,
};

// ── Family catalog ──────────────────────────────────────────────────────────

/** y log(y) with the convention 0 log 0 = 0. */
const yLogY = (y: number): number => (y > 0 ? y * Math.log(y) : 0);

const bernoulliFamily: GLMFamily = {
  name: 'bernoulli',
  canonicalLink: logitLink,
  variance: bernoulliVar,
  logLik: (y, mu) => {
    const p = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    return y * Math.log(p) + (1 - y) * Math.log(1 - p);
  },
  logLikSaturated: () => 0, // y∈{0,1} ⇒ y log(y) + (1-y) log(1-y) = 0
  devianceContribution: (y, mu) => {
    const p = clip(mu, PROB_CLIP_LO, PROB_CLIP_HI);
    const a = y > 0 ? y * Math.log(y / p) : 0;
    const b = y < 1 ? (1 - y) * Math.log((1 - y) / (1 - p)) : 0;
    return 2 * (a + b);
  },
  initMu: (y) => {
    // Smooth toward 0.5 to avoid logit(0) / logit(1) on first iteration.
    return y.map((yi) => (yi + 0.5) / 2);
  },
  fixedDispersion: true,
};

const poissonFamily: GLMFamily = {
  name: 'poisson',
  canonicalLink: logLink,
  variance: poissonVar,
  logLik: (y, mu) => {
    const m = Math.max(mu, POSITIVE_CLIP);
    return y * Math.log(m) - m; // drop log Γ(y+1) — cancels in deviance
  },
  logLikSaturated: (y) => yLogY(y) - y,
  devianceContribution: (y, mu) => {
    const m = Math.max(mu, POSITIVE_CLIP);
    const term1 = y > 0 ? y * Math.log(y / m) : 0;
    return 2 * (term1 - (y - m));
  },
  initMu: (y) => y.map((yi) => yi + 0.1),
  fixedDispersion: true,
};

const gammaFamily: GLMFamily = {
  name: 'gamma',
  canonicalLink: inverseLink,
  variance: gammaVar,
  logLik: (y, mu, phi) => {
    // Up to dispersion-only constants; ν = 1/φ.
    const m = Math.max(mu, POSITIVE_CLIP);
    const yClip = Math.max(y, POSITIVE_CLIP);
    return -Math.log(m) - yClip / m + (Math.log(yClip) - 1) / Math.max(phi, POSITIVE_CLIP);
  },
  logLikSaturated: (y) => {
    const yClip = Math.max(y, POSITIVE_CLIP);
    return -Math.log(yClip) - 1; // matches logLik with mu = y
  },
  devianceContribution: (y, mu) => {
    const m = Math.max(mu, POSITIVE_CLIP);
    // y clipped only inside the deviance-term log, NOT in the fit (G5).
    const yLog = Math.max(y, POSITIVE_CLIP);
    return 2 * (-Math.log(yLog / m) + (y - m) / m);
  },
  initMu: (y) => y.map((yi) => Math.max(yi, POSITIVE_CLIP)),
  fixedDispersion: false,
};

const normalFamily: GLMFamily = {
  name: 'normal',
  canonicalLink: identityLink,
  variance: normalVar,
  logLik: (y, mu, phi) => {
    const sigmaSq = Math.max(phi, POSITIVE_CLIP);
    const r = y - mu;
    return -0.5 * (Math.log(2 * Math.PI * sigmaSq) + (r * r) / sigmaSq);
  },
  logLikSaturated: (_y, phi) => {
    const sigmaSq = Math.max(phi, POSITIVE_CLIP);
    return -0.5 * Math.log(2 * Math.PI * sigmaSq);
  },
  devianceContribution: (y, mu) => {
    const r = y - mu;
    return r * r;
  },
  initMu: (y) => y.slice(),
  fixedDispersion: false,
};

export const FAMILIES: Record<string, GLMFamily> = {
  bernoulli: bernoulliFamily,
  poisson: poissonFamily,
  gamma: gammaFamily,
  normal: normalFamily,
};

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.I — IRLS solver (glmFit, irlsStep, predictGLM) (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────

export interface GLMFit {
  beta: number[];
  mu: number[];
  eta: number[];
  weights: number[];
  deviance: number;
  nullDeviance: number;
  phi: number;
  nIter: number;
  converged: boolean;
  X: number[][];
  y: number[];
  family: GLMFamily;
  link: LinkFunction;
  /** Asymptotic variance-covariance: φ · (X^T W X)^{-1} */
  vcov: number[][];
  offset: number[] | null;
}

/** Solve A β = b for symmetric positive-definite A via Cholesky. */
function choleskySolve(A: number[][], b: number[]): number[] {
  const n = A.length;
  // L is the lower-triangular Cholesky factor: A = L L^T.
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) {
          throw new Error(
            `choleskySolve: matrix not positive-definite at pivot ${i} (diagonal=${s.toExponential(3)})`,
          );
        }
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  // Forward solve L u = b.
  const u: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * u[k];
    u[i] = s / L[i][i];
  }
  // Backward solve L^T x = u.
  const x: number[] = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = u[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** Compute X^T diag(w) X (symmetric). */
function xtwx(X: number[][], w: number[]): number[][] {
  const n = X.length;
  const m = X[0].length;
  const out: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    const wi = w[i];
    const xi = X[i];
    for (let j = 0; j < m; j++) {
      const xij = xi[j];
      for (let k = j; k < m; k++) {
        out[j][k] += wi * xij * xi[k];
      }
    }
  }
  // Mirror upper triangle into lower.
  for (let j = 0; j < m; j++) {
    for (let k = 0; k < j; k++) out[j][k] = out[k][j];
  }
  return out;
}

/** Compute X^T diag(w) z. */
function xtwz(X: number[][], w: number[], z: number[]): number[] {
  const n = X.length;
  const m = X[0].length;
  const out: number[] = new Array(m).fill(0);
  for (let i = 0; i < n; i++) {
    const wzi = w[i] * z[i];
    const xi = X[i];
    for (let j = 0; j < m; j++) out[j] += xi[j] * wzi;
  }
  return out;
}

/**
 * Single IRLS iteration step. Exposed so `IRLSVisualizer` can animate one step
 * at a time. Returns the next β plus per-observation diagnostics needed for
 * the right panel of the visualizer (current μ, η, weights, adjusted response).
 *
 * Uses the general-link IRLS form W_ii = 1/[V(μ_i) · g'(μ_i)²], which collapses
 * to W_ii = V(μ_i) under the canonical link.
 */
export function irlsStep(
  X: number[][],
  y: number[],
  betaCurrent: number[],
  family: GLMFamily,
  link: LinkFunction,
  offset?: number[],
): {
  betaNext: number[];
  mu: number[];
  eta: number[];
  weights: number[];
  adjustedResponse: number[];
  logLik: number;
} {
  const n = X.length;
  const off = offset ?? new Array<number>(n).fill(0);
  // Current η, μ.
  const linPred = matVec(X, betaCurrent);
  const eta = new Array<number>(n);
  const mu = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    eta[i] = linPred[i] + off[i];
    mu[i] = link.gInv(eta[i]);
  }
  // IRLS weights and adjusted response.
  const weights = new Array<number>(n);
  const z = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const gp = link.gPrime(mu[i]);
    const V = family.variance.V(mu[i]);
    const denom = V * gp * gp;
    weights[i] = 1 / Math.max(denom, WEIGHT_CLIP_LO);
    // Adjusted response in the predictor scale; remove offset before the WLS solve.
    z[i] = (eta[i] - off[i]) + gp * (y[i] - mu[i]);
  }
  const A = xtwx(X, weights);
  const b = xtwz(X, weights, z);
  const betaNext = choleskySolve(A, b);

  let ll = 0;
  // φ for Gamma/Normal not yet known at this iteration; pass 1 (deviance is
  // φ-independent for Bernoulli/Poisson, and for Gamma/Normal the per-step
  // log-lik is only used for the IRLSVisualizer readout).
  for (let i = 0; i < n; i++) ll += family.logLik(y[i], mu[i], 1);

  return { betaNext, mu, eta, weights, adjustedResponse: z, logLik: ll };
}

/**
 * IRLS / Fisher scoring solver for a GLM. Returns the full GLMFit object.
 *
 * Convergence is declared when ‖β^(t+1) - β^(t)‖_2 < tol. On non-convergence
 * (max-iter reached), returns a GLMFit with converged=false and a console
 * warning — does NOT throw. This is critical for `IRLSVisualizer` near-
 * separation preset (Bernoulli MLE diverges to ±∞ under quasi-complete
 * separation).
 */
export function glmFit(
  X: number[][],
  y: number[],
  family: GLMFamily,
  link?: LinkFunction,
  options?: {
    offset?: number[];
    maxIter?: number;
    tol?: number;
    startBeta?: number[];
  },
): GLMFit {
  const n = X.length;
  const m = X[0].length;
  const linkFn = link ?? family.canonicalLink;
  const maxIter = options?.maxIter ?? 25;
  const tol = options?.tol ?? 1e-8;
  const off = options?.offset ?? new Array<number>(n).fill(0);

  // Start: either user-provided β^(0), or g(initMu(y)) − offset projected onto X.
  let beta: number[];
  if (options?.startBeta && options.startBeta.length === m) {
    beta = options.startBeta.slice();
  } else {
    const mu0 = family.initMu(y);
    const eta0 = mu0.map((mi, i) => linkFn.g(mi) - off[i]);
    // Initial OLS-style solve to land β^(0) in the right neighborhood.
    try {
      const A0 = xtx(X);
      const b0 = xty(X, eta0);
      beta = choleskySolve(A0, b0);
    } catch {
      beta = new Array<number>(m).fill(0);
    }
  }

  let nIter = 0;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    nIter = iter + 1;
    let step;
    try {
      step = irlsStep(X, y, beta, family, linkFn, off);
    } catch (err) {
      // Cholesky failure → singular X^T W X (e.g. near separation). Bail with
      // converged=false rather than throw.
      // eslint-disable-next-line no-console
      console.warn(
        `glmFit: IRLS Cholesky failure at iter ${iter} (likely separation or rank-deficient design). Returning unconverged fit.`,
        err,
      );
      converged = false;
      break;
    }

    let diffSq = 0;
    for (let j = 0; j < m; j++) {
      const d = step.betaNext[j] - beta[j];
      diffSq += d * d;
    }
    beta = step.betaNext;

    if (Math.sqrt(diffSq) < tol) {
      converged = true;
      break;
    }
  }

  if (!converged) {
    // eslint-disable-next-line no-console
    console.warn(
      `glmFit: did not converge in ${maxIter} iterations (family=${family.name}, link=${linkFn.name}). ` +
        `Returning best-effort fit with converged=false.`,
    );
  }

  // Final μ, η at the converged β.
  const eta = new Array<number>(n);
  const mu = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    eta[i] = matVec([X[i]], beta)[0] + off[i];
    mu[i] = linkFn.gInv(eta[i]);
  }
  // Refresh weights at converged β.
  const weights = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const gp = linkFn.gPrime(mu[i]);
    const V = family.variance.V(mu[i]);
    weights[i] = 1 / Math.max(V * gp * gp, WEIGHT_CLIP_LO);
  }

  // Deviance (φ-free), phi (estimated for Gamma/Normal via Pearson statistic),
  // and null deviance (intercept-only refit).
  let dev = 0;
  for (let i = 0; i < n; i++) dev += family.devianceContribution(y[i], mu[i]);

  let phi = 1;
  if (!family.fixedDispersion) {
    let ssr = 0;
    for (let i = 0; i < n; i++) {
      const r = y[i] - mu[i];
      ssr += (r * r) / Math.max(family.variance.V(mu[i]), POSITIVE_CLIP);
    }
    phi = ssr / Math.max(n - m, 1);
  }

  // Null deviance: refit intercept-only model.
  let nullDeviance = dev;
  if (m > 1) {
    const Xnull = X.map(() => [1]);
    try {
      const nullFit = glmFit(Xnull, y, family, linkFn, {
        offset: off.slice(),
        maxIter,
        tol,
      });
      nullDeviance = nullFit.deviance;
    } catch {
      // Intercept-only refit can rarely fail; fall back to NaN-safe value.
      nullDeviance = NaN;
    }
  }

  // Asymptotic variance-covariance: φ · (X^T W X)^{-1}.
  let vcov: number[][];
  try {
    const A = xtwx(X, weights);
    const Ainv = choleskyInverse(A);
    vcov = Ainv.map((row) => row.map((v) => v * phi));
  } catch {
    vcov = Array.from({ length: m }, () => new Array<number>(m).fill(NaN));
  }

  return {
    beta,
    mu,
    eta,
    weights,
    deviance: dev,
    nullDeviance,
    phi,
    nIter,
    converged,
    X,
    y,
    family,
    link: linkFn,
    vcov,
    offset: options?.offset ?? null,
  };
}

/** Predict μ (or η, with returnEta=true) at new x via the fitted link. */
export function predictGLM(
  fit: GLMFit,
  Xnew: number[][],
  options?: { offsetNew?: number[]; returnEta?: boolean },
): number[] {
  const n = Xnew.length;
  const off = options?.offsetNew ?? new Array<number>(n).fill(0);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const eta = matVec([Xnew[i]], fit.beta)[0] + off[i];
    out[i] = options?.returnEta ? eta : fit.link.gInv(eta);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.J — Deviance and residuals (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────

/** D = 2φ[ℓ_sat - ℓ(β̂)]. Stored on the fit; this getter exists for API
 *  symmetry with `pearsonResiduals` / `devianceResiduals`. */
export function deviance(fit: GLMFit): number {
  return fit.deviance;
}

/** Pearson residuals: r_i^P = (y_i - μ_i) / sqrt(V(μ_i) · φ). */
export function pearsonResiduals(fit: GLMFit): number[] {
  return fit.y.map((yi, i) => {
    const V = fit.family.variance.V(fit.mu[i]);
    return (yi - fit.mu[i]) / Math.sqrt(Math.max(V * fit.phi, POSITIVE_CLIP));
  });
}

/** Deviance residuals: r_i^D = sign(y - μ) · sqrt(d_i). */
export function devianceResiduals(fit: GLMFit): number[] {
  return fit.y.map((yi, i) => {
    const di = Math.max(0, fit.family.devianceContribution(yi, fit.mu[i]));
    return Math.sign(yi - fit.mu[i]) * Math.sqrt(di);
  });
}

export interface DevianceTestResult {
  devianceFull: number;
  devianceReduced: number;
  diff: number;
  df: number;
  pValue: number;
  reject: boolean;
  dispersionEstimated: boolean;
}

/**
 * Nested-model deviance test. Fits both models, returns (D_0 - D_1), df=k,
 * p-value. For fixed-φ families (Bernoulli, Poisson) uses χ²_k reference.
 * For estimated-φ families (Gamma, Normal), divides by the full-model φ̂ and
 * uses an F_{k, n-p-1} reference.
 */
export function devianceTestNested(
  Xfull: number[][],
  Xreduced: number[][],
  y: number[],
  family: GLMFamily,
  link: LinkFunction,
  alpha: number,
  options?: { offset?: number[] },
): DevianceTestResult {
  const fitFull = glmFit(Xfull, y, family, link, { offset: options?.offset });
  const fitRed = glmFit(Xreduced, y, family, link, { offset: options?.offset });
  const diff = fitRed.deviance - fitFull.deviance;
  const k = Xfull[0].length - Xreduced[0].length;

  let pValue: number;
  if (family.fixedDispersion) {
    pValue = 1 - chiSquaredCDF(diff, k);
  } else {
    const phi = fitFull.phi;
    const fStat = diff / k / phi;
    const df1 = k;
    const df2 = y.length - Xfull[0].length;
    pValue = 1 - fCDF(fStat, df1, df2);
  }

  return {
    devianceFull: fitFull.deviance,
    devianceReduced: fitRed.deviance,
    diff,
    df: k,
    pValue,
    reject: pValue < alpha,
    dispersionEstimated: !family.fixedDispersion,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.K — Sandwich variance + profile-likelihood CI (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────

export type SandwichType = 'HC0' | 'HC1' | 'HC2' | 'HC3';

/**
 * Sandwich variance estimator A^{-1} B A^{-1} for a fitted GLM.
 *
 * - **HC0** (White 1980): B_HC0 = X^T diag(r_i²) X with r_i = y_i - μ_i.
 * - **HC1** (MacKinnon-White 1985): scale HC0 by n/(n-p-1).
 * - **HC2**: divide r_i² by (1 - h_ii).
 * - **HC3**: divide r_i² by (1 - h_ii)². Leverage h_ii capped at 0.9999 with
 *   a console warning (G5 numerical stability).
 *
 * A is the GLM information matrix X^T W X (Fisher), so A^{-1} is the inverse
 * of the φ-free vcov. Under correct mean-specification + arbitrary variance
 * misspecification, this is consistent for the QMLE asymptotic variance
 * (Huber 1967, White 1980, Wedderburn 1974).
 */
export function sandwichVCov(fit: GLMFit, type: SandwichType = 'HC3'): number[][] {
  const n = fit.X.length;
  const m = fit.X[0].length;

  // A = X^T W X (φ-free), Ainv from the stored vcov by un-scaling.
  const A = xtwx(fit.X, fit.weights);
  const Ainv = choleskyInverse(A);

  // Hat-equivalent leverages h_ii = (W^{1/2} X (X^T W X)^{-1} X^T W^{1/2})_{ii}.
  // For HC2/HC3 we need them; HC0/HC1 do not.
  let hatDiag: number[] = [];
  if (type === 'HC2' || type === 'HC3') {
    hatDiag = new Array(n).fill(0);
    let cappedCount = 0;
    for (let i = 0; i < n; i++) {
      const xi = fit.X[i];
      // h_ii = w_i · x_i^T A^{-1} x_i.
      let q = 0;
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let k = 0; k < m; k++) s += Ainv[j][k] * xi[k];
        q += xi[j] * s;
      }
      let hi = fit.weights[i] * q;
      if (hi > HC3_LEVERAGE_CAP) {
        hi = HC3_LEVERAGE_CAP;
        cappedCount++;
      }
      if (hi < 0) hi = 0;
      hatDiag[i] = hi;
    }
    if (cappedCount > 0) {
      // eslint-disable-next-line no-console
      console.warn(
        `sandwichVCov(${type}): clipped ${cappedCount} leverage values at ${HC3_LEVERAGE_CAP} (G5 numerical stability).`,
      );
    }
  }

  // B = X^T diag(s_i² · score_i²) X where score_i = w_i · (y_i - μ_i) · g'(μ_i)^{-1}
  // For canonical link the per-observation score becomes x_i (y_i - μ_i), and
  // we collect B = X^T diag((y_i - μ_i)² × adjustment) X.
  // For non-canonical, the working residual is r_i^W = (y_i - μ_i)/g'(μ_i),
  // i.e. the score per-obs contribution is x_i · w_i · g'(μ_i) · (y_i - μ_i).
  // Equivalent compact form for the empirical estimate: weight by
  // w_i² · g'(μ_i)² · (y_i - μ_i)² = (since w_i = 1/[V·g'²]) (y - μ)² / V².
  // Matching standard software (R `sandwich`), we use the working-residual form.
  const sScore = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const r = fit.y[i] - fit.mu[i];
    const gp = fit.link.gPrime(fit.mu[i]);
    // u_i = w_i · g'(μ_i) · r_i  is the per-observation score in β-space (up to x_i).
    const ui = fit.weights[i] * gp * r;
    sScore[i] = ui;
  }

  // Apply HC adjustment to s_i² then form B = X^T diag(s_i² adj) X.
  const adj = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let a = sScore[i] * sScore[i];
    if (type === 'HC2') a /= 1 - hatDiag[i];
    else if (type === 'HC3') {
      const denom = 1 - hatDiag[i];
      a /= denom * denom;
    }
    adj[i] = a;
  }
  const B: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    const xi = fit.X[i];
    const ai = adj[i];
    for (let j = 0; j < m; j++) {
      const xij = xi[j];
      for (let k = j; k < m; k++) B[j][k] += ai * xij * xi[k];
    }
  }
  for (let j = 0; j < m; j++) for (let k = 0; k < j; k++) B[j][k] = B[k][j];

  // Sandwich = Ainv · B · Ainv.
  const tmp: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += Ainv[i][k] * B[k][j];
      tmp[i][j] = s;
    }
  }
  const out: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += tmp[i][k] * Ainv[k][j];
      out[i][j] = s;
    }
  }
  // HC1 small-sample scale.
  if (type === 'HC1') {
    const scale = n / Math.max(n - m, 1);
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) out[i][j] *= scale;
  }
  return out;
}

/** Per-coefficient sandwich SE. Convenience for CI-drawing. */
export function sandwichSE(fit: GLMFit, type: SandwichType = 'HC3'): number[] {
  const V = sandwichVCov(fit, type);
  return V.map((row, j) => Math.sqrt(Math.max(row[j], 0)));
}

/** Sandwich-Wald CI for each coefficient at level 1-α. */
export function coefCIWaldSandwich(
  fit: GLMFit,
  alpha: number,
  type: SandwichType = 'HC3',
): Array<{ lower: number; upper: number; center: number }> {
  const ses = sandwichSE(fit, type);
  const z = standardNormalInvCDF(1 - alpha / 2);
  return fit.beta.map((b, j) => ({
    lower: b - z * ses[j],
    upper: b + z * ses[j],
    center: b,
  }));
}

/**
 * Profile-likelihood CI for a single coefficient β_j. The constrained-fit
 * trick: move β_j · X[:,j] into the offset, drop column j from X, refit. This
 * is exact (the constrained MLE on the reduced design with the right offset
 * IS the profile MLE). Solve for the two β_j values where
 *   D(β_j^0) - D(β̂) = χ²_{1, 1-α}
 * via bisection bracketed by ±gridBracket × Wald-SE around β̂_j.
 *
 * Renamed from the brief's `coefCIProfile` to avoid colliding with Topic 21's
 * existing OLS `coefCIProfile(fit, alpha): CIEntry[]` at line 641.
 */
export function coefCIProfileGLM(
  fit: GLMFit,
  j: number,
  alpha: number,
  options?: { maxBisect?: number; gridBracket?: number },
): { lower: number; upper: number; center: number } {
  const maxBisect = options?.maxBisect ?? 50;
  const gridBracket = options?.gridBracket ?? 8;
  const target = chiSquaredInvCDF(1 - alpha, 1);
  const center = fit.beta[j];
  const seWald = Math.sqrt(Math.max(fit.vcov[j][j], POSITIVE_CLIP));

  const profileDeviance = (bjFixed: number): number => {
    // Remove column j from X; absorb β_j × X[:,j] into the offset.
    const Xred = fit.X.map((row) => row.filter((_, k) => k !== j));
    const baseOff = fit.offset ?? new Array<number>(fit.X.length).fill(0);
    const newOff = baseOff.map((o, i) => o + bjFixed * fit.X[i][j]);
    const refit = glmFit(Xred, fit.y, fit.family, fit.link, {
      offset: newOff,
      // Warm-start drops the j-th coefficient.
      startBeta: fit.beta.filter((_, k) => k !== j),
    });
    return refit.deviance;
  };

  const baseDev = fit.deviance;
  const objective = (bj: number): number => profileDeviance(bj) - baseDev - target;

  // Bisect on the upper side: find b ∈ [center, center + gridBracket·seWald] where objective changes sign.
  const bisect = (lo: number, hi: number): number => {
    let a = lo;
    let b = hi;
    let fa = objective(a);
    let fb = objective(b);
    if (fa * fb > 0) {
      // Bracket failed — Wald-SE-based bracket too tight. Widen geometrically.
      for (let k = 0; k < 5 && fa * fb > 0; k++) {
        b = center + (b - center) * 2;
        fb = objective(b);
      }
      if (fa * fb > 0) {
        // Couldn't find sign change — return the widest bracket as fallback.
        return b;
      }
    }
    for (let it = 0; it < maxBisect; it++) {
      const mid = (a + b) / 2;
      const fm = objective(mid);
      if (Math.abs(fm) < 1e-6 || (b - a) / 2 < 1e-8) return mid;
      if (fa * fm < 0) {
        b = mid;
        fb = fm;
      } else {
        a = mid;
        fa = fm;
      }
    }
    return (a + b) / 2;
  };

  const upper = bisect(center, center + gridBracket * seWald);
  const lower = (() => {
    const a0 = center - gridBracket * seWald;
    const b0 = center;
    let a = a0;
    let b = b0;
    let fa = objective(a);
    let fb = objective(b);
    for (let k = 0; k < 5 && fa * fb > 0; k++) {
      a = center - (center - a) * 2;
      fa = objective(a);
    }
    if (fa * fb > 0) return a;
    for (let it = 0; it < maxBisect; it++) {
      const mid = (a + b) / 2;
      const fm = objective(mid);
      if (Math.abs(fm) < 1e-6 || (b - a) / 2 < 1e-8) return mid;
      if (fa * fm < 0) {
        b = mid;
        fb = fm;
      } else {
        a = mid;
        fa = fm;
      }
    }
    return (a + b) / 2;
  })();

  return { lower, upper, center };
}

// ─────────────────────────────────────────────────────────────────────────────
// §6.1.L — Simulation harness for GLM DGPs (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Simulate a response vector from a given GLM family, link, and coefficient.
 * Bernoulli / Poisson / Gamma / Normal supported via the convergence.ts
 * sampler primitives (bernoulliSample, poissonSample, gammaSample,
 * normalSample). The Gamma sampler uses a shape ν = 1/dispersion; pass
 * dispersion ∈ (0, ∞) (default 1).
 */
export function simulateGLM(
  X: number[][],
  beta: number[],
  family: GLMFamily,
  link: LinkFunction,
  seed: number,
  options?: { offset?: number[]; dispersion?: number },
): number[] {
  const n = X.length;
  const off = options?.offset ?? new Array<number>(n).fill(0);
  const phi = options?.dispersion ?? 1;
  const rng = seededRandom(seed);
  const eta = matVec(X, beta).map((v, i) => v + off[i]);
  const mu = eta.map((e) => link.gInv(e));
  const out = new Array<number>(n);
  switch (family.name) {
    case 'bernoulli':
      for (let i = 0; i < n; i++) out[i] = bernoulliSample(clip(mu[i], 0, 1), rng);
      break;
    case 'poisson':
      for (let i = 0; i < n; i++) out[i] = poissonSample(Math.max(mu[i], POSITIVE_CLIP), rng);
      break;
    case 'gamma': {
      const nu = 1 / Math.max(phi, POSITIVE_CLIP); // shape parameter
      for (let i = 0; i < n; i++) {
        const m = Math.max(mu[i], POSITIVE_CLIP);
        // Gamma(shape=ν, rate=ν/μ) ⇒ E = μ, Var = μ²/ν = μ²·φ.
        out[i] = gammaSample(nu, nu / m, rng);
      }
      break;
    }
    case 'normal':
      for (let i = 0; i < n; i++) out[i] = normalSample(mu[i], Math.sqrt(phi), rng);
      break;
  }
  return out;
}

/**
 * Simulate overdispersed Poisson via a Negative-Binomial-equivalent trick:
 * Y_i | λ_i ~ Poisson(λ_i), λ_i ~ Gamma(shape=μ_i/(r-1), rate=1/(r-1))
 * gives E[Y_i] = μ_i and Var(Y_i) = μ_i + μ_i²·(r-1)/μ_i = μ_i · r.
 *
 * For r = 1 (no overdispersion), reduces to plain Poisson.
 */
export function simulateOverdispersedPoisson(
  X: number[][],
  beta: number[],
  dispersionRatio: number,
  seed: number,
  options?: { offset?: number[] },
): number[] {
  const n = X.length;
  const off = options?.offset ?? new Array<number>(n).fill(0);
  const rng = seededRandom(seed);
  const eta = matVec(X, beta).map((v, i) => v + off[i]);
  const mu = eta.map((e) => Math.exp(e));
  const out = new Array<number>(n);
  if (Math.abs(dispersionRatio - 1) < 1e-10) {
    for (let i = 0; i < n; i++) out[i] = poissonSample(Math.max(mu[i], POSITIVE_CLIP), rng);
    return out;
  }
  // Negative-Binomial-style mixture.
  const k = dispersionRatio - 1;
  for (let i = 0; i < n; i++) {
    const m = Math.max(mu[i], POSITIVE_CLIP);
    const shape = m / k;
    const rate = 1 / k;
    const lambda = gammaSample(shape, rate, rng);
    out[i] = poissonSample(Math.max(lambda, POSITIVE_CLIP), rng);
  }
  return out;
}

/**
 * Monte-Carlo coverage estimator for a GLM CI constructor under either the
 * correctly-specified DGP (default) or an arbitrary user-supplied
 * `misspecify` simulator. Returns per-coefficient empirical coverage.
 */
export function coverageSimulatorGLM(
  X: number[][],
  betaTrue: number[],
  family: GLMFamily,
  link: LinkFunction,
  iterations: number,
  ciConstructor: (
    fit: GLMFit,
    alpha: number,
  ) => Array<{ lower: number; upper: number }>,
  alpha: number,
  seed: number,
  options?: {
    misspecify?: (X: number[][], beta: number[], seed: number) => number[];
    offset?: number[];
  },
): number[] {
  const m = betaTrue.length;
  const hits = new Array<number>(m).fill(0);
  for (let iter = 0; iter < iterations; iter++) {
    const y = options?.misspecify
      ? options.misspecify(X, betaTrue, seed + iter)
      : simulateGLM(X, betaTrue, family, link, seed + iter, { offset: options?.offset });
    let fit: GLMFit;
    try {
      fit = glmFit(X, y, family, link, { offset: options?.offset });
    } catch {
      continue; // fit failure → don't count toward iteration total
    }
    if (!fit.converged) continue;
    const cis = ciConstructor(fit, alpha);
    for (let j = 0; j < m; j++) {
      if (cis[j].lower <= betaTrue[j] && betaTrue[j] <= cis[j].upper) hits[j]++;
    }
  }
  return hits.map((h) => h / iterations);
}
