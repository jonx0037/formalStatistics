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
} from './testing';
import { seededRandom } from './probability';
import { normalSample } from './convergence';

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
  if (n === 0) throw new Error('qrSolve: empty design matrix');
  const m = X[0].length;
  if (m === 0) throw new Error('qrSolve: zero-column design matrix');
  if (n < m) {
    throw new Error(`qrSolve: underdetermined system (n=${n} < p+1=${m})`);
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
        `qrSolve: rank-deficient design matrix (column ${k} has near-zero norm)`,
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

/** Internally studentized residuals: e_i / (σ̂ √(1 - h_ii)). */
export function studentizedResiduals(fit: OLSFit, X: number[][]): number[] {
  const h = leverage(X);
  const sigma = Math.sqrt(fit.sigmaSquared);
  return fit.residuals.map((ei, i) => {
    const denom = sigma * Math.sqrt(Math.max(1 - h[i], 1e-12));
    return ei / denom;
  });
}

/**
 * Cook's distance: D_i = (r_i² / (p+1)) * (h_ii / (1 - h_ii)), where r_i is
 * the internally studentized residual. Flags observations whose removal would
 * substantially shift the fit.
 */
export function cooksDistance(fit: OLSFit, X: number[][]): number[] {
  const h = leverage(X);
  const r = studentizedResiduals(fit, X);
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
 * Non-central F_{df1, df2}(λ) density at x via Poisson-mixture series:
 *   f(x; df1, df2, λ) = Σ_j Poisson(λ/2; j) · (df1 / (df1 + 2j))
 *                       · f_F(x df1/(df1+2j); df1 + 2j, df2)
 * Truncates when the running Poisson weight drops below 1e-12 (plus a hard
 * cap at j = 10⁴). For large λ the mode of the Poisson is near λ/2, so
 * centering the truncation window there would be faster, but the naive
 * forward sweep is adequate for λ ≤ 200 which covers every application in
 * Topic 21's components.
 */
export function noncentralFDensity(
  x: number,
  df1: number,
  df2: number,
  lambda: number,
): number {
  if (x <= 0 || df1 <= 0 || df2 <= 0 || lambda < 0) return 0;
  if (lambda === 0) return fPDF(x, df1, df2);

  const halfLambda = lambda / 2;
  let weight = Math.exp(-halfLambda); // Poisson(λ/2; 0)
  let total = 0;
  const maxJ = 10000;
  const eps = 1e-12;
  for (let j = 0; j < maxJ; j++) {
    const df1Eff = df1 + 2 * j;
    const xEff = (x * df1) / df1Eff;
    const term = weight * (df1 / df1Eff) * fPDF(xEff, df1Eff, df2);
    total += term;
    // Forward recursion: Poisson(λ/2; j+1) = Poisson(λ/2; j) · (λ/2) / (j+1).
    weight *= halfLambda / (j + 1);
    // Break once the remaining Poisson tail is negligible AND we've passed the mode.
    if (weight < eps && j > halfLambda) break;
  }
  return total;
}

/**
 * Non-central F CDF at x via the same Poisson-mixture representation —
 *   F(x; df1, df2, λ) = Σ_j Poisson(λ/2; j) · F_F(x df1/(df1+2j); df1 + 2j, df2)
 * Truncation rule matches noncentralFDensity.
 */
function noncentralFCDF(
  x: number,
  df1: number,
  df2: number,
  lambda: number,
): number {
  if (x <= 0 || df1 <= 0 || df2 <= 0 || lambda < 0) return 0;
  if (lambda === 0) return fCDF(x, df1, df2);

  const halfLambda = lambda / 2;
  let weight = Math.exp(-halfLambda);
  let total = 0;
  const maxJ = 10000;
  const eps = 1e-12;
  for (let j = 0; j < maxJ; j++) {
    const df1Eff = df1 + 2 * j;
    const xEff = (x * df1) / df1Eff;
    total += weight * fCDF(xEff, df1Eff, df2);
    weight *= halfLambda / (j + 1);
    if (weight < eps && j > halfLambda) break;
  }
  return total;
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
 * Uses the Topic-17 seeded Mersenne Twister (`seededRandom`) and Box–Muller
 * normal sampling (`normalSample`) for reproducibility.
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
