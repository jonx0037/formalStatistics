/**
 * polynomial.ts — shared 1-D polynomial utilities used by Topic 24's
 * IC components (ICSelector, CVvsICComparator) and any other visualization
 * that needs polynomial design construction, scalar evaluation, or QR-based
 * fitting that can survive monomial Vandermonde ill-conditioning at high d.
 *
 * Why a dedicated module: regression.ts is general-purpose (any design
 * matrix); polynomial-specific helpers (polyDesign, polyEval) don't belong
 * there. polyFitOLS is the QR-based OLS fit that returns an OLSFit-shaped
 * object — used wherever olsFit's choleskyInverse-on-(XᵀX) step is fragile
 * (notably d ≥ 12 monomial designs; the notebook uses numpy's SVD-based
 * lstsq to sidestep the same issue).
 */

import { olsFit, qrSolve, type OLSFit } from './regression';

/**
 * Build a degree-`d` polynomial design matrix `[1, x, x², …, x^d]` of shape
 * `(n, d+1)`. Column 0 is the intercept.
 */
export function polyDesign(x: number[], d: number): number[][] {
  return x.map((xi) => {
    const row = new Array<number>(d + 1);
    let pwr = 1;
    for (let j = 0; j <= d; j++) {
      row[j] = pwr;
      pwr *= xi;
    }
    return row;
  });
}

/**
 * Scalar evaluation of a polynomial with coefficients `beta` (low → high)
 * at `xNew`. Mirrors `polyDesign(.).dot(beta)` for a single point.
 */
export function polyEval(beta: number[], xNew: number, d: number): number {
  let yhat = 0;
  let pwr = 1;
  for (let j = 0; j <= d; j++) {
    yhat += beta[j] * pwr;
    pwr *= xNew;
  }
  return yhat;
}

/**
 * Polynomial OLS fit with two-tier strategy:
 *   1. Try the fully-instrumented `olsFit` first. If it succeeds, the result
 *      has every OLSFit field populated correctly (`xtxInv` from
 *      choleskyInverse, `rSquared` / `adjustedRSquared` / `sst` / `ssr` from
 *      the standard formulas) and downstream helpers like `coefSE` work
 *      without surprise. This is the path for d ≤ ~11 on most x-domains.
 *   2. If `olsFit` throws because `choleskyInverse(XᵀX)` failed the
 *      positive-definite check (the monomial Vandermonde at d ≥ 12 hits
 *      this), fall back to a QR-based fit. `xtxInv` is left as an empty
 *      array in this branch — callers that use it (e.g., coefSE) MUST
 *      detect `xtxInv.length === 0` and either error out or compute a
 *      QR-based variance estimator. The IC consumers (aic, aicc, bic,
 *      mallowsCp, nestedICRanking) only read beta / sse / n /
 *      sigmaSquared, so they work in either branch.
 *
 * Pre-flight: throws when n ≤ d + 1 (no residual degrees of freedom). This
 * matches `olsFit`'s behaviour on saturated designs and prevents downstream
 * `aic` / `bic` calls from quietly returning -∞ when `log(SSE/n)` underflows
 * on a perfect-fit polynomial.
 *
 * @param x  predictor vector (length n)
 * @param y  response vector (length n)
 * @param d  polynomial degree
 */
export function polyFitOLS(x: number[], y: number[], d: number): OLSFit {
  const n = x.length;
  const m = d + 1;
  if (n <= m) {
    throw new Error(
      `polyFitOLS: residual df = ${n - m} ≤ 0 (n=${n}, d+1=${m}); ` +
        `polynomial fit is unidentifiable`,
    );
  }
  const X = polyDesign(x, d);
  // Tier 1: fully-instrumented olsFit. Succeeds for low-d designs where
  // XᵀX is well-conditioned enough for Cholesky.
  try {
    return olsFit(X, y);
  } catch (e) {
    if (!String((e as Error).message).includes('positive-definite')) {
      throw e;
    }
    // Tier 2: QR fallback. Cholesky failed because the monomial Vandermonde
    // is too ill-conditioned at this d. QR (via Householder reflections in
    // qrSolve) is the standard SVD-equivalent fallback.
  }
  const beta = qrSolve(X, y);
  const fitted = new Array<number>(n);
  let sse = 0;
  for (let i = 0; i < n; i++) {
    let yhat = 0;
    for (let j = 0; j <= d; j++) yhat += beta[j] * X[i][j];
    fitted[i] = yhat;
    sse += (y[i] - yhat) * (y[i] - yhat);
  }
  const residuals = y.map((yi, i) => yi - fitted[i]);
  const sigmaSquared = sse / (n - m);
  // Standard SST/SSR/R² computations from the raw response vector — no
  // dependence on xtxInv, so these stay valid in the QR fallback.
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  let sst = 0;
  for (const yi of y) sst += (yi - yMean) * (yi - yMean);
  const ssr = sst - sse;
  const rSquared = sst > 0 ? 1 - sse / sst : 1;
  const adjustedRSquared = sst > 0 ? 1 - ((1 - rSquared) * (n - 1)) / (n - m) : rSquared;
  return {
    beta,
    residuals,
    fitted,
    sigmaSquared,
    sse,
    sst,
    ssr,
    // xtxInv intentionally [] for the QR fallback — downstream helpers that
    // need it (coefSE, workingHotellingBand, …) must check `length === 0`
    // and either error or compute a QR-based variance themselves. polyFitOLS
    // is intended for IC computation, not coefficient-level inference at
    // d ≥ 12.
    xtxInv: [],
    rSquared,
    adjustedRSquared,
    n,
    p: d,
  };
}
