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

import { qrSolve, type OLSFit } from './regression';

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
 * QR-based polynomial OLS fit. Returns the OLSFit-shape fields that the
 * `aic` / `aicc` / `bic` / `mallowsCp` / `nestedICRanking` consumers actually
 * read (`beta`, `sse`, `n`, `sigmaSquared`); the remaining OLSFit fields
 * (`xtxInv`, `rSquared`, `adjustedRSquared`, etc.) are populated with stub
 * defaults since they're not used by the IC framework.
 *
 * QR-based instead of olsFit's Cholesky path: the monomial Vandermonde
 * design at d ≥ 12 has a near-singular XᵀX that fails Cholesky's
 * positive-definite check; QR is the standard SVD-equivalent fallback.
 *
 * @param x  predictor vector (length n)
 * @param y  response vector (length n)
 * @param d  polynomial degree
 */
export function polyFitOLS(x: number[], y: number[], d: number): OLSFit {
  const X = polyDesign(x, d);
  const beta = qrSolve(X, y);
  const fitted = new Array<number>(x.length);
  let sse = 0;
  for (let i = 0; i < x.length; i++) {
    let yhat = 0;
    for (let j = 0; j <= d; j++) yhat += beta[j] * X[i][j];
    fitted[i] = yhat;
    sse += (y[i] - yhat) * (y[i] - yhat);
  }
  const residuals = y.map((yi, i) => yi - fitted[i]);
  const n = x.length;
  const m = d + 1;
  const sigmaSquared = n > m ? sse / (n - m) : sse / n;
  return {
    beta,
    residuals,
    fitted,
    sigmaSquared,
    sse,
    sst: 0,
    ssr: 0,
    xtxInv: [] as number[][],
    rSquared: 0,
    adjustedRSquared: 0,
    n,
    p: d,
  };
}
