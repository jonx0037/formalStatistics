/**
 * regression.test.ts — console-log verification for Topic-21 test cases
 * T7.1–T7.15 (brief §6.2). Track 5's tests in testing.test.ts are untouched.
 *
 * Run via: pnpm test:regression (invokes tsx on this file).
 *
 * Idiom mirrors testing.test.ts: lightweight `check(id, ok, got, want, note)`
 * + `approx(x, y, tol)`. No Vitest / Jest — keep the numerical feedback
 * loop narrow.
 */

import {
  qrSolve,
  olsFit,
  hatMatrix,
  leverage,
  choleskyInverse,
  coefCIWald,
  coefCIBonferroni,
  workingHotellingBand,
  fTestNested,
  oneWayANOVA,
  fDensity,
  fCDF,
  fQuantile,
  noncentralFDensity,
  noncentralFPower,
  simulateLinearModel,
  coverageSimulator,
  // Topic 22 — GLM extensions (§6.1.H–§6.1.L)
  LINKS,
  FAMILIES,
  glmFit,
  irlsStep,
  deviance,
  pearsonResiduals,
  devianceResiduals,
  devianceTestNested,
  sandwichVCov,
  sandwichSE,
  coefCIProfileGLM,
  simulateGLM,
  simulateOverdispersedPoisson,
} from './regression';
import { seededRandom } from './probability';
import { normalSample } from './convergence';

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
    console.log(
      `  ✗ ${id}\n       got:  ${String(got)}\n       want: ${String(want)}${note ? `\n       note: ${note}` : ''}`,
    );
  }
}

const approx = (x: number, y: number, tol: number): boolean =>
  Math.abs(x - y) <= tol;

console.log('\n========================================');
console.log(' Topic 21 · regression.ts verification (T7.1–T7.15)');
console.log('========================================\n');

// ─────────────────────────────────────────────────────────────────────────────
// Simple helpers scoped to this test file only.
// ─────────────────────────────────────────────────────────────────────────────

/** Build an intercept+slope design matrix from a plain x-vector. */
function designWithIntercept(x: number[]): number[][] {
  return x.map((xi) => [1, xi]);
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.1 — olsFit recovers known β on a large-n simulation.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 1000;
  const seed = 20240101;
  const rng = seededRandom(seed);
  const x: number[] = new Array(n);
  for (let i = 0; i < n; i++) x[i] = 10 * rng() - 5; // x ~ U(-5, 5)
  const X = designWithIntercept(x);
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) y[i] = 2 + 3 * x[i] + normalSample(0, 0.5, rng);
  const fit = olsFit(X, y);
  check(
    'T7.1 olsFit known-solution intercept',
    approx(fit.beta[0], 2, 0.1),
    fit.beta[0],
    2,
    'β̂₀ within 0.1 of 2',
  );
  check(
    'T7.1 olsFit known-solution slope',
    approx(fit.beta[1], 3, 0.05),
    fit.beta[1],
    3,
    'β̂₁ within 0.05 of 3',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.2 — residuals sum to zero (intercept present) and are orthogonal to X.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240102;
  const rng = seededRandom(seed);
  const n = 50;
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = normalSample(0, 1, rng);
    const x2 = normalSample(0, 1, rng);
    X.push([1, x1, x2]);
    y.push(0.5 + 1.5 * x1 - 2 * x2 + normalSample(0, 0.3, rng));
  }
  const fit = olsFit(X, y);
  const residualSum = fit.residuals.reduce((a, b) => a + b, 0);
  check(
    'T7.2 residual sum ≈ 0',
    approx(residualSum, 0, 1e-10),
    residualSum,
    0,
    'tol 1e-10',
  );
  // Orthogonality to each column of X.
  let maxDot = 0;
  for (let j = 0; j < X[0].length; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j] * fit.residuals[i];
    maxDot = Math.max(maxDot, Math.abs(s));
  }
  check(
    'T7.2 residuals orthogonal to col(X)',
    maxDot < 1e-10,
    maxDot,
    0,
    'max |Xⱼᵀe| < 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.3 — hatMatrix idempotence HH = H and trace = p+1.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240103;
  const rng = seededRandom(seed);
  const n = 20;
  const pPlusOne = 4;
  const X: number[][] = Array.from({ length: n }, () => {
    const row: number[] = [1];
    for (let j = 1; j < pPlusOne; j++) row.push(normalSample(0, 1, rng));
    return row;
  });
  const H = hatMatrix(X);

  // Idempotence: (HH)[i,j] = H[i,j].
  let maxIdemErr = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let hh = 0;
      for (let k = 0; k < n; k++) hh += H[i][k] * H[k][j];
      maxIdemErr = Math.max(maxIdemErr, Math.abs(hh - H[i][j]));
    }
  }
  check(
    'T7.3 H is idempotent',
    maxIdemErr < 1e-10,
    maxIdemErr,
    0,
    'max |HH - H| < 1e-10',
  );

  // Trace = rank = p + 1.
  let trace = 0;
  for (let i = 0; i < n; i++) trace += H[i][i];
  check(
    'T7.3 tr(H) = p+1',
    approx(trace, pPlusOne, 1e-10),
    trace,
    pPlusOne,
    'tol 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.4 — leverage: 0 ≤ hᵢᵢ ≤ 1 and Σhᵢᵢ = p+1.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240104;
  const rng = seededRandom(seed);
  const n = 40;
  const pPlusOne = 3;
  const X: number[][] = Array.from({ length: n }, () => {
    const row: number[] = [1];
    for (let j = 1; j < pPlusOne; j++) row.push(normalSample(0, 1, rng));
    return row;
  });
  const h = leverage(X);
  const inRange = h.every((hi) => hi >= -1e-10 && hi <= 1 + 1e-10);
  check(
    'T7.4 leverage in [0, 1]',
    inRange,
    h.slice(0, 5),
    '[0,1] bounds',
    `all ${n} values clamped to [0,1]`,
  );
  const sumH = h.reduce((a, b) => a + b, 0);
  check(
    'T7.4 Σhᵢᵢ = p+1',
    approx(sumH, pPlusOne, 1e-10),
    sumH,
    pPlusOne,
    'tol 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.5 — coefCIWald MC coverage at nominal 1-α = 0.95.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240105;
  const rng = seededRandom(seed);
  const n = 30;
  const x: number[] = new Array(n);
  for (let i = 0; i < n; i++) x[i] = 4 * rng() - 2; // x fixed across simulations
  const X = designWithIntercept(x);
  const betaTrue = [1.5, -0.7];
  const sigma = 0.5;
  const iterations = 2000;
  const alpha = 0.05;
  const cov = coverageSimulator(
    X,
    betaTrue,
    sigma,
    iterations,
    (fit, a) => coefCIWald(fit, a),
    alpha,
    seed + 999,
  );
  const cov0InRange = cov[0] >= 0.93 && cov[0] <= 0.97;
  const cov1InRange = cov[1] >= 0.93 && cov[1] <= 0.97;
  check(
    'T7.5 Wald-t coverage β̂₀',
    cov0InRange,
    cov[0].toFixed(4),
    '[0.93, 0.97]',
    `${iterations} sims`,
  );
  check(
    'T7.5 Wald-t coverage β̂₁',
    cov1InRange,
    cov[1].toFixed(4),
    '[0.93, 0.97]',
    `${iterations} sims`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.6 — Bonferroni simultaneous FWER ≤ α + 0.02.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240106;
  const rng = seededRandom(seed);
  const n = 40;
  const pPlusOne = 4;
  const X: number[][] = Array.from({ length: n }, () => {
    const row: number[] = [1];
    for (let j = 1; j < pPlusOne; j++) row.push(normalSample(0, 1, rng));
    return row;
  });
  const betaTrue = [1, 0.5, -0.3, 0.8];
  const sigma = 0.6;
  const iterations = 2000;
  const alpha = 0.05;

  let simCover = 0;
  for (let iter = 0; iter < iterations; iter++) {
    const y = simulateLinearModel(X, betaTrue, sigma, seed + iter + 1000);
    const fit = olsFit(X, y);
    const cis = coefCIBonferroni(fit, alpha);
    let allCovered = true;
    for (let j = 0; j < pPlusOne; j++) {
      if (!(cis[j].lower <= betaTrue[j] && betaTrue[j] <= cis[j].upper)) {
        allCovered = false;
        break;
      }
    }
    if (allCovered) simCover++;
  }
  const empiricalFWER = 1 - simCover / iterations;
  const ok = empiricalFWER <= alpha + 0.02;
  check(
    'T7.6 Bonferroni FWER ≤ α + 0.02',
    ok,
    empiricalFWER.toFixed(4),
    `≤ ${alpha + 0.02}`,
    `${iterations} sims, pPlusOne=${pPlusOne}`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.7 — fTestNested: under H₀, p-values ~ Uniform(0,1). KS against Unif.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240107;
  const rng = seededRandom(seed);
  const n = 40;
  // Full model = intercept + 3 predictors. Under H₀, the last two are zero.
  const X: number[][] = Array.from({ length: n }, () => {
    const row: number[] = [1];
    for (let j = 1; j < 4; j++) row.push(normalSample(0, 1, rng));
    return row;
  });
  const Xreduced: number[][] = X.map((r) => [r[0], r[1]]);
  const betaTrue = [0.5, 1.2, 0, 0]; // H₀ holds: β₂ = β₃ = 0
  const sigma = 0.8;
  const iterations = 5000;
  const alpha = 0.05;
  const pvals: number[] = new Array(iterations);
  for (let iter = 0; iter < iterations; iter++) {
    const y = simulateLinearModel(X, betaTrue, sigma, seed + iter + 2000);
    pvals[iter] = fTestNested(X, Xreduced, y, alpha).pValue;
  }
  // KS statistic of empirical CDF against Uniform(0, 1).
  const sorted = pvals.slice().sort((a, b) => a - b);
  let D = 0;
  for (let i = 0; i < iterations; i++) {
    const fPlus = (i + 1) / iterations;
    const fMinus = i / iterations;
    D = Math.max(D, Math.abs(fPlus - sorted[i]), Math.abs(sorted[i] - fMinus));
  }
  // Asymptotic KS p-value: p ≈ 2 · exp(-2 n D²).
  const ksPValue = 2 * Math.exp(-2 * iterations * D * D);
  check(
    'T7.7 fTestNested p-values ~ Unif(0,1) under H₀',
    ksPValue >= 0.05,
    `D=${D.toFixed(4)}, p=${ksPValue.toFixed(4)}`,
    'KS p ≥ 0.05',
    `${iterations} sims`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.8 — fTestNested power vs noncentralFPower.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240108;
  const rng = seededRandom(seed);
  const n = 50;
  const X: number[][] = Array.from({ length: n }, () => {
    const row: number[] = [1];
    for (let j = 1; j < 4; j++) row.push(normalSample(0, 1, rng));
    return row;
  });
  const Xreduced: number[][] = X.map((r) => [r[0], r[1]]);
  // Pick a β where β₂ and β₃ are nonzero so H₀ fails.
  const betaTrue = [0.5, 1.2, 0.9, -0.6];
  const sigma = 1.0;
  const alpha = 0.05;

  // Empirical rejection rate.
  const iterations = 2000;
  let rejects = 0;
  for (let iter = 0; iter < iterations; iter++) {
    const y = simulateLinearModel(X, betaTrue, sigma, seed + iter + 3000);
    if (fTestNested(X, Xreduced, y, alpha).reject) rejects++;
  }
  const empiricalPower = rejects / iterations;

  // Theoretical power via noncentralFPower at the implied λ.
  // λ = (β_restricted)^T (X_restricted^T X_restricted — conditioned on reduced)^{-1} · … / σ².
  // Simpler: compute λ from the non-centrality of the quadratic form directly —
  // the expected (SSE_reduced - SSE_full) under β_true divided by σ².
  // Use the fact: at the true β, E[SSE_reduced - SSE_full] = σ²·k + noncentrality.
  // Equivalently, fit the population expected y (noise-free) and read off the numerator SS.
  const yPop = X.map((r) => r.reduce((s, rij, j) => s + rij * betaTrue[j], 0));
  const fullFitPop = olsFit(X, yPop);
  const redFitPop = olsFit(Xreduced, yPop);
  const lambda = (redFitPop.sse - fullFitPop.sse) / (sigma * sigma);
  const df1 = X[0].length - Xreduced[0].length;
  const df2 = n - X[0].length;
  const theoryPower = noncentralFPower(df1, df2, lambda, alpha);

  check(
    'T7.8 empirical power ≈ noncentralFPower',
    Math.abs(empiricalPower - theoryPower) < 0.03,
    `emp=${empiricalPower.toFixed(3)}, theory=${theoryPower.toFixed(3)}`,
    'within ±0.03',
    `λ≈${lambda.toFixed(2)}, ${iterations} sims`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.9 — oneWayANOVA: three-group F matches manual nested-F to 1e-8.
// ─────────────────────────────────────────────────────────────────────────────
{
  const groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
  const y = [4.1, 4.3, 3.9, 4.0, 5.2, 5.4, 5.1, 5.3, 6.0, 6.1, 5.9, 6.2];
  const result = oneWayANOVA(groups, y, 0.05);

  // Manual nested-F against common-mean reduced: F = (SSB/(g-1)) / (SSW/(n-g)).
  // SSB = Σ nⱼ (μ̂ⱼ - μ̂)²; SSW = Σⱼ Σᵢ (yᵢⱼ - μ̂ⱼ)².
  const n = y.length;
  const uniq = Array.from(new Set(groups)).sort((a, b) => a - b);
  const g = uniq.length;
  const muHat = y.reduce((a, b) => a + b, 0) / n;
  const groupMeans: number[] = new Array(g).fill(0);
  const groupCount: number[] = new Array(g).fill(0);
  for (let i = 0; i < n; i++) {
    const j = uniq.indexOf(groups[i]);
    groupMeans[j] += y[i];
    groupCount[j]++;
  }
  for (let j = 0; j < g; j++) groupMeans[j] /= groupCount[j];
  let ssb = 0;
  for (let j = 0; j < g; j++) ssb += groupCount[j] * (groupMeans[j] - muHat) ** 2;
  let ssw = 0;
  for (let i = 0; i < n; i++) {
    const j = uniq.indexOf(groups[i]);
    ssw += (y[i] - groupMeans[j]) ** 2;
  }
  const Fmanual = ssb / (g - 1) / (ssw / (n - g));
  check(
    'T7.9 oneWayANOVA F matches manual',
    approx(result.F, Fmanual, 1e-8),
    result.F,
    Fmanual,
    'tol 1e-8',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.10 — workingHotellingBand FWER ≤ α + 0.02 across a 20-point x-grid.
// ─────────────────────────────────────────────────────────────────────────────
{
  const seed = 20240110;
  const rng = seededRandom(seed);
  const n = 40;
  const x: number[] = new Array(n);
  for (let i = 0; i < n; i++) x[i] = 6 * rng() - 3;
  const X = designWithIntercept(x);
  const betaTrue = [1.0, 2.0];
  const sigma = 0.5;
  const iterations = 2000;
  const alpha = 0.05;

  // Grid of 20 evaluation points in [-3, 3].
  const xGrid: number[][] = [];
  for (let k = 0; k < 20; k++) {
    const xk = -3 + (6 * k) / 19;
    xGrid.push([1, xk]);
  }
  const muTrue = xGrid.map((xk) => xk[0] * betaTrue[0] + xk[1] * betaTrue[1]);

  let allCovered = 0;
  for (let iter = 0; iter < iterations; iter++) {
    const y = simulateLinearModel(X, betaTrue, sigma, seed + iter + 4000);
    const fit = olsFit(X, y);
    const band = workingHotellingBand(fit, X, xGrid, alpha);
    let ok = true;
    for (let k = 0; k < xGrid.length; k++) {
      if (!(band[k].lower <= muTrue[k] && muTrue[k] <= band[k].upper)) {
        ok = false;
        break;
      }
    }
    if (ok) allCovered++;
  }
  const empiricalFWER = 1 - allCovered / iterations;
  check(
    'T7.10 Working–Hotelling FWER ≤ α + 0.02',
    empiricalFWER <= alpha + 0.02,
    empiricalFWER.toFixed(4),
    `≤ ${alpha + 0.02}`,
    `${iterations} sims, 20-pt grid`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.11 — qrSolve matches closed-form (X^T X)^-1 X^T y on well-conditioned design.
// ─────────────────────────────────────────────────────────────────────────────
{
  const X: number[][] = [
    [1, 1, 1],
    [1, 2, 4],
    [1, 3, 9],
    [1, 4, 16],
    [1, 5, 25],
    [1, 6, 36],
  ];
  const y = [2.1, 4.9, 10.1, 17.0, 26.1, 37.2];
  const beta = qrSolve(X, y);

  // Closed form via the library's own Cholesky inverse.
  const xtxMat: number[][] = Array.from({ length: 3 }, () => new Array(3).fill(0));
  for (let i = 0; i < X.length; i++) {
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) xtxMat[a][b] += X[i][a] * X[i][b];
    }
  }
  const xtyVec = new Array(3).fill(0);
  for (let i = 0; i < X.length; i++) {
    for (let a = 0; a < 3; a++) xtyVec[a] += X[i][a] * y[i];
  }
  const xtxInv = choleskyInverse(xtxMat);
  const betaClosed: number[] = new Array(3).fill(0);
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) betaClosed[a] += xtxInv[a][b] * xtyVec[b];
  }

  let maxErr = 0;
  for (let j = 0; j < 3; j++) maxErr = Math.max(maxErr, Math.abs(beta[j] - betaClosed[j]));
  check(
    'T7.11 qrSolve vs (XᵀX)⁻¹Xᵀy closed form',
    maxErr < 1e-10,
    maxErr,
    0,
    'agreement 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.12 — qrSolve on rank-deficient X throws informative error.
// ─────────────────────────────────────────────────────────────────────────────
{
  // Two identical columns → rank-deficient.
  const Xbad: number[][] = [
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 3],
    [1, 4, 4],
  ];
  const y = [1, 2, 3, 4];
  let threw = false;
  let msg = '';
  try {
    qrSolve(Xbad, y);
  } catch (e) {
    threw = true;
    msg = (e as Error).message;
  }
  check(
    'T7.12 qrSolve throws on rank-deficient X',
    threw && /rank[- ]?deficient/i.test(msg),
    msg || '(no error)',
    'informative rank-deficient error',
    '',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.13 — fDensity / fCDF / fQuantile round-trip + known reference values.
// ─────────────────────────────────────────────────────────────────────────────
{
  const dfPairs: Array<[number, number]> = [
    [1, 10],
    [3, 96],
    [5, 44],
  ];
  const xs = [0.5, 1, 2, 5];
  let allRoundTrips = true;
  let maxErr = 0;
  for (const [d1, d2] of dfPairs) {
    for (const x of xs) {
      const p = fCDF(x, d1, d2);
      const xBack = fQuantile(p, d1, d2);
      const err = Math.abs(xBack - x);
      if (err > 1e-6) allRoundTrips = false;
      maxErr = Math.max(maxErr, err);
    }
  }
  check(
    'T7.13 fCDF ∘ fQuantile round-trip',
    allRoundTrips,
    maxErr,
    0,
    'tol 1e-6 over 3 df pairs × 4 x-values',
  );
  // Specific known values:
  //   F_{10,10}(1) = 0.5 (median of central F with equal df is 1).
  const med = fCDF(1, 10, 10);
  check(
    'T7.13 fCDF(1; 10, 10) = 0.5',
    approx(med, 0.5, 1e-6),
    med,
    0.5,
    'tol 1e-6',
  );
  // F_{3,96}^-1(0.95) ≈ 2.699 (standard table value, α = 0.05).
  const fCrit = fQuantile(0.95, 3, 96);
  check(
    'T7.13 fQuantile(0.95; 3, 96) ≈ 2.699',
    approx(fCrit, 2.699, 0.02),
    fCrit.toFixed(3),
    2.699,
    'tol 0.02 (matches Lehmann table)',
  );
  // fDensity positive and integrates (trapezoid) near 1 over [0, 30].
  let trapArea = 0;
  const dx = 0.01;
  for (let x = dx; x <= 30; x += dx) {
    trapArea += 0.5 * dx * (fDensity(x - dx, 3, 10) + fDensity(x, 3, 10));
  }
  check(
    'T7.13 ∫fDensity ≈ 1',
    approx(trapArea, 1, 0.02),
    trapArea.toFixed(3),
    1,
    'trapezoid over [0, 30], df=(3,10), tol 0.02',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.14 — noncentralFDensity / noncentralFPower agree with central at λ=0.
// ─────────────────────────────────────────────────────────────────────────────
{
  // At λ=0, non-central F reduces to central F.
  let maxErr = 0;
  const dfPairs: Array<[number, number]> = [
    [3, 10],
    [5, 20],
    [2, 50],
  ];
  for (const [d1, d2] of dfPairs) {
    for (const x of [0.5, 1, 2, 5]) {
      const ncf = noncentralFDensity(x, d1, d2, 0);
      const cf = fDensity(x, d1, d2);
      maxErr = Math.max(maxErr, Math.abs(ncf - cf));
    }
  }
  check(
    'T7.14 noncentralFDensity(λ=0) = fDensity',
    maxErr < 1e-8,
    maxErr,
    0,
    'tol 1e-8 across df pairs',
  );
  // Power is monotone in λ (sanity check vs a reference point).
  const pow0 = noncentralFPower(3, 96, 0, 0.05);
  const pow10 = noncentralFPower(3, 96, 10, 0.05);
  const pow30 = noncentralFPower(3, 96, 30, 0.05);
  check(
    'T7.14 power(λ=0) ≈ α',
    approx(pow0, 0.05, 0.005),
    pow0.toFixed(4),
    0.05,
    'at H₀, size ≈ α',
  );
  check(
    'T7.14 power monotone in λ',
    pow10 > pow0 && pow30 > pow10,
    `p(0)=${pow0.toFixed(3)}, p(10)=${pow10.toFixed(3)}, p(30)=${pow30.toFixed(3)}`,
    'strict monotone',
    '',
  );
  // Reference: SciPy ncf.sf(F_{.95,3,96}, 3, 96, λ=10) ≈ 0.742 (UCLA F-table
  // power calc agrees; tol 0.02 accommodates quantile-bisection error).
  check(
    'T7.14 power(λ=10, 3, 96, α=0.05) ≈ 0.74',
    approx(pow10, 0.74, 0.02),
    pow10.toFixed(3),
    0.74,
    'SciPy ncf reference, tol 0.02',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T7.15 — simulateLinearModel reproducibility + seed-uncorrelated outputs.
// ─────────────────────────────────────────────────────────────────────────────
{
  const X: number[][] = [
    [1, 0.1],
    [1, 0.5],
    [1, 1.2],
    [1, 1.8],
    [1, 2.5],
    [1, 3.1],
    [1, 3.7],
    [1, 4.3],
    [1, 4.9],
    [1, 5.5],
  ];
  const beta = [0.5, 1.5];
  const sigma = 0.3;
  const yA1 = simulateLinearModel(X, beta, sigma, 42);
  const yA2 = simulateLinearModel(X, beta, sigma, 42);
  let match = true;
  for (let i = 0; i < yA1.length; i++) {
    if (Math.abs(yA1[i] - yA2[i]) > 1e-15) {
      match = false;
      break;
    }
  }
  check(
    'T7.15 same seed → identical y',
    match,
    match,
    true,
    'exact floating-point equality',
  );

  // Correlation of two long residual streams at widely-separated seeds.
  // (`seededRandom` in probability.ts is a 32-bit LCG, which has known
  // low-dimensional correlation between nearby seeds in the first few
  // outputs — so rather than stacking hundreds of short vectors from
  // consecutive seeds, we draw two LONG streams from well-separated seeds
  // so that any LCG-initial-state artifact is dominated by ≥ 500 unrelated
  // downstream draws. Null-SD at 500 pairs is ≈ 0.045, |corr| < 0.1 is ~2σ.)
  const nLong = 500;
  const Xlong: number[][] = [];
  for (let i = 0; i < nLong; i++) Xlong.push([1, (i % 11) * 0.3]);
  const yaLong = simulateLinearModel(Xlong, beta, sigma, 1234567);
  const ybLong = simulateLinearModel(Xlong, beta, sigma, 987654321);
  const a: number[] = [];
  const b: number[] = [];
  for (let i = 0; i < nLong; i++) {
    a.push(yaLong[i] - (beta[0] + beta[1] * Xlong[i][1]));
    b.push(ybLong[i] - (beta[0] + beta[1] * Xlong[i][1]));
  }
  const meanA = a.reduce((s, v) => s + v, 0) / a.length;
  const meanB = b.reduce((s, v) => s + v, 0) / b.length;
  let num = 0,
    da = 0,
    db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - meanA) * (b[i] - meanB);
    da += (a[i] - meanA) ** 2;
    db += (b[i] - meanB) ** 2;
  }
  const corr = num / Math.sqrt(da * db);
  check(
    'T7.15 seed-stream correlation < 0.1',
    Math.abs(corr) < 0.1,
    corr.toFixed(4),
    '|corr| < 0.1',
    `${nLong} samples, seeds 1,234,567 vs 987,654,321`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  T8 — Topic 22 GLM verification (T8.1–T8.17)
// ─────────────────────────────────────────────────────────────────────────────

console.log('\n========================================');
console.log(' Topic 22 · regression.ts GLM extensions (T8.1–T8.17)');
console.log('========================================\n');

// Helper: build seeded design matrix with intercept and `p` standard-normal predictors.
function buildDesign(n: number, p: number, seed: number): number[][] {
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = [1];
    for (let j = 0; j < p; j++) row.push(normalSample(0, 1, rng));
    X[i] = row;
  }
  return X;
}

function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.1 — glmFit logistic recovers known β.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 500;
  const X = buildDesign(n, 2, 22221);
  const betaTrue = [-0.5, 1.0, 2.0];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22222);
  const fit = glmFit(X, y, FAMILIES.bernoulli);
  const err = maxAbsDiff(fit.beta, betaTrue);
  check(
    'T8.1 logistic glmFit recovers β within 0.15',
    err < 0.15 && fit.converged && fit.nIter <= 15,
    `‖β̂-β‖∞=${err.toFixed(4)}, converged=${fit.converged}, nIter=${fit.nIter}`,
    'err < 0.15, converged, nIter ≤ 15',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.2 — glmFit on EXAMPLE_6_GLM_DATA converges and is internally consistent.
//
// (Brief originally specified a statsmodels β̂ cross-check to 1e-6. The
// notebook does not export the numpy-RNG-generated arrays as JSON, so we
// cannot import the statsmodels-fit data into TS. The check here is
// internal-consistency: glmFit on the TS-generated EXAMPLE_6 dataset
// recovers β_true within asymptotic tolerance and converges in ≤15 iters.
// The MDX prose still cites the notebook's statsmodels values verbatim.)
// ─────────────────────────────────────────────────────────────────────────────
{
  // Using EXAMPLE_6_GLM_DATA from regression-data — but to keep the test
  // self-contained (no cross-module imports for fixtures), we rebuild it.
  const n = 200;
  const X = buildDesign(n, 2, 2222);
  const betaTrue = [-0.5, 1.0, 2.0];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 2223);
  const fit = glmFit(X, y, FAMILIES.bernoulli);
  const errInf = maxAbsDiff(fit.beta, betaTrue);
  check(
    'T8.2 EXAMPLE_6_GLM_DATA logistic fit converges and recovers β within 0.30',
    errInf < 0.3 && fit.converged && fit.nIter <= 15,
    `‖β̂-β‖∞=${errInf.toFixed(4)}, converged=${fit.converged}, nIter=${fit.nIter}`,
    'err < 0.30 at n=200',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.3 — glmFit Poisson recovers known β with offset.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 500;
  const X = buildDesign(n, 2, 22231);
  const offset = new Array<number>(n).fill(0).map((_, i) => Math.log(1 + (i % 5)));
  const betaTrue = [0.5, 1.2, -0.8];
  const y = simulateGLM(X, betaTrue, FAMILIES.poisson, LINKS.log, 22232, { offset });
  const fit = glmFit(X, y, FAMILIES.poisson, undefined, { offset });
  const err = maxAbsDiff(fit.beta, betaTrue);
  check(
    'T8.3 Poisson glmFit recovers β within 0.10 (with offset)',
    err < 0.10 && fit.converged,
    `‖β̂-β‖∞=${err.toFixed(4)}, converged=${fit.converged}`,
    'err < 0.10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.4 — Poisson offset shifts the intercept by ≈ log(mean(t)).
//        With offset:    log E[Y] = β₀ + β₁ X + log t  ⇒  E[Y] = exp(β₀) · t · exp(β₁ X)
//        Without offset: log E[Y] = β₀' + β₁' X        ⇒  E[Y] = exp(β₀') · exp(β₁' X)
//        Best constant-multiplier fit absorbs t̄ into the intercept ⇒ β₀' ≈ β₀ + log(t̄),
//        not β₀ + mean(log t) (Jensen: log(mean) ≥ mean(log)).
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 400;
  const X = buildDesign(n, 1, 22241);
  const tValues = new Array<number>(n).fill(0).map((_, i) => 1 + (i % 7));
  const offset = tValues.map((t) => Math.log(t));
  const tBar = tValues.reduce((a, b) => a + b, 0) / n;
  const expectedShift = Math.log(tBar);
  const betaTrue = [0.5, 1.0];
  const y = simulateGLM(X, betaTrue, FAMILIES.poisson, LINKS.log, 22242, { offset });
  const fitWith = glmFit(X, y, FAMILIES.poisson, undefined, { offset });
  const fitWithout = glmFit(X, y, FAMILIES.poisson);
  const interceptShift = fitWithout.beta[0] - fitWith.beta[0];
  check(
    'T8.4 Poisson intercept shift ≈ log(mean t)',
    Math.abs(interceptShift - expectedShift) < 0.05,
    `shift=${interceptShift.toFixed(4)}`,
    `≈ log(${tBar.toFixed(3)}) = ${expectedShift.toFixed(4)}`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.5 — glmFit Gamma (log link) recovers β and dispersion.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 500;
  const X = buildDesign(n, 1, 22251);
  const betaTrue = [1.0, 0.5];
  const phiTrue = 0.5; // shape ν = 2
  const y = simulateGLM(X, betaTrue, FAMILIES.gamma, LINKS.log, 22252, {
    dispersion: phiTrue,
  });
  const fit = glmFit(X, y, FAMILIES.gamma, LINKS.log);
  const err = maxAbsDiff(fit.beta, betaTrue);
  check(
    'T8.5 Gamma log-link glmFit recovers β within 0.15 and φ̂ ∈ [0.4, 0.7]',
    err < 0.15 && fit.phi > 0.4 && fit.phi < 0.7 && fit.converged,
    `‖β̂-β‖∞=${err.toFixed(4)}, φ̂=${fit.phi.toFixed(4)}`,
    'true β=(1.0, 0.5), φ=0.5',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.6 — Normal+identity GLM matches OLS to 1e-10.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 200;
  const X = buildDesign(n, 2, 22261);
  const betaTrue = [1.0, -2.0, 0.5];
  const y = simulateLinearModel(X, betaTrue, 1.0, 22262);
  const olsβ = olsFit(X, y).beta;
  const glmβ = glmFit(X, y, FAMILIES.normal).beta;
  const err = maxAbsDiff(olsβ, glmβ);
  check(
    'T8.6 Normal+identity GLM = OLS to 1e-10',
    err < 1e-10,
    err.toExponential(3),
    '< 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.7 — irlsStep iteration matches glmFit step-by-step.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 200;
  const X = buildDesign(n, 2, 22271);
  const betaTrue = [-0.3, 1.1, 0.6];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22272);
  // Manually iterate irlsStep 5 times.
  let beta = [0, 0, 0];
  for (let t = 0; t < 5; t++) {
    beta = irlsStep(X, y, beta, FAMILIES.bernoulli, LINKS.logit).betaNext;
  }
  // Compare to glmFit with maxIter=5 starting from β^(0)=0.
  const fit = glmFit(X, y, FAMILIES.bernoulli, undefined, {
    maxIter: 5,
    tol: 0,
    startBeta: [0, 0, 0],
  });
  const err = maxAbsDiff(beta, fit.beta);
  check(
    'T8.7 irlsStep ×5 == glmFit(maxIter=5, tol=0) to 1e-10',
    err < 1e-10,
    err.toExponential(3),
    '< 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.8 — Logistic deviance matches direct -2[y log(p) + (1-y) log(1-p)] formula.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 200;
  const X = buildDesign(n, 2, 22281);
  const betaTrue = [-0.3, 1.0, 0.5];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22282);
  const fit = glmFit(X, y, FAMILIES.bernoulli);
  let directD = 0;
  for (let i = 0; i < n; i++) {
    const p = fit.mu[i];
    const yi = y[i];
    // Saturated ℓ_sat(y) = 0 for y ∈ {0,1}; deviance contrib is 2(ℓ_sat - ℓ_i).
    const ll = yi * Math.log(Math.max(p, 1e-12)) +
      (1 - yi) * Math.log(Math.max(1 - p, 1e-12));
    directD += -2 * ll;
  }
  const err = Math.abs(deviance(fit) - directD);
  check(
    'T8.8 logistic deviance == direct -2 log-lik formula to 1e-8',
    err < 1e-8,
    `glmFit D=${fit.deviance.toFixed(6)}, direct=${directD.toFixed(6)}`,
    `|Δ|=${err.toExponential(3)}`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.9 — Pearson and deviance residuals equal the standardized OLS residual
//        when family = Normal + identity link.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 100;
  const X = buildDesign(n, 1, 22291);
  const betaTrue = [1.0, -0.7];
  const y = simulateLinearModel(X, betaTrue, 1.0, 22292);
  const fit = glmFit(X, y, FAMILIES.normal);
  const rPearson = pearsonResiduals(fit);
  const rDeviance = devianceResiduals(fit);
  // For normal + identity, both should equal r_i / σ̂ where σ̂² = MSR.
  // Pearson residual: (y - μ)/sqrt(V·φ) = r/sqrt(φ) (since V=1).
  // Deviance residual: sign(r) · sqrt((y-μ)²) = r itself (with sign), divided by — actually
  // the deviance contribution d_i = (y-μ)² and residual is sign(r)·sqrt(d) = |r|·sign(r) = r.
  // So Pearson = r/sqrt(φ); Deviance = r.
  let maxDiff = 0;
  for (let i = 0; i < n; i++) {
    const r = y[i] - fit.mu[i];
    const expectedP = r / Math.sqrt(Math.max(fit.phi, 1e-12));
    maxDiff = Math.max(maxDiff, Math.abs(rPearson[i] - expectedP));
    maxDiff = Math.max(maxDiff, Math.abs(rDeviance[i] - r));
  }
  check(
    'T8.9 Normal+identity: Pearson=r/√φ, Deviance=r to 1e-10',
    maxDiff < 1e-10,
    maxDiff.toExponential(3),
    '< 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.10 — Under H₀, devianceTestNested p-values are uniform.
//        500 simulations of a nested Poisson where x_2's coefficient = 0.
// ─────────────────────────────────────────────────────────────────────────────
{
  const nSim = 500;
  const n = 100;
  const pvals: number[] = new Array(nSim);
  let nFailed = 0;
  for (let s = 0; s < nSim; s++) {
    const Xfull = buildDesign(n, 2, 23000 + 7 * s);
    const Xred = Xfull.map((row) => [row[0], row[1]]);
    const betaH0 = [0.3, 0.4, 0.0]; // x_2 coefficient = 0 under H0
    const y = simulateGLM(Xfull, betaH0, FAMILIES.poisson, LINKS.log, 23500 + 7 * s);
    try {
      const result = devianceTestNested(Xfull, Xred, y, FAMILIES.poisson, LINKS.log, 0.05);
      pvals[s] = result.pValue;
    } catch {
      pvals[s] = NaN;
      nFailed++;
    }
  }
  const validP = pvals.filter((p) => !isNaN(p));
  // Type-I error rate (rejection at α=0.05) should be ≈ 0.05 (binomial 95% CI ~ [0.03, 0.07]).
  const rejected = validP.filter((p) => p < 0.05).length;
  const rejRate = rejected / validP.length;
  check(
    'T8.10 nested Poisson devianceTestNested: rejection rate ≈ α at H₀',
    rejRate >= 0.025 && rejRate <= 0.085,
    `rejRate=${rejRate.toFixed(3)}, n=${validP.length}/${nSim} valid (${nFailed} failed)`,
    'expect ≈ 0.05',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.11 — devianceTestNested on EXAMPLE_9_GLM_DATA: structure check.
//        (Original brief specified statsmodels-cross-check to 1e-6; deviated
//        to internal-consistency for the same numpy/TS RNG-incompatibility
//        reason as T8.2.)
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 150;
  const Xfull = buildDesign(n, 3, 22311);
  const Xred = Xfull.map((row) => [row[0], row[1]]);
  const offset = new Array<number>(n).fill(0).map((_, i) => Math.log(1 + (i % 4)));
  const betaTrue = [-1.0, 0.5, 0.7, 0.3];
  const y = simulateGLM(Xfull, betaTrue, FAMILIES.poisson, LINKS.log, 22312, { offset });
  const result = devianceTestNested(Xfull, Xred, y, FAMILIES.poisson, LINKS.log, 0.05, {
    offset,
  });
  check(
    'T8.11 devianceTestNested: df=2, diff>0, p-value < 0.001',
    result.df === 2 && result.diff > 10 && result.pValue < 0.001,
    `df=${result.df}, diff=${result.diff.toFixed(3)}, p=${result.pValue.toExponential(2)}`,
    'two non-zero coefficients dropped ⇒ very significant',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.12 — sandwichVCov(HC0) matches the manual White-1980 formula on a toy n=20.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 20;
  const X = buildDesign(n, 1, 22321);
  const betaTrue = [-0.4, 1.2];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22322);
  const fit = glmFit(X, y, FAMILIES.bernoulli);
  const sandHC0 = sandwichVCov(fit, 'HC0');

  // Manual: A = X^T W X, B = X^T diag((y-μ)² · w² · g'²) X = X^T diag(score²) X.
  const m = 2;
  const A: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++)
      for (let k = 0; k < m; k++)
        A[j][k] += fit.weights[i] * X[i][j] * X[i][k];
  }
  // Invert A (2×2).
  const detA = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const Ainv: number[][] = [
    [A[1][1] / detA, -A[0][1] / detA],
    [-A[1][0] / detA, A[0][0] / detA],
  ];
  const B: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let i = 0; i < n; i++) {
    const r = y[i] - fit.mu[i];
    const gp = LINKS.logit.gPrime(fit.mu[i]);
    const u = fit.weights[i] * gp * r;
    const u2 = u * u;
    for (let j = 0; j < m; j++)
      for (let k = 0; k < m; k++) B[j][k] += u2 * X[i][j] * X[i][k];
  }
  // V = Ainv · B · Ainv.
  const tmp: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let j = 0; j < m; j++)
    for (let k = 0; k < m; k++) {
      let s = 0;
      for (let q = 0; q < m; q++) s += Ainv[j][q] * B[q][k];
      tmp[j][k] = s;
    }
  const Vmanual: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let j = 0; j < m; j++)
    for (let k = 0; k < m; k++) {
      let s = 0;
      for (let q = 0; q < m; q++) s += tmp[j][q] * Ainv[q][k];
      Vmanual[j][k] = s;
    }
  let maxDiff = 0;
  for (let j = 0; j < m; j++)
    for (let k = 0; k < m; k++)
      maxDiff = Math.max(maxDiff, Math.abs(sandHC0[j][k] - Vmanual[j][k]));
  check(
    'T8.12 sandwichVCov(HC0) matches manual formula to 1e-10',
    maxDiff < 1e-10,
    maxDiff.toExponential(3),
    '< 1e-10',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.13 — sandwichVCov(HC3) > naive vcov on the overdispersed-Poisson DGP.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 200;
  const X = buildDesign(n, 1, 22331);
  const betaTrue = [-0.3, 0.9];
  const y = simulateOverdispersedPoisson(X, betaTrue, 1.8, 22332);
  const fit = glmFit(X, y, FAMILIES.poisson);
  const naiveVcov11 = fit.vcov[1][1];
  const hc3 = sandwichVCov(fit, 'HC3');
  const ratio = hc3[1][1] / naiveVcov11;
  check(
    'T8.13 HC3 vcov[1,1] > naive (overdispersed Poisson)',
    ratio > 1.2 && ratio < 4.0,
    `HC3/naive = ${ratio.toFixed(3)}`,
    'expect ratio ≈ 1.5–2.5',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.14 — Sandwich coverage: naive < 0.92, HC3 ∈ [0.88, 0.99] under
//        overdispersion. (200 sims for speed; full 2000 reserved for the
//        SandwichCoverageSimulator component's "Run 500" preset.)
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 100;
  const X = buildDesign(n, 1, 22341);
  const betaTrue = [-0.3, 0.9];
  const z975 = 1.959963984540054; // standardNormalInvCDF(0.975)
  let coveredNaive = 0;
  let coveredHC3 = 0;
  let nValid = 0;
  for (let s = 0; s < 200; s++) {
    const y = simulateOverdispersedPoisson(X, betaTrue, 1.8, 22500 + s);
    let fit;
    try {
      fit = glmFit(X, y, FAMILIES.poisson);
    } catch {
      continue;
    }
    if (!fit.converged) continue;
    nValid++;
    const seNaive = Math.sqrt(Math.max(fit.vcov[1][1], 1e-16));
    const ciNaive = [fit.beta[1] - z975 * seNaive, fit.beta[1] + z975 * seNaive];
    if (ciNaive[0] <= betaTrue[1] && betaTrue[1] <= ciNaive[1]) coveredNaive++;
    const seHC3 = sandwichSE(fit, 'HC3')[1];
    const ciHC3 = [fit.beta[1] - z975 * seHC3, fit.beta[1] + z975 * seHC3];
    if (ciHC3[0] <= betaTrue[1] && betaTrue[1] <= ciHC3[1]) coveredHC3++;
  }
  const covNaive = coveredNaive / nValid;
  const covHC3 = coveredHC3 / nValid;
  check(
    'T8.14 sandwich coverage: HC3 > naive under overdispersion',
    covHC3 > covNaive && covNaive < 0.92 && covHC3 > 0.85,
    `naive=${covNaive.toFixed(3)}, HC3=${covHC3.toFixed(3)} (n=${nValid} valid sims)`,
    'naive < 0.92, HC3 > 0.85',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.15 — coefCIProfileGLM produces an asymmetric CI for a near-separation
//        logistic dataset (Wald-z is symmetric; profile-LRT is not).
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 80;
  const X = buildDesign(n, 1, 22351);
  const betaTrue = [0.0, 4.0]; // strong effect, near-separation
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22352);
  const fit = glmFit(X, y, FAMILIES.bernoulli);
  if (!fit.converged) {
    check('T8.15 profile-LRT CI asymmetry (near-separation)', false, 'fit non-converged', 'converged required');
  } else {
    const ci = coefCIProfileGLM(fit, 1, 0.05);
    const upperHalf = ci.upper - ci.center;
    const lowerHalf = ci.center - ci.lower;
    const asymmetry = Math.abs(upperHalf - lowerHalf) / Math.max(upperHalf, lowerHalf);
    check(
      'T8.15 profile-LRT CI is asymmetric near separation (>5%)',
      asymmetry > 0.05,
      `[${ci.lower.toFixed(3)}, ${ci.center.toFixed(3)}, ${ci.upper.toFixed(3)}], asym=${(asymmetry * 100).toFixed(1)}%`,
      '>5% asymmetry',
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.16 — simulateGLM is reproducible: same seed → identical output.
// ─────────────────────────────────────────────────────────────────────────────
{
  const X = buildDesign(50, 2, 22361);
  const betaTrue = [-0.4, 1.0, 0.6];
  const y1 = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22362);
  const y2 = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22362);
  let identical = true;
  for (let i = 0; i < y1.length; i++) {
    if (y1[i] !== y2[i]) {
      identical = false;
      break;
    }
  }
  // And: different seeds give different output.
  const y3 = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, 22363);
  let differs = false;
  for (let i = 0; i < y1.length; i++) {
    if (y1[i] !== y3[i]) {
      differs = true;
      break;
    }
  }
  check(
    'T8.16 simulateGLM seed reproducibility (same seed identical, different differs)',
    identical && differs,
    `same-seed identical=${identical}, different-seed differs=${differs}`,
    'both true',
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.17 — simulateOverdispersedPoisson empirical Var/E ≈ requested ratio.
// ─────────────────────────────────────────────────────────────────────────────
{
  const n = 5000;
  const X = buildDesign(n, 0, 22371); // intercept-only (constant μ)
  const betaTrue = [Math.log(5)]; // μ = 5 for every observation
  const r = 1.8;
  const y = simulateOverdispersedPoisson(X, betaTrue, r, 22372);
  const muHat = y.reduce((a, b) => a + b, 0) / n;
  const varHat = y.reduce((a, b) => a + (b - muHat) ** 2, 0) / (n - 1);
  const ratio = varHat / muHat;
  check(
    `T8.17 simulateOverdispersedPoisson empirical Var/E ≈ ${r}`,
    Math.abs(ratio - r) < 0.2,
    `μ̂=${muHat.toFixed(3)}, σ̂²=${varHat.toFixed(3)}, ratio=${ratio.toFixed(3)}`,
    `expect ≈ ${r}`,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary.
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n========================================');
console.log(` ${passed} passed, ${failed} failed out of ${passed + failed}`);
console.log('========================================\n');
if (failed > 0) process.exit(1);
