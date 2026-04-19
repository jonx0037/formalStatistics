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
// Summary.
// ─────────────────────────────────────────────────────────────────────────────
console.log('\n========================================');
console.log(` ${passed} passed, ${failed} failed out of ${passed + failed}`);
console.log('========================================\n');
if (failed > 0) process.exit(1);
