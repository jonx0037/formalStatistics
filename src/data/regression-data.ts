/**
 * regression-data.ts — Track 6 data module, seeded by Topic 21.
 *
 * Surface:
 *   §7.A  Canonical small datasets used in MDX examples
 *         (Example 2 five-point fit, Example 5 OLS-vs-naive comparison,
 *          Example 6 hat-matrix worked computation, Anscombe quartet)
 *   §7.B  Component presets consumed by the four interactive components
 *   §7.C  DGP dispatcher (`generateLinearModelData`) for simulation-based
 *         components — CoefficientCIBands and RegressionDiagnosticsExplorer
 *
 * Topics 22–24 extend this module with GLM / ridge-lasso / model-selection
 * datasets. Keep the subsection ordering for Topic-22's append.
 */

import { seededRandom } from '../components/viz/shared/probability';
import {
  normalSample,
  bernoulliSample,
  uniformSample,
} from '../components/viz/shared/convergence';
import {
  simulateLinearModel,
  simulateGLM,
  simulateOverdispersedPoisson,
  regularizationPath,
  crossValidate,
  FAMILIES,
  LINKS,
  type GLMFamily,
  type LinkFunction,
  type RegularizationPathResult,
  type CVResult,
} from '../components/viz/shared/regression';

// ─────────────────────────────────────────────────────────────────────────────
// §7.A — Canonical small datasets for MDX examples
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Example 2 (§21.2): 5-point worked example for OLS hand-computation.
 * Hand-computable slope = 0.96, intercept = 0.22 (ȳ = 3.1, x̄ = 3).
 */
export const EXAMPLE_2_DATA = {
  x: [1, 2, 3, 4, 5],
  y: [1.2, 1.9, 3.3, 4.1, 5.0],
};

/**
 * Example 5 (§21.6): 4-point OLS-vs-naive comparison (n=4, p=1).
 * True model: β_0 = 1, β_1 = 2, σ = 0.3, Normal errors.
 * `yTruth(seed)` returns a fresh y-vector sampled at the given seed via
 * `simulateLinearModel` — reproducible across sessions.
 */
export const EXAMPLE_5_DATA = {
  x: [1, 2, 3, 4],
  /** Seeded realization of y for a single run. */
  yTruth: (seed: number): number[] => {
    const X = EXAMPLE_5_DATA.x.map((xi) => [1, xi]);
    return simulateLinearModel(X, [1, 2], 0.3, seed);
  },
};

/**
 * Example 6 (§21.7): 3-point hat-matrix explicit computation.
 * Used to walk through H = X (X^T X)^-1 X^T for n=3, p=1.
 */
export const EXAMPLE_6_DATA = {
  x: [0, 1, 2],
  y: [1, 3, 4],
};

/**
 * Example 10 (§21.9): the Anscombe quartet (Anscombe 1973). Four datasets
 * sharing mean, variance, regression line, and R² to two decimal places —
 * but with wildly different residual structures. The punchline that
 * residual plots are irreducible, not optional.
 */
export const ANSCOMBE_QUARTET = {
  I: {
    x: [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    y: [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
  },
  II: {
    x: [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    y: [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
  },
  III: {
    x: [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    y: [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
  },
  IV: {
    x: [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
    y: [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// §7.B — Component presets
// ─────────────────────────────────────────────────────────────────────────────

export interface LeastSquaresPreset {
  points: Array<[number, number]>;
  label: string;
}

export const LEAST_SQUARES_DRAGGABLE_PRESETS: Record<string, LeastSquaresPreset> = {
  linearTrend: {
    label: 'Linear trend',
    points: [
      [1, 2.1],
      [2, 3.9],
      [3, 5.8],
      [4, 8.2],
      [5, 9.9],
      [6, 12.1],
      [7, 13.8],
    ],
  },
  noRelationship: {
    label: 'No relationship',
    points: [
      [1, 5.2],
      [2, 4.8],
      [3, 5.1],
      [4, 4.9],
      [5, 5.3],
      [6, 4.7],
      [7, 5.0],
    ],
  },
  outlierPresent: {
    label: 'Outlier present',
    points: [
      [1, 2.1],
      [2, 3.9],
      [3, 5.8],
      [4, 15.0],
      [5, 9.9],
      [6, 12.1],
      [7, 13.8],
    ],
  },
  perfectFit: {
    label: 'Perfect fit',
    points: [
      [1, 2],
      [2, 4],
      [3, 6],
      [4, 8],
      [5, 10],
      [6, 12],
      [7, 14],
    ],
  },
};

export interface FTestPreset {
  df1: number;
  df2: number;
  lambda: number;
  alpha: number;
  label: string;
}

export const F_TEST_EXPLORER_PRESETS: Record<string, FTestPreset> = {
  oneWayAnova3Groups: {
    df1: 2,
    df2: 27,
    lambda: 8,
    alpha: 0.05,
    label: 'One-way ANOVA (3 groups, n=30)',
  },
  partialFTest: {
    df1: 2,
    df2: 46,
    lambda: 12,
    alpha: 0.05,
    label: 'Partial F-test (2 coefs, n=50)',
  },
  fullSignificance: {
    df1: 5,
    df2: 44,
    lambda: 20,
    alpha: 0.05,
    label: 'Full significance (5 predictors, n=50)',
  },
  underpowered: {
    df1: 3,
    df2: 12,
    lambda: 2,
    alpha: 0.05,
    label: 'Underpowered (n=16, k=3)',
  },
};

export interface CoefficientCIPreset {
  n: number;
  sigma: number;
  alpha: number;
  designKind: 'wellConditioned' | 'collinear';
  label: string;
}

export const COEFFICIENT_CI_BANDS_PRESETS: Record<string, CoefficientCIPreset> = {
  smallSample: {
    n: 25,
    sigma: 1,
    alpha: 0.05,
    designKind: 'wellConditioned',
    label: 'Small sample (n=25)',
  },
  largeSample: {
    n: 500,
    sigma: 1,
    alpha: 0.05,
    designKind: 'wellConditioned',
    label: 'Large sample (n=500)',
  },
  highCollinearity: {
    n: 100,
    sigma: 1,
    alpha: 0.05,
    designKind: 'collinear',
    label: 'High collinearity',
  },
  wellConditioned: {
    n: 100,
    sigma: 1,
    alpha: 0.05,
    designKind: 'wellConditioned',
    label: 'Well-conditioned (n=100)',
  },
};

export type DiagnosticKind =
  | 'ideal'
  | 'heteroscedastic'
  | 'outlier'
  | 'nonLinear'
  | 'highLeverage';

export interface DiagnosticPreset {
  kind: DiagnosticKind;
  label: string;
  diagnosis: string;
}

export const REGRESSION_DIAGNOSTICS_PRESETS: Record<string, DiagnosticPreset> = {
  ideal: {
    kind: 'ideal',
    label: 'Ideal (textbook)',
    diagnosis:
      'All assumptions met. No pattern in residuals. Q-Q plot is near-linear. Leverage values cluster below 2(p+1)/n.',
  },
  heteroscedastic: {
    kind: 'heteroscedastic',
    label: 'Heteroscedastic',
    diagnosis:
      'Fan shape in residual-vs-fitted — variance grows with the fitted value. See §21.9 Rem 19 (Topic 22 HC-robust SEs) and Rem 21 (WLS re-weighting).',
  },
  outlier: {
    kind: 'outlier',
    label: 'Outlier',
    diagnosis:
      'One observation pulls the fit. High studentized residual and high Cook’s distance. See §21.9 Rem 20 — robust regression (Topic 15 M-estimators) or outlier diagnostics.',
  },
  nonLinear: {
    kind: 'nonLinear',
    label: 'Non-linear',
    diagnosis:
      'Curvature in residual-vs-fitted. The linear model is misspecified; add a polynomial term or transform. See §21.9 Rem 18 on Q-Q diagnostics and §21.1 Rem 1 on linear-in-parameters.',
  },
  highLeverage: {
    kind: 'highLeverage',
    label: 'High leverage',
    diagnosis:
      'One observation has h_ii ≫ 2(p+1)/n. May dominate the fit even with small residual. See §21.9 Rem 13 and Cook’s distance.',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// §7.C — DGP dispatcher for simulation-based components
// ─────────────────────────────────────────────────────────────────────────────

export interface LinearModelDGP {
  n: number;
  p: number; // non-intercept predictors
  beta: number[]; // length p+1
  sigma: number;
  designKind:
    | 'wellConditioned'
    | 'collinear'
    | 'ideal'
    | 'heteroscedastic'
    | 'outlier'
    | 'nonLinear'
    | 'highLeverage';
  seed: number;
}

/**
 * Given a DGP spec, generate a concrete (X, y) realization. The design-kind
 * switch controls both the structure of X and any pathology injected into y
 * — heteroscedasticity re-weights errors, outlier adds a large residual at
 * the last observation, nonLinear adds a hidden quadratic, highLeverage
 * places one x far from the rest.
 */
export function generateLinearModelData(dgp: LinearModelDGP): {
  X: number[][];
  y: number[];
} {
  const rng = seededRandom(dgp.seed);
  const { n, p, beta, sigma, designKind } = dgp;

  // Invariants: fail loud rather than silently fitting a wrong-length β.
  if (beta.length !== p + 1) {
    throw new Error(
      `generateLinearModelData: beta.length (${beta.length}) must equal p + 1 (${p + 1}) for designKind="${designKind}"`,
    );
  }
  const diagnosticKinds: Array<LinearModelDGP['designKind']> = [
    'ideal',
    'heteroscedastic',
    'outlier',
    'nonLinear',
    'highLeverage',
  ];
  if (diagnosticKinds.includes(designKind) && p !== 1) {
    throw new Error(
      `generateLinearModelData: diagnostic designKind="${designKind}" requires p === 1 (got p=${p})`,
    );
  }

  // Build X.
  const X: number[][] = [];
  if (designKind === 'wellConditioned') {
    for (let i = 0; i < n; i++) {
      const row: number[] = [1];
      for (let j = 0; j < p; j++) row.push(normalSample(0, 1, rng));
      X.push(row);
    }
  } else if (designKind === 'collinear') {
    // First non-intercept column is a Normal z; second column = z + small noise
    // (near-collinear pair). Remaining columns are iid Normal.
    for (let i = 0; i < n; i++) {
      const z = normalSample(0, 1, rng);
      const row: number[] = [1, z];
      if (p >= 2) row.push(z + normalSample(0, 0.05, rng));
      for (let j = 2; j < p; j++) row.push(normalSample(0, 1, rng));
      X.push(row);
    }
  } else {
    // Diagnostic-preset designs use p = 1 (intercept + single x).
    if (designKind === 'highLeverage') {
      for (let i = 0; i < n - 1; i++) X.push([1, normalSample(0, 1, rng)]);
      X.push([1, 10]); // a single high-leverage point far from the cloud
    } else {
      for (let i = 0; i < n; i++) X.push([1, normalSample(0, 1.5, rng)]);
    }
  }

  // Compute noise-free fitted values (with possible hidden-quadratic).
  // beta.length === X[i].length by the invariants above.
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let mu = 0;
    for (let j = 0; j < beta.length; j++) mu += X[i][j] * beta[j];

    // Inject any structural pathology from designKind.
    if (designKind === 'nonLinear') {
      mu += 0.5 * X[i][1] * X[i][1];
    }

    // Error term — design-specific scaling.
    let eps: number;
    if (designKind === 'heteroscedastic') {
      // σ grows with the fitted value: cheap multiplicative scheme.
      const scale = 1 + Math.abs(X[i][1]);
      eps = normalSample(0, sigma * scale, rng);
    } else {
      eps = normalSample(0, sigma, rng);
    }

    // Outlier injection — last observation.
    if (designKind === 'outlier' && i === n - 1) {
      eps += sigma * 8;
    }

    y[i] = mu + eps;
  }

  return { X, y };
}

// ─────────────────────────────────────────────────────────────────────────────
// §7.C — GLM preset datasets (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────
//
// Each dataset is generated at module-load time from a fixed seed via
// `simulateGLM` (or a small DGP-specific helper). Reproducible across
// sessions and components: the array references are stable for the lifetime
// of the module instance.
//
// **Note on reference values.** The reference β̂ printed in the §22.4–§22.6
// MDX (e.g. β̂_logistic ≈ [-0.972, 1.319, 1.869]) come from the notebook's
// numpy/statsmodels run on numpy-RNG-generated data. The TypeScript datasets
// below use the convergence.ts samplers on a different (LCG) RNG — same DGP
// recipe, different sample. The internal-consistency tests (T8.x) verify that
// glmFit recovers the true β to within asymptotic tolerance and matches
// closed-form formulas where applicable.

interface GLMDataset {
  X: number[][];
  y: number[];
  betaTrue: number[];
}

function buildLogisticCreditDefault(seed: number, n: number): GLMDataset {
  // X = [intercept, age_z, income_z]; β_true = (-0.5, 1.0, 2.0)
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = [1, normalSample(0, 1, rng), normalSample(0, 1, rng)];
  }
  const betaTrue = [-0.5, 1.0, 2.0];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, seed + 1);
  return { X, y, betaTrue };
}

function buildPoissonInsuranceFreq(
  seed: number,
  n: number,
): GLMDataset & { offset: number[] } {
  // X = [intercept, age_z, region_indicator]; offset = log(policy-years)
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  const offset = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    X[i] = [
      1,
      normalSample(0, 1, rng),
      bernoulliSample(0.45, rng), // 45% in region 1
    ];
    const py = uniformSample(1, 10, rng);
    offset[i] = Math.log(py);
  }
  const betaTrue = [-1.2, 0.6, 0.5];
  const y = simulateGLM(X, betaTrue, FAMILIES.poisson, LINKS.log, seed + 1, {
    offset,
  });
  return { X, y, betaTrue, offset };
}

function buildGammaInsuranceAmount(
  seed: number,
  n: number,
): GLMDataset & { dispersion: number } {
  // X = [intercept, severity_z]; log link; shape ν = 2 ⇒ φ = 0.5
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = [1, normalSample(0, 1, rng)];
  }
  const betaTrue = [1.0, 0.5];
  const phi = 0.5;
  const y = simulateGLM(X, betaTrue, FAMILIES.gamma, LINKS.log, seed + 1, {
    dispersion: phi,
  });
  return { X, y, betaTrue, dispersion: phi };
}

function buildNestedPoisson(
  seed: number,
  n: number,
): {
  Xfull: number[][];
  Xreduced: number[][];
  y: number[];
  offset: number[];
  betaTrue: number[];
} {
  // Full: [intercept, x1, x2, x3] with offset; Reduced drops x2, x3.
  const rng = seededRandom(seed);
  const Xfull: number[][] = new Array(n);
  const Xreduced: number[][] = new Array(n);
  const offset = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x1 = normalSample(0, 1, rng);
    const x2 = normalSample(0, 1, rng);
    const x3 = normalSample(0, 1, rng);
    Xfull[i] = [1, x1, x2, x3];
    Xreduced[i] = [1, x1];
    offset[i] = Math.log(uniformSample(1, 5, rng));
  }
  const betaTrue = [-1.0, 0.5, 0.7, 0.3];
  const y = simulateGLM(Xfull, betaTrue, FAMILIES.poisson, LINKS.log, seed + 1, {
    offset,
  });
  return { Xfull, Xreduced, y, offset, betaTrue };
}

function buildNearSeparationLogistic(
  seed: number,
  n: number,
): GLMDataset {
  // Quasi-complete separation: very strong slope, moderate sample.
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = [1, normalSample(0, 1, rng)];
  }
  const betaTrue = [0.0, 5.0];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, seed + 1);
  return { X, y, betaTrue };
}

function buildOverdispersedPoisson(
  seed: number,
  n: number,
  dispersionRatio: number,
): {
  X: number[][];
  y: number[];
  betaTrue: number[];
  dispersionRatio: number;
} {
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = [1, normalSample(0, 1, rng)];
  }
  const betaTrue = [-0.3, 0.9];
  const y = simulateOverdispersedPoisson(X, betaTrue, dispersionRatio, seed + 1);
  return { X, y, betaTrue, dispersionRatio };
}

/**
 * Example 6 (§22.4): synthetic credit-default logistic regression.
 * n = 200, predictors = [intercept, age_z, income_z], β_true = (-0.5, 1.0, 2.0).
 */
export const EXAMPLE_6_GLM_DATA = buildLogisticCreditDefault(2222, 200);

/**
 * Example 7 (§22.5): synthetic insurance claim-frequency Poisson regression.
 * n = 300, β_true = (-1.2, 0.6, 0.5), offset = log(policy-years ∈ [1, 10]).
 */
export const EXAMPLE_7_GLM_DATA = buildPoissonInsuranceFreq(2223, 300);

/**
 * Example 8 (§22.6): synthetic insurance claim-amount Gamma GLM (log link).
 * n = 250, β_true = (1.0, 0.5), shape ν = 2 ⇒ dispersion φ = 0.5.
 */
export const EXAMPLE_8_GLM_DATA = buildGammaInsuranceAmount(2224, 250);

/**
 * Example 9 (§22.7): nested Poisson deviance test.
 * Full model: intercept + 3 predictors + log-offset.
 * Reduced model: intercept + 1 predictor + log-offset.
 */
export const EXAMPLE_9_GLM_DATA = buildNestedPoisson(2225, 150);

/**
 * Example 10 (§22.8): near-separation logistic (n=80) where Wald-z fails and
 * profile-LRT recovers an asymmetric, conservative CI.
 */
export const EXAMPLE_10_GLM_DATA = buildNearSeparationLogistic(2226, 80);

/**
 * Example 12 (§22.9): overdispersed-Poisson DGP (Var/E ≈ 1.8) for the
 * sandwich-coverage simulation.
 */
export const EXAMPLE_12_GLM_DATA = buildOverdispersedPoisson(2232, 200, 1.8);

// ─────────────────────────────────────────────────────────────────────────────
// §7.D — Component presets (Topic 22)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * IRLSVisualizer presets — toy 2-predictor logistic datasets covering three
 * convergence regimes.
 */
function buildIRLSPreset(
  seed: number,
  n: number,
  betaTrue: number[],
): GLMDataset {
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = [1, normalSample(0, 1, rng)];
  }
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, seed + 1);
  return { X, y, betaTrue };
}

export const IRLS_VISUALIZER_PRESETS = {
  wellSeparated: buildIRLSPreset(3001, 60, [-1.0, 2.0]),
  noisy: buildIRLSPreset(3002, 60, [0.0, 0.8]),
  nearSeparation: buildIRLSPreset(3003, 40, [0.0, 4.5]),
};

/**
 * GLMExplorer presets — six datasets covering the three response families
 * with both "real-scale" and "toy-fast-demo" sizes.
 */
function buildToyBernoulli(seed: number, n: number): GLMDataset {
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) X[i] = [1, normalSample(0, 1, rng)];
  const betaTrue = [0.2, 1.5];
  const y = simulateGLM(X, betaTrue, FAMILIES.bernoulli, LINKS.logit, seed + 1);
  return { X, y, betaTrue };
}

function buildToyPoisson(seed: number, n: number): GLMDataset {
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) X[i] = [1, normalSample(0, 1, rng)];
  const betaTrue = [0.5, 0.4];
  const y = simulateGLM(X, betaTrue, FAMILIES.poisson, LINKS.log, seed + 1);
  return { X, y, betaTrue };
}

function buildToyGamma(seed: number, n: number): GLMDataset {
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) X[i] = [1, normalSample(0, 1, rng)];
  const betaTrue = [0.5, 0.3];
  const y = simulateGLM(X, betaTrue, FAMILIES.gamma, LINKS.log, seed + 1, {
    dispersion: 0.5,
  });
  return { X, y, betaTrue };
}

export const GLM_EXPLORER_PRESETS = {
  creditDefault: EXAMPLE_6_GLM_DATA,
  insuranceFrequency: EXAMPLE_7_GLM_DATA,
  insuranceAmounts: EXAMPLE_8_GLM_DATA,
  toyBernoulli: buildToyBernoulli(4001, 30),
  toyPoisson: buildToyPoisson(4002, 30),
  toyGamma: buildToyGamma(4003, 30),
};

/**
 * DevianceTestExplorer presets — H₀ true through large effect.
 * Each preset returns the full and reduced design matrices, the response, and
 * the offset (zero-vector for those without an offset).
 */
function buildDevianceTestPreset(
  seed: number,
  n: number,
  betaFull: number[],
): {
  Xfull: number[][];
  Xreduced: number[][];
  y: number[];
  offset: number[];
  betaTrue: number[];
} {
  const rng = seededRandom(seed);
  const Xfull: number[][] = new Array(n);
  const Xreduced: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const x1 = normalSample(0, 1, rng);
    const x2 = normalSample(0, 1, rng);
    Xfull[i] = [1, x1, x2];
    Xreduced[i] = [1, x1];
  }
  const offset = new Array<number>(n).fill(0);
  const y = simulateGLM(Xfull, betaFull, FAMILIES.poisson, LINKS.log, seed + 1);
  return { Xfull, Xreduced, y, offset, betaTrue: betaFull };
}

export const DEVIANCE_TEST_EXPLORER_PRESETS = {
  nullTrue: buildDevianceTestPreset(5001, 100, [0.5, 0.4, 0.0]),
  smallEffect: buildDevianceTestPreset(5002, 100, [0.5, 0.4, 0.2]),
  mediumEffect: buildDevianceTestPreset(5003, 100, [0.5, 0.4, 0.5]),
  largeEffect: buildDevianceTestPreset(5004, 100, [0.5, 0.4, 1.0]),
};

/**
 * SandwichCoverageSimulator presets — three misspecification flavors.
 * Each preset is a fixed seeded dataset generated once at module load (used
 * as the canonical "shape" of the DGP); the simulator re-runs its own seeded
 * draws via the GLM simulators below to vary the random response per
 * iteration. For the clustered-Binomial flavor the simulator must call
 * `simulateClusteredBinomial` (re-exported below) per draw rather than
 * re-using this fixed sample.
 */
function buildClusteredBinomial(
  seed: number,
  n: number,
): { X: number[][]; y: number[]; betaTrue: number[] } {
  // Cluster Bernoulli into groups of 5 with within-cluster correlation.
  const rng = seededRandom(seed);
  const X: number[][] = new Array(n);
  for (let i = 0; i < n; i++) X[i] = [1, normalSample(0, 1, rng)];
  const betaTrue = [0.0, 1.0];
  const y = new Array<number>(n);
  const eta = X.map((row) => row[0] * betaTrue[0] + row[1] * betaTrue[1]);
  const clusterSize = 5;
  const rho = 0.3;
  for (let g = 0; g < n / clusterSize; g++) {
    const u = normalSample(0, Math.sqrt(rho), rng);
    for (let k = 0; k < clusterSize && g * clusterSize + k < n; k++) {
      const i = g * clusterSize + k;
      const indiv = normalSample(0, Math.sqrt(1 - rho), rng);
      const latent = eta[i] + u + indiv;
      y[i] = latent > 0 ? 1 : 0;
    }
  }
  return { X, y, betaTrue };
}

export const SANDWICH_COVERAGE_PRESETS = {
  overdispersedPoisson: EXAMPLE_12_GLM_DATA,
  misspecifiedGamma: buildGammaInsuranceAmount(2241, 200),
  clusteredBinomial: buildClusteredBinomial(2242, 200),
};

// ═══════════════════════════════════════════════════════════════════════════
// §7.E — Topic 23: Regularization & Penalized Estimation data
// ═══════════════════════════════════════════════════════════════════════════
//
// Surface (brief §7.E):
//   §7.E.1  PROSTATE_CANCER_DATA — HTF Table 3.3 (canonical ridge/lasso bench)
//   §7.E.2  CREDIT_DEFAULT_EXPANDED_DATA — Topic 22 EX 6 × polynomial degree 3
//           plus polynomialFeatures helper exported for reuse
//   §7.E.3  Precomputed regularization paths + CV curves on prostate (IIFE)
//   §7.E.5  CORRELATED_FEATURES_DATA — synthetic n=200, p=10 with ρ=0.9 pairs
//
// §7.E.4 (reuse of EXAMPLE_10_GLM_DATA for Topic 23 §23.6 Ex 8) requires no
// new data — the existing Topic-22 export is consumed verbatim.
//
// Numeric values for PROSTATE_CANCER_DATA come from the HTF / ESL companion
// dataset at <https://hastie.su.domains/ElemStatLearn/datasets/prostate.data>;
// fetched once at authoring time and embedded here as TS literals so the site
// is fully offline-capable.

// ── §7.E.1  Prostate cancer (HTF Table 3.3) ─────────────────────────────────

/**
 * Hastie / Tibshirani / Friedman (2009 ESL 2nd ed.) Table 3.3 prostate-cancer
 * dataset, reproduced from Stamey et al. 1989 via the Stanford ESL companion
 * file `prostate.data`.
 *
 * n = 97 observations × 8 predictors. Response: lpsa (log prostate-specific
 * antigen). HTF's canonical 67/30 train/test split is provided.
 *
 * Predictors are NOT standardized — callers must center/scale before ridge
 * or lasso (the fit routines in regression.ts do this automatically when
 * `standardize: true`, the default).
 */
export const PROSTATE_CANCER_DATA: {
  X: number[][];
  y: number[];
  featureNames: readonly string[];
  train: boolean[];
} = {
  featureNames: ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45'] as const,
  X: [[-0.579818495,2.769459,50,-1.38629436,0,-1.38629436,6,0],[-0.994252273,3.319626,58,-1.38629436,0,-1.38629436,6,0],[-0.510825624,2.691243,74,-1.38629436,0,-1.38629436,7,20],[-1.203972804,3.282789,58,-1.38629436,0,-1.38629436,6,0],[0.751416089,3.432373,62,-1.38629436,0,-1.38629436,6,0],[-1.049822124,3.228826,50,-1.38629436,0,-1.38629436,6,0],[0.737164066,3.473518,64,0.61518564,0,-1.38629436,6,0],[0.693147181,3.539509,58,1.53686722,0,-1.38629436,6,0],[-0.776528789,3.539509,47,-1.38629436,0,-1.38629436,6,0],[0.223143551,3.244544,63,-1.38629436,0,-1.38629436,6,0],[0.254642218,3.604138,65,-1.38629436,0,-1.38629436,6,0],[-1.347073648,3.598681,63,1.2669476,0,-1.38629436,6,0],[1.613429934,3.022861,63,-1.38629436,0,-0.597837,7,30],[1.477048724,2.998229,67,-1.38629436,0,-1.38629436,7,5],[1.205970807,3.442019,57,-1.38629436,0,-0.43078292,7,5],[1.541159072,3.061052,66,-1.38629436,0,-1.38629436,6,0],[-0.415515444,3.516013,70,1.24415459,0,-0.597837,7,30],[2.288486169,3.649359,66,-1.38629436,0,0.37156356,6,0],[-0.562118918,3.267666,41,-1.38629436,0,-1.38629436,6,0],[0.182321557,3.825375,70,1.65822808,0,-1.38629436,6,0],[1.147402453,3.419365,59,-1.38629436,0,-1.38629436,6,0],[2.059238834,3.501043,60,1.47476301,0,1.34807315,7,20],[-0.544727175,3.37588,59,-0.7985077,0,-1.38629436,6,0],[1.781709133,3.451574,63,0.43825493,0,1.178655,7,60],[0.385262401,3.6674,69,1.59938758,0,-1.38629436,6,0],[1.446918983,3.124565,68,0.30010459,0,-1.38629436,6,0],[0.512823626,3.719651,65,-1.38629436,0,-0.7985077,7,70],[-0.400477567,3.865979,67,1.81645208,0,-1.38629436,7,20],[1.040276712,3.128951,67,0.22314355,0,0.04879016,7,80],[2.409644165,3.37588,65,-1.38629436,0,1.61938824,6,0],[0.285178942,4.090169,65,1.96290773,0,-0.7985077,6,0],[0.182321557,3.804438,65,1.70474809,0,-1.38629436,6,0],[1.2753628,3.037354,71,1.2669476,0,-1.38629436,6,0],[0.009950331,3.267666,54,-1.38629436,0,-1.38629436,6,0],[-0.010050336,3.216874,63,-1.38629436,0,-0.7985077,6,0],[1.30833282,4.11985,64,2.17133681,0,-1.38629436,7,5],[1.423108334,3.657131,73,-0.5798185,0,1.65822808,8,15],[0.457424847,2.374906,64,-1.38629436,0,-1.38629436,7,15],[2.660958594,4.085136,68,1.37371558,1,1.83258146,7,35],[0.797507196,3.013081,56,0.93609336,0,-0.16251893,7,5],[0.620576488,3.141995,60,-1.38629436,0,-1.38629436,9,80],[1.442201993,3.68261,68,-1.38629436,0,-1.38629436,7,10],[0.58221562,3.865979,62,1.71379793,0,-0.43078292,6,0],[1.771556762,3.896909,61,-1.38629436,0,0.81093022,7,6],[1.486139696,3.409496,66,1.74919985,0,-0.43078292,7,20],[1.663926098,3.392829,61,0.61518564,0,-1.38629436,7,15],[2.727852828,3.995445,79,1.87946505,1,2.65675691,9,100],[1.16315081,4.035125,68,1.71379793,0,-0.43078292,7,40],[1.745715531,3.498022,43,-1.38629436,0,-1.38629436,6,0],[1.220829921,3.568123,70,1.37371558,0,-0.7985077,6,0],[1.091923301,3.993603,68,-1.38629436,0,-1.38629436,7,50],[1.660131027,4.234831,64,2.07317193,0,-1.38629436,6,0],[0.512823626,3.633631,64,1.4929041,0,0.04879016,7,70],[2.12704052,4.121473,68,1.76644166,0,1.44691898,7,40],[3.153590358,3.516013,59,-1.38629436,0,-1.38629436,7,5],[1.266947603,4.280132,66,2.12226154,0,-1.38629436,7,15],[0.97455964,2.865054,47,-1.38629436,0,0.50077529,7,4],[0.463734016,3.764682,49,1.42310833,0,-1.38629436,6,0],[0.542324291,4.178226,70,0.43825493,0,-1.38629436,7,20],[1.061256502,3.851211,61,1.29472717,0,-1.38629436,7,40],[0.457424847,4.524502,73,2.32630162,0,-1.38629436,6,0],[1.997417706,3.719651,63,1.61938824,1,1.9095425,7,40],[2.77570885,3.524889,72,-1.38629436,0,1.55814462,9,95],[2.034705648,3.917011,66,2.00821403,1,2.1102132,7,60],[2.073171929,3.623007,64,-1.38629436,0,-1.38629436,6,0],[1.458615023,3.836221,61,1.32175584,0,-0.43078292,7,20],[2.02287119,3.878466,68,1.78339122,0,1.32175584,7,70],[2.198335072,4.050915,72,2.30757263,0,-0.43078292,7,10],[-0.446287103,4.408547,69,-1.38629436,0,-1.38629436,6,0],[1.193922468,4.780383,72,2.32630162,0,-0.7985077,7,5],[1.864080131,3.593194,60,-1.38629436,1,1.32175584,7,60],[1.160020917,3.341093,77,1.74919985,0,-1.38629436,7,25],[1.214912744,3.825375,69,-1.38629436,1,0.22314355,7,20],[1.838961071,3.236716,60,0.43825493,1,1.178655,9,90],[2.999226163,3.849083,69,-1.38629436,1,1.9095425,7,20],[3.141130476,3.263849,68,-0.05129329,1,2.42036813,7,50],[2.010894999,4.433789,72,2.12226154,0,0.50077529,7,60],[2.537657215,4.354784,78,2.32630162,0,-1.38629436,7,10],[2.648300197,3.582129,69,-1.38629436,1,2.58399755,7,70],[2.779440197,3.823192,63,-1.38629436,0,0.37156356,7,50],[1.467874348,3.070376,66,0.55961579,0,0.22314355,7,40],[2.513656063,3.473518,57,0.43825493,0,2.32727771,7,60],[2.613006652,3.888754,77,-0.52763274,1,0.55961579,7,30],[2.677590994,3.838376,65,1.11514159,0,1.74919985,9,70],[1.562346305,3.709907,60,1.69561561,0,0.81093022,7,30],[3.302849259,3.51898,64,-1.38629436,1,2.32727771,7,60],[2.024193067,3.731699,58,1.63899671,0,-1.38629436,6,0],[1.731655545,3.369018,62,-1.38629436,1,0.30010459,7,30],[2.807593831,4.718052,65,-1.38629436,1,2.46385324,7,60],[1.562346305,3.69511,76,0.93609336,1,0.81093022,7,75],[3.246490992,4.101817,68,-1.38629436,0,-1.38629436,6,0],[2.532902848,3.677566,61,1.34807315,1,-1.38629436,7,15],[2.830267834,3.876396,68,-1.38629436,1,1.32175584,7,60],[3.821003607,3.896909,44,-1.38629436,1,2.1690537,7,40],[2.907447359,3.396185,52,-1.38629436,1,2.46385324,7,10],[2.882563575,3.77391,68,1.55814462,1,1.55814462,7,80],[3.471966453,3.974998,68,0.43825493,1,2.90416508,7,20]],
  y: [-0.4307829,-0.1625189,-0.1625189,-0.1625189,0.3715636,0.7654678,0.7654678,0.8544153,1.047319,1.047319,1.2669476,1.2669476,1.2669476,1.3480731,1.3987169,1.446919,1.4701758,1.4929041,1.5581446,1.5993876,1.6389967,1.6582281,1.6956156,1.7137979,1.7316555,1.7664417,1.8000583,1.8164521,1.8484548,1.8946169,1.9242487,2.008214,2.008214,2.0215476,2.0476928,2.0856721,2.1575593,2.1916535,2.2137539,2.2772673,2.2975726,2.3075726,2.3272777,2.3749058,2.5217206,2.5533438,2.5687881,2.5687881,2.5915164,2.5915164,2.6567569,2.677591,2.6844403,2.6912431,2.7047113,2.7180005,2.7880929,2.7942279,2.8063861,2.8124102,2.8419982,2.8535925,2.8535925,2.8820035,2.8820035,2.8875901,2.9204698,2.9626924,2.9626924,2.9729753,3.0130809,3.0373539,3.0563569,3.0750055,3.2752562,3.3375474,3.3928291,3.4355988,3.4578927,3.5130369,3.5160131,3.5307626,3.5652984,3.5709402,3.5876769,3.6309855,3.6800909,3.7123518,3.9843437,3.993603,4.029806,4.1295508,4.3851468,4.6844434,5.1431245,5.477509,5.5829322],
  train: [true,true,true,true,true,true,false,true,false,false,true,true,true,true,false,true,true,true,true,true,true,false,true,true,false,false,true,false,true,true,true,false,true,false,true,false,true,true,true,true,true,false,true,false,true,true,true,false,false,false,true,true,false,false,false,true,false,true,true,true,true,false,true,false,false,false,true,true,true,true,true,true,false,false,true,true,true,true,true,false,true,true,true,false,true,true,true,true,true,true,true,true,true,true,false,true,false],
};

// ── §7.E.2  Polynomial-feature expansion + credit-default high-dim preset ──

/**
 * Polynomial feature expansion up to a given degree (inclusive). Mirrors
 * sklearn.preprocessing.PolynomialFeatures(include_bias=False). Generates
 * all unique combinations with replacement of the original features —
 * e.g. for p = 3, degree = 2 the columns are
 *   [x1, x2, x3, x1², x1·x2, x1·x3, x2², x2·x3, x3²].
 * For p original features and degree d, the expansion has
 *   C(p + d, d) − 1  total columns.
 */
export function polynomialFeatures(
  X: number[][],
  degree: number,
  options?: { includeBias?: boolean },
): { X: number[][]; featureNames: string[] } {
  const includeBias = options?.includeBias ?? false;
  if (X.length === 0) return { X: [], featureNames: [] };
  const p = X[0].length;
  if (degree < 1) throw new Error(`polynomialFeatures: degree ${degree} < 1`);

  // Enumerate every combinations-with-replacement multi-index up to degree.
  // Each multi-index is an array of feature indices (sorted, with repetition).
  const indices: number[][] = [];
  if (includeBias) indices.push([]);
  const enumerate = (current: number[], start: number, remaining: number): void => {
    if (current.length > 0) indices.push(current.slice());
    if (remaining === 0) return;
    for (let j = start; j < p; j++) {
      current.push(j);
      enumerate(current, j, remaining - 1);
      current.pop();
    }
  };
  enumerate([], 0, degree);

  const featureNames: string[] = indices.map((mi) => {
    if (mi.length === 0) return '1';
    // Group consecutive duplicates and emit "x_j^k" for k > 1.
    const counts = new Map<number, number>();
    for (const idx of mi) counts.set(idx, (counts.get(idx) ?? 0) + 1);
    const parts: string[] = [];
    for (const [idx, c] of counts.entries()) {
      parts.push(c === 1 ? `x${idx + 1}` : `x${idx + 1}^${c}`);
    }
    return parts.join('·');
  });

  const Xout: number[][] = X.map((row) =>
    indices.map((mi) => {
      let prod = 1;
      for (const idx of mi) prod *= row[idx];
      return prod;
    }),
  );

  return { X: Xout, featureNames };
}

/**
 * Credit-default logistic regression (Topic 22 §7.C EXAMPLE_6_GLM_DATA) with
 * a degree-3 polynomial expansion of its predictors. Used by §23.9 Ex 15
 * (lasso Poisson high-dim demo) and the `CrossValidationExplorer`'s
 * "credit-default" toggle. Drops the leading intercept column from
 * EXAMPLE_6's X (which is `[1, age_z, income_z]`) before expansion so the
 * polynomial features are purely on the two predictors.
 */
export const CREDIT_DEFAULT_EXPANDED_DATA: {
  X: number[][];
  y: number[];
  featureNames: string[];
} = (() => {
  // Strip EXAMPLE_6's intercept column (it's [1, age_z, income_z]; we want
  // the 2 real predictors only — polynomialFeatures will not create a bias).
  const Xraw = EXAMPLE_6_GLM_DATA.X.map((row) => row.slice(1));
  const expanded = polynomialFeatures(Xraw, 3);
  return {
    X: expanded.X,
    y: EXAMPLE_6_GLM_DATA.y,
    featureNames: expanded.featureNames,
  };
})();

// ── §7.E.5  Correlated-features synthetic data (LevelSetExplorer driver) ───

/**
 * Synthetic correlated-features design for §23.5 Ex 7 (grouping effect).
 * n = 200, p = 10. Pairs (X₁, X₂), (X₃, X₄), (X₅, X₆) have within-pair
 * correlation ρ = 0.9; the remaining four predictors are independent.
 * True β has nonzero entries only at indices 0 (X₁) and 2 (X₃).
 *
 * Elastic net (α ≈ 0.5) should activate the correlated partners X₂ and X₄
 * thanks to the grouping effect; pure lasso (α = 1) typically picks one
 * member of each correlated pair and zeroes the other.
 *
 * Seed: 42 — the figure must match the text.
 */
export const CORRELATED_FEATURES_DATA: {
  X: number[][];
  y: number[];
  betaTrue: number[];
  correlationStructure: { indices: [number, number]; rho: number }[];
} = (() => {
  const n = 200;
  const p = 10;
  const rho = 0.9;
  const rng = seededRandom(42);
  const correlatedPairs: [number, number][] = [
    [0, 1],
    [2, 3],
    [4, 5],
  ];
  const isCorrelated = new Array<number | undefined>(p);
  for (const [a, b] of correlatedPairs) isCorrelated[b] = a;

  const X: number[][] = new Array(n);
  const sigEpsilon = Math.sqrt(1 - rho * rho);
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(p);
    for (let j = 0; j < p; j++) {
      const partner = isCorrelated[j];
      if (partner !== undefined) {
        // X_j = ρ · X_partner + sqrt(1 − ρ²) · ε  (so correlation is exactly ρ).
        row[j] = rho * row[partner] + sigEpsilon * normalSample(0, 1, rng);
      } else {
        row[j] = normalSample(0, 1, rng);
      }
    }
    X[i] = row;
  }

  const betaTrue = new Array<number>(p).fill(0);
  betaTrue[0] = 2.0;
  betaTrue[2] = 1.5;
  const y = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let yi = 0;
    for (let j = 0; j < p; j++) yi += betaTrue[j] * X[i][j];
    yi += normalSample(0, 0.5, rng); // moderate noise
    y[i] = yi;
  }

  return {
    X,
    y,
    betaTrue,
    correlationStructure: correlatedPairs.map(([a, b]) => ({
      indices: [a, b] as [number, number],
      rho,
    })),
  };
})();

// ── §7.E.3  Precomputed regularization paths + CV curves (IIFE) ────────────
//
// Computed at module-load time using the fit functions in regression.ts.
// The cost is one ~50-150 ms compute per JS bundle execution (SSG: once at
// build; client-side: once per browser hydration of any component that
// imports these constants). The brief recommends precomputing to a JSON
// literal via a build script for browser-load avoidance — that's deferred
// tech debt; IIFE is acceptable for the prostate path size (n = 97, p = 8).

/** Ridge regularization path on prostate-cancer (100-point log-grid). */
export const PROSTATE_RIDGE_PATH: RegularizationPathResult = regularizationPath(
  PROSTATE_CANCER_DATA.X,
  PROSTATE_CANCER_DATA.y,
  { penalty: 'ridge', nLambda: 100 },
);

/** Lasso regularization path on prostate-cancer (100-point log-grid, warm-started). */
export const PROSTATE_LASSO_PATH: RegularizationPathResult = regularizationPath(
  PROSTATE_CANCER_DATA.X,
  PROSTATE_CANCER_DATA.y,
  { penalty: 'lasso', nLambda: 100 },
);

/** Elastic-net (α = 0.5) regularization path on prostate-cancer. */
export const PROSTATE_ELASTIC_NET_PATH: RegularizationPathResult = regularizationPath(
  PROSTATE_CANCER_DATA.X,
  PROSTATE_CANCER_DATA.y,
  { penalty: 'elasticnet', alpha: 0.5, nLambda: 100 },
);

/**
 * Precomputed CV results for `CrossValidationExplorer`. Keys:
 *   'prostate-ridge-k10-s42'
 *   'prostate-lasso-k10-s42'
 *   'prostate-elasticnet-k10-s42'   (α = 0.5)
 */
export const CV_RESULTS: Record<string, CVResult> = {
  'prostate-ridge-k10-s42': crossValidate(
    PROSTATE_CANCER_DATA.X,
    PROSTATE_CANCER_DATA.y,
    { penalty: 'ridge', k: 10, seed: 42, nLambda: 100 },
  ),
  'prostate-lasso-k10-s42': crossValidate(
    PROSTATE_CANCER_DATA.X,
    PROSTATE_CANCER_DATA.y,
    { penalty: 'lasso', k: 10, seed: 42, nLambda: 100 },
  ),
  'prostate-elasticnet-k10-s42': crossValidate(
    PROSTATE_CANCER_DATA.X,
    PROSTATE_CANCER_DATA.y,
    { penalty: 'elasticnet', alpha: 0.5, k: 10, seed: 42, nLambda: 100 },
  ),
};

/** Re-export the catalogs so component code can import everything from one place. */
export { FAMILIES, LINKS };
export type { GLMFamily, LinkFunction };
