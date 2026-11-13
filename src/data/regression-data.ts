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
  FAMILIES,
  LINKS,
  type GLMFamily,
  type LinkFunction,
} from '../components/viz/shared/regression';

// ─────────────────────────────────────────────────────────────────────────────
// §7.A — Canonical small datasets for MDX examples
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Example 2 (§21.2): 5-point worked example for OLS hand-computation.
 * Hand-computable slope = 0.98, intercept = 0.16 (ȳ = 3.1, x̄ = 3, Sxy = 9.8).
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
      'All assumptions met. No pattern in residuals. Q-Q plot is near-linear. Leverage values cluster below 2(p+1)/n. The Cook’s D > 4/n rule still flags a few points in red by chance — it is a screening threshold, not a test.',
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
      'One observation has h_ii ≫ 2(p+1)/n but follows the same line, so its residual is ordinary. High leverage on its own mainly sharpens the slope estimate (higher precision); influence needs leverage and a large residual together. Cook’s D = r²/(p+1) · h/(1−h) combines both, and with h/(1−h) this large even an ordinary residual can cross the 4/n screen — check the residual before calling the point influential. See §21.9 Rem 13 and Rem 20.',
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
  // Matches §22.9 Ex 12 and the notebook coverage table: β₀ = (−0.3, 0.8).
  const betaTrue = [-0.3, 0.8];
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
 * Example 10 (§22.8): near-separation logistic, n = 80, x ~ N(0, 2²),
 * β_true = (0, 4). The literal draw from the companion notebook's Example 10
 * cell: numpy `default_rng(14992)`, x first, then y ~ Bernoulli(expit(4x)).
 * Stored verbatim because numpy's PCG64 stream can't be replayed here.
 * β̂ = (0.293, 4.136); 95% CIs for β₁: Wald-z [1.416, 6.856], profile-LRT
 * [2.160, 7.793] (upper half ≈ 1.85× lower half) — T8.15b checks all three.
 */
const EXAMPLE_10_X: number[] = [
  -1.1742625951829107, -2.0158260391698608, +0.894137232833631, -2.5607130426286466,
  -1.5656423233191374, +0.27785691629007592, -0.4141551015106314, +0.089992657396867917,
  -1.2656865389578096, +0.94048825648901291, -4.1744993212750652, -0.52160907403047474,
  +2.8624168105124994, +0.18478170606918445, -0.29241187423065029, -2.8702280622792471,
  -0.13911329865950917, -3.4882628415826917, -2.4778836822856589, +2.0699565791288523,
  +4.25775646909418, -1.4341800432804095, -4.5268529173924819, -1.204311161354666,
  -2.3295498127636978, -2.8529031291648757, -4.0938024333350995, +1.7006450169893743,
  -0.016569000866683187, +0.81343193436574868, +2.2697489909488575, -2.3701519188346354,
  -2.0709848603412793, -0.2946549733638727, +0.49443271593507654, +1.0228655749612412,
  -0.12582196997527839, -2.8484650358753147, +1.2199256413785753, -2.5448164652251872,
  +0.27030449914450938, +1.1630468377193066, -1.2296182931942514, -2.4753733796352111,
  -2.0297087751136891, -1.6533124538586026, +2.5371697216180911, -0.5336641083776833,
  +0.89391146719862502, -0.96483491113223141, -1.8938206884100695, +1.6494784773927922,
  +0.040435903498982691, +2.5474670928117664, -0.74022856391028435, -0.18482477685949253,
  -0.17230211717074756, +1.8011137153198584, +2.9918882927321304, +1.0910333403006343,
  -2.3929948367301095, -0.44242675757505567, -1.1419019668546915, +0.89076270569728133,
  +0.13916644875761233, -0.79880612524559369, -1.5658457308872269, -0.37399784325130658,
  -4.9190605883381409, +2.207971425694208, -1.9036884144886144, +3.7600883117739792,
  -0.018302733462510729, -2.062733492279186, +1.3467847685498882, -1.7406492702019005,
  -1.7851375254288979, -5.1245869026035962, +2.6375763855045817, -0.066021974964439151,
];
const EXAMPLE_10_Y: number[] = [
  0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1,
  1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0,
  0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1,
  0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
];

export const EXAMPLE_10_GLM_DATA: GLMDataset = {
  X: EXAMPLE_10_X.map((xi) => [1, xi]),
  y: EXAMPLE_10_Y,
  betaTrue: [0.0, 4.0],
};

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
 * convergence regimes. `separated` is completely separated (seed 3184: every
 * y = 0 point has x ≤ −0.043 and every y = 1 point has x ≥ 0.222), so the
 * logistic MLE does not exist and IRLS iterates diverge (‖β‖ grows by ≈ 7.6
 * per step after the first few iterations). The seed was chosen by search
 * under the same DGP the old near-separation preset used, (n, β) = (40, (0, 4.5)).
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
  separated: buildIRLSPreset(3184, 40, [0.0, 4.5]),
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

/** Re-export the catalogs so component code can import everything from one place. */
export { FAMILIES, LINKS };
export type { GLMFamily, LinkFunction };
