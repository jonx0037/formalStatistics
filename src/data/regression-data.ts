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
import { normalSample } from '../components/viz/shared/convergence';
import { simulateLinearModel } from '../components/viz/shared/regression';

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
