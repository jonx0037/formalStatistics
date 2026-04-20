/**
 * model-selection-data.ts — Topic 24 data presets + precomputed IC paths.
 *
 * Surface (brief §7):
 *   POLY_DGP                   — sin(2πx) + N(0,σ²); the canonical Topic-24 example,
 *                                consumed by ICSelector and CVvsICComparator
 *   POLY_TRUE_3 / 5 / 10       — DGP variants where the truth IS a polynomial of
 *                                the named degree (for the "true degree" radio)
 *   YANG_INCOMPAT_WELL         — Yang-race Tab A: truth = degree-3 polynomial,
 *                                BIC consistency holds
 *   YANG_INCOMPAT_MIS          — Yang-race Tab B: truth = sin(2πx) (polynomial-
 *                                misspecified), AIC minimax-rate dominates
 *   NESTED_POISSON             — §24.7 Ex 8: η = 1 + 0.8 x₁ - 0.5 x₂; x₃ null
 *   POLY_DGP_PRECOMPUTED       — IIFE-materialized POLY_DGP sample, per-degree
 *                                fits, and AIC/AICc/BIC/Cp/CV value arrays.
 *                                Computed once at module load (~50–150 ms);
 *                                ICSelector and CVvsICComparator import the
 *                                precomputed object verbatim and re-render in
 *                                O(1) on slider/toggle changes.
 *
 * Why this lives in its own file (not `regression-data.ts` or `regularization-
 * data.ts`): Topic 23's PR #27 split set the precedent — Topic-N data lives in
 * `Topic-N-data.ts` so the 16+ unrelated visualization bundles don't pay the
 * IIFE cost. Topic 24's only data-importing components are ICSelector,
 * CVvsICComparator, ConsistencyEfficiencyRace, and (for §24.8 Ex 10) the lasso
 * AIC-overlay, which re-imports PROSTATE_CANCER_DATA from regularization-data.ts.
 */

import { seededRandom } from '../components/viz/shared/probability';
import { normalSample } from '../components/viz/shared/convergence';
import {
  aic,
  aicc,
  bic,
  mallowsCp,
  kFoldCV,
  looCV,
  type OLSFit,
} from '../components/viz/shared/regression';
import { polyDesign, polyFitOLS } from '../components/viz/shared/polynomial';
import { PROSTATE_CANCER_DATA } from './regularization-data';

// Re-export so Topic-24 consumers can import everything from one module.
export { PROSTATE_CANCER_DATA };

// ─────────────────────────────────────────────────────────────────────────────
// Type definitions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Spec for a 1D polynomial-DGP preset. The DGP itself is materialized inside
 * the consuming widget (so seed/n/sigma overrides are cheap); this preset
 * captures the metadata for menu rendering and the default parameter values.
 */
export interface PolynomialPreset {
  /** Display name shown in the preset dropdown. */
  name: string;
  /** One-line description shown under the dropdown. */
  description: string;
  /** Default sample size. */
  n: number;
  /** Default Gaussian noise σ. */
  sigma: number;
  /** Seed for reproducibility (passed to the widget's RNG). */
  seed: number;
  /**
   * Human-readable description of the true regression function. The widget
   * also overlays this when "Show true model" is on. Examples:
   *   'sin(2πx)'
   *   'polynomial-d3 with [0.5, -1.5, 2.0, -1.0]'
   */
  trueFunction: string;
  /** True polynomial coefficients (low → high), if the truth is a polynomial. */
  trueCoefs?: number[];
  /** True polynomial degree, if the truth is a polynomial. */
  trueDegree?: number;
  /** [xMin, xMax]; uniform sampling on this interval. */
  xDomain: [number, number];
  /** Largest candidate model degree (cap on the slider). */
  maxDegree: number;
}

/** Spec for a Yang-incompatibility race scenario. */
export interface RacePreset {
  /** Display name shown in the tab radio. */
  name: string;
  /**
   * Description of the true model: 'polynomial-dN' or 'sin-2pi-x' (extensible
   * — components dispatch on this string).
   */
  trueModel: 'polynomial-d3' | 'sin-2pi-x';
  /** True polynomial coefficients (only when trueModel is 'polynomial-dN'). */
  trueCoefs?: number[];
  /** Candidate model degrees the race ranges over. */
  candidateDegrees: number[];
  /** Gaussian noise σ for the race DGP. */
  sigma: number;
  /** Sample sizes plotted on the x-axis of the risk-vs-n panel. */
  nSweep: number[];
  /** Base seed. Per-replicate seeds are derived as seed + repIndex. */
  seed: number;
}

/** Spec for a GLM-DGP preset (currently nested-Poisson only). */
export interface GLMPreset {
  /** Display name. */
  name: string;
  /** Exponential-family name; passed to glmFit's family lookup. */
  family: 'poisson' | 'binomial' | 'gamma' | 'normal';
  /** Canonical-link name; passed to glmFit's link lookup. */
  link: 'log' | 'logit' | 'identity' | 'inverse';
  /**
   * True coefficient vector. The DGP draws x_j ~ U(-1, 1) for each non-
   * intercept slot and sets η = X · trueCoefs. Trailing zero entries
   * indicate null-effect predictors (the §24.7 hook for nested LRT).
   */
  trueCoefs: number[];
  /** Sample size. */
  n: number;
  /** Seed for reproducibility. */
  seed: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Polynomial DGPs
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Standard POLY_DGP: y = sin(2πx) + σ·N(0,1), n=80, σ=0.25, seed=42.
 *
 * The canonical Topic-24 DGP, reused across §24.1–§24.9 and Figures 1, 2, 3,
 * 5, 7, 8 (all except 4 BIC-Laplace toy, 7 nested-GLM, 8 prostate lasso).
 * Argmin AIC/AICc/Cp/CV all land at d=6; argmin BIC at d=3 (BIC favors
 * sparser). See T10.1–T10.11 for pinned IC values.
 */
export const POLY_DGP: PolynomialPreset = {
  name: 'sin(2πx) + Gaussian noise (Topic-24 canonical)',
  description: 'Smooth-but-not-polynomial truth — the running example.',
  n: 80,
  sigma: 0.25,
  seed: 42,
  trueFunction: 'sin(2πx)',
  xDomain: [0, 1],
  maxDegree: 12,
};

/** Polynomial truth at degree 3 — when the truth IS in the candidate set. */
export const POLY_TRUE_3: PolynomialPreset = {
  name: 'Polynomial degree 3 (well-specified)',
  description: 'True model is a polynomial — BIC selection-consistent.',
  n: 80,
  sigma: 0.3,
  seed: 42,
  trueFunction: 'polynomial-d3 with [0.5, -1.5, 2.0, -1.0]',
  trueCoefs: [0.5, -1.5, 2.0, -1.0],
  trueDegree: 3,
  xDomain: [0, 1],
  maxDegree: 12,
};

/** Polynomial truth at degree 5. */
export const POLY_TRUE_5: PolynomialPreset = {
  name: 'Polynomial degree 5 (well-specified)',
  description: 'Higher-degree polynomial truth — illustrates AIC/BIC at moderate complexity.',
  n: 80,
  sigma: 0.3,
  seed: 42,
  trueFunction: 'polynomial-d5 with [0.2, 0.8, -1.6, 1.0, -0.7, 0.4]',
  trueCoefs: [0.2, 0.8, -1.6, 1.0, -0.7, 0.4],
  trueDegree: 5,
  xDomain: [0, 1],
  maxDegree: 12,
};

/** Polynomial truth at degree 10 — near the candidate-set ceiling. */
export const POLY_TRUE_10: PolynomialPreset = {
  name: 'Polynomial degree 10 (well-specified, high-dim)',
  description: 'Truth near the candidate-set ceiling — AICc correction matters.',
  n: 80,
  sigma: 0.3,
  seed: 42,
  trueFunction:
    'polynomial-d10 with [0.1, 0.5, -0.8, 0.6, -0.4, 0.3, -0.2, 0.15, -0.1, 0.08, -0.05]',
  trueCoefs: [0.1, 0.5, -0.8, 0.6, -0.4, 0.3, -0.2, 0.15, -0.1, 0.08, -0.05],
  trueDegree: 10,
  xDomain: [0, 1],
  maxDegree: 12,
};

// ─────────────────────────────────────────────────────────────────────────────
// Yang-incompatibility race presets
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Yang race Tab A — well-specified.
 *
 * Truth is a degree-3 polynomial; candidate set d ∈ [0, 8] includes the truth.
 * As n grows, BIC selects d=3 with probability → 1 (selection consistency).
 * AIC has a persistent positive probability of selecting d ≥ 4 (asymptotic
 * over-fitting). At any finite n, both have similar prediction risk; the
 * divergence is in selection behavior.
 */
export const YANG_INCOMPAT_WELL: RacePreset = {
  name: 'Yang Tab A: truth is degree-3 polynomial',
  trueModel: 'polynomial-d3',
  trueCoefs: [0.5, -1.5, 2.0, -1.0],
  candidateDegrees: [0, 1, 2, 3, 4, 5, 6, 7, 8],
  sigma: 0.3,
  nSweep: [50, 100, 200, 500, 1000, 2000, 5000],
  seed: 42,
};

/**
 * Yang race Tab B — misspecified.
 *
 * Truth is sin(2πx); candidate set is polynomials d ∈ [0, 12] (truth NOT in
 * the candidate set). BIC's selection target shifts with n (each n has a
 * different "best polynomial approximation"); AIC tracks the minimax-rate
 * polynomial approximation more aggressively. The divergence in prediction
 * risk is the qualitative content of Yang 2005's incompatibility theorem.
 */
export const YANG_INCOMPAT_MIS: RacePreset = {
  name: 'Yang Tab B: truth is sin(2πx) (polynomial-misspecified)',
  trueModel: 'sin-2pi-x',
  candidateDegrees: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  sigma: 0.25,
  nSweep: [50, 100, 200, 500, 1000, 2000, 5000],
  seed: 42,
};

// ─────────────────────────────────────────────────────────────────────────────
// Nested-Poisson preset
// ─────────────────────────────────────────────────────────────────────────────

/**
 * §24.7 Ex 8 nested-Poisson preset: x₃ has no true effect.
 *
 * Generated via NumPy's default_rng(123) to match the brief §6.2 pinned T10.12–
 * T10.19 reference values (β̂_red[0] ≈ 0.9815, LR ≈ 0.5547, ΔAIC ≈ 1.4453,
 * ΔBIC ≈ 4.7436). A consuming widget should replicate the same seed strategy
 * (default_rng), or import the test fixtures directly if the visualization
 * needs to display the same values shown in the §24.7 worked-example table.
 */
export const NESTED_POISSON: GLMPreset = {
  name: 'Nested Poisson: x₃ null effect',
  family: 'poisson',
  link: 'log',
  trueCoefs: [1.0, 0.8, -0.5, 0.0], // intercept, x1, x2, x3 (null)
  n: 200,
  seed: 123,
};

// ─────────────────────────────────────────────────────────────────────────────
// Convenience aggregations
// ─────────────────────────────────────────────────────────────────────────────

/** All polynomial presets, in dropdown-display order. */
export const POLYNOMIAL_PRESETS: readonly PolynomialPreset[] = [
  POLY_DGP,
  POLY_TRUE_3,
  POLY_TRUE_5,
  POLY_TRUE_10,
] as const;

/** All Yang-race presets, in tab-display order. */
export const YANG_RACE_PRESETS: readonly RacePreset[] = [
  YANG_INCOMPAT_WELL,
  YANG_INCOMPAT_MIS,
] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Precomputed POLY_DGP sample + per-degree fits + IC paths
// ─────────────────────────────────────────────────────────────────────────────

/** Inclusive degree range used for IC computation across components. */
const POLY_DEG_MIN = 0;
const POLY_DEG_MAX = 12;
/** Deterministic seed for the CV fold partitions. Held constant across renders. */
const POLY_CV_SEED = 1234;
/** Maximum degree for which the LOO hat-matrix shortcut is numerically safe. */
const LOO_MAX_DEG = 11;

/**
 * Materialized POLY_DGP sample + the full IC ranking machinery the §24.3 / §24.5
 * components consume. Computed once at module load via the IIFE pattern that
 * Topic 23's `PROSTATE_RIDGE_PATH` / `PROSTATE_LASSO_PATH` established. Cost
 * for n=80 / 13 fits / 10-fold CV is ≈ 150 OLS solves per IIFE entry, well
 * inside Topic 23's 50–150 ms budget.
 *
 * The raw `x` and `y` vectors are also exposed so the components can render
 * scatter plots of the underlying data without re-generating it from the seed.
 */
export interface PolyDGPPrecomputed {
  /** Predictor sample (length n), uniform on [0, 1]. */
  x: number[];
  /** Response sample (length n) from sin(2πx) + N(0, σ²). */
  y: number[];
  /** Sample size n (= POLY_DGP.n). */
  n: number;
  /** Noise standard deviation σ (= POLY_DGP.sigma). */
  sigma: number;
  /** Candidate polynomial degrees [0, 1, …, 12]. */
  degrees: number[];
  /** OLSFit per degree. fits[i] is the fit at degrees[i]. */
  fits: OLSFit[];
  /** σ̂²_ref = MLE σ² from the largest-degree fit (used by Mallows' Cp). */
  sigmaSqRef: number;
  /** Per-degree IC values (parallel arrays indexed by `degrees`). */
  values: {
    aic: number[];
    aicc: number[];
    bic: number[];
    cp: number[];
    /** 5-fold CV mean squared error per degree. */
    cv5: number[];
    /** 10-fold CV mean squared error per degree (Stone-equivalence default). */
    cv10: number[];
    /** Leave-one-out CV via hat-matrix shortcut. NaN for degrees where the
     *  monomial Vandermonde is too ill-conditioned (d > LOO_MAX_DEG = 11). */
    looCV: number[];
  };
  /** Argmin degree per criterion. NaN-safe for `looCV`. */
  argmins: {
    aic: number;
    aicc: number;
    bic: number;
    cp: number;
    cv5: number;
    cv10: number;
    looCV: number;
  };
}

export const POLY_DGP_PRECOMPUTED: PolyDGPPrecomputed = (() => {
  const { n, sigma, seed } = POLY_DGP;
  const rng = seededRandom(seed);
  const x: number[] = new Array(n);
  for (let i = 0; i < n; i++) x[i] = rng();
  const y: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    y[i] = Math.sin(2 * Math.PI * x[i]) + normalSample(0, sigma, rng);
  }
  const degrees = Array.from(
    { length: POLY_DEG_MAX - POLY_DEG_MIN + 1 },
    (_, i) => POLY_DEG_MIN + i,
  );
  const fits = degrees.map((d) => polyFitOLS(x, y, d));
  // σ̂²_ref: MLE from the largest model (Mallows' Cp convention).
  const lastFit = fits[fits.length - 1];
  const sigmaSqRef = lastFit.sse / lastFit.n;
  const aicArr = fits.map((f) => aic(f));
  const aiccArr = fits.map((f) => aicc(f, n));
  const bicArr = fits.map((f) => bic(f, n));
  const cpArr = fits.map((f) => mallowsCp(f, sigmaSqRef));
  const cv5Arr = degrees.map((d) =>
    kFoldCV(polyDesign(x, d), y, 5, undefined, POLY_CV_SEED),
  );
  const cv10Arr = degrees.map((d) =>
    kFoldCV(polyDesign(x, d), y, 10, undefined, POLY_CV_SEED),
  );
  // LOO via hat-matrix shortcut throws on near-singular designs; fall back
  // to NaN at the high-d tail where the monomial Vandermonde fails the
  // 0.999-leverage check inside looCV.
  const looArr = degrees.map((d) => {
    if (d > LOO_MAX_DEG) return NaN;
    try {
      return looCV(polyDesign(x, d), y);
    } catch {
      return NaN;
    }
  });
  /** Argmin over finite values; returns NaN if every value is NaN. */
  const argmin = (arr: number[]): number => {
    let best = -1;
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) continue;
      if (best === -1 || arr[i] < arr[best]) best = i;
    }
    return best === -1 ? NaN : degrees[best];
  };
  return {
    x,
    y,
    n,
    sigma,
    degrees,
    fits,
    sigmaSqRef,
    values: {
      aic: aicArr,
      aicc: aiccArr,
      bic: bicArr,
      cp: cpArr,
      cv5: cv5Arr,
      cv10: cv10Arr,
      looCV: looArr,
    },
    argmins: {
      aic: argmin(aicArr),
      aicc: argmin(aiccArr),
      bic: argmin(bicArr),
      cp: argmin(cpArr),
      cv5: argmin(cv5Arr),
      cv10: argmin(cv10Arr),
      looCV: argmin(looArr),
    },
  };
})();
