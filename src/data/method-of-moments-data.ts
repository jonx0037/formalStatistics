/**
 * method-of-moments-data.ts — preset configurations for the five
 * Topic-15 interactive components.
 *
 * Conventions: every preset is a plain object literal so components can
 * `import { gammaPresets }` and bind directly to UI controls. Numeric
 * defaults match the brief and the reference notebook; descriptions are
 * the short copy that surfaces in the explorer's "info" tooltip.
 */

// ── Gamma presets for MomGammaSolver ────────────────────────────────────────

export interface GammaPreset {
  name: string;
  trueAlpha: number;
  trueBeta: number;
  defaultN: number;
  description: string;
}

export const gammaPresets: GammaPreset[] = [
  {
    name: 'Gamma(3, 2) — balanced',
    trueAlpha: 3,
    trueBeta: 2,
    defaultN: 100,
    description:
      "The canonical 'MoM is two lines of algebra, MLE is Newton-Raphson' example. ARE ≈ 0.68 at α = 3.",
  },
  {
    name: 'Gamma(0.8, 1) — near the shape boundary',
    trueAlpha: 0.8,
    trueBeta: 1,
    defaultN: 200,
    description:
      'At small α, MoM is visibly unstable — S² is small and the ratio X̄²/S² is noisy.',
  },
  {
    name: 'Gamma(10, 1) — large shape',
    trueAlpha: 10,
    trueBeta: 1,
    defaultN: 100,
    description:
      'At large α, Gamma approaches Normal; MoM and MLE become nearly equivalent. ARE ≈ 0.95.',
  },
];

// ── Family presets for AREExplorer ──────────────────────────────────────────

export interface AREFamilyPreset {
  name: string;
  family: 'gamma-shape' | 'exponential-rate' | 'normal-variance' | 'uniform-endpoint';
  defaultParams: Record<string, number>;
  theoreticalAREFn: string;
  description: string;
  paramGrid: number[];
  sweepParam: string;
}

export const areFamilyPresets: AREFamilyPreset[] = [
  {
    name: 'Gamma shape α — ARE < 1',
    family: 'gamma-shape',
    defaultParams: { alpha: 3, beta: 2 },
    theoreticalAREFn: '1 / [2(α+1)(α·ψ′(α) − 1)]',
    description:
      'The canonical ARE < 1 example. Depends on the trigamma function ψ′(α). At α = 3, ARE ≈ 0.68.',
    paramGrid: [0.5, 1, 2, 3, 5, 10, 20],
    sweepParam: 'alpha',
  },
  {
    name: 'Exponential rate λ — ARE = 1',
    family: 'exponential-rate',
    defaultParams: { lambda: 1 },
    theoreticalAREFn: '1',
    description:
      'Exponential family in the natural parameter: MoM ≡ MLE exactly. The ARE = 1 boundary.',
    paramGrid: [0.5, 1, 2, 5],
    sweepParam: 'lambda',
  },
  {
    name: 'Normal variance σ² — both biased, both equal',
    family: 'normal-variance',
    defaultParams: { mu: 0, sigma2: 4 },
    theoreticalAREFn: '1',
    description:
      'Both MoM and MLE use the biased (1/n) estimator. ARE = 1, but both are dominated by the unbiased S²_{n-1}.',
    paramGrid: [1, 4, 9],
    sweepParam: 'sigma2',
  },
  {
    name: 'Uniform(0, θ) — MoM dramatically worse',
    family: 'uniform-endpoint',
    defaultParams: { theta: 1, n: 50 },
    theoreticalAREFn: '3 / (n + 2)  →  0',
    description:
      'MoM is 2X̄ (variance O(1/n)); MLE is X_(n) (variance O(1/n²)). ARE → 0 as n grows — the classic cautionary example.',
    paramGrid: [10, 25, 50, 100, 250, 500],
    sweepParam: 'n',
  },
];

// ── Pathology presets for CauchyPathologyExplorer ───────────────────────────

export interface PathologyPreset {
  name: string;
  family: 'cauchy' | 'pareto' | 'normal';
  params: Record<string, number>;
  hasMean: boolean;
  hasVariance: boolean;
  medianExists: boolean;
  trueMedian: number;
  description: string;
}

export const pathologyPresets: PathologyPreset[] = [
  {
    name: 'Cauchy(0, 1) — no mean, MoM undefined',
    family: 'cauchy',
    params: { location: 0, scale: 1 },
    hasMean: false,
    hasVariance: false,
    medianExists: true,
    trueMedian: 0,
    description:
      'The textbook failure case. E[|X|] = ∞, so the SLLN does not apply — the sample mean never converges. The median converges to 0 smoothly.',
  },
  {
    name: 'Pareto(α = 0.8) — mean is infinite',
    family: 'pareto',
    params: { alpha: 0.8, xm: 1 },
    hasMean: false,
    hasVariance: false,
    medianExists: true,
    trueMedian: Math.pow(2, 1 / 0.8), // xm · 2^{1/α}
    description:
      'Tail index α ≤ 1 means E[X] = ∞. The sample mean grows without bound as n grows — there is no population moment to match.',
  },
  {
    name: 'Pareto(α = 1.5) — mean exists, variance does not',
    family: 'pareto',
    params: { alpha: 1.5, xm: 1 },
    hasMean: true,
    hasVariance: false,
    medianExists: true,
    trueMedian: Math.pow(2, 1 / 1.5),
    description:
      'E[X] = xₘ · α/(α−1) = 3, but E[X²] = ∞. MoM for the mean is consistent; MoM for the variance fails.',
  },
  {
    name: 'Normal(0, 1) — baseline',
    family: 'normal',
    params: { mu: 0, sigma2: 1 },
    hasMean: true,
    hasVariance: true,
    medianExists: true,
    trueMedian: 0,
    description:
      'All moments exist; both mean and median converge. The contrast case.',
  },
];

// ── WarmStartExplorer presets ───────────────────────────────────────────────

export interface WarmStartPreset {
  name: string;
  family: 'Gamma';
  paramName: 'alpha';
  trueAlpha: number;
  trueBeta: number;
  defaultN: number;
  startingPoints: Array<
    | { label: string; strategy: 'mom' }
    | { label: string; strategy: 'fixed'; value: number }
    | { label: string; strategy: '2x-mom' }
  >;
  expectedIterations: { mom: number; small: number; large: number };
  description: string;
}

export const warmStartPresets: WarmStartPreset[] = [
  {
    name: 'Gamma(3, 2) shape — MoM init vs alternatives',
    family: 'Gamma',
    paramName: 'alpha',
    trueAlpha: 3,
    trueBeta: 2,
    defaultN: 100,
    startingPoints: [
      { label: 'MoM init (α̂_MoM)', strategy: 'mom' },
      { label: 'Small (α = 0.5)', strategy: 'fixed', value: 0.5 },
      { label: 'Large (2 × α̂_MoM)', strategy: '2x-mom' },
    ],
    expectedIterations: { mom: 4, small: 9, large: 10 },
    description:
      'MoM init typically converges in 3–5 Newton steps. Small init wanders through the left tail; large init can overshoot on the first step.',
  },
];

// ── Robust-estimation contamination presets for MEstimatorGallery ───────────

export interface ContaminationPreset {
  name: string;
  base: { family: 'Normal'; mu: number; sigma: number };
  contamination: { fraction: number; magnitude: number };
  description: string;
}

export const robustContaminationPresets: ContaminationPreset[] = [
  {
    name: 'Clean Normal(5, 1)',
    base: { family: 'Normal', mu: 5, sigma: 1 },
    contamination: { fraction: 0, magnitude: 0 },
    description:
      'The uncontaminated baseline. All four estimators (mean, median, Huber, Tukey) should agree.',
  },
  {
    name: 'Normal(5, 1) + 10% outliers at +20σ',
    base: { family: 'Normal', mu: 5, sigma: 1 },
    contamination: { fraction: 0.1, magnitude: 20 },
    description:
      'Mean is pulled to ≈ 7; median stays at 5; Huber (k = 1.345) stays near 5; Tukey (c = 4.685) stays at 5.',
  },
  {
    name: 'Normal(5, 1) + 30% outliers at +10σ',
    base: { family: 'Normal', mu: 5, sigma: 1 },
    contamination: { fraction: 0.3, magnitude: 10 },
    description:
      "Empirically, the Huber estimate drifts here because the MAD scale itself is corrupted by 30% outliers. Tukey's redescending ψ holds longer. Both have asymptotic breakdown 1/2.",
  },
  {
    name: 'Normal(5, 1) + 40% contamination',
    base: { family: 'Normal', mu: 5, sigma: 1 },
    contamination: { fraction: 0.4, magnitude: 15 },
    description:
      'Approaching the asymptotic breakdown of 1/2. All three robust estimators (median, Huber-with-MAD, Tukey) drift in finite samples; only the median keeps a tight bound. Pareto frontier of efficiency vs robustness.',
  },
];

// ── Tuning constants with provenance ────────────────────────────────────────

export const tuningConstants = {
  huber: {
    default: 1.345,
    rationale:
      '95% asymptotic efficiency under Normal data while bounding the influence function.',
    source: 'Huber (1964)',
  },
  tukey: {
    default: 4.685,
    rationale:
      '95% asymptotic efficiency under Normal with a redescending ψ that rejects outliers beyond |u| > 4.685σ.',
    source: 'Beaton & Tukey (1974)',
  },
  trimmedMean: {
    default: 0.1,
    rationale:
      '10% trimming: standard recommendation for moderate robustness with minimal efficiency loss.',
    source: 'Tukey (1960)',
  },
};

// ── §15.12 GMM preview (text snippets for the MDX, single source of truth) ──

export const gmmPreview = {
  definition:
    'GMM extends MoM to over-identified systems: more moment equations than parameters. Given moment conditions g(X; θ) with E[g(X; θ₀)] = 0, the GMM estimator minimizes ĝₙ(θ)ᵀ W ĝₙ(θ) for a weighting matrix W.',
  efficientGMM:
    'The efficient GMM uses the optimal weighting matrix (the inverse of the asymptotic variance of the moment conditions), achieving the lowest asymptotic variance among GMM estimators.',
  applications: [
    'Instrumental variables (IV) estimation in econometrics',
    'Euler-equation estimation for consumption models',
    'Asset-pricing moment conditions',
    'Dynamic panel data models (Arellano-Bond)',
  ],
  reference: 'Hansen (1982), Econometrica 50(4): 1029–1054.',
};
