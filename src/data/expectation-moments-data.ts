/**
 * expectation-moments-data.ts — Preset data for Topic 4 interactive components.
 *
 * Mixture model presets, distribution presets, MGF presets, inequality presets,
 * and Jensen function presets used by ExpectationBalanceExplorer,
 * VarianceDecompositionExplorer, InequalityExplorer, LawOfTotalExpectationExplorer,
 * and MGFExplorer.
 */

// ── Mixture Model Presets (LawOfTotalExpectationExplorer) ────────────────────

export interface MixtureSegment {
  label: string;
  weight: number;
  mean: number;
  std: number;
}

export interface MixturePreset {
  name: string;
  segments: MixtureSegment[];
  description: string;
}

export const mixtureModelPresets: MixturePreset[] = [
  {
    name: 'Customer Segments',
    segments: [
      { label: 'Casual', weight: 0.6, mean: 50, std: 20 },
      { label: 'Power User', weight: 0.4, mean: 120, std: 50 },
    ],
    description: 'Two customer segments with different spending patterns',
  },
  {
    name: 'Exam Scores (3 Sections)',
    segments: [
      { label: 'Section A', weight: 0.4, mean: 72, std: 8 },
      { label: 'Section B', weight: 0.35, mean: 78, std: 12 },
      { label: 'Section C', weight: 0.25, mean: 85, std: 5 },
    ],
    description: 'Three class sections with different performance profiles',
  },
  {
    name: 'Manufacturing Shifts',
    segments: [
      { label: 'Day Shift', weight: 0.5, mean: 10.0, std: 0.5 },
      { label: 'Night Shift', weight: 0.3, mean: 10.2, std: 0.8 },
      { label: 'Weekend', weight: 0.2, mean: 9.8, std: 1.2 },
    ],
    description: 'Product weight (grams) across three manufacturing shifts',
  },
];

// ── Expectation Presets (ExpectationBalanceExplorer) ─────────────────────────

export interface DiscretePreset {
  name: string;
  values: number[];
  probabilities: number[];
}

export interface ContinuousPreset {
  name: string;
  distribution: string;
  params: Record<string, number>;
}

export const expectationPresets = {
  discrete: [
    {
      name: 'Fair Die',
      values: [1, 2, 3, 4, 5, 6],
      probabilities: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    },
    {
      name: 'Loaded Die',
      values: [1, 2, 3, 4, 5, 6],
      probabilities: [0.05, 0.05, 0.10, 0.15, 0.25, 0.40],
    },
    {
      name: 'Bernoulli(0.7)',
      values: [0, 1],
      probabilities: [0.3, 0.7],
    },
    {
      name: 'A/B Test Variant A',
      values: [0, 5],
      probabilities: [0.6, 0.4],
    },
    {
      name: 'A/B Test Variant B',
      values: [0, 20],
      probabilities: [0.9, 0.1],
    },
  ] as DiscretePreset[],
  continuous: [
    { name: 'Normal(0, 1)', distribution: 'Normal', params: { mu: 0, sigma2: 1 } },
    { name: 'Normal(3, 0.5)', distribution: 'Normal', params: { mu: 3, sigma2: 0.5 } },
    { name: 'Exponential(1)', distribution: 'Exponential', params: { lambda: 1 } },
    { name: 'Exponential(3)', distribution: 'Exponential', params: { lambda: 3 } },
    { name: 'Uniform(0, 1)', distribution: 'Uniform', params: { a: 0, b: 1 } },
  ] as ContinuousPreset[],
};

// ── MGF Presets (MGFExplorer) ────��──────────────────────────────────────────

export interface MGFParamDef {
  name: string;
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface MGFPreset {
  name: string;
  paramDefs: MGFParamDef[];
  mgf: (t: number, params: Record<string, number>) => number;
  moments: (params: Record<string, number>) => { mean: number; variance: number };
  domain: [number, number | null];
}

export const mgfPresets: MGFPreset[] = [
  {
    name: 'Bernoulli(p)',
    paramDefs: [{ name: 'p', default: 0.5, min: 0.01, max: 0.99, step: 0.01 }],
    mgf: (t, params) => (1 - params.p) + params.p * Math.exp(t),
    moments: (params) => ({
      mean: params.p,
      variance: params.p * (1 - params.p),
    }),
    domain: [-3, 3],
  },
  {
    name: 'Binomial(n, p)',
    paramDefs: [
      { name: 'n', default: 10, min: 1, max: 30, step: 1 },
      { name: 'p', default: 0.3, min: 0.01, max: 0.99, step: 0.01 },
    ],
    mgf: (t, params) =>
      Math.pow((1 - params.p) + params.p * Math.exp(t), params.n),
    moments: (params) => ({
      mean: params.n * params.p,
      variance: params.n * params.p * (1 - params.p),
    }),
    domain: [-2, 2],
  },
  {
    name: 'Normal(\u03BC, \u03C3\u00B2)',
    paramDefs: [
      { name: 'mu', default: 0, min: -5, max: 5, step: 0.1 },
      { name: 'sigma2', default: 1, min: 0.1, max: 5, step: 0.1 },
    ],
    mgf: (t, params) =>
      Math.exp(params.mu * t + params.sigma2 * t * t / 2),
    moments: (params) => ({
      mean: params.mu,
      variance: params.sigma2,
    }),
    domain: [-2, 2],
  },
  {
    name: 'Exponential(\u03BB)',
    paramDefs: [{ name: 'lambda', default: 1, min: 0.1, max: 5, step: 0.1 }],
    mgf: (t, params) =>
      t < params.lambda ? params.lambda / (params.lambda - t) : NaN,
    moments: (params) => ({
      mean: 1 / params.lambda,
      variance: 1 / (params.lambda * params.lambda),
    }),
    domain: [-2, null], // upper bound is lambda - 0.1
  },
  {
    name: 'Poisson(\u03BB)',
    paramDefs: [{ name: 'lambda', default: 3, min: 0.5, max: 15, step: 0.5 }],
    mgf: (t, params) =>
      Math.exp(params.lambda * (Math.exp(t) - 1)),
    moments: (params) => ({
      mean: params.lambda,
      variance: params.lambda,
    }),
    domain: [-2, 2],
  },
];

// ── Inequality Presets (InequalityExplorer) ───��──────────────────────────────

export interface InequalityPreset {
  name: string;
  type: 'discrete' | 'continuous';
  distribution: string;
  params: Record<string, number>;
  defaultMarkovThreshold: number;
  defaultChebyshevK: number;
}

export const inequalityPresets: InequalityPreset[] = [
  {
    name: 'Exponential(\u03BB=0.5)',
    type: 'continuous',
    distribution: 'Exponential',
    params: { lambda: 0.5 },
    defaultMarkovThreshold: 4,
    defaultChebyshevK: 2,
  },
  {
    name: 'Normal(0, 1)',
    type: 'continuous',
    distribution: 'Normal',
    params: { mu: 0, sigma2: 1 },
    defaultMarkovThreshold: 2,
    defaultChebyshevK: 2,
  },
  {
    name: 'Uniform(0, 10)',
    type: 'continuous',
    distribution: 'Uniform',
    params: { a: 0, b: 10 },
    defaultMarkovThreshold: 7,
    defaultChebyshevK: 2,
  },
  {
    name: 'Binomial(20, 0.3)',
    type: 'discrete',
    distribution: 'Binomial',
    params: { n: 20, p: 0.3 },
    defaultMarkovThreshold: 10,
    defaultChebyshevK: 2,
  },
];

// ── Jensen Presets (InequalityExplorer) ──────────────────────────────────────

export interface JensenPreset {
  name: string;
  g: (x: number) => number;
  gPrime: (x: number) => number;
  label: string;
}

export const jensenPresets: JensenPreset[] = [
  {
    name: 'g(x) = x\u00B2',
    g: (x) => x * x,
    gPrime: (x) => 2 * x,
    label: '$x^2$',
  },
  {
    name: 'g(x) = e\u02E3',
    g: (x) => Math.exp(x),
    gPrime: (x) => Math.exp(x),
    label: '$e^x$',
  },
  {
    name: 'g(x) = -log(x)',
    g: (x) => -Math.log(x),
    gPrime: (x) => -1 / x,
    label: '$-\\log(x)$',
  },
];
