/**
 * random-variables-data.ts — Preset data for Topic 3 interactive components.
 */

// ── Random Variable Mapping Presets ─────────────────────────────────────────

export interface RVMappingPreset {
  name: string;
  omega: string[];
  mappings: Record<string, Record<string, number>>;
}

/** Preset random variable mappings for RandomVariableMappingExplorer. */
export const rvMappingPresets: RVMappingPreset[] = [
  {
    name: 'Die Roll',
    omega: ['1', '2', '3', '4', '5', '6'],
    mappings: {
      'Identity X(ω) = ω': { '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6 },
      'Squared X(ω) = ω²': { '1': 1, '2': 4, '3': 9, '4': 16, '5': 25, '6': 36 },
      'Even Indicator': { '1': 0, '2': 1, '3': 0, '4': 1, '5': 0, '6': 1 },
    },
  },
  {
    name: 'Two Coins',
    omega: ['HH', 'HT', 'TH', 'TT'],
    mappings: {
      '# Heads': { 'HH': 2, 'HT': 1, 'TH': 1, 'TT': 0 },
      'First Coin (H=1)': { 'HH': 1, 'HT': 1, 'TH': 0, 'TT': 0 },
      'Match Indicator': { 'HH': 1, 'HT': 0, 'TH': 0, 'TT': 1 },
    },
  },
  {
    name: 'Two Dice',
    omega: (() => {
      const outcomes: string[] = [];
      for (let i = 1; i <= 6; i++) {
        for (let j = 1; j <= 6; j++) {
          outcomes.push(`(${i},${j})`);
        }
      }
      return outcomes;
    })(),
    mappings: {
      'Sum': (() => {
        const m: Record<string, number> = {};
        for (let i = 1; i <= 6; i++) {
          for (let j = 1; j <= 6; j++) {
            m[`(${i},${j})`] = i + j;
          }
        }
        return m;
      })(),
      'Max': (() => {
        const m: Record<string, number> = {};
        for (let i = 1; i <= 6; i++) {
          for (let j = 1; j <= 6; j++) {
            m[`(${i},${j})`] = Math.max(i, j);
          }
        }
        return m;
      })(),
    },
  },
];

// ── Distribution Presets ────────────────────────────────────────────────────

export interface DistributionPreset {
  name: string;
  params: Record<string, number>;
  paramRanges: Record<string, [number, number]>;
}

/** Discrete distribution presets for PMFPDFExplorer. */
export const discreteDistributionPresets: DistributionPreset[] = [
  { name: 'Bernoulli', params: { p: 0.5 }, paramRanges: { p: [0.01, 0.99] } },
  { name: 'Binomial', params: { n: 10, p: 0.3 }, paramRanges: { n: [1, 30], p: [0.01, 0.99] } },
  { name: 'Geometric', params: { p: 0.3 }, paramRanges: { p: [0.05, 0.95] } },
  { name: 'Poisson', params: { lambda: 5 }, paramRanges: { lambda: [0.5, 20] } },
];

/** Continuous distribution presets for PMFPDFExplorer. */
export const continuousDistributionPresets: DistributionPreset[] = [
  { name: 'Uniform', params: { a: 0, b: 1 }, paramRanges: { a: [-5, 5], b: [-5, 10] } },
  { name: 'Normal', params: { mu: 0, sigma2: 1 }, paramRanges: { mu: [-5, 5], sigma2: [0.1, 10] } },
  { name: 'Exponential', params: { lambda: 1 }, paramRanges: { lambda: [0.1, 5] } },
];

// ── Transformation Presets ──────────────────────────────────────────────────

export interface TransformationPreset {
  name: string;
  source: string;
  formula: string;
  gFunc: (x: number, a?: number, b?: number) => number;
  hasParams?: boolean;
  paramDefaults?: { a: number; b: number };
}

/** Transformation presets for TransformationExplorer. */
export const transformationPresets: TransformationPreset[] = [
  {
    name: 'Y = X²',
    source: 'Normal(0,1)',
    formula: 'x² → χ²(1)',
    gFunc: (x: number) => x * x,
  },
  {
    name: 'Y = eˣ',
    source: 'Normal(0,1)',
    formula: 'eˣ → Lognormal',
    gFunc: (x: number) => Math.exp(x),
  },
  {
    name: 'Y = |X|',
    source: 'Normal(0,1)',
    formula: '|X| → Half-Normal',
    gFunc: (x: number) => Math.abs(x),
  },
  {
    name: 'Y = aX + b',
    source: 'Normal(0,1)',
    formula: 'Linear → Normal(b, a²)',
    gFunc: (x: number, a: number = 2, b: number = 1) => a * x + b,
    hasParams: true,
    paramDefaults: { a: 2, b: 1 },
  },
];

// ── Bivariate Normal Presets ────────────────────────────────────────────────

export interface BivariateNormalPreset {
  name: string;
  rho: number;
}

/** Bivariate normal presets for JointDistributionExplorer. */
export const bivariateNormalPresets: BivariateNormalPreset[] = [
  { name: 'Independent (ρ = 0)', rho: 0 },
  { name: 'Moderate positive (ρ = 0.5)', rho: 0.5 },
  { name: 'Strong positive (ρ = 0.85)', rho: 0.85 },
  { name: 'Negative (ρ = −0.6)', rho: -0.6 },
];
