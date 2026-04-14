/**
 * law-of-large-numbers-data.ts — Data module for Topic 10.
 *
 * Distribution presets, theorem progression, and convergence rate
 * presets used by the five LLN interactive components.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface LLNDistribution {
  id: string;
  name: string;
  mu: number | undefined;
  sigmaSquared: number;
  hasMean: boolean;
  hasVariance: boolean;
  /** "slln" = both WLLN and SLLN apply; "wlln" = only WLLN; "none" = neither */
  llnApplies: 'slln' | 'wlln' | 'none';
  description: string;
}

export interface LLNTheorem {
  id: string;
  name: string;
  number: number;
  assumption: string;
  conclusion: string;
  proofTool: string;
  /** "probability" | "almost-sure" | "tool" (for maximal inequality) */
  mode: 'probability' | 'almost-sure' | 'tool';
}

export interface RatePreset {
  name: string;
  formula: string;
  type: 'polynomial' | 'exponential' | 'envelope';
  color: string;
}

// ── Distribution Presets ───────────────────────────────────────────────────

export const llnDistributions: LLNDistribution[] = [
  {
    id: 'normal',
    name: 'Normal(3, 4)',
    mu: 3,
    sigmaSquared: 4,
    hasMean: true,
    hasVariance: true,
    llnApplies: 'slln',
    description:
      'The baseline: both WLLN and SLLN apply with fast 1/\u221An convergence.',
  },
  {
    id: 'exponential',
    name: 'Exponential(1)',
    mu: 1,
    sigmaSquared: 1,
    hasMean: true,
    hasVariance: true,
    llnApplies: 'slln',
    description:
      'Skewed but well-behaved. SLLN applies; asymmetry visible at small n.',
  },
  {
    id: 't-light',
    name: 't(3)',
    mu: 0,
    sigmaSquared: 3, // ν/(ν−2) = 3
    hasMean: true,
    hasVariance: true,
    llnApplies: 'slln',
    description:
      'Heavier tails than Normal but still has finite variance. SLLN applies.',
  },
  {
    id: 't-heavy',
    name: 't(1.5)',
    mu: 0,
    sigmaSquared: Infinity,
    hasMean: true,
    hasVariance: false,
    llnApplies: 'slln',
    description:
      'Finite mean but infinite variance. SLLN still applies (Khintchine), but convergence is very slow.',
  },
  {
    id: 'cauchy',
    name: 'Cauchy (t(1))',
    mu: undefined,
    sigmaSquared: Infinity,
    hasMean: false,
    hasVariance: false,
    llnApplies: 'none',
    description:
      'No finite mean. Neither WLLN nor SLLN applies. X\u0304\u2099 ~ Cauchy for all n.',
  },
  {
    id: 'pareto-finite',
    name: 'Pareto(\u03B1=2)',
    mu: 2,
    sigmaSquared: Infinity,
    hasMean: true,
    hasVariance: false,
    llnApplies: 'slln',
    description:
      'Finite mean (\u03B1 > 1) but infinite variance (\u03B1 \u2264 2). SLLN applies via Khintchine.',
  },
  {
    id: 'pareto-no-mean',
    name: 'Pareto(\u03B1=0.8)',
    mu: undefined,
    sigmaSquared: Infinity,
    hasMean: false,
    hasVariance: false,
    llnApplies: 'none',
    description: '\u03B1 < 1: no finite mean. LLN fails completely.',
  },
  {
    id: 'bernoulli',
    name: 'Bernoulli(0.3)',
    mu: 0.3,
    sigmaSquared: 0.21,
    hasMean: true,
    hasVariance: true,
    llnApplies: 'slln',
    description:
      'Bounded, so Hoeffding gives exponential concentration. The fastest convergence.',
  },
];

// ── Theorem Progression ────────────────────────────────────────────────────

export const llnTheorems: LLNTheorem[] = [
  {
    id: 'wlln-chebyshev',
    name: 'WLLN (Chebyshev)',
    number: 1,
    assumption: 'iid, \u03C3\u00B2 < \u221E',
    conclusion: 'X\u0304\u2099 \u2192\u1D3E \u03BC',
    proofTool: 'Chebyshev',
    mode: 'probability',
  },
  {
    id: 'wlln-uncorrelated',
    name: 'WLLN (Uncorrelated)',
    number: 2,
    assumption: 'uncorrelated, \u03A3\u03C3\u1D62\u00B2/n\u00B2 \u2192 0',
    conclusion: 'X\u0304\u2099 \u2212 \u03BC\u0304\u2099 \u2192\u1D3E 0',
    proofTool: 'Variance of sums',
    mode: 'probability',
  },
  {
    id: 'wlln-khintchine',
    name: 'WLLN (Khintchine)',
    number: 3,
    assumption: 'iid, E[|X|] < \u221E',
    conclusion: 'X\u0304\u2099 \u2192\u1D3E \u03BC',
    proofTool: 'Truncation + DCT',
    mode: 'probability',
  },
  {
    id: 'kolmogorov-maximal',
    name: 'Kolmogorov Maximal',
    number: 4,
    assumption: 'independent, E[X\u1D62] = 0',
    conclusion: 'P(max|S\u2096| > \u03B5) \u2264 Var(S\u2099)/\u03B5\u00B2',
    proofTool: 'Stopping time decomposition',
    mode: 'tool',
  },
  {
    id: 'slln',
    name: 'SLLN (Etemadi)',
    number: 5,
    assumption: 'pairwise iid, E[|X|] < \u221E',
    conclusion: 'X\u0304\u2099 \u2192\u1D43\u02E2 \u03BC',
    proofTool: 'Truncation + maximal + subsequence',
    mode: 'almost-sure',
  },
  {
    id: 'glivenko-cantelli',
    name: 'Glivenko\u2013Cantelli',
    number: 6,
    assumption: 'iid',
    conclusion: 'sup|F\u2099 \u2212 F| \u2192\u1D43\u02E2 0',
    proofTool: 'SLLN for indicators',
    mode: 'almost-sure',
  },
];

// ── Rate Comparison Presets ────────────────────────────────────────────────

export const ratePresets: RatePreset[] = [
  {
    name: 'Chebyshev',
    formula: '\u03C3\u00B2/(n\u03B5\u00B2)',
    type: 'polynomial',
    color: '#059669', // emerald
  },
  {
    name: 'Hoeffding',
    formula: '2exp(\u22122n\u03B5\u00B2/(b\u2212a)\u00B2)',
    type: 'exponential',
    color: '#d97706', // amber
  },
  {
    name: 'LIL envelope',
    formula: '\u03C3\u221A(2 ln ln n / n)',
    type: 'envelope',
    color: '#2563eb', // blue
  },
];
