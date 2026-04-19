import * as d3 from 'd3';

/**
 * Domain color scale — maps track domain strings to distinct colors.
 * Used for curriculum graph and topic cards.
 */
export const domainColorScale = d3
  .scaleOrdinal<string>()
  .domain([
    'foundations-of-probability',
    'core-distributions',
    'convergence-limit-theorems',
    'statistical-estimation',
    'hypothesis-testing-confidence',
    'regression-linear-models',
    'bayesian-statistics',
    'high-dimensional-nonparametric',
  ])
  .range([
    '#2563EB', // blue — probability foundations
    '#059669', // emerald — distributions
    '#7C3AED', // violet — convergence
    '#D97706', // amber — estimation
    '#DC2626', // red — testing
    '#0891B2', // cyan — regression
    '#4F46E5', // indigo — Bayesian
    '#BE185D', // pink — nonparametric
  ]);

/**
 * Distribution visualization colors.
 */
export const distributionColors = {
  pdf: '#2563EB',
  cdf: '#059669',
  area: '#BFDBFE',
  sample: '#DC2626',
} as const;

/**
 * Inference visualization colors.
 */
export const inferenceColors = {
  null: '#6B7280',
  alternative: '#DC2626',
  confidence: '#BBF7D0',
  rejection: '#FECACA',
} as const;

/**
 * Topic 20 — Multiple Testing & False Discovery palette.
 *
 * Four FWER procedures in a purple gradient (deepest = most conservative):
 *   bonf (deepest) → holm → sidak → hoch (lightest).
 *
 * Three FDR procedures in a green gradient (deepest = most conservative):
 *   by (deepest, arbitrary dependence) → bh (independence) → storey (adaptive).
 *
 * `bh` intentionally shares hex #10B981 with the Topic 18 LRT / Topic 19
 * test-CI-duality green — BH is the FDR inheritor of the same visual family.
 *
 * `nullMixture` / `altMixture` are used in the §20.4 mixture decomposition
 * figures (slate for Uniform nulls, amber for concentrated alternatives).
 */
export const multipleTestingColors = {
  bonf: '#6B21A8',
  holm: '#9333EA',
  sidak: '#A855F7',
  hoch: '#C084FC',
  bh: '#10B981',
  by: '#047857',
  storey: '#34D399',
  nullMixture: '#94A3B8',
  altMixture: '#F59E0B',
} as const;

export type MultipleTestingColorKey = keyof typeof multipleTestingColors;
