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
