/**
 * continuous-distributions-data.ts — Preset data for Topic 6 interactive components.
 *
 * Provides distribution configurations, catalog presets, and specialized
 * preset arrays for the five continuous distribution visualizations.
 */

// ── Types ───────────────────────────────────────────────────────────────────

export interface ContinuousParamDef {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface ContinuousDistributionConfig {
  name: string;
  key: string;
  params: ContinuousParamDef[];
  supportRange: (params: Record<string, number>) => [number, number];
  color: string;
  mgfFormula: string;
  expFamily: boolean;
}

// ── Distribution Configurations ─────────────────────────────────────────────

export const continuousDistributionConfigs: ContinuousDistributionConfig[] = [
  {
    name: 'Uniform',
    key: 'Uniform',
    params: [
      { name: 'a', label: 'a (lower)', min: -5, max: 4, step: 0.5, default: 0 },
      { name: 'b', label: 'b (upper)', min: -4, max: 5, step: 0.5, default: 1 },
    ],
    supportRange: (p) => [p.a - 0.5, p.b + 0.5],
    color: '#6366f1',
    mgfFormula: '\\frac{e^{tb} - e^{ta}}{t(b-a)}',
    expFamily: false,
  },
  {
    name: 'Normal',
    key: 'Normal',
    params: [
      { name: 'mu', label: 'μ (mean)', min: -5, max: 5, step: 0.1, default: 0 },
      { name: 'sigma2', label: 'σ² (variance)', min: 0.1, max: 10, step: 0.1, default: 1 },
    ],
    supportRange: (p) => {
      const sigma = Math.sqrt(p.sigma2);
      return [p.mu - 4 * sigma, p.mu + 4 * sigma];
    },
    color: '#2563eb',
    mgfFormula: 'e^{\\mu t + \\sigma^2 t^2/2}',
    expFamily: true,
  },
  {
    name: 'Exponential',
    key: 'Exponential',
    params: [
      { name: 'lambda', label: 'λ (rate)', min: 0.1, max: 5, step: 0.1, default: 1 },
    ],
    supportRange: (p) => [0, Math.max(5 / p.lambda, 3)],
    color: '#16a34a',
    mgfFormula: '\\frac{\\lambda}{\\lambda - t}',
    expFamily: true,
  },
  {
    name: 'Gamma',
    key: 'Gamma',
    params: [
      { name: 'alpha', label: 'α (shape)', min: 0.1, max: 20, step: 0.1, default: 3 },
      { name: 'beta', label: 'β (rate)', min: 0.1, max: 10, step: 0.1, default: 1 },
    ],
    supportRange: (p) => [0, Math.max((p.alpha + 4 * Math.sqrt(p.alpha)) / p.beta, 3)],
    color: '#d97706',
    mgfFormula: '\\left(\\frac{\\beta}{\\beta - t}\\right)^\\alpha',
    expFamily: true,
  },
  {
    name: 'Beta',
    key: 'Beta',
    params: [
      { name: 'a', label: 'α', min: 0.1, max: 20, step: 0.1, default: 2 },
      { name: 'b', label: 'β', min: 0.1, max: 20, step: 0.1, default: 5 },
    ],
    supportRange: () => [-0.05, 1.05],
    color: '#dc2626',
    mgfFormula: '\\text{(no closed form)}',
    expFamily: true,
  },
  {
    name: 'Chi-squared',
    key: 'Chi2',
    params: [
      { name: 'k', label: 'k (degrees of freedom)', min: 1, max: 30, step: 1, default: 5 },
    ],
    supportRange: (p) => [0, Math.max(p.k + 4 * Math.sqrt(2 * p.k), 10)],
    color: '#9333ea',
    mgfFormula: '(1 - 2t)^{-k/2}',
    expFamily: true,
  },
  {
    name: "Student's t",
    key: 'StudentT',
    params: [
      { name: 'nu', label: 'ν (degrees of freedom)', min: 1, max: 30, step: 1, default: 5 },
    ],
    supportRange: () => [-6, 6],
    color: '#0891b2',
    mgfFormula: '\\text{(does not exist)}',
    expFamily: false,
  },
  {
    name: 'F',
    key: 'F',
    params: [
      { name: 'd1', label: 'd₁', min: 1, max: 30, step: 1, default: 5 },
      { name: 'd2', label: 'd₂', min: 1, max: 30, step: 1, default: 20 },
    ],
    supportRange: (p) => {
      const mean = p.d2 > 2 ? p.d2 / (p.d2 - 2) : 2;
      return [0, Math.max(mean * 3, 5)];
    },
    color: '#be185d',
    mgfFormula: '\\text{(no closed form)}',
    expFamily: false,
  },
];

// ── Catalog Presets ──────────────────────────────────────────────────────────

export const continuousDistributionPresets = [
  { name: 'Uniform(0, 1)', distribution: 'Uniform', params: { a: 0, b: 1 } },
  { name: 'Normal(0, 1)', distribution: 'Normal', params: { mu: 0, sigma2: 1 } },
  { name: 'Normal(3, 4)', distribution: 'Normal', params: { mu: 3, sigma2: 4 } },
  { name: 'Exponential(1)', distribution: 'Exponential', params: { lambda: 1 } },
  { name: 'Exponential(0.5)', distribution: 'Exponential', params: { lambda: 0.5 } },
  { name: 'Gamma(3, 1)', distribution: 'Gamma', params: { alpha: 3, beta: 1 } },
  { name: 'Gamma(0.5, 1)', distribution: 'Gamma', params: { alpha: 0.5, beta: 1 } },
  { name: 'Beta(2, 5)', distribution: 'Beta', params: { a: 2, b: 5 } },
  { name: 'Beta(0.5, 0.5)', distribution: 'Beta', params: { a: 0.5, b: 0.5 } },
  { name: 'Chi-squared(5)', distribution: 'Chi2', params: { k: 5 } },
  { name: "Student's t(5)", distribution: 'StudentT', params: { nu: 5 } },
  { name: 'F(5, 20)', distribution: 'F', params: { d1: 5, d2: 20 } },
];

// ── Beta Shape Presets ──────────────────────────────────────────────────────

export const betaShapePresets = [
  { name: 'Uniform: α=β=1', params: { a: 1, b: 1 } },
  { name: 'Symmetric bell: α=β=5', params: { a: 5, b: 5 } },
  { name: 'Skewed: α=2, β=5', params: { a: 2, b: 5 } },
  { name: 'U-shaped: α=β=0.5', params: { a: 0.5, b: 0.5 } },
  { name: 'J-shaped: α=0.5, β=2', params: { a: 0.5, b: 2 } },
];

// ── Gamma Special Cases ─────────────────────────────────────────────────────

export const gammaSpecialCases = [
  { name: 'Exponential(1) = Gamma(1, 1)', alpha: 1, beta: 1 },
  { name: 'χ²(4) = Gamma(2, 0.5)', alpha: 2, beta: 0.5 },
  { name: 'χ²(10) = Gamma(5, 0.5)', alpha: 5, beta: 0.5 },
  { name: 'Erlang(3, 2) = Gamma(3, 2)', alpha: 3, beta: 2 },
];

// ── Conjugate Prior Presets ─────────────────────────────────────────────────

export const conjugatePriorPresets = [
  { name: 'Uniform prior', alpha0: 1, beta0: 1, description: 'No prior information' },
  { name: 'Weak prior (α=β=2)', alpha0: 2, beta0: 2, description: 'Mild preference for θ=0.5' },
  { name: 'Informative prior (α=10, β=10)', alpha0: 10, beta0: 10, description: 'Strong belief θ≈0.5' },
  { name: 'Asymmetric prior (α=2, β=8)', alpha0: 2, beta0: 8, description: 'Believe θ is small' },
  { name: 'Jeffreys prior (α=β=0.5)', alpha0: 0.5, beta0: 0.5, description: 'Non-informative reference prior' },
];
