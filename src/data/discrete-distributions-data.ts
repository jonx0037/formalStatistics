/**
 * discrete-distributions-data.ts — Presets and configuration for Topic 5
 * interactive components.
 *
 * Each preset array corresponds to one of the five interactive components:
 *   1. DistributionCatalogExplorer — unified 7-distribution interface
 *   2. PoissonLimitExplorer — Binomial→Poisson convergence
 *   3. SamplingComparisonExplorer — Binomial vs Hypergeometric
 *   4. PGFExplorer — PGF curves and compound distributions
 *   5. MemorylessPropertyExplorer — no presets needed (all client-side)
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface DistributionPreset {
  name: string;
  distribution: string;
  params: Record<string, number>;
  supportRange: [number, number];
  color: string;
}

export interface ParamDef {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface DistributionConfig {
  name: string;
  key: string;
  params: ParamDef[];
  supportRange: (params: Record<string, number>) => [number, number];
  color: string;
  mgfFormula: string;
  pgfFormula: string;
  expFamily: boolean;
}

// ── Distribution Configurations ────────────────────────────────────────────

export const distributionConfigs: DistributionConfig[] = [
  {
    name: 'Bernoulli',
    key: 'Bernoulli',
    params: [
      { name: 'p', label: 'p', min: 0.01, max: 0.99, step: 0.01, default: 0.6 },
    ],
    supportRange: () => [0, 1],
    color: 'var(--color-primary)',
    mgfFormula: 'M(t) = (1-p) + pe^t',
    pgfFormula: 'G(s) = (1-p) + ps',
    expFamily: true,
  },
  {
    name: 'Binomial',
    key: 'Binomial',
    params: [
      { name: 'n', label: 'n', min: 1, max: 50, step: 1, default: 15 },
      { name: 'p', label: 'p', min: 0.01, max: 0.99, step: 0.01, default: 0.4 },
    ],
    supportRange: (p) => [0, p.n],
    color: '#8b5cf6',
    mgfFormula: 'M(t) = ((1-p) + pe^t)^n',
    pgfFormula: 'G(s) = ((1-p) + ps)^n',
    expFamily: true,
  },
  {
    name: 'Geometric',
    key: 'Geometric',
    params: [
      { name: 'p', label: 'p', min: 0.05, max: 0.9, step: 0.01, default: 0.3 },
    ],
    supportRange: (p) => [1, Math.min(Math.ceil(5 / p.p), 30)],
    color: '#f59e0b',
    mgfFormula: 'M(t) = pe^t / (1 - (1-p)e^t)',
    pgfFormula: 'G(s) = ps / (1 - (1-p)s)',
    expFamily: true,
  },
  {
    name: 'Negative Binomial',
    key: 'NegBin',
    params: [
      { name: 'r', label: 'r', min: 1, max: 20, step: 1, default: 3 },
      { name: 'p', label: 'p', min: 0.05, max: 0.9, step: 0.01, default: 0.4 },
    ],
    supportRange: (p) => [p.r, Math.min(p.r + Math.ceil(4 * Math.sqrt(p.r * (1 - p.p) / (p.p * p.p))), 50)],
    color: '#ec4899',
    mgfFormula: 'M(t) = (pe^t / (1 - (1-p)e^t))^r',
    pgfFormula: 'G(s) = (ps / (1 - (1-p)s))^r',
    expFamily: true,
  },
  {
    name: 'Poisson',
    key: 'Poisson',
    params: [
      { name: 'lambda', label: 'λ', min: 0.1, max: 25, step: 0.1, default: 5 },
    ],
    supportRange: (p) => [0, Math.min(Math.ceil(p.lambda + 4 * Math.sqrt(p.lambda)), 40)],
    color: '#ef4444',
    mgfFormula: 'M(t) = e^{λ(e^t - 1)}',
    pgfFormula: 'G(s) = e^{λ(s - 1)}',
    expFamily: true,
  },
  {
    name: 'Hypergeometric',
    key: 'Hypergeometric',
    params: [
      { name: 'N', label: 'N', min: 10, max: 500, step: 1, default: 50 },
      { name: 'K', label: 'K', min: 1, max: 499, step: 1, default: 20 },
      { name: 'n', label: 'n', min: 1, max: 50, step: 1, default: 10 },
    ],
    supportRange: (p) => [Math.max(0, p.n - (p.N - p.K)), Math.min(p.n, p.K)],
    color: '#0ea5e9',
    mgfFormula: '(no closed form)',
    pgfFormula: '(hypergeometric series)',
    expFamily: false,
  },
  {
    name: 'Discrete Uniform',
    key: 'DiscreteUniform',
    params: [
      { name: 'a', label: 'a', min: 0, max: 20, step: 1, default: 1 },
      { name: 'b', label: 'b', min: 1, max: 30, step: 1, default: 6 },
    ],
    supportRange: (p) => [p.a, p.b],
    color: '#6b7280',
    mgfFormula: 'M(t) = e^{ta}(1 - e^{nt}) / (n(1 - e^t))',
    pgfFormula: 'G(s) = s^a(1 - s^n) / (n(1 - s))',
    expFamily: false,
  },
];

// ── Catalog Presets ────────────────────────────────────────────────────────

export const catalogPresets: DistributionPreset[] = [
  { name: 'Bernoulli(p=0.6)', distribution: 'Bernoulli', params: { p: 0.6 }, supportRange: [0, 1], color: 'var(--color-primary)' },
  { name: 'Binomial(n=15, p=0.4)', distribution: 'Binomial', params: { n: 15, p: 0.4 }, supportRange: [0, 15], color: '#8b5cf6' },
  { name: 'Geometric(p=0.3)', distribution: 'Geometric', params: { p: 0.3 }, supportRange: [1, 15], color: '#f59e0b' },
  { name: 'NegBin(r=3, p=0.4)', distribution: 'NegBin', params: { r: 3, p: 0.4 }, supportRange: [3, 20], color: '#ec4899' },
  { name: 'Poisson(λ=5)', distribution: 'Poisson', params: { lambda: 5 }, supportRange: [0, 15], color: '#ef4444' },
  { name: 'Hypergeometric(N=50, K=20, n=10)', distribution: 'Hypergeometric', params: { N: 50, K: 20, n: 10 }, supportRange: [0, 10], color: '#0ea5e9' },
  { name: 'DiscreteUniform(1, 6)', distribution: 'DiscreteUniform', params: { a: 1, b: 6 }, supportRange: [1, 6], color: '#6b7280' },
];

// ── Poisson Limit Presets ──────────────────────────────────────────────────

export const poissonLimitPresets = [
  { name: 'λ = 2 (rare events)', lambda: 2 },
  { name: 'λ = 5 (moderate)', lambda: 5 },
  { name: 'λ = 10 (frequent)', lambda: 10 },
  { name: 'λ = 20 (high rate)', lambda: 20 },
];

// ── Sampling Comparison Presets ────────────────────────────────────────────

export const samplingPresets = [
  { name: 'Small batch (N=50)', N: 50, K: 20, n: 10 },
  { name: 'Quality control (N=200)', N: 200, K: 40, n: 30 },
  { name: 'Survey (N=1000)', N: 1000, K: 400, n: 50 },
  { name: 'Census-scale (N=10000)', N: 10000, K: 4000, n: 100 },
];

// ── PGF Presets ────────────────────────────────────────────────────────────

export const pgfPresets = [
  { name: 'Bernoulli(p=0.4)', distribution: 'Bernoulli', params: { p: 0.4 } },
  { name: 'Binomial(n=5, p=0.4)', distribution: 'Binomial', params: { n: 5, p: 0.4 } },
  { name: 'Poisson(λ=3)', distribution: 'Poisson', params: { lambda: 3 } },
  { name: 'Geometric(p=0.4)', distribution: 'Geometric', params: { p: 0.4 } },
];

export const compoundPresets = [
  {
    name: 'Poisson-thinning',
    N: { distribution: 'Poisson', params: { lambda: 3 } },
    X: { distribution: 'Bernoulli', params: { p: 0.5 } },
    result: 'Poisson(λp = 1.5)',
  },
  {
    name: 'Poisson-Geometric compound',
    N: { distribution: 'Poisson', params: { lambda: 2 } },
    X: { distribution: 'Geometric', params: { p: 0.5 } },
    result: 'Compute via PGF',
  },
];
