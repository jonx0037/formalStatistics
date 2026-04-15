/**
 * point-estimation-data.ts — Data module for Topic 13.
 *
 * Distribution families, estimator specs, and scenario presets used by the
 * five Point-Estimation interactive components (EstimatorSamplingExplorer,
 * BiasVarianceMSEExplorer, ConsistencyExplorer, FisherInformationExplorer,
 * CramerRaoExplorer).
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DistributionPreset {
  name: string;
  family: 'Normal' | 'Exponential' | 'Bernoulli' | 'Uniform' | 'Poisson';
  params: Record<string, number>;
  /**
   * Map of estimator-target names to their true value under the preset.
   * Populated by consumers to pick the right truth for the selected estimator.
   */
  trueParam: Record<string, number>;
  description: string;
}

export interface EstimatorPreset {
  name: string;
  id: 'mean' | 'median' | 'trimmed' | 'var_biased' | 'var_unbiased';
  isUnbiased: boolean;
  description: string;
}

export interface ShrinkagePreset {
  name: string;
  mu: number;
  sigma2: number;
  n: number;
  cRange: [number, number];
  description: string;
}

export interface ConsistencyPreset {
  name: string;
  estimator: 'mean' | 'variance' | 'first' | 'constant';
  consistent: boolean;
  description: string;
}

export interface FisherInfoPreset {
  name: string;
  family: 'Normal' | 'Bernoulli' | 'Exponential' | 'Poisson';
  paramName: 'mu' | 'p' | 'lambda';
  formula: string;
  description: string;
}

export interface AREPreset {
  distribution: 'Normal' | 'Exponential' | 'Cauchy';
  meanEfficiency: number;
  medianEfficiency: number;
  trimmedEfficiency: number;
  description: string;
}

export interface JamesSteinPreset {
  d: number;
  sigma2: number;
  thetaNormRange: [number, number];
  description: string;
}

// ── Distribution Presets (EstimatorSamplingExplorer) ─────────────────────────

export const distributionPresets: DistributionPreset[] = [
  {
    name: 'Normal(5, 4)',
    family: 'Normal',
    params: { mu: 5, sigma2: 4 },
    trueParam: { mean: 5, mu: 5, variance: 4, sigma2: 4 },
    description: 'Symmetric, finite variance — the CLT baseline.',
  },
  {
    name: 'Exponential(1)',
    family: 'Exponential',
    params: { lambda: 1 },
    trueParam: { mean: 1, lambda: 1, variance: 1 },
    description: 'Skewed — asymptotic normality kicks in slowly.',
  },
  {
    name: 'Bernoulli(0.3)',
    family: 'Bernoulli',
    params: { p: 0.3 },
    trueParam: { mean: 0.3, p: 0.3, variance: 0.21 },
    description: 'Discrete — estimating a proportion.',
  },
  {
    name: 'Uniform(0, 10)',
    family: 'Uniform',
    params: { a: 0, b: 10 },
    trueParam: { mean: 5, a: 0, b: 10, variance: 100 / 12 },
    description: 'Bounded — finite support, mean vs midrange comparison.',
  },
];

// ── Estimator Presets (EstimatorSamplingExplorer) ────────────────────────────

export const estimatorPresets: EstimatorPreset[] = [
  {
    name: 'Sample Mean',
    id: 'mean',
    isUnbiased: true,
    description: 'X̄ = (1/n)ΣXᵢ — unbiased, efficient for Normal',
  },
  {
    name: 'Sample Median',
    id: 'median',
    // Unbiased for the population median only under symmetry; biased for the
    // population mean on skewed families (e.g. Exponential). The EstimatorSamplingExplorer
    // targets the population median, so this flag reflects general-case behaviour.
    isUnbiased: false,
    description: 'Middle order statistic — robust to outliers, less efficient',
  },
  {
    name: 'Trimmed Mean (10%)',
    id: 'trimmed',
    // Unbiased for μ only under symmetry; biased for skewed populations where
    // the trimmed expectation differs from E[X].
    isUnbiased: false,
    description: 'Remove 10% from each tail — compromise between mean and median',
  },
  {
    name: 'Sample Variance (1/n)',
    id: 'var_biased',
    isUnbiased: false,
    description: 'Biased — E[S²] = (n−1)σ²/n',
  },
  {
    name: 'Sample Variance (1/(n-1))',
    id: 'var_unbiased',
    isUnbiased: true,
    description: "Unbiased — Bessel's correction",
  },
];

// ── Shrinkage Presets (BiasVarianceMSEExplorer) ──────────────────────────────

export const shrinkagePresets: ShrinkagePreset[] = [
  {
    name: 'Normal(5, 4), n=25',
    mu: 5,
    sigma2: 4,
    n: 25,
    cRange: [0, 1.5],
    description: 'Moderate SNR — optimal c* close to 1',
  },
  {
    name: 'Normal(1, 16), n=10',
    mu: 1,
    sigma2: 16,
    n: 10,
    cRange: [0, 1.5],
    description: 'Low SNR — optimal c* well below 1',
  },
  {
    name: 'Normal(10, 1), n=50',
    mu: 10,
    sigma2: 1,
    n: 50,
    cRange: [0, 1.5],
    description: 'High SNR — optimal c* very close to 1',
  },
];

// ── Consistency Presets (ConsistencyExplorer) ────────────────────────────────

export const consistencyPresets: ConsistencyPreset[] = [
  {
    name: 'Mean → μ (consistent)',
    estimator: 'mean',
    consistent: true,
    description: 'SLLN guarantees X̄ₙ → μ a.s.',
  },
  {
    name: 'Variance → σ² (consistent)',
    estimator: 'variance',
    consistent: true,
    description: 'LLN applied to (Xᵢ − X̄)²',
  },
  {
    name: 'First obs X₁ (inconsistent)',
    estimator: 'first',
    consistent: false,
    description: 'Ignores all data after first observation — Var stays σ²',
  },
  {
    name: 'Constant θ̂ = 7 (inconsistent)',
    estimator: 'constant',
    consistent: false,
    description: 'Does not depend on data at all',
  },
];

// ── Fisher-Information Presets (FisherInformationExplorer) ───────────────────

export const fisherInfoPresets: FisherInfoPreset[] = [
  {
    name: 'Normal(μ, σ²) w.r.t. μ',
    family: 'Normal',
    paramName: 'mu',
    formula: 'I(μ) = 1/σ²',
    description: 'Fisher info is constant in μ, depends on σ²',
  },
  {
    name: 'Bernoulli(p)',
    family: 'Bernoulli',
    paramName: 'p',
    formula: 'I(p) = 1/(p(1−p))',
    description: 'Maximum info at p = 0.5, diverges at p → 0 or 1',
  },
  {
    name: 'Exponential(λ)',
    family: 'Exponential',
    paramName: 'lambda',
    formula: 'I(λ) = 1/λ²',
    description: 'Info decreases as λ increases',
  },
  {
    name: 'Poisson(λ)',
    family: 'Poisson',
    paramName: 'lambda',
    formula: 'I(λ) = 1/λ',
    description: 'Info decreases as λ increases',
  },
];

// ── ARE Presets (CramerRaoExplorer) ──────────────────────────────────────────

export const arePresets: AREPreset[] = [
  {
    distribution: 'Normal',
    meanEfficiency: 1.0,
    medianEfficiency: 2 / Math.PI, // ≈ 0.637
    trimmedEfficiency: 0.95,
    description: 'Mean is efficient; median wastes π/2 ≈ 57% of the data',
  },
  {
    distribution: 'Exponential',
    meanEfficiency: 1.0,
    medianEfficiency: Math.log(2) ** 2 * 4, // ≈ 1.922 (worse)
    trimmedEfficiency: 1.1,
    description: 'Mean is efficient; median even worse due to asymmetry',
  },
  {
    distribution: 'Cauchy',
    meanEfficiency: 0, // mean has infinite variance
    medianEfficiency: 1.0,
    trimmedEfficiency: 0.8,
    description: 'No CRLB (mean has infinite variance); median is optimal',
  },
];

// ── James-Stein Presets (CramerRaoExplorer) ──────────────────────────────────

export const jamesSteinPresets: JamesSteinPreset[] = [
  { d: 3, sigma2: 1, thetaNormRange: [0, 8], description: 'Threshold case d = 3' },
  { d: 5, sigma2: 1, thetaNormRange: [0, 8], description: 'Moderate improvement d = 5' },
  { d: 10, sigma2: 1, thetaNormRange: [0, 8], description: 'Large improvement d = 10' },
  { d: 20, sigma2: 1, thetaNormRange: [0, 8], description: 'Strong shrinkage d = 20' },
];
