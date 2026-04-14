/**
 * central-limit-theorem-data.ts — Data module for Topic 11.
 *
 * Distribution presets, Lindeberg-condition scenarios, Berry–Esseen
 * comparison presets, and theorem progression used by the five
 * CLT interactive components.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export interface CLTDistribution {
  id: string;
  name: string;
  mu: number;
  sigma: number;
  skewness: number;
  /** E[|X − μ|³] / σ³ — the absolute third-moment ratio that drives Berry–Esseen */
  thirdMomentRatio: number;
  description: string;
}

export interface LindebergDistributionEntry {
  name: string;
  variance: number;
}

export interface LindebergPreset {
  id: string;
  name: string;
  description: string;
  lindebergHolds: boolean;
  /**
   * Either a concrete list of summands (fixed-length preset) or the sentinel
   * "dynamic" meaning the summands are constructed from n at render time.
   */
  distributions: LindebergDistributionEntry[] | 'dynamic';
}

export interface BerryEsseenPreset {
  id: string;
  name: string;
  /** Absolute third-moment ratio ρ = E[|X − μ|³] / σ³ */
  rho: number;
  skewness: number;
  convergenceSpeed: 'fast' | 'slow' | 'very slow';
  color: string;
}

export interface CLTTheorem {
  id: string;
  name: string;
  number: number;
  assumption: string;
  conclusion: string;
  proofTool: string;
  /** Year of first publication, or "—" when no single attribution applies */
  year: number | string;
}

// ── Distribution Presets (CLTExplorer) ─────────────────────────────────────

export const cltDistributions: CLTDistribution[] = [
  {
    id: 'uniform',
    name: 'Uniform(0, 1)',
    mu: 0.5,
    sigma: Math.sqrt(1 / 12),
    skewness: 0,
    thirdMomentRatio: 1.8,
    description:
      'Symmetric, bounded. Fast convergence — the CLT works well even at n = 5.',
  },
  {
    id: 'exponential',
    name: 'Exponential(1)',
    mu: 1,
    sigma: 1,
    skewness: 2,
    thirdMomentRatio: 6,
    description:
      'Skewed right. Moderate convergence — visible non-normality at n = 10, good by n = 50.',
  },
  {
    id: 'bernoulli',
    name: 'Bernoulli(0.3)',
    mu: 0.3,
    sigma: Math.sqrt(0.21),
    skewness: 0.873, // (1 − 2p)/√(p(1−p)) with p = 0.3
    thirdMomentRatio: 2.34,
    description:
      'Discrete and asymmetric. The de Moivre–Laplace theorem is this CLT applied to sums of Bernoullis.',
  },
  {
    id: 'poisson',
    name: 'Poisson(5)',
    mu: 5,
    sigma: Math.sqrt(5),
    skewness: 1 / Math.sqrt(5),
    thirdMomentRatio: 1.45,
    description:
      'Discrete count data. Moderate skewness — converges reasonably fast.',
  },
  {
    id: 'beta',
    name: 'Beta(2, 5)',
    mu: 2 / 7,
    sigma: Math.sqrt(10 / (49 * 8)),
    skewness: 0.596,
    thirdMomentRatio: 1.9,
    description:
      'Bounded, right-skewed. Common in Bayesian posterior analysis.',
  },
  {
    id: 'chi-squared',
    name: 'Chi-squared(3)',
    mu: 3,
    sigma: Math.sqrt(6),
    skewness: Math.sqrt(8 / 3),
    thirdMomentRatio: 4.35,
    description:
      'Highly skewed right. Slow convergence — needs n ≈ 100 for good Normal approximation.',
  },
  {
    id: 't5',
    name: 't(5)',
    mu: 0,
    sigma: Math.sqrt(5 / 3),
    skewness: 0,
    thirdMomentRatio: 6.0,
    description:
      'Symmetric but heavy-tailed. Symmetric → faster convergence from the skewness term, but heavy tails inflate the Berry–Esseen constant.',
  },
  {
    id: 'gamma',
    name: 'Gamma(2, 1)',
    mu: 2,
    sigma: Math.sqrt(2),
    skewness: Math.sqrt(2),
    thirdMomentRatio: 3.77,
    description:
      'Right-skewed. Shape α controls skewness: larger α → more symmetric → faster CLT convergence.',
  },
  {
    id: 'dice',
    name: 'Fair Die {1,…,6}',
    mu: 3.5,
    sigma: Math.sqrt(35 / 12),
    skewness: 0,
    thirdMomentRatio: 1.64,
    description:
      'The most familiar random variable. Symmetric, so convergence is fast — rolling 30 dice gives a remarkably Normal sum.',
  },
];

// ── Lindeberg Presets (LindebergExplorer) ──────────────────────────────────

export const lindebergPresets: LindebergPreset[] = [
  {
    id: 'mixed-equal-var',
    name: 'Mixed Distributions (equal variance)',
    description:
      'X₁ ~ N(0,1), X₂ ~ Exp(1)−1, X₃ ~ Uniform(−√3, √3). All variance 1. Lindeberg holds.',
    lindebergHolds: true,
    distributions: [
      { name: 'Normal(0,1)', variance: 1 },
      { name: 'Exponential(1)−1', variance: 1 },
      { name: 'Uniform(−√3, √3)', variance: 1 },
    ],
  },
  {
    id: 'increasing-var',
    name: 'Increasing Variance',
    description:
      'Xᵢ ~ N(0, i). Variance grows linearly. Lindeberg holds: max σᵢ²/sₙ² = n / (n(n+1)/2) → 0.',
    lindebergHolds: true,
    distributions: 'dynamic',
  },
  {
    id: 'one-dominant',
    name: 'One Dominant Variable',
    description:
      'X₁ ~ N(0, n²), X₂,…,Xₙ ~ N(0, 1). Var(X₁)/sₙ² = n² / (n² + n − 1) → 1. Lindeberg fails.',
    lindebergHolds: false,
    distributions: 'dynamic',
  },
  {
    id: 'bernoulli-varying-p',
    name: 'Bernoulli with Varying p',
    description:
      'Xᵢ ~ Bernoulli(pᵢ) with pᵢ = 1/(i+1). Different success probabilities. Lindeberg holds as long as no single pᵢ(1−pᵢ) dominates.',
    lindebergHolds: true,
    distributions: 'dynamic',
  },
];

// ── Berry–Esseen Comparison Presets ────────────────────────────────────────

export const berryEsseenPresets: BerryEsseenPreset[] = [
  {
    id: 'uniform',
    name: 'Uniform(0, 1)',
    rho: 1.8,
    skewness: 0,
    convergenceSpeed: 'fast',
    color: '#059669', // emerald
  },
  {
    id: 'normal',
    name: 'Normal(0, 1)',
    rho: 2.0,
    skewness: 0,
    convergenceSpeed: 'fast',
    color: '#2563eb', // blue
  },
  {
    id: 'bernoulli-half',
    name: 'Bernoulli(0.5)',
    rho: 1.0,
    skewness: 0,
    convergenceSpeed: 'fast',
    color: '#7c3aed', // violet
  },
  {
    id: 'exponential',
    name: 'Exponential(1)',
    rho: 6.0,
    skewness: 2,
    convergenceSpeed: 'slow',
    color: '#d97706', // amber
  },
  {
    id: 'chi-squared-1',
    name: 'Chi-squared(1)',
    rho: 8.0,
    skewness: 2 * Math.sqrt(2),
    convergenceSpeed: 'very slow',
    color: '#dc2626', // red
  },
  {
    id: 'bernoulli-01',
    name: 'Bernoulli(0.1)',
    rho: 3.16,
    skewness: 2.67,
    convergenceSpeed: 'slow',
    color: '#ea580c', // orange
  },
];

// ── Theorem Progression (Summary Diagram) ──────────────────────────────────

export const cltTheorems: CLTTheorem[] = [
  {
    id: 'de-moivre-laplace',
    name: 'De Moivre–Laplace',
    number: 1,
    assumption: 'Sₙ ~ Bin(n, p)',
    conclusion: '(Sₙ − np) / √(np(1−p)) →d N(0,1)',
    proofTool: 'Stirling’s approximation',
    year: 1733,
  },
  {
    id: 'clt-lindeberg-levy',
    name: 'CLT (Lindeberg–Lévy)',
    number: 2,
    assumption: 'iid, σ² < ∞',
    conclusion: '√n(X̄ₙ − μ)/σ →d N(0,1)',
    proofTool: 'MGF Taylor expansion + Lévy continuity',
    year: 1922,
  },
  {
    id: 'levy-cf',
    name: 'Lévy Continuity (CF version)',
    number: 3,
    assumption: 'φₙ(t) → φ(t) pointwise, φ continuous at 0',
    conclusion: 'Xₙ →d X',
    proofTool: 'Characteristic functions',
    year: 1925,
  },
  {
    id: 'lindeberg-clt',
    name: 'Lindeberg CLT',
    number: 4,
    assumption: 'independent, Lindeberg condition',
    conclusion: 'Sₙ/sₙ →d N(0,1)',
    proofTool: 'Lindeberg replacement + CF Taylor',
    year: 1922,
  },
  {
    id: 'lyapunov-clt',
    name: 'Lyapunov CLT',
    number: 5,
    assumption: 'independent, Σ E[|Xₖ|^{2+δ}] / sₙ^{2+δ} → 0',
    conclusion: 'Sₙ/sₙ →d N(0,1)',
    proofTool: 'Lyapunov condition ⟹ Lindeberg',
    year: 1901,
  },
  {
    id: 'berry-esseen',
    name: 'Berry–Esseen',
    number: 6,
    assumption: 'iid, E[|X|³] < ∞',
    conclusion: 'sup|Fₙ(x) − Φ(x)| ≤ C·ρ/√n',
    proofTool: 'Fourier analysis of CFs',
    year: 1941,
  },
  {
    id: 'multivariate-clt',
    name: 'Multivariate CLT',
    number: 7,
    assumption: 'iid vectors, Σ finite',
    conclusion: '√n(X̄ₙ − μ) →d N(0, Σ)',
    proofTool: 'Cramér–Wold + univariate CLT',
    year: '—',
  },
  {
    id: 'delta-method-clt',
    name: 'Delta Method (CLT version)',
    number: 8,
    assumption: '√n(X̄ₙ − μ) →d N(0, σ²), g differentiable',
    conclusion: '√n(g(X̄ₙ) − g(μ)) →d N(0, [g′(μ)]²σ²)',
    proofTool: 'Taylor + Slutsky',
    year: '—',
  },
];
