/**
 * large-deviations-data.ts — Data module for Topic 12.
 *
 * Distribution presets for concentration-bound comparison, sub-Gaussian
 * classification, Cramér rate functions, and the tail-bound cheat sheet
 * used by the five Large Deviations interactive components.
 */

// ── Types ──────────────────────────────────────────────────────────────────

export type TailClassification =
  | 'sub-gaussian'
  | 'sub-exponential'
  | 'heavy-tailed';

export interface ConcentrationDistribution {
  id: string;
  name: string;
  mu: number;
  sigma2: number;
  /** [a, b] range for bounded variables; undefined if unbounded. */
  range: [number, number] | undefined;
  /** a.s. bound on |Xᵢ − μ| for Bernstein; undefined if unbounded. */
  M: number | undefined;
  /** Sub-Gaussian parameter σ_sg for bounded/Gaussian-like variables. */
  subGaussianParam: number | undefined;
  classification: TailClassification;
  description: string;
}

export interface SubGaussianPreset {
  id: string;
  name: string;
  classification: TailClassification;
  subGaussianParam?: number;
  subExponentialParams?: { nu2: number; b: number };
  description: string;
}

export interface RateFunctionPreset {
  id: string;
  name: string;
  /** Plain-text formula for the Cramér rate function I(x). */
  rateFunction: string;
  /** Plain-text formula for the log-MGF Λ(t) = log Mₓ(t). */
  logMgf: string;
  /** True if I(x) has a closed form; false means numerical only. */
  closedForm: boolean;
  /** Default parameter values — exact field names depend on the family. */
  defaultP?: number;
  defaultMu?: number;
  defaultSigma2?: number;
  defaultLambda?: number;
  description: string;
}

export interface TailBoundTheoremEntry {
  id: string;
  name: string;
  /** Numbering as it appears in the site's prose ("4.11", "12.3", etc.). */
  number: string;
  source: string;
  assumption: string;
  bound: string;
  rate: string;
  information: string;
  /** Tailwind-compatible hex colour for the cheat-sheet plot. */
  color: string;
}

// ── Concentration bound presets ─────────────────────────────────────────────

export const concentrationDistributions: ConcentrationDistribution[] = [
  {
    id: 'bernoulli-05',
    name: 'Bernoulli(0.5)',
    mu: 0.5,
    sigma2: 0.25,
    range: [0, 1],
    M: 0.5,
    subGaussianParam: 0.5,
    classification: 'sub-gaussian',
    description:
      'Symmetric coin flip. Bounded, so sub-Gaussian with σ_sg = (b−a)/2 = 0.5. Hoeffding and Bernstein both apply.',
  },
  {
    id: 'bernoulli-01',
    name: 'Bernoulli(0.1)',
    mu: 0.1,
    sigma2: 0.09,
    range: [0, 1],
    M: 0.9,
    subGaussianParam: 0.5,
    classification: 'sub-gaussian',
    description:
      'Rare event. Low variance (σ² = 0.09) makes Bernstein much tighter than Hoeffding.',
  },
  {
    id: 'uniform',
    name: 'Uniform(0, 1)',
    mu: 0.5,
    sigma2: 1 / 12,
    range: [0, 1],
    M: 0.5,
    subGaussianParam: 0.5,
    classification: 'sub-gaussian',
    description:
      'Bounded and symmetric. σ² = 1/12 ≪ (b−a)²/4 = 1/4, so Bernstein improves on Hoeffding.',
  },
  {
    id: 'normal-truncated',
    name: 'Truncated N(0, 1) on [−3, 3]',
    mu: 0,
    sigma2: 0.9973,
    range: [-3, 3],
    M: 3,
    subGaussianParam: 1,
    classification: 'sub-gaussian',
    description:
      'Bounded approximation of a Gaussian. Range [−3, 3] covers 99.7% of N(0,1).',
  },
  {
    id: 'exponential',
    name: 'Exponential(1)',
    mu: 1,
    sigma2: 1,
    range: undefined,
    M: undefined,
    subGaussianParam: undefined,
    classification: 'sub-exponential',
    description:
      'Unbounded, sub-exponential but NOT sub-Gaussian. Hoeffding does not apply directly; Bernstein applies with a truncation argument.',
  },
];

// ── Sub-Gaussian classifier presets ─────────────────────────────────────────

export const subGaussianPresets: SubGaussianPreset[] = [
  {
    id: 'normal',
    name: 'Normal(0, 1)',
    classification: 'sub-gaussian',
    subGaussianParam: 1,
    description: 'The prototype sub-Gaussian. σ_sg = σ = 1.',
  },
  {
    id: 'uniform-symmetric',
    name: 'Uniform(−1, 1)',
    classification: 'sub-gaussian',
    subGaussianParam: 1,
    description:
      'Bounded variables are sub-Gaussian with σ_sg = (b−a)/2 = 1.',
  },
  {
    id: 'rademacher',
    name: 'Rademacher (±1)',
    classification: 'sub-gaussian',
    subGaussianParam: 1,
    description:
      '±1 with equal probability. Sub-Gaussian with σ_sg = 1. The building block of Rademacher complexity.',
  },
  {
    id: 'exponential',
    name: 'Exponential(1) − 1',
    classification: 'sub-exponential',
    subExponentialParams: { nu2: 4, b: 1 },
    description:
      'Centered Exponential. Sub-exponential but NOT sub-Gaussian: the right tail decays like e^{−t}, not e^{−t²}.',
  },
  {
    id: 'chi-squared',
    name: 'χ²(1) − 1',
    classification: 'sub-exponential',
    subExponentialParams: { nu2: 4, b: 4 },
    description:
      'Centered chi-squared. Sub-exponential (product of two sub-Gaussians). NOT sub-Gaussian — its right tail is too heavy.',
  },
  {
    id: 'poisson',
    name: 'Poisson(5) − 5',
    classification: 'sub-exponential',
    subExponentialParams: { nu2: 10, b: 1 },
    description:
      'Centered Poisson. Sub-exponential via the Poisson MGF. NOT sub-Gaussian for any σ.',
  },
  {
    id: 't3',
    name: 't(3)',
    classification: 'heavy-tailed',
    description:
      'Student-t with 3 df. Finite variance but polynomial (not exponential) tails. NOT sub-exponential.',
  },
  {
    id: 't5',
    name: 't(5)',
    classification: 'heavy-tailed',
    description:
      'Student-t with 5 df. Heavier tails than Gaussian despite symmetry. NOT sub-exponential.',
  },
  {
    id: 'cauchy',
    name: 'Cauchy(0, 1)',
    classification: 'heavy-tailed',
    description:
      'No finite mean. The canonical counterexample: no concentration inequality applies. Even the LLN fails.',
  },
  {
    id: 'pareto',
    name: 'Pareto(2, 1) − 2',
    classification: 'heavy-tailed',
    description:
      'Power-law tails: P(X > t) ~ t^{−α}. Finite mean but infinite variance for α ≤ 2. The enemy of concentration.',
  },
];

// ── Rate function presets ───────────────────────────────────────────────────

export const rateFunctionPresets: RateFunctionPreset[] = [
  {
    id: 'bernoulli',
    name: 'Bernoulli(p)',
    defaultP: 0.3,
    rateFunction: 'D_KL(Ber(x) || Ber(p)) = x·ln(x/p) + (1−x)·ln((1−x)/(1−p))',
    logMgf: 'ln(1 − p + p·eᵗ)',
    closedForm: true,
    description:
      'The rate function is the KL divergence between Ber(x) and Ber(p). This connects large deviations to information theory.',
  },
  {
    id: 'normal',
    name: 'Normal(μ, σ²)',
    defaultMu: 0,
    defaultSigma2: 1,
    rateFunction: '(x − μ)² / (2σ²)',
    logMgf: 'μt + σ²t²/2',
    closedForm: true,
    description:
      'The rate function is a parabola — quadratic decay in the tails. The Normal is the unique distribution whose rate function is exactly quadratic.',
  },
  {
    id: 'exponential',
    name: 'Exponential(λ)',
    defaultLambda: 1,
    rateFunction: 'λx − 1 − ln(λx) for x > 0',
    logMgf: '−ln(1 − t/λ) for t < λ',
    closedForm: true,
    description:
      'The rate function grows logarithmically in x. The MGF has a pole at t = λ, which limits the domain of the Legendre transform.',
  },
  {
    id: 'poisson',
    name: 'Poisson(λ)',
    defaultLambda: 5,
    rateFunction: 'x·ln(x/λ) − x + λ for x ≥ 0',
    logMgf: 'λ(eᵗ − 1)',
    closedForm: true,
    description:
      'The Poisson rate function. Minimum at x = λ. Asymmetric: right tail decays faster than left tail.',
  },
];

// ── Theorem progression for the summary cheat sheet ─────────────────────────

export const tailBoundTheorems: TailBoundTheoremEntry[] = [
  {
    id: 'markov',
    name: 'Markov',
    number: '4.11',
    source: 'Topic 4',
    assumption: 'X ≥ 0, finite mean',
    bound: 'P(X ≥ a) ≤ E[X] / a',
    rate: 'O(1/ε)',
    information: 'Mean only',
    color: '#6b7280',
  },
  {
    id: 'chebyshev',
    name: 'Chebyshev',
    number: '4.12',
    source: 'Topic 4',
    assumption: 'Finite variance',
    bound: 'P(|X − μ| ≥ ε) ≤ σ² / ε²',
    rate: 'O(1/(nε²))',
    information: 'Mean + variance',
    color: '#d97706',
  },
  {
    id: 'hoeffding',
    name: 'Hoeffding',
    number: '12.3',
    source: 'Topic 12',
    assumption: 'Independent, bounded [aᵢ, bᵢ]',
    bound: 'P(|X̄ₙ − μ| ≥ ε) ≤ 2·exp(−2nε²/Σ(bᵢ−aᵢ)²)',
    rate: 'O(exp(−cnε²))',
    information: 'Mean + range',
    color: '#059669',
  },
  {
    id: 'bernstein',
    name: 'Bernstein',
    number: '12.4',
    source: 'Topic 12',
    assumption: 'Independent, bounded, known variance',
    bound: 'P(|X̄ₙ − μ| ≥ ε) ≤ 2·exp(−nε²/(2σ² + 2Mε/3))',
    rate: 'O(exp(−cnε²)) small ε; O(exp(−cnε)) large ε',
    information: 'Mean + variance + range',
    color: '#2563eb',
  },
  {
    id: 'sub-gaussian',
    name: 'Sub-Gaussian',
    number: '12.5',
    source: 'Topic 12',
    assumption: 'Independent sub-Gaussian(σ)',
    bound: 'P(|X̄ₙ − μ| ≥ ε) ≤ 2·exp(−nε²/(2σ²))',
    rate: 'O(exp(−cnε²))',
    information: 'Sub-Gaussian parameter',
    color: '#7c3aed',
  },
  {
    id: 'chernoff',
    name: 'Chernoff (generic)',
    number: '12.1',
    source: 'Topic 12',
    assumption: 'Finite MGF in a neighborhood of 0',
    bound: 'P(X ≥ a) ≤ inf_t exp(−ta)·Mₓ(t)',
    rate: 'O(exp(−nI(a)))',
    information: 'Full MGF',
    color: '#0891b2',
  },
  {
    id: 'mcdiarmid',
    name: 'McDiarmid',
    number: '12.8',
    source: 'Topic 12',
    assumption: 'f satisfies bounded differences cᵢ',
    bound: 'P(|f − E[f]| ≥ t) ≤ 2·exp(−2t²/Σcᵢ²)',
    rate: 'O(exp(−ct²))',
    information: 'Bounded-differences constants',
    color: '#dc2626',
  },
];
