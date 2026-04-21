/**
 * Preset data for Topic 25: Bayesian Foundations & Prior Selection.
 *
 * Used by PriorPosteriorExplorer (§25.2 + §25.5 anchor component),
 * ConjugatePairBrowser (§25.5 5-tab browser), BernsteinVonMisesAnimator
 * (§25.8 featured component), and PriorSensitivityComparator (§25.7
 * optional fourth).
 *
 * Preset descriptions are intentionally short (under 500 chars each) —
 * they appear as tooltip / subtitle text in the components. Expository
 * context lives in the MDX body, not here.
 */

// ── PriorPosteriorExplorer presets ─────────────────────────────────────────

export const priorPosteriorPresets = [
  {
    name: 'Uniform Beta(1, 1)',
    alpha0: 1, beta0: 1,
    description: 'No prior information — all probabilities equally likely.',
  },
  {
    name: 'Jeffreys Beta(½, ½)',
    alpha0: 0.5, beta0: 0.5,
    description: 'Non-informative reference prior, reparameterization-invariant.',
  },
  {
    name: 'Weakly informative Beta(2, 2)',
    alpha0: 2, beta0: 2,
    description: 'Mild preference for θ = ½; pseudo-sample-size 4.',
  },
  {
    name: 'Informative heads-favoring Beta(10, 3)',
    alpha0: 10, beta0: 3,
    description: 'Strong prior belief θ ≈ 0.77; pseudo-sample-size 13.',
  },
  {
    name: 'Informative tails-favoring Beta(3, 10)',
    alpha0: 3, beta0: 10,
    description: 'Strong prior belief θ ≈ 0.23; pseudo-sample-size 13.',
  },
] as const;

export type PriorPosteriorPreset = typeof priorPosteriorPresets[number];

// ── ConjugatePairBrowser presets (per conjugate family) ────────────────────

export const betaBinomialPresets = [
  {
    name: 'A/B test conversion (weak prior)',
    alpha0: 2, beta0: 2, n: 50, k: 10,
    mNew: 50,
    description:
      'Weak prior + observed 10/50 → Beta(12, 42). Posterior mean 0.222, 95% CrI [0.123, 0.341].',
  },
  {
    name: 'Quality-control rejection rate (Jeffreys)',
    alpha0: 0.5, beta0: 0.5, n: 20, k: 2,
    mNew: 20,
    description:
      'Jeffreys prior + 2 rejects in 20 → Beta(2.5, 18.5). Posterior mean 0.119.',
  },
  {
    name: 'Vaccine efficacy (strong success prior)',
    alpha0: 15, beta0: 5, n: 100, k: 85,
    mNew: 100,
    description:
      'Strong prior on high efficacy + data → Beta(100, 20). Posterior mean 0.833, tight CrI.',
  },
] as const;

export type BetaBinomialPreset = typeof betaBinomialPresets[number];

export const normalNormalPresets = [
  {
    name: 'Weight estimation (weak prior)',
    mu0: 70, sigma0_sq: 100, sigma_sq: 4, n: 10, yBar: 72.5,
    description:
      'Prior N(70, 100) + 10 obs with mean 72.5, σ²=4 → posterior mean ≈ 72.49.',
  },
  {
    name: 'IQ measurement (strong prior)',
    mu0: 100, sigma0_sq: 225, sigma_sq: 100, n: 5, yBar: 115,
    description:
      'Prior N(100, 225) + 5 obs with mean 115, σ²=100 → posterior mean 110.71.',
  },
  {
    name: 'Sensor calibration (noninformative)',
    mu0: 0, sigma0_sq: 10000, sigma_sq: 1, n: 30, yBar: 0.85,
    description:
      'Nearly-flat prior → posterior mean ≈ MLE = 0.85.',
  },
] as const;

export type NormalNormalPreset = typeof normalNormalPresets[number];

export const normalNormalIGPresets = [
  {
    name: 'Student-t arising naturally',
    mu0: 0, kappa0: 1, alpha0: 2, beta0: 2, n: 20, yBar: 0.5, s2: 1.2,
    description:
      'Unknown σ²: marginal posterior on μ is Student-t, not Normal.',
  },
] as const;

export type NormalNormalIGPreset = typeof normalNormalIGPresets[number];

export const gammaPoissonPresets = [
  {
    name: 'Call-center arrival rate',
    alpha0: 2, beta0: 1, n: 10, S: 45,
    mNew: 10,
    description:
      'Prior mean 2 calls/interval + 45 calls in 10 intervals → posterior mean 47/11 ≈ 4.27.',
  },
  {
    name: 'Defect rate (rare-event)',
    alpha0: 1, beta0: 10, n: 5, S: 2,
    mNew: 5,
    description:
      'Prior favoring low rate + 2 defects in 5 units → posterior mean 3/15 = 0.2.',
  },
] as const;

export type GammaPoissonPreset = typeof gammaPoissonPresets[number];

export const dirichletMultinomialPresets = [
  {
    name: 'Three-way topic proportions',
    alpha0: [1, 1, 1] as const, counts: [12, 10, 8] as const,
    description:
      'Uniform prior on the simplex + observed (12, 10, 8) → posterior Dir(13, 11, 9).',
  },
  {
    name: 'Strong prior on dominant category',
    alpha0: [10, 2, 2] as const, counts: [5, 5, 5] as const,
    description:
      'Prior concentrated at (0.71, 0.14, 0.14) pulls the posterior toward category 1 despite balanced data.',
  },
] as const;

export type DirichletMultinomialPreset = typeof dirichletMultinomialPresets[number];

// ── BernsteinVonMisesAnimator presets ──────────────────────────────────────

/**
 * BvM presets span three conjugate families. The `family` tag discriminates
 * the hyperparameter shape and the true-parameter field. See §25.8 Thm 5.
 */
export type BvMPreset =
  | {
      name: string;
      family: 'beta-binomial';
      prior: { alpha0: number; beta0: number };
      trueTheta: number;
      description: string;
    }
  | {
      name: string;
      family: 'gamma-poisson';
      prior: { alpha0: number; beta0: number };
      trueLambda: number;
      description: string;
    }
  | {
      name: string;
      family: 'normal-normal';
      prior: { mu0: number; sigma0_sq: number; sigma_sq: number };
      trueMu: number;
      description: string;
    };

export const bvmPresets: BvMPreset[] = [
  {
    name: 'Beta-Binomial with Jeffreys prior',
    family: 'beta-binomial',
    prior: { alpha0: 0.5, beta0: 0.5 },
    trueTheta: 0.3,
    description:
      'At n=500 the posterior is visually indistinguishable from the Normal-at-MLE.',
  },
  {
    name: 'Beta-Binomial with strong misspecified prior',
    family: 'beta-binomial',
    prior: { alpha0: 10, beta0: 40 },
    trueTheta: 0.7, // deliberate mismatch: prior mean 0.2 vs true θ₀ 0.7
    description:
      'Strong prior at wrong location → posterior takes longer to concentrate on MLE; BvM still holds.',
  },
  {
    name: 'Gamma-Poisson',
    family: 'gamma-poisson',
    prior: { alpha0: 2, beta0: 1 },
    trueLambda: 3,
    description:
      'Gamma prior + Poisson likelihood; BvM holds by n ≈ 100.',
  },
  {
    name: 'Normal-Normal known σ²',
    family: 'normal-normal',
    prior: { mu0: 0, sigma0_sq: 4, sigma_sq: 1 },
    trueMu: 2,
    description:
      'BvM is exact for every n when prior is Normal (no limit needed).',
  },
];

// ── PriorSensitivityComparator presets ─────────────────────────────────────

export const prior3WayPresets = [
  {
    name: 'Canonical Beta-Binomial contrast (10/50)',
    n: 50, k: 10,
    prior1: { alpha: 2, beta: 8, label: 'Informative (mean 0.2)' },
    prior2: { alpha: 2, beta: 2, label: 'Weakly informative' },
    prior3: { alpha: 0.5, beta: 0.5, label: 'Jeffreys (non-informative)' },
    description:
      'At moderate n, informative prior dominates; at n=1000 all three posteriors align.',
  },
  {
    name: 'Small-sample A/B test (3/10)',
    n: 10, k: 3,
    prior1: { alpha: 8, beta: 2, label: 'Informative heads-favoring' },
    prior2: { alpha: 1, beta: 1, label: 'Uniform' },
    prior3: { alpha: 0.5, beta: 0.5, label: 'Jeffreys' },
    description:
      'Small data → priors dominate; posterior means range across three different ballparks.',
  },
] as const;

export type Prior3WayPreset = typeof prior3WayPresets[number];
