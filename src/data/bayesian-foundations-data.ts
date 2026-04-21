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

// ═══════════════════════════════════════════════════════════════════════════
// Topic 26 — Bayesian Computation & MCMC presets
//
// Four preset groups (one per interactive component). Target densities and
// gradients are function literals (not strings) — typed callbacks that the
// components invoke directly; no runtime eval / code-string compilation.
// ═══════════════════════════════════════════════════════════════════════════

// ── MetropolisHastingsTuner presets ─────────────────────────────────────────

export interface MHTunerPreset {
  readonly id: string;
  readonly label: string;
  /** log π(x) — 1-D accepts a scalar, 2-D accepts a 2-vector. */
  readonly logPi: (x: number | number[]) => number;
  readonly dimension: 1 | 2;
  /** Plot support; scalar for 1-D, pair for 2-D. */
  readonly support: [number, number] | [[number, number], [number, number]];
  readonly optimalScale: number;
  readonly optimalAcceptance: number;
  readonly description: string;
}

export const mhTunerPresets: readonly MHTunerPreset[] = [
  {
    id: 'standard-normal',
    label: 'Standard Normal 𝒩(0, 1)',
    logPi: (x) => {
      const v = x as number;
      return -0.5 * v * v;
    },
    dimension: 1,
    support: [-5, 5],
    optimalScale: 2.38,
    optimalAcceptance: 0.44,
    description:
      'Canonical 1-D target. Roberts-Gelman-Gilks optimal scale 2.38 gives ~44% acceptance.',
  },
  {
    id: 'banana',
    label: 'Banana (Rosenbrock, d=2)',
    logPi: (q) => {
      const v = q as number[];
      return -0.5 * ((1 - v[0]) ** 2 + 10 * (v[1] - v[0] ** 2) ** 2);
    },
    dimension: 2,
    support: [
      [-2, 2],
      [-1, 3],
    ],
    optimalScale: 0.5,
    optimalAcceptance: 0.234,
    description:
      'Curved ridge — RWM struggles. Illustrates why HMC / gradient-aware samplers win on correlated geometries.',
  },
  {
    id: 'bimodal-mixture',
    label: 'Bimodal 0.5·𝒩(−2, 0.5²) + 0.5·𝒩(2, 0.5²)',
    logPi: (x) => {
      const v = x as number;
      return Math.log(
        0.5 * Math.exp(-2 * (v + 2) ** 2) +
          0.5 * Math.exp(-2 * (v - 2) ** 2),
      );
    },
    dimension: 1,
    support: [-6, 6],
    optimalScale: 1.5,
    optimalAcceptance: 0.25,
    description:
      'Two equally-weighted modes separated by a low-density valley. Stuck-chain failure mode (§26.8 Ex 10).',
  },
  {
    id: 't-distribution-df3',
    label: 'Heavy-tail Student-t (ν = 3)',
    logPi: (x) => {
      const v = x as number;
      return -2 * Math.log(1 + (v * v) / 3);
    },
    dimension: 1,
    support: [-10, 10],
    optimalScale: 3.0,
    optimalAcceptance: 0.35,
    description:
      'Polynomial tails challenge geometric ergodicity (ROB2004 §7.3 — non-geometrically ergodic under Gaussian proposals).',
  },
];

// ── GibbsStepper presets (bivariate Normal) ─────────────────────────────────

export interface GibbsStepperPreset {
  readonly id: string;
  readonly label: string;
  readonly rho: number;
  readonly description: string;
}

export const gibbsStepperPresets: readonly GibbsStepperPreset[] = [
  {
    id: 'weak',
    label: 'ρ = 0.3 (weak correlation)',
    rho: 0.3,
    description:
      'Near-decorrelated; Gibbs mixes quickly — each coordinate update is nearly independent of the previous.',
  },
  {
    id: 'moderate',
    label: 'ρ = 0.6',
    rho: 0.6,
    description:
      'Moderate correlation; mixing slows but full-conditionals remain distinct enough for standard Gibbs.',
  },
  {
    id: 'strong',
    label: 'ρ = 0.8 (canonical demo)',
    rho: 0.8,
    description:
      'Brief §3.1 Ex 3 canonical setup — strong correlation; axis-aligned Gibbs moves show visible staircase pattern.',
  },
  {
    id: 'extreme',
    label: 'ρ = 0.95 (slow mixing)',
    rho: 0.95,
    description:
      'Near-degenerate ridge; Gibbs crawls along the diagonal. Reparameterization or block-update would be better.',
  },
];

// ── HamiltonianTrajectoryAnimator presets (FEATURED component) ──────────────

export interface HMCAnimatorPreset {
  readonly id: string;
  readonly label: string;
  readonly logPi: (q: number[]) => number;
  /** Gradient of U(q) = -log π(q). */
  readonly gradU: (q: number[]) => number[];
  readonly dimension: 2;
  readonly defaultStart: readonly [number, number];
  readonly recommendedEpsilon: number;
  readonly recommendedSteps: number;
  readonly description: string;
}

export const hmcAnimatorPresets: readonly HMCAnimatorPreset[] = [
  {
    id: 'standard-normal-2d',
    label: '2-D Standard Normal',
    logPi: (q) => -0.5 * (q[0] ** 2 + q[1] ** 2),
    gradU: (q) => [q[0], q[1]],
    dimension: 2,
    defaultStart: [1, 1],
    recommendedEpsilon: 0.1,
    recommendedSteps: 25,
    description:
      'Rotational trajectory on a radially symmetric target. Energy oscillates as O(ε²) per Störmer-Verlet.',
  },
  {
    id: 'banana',
    label: 'Banana (Rosenbrock)',
    logPi: (q) => -0.5 * ((1 - q[0]) ** 2 + 10 * (q[1] - q[0] ** 2) ** 2),
    gradU: (q) => [
      -(1 - q[0]) + 20 * (q[1] - q[0] ** 2) * (-2 * q[0]),
      20 * (q[1] - q[0] ** 2),
    ],
    dimension: 2,
    defaultStart: [-1, 1],
    recommendedEpsilon: 0.05,
    recommendedSteps: 25,
    description:
      'Curved ridge — HMC flows along curvature where RWM rejects. Featured-figure geometry (§26.4 Fig 4).',
  },
  {
    id: 'donut',
    label: 'Donut (ring-shaped target)',
    logPi: (q) => {
      const r = Math.sqrt(q[0] ** 2 + q[1] ** 2);
      return -10 * (r - 1) ** 2;
    },
    gradU: (q) => {
      const r = Math.max(Math.sqrt(q[0] ** 2 + q[1] ** 2), 1e-6);
      return [(20 * (r - 1) * q[0]) / r, (20 * (r - 1) * q[1]) / r];
    },
    dimension: 2,
    defaultStart: [1, 0],
    recommendedEpsilon: 0.05,
    recommendedSteps: 30,
    description:
      'Unimodal but non-convex support. HMC orbits the ring; RWM diffuses slowly across the hole.',
  },
  {
    id: 'correlated-normal',
    label: 'Correlated Normal (ρ = 0.9)',
    logPi: (q) =>
      -0.5 / (1 - 0.81) * (q[0] ** 2 - 1.8 * q[0] * q[1] + q[1] ** 2),
    gradU: (q) => {
      const f = 1 / (1 - 0.81);
      return [f * (q[0] - 0.9 * q[1]), f * (q[1] - 0.9 * q[0])];
    },
    dimension: 2,
    defaultStart: [2, -2],
    recommendedEpsilon: 0.1,
    recommendedSteps: 20,
    description:
      'Near-degenerate ridge; HMC with mass matrix M = Σ⁻¹ rescales to well-conditioned dynamics (§26.4 Rem 13).',
  },
];

// ── ConvergenceDiagnosticDashboard presets ──────────────────────────────────

export interface ConvergenceDashboardPreset {
  readonly id: string;
  readonly label: string;
  /** Matches an `id` in `mhTunerPresets` — dashboard pulls the target from there. */
  readonly targetId: string;
  readonly defaultChains: number;
  readonly defaultDispersion: number;
  readonly expectedRhat: number;
  readonly description: string;
}

export const convergenceDashboardPresets: readonly ConvergenceDashboardPreset[] = [
  {
    id: 'well-mixed-normal',
    label: 'Well-mixed chains on 𝒩(0, 1)',
    targetId: 'standard-normal',
    defaultChains: 4,
    defaultDispersion: 3,
    expectedRhat: 1.01,
    description:
      'Dispersed starts collapse onto the mode within ~200 iter. R̂ drops from ≈ 1.3 → < 1.01 — the happy path.',
  },
  {
    id: 'stuck-bimodal',
    label: 'Stuck chains on bimodal target',
    targetId: 'bimodal-mixture',
    defaultChains: 4,
    defaultDispersion: 4,
    expectedRhat: 1.5,
    description:
      'All chains stuck in the mode they started in — R̂ stays high, visually demonstrates §26.8 Ex 10.',
  },
  {
    id: 'slow-mixing-banana',
    label: 'Slow-mixing RWM on banana',
    targetId: 'banana',
    defaultChains: 4,
    defaultDispersion: 2,
    expectedRhat: 1.1,
    description:
      'Curvature traps RWM at short lags — R̂ improves slowly, motivating gradient-aware methods.',
  },
];
