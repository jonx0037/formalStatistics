/**
 * maximum-likelihood-data.ts — Data module for Topic 14.
 *
 * Distribution families, transformation specs, and scenario presets used by
 * the five Maximum-Likelihood interactive components:
 *   LikelihoodSurfaceExplorer, MLESamplingExplorer, NewtonRaphsonExplorer,
 *   MLEInvarianceExplorer, ObservedVsExpectedExplorer.
 */

// ── Types ────────────────────────────────────────────────────────────────────

export type MLEFamily = 'Normal' | 'Bernoulli' | 'Exponential' | 'Poisson' | 'Gamma';
export type MLEParamName = 'mu' | 'sigma2' | 'p' | 'lambda' | 'alpha';

export interface MLEDistributionPreset {
  name: string;
  family: MLEFamily;
  paramName: MLEParamName;
  trueParam: number;
  otherParams?: Record<string, number>;
  closedForm: boolean;
  mleFormula: string;
  fisherInfo: string;
  /** Plotting window for the log-likelihood curve: [θ_min, θ_max]. */
  thetaRange: [number, number];
  description: string;
}

export interface InvariancePreset {
  name: string;
  baseFamily: MLEFamily;
  baseParam: MLEParamName;
  trueParam: number;
  otherParams?: Record<string, number>;
  transformLabel: string;
  transform: (theta: number) => number;
  inverse: (phi: number) => number;
  /** Plotting windows on the two axes: [θ_min, θ_max] and [φ_min, φ_max]. */
  thetaRange: [number, number];
  phiRange: [number, number];
  description: string;
}

export interface NewtonRaphsonPreset {
  name: string;
  family: MLEFamily | 'Logistic';
  paramName: string;
  trueParam: number | [number, number];
  otherParams?: Record<string, number>;
  defaultN: number;
  defaultInit: number | [number, number];
  description: string;
}

export interface ObservedVsExpectedPreset {
  name: string;
  family: MLEFamily;
  paramName: MLEParamName;
  trueParam: number;
  otherParams?: Record<string, number>;
  /**
   * Descriptive metadata flag for whether the observed information J(θ̂)
   * and the expected information nI(θ̂) are identically equal at the MLE
   * for this family (a property of the natural parameterization of some
   * exponential families). This flag is retained for preset annotation
   * and documentation — the explorer derives visible behaviour directly
   * from the MC-computed values, not from this flag.
   */
  observedEqualsExpected: boolean;
  description: string;
}

// ── Distribution Presets (LikelihoodSurface, MLESampling, ObservedVsExpected) ─

export const mleDistributionPresets: MLEDistributionPreset[] = [
  {
    name: 'Normal(5, 4) — mean μ',
    family: 'Normal',
    paramName: 'mu',
    trueParam: 5,
    otherParams: { sigma2: 4 },
    closedForm: true,
    mleFormula: 'μ̂ = X̄',
    fisherInfo: 'I(μ) = 1/σ² = 0.25',
    thetaRange: [2, 8],
    description:
      'The simplest MLE — the sample mean. Closed-form, efficient, exact Normal sampling distribution.',
  },
  {
    name: 'Bernoulli(0.3) — proportion p',
    family: 'Bernoulli',
    paramName: 'p',
    trueParam: 0.3,
    closedForm: true,
    mleFormula: 'p̂ = X̄ = k/n',
    fisherInfo: 'I(p) = 1/(p(1−p)) ≈ 4.76',
    thetaRange: [0.02, 0.98],
    description:
      'MLE is the sample proportion. Discrete data — asymptotic normality kicks in slowly near p ≈ 0 or 1.',
  },
  {
    name: 'Exponential(1) — rate λ',
    family: 'Exponential',
    paramName: 'lambda',
    trueParam: 1,
    closedForm: true,
    mleFormula: 'λ̂ = 1/X̄',
    fisherInfo: 'I(λ) = 1/λ² = 1',
    thetaRange: [0.2, 3],
    description:
      'MLE is the reciprocal of the sample mean — biased for finite n (Jensen), but consistent and efficient.',
  },
  {
    name: 'Poisson(3) — rate λ',
    family: 'Poisson',
    paramName: 'lambda',
    trueParam: 3,
    closedForm: true,
    mleFormula: 'λ̂ = X̄',
    fisherInfo: 'I(λ) = 1/λ ≈ 0.333',
    thetaRange: [0.5, 7],
    description:
      'MLE is the sample mean. Discrete data — histograms at small n show visible discreteness.',
  },
  {
    name: 'Gamma(3, 2) — shape α',
    family: 'Gamma',
    paramName: 'alpha',
    trueParam: 3,
    otherParams: { beta: 2 },
    closedForm: false,
    mleFormula: 'Newton-Raphson (no closed form)',
    fisherInfo: "I(α) = ψ'(α) (trigamma)",
    thetaRange: [0.5, 7],
    description:
      'The canonical "no closed-form MLE" example. Newton-Raphson converges in 4–6 iterations from a method-of-moments start.',
  },
];

// ── Invariance Presets (MLEInvarianceExplorer) ───────────────────────────────

export const invariancePresets: InvariancePreset[] = [
  {
    name: 'σ² → σ  (standard deviation)',
    baseFamily: 'Normal',
    baseParam: 'sigma2',
    trueParam: 4,
    otherParams: { mu: 0 },
    transformLabel: '√(·)',
    transform: (s2) => Math.sqrt(Math.max(s2, 0)),
    inverse: (sigma) => sigma * sigma,
    thetaRange: [0.5, 10],
    phiRange: [Math.sqrt(0.5), Math.sqrt(10)],
    description:
      'If σ̂² is the MLE of σ², then √(σ̂²) is the MLE of σ — a direct application of invariance.',
  },
  {
    name: 'p → p/(1−p)  (odds)',
    baseFamily: 'Bernoulli',
    baseParam: 'p',
    trueParam: 0.3,
    transformLabel: '·/(1−·)',
    transform: (p) => p / Math.max(1 - p, 1e-9),
    inverse: (o) => o / (1 + o),
    thetaRange: [0.02, 0.98],
    phiRange: [0.02 / 0.98, 0.98 / 0.02],
    description:
      'If p̂ is the MLE of p, then p̂/(1−p̂) is the MLE of the odds — the natural parameter in logistic regression.',
  },
  {
    name: 'λ → 1/λ  (mean)',
    baseFamily: 'Exponential',
    baseParam: 'lambda',
    trueParam: 1,
    transformLabel: '1/(·)',
    transform: (lam) => 1 / Math.max(lam, 1e-9),
    inverse: (mu) => 1 / Math.max(mu, 1e-9),
    thetaRange: [0.2, 3],
    phiRange: [1 / 3, 1 / 0.2],
    description: 'If λ̂ is the MLE of the rate, then 1/λ̂ = X̄ is the MLE of the mean.',
  },
];

// ── Newton-Raphson Presets (NewtonRaphsonExplorer) ───────────────────────────

export const newtonRaphsonPresets: NewtonRaphsonPreset[] = [
  {
    name: 'Gamma(3, 2) shape — no closed form',
    family: 'Gamma',
    paramName: 'alpha',
    trueParam: 3,
    otherParams: { beta: 2 },
    defaultN: 50,
    defaultInit: 1.0,
    description:
      'The score involves the digamma function ψ(α). Newton-Raphson typically converges in 4–6 iterations.',
  },
  {
    name: 'Poisson(3) rate — verify one-step convergence',
    family: 'Poisson',
    paramName: 'lambda',
    trueParam: 3,
    defaultN: 50,
    defaultInit: 1.5,
    description:
      'The log-likelihood is concave with a closed-form MLE — Newton converges in one step from most starts.',
  },
  {
    name: 'Logistic regression (preview)',
    family: 'Logistic',
    paramName: 'beta',
    trueParam: [0.5, 1.5],
    defaultN: 100,
    defaultInit: [0, 0],
    description:
      'No closed-form MLE — Newton-Raphson here is IRLS. The full GLM theory is in Topic 21.',
  },
];

// ── Observed-vs-Expected Presets (ObservedVsExpectedExplorer) ────────────────

export const observedVsExpectedPresets: ObservedVsExpectedPreset[] = [
  {
    name: 'Normal(5, 4) — observed = expected',
    family: 'Normal',
    paramName: 'mu',
    trueParam: 5,
    otherParams: { sigma2: 4 },
    observedEqualsExpected: true,
    description:
      'For Normal mean, J(μ) = n/σ² is non-random — every sample gives the same observed information.',
  },
  {
    name: 'Exponential(1) — J(λ̂) = nI(λ̂) at MLE, varies with data',
    family: 'Exponential',
    paramName: 'lambda',
    trueParam: 1,
    observedEqualsExpected: false,
    description:
      'Both J(λ̂) and nI(λ̂) equal n/λ̂² at the MLE, so the scatter sits on the identity but each point varies sample-to-sample through λ̂.',
  },
  {
    name: 'Poisson(3)',
    family: 'Poisson',
    paramName: 'lambda',
    trueParam: 3,
    observedEqualsExpected: false,
    description:
      'J(λ̂) = nX̄/λ̂² and nI(λ̂) = n/λ̂ coincide at the MLE (since λ̂ = X̄) — the scatter lies on the identity and contracts toward (n/3, n/3) as n grows.',
  },
  {
    name: 'Bernoulli(0.3)',
    family: 'Bernoulli',
    paramName: 'p',
    trueParam: 0.3,
    observedEqualsExpected: false,
    description:
      'At the MLE p̂, both J(p̂) and nI(p̂) equal n/(p̂(1−p̂)). The scatter is on the identity; its spread shrinks with n as p̂ concentrates around p₀.',
  },
];

// ── Machine-learning Connection Presets (§9 content) ─────────────────────────

export interface MLConnectionPresets {
  crossEntropy: {
    families: Array<{ family: MLEFamily; note: string }>;
  };
  mapRegularization: {
    gaussianPrior: { mean: number; variance: number; label: string; penalty: string };
    laplacePrior: { location: number; scale: number; label: string; penalty: string };
  };
}

export const mlConnectionPresets: MLConnectionPresets = {
  crossEntropy: {
    families: [
      {
        family: 'Normal',
        note: 'Normal-likelihood MLE ≡ minimizing mean-squared error under fixed σ².',
      },
      {
        family: 'Bernoulli',
        note: 'Bernoulli MLE ≡ minimizing binary cross-entropy — the loss function of logistic regression.',
      },
      {
        family: 'Poisson',
        note: 'Poisson MLE ≡ minimizing the Poisson deviance — the loss function of count-regression GLMs.',
      },
    ],
  },
  mapRegularization: {
    gaussianPrior: {
      mean: 0,
      variance: 1,
      label: 'β ~ N(0, τ²)',
      penalty: 'L² penalty: −(1/(2τ²))‖β‖²',
    },
    laplacePrior: {
      location: 0,
      scale: 1,
      label: 'β ~ Laplace(0, b)',
      penalty: 'L¹ penalty: −(1/b)‖β‖₁',
    },
  },
};

// ── Multivariate Sidebar (§5 content) ────────────────────────────────────────

export interface MultivariateSidebar {
  fisherMatrix: string;
  asymptoticNormality: string;
  waldCI: string;
  note: string;
}

export const multivariateSidebar: MultivariateSidebar = {
  fisherMatrix:
    "I(θ) = E[s(θ)s(θ)ᵀ] = −E[∂²ℓ/∂θᵢ∂θⱼ]  (a d×d positive semi-definite matrix)",
  asymptoticNormality: '√n(θ̂ − θ₀) →ᵈ N(0, I(θ₀)⁻¹)',
  waldCI: 'θ̂ⱼ ± z_{α/2} · √([I(θ̂)⁻¹/n]ⱼⱼ)',
  note:
    'The full multivariate treatment — including the role of the Cramér-Rao matrix inequality and the block structure of information matrices — appears in Topics 20–21 (Linear Regression, Generalized Linear Models).',
};
