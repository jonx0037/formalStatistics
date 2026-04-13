/**
 * modes-of-convergence-data.ts — Static data for Topic 9 interactive components.
 *
 * Provides mode definitions, hierarchy arrows, sequence presets,
 * CDF convergence examples, and delta method transformation presets.
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface ConvergenceMode {
  id: string;
  name: string;
  shortName: string;
  symbol: string;
  intuition: string;
  mlUseCase: string;
  strength: number; // 1 = strongest, 3 = weakest
}

export interface HierarchyArrow {
  from: string;
  to: string;
  type: 'implication' | 'non-implication' | 'partial';
  proof?: string;
  counterexample?: string;
  theoremNum?: number;
  caveat?: string;
}

export interface SamplePathPreset {
  name: string;
  id: string;
  description: string;
  target: number;
  convergenceMode: string;
  defaultEpsilon: number;
}

export interface CDFConvergenceExample {
  name: string;
  id: string;
  description: string;
  defaultLambda?: number;
  underlyingDistributions?: string[];
}

export interface DeltaMethodTransformation {
  name: string;
  g: (x: number) => number;
  gPrime: (x: number) => number;
  label: string;
}

export interface DeltaMethodDistribution {
  name: string;
  mu: number;
  sigmaSquared: number;
}

// ── Convergence Modes ────────────────────────────────────────────────────────

export const convergenceModes: ConvergenceMode[] = [
  {
    id: 'as',
    name: 'Almost Sure',
    shortName: 'a.s.',
    symbol: 'P(\\lim X_n = X) = 1',
    intuition: 'Every single sample path eventually settles down to the limit.',
    mlUseCase: 'SGD convergence (Robbins\u2013Monro), value iteration in RL, SLLN',
    strength: 1,
  },
  {
    id: 'lp',
    name: 'L^p (Mean)',
    shortName: 'L^p',
    symbol: 'E[|X_n - X|^p] \\to 0',
    intuition: 'The average p-th power deviation vanishes. Controls moment convergence.',
    mlUseCase: 'Mean-squared error convergence, L^p approximation bounds, L^2 risk convergence',
    strength: 1,
  },
  {
    id: 'prob',
    name: 'In Probability',
    shortName: 'in prob',
    symbol: '\\forall\\varepsilon > 0:\\; P(|X_n - X| > \\varepsilon) \\to 0',
    intuition: 'Deviations become rare \u2014 but some paths may still wander occasionally.',
    mlUseCase: 'PAC learning bounds, WLLN, consistency of estimators',
    strength: 2,
  },
  {
    id: 'dist',
    name: 'In Distribution',
    shortName: 'in dist',
    symbol: 'F_n(x) \\to F(x) \\text{ at continuity points}',
    intuition: 'The CDFs converge \u2014 the shape of the distribution stabilizes.',
    mlUseCase: 'CLT, asymptotic normality of MLE, Bernstein\u2013von Mises theorem',
    strength: 3,
  },
];

// ── Hierarchy Arrows ─────────────────────────────────────────────────────────

export const hierarchyArrows: HierarchyArrow[] = [
  {
    from: 'as',
    to: 'prob',
    type: 'implication',
    proof: 'Bounded convergence / measure argument: if P(lim X_n = X) = 1, then for any \u03b5, P(|X_n - X| > \u03b5) \u2192 0.',
    theoremNum: 5,
  },
  {
    from: 'lp',
    to: 'prob',
    type: 'implication',
    proof: "Markov's inequality applied to |X_n - X|^p: P(|X_n - X| > \u03b5) \u2264 E[|X_n - X|^p] / \u03b5^p \u2192 0.",
    theoremNum: 6,
  },
  {
    from: 'prob',
    to: 'dist',
    type: 'implication',
    proof: 'CDF squeeze: if P(|X_n - X| > \u03b5) \u2192 0, then F_{X_n}(x) \u2192 F_X(x) at continuity points of F_X.',
    theoremNum: 7,
  },
  {
    from: 'prob',
    to: 'as',
    type: 'partial',
    proof: 'First Borel\u2013Cantelli: from convergence in probability, extract a subsequence n_k with P(|X_{n_k} - X| > 1/k) < 2^{-k}. Then \u03a3 P(...) < \u221e, so X_{n_k} \u2192 X a.s.',
    theoremNum: 8,
    caveat: 'only along a subsequence',
  },
  {
    from: 'as',
    to: 'lp',
    type: 'non-implication',
    counterexample: 'X_n = n \u00b7 1(U < 1/n): converges a.s. to 0, but E[X_n] = 1 always (escape to infinity).',
  },
  {
    from: 'lp',
    to: 'as',
    type: 'non-implication',
    counterexample: 'Typewriter sequence (L\u00b2 version): E[X_n\u00b2] = 1/k \u2192 0 (L\u00b2 convergence), but every path has X_n = 1 infinitely often.',
  },
  {
    from: 'prob',
    to: 'as',
    type: 'non-implication',
    counterexample: 'Typewriter sequence: P(X_n \u2260 0) = 1/k \u2192 0, but every path has X_n = 1 infinitely often.',
  },
  {
    from: 'dist',
    to: 'prob',
    type: 'non-implication',
    counterexample: 'X_n = \u2212X where X ~ N(0,1): X_n has the same distribution as X (so X_n \u2192_d X), but P(|X_n - X| > \u03b5) = P(|2X| > \u03b5) \u2260 0.',
  },
];

// ── Sample Path Presets ──────────────────────────────────────────────────────

export const samplePathPresets: SamplePathPreset[] = [
  {
    name: 'Sample mean (a.s. convergence)',
    id: 'sample-mean',
    description: 'X\u0304_n of iid N(0,1) \u2014 converges a.s. and in L\u00b2 to 0',
    target: 0,
    convergenceMode: 'as',
    defaultEpsilon: 0.5,
  },
  {
    name: 'Z/n (a.s. & in probability)',
    id: 'z-over-n',
    description: 'X_n = Z/n where Z ~ N(0,1) \u2014 every path converges',
    target: 0,
    convergenceMode: 'as',
    defaultEpsilon: 0.1,
  },
  {
    name: 'Typewriter (in prob, NOT a.s.)',
    id: 'typewriter',
    description: 'Sliding indicators cycling through [0,1] \u2014 P(X_n \u2260 0) \u2192 0, but no path converges',
    target: 0,
    convergenceMode: 'prob',
    defaultEpsilon: 0.5,
  },
  {
    name: 'Escape to infinity (a.s., NOT L\u00b9)',
    id: 'escape',
    description: 'X_n = n \u00b7 1(U < 1/n) \u2014 converges to 0 a.s., but E[X_n] = 1 always',
    target: 0,
    convergenceMode: 'as',
    defaultEpsilon: 0.5,
  },
];

// ── CDF Convergence Examples ─────────────────────────────────────────────────

export const cdfConvergenceExamples: CDFConvergenceExample[] = [
  {
    name: 'Binomial \u2192 Poisson',
    id: 'poisson-limit',
    description: 'Bin(n, \u03bb/n) \u2192 Poisson(\u03bb)',
    defaultLambda: 5,
  },
  {
    name: 'Student-t \u2192 Normal',
    id: 'student-normal',
    description: 't(n) \u2192 N(0,1) as n \u2192 \u221e',
  },
  {
    name: 'CLT preview',
    id: 'clt-preview',
    description: '(X\u0304_n \u2212 \u03bc)/(\u03c3/\u221an) \u2192 N(0,1)',
    underlyingDistributions: ['Exponential(1)', 'Uniform(0,1)', 'Bernoulli(0.3)'],
  },
];

// ── Delta Method Presets ─────────────────────────────────────────────────────

export const deltaMethodTransformations: DeltaMethodTransformation[] = [
  { name: 'log(x)', g: Math.log, gPrime: (x: number) => 1 / x, label: 'ln(x)' },
  { name: '\u221ax', g: Math.sqrt, gPrime: (x: number) => 1 / (2 * Math.sqrt(x)), label: '\u221ax' },
  { name: '1/x', g: (x: number) => 1 / x, gPrime: (x: number) => -1 / (x * x), label: '1/x' },
  { name: 'x\u00b2', g: (x: number) => x * x, gPrime: (x: number) => 2 * x, label: 'x\u00b2' },
  { name: 'e^x', g: Math.exp, gPrime: Math.exp, label: 'e^x' },
];

export const deltaMethodDistributions: DeltaMethodDistribution[] = [
  { name: 'Normal(5, 4)', mu: 5, sigmaSquared: 4 },
  { name: 'Exponential(1)', mu: 1, sigmaSquared: 1 },
  { name: 'Poisson(5)', mu: 5, sigmaSquared: 5 },
  { name: 'Uniform(0, 10)', mu: 5, sigmaSquared: 100 / 12 },
];
