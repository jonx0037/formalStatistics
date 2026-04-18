/**
 * confidence-intervals-data.ts — Topic 19 preset data.
 *
 * Powers the four interactive components: CITestDualityVisualizer,
 * BinomialCoverageComparison, ProfileLikelihoodExplorer, and the optional
 * ReparamCIInvariance. Presets mirror notebook reference values where
 * applicable so visual exploration and numerical ground truth line up.
 */

export interface DualityPreset {
  name: string;
  family: 'normal-mean' | 'bernoulli' | 'poisson';
  n: number;
  alpha: number;
  trueParam: number;
  observedStatistic: number;
  sigmaKnown?: number;
}

export const dualityPresets: DualityPreset[] = [
  {
    name: 'Normal mean (σ = 1)',
    family: 'normal-mean',
    n: 30,
    alpha: 0.05,
    trueParam: 0,
    observedStatistic: 0.25,
    sigmaKnown: 1,
  },
  {
    name: 'Bernoulli proportion',
    family: 'bernoulli',
    n: 50,
    alpha: 0.05,
    trueParam: 0.4,
    observedStatistic: 0.36,
  },
  {
    name: 'Poisson rate',
    family: 'poisson',
    n: 40,
    alpha: 0.05,
    trueParam: 2.0,
    observedStatistic: 1.9,
  },
];

export interface CoveragePreset {
  name: string;
  n: number;
  alpha: number;
}

export const coveragePresets: CoveragePreset[] = [
  { name: 'n = 20 (small)', n: 20, alpha: 0.05 },
  { name: 'n = 100 (moderate)', n: 100, alpha: 0.05 },
  { name: 'n = 500 (large)', n: 500, alpha: 0.05 },
];

export interface ProfilePreset {
  name: string;
  scenario: 'normal-2d' | 'gamma-2d';
  n: number;
  observedThetaHat: number;
  observedNuisance: number;
  alpha: number;
  thetaRange: [number, number];
  psiRange: [number, number];
}

export const profilePresets: ProfilePreset[] = [
  {
    name: 'Normal mean, σ unknown',
    scenario: 'normal-2d',
    n: 20,
    observedThetaHat: 0.5,
    observedNuisance: 1.2,
    alpha: 0.05,
    thetaRange: [-1, 2],
    psiRange: [0.1, 3],
  },
  {
    name: 'Gamma shape, rate unknown',
    scenario: 'gamma-2d',
    n: 30,
    observedThetaHat: 2.5,
    observedNuisance: 1.8,
    alpha: 0.05,
    thetaRange: [0.5, 6],
    psiRange: [0.1, 5],
  },
];

export interface ReparamCIPreset {
  name: string;
  pHat: number;
  n: number;
  alpha: number;
  reparam: 'logit' | 'log' | 'probit';
}

export const reparamCIPresets: ReparamCIPreset[] = [
  {
    name: 'Bernoulli p → logit η',
    pHat: 0.3,
    n: 50,
    alpha: 0.05,
    reparam: 'logit',
  },
  {
    name: 'Bernoulli p → log p (rare event)',
    pHat: 0.05,
    n: 100,
    alpha: 0.05,
    reparam: 'log',
  },
  {
    name: 'Bernoulli p → probit',
    pHat: 0.35,
    n: 40,
    alpha: 0.05,
    reparam: 'probit',
  },
];
