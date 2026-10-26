/**
 * Track 9 preset parameters for the time-series components and the
 * timeseries.test.ts harness.
 *
 * Topic 34 — Hidden Markov Models: values locked to notebook Cell 4 / Cell 11
 * (notebooks/hidden-markov-models/34_hidden_markov_models.ipynb); see that
 * handoff brief §7.
 *
 * Topic 35 — Stationary Time Series & ARMA: root-plane, periodogram, GARCH and
 * Wold presets, locked to notebook Cell 4
 * (notebooks/stationary-time-series-and-arma/35_stationary_time_series.ipynb);
 * see docs/formalstatistics-stationary-time-series-and-arma-handoff-brief.md §7.
 *
 * Topic 36 — State-Space Models & Kalman Smoothing: smoother-explorer,
 * ellipse-stepper, likelihood-surface and structural presets, locked to
 * notebook Cells 4–12
 * (notebooks/state-space-models-and-kalman-smoothing/36_state_space_models.ipynb);
 * see docs/formalstatistics-state-space-models-and-kalman-smoothing-handoff-brief.md §7.
 *
 * Typed presets, not `as const` (readonly arrays do not type-check against
 * HMMParams / ARMAParams — Topic 33 Appendix B.10).
 */

import {
  stationaryDistribution,
  type HMMParams,
  type ARMAParams,
  type Complex,
  type GarchParams,
} from '../components/viz/shared/timeseries';

const marketA = [
  [0.98, 0.02],
  [0.05, 0.95],
];
const market3A = [
  [0.97, 0.015, 0.015],
  [0.035, 0.93, 0.035],
  [0.075, 0.075, 0.85],
];

/** Dishonest casino (DUR1998 §3.1): state 0 = fair, 1 = loaded; symbols 0–5 = faces 1–6. */
export const casinoParams: HMMParams = {
  initial: [2 / 3, 1 / 3],
  transition: [
    [0.95, 0.05],
    [0.1, 0.9],
  ],
  emission: {
    kind: 'categorical',
    probs: [
      [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
      [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    ],
  },
};

/** Two-regime market (Ex 1): state 0 = calm, 1 = turbulent; returns in percent. */
export const marketParams: HMMParams = {
  initial: stationaryDistribution(marketA),
  transition: marketA,
  emission: { kind: 'gaussian', means: [0.05, -0.1], sds: [0.7, 2.0] },
};

/** Three-regime market (Ex 13–14): calm / normal / crisis. */
export const market3Params: HMMParams = {
  initial: stationaryDistribution(market3A),
  transition: market3A,
  emission: { kind: 'gaussian', means: [0.06, 0.0, -0.3], sds: [0.5, 1.0, 3.0] },
};

/** Ex 10: pointwise posterior decoding returns (3, 1, 1, 1, 1), which uses the forbidden edge 3 → 1. */
export const forbiddenTransitionExample: { params: HMMParams; obs: number[] } = {
  params: {
    initial: [1 / 3, 1 / 3, 1 / 3],
    transition: [
      [0.9, 0.1, 0],
      [0.5, 0.2, 0.3],
      [0, 0.2, 0.8],
    ],
    emission: {
      kind: 'categorical',
      probs: [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.9, 0.1],
      ],
    },
  },
  obs: [0, 1, 1, 1, 0],
};

export interface HMMPreset {
  id: string;
  label: string;
  params: HMMParams;
  T: number;
  seed: number;
  /** Display names for the hidden states, in state order. */
  stateNames: string[];
}

const stickyThreeA = [
  [0.96, 0.03, 0.01],
  [0.02, 0.95, 0.03],
  [0.01, 0.04, 0.95],
];

export const forwardBackwardPresets: HMMPreset[] = [
  { id: 'casino', label: 'Casino', params: casinoParams, T: 120, seed: 3405, stateNames: ['fair', 'loaded'] },
  { id: 'market', label: 'Market', params: marketParams, T: 200, seed: 3405, stateNames: ['calm', 'turbulent'] },
  {
    id: 'sticky3',
    label: 'Sticky three-state',
    params: {
      initial: stationaryDistribution(stickyThreeA),
      transition: stickyThreeA,
      emission: { kind: 'gaussian', means: [-1.5, 0, 1.5], sds: [0.8, 0.8, 0.8] },
    },
    T: 200,
    seed: 3405,
    stateNames: ['low', 'middle', 'high'],
  },
];

export interface RegimeSimulatorPreset {
  id: string;
  label: string;
  stay: [number, number];
  means: [number, number];
  sds: [number, number];
}

export const regimeSimulatorPresets: RegimeSimulatorPreset[] = [
  { id: 'calm-turbulent', label: 'Calm / turbulent', stay: [0.98, 0.95], means: [0.05, -0.1], sds: [0.7, 2.0] },
  // Every row of A equal to π = (5/7, 2/7): a1→1 = 5/7 and a2→2 = 2/7, so the labels are i.i.d.
  { id: 'no-memory', label: 'No memory', stay: [5 / 7, 2 / 7], means: [0.05, -0.1], sds: [0.7, 2.0] },
  { id: 'same-mean', label: 'Same mean, different volatility', stay: [0.98, 0.95], means: [0, 0], sds: [0.7, 2.0] },
];

export const viterbiPresets: HMMPreset[] = [
  { id: 'casino', label: 'Casino', params: casinoParams, T: 150, seed: 3412, stateNames: ['fair', 'loaded'] },
];

export const baumWelchPresets: HMMPreset[] = [
  { id: 'market2', label: 'Market, K = 2', params: marketParams, T: 1000, seed: 3413, stateNames: ['calm', 'turbulent'] },
  {
    id: 'market3',
    label: 'Market, K = 3',
    params: market3Params,
    T: 1500,
    seed: 3413,
    stateNames: ['calm', 'normal', 'crisis'],
  },
];

// ═══════════════════════════════════════════════════════════════════════════
// Topic 35 — Stationary Time Series & ARMA
// ═══════════════════════════════════════════════════════════════════════════

export interface ARMARootPreset {
  id: string;
  label: string;
  /** Roots of φ(z); complex roots listed with their conjugates. */
  arRoots: Complex[];
  /** Roots of θ(z). */
  maRoots: Complex[];
  sigma2: number;
  n: number;
  seed: number;
}

const rootPreset = (id: string, label: string, arRoots: Complex[], maRoots: Complex[]): ARMARootPreset => ({
  id,
  label,
  arRoots,
  maRoots,
  sigma2: 1,
  n: 300,
  seed: 3541,
});

const peakRe = Math.cos(Math.PI / 5) / 0.9;
const peakIm = Math.sin(Math.PI / 5) / 0.9;
const ma2Im = Math.sqrt(1.24) / 0.8;

/** ARMARootExplorer presets; `ar2peak` is the default. */
export const armaRootPresets: ARMARootPreset[] = [
  rootPreset('ar1', 'AR(1), φ = 0.7', [{ re: 1 / 0.7, im: 0 }], []),
  rootPreset('ar2peak', 'AR(2), spectral peak', [{ re: peakRe, im: peakIm }, { re: peakRe, im: -peakIm }], []),
  rootPreset('ar2osc', 'AR(2), roots 1 ± i', [{ re: 1, im: 1 }, { re: 1, im: -1 }], []),
  rootPreset('ma2', 'MA(2), θ = (0.6, 0.4)', [], [{ re: -0.75, im: ma2Im }, { re: -0.75, im: -ma2Im }]),
  rootPreset('arma11', 'ARMA(1,1), φ = 0.7, θ = 0.4', [{ re: 1 / 0.7, im: 0 }], [{ re: -2.5, im: 0 }]),
  rootPreset('unitroot', 'Unit root (ARIMA in disguise)', [{ re: 1, im: 0 }, { re: 5, im: 0 }], []),
];

export interface PeriodogramPreset {
  id: string;
  label: string;
  kind: 'arma' | 'sunspots';
  params?: ARMAParams;
  n: number;
  m: number;
  seed: number;
}

/** PeriodogramSmoother presets; `ar2peak` is the default. */
export const periodogramPresets: PeriodogramPreset[] = [
  {
    id: 'ar2peak',
    label: 'AR(2) peak',
    kind: 'arma',
    params: { phi: [1.8 * Math.cos(Math.PI / 5), -0.81], theta: [], sigma2: 1 },
    n: 1024,
    m: 4,
    seed: 3513,
  },
  { id: 'ar1', label: 'AR(1), φ = 0.9', kind: 'arma', params: { phi: [0.9], theta: [], sigma2: 1 }, n: 1024, m: 4, seed: 3513 },
  { id: 'ma1', label: 'MA(1), θ = −0.8', kind: 'arma', params: { phi: [], theta: [-0.8], sigma2: 1 }, n: 1024, m: 4, seed: 3513 },
  { id: 'wn', label: 'White noise', kind: 'arma', params: { phi: [], theta: [], sigma2: 1 }, n: 1024, m: 4, seed: 3513 },
  { id: 'sunspots', label: 'Sunspots', kind: 'sunspots', n: 309, m: 2, seed: 3513 },
];

export interface GarchPreset {
  id: string;
  label: string;
  /** null → simulate Topic 34's market HMM (`marketParams`). */
  params: GarchParams | null;
  T: number;
  seed: number;
}

/** GarchVsHmmSimulator presets; `textbook` is the default. */
export const garchPresets: GarchPreset[] = [
  { id: 'textbook', label: 'Textbook (0.05, 0.1, 0.85)', params: { omega: 0.05, alpha: 0.1, beta: 0.85 }, T: 1000, seed: 3514 },
  { id: 'fitToHmm', label: 'Fit to HMM data (0.0803, 0.2821, 0.7028)', params: { omega: 0.0803, alpha: 0.2821, beta: 0.7028 }, T: 1000, seed: 3514 },
  { id: 'nearIgarch', label: 'Near-IGARCH (0.01, 0.09, 0.9)', params: { omega: 0.01, alpha: 0.09, beta: 0.9 }, T: 1000, seed: 3514 },
  { id: 'hmmMarket', label: 'HMM market (Topic 34)', params: null, T: 1000, seed: 3514 },
];

export interface WoldPreset {
  id: string;
  label: string;
  params: ARMAParams;
  seed: number;
}

/** WoldPredictionProjector presets. */
export const woldPresets: WoldPreset[] = [
  { id: 'ar1', label: 'AR(1), φ = 0.9', params: { phi: [0.9], theta: [], sigma2: 1 }, seed: 3542 },
  { id: 'ar2osc', label: 'AR(2), roots 1 ± i', params: { phi: [1, -0.5], theta: [], sigma2: 1 }, seed: 3542 },
  { id: 'ma2', label: 'MA(2), θ = (0.6, 0.4)', params: { phi: [], theta: [0.6, 0.4], sigma2: 1 }, seed: 3542 },
  { id: 'arma11', label: 'ARMA(1,1), φ = 0.7, θ = 0.4', params: { phi: [0.7], theta: [0.4], sigma2: 1 }, seed: 3542 },
];

// ═══ Topic 36 — State-Space Models & Kalman Smoothing ═══

export interface KalmanExplorerPreset {
  id: string;
  label: string;
  source: 'nile' | 'simulated';
  /** 0-based half-open index ranges [a, b) set to missing. */
  gaps: [number, number][];
  q: number;
  r: number;
  seed: number;
}

/** KalmanSmootherExplorer presets (q, r = the Nile direct MLE, Ex 1). */
export const kalmanExplorerPresets: KalmanExplorerPreset[] = [
  { id: 'nile', label: 'Nile 1871–1970', source: 'nile', gaps: [], q: 1463.5474, r: 15108.3145, seed: 3616 },
  { id: 'nileGaps', label: 'Nile, DK gaps', source: 'nile', gaps: [[20, 40], [60, 80]], q: 1463.5474, r: 15108.3145, seed: 3616 },
  { id: 'simulated', label: 'Simulated local level', source: 'simulated', gaps: [], q: 1463.5474, r: 15108.3145, seed: 3616 },
];

export interface EllipseStepperPreset {
  id: string;
  label: string;
  dt: number;
  qa: number;
  r: number;
  observe: 'position' | 'velocity' | 'both';
  seed: number;
}

/** CovarianceEllipseStepper presets (Ex 3's constant-velocity model). */
export const ellipseStepperPresets: EllipseStepperPreset[] = [
  { id: 'ex3', label: 'Ex 3: observe position', dt: 1, qa: 0.3, r: 0.4, observe: 'position', seed: 3602 },
  { id: 'velocityOnly', label: 'Observe velocity only', dt: 1, qa: 0.3, r: 0.4, observe: 'velocity', seed: 3602 },
  { id: 'noisy', label: 'Noisy position (r = 4)', dt: 1, qa: 0.3, r: 4, observe: 'position', seed: 3602 },
];

export interface LikelihoodSurfacePreset {
  id: string;
  label: string;
  mode: 'localLevel' | 'arma11';
  emStart?: [number, number];
  /** localLevel: log10 q × log10 r; arma11: φ × θ. */
  grid: { x: [number, number]; y: [number, number]; n: number };
}

/** StateSpaceLikelihoodSurface presets (mode A = Nile (q, r); mode B = ARMA(1,1) (φ, θ)). */
export const likelihoodSurfacePresets: LikelihoodSurfacePreset[] = [
  { id: 'nile', label: 'A: Nile local level (q, r)', mode: 'localLevel', emStart: [10000, 5000], grid: { x: [2, 4.6], y: [3.6, 4.6], n: 60 } },
  { id: 'arma11', label: 'B: ARMA(1,1) (φ, θ)', mode: 'arma11', grid: { x: [0.45, 0.85], y: [0.2, 0.6], n: 50 } },
];

export interface StructuralPreset {
  id: string;
  label: string;
  level: number;
  slope: number | null;
  seasonal: number | null;
  irregular: number;
}

/** StructuralDecomposer presets: the UnobservedComponents MLE of Ex 13 (null = component off). */
export const structuralPresets: StructuralPreset[] = [
  { id: 'co2Mle', label: 'Trend + seasonal (MLE)', level: 0.050844309968, slope: 3.450003e-6, seasonal: 1.0336517e-5, irregular: 0.024022091168 },
  { id: 'levelOnly', label: 'Level only', level: 0.050844309968, slope: null, seasonal: null, irregular: 0.024022091168 },
];
