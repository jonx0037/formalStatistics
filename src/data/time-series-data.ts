/**
 * Topic 34 — Hidden Markov Models: preset parameters for the four §34
 * components and the timeseries.test.ts harness.
 *
 * Values are locked to notebook Cell 4 / Cell 11
 * (notebooks/hidden-markov-models/34_hidden_markov_models.ipynb); see
 * handoff brief §7. Typed presets, not `as const` (readonly arrays do not
 * type-check against HMMParams — Topic 33 Appendix B.10).
 */

import { stationaryDistribution, type HMMParams } from '../components/viz/shared/timeseries';

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
