/**
 * multi-testing-data.ts — Topic 20 preset data.
 *
 * Powers the four Topic 20 interactive components:
 *   MultipleTestingProcedureExplorer, FWERvsFDRComparator,
 *   BHAlgorithmVisualizer, SimultaneousCIBands.
 *
 * Presets mirror notebook Cell 12 reference values and the §6.2 T1 fixed
 * vector so that numerical ground truth in the brief aligns with the
 * interactive exploration surface.
 */

export interface ProcedureExplorerPreset {
  id: string;
  label: string;
  m: number;
  pi0: number;
  signalStrength: number;
  alpha: number;
  description: string;
}

export const procedureExplorerPresets: ProcedureExplorerPreset[] = [
  {
    id: 'dense-signal',
    label: 'Dense signal (π₀ = 0.5)',
    m: 100,
    pi0: 0.5,
    signalStrength: 2.5,
    alpha: 0.05,
    description:
      'Half of the hypotheses are true alternatives. BH power ≈ 0.7; Bonferroni loses substantial power.',
  },
  {
    id: 'moderate-sparse',
    label: 'Moderate sparse (π₀ = 0.8)',
    m: 100,
    pi0: 0.8,
    signalStrength: 3.0,
    alpha: 0.05,
    description:
      'Default exploration configuration; realistic for behavioral-science multi-test scenarios.',
  },
  {
    id: 'gwas-like',
    label: 'GWAS-like (π₀ = 0.99, m large)',
    m: 1000,
    pi0: 0.99,
    signalStrength: 3.5,
    alpha: 0.05,
    description:
      'Extreme sparsity regime; Bonferroni and Holm converge; BH retains most of the signal.',
  },
  {
    id: 'no-signal',
    label: 'No signal (π₀ = 1.0)',
    m: 100,
    pi0: 1.0,
    signalStrength: 0.0,
    alpha: 0.05,
    description:
      'Null-only scenario. All procedures should reject ≤ α·m hypotheses on average.',
  },
];

export interface FWERvsFDRPreset {
  id: string;
  label: string;
  procedure: 'bonferroni' | 'holm' | 'sidak' | 'hochberg' | 'bh' | 'by' | 'storey';
  m: number;
  pi0: number;
  alpha: number;
  nTrials: number;
}

export const fwerVsFdrPresets: FWERvsFDRPreset[] = [
  {
    id: 'bh-moderate',
    label: 'BH, moderate sparse',
    procedure: 'bh',
    m: 100,
    pi0: 0.8,
    alpha: 0.05,
    nTrials: 300,
  },
  {
    id: 'bonf-moderate',
    label: 'Bonferroni, moderate sparse',
    procedure: 'bonferroni',
    m: 100,
    pi0: 0.8,
    alpha: 0.05,
    nTrials: 300,
  },
  {
    id: 'by-dependent',
    label: 'BY under any dependence',
    procedure: 'by',
    m: 100,
    pi0: 0.8,
    alpha: 0.05,
    nTrials: 300,
  },
];

export interface BHAlgorithmPreset {
  id: string;
  label: string;
  /**
   * Either a fixed p-value vector OR a generator config (simulatePValues args)
   * for runtime generation. Components should use `pvals` directly when set;
   * otherwise call `simulatePValues(generator.m, generator.pi0, …, generator.seed)`.
   */
  pvals?: number[];
  generator?: {
    m: number;
    pi0: number;
    signalStrength: number;
    seed: number;
  };
  alpha: number;
  expectedKStar: number;
  description: string;
}

export const bhAlgorithmPresets: BHAlgorithmPreset[] = [
  {
    id: 't1-canonical',
    label: 'Canonical test vector (m = 10, α = 0.05)',
    pvals: [0.0005, 0.003, 0.011, 0.021, 0.043, 0.051, 0.087, 0.21, 0.33, 0.55],
    alpha: 0.05,
    expectedKStar: 3,
    description:
      'Regression-test vector from §6.2 T1. BH rejects ranks 1–3 (p_(3) = 0.011 passes threshold 0.015; p_(4) = 0.021 fails 0.020).',
  },
  {
    id: 'fig5-random',
    label: 'Figure 5 random sample (m = 20, α = 0.10)',
    generator: { m: 20, pi0: 0.5, signalStrength: 2.8, seed: 43 },
    alpha: 0.1,
    expectedKStar: 10,
    description:
      'Random-sample illustration; matches Figure 5 seed and parameters.',
  },
  {
    id: 'no-rejection',
    label: 'No rejection (all p > α)',
    pvals: [0.15, 0.22, 0.31, 0.48, 0.65],
    alpha: 0.05,
    expectedKStar: 0,
    description: 'Degenerate case — k* = 0, no rejections.',
  },
  {
    id: 'dense-all-reject',
    label: 'Dense signal (reject all, m = 10)',
    pvals: [
      0.0001, 0.0005, 0.002, 0.004, 0.007, 0.009, 0.012, 0.017, 0.022, 0.028,
    ],
    alpha: 0.1,
    expectedKStar: 10,
    description:
      'All ten hypotheses satisfy the BH threshold at α = 0.10; k* = m.',
  },
];

export interface SimultaneousCIPreset {
  id: string;
  label: string;
  m: number;
  n: number;
  alpha: number;
  /** Length-m vector of true parameter values used for empirical joint coverage. */
  trueTheta: number[];
  description: string;
}

export const simultaneousCIPresets: SimultaneousCIPreset[] = [
  {
    id: 'fig8-default',
    label: 'Figure 8 default (m = 10)',
    m: 10,
    n: 25,
    alpha: 0.05,
    trueTheta: [1.5, 2.0, 0.8, 3.1, 2.5, 1.2, 2.8, 0.5, 1.9, 2.2],
    description: 'Matches Figure 8 parameter configuration exactly.',
  },
  {
    id: 'small-m',
    label: 'Small family (m = 3)',
    m: 3,
    n: 30,
    alpha: 0.05,
    trueTheta: [1.0, 2.0, 3.0],
    description:
      'Minimal-m case; Bonferroni width inflation is modest (~1.15× unadjusted).',
  },
  {
    id: 'large-m',
    label: 'Large family (m = 30)',
    m: 30,
    n: 25,
    alpha: 0.05,
    trueTheta: Array.from({ length: 30 }, (_, i) => 1 + 0.1 * i),
    description:
      'Larger family — Bonferroni width ~1.55× unadjusted; unadjusted joint coverage drops to ≈ 0.78.',
  },
];
