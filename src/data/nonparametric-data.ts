/**
 * src/data/nonparametric-data.ts
 *
 * Preset data for Track 8 interactive components. Consumed by:
 *   • QuantileAsymptoticsExplorer (§29.6)
 *   • ECDFDKWBandExplorer        (§29.5)
 *   • OrderStatisticDensityBrowser (§29.3)
 *   • KSGoodnessExplorer          (§29.8)
 *
 * Each DistributionPreset bundles a CDF, PDF, quantile, a single-draw sampler,
 * and a plotting domain. The CDFs/PDFs reuse the numerical primitives already
 * exported by `src/components/viz/shared/distributions.ts` (Topic 6 module) —
 * no new math primitives are introduced here.
 */

import {
  cdfStdNormal,
  pdfStdNormal,
  quantileStdNormal,
  cdfBeta,
  pdfBeta,
} from '../components/viz/shared/distributions';
import { type SeededRng } from '../components/viz/shared/bayes';

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type DistributionKey =
  | 'normal'
  | 'exp'
  | 'beta'
  | 'cauchy'
  | 'uniform'
  | 'bimodal'
  | 't3';

/**
 * Keys of scalar statistics the Topic 31 components track. Tightens the
 * `trueParameterByStat` typing so misspellings like `.meam` fail to compile.
 * Extend cautiously: components iterate over these keys for their statistic
 * selector, so adding a new entry means every consumer must handle it.
 */
export type StatKey = 'mean' | 'median' | 'variance';

export interface DistributionPreset {
  /** Display name, e.g. "Normal(0, 1)". */
  name: string;
  /** Stable identifier for UI state. */
  key: DistributionKey;
  /** F(x). */
  cdf: (x: number) => number;
  /** f(x). */
  pdf: (x: number) => number;
  /** True population quantile ξ_p = F⁻¹(p) for the QuantileAsymptoticsExplorer limit overlay. */
  quantile: (p: number) => number;
  /** Single draw from F given a U(0,1) source. */
  sampler: (rng: () => number) => number;
  /** Plot-window x-range; heavy-tailed distributions clip for readability. */
  domain: [number, number];

  // ── Topic 30 KDE extensions (all optional; backward-compatible) ──────────

  /**
   * Second derivative f''(x). Required by `BandwidthExplorer` to compute the
   * curvature integral R(f'') = ∫f''² and the AMISE-optimal bandwidth h*.
   * If absent, the AMISE-oracle panel falls back to Silverman's rule.
   */
  d2pdf?: (x: number) => number;

  /**
   * Curvature integral R(f'') = ∫ f''(x)² dx. Closed-form if available
   * (3/(8√π) for Normal(0,1)); undefined for densities where numerical
   * integration is required at component mount (e.g. Bimodal Normal mixture).
   */
  R_f_dd?: number;

  /**
   * Support [a, b] of f — the interval outside which f(x) = 0. Used by
   * `BoundaryBiasDemo` to trigger reflection correction at the boundary.
   * Absent means full ℝ (or effectively so for unbounded-support densities).
   */
  support?: [number, number];

  // ── Topic 31 Bootstrap extensions (all optional; backward-compatible) ────

  /**
   * Bulk SeededRng-based sampler: returns a length-n array of iid draws.
   * Distinct from `sampler` (single draw from a U(0,1) callable) — the
   * Topic 31 CI components use this path so they can share one `SeededRng`
   * across bootstrap replicates without packaging a U(0,1) closure.
   * Existing Topic 30 components ignore this field.
   */
  sample?: (n: number, rng: SeededRng) => number[];

  /**
   * True population value of named scalar statistics. Used by
   * `BootstrapCIComparator` to draw a vertical reference line for coverage.
   * Keys are constrained to `StatKey` so misspellings fail at compile time.
   * Omitted for presets where one or more statistics don't exist (Cauchy's
   * mean, Uniform's skewness, etc.); partial coverage is allowed.
   */
  trueParameterByStat?: Partial<Record<StatKey, number>>;
}

// ────────────────────────────────────────────────────────────────────────────
// Presets
// ────────────────────────────────────────────────────────────────────────────

/** Standard normal N(0, 1). */
export const normalPreset: DistributionPreset = {
  name: 'Normal(0, 1)',
  key: 'normal',
  cdf: cdfStdNormal,
  pdf: pdfStdNormal,
  quantile: quantileStdNormal,
  sampler: (rng) => {
    // Box-Muller: U₁, U₂ → √(−2 ln U₁) · cos(2π U₂).
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  },
  domain: [-4, 4],
  // Topic 30: f''(x) = (x² − 1) · φ(x); R(f'') = 3/(8√π).
  d2pdf: (x) => (x * x - 1) * pdfStdNormal(x),
  R_f_dd: 3 / (8 * Math.sqrt(Math.PI)),
};

/** Exp(rate = 1). */
export const expPreset: DistributionPreset = {
  name: 'Exp(1)',
  key: 'exp',
  cdf: (x) => (x >= 0 ? 1 - Math.exp(-x) : 0),
  pdf: (x) => (x >= 0 ? Math.exp(-x) : 0),
  quantile: (p) => -Math.log(1 - p),
  sampler: (rng) => -Math.log(1 - rng()),
  domain: [0, 6],
  // Topic 30: f''(x) = e^{-x} on x ≥ 0; R(f'') = ∫₀^∞ e^{-2x} dx = 1/2.
  d2pdf: (x) => (x >= 0 ? Math.exp(-x) : 0),
  R_f_dd: 0.5,
  support: [0, Infinity],
};

/** Beta(2, 5) — right-skewed, mode at 0.2, compact support [0, 1]. */
export const betaPreset: DistributionPreset = {
  name: 'Beta(2, 5)',
  key: 'beta',
  cdf: (x) => cdfBeta(x, 2, 5),
  pdf: (x) => pdfBeta(x, 2, 5),
  quantile: (p) => {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    // Bisection on cdfBeta; fewer iterations than the shared quantileBeta's.
    let lo = 1e-6;
    let hi = 1 - 1e-6;
    while (hi - lo > 1e-6) {
      const mid = 0.5 * (lo + hi);
      if (cdfBeta(mid, 2, 5) < p) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  },
  sampler: (rng) => {
    // Mode-envelope rejection. Mode of Beta(2, 5) is (α−1)/(α+β−2) = 1/5.
    const mode = 0.2;
    const envelope = pdfBeta(mode, 2, 5); // ≈ 2.4576
    for (let iter = 0; iter < 256; iter++) {
      const x = rng();
      if (rng() * envelope < pdfBeta(x, 2, 5)) return x;
    }
    return mode; // safety fallback; probability of 256 rejections is < 1e-50.
  },
  domain: [0, 1],
  // Topic 30: f(x) = 30 x (1−x)⁴ on [0,1];
  // f'(x)  = 30 (1−x)³ (1 − 5x);
  // f''(x) = 60 (1−x)² (10x − 4).
  d2pdf: (x) => (x < 0 || x > 1 ? 0 : 60 * (1 - x) ** 2 * (10 * x - 4)),
  // R(f'') = ∫₀¹ [60(1−x)²(10x−4)]² dx ≈ 5348.57 (numerical; the curvature
  // piles up near the mode at x = 0.2 so AMISE-optimal h is very small).
  R_f_dd: 5348.571428571429,
  support: [0, 1],
};

/** Standard Cauchy — the money-shot preset for QuantileAsymptoticsExplorer. */
export const cauchyPreset: DistributionPreset = {
  name: 'Cauchy(0, 1)',
  key: 'cauchy',
  cdf: (x) => 0.5 + Math.atan(x) / Math.PI,
  pdf: (x) => 1 / (Math.PI * (1 + x * x)),
  quantile: (p) => Math.tan(Math.PI * (p - 0.5)),
  sampler: (rng) => Math.tan(Math.PI * (rng() - 0.5)),
  // Heavy tails — clip plotting domain to keep histograms readable. The MC
  // statistics still see the raw (unclipped) draws.
  domain: [-6, 6],
};

/** Uniform(0, 1). */
export const uniformPreset: DistributionPreset = {
  name: 'Uniform(0, 1)',
  key: 'uniform',
  cdf: (x) => Math.max(0, Math.min(1, x)),
  pdf: (x) => (x >= 0 && x <= 1 ? 1 : 0),
  quantile: (p) => p,
  sampler: (rng) => rng(),
  domain: [0, 1],
};

/**
 * Bimodal equal-mix of two unit-variance Gaussians centered at ±1.5:
 *   f(x) = 0.5 · φ(x + 1.5) + 0.5 · φ(x − 1.5).
 *
 * Mean 0, variance 1 + 1.5² = 3.25. The separation |μ₁ − μ₂| = 3 (roughly 3σ
 * in each component) makes this a bimodal-resolvable case at moderate n —
 * the flagship preset for the `BandwidthExplorer` and `PluginBandwidthComparator`
 * components, and the preset where Silverman's rule over-smooths visibly.
 * (Topic 30 §30.8 Ex 10)
 */
export const bimodalNormalPreset: DistributionPreset = {
  name: '0.5·N(−1.5, 1) + 0.5·N(1.5, 1)',
  key: 'bimodal',
  pdf: (x) => 0.5 * pdfStdNormal(x + 1.5) + 0.5 * pdfStdNormal(x - 1.5),
  cdf: (x) => 0.5 * cdfStdNormal(x + 1.5) + 0.5 * cdfStdNormal(x - 1.5),
  quantile: (p) => {
    // Bisection on the bimodal CDF. Mixture CDF is strictly monotone so
    // bisection converges in ~30 iterations to 1e-8 tolerance.
    if (p <= 0) return -6;
    if (p >= 1) return 6;
    const mixCdf = (x: number): number =>
      0.5 * cdfStdNormal(x + 1.5) + 0.5 * cdfStdNormal(x - 1.5);
    let lo = -10;
    let hi = 10;
    for (let i = 0; i < 60 && hi - lo > 1e-8; i++) {
      const mid = 0.5 * (lo + hi);
      if (mixCdf(mid) < p) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  },
  sampler: (rng) => {
    // Single draw: pick a component uniformly, then Box-Muller within it.
    const mu = rng() < 0.5 ? -1.5 : 1.5;
    const u1 = rng();
    const u2 = rng();
    return mu + Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  },
  domain: [-5, 5],
  /**
   * Closed-form f''(x) for the mixture:
   *   f''(x) = 0.5 · [(x+1.5)² − 1] · φ(x+1.5)
   *          + 0.5 · [(x−1.5)² − 1] · φ(x−1.5).
   * R(f'') has no clean closed form — `BandwidthExplorer` integrates on mount
   * via Simpson's rule on a padded grid when computing the oracle h*.
   */
  d2pdf: (x) => {
    const phi1 = pdfStdNormal(x + 1.5);
    const phi2 = pdfStdNormal(x - 1.5);
    return 0.5 * ((x + 1.5) ** 2 - 1) * phi1 + 0.5 * ((x - 1.5) ** 2 - 1) * phi2;
  },
  // Numerical R(f'') via scipy.integrate.quad on [-15, 15] — captures both
  // modes of the mixture. Value is stable to 12 digits.
  R_f_dd: 0.091848403647,
};

// ────────────────────────────────────────────────────────────────────────────
// Component-specific preset groupings
// ────────────────────────────────────────────────────────────────────────────

/** `QuantileAsymptoticsExplorer` (§29.6). The Cauchy preset is the money-shot. */
export const quantileAsymptoticsPresets: Record<string, DistributionPreset> = {
  normal: normalPreset,
  exp: expPreset,
  beta: betaPreset,
  cauchy: cauchyPreset,
};

/** `ECDFDKWBandExplorer` (§29.5). */
export const ecdfDkwPresets: Record<string, DistributionPreset> = {
  normal: normalPreset,
  exp: expPreset,
  uniform: uniformPreset,
  cauchy: cauchyPreset,
};

/** `OrderStatisticDensityBrowser` (§29.3) — the Uniform case is the theorem's payoff. */
export const orderStatPresets: Record<string, DistributionPreset> = {
  uniform: uniformPreset,
  exp: expPreset,
  normal: normalPreset,
};

/** `KSGoodnessExplorer` null-distribution presets (§29.8). */
export const ksPresets: Record<string, DistributionPreset> = {
  normal: normalPreset,
  uniform: uniformPreset,
  exp: expPreset,
};

/**
 * Topic 30 KDE components (`KernelChoiceExplorer`, `BandwidthExplorer`,
 * `PluginBandwidthComparator`, `BoundaryBiasDemo`) share this three-preset
 * bundle. Order is the default segmented-control sequence: Unimodal Normal
 * (the textbook case), Bimodal mixture (the case where data-driven
 * selectors earn their keep), and Beta(2, 5) (compact support for the
 * boundary-bias demo).
 */
export const kdePresets = [
  normalPreset,
  bimodalNormalPreset,
  betaPreset,
] as const;

// ────────────────────────────────────────────────────────────────────────────
// Topic 31 heavy-tailed preset (Student-t, df = 3). Promoted from Topic 30's
// `PluginBandwidthComparator` inline and re-used by the three Topic 31
// interactive components (`BootstrapDistributionExplorer`,
// `BootstrapCIComparator`, `SmoothBootstrapDemo`). Finite variance (= 3) but
// tails heavy enough that parametric-Wald CIs degrade visibly — the motivating
// case for bootstrap's distribution-free advantage.
// ────────────────────────────────────────────────────────────────────────────

/** Closed-form Student-t_3 CDF (Abramowitz & Stegun 26.7.3). */
const t3Cdf = (x: number): number => {
  const c = x / Math.sqrt(3);
  return 0.5 + (1 / Math.PI) * (Math.atan(c) + c / (1 + c * c));
};

/** Student-t_3 PDF: (2/(π√3))·(1 + x²/3)^{−2}. */
const t3Pdf = (x: number): number =>
  (2 / (Math.PI * Math.sqrt(3))) * Math.pow(1 + (x * x) / 3, -2);

/** Bisection-inverted quantile for t_3 (no closed-form). Tol ≈ 1e-10 in 60 iters. */
const t3Quantile = (p: number): number => {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  let lo = -100;
  let hi = 100;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (t3Cdf(mid) < p) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
};

/** Student-t distribution with 3 degrees of freedom. */
export const heavyTailedPreset: DistributionPreset = {
  name: 'Student-t (df = 3)',
  key: 't3',
  cdf: t3Cdf,
  pdf: t3Pdf,
  quantile: t3Quantile,
  // Single-draw sampler via ratio Z / √(V/df): 1 Box-Muller normal + 3 BM
  // normals for the chi-squared, so 8 U(0,1) draws per t_3 sample.
  sampler: (rng) => {
    const bm = (): number => {
      const u1 = rng();
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
    };
    const z = bm();
    const v = bm() ** 2 + bm() ** 2 + bm() ** 2;
    return z / Math.sqrt(v / 3);
  },
  // Bulk SeededRng-based sampler. Uses rng.chiSquared(3) directly.
  sample: (n, rng) => {
    const out: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = rng.normal() / Math.sqrt(rng.chiSquared(3) / 3);
    }
    return out;
  },
  domain: [-6, 6],
  // True parameters: t_3 is symmetric about 0 with finite variance = 3.
  trueParameterByStat: {
    mean: 0,
    median: 0,
    variance: 3,
  },
};

/**
 * Combined preset set for Topic 31 components: Topic 30's three KDE presets
 * plus the new heavy-tailed Student-t_3. Topic 31 components iterate over
 * `allPresets` for the preset selector; Topic 30 components continue to
 * consume `kdePresets` unchanged.
 */
export const allPresets = [...kdePresets, heavyTailedPreset] as const;

// ────────────────────────────────────────────────────────────────────────────
// Seeded RNG helper — Mulberry32 for reproducibility across component renders.
// Not cryptographically strong; fine for Monte Carlo.
// ────────────────────────────────────────────────────────────────────────────

/**
 * Returns a deterministic U(0, 1) draw generator seeded by `seed`. Uses
 * Mulberry32 (Tommy Ettinger 2017) — a fast 32-bit counter-based RNG with a
 * period of 2^32. Each component uses this to make the "Resample" button
 * predictable: same seed, same picture.
 */
export function seededUniform(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
