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

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type DistributionKey = 'normal' | 'exp' | 'beta' | 'cauchy' | 'uniform';

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
};

/** Beta(2, 5) — right-skewed, mode at 0.2. */
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

// ────────────────────────────────────────────────────────────────────────────
// Seeded RNG helper — linear congruential generator (LCG) for reproducibility
// across component renders. Not cryptographically strong; fine for Monte Carlo.
// ────────────────────────────────────────────────────────────────────────────

/**
 * Returns a deterministic U(0, 1) draw generator seeded by `seed`. Standard
 * mulberry32 LCG (m=2³², 32-bit). Each component uses this to make the
 * "Resample" button predictable — same seed, same picture.
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
