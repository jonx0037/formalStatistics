/**
 * src/data/empirical-processes-data.ts
 *
 * Preset data for Topic 32's interactive components:
 *   • EmpiricalProcessExplorer  (§32.5)  — baseDistributions
 *   • FunctionalDeltaExplorer   (§32.7)  — functionalPresets
 *   • VCShatteringDemo          (§32.4)  — vcClasses
 *
 * Numerical primitives reused from `src/components/viz/shared/distributions.ts`
 * (Topic 6) and `src/components/viz/shared/bayes.ts` (Topic 26). No new math
 * primitives introduced here — this file is pure configuration.
 *
 * Distribution coverage rationale (§32.5 / §32.7 design decision):
 *   • Uniform(0, 1)  — canonical example; CDF is identity, so the empirical
 *                      process visual is the classical one from Doob (1949).
 *   • Normal(0, 1)   — lets us show that Donsker's theorem is distribution-
 *                      free for the sup-norm: K(x) overlay works identically
 *                      after the probability-integral transform.
 *   • Exponential(1) — asymmetric, unbounded-support test case; verifies the
 *                      student's mental model survives leaving [0, 1].
 *
 * The three-dist × three-functional matrix (§7 preset tables) was locked after
 * dropping the KS functional from the brief's four-functional draft: under
 * F = F₀ the KS functional is not Hadamard differentiable, and under F ≠ F₀
 * it's asymptotically Gaussian with a degenerate influence function — neither
 * scenario illustrates the delta method cleanly. The CvM functional carries
 * the non-Gaussian-limit pedagogical payload the KS functional would have.
 */

import {
  cdfStdNormal,
  pdfStdNormal,
} from '../components/viz/shared/distributions';
import type { SeededRng } from '../components/viz/shared/bayes';

// ────────────────────────────────────────────────────────────────────────────
// Base distributions for EmpiricalProcessExplorer (§32.5)
// ────────────────────────────────────────────────────────────────────────────

export type BaseDistKey = 'uniform' | 'normal' | 'exponential';

export interface BaseDistributionSpec {
  key: BaseDistKey;
  label: string;
  /** Single-sample draw via `rng.random()` / `rng.normal()`. */
  sampler: (rng: SeededRng) => number;
  /** F(t) on the plotted grid. */
  Fcdf: (t: number) => number;
  /** Display grid for 𝔾_n paths — 300-point discretisation of the support. */
  tGrid: readonly number[];
  /** Axis-limit range for path rendering. */
  tGridRange: readonly [number, number];
  /** Horizontal-axis label. */
  xAxisLabel: string;
}

const GRID_N = 300;
const gridOver = (a: number, b: number): number[] =>
  Array.from({ length: GRID_N }, (_, i) => a + (b - a) * (i / (GRID_N - 1)));

export const uniformBase: BaseDistributionSpec = {
  key: 'uniform',
  label: 'Uniform(0, 1)',
  sampler: (rng) => rng.random(),
  Fcdf: (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t),
  tGrid: gridOver(0, 1),
  tGridRange: [0, 1] as const,
  xAxisLabel: 't',
};

export const normalBase: BaseDistributionSpec = {
  key: 'normal',
  label: 'Normal(0, 1)',
  sampler: (rng) => rng.normal(),
  Fcdf: cdfStdNormal,
  tGrid: gridOver(-3, 3),
  tGridRange: [-3, 3] as const,
  xAxisLabel: 'x',
};

export const exponentialBase: BaseDistributionSpec = {
  key: 'exponential',
  // Inverse-CDF: rate 1 → X = −log(1 − U). Use (1 − U) directly so we never
  // log(0) when U = 0 — the LCG returns zero only on a single state (period
  // 2³¹−1), but belt-and-braces costs nothing.
  label: 'Exponential(1)',
  sampler: (rng) => -Math.log(1 - rng.random()),
  Fcdf: (t) => (t <= 0 ? 0 : 1 - Math.exp(-t)),
  tGrid: gridOver(0, 6),
  tGridRange: [0, 6] as const,
  xAxisLabel: 'x',
};

export const baseDistributions: readonly BaseDistributionSpec[] = [
  uniformBase,
  normalBase,
  exponentialBase,
] as const;

// ────────────────────────────────────────────────────────────────────────────
// Functionals for FunctionalDeltaExplorer (§32.7)
// ────────────────────────────────────────────────────────────────────────────

export type FunctionalKey = 'mean' | 'median' | 'cvm';

/**
 * Per (functional × distribution) lookup. `asymptoticVariance = 0` is a
 * sentinel: the component draws the weighted-χ² Cramér-von-Mises overlay
 * from the preset histogram instead of a Gaussian density.
 */
export interface FunctionalPreset {
  key: FunctionalKey;
  label: string;
  /** φ applied to a finite sample. */
  phi: (sample: readonly number[]) => number;
  /** φ(F) for overlay centering. */
  phiAtF: (dist: BaseDistKey) => number;
  /** Scalar Hadamard / influence-function variance P(IC²). Zero → non-Gaussian. */
  asymptoticVariance: (dist: BaseDistKey) => number;
  /**
   * One-sentence pedagogical caption rendered under the histogram. Written to
   * match the adjacent MDX prose — small diction matters.
   */
  description: string;
}

// ── Sample statistics (shared closures) ──────────────────────────────────────

const sampleMean = (s: readonly number[]): number => {
  let t = 0;
  for (const v of s) t += v;
  return t / s.length;
};

const sampleMedian = (s: readonly number[]): number => {
  const n = s.length;
  const sorted = [...s].sort((a, b) => a - b);
  return n % 2 === 0
    ? 0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    : sorted[(n - 1) / 2];
};

// CvM statistic W²_n = 1/(12n) + Σ_{i=1}^{n} [F₀(X_(i)) − (2i − 1)/(2n)]²
// with F₀ = Standard Normal. The 1/(12n) offset + Anderson-Darling-style
// alignment is the canonical closed form (see VDV2000 §19.4).
const cvmStatistic = (sample: readonly number[]): number => {
  const n = sample.length;
  const sorted = [...sample].sort((a, b) => a - b);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const Fxi = cdfStdNormal(sorted[i]);
    const target = (2 * i + 1) / (2 * n);
    const d = Fxi - target;
    acc += d * d;
  }
  return 1 / (12 * n) + acc;
};

// ── Per-distribution constants (memoisation would be overkill) ───────────────

const meanAtF = (d: BaseDistKey): number =>
  d === 'uniform' ? 0.5 : d === 'normal' ? 0 : 1;

const medianAtF = (d: BaseDistKey): number =>
  d === 'uniform' ? 0.5 : d === 'normal' ? 0 : Math.LN2;

const densityAtMedian = (d: BaseDistKey): number =>
  d === 'uniform' ? 1 : d === 'normal' ? pdfStdNormal(0) : 0.5; // exp(1): f(ln2) = 0.5

export const meanFunctional: FunctionalPreset = {
  key: 'mean',
  label: 'Sample mean',
  phi: sampleMean,
  phiAtF: meanAtF,
  asymptoticVariance: (d) =>
    d === 'uniform' ? 1 / 12 : d === 'normal' ? 1 : 1, // exp(1) var = 1
  description: 'φ(F) = ∫ x dF. Influence function IC(x) = x − μ; asymptotic variance P(IC²) = Var(X).',
};

export const medianFunctional: FunctionalPreset = {
  key: 'median',
  label: 'Sample median',
  phi: sampleMedian,
  phiAtF: medianAtF,
  asymptoticVariance: (d) => {
    const f = densityAtMedian(d);
    return 1 / (4 * f * f);
  },
  description: 'φ(F) = F⁻¹(½). IC(x) = −sign(x − m) / (2 f(m)); asymptotic variance = 1 / (4 f(m)²).',
};

export const cvmFunctional: FunctionalPreset = {
  key: 'cvm',
  label: 'Cramér–von-Mises (vs N(0, 1))',
  phi: cvmStatistic,
  phiAtF: (d) => (d === 'normal' ? 0 : 0.1),
  // Sentinel: under F = F₀, W²_n ⟹ Σ_{k≥1} χ²₁ / (k π)² — a weighted-χ² limit,
  // NOT Gaussian. Under F ≠ F₀, the √n scaling produces a Gaussian by the
  // delta method, but the influence function is nontrivial. The component
  // branches on asymptoticVariance === 0 to pick the correct overlay.
  asymptoticVariance: () => 0,
  description: 'φ(F) = ∫ (F − F₀)² dF₀. Limit under H₀ is Σ χ²₁ / (kπ)² (non-Gaussian); see §32.8 Ex 10.',
};

export const functionalPresets: readonly FunctionalPreset[] = [
  meanFunctional,
  medianFunctional,
  cvmFunctional,
] as const;

// ────────────────────────────────────────────────────────────────────────────
// VC classes for VCShatteringDemo (§32.4)
// ────────────────────────────────────────────────────────────────────────────

export type VCClassKey = 'halfline' | 'halfspace-2d' | 'rectangle-2d';

export interface VCClassSpec {
  key: VCClassKey;
  label: string;
  /** VC dimension — used for the Sauer-bound readout and the labelings grid. */
  vcDim: number;
  /** Ambient dimension of the input space. */
  inputDim: 1 | 2;
  /**
   * Canonical shatter-witness configuration rendered when the user first
   * switches to this class. For halflines a single point on the number line,
   * for halfspaces three non-collinear points, for rectangles a 4-point
   * diamond (the canonical 4-point shatter witness).
   */
  witnessPoints: readonly (number | readonly [number, number])[];
  description: string;
}

export const halflineClass: VCClassSpec = {
  key: 'halfline',
  label: 'Halflines on ℝ',
  vcDim: 1,
  inputDim: 1,
  witnessPoints: [0.3] as const,
  description: 'Classifiers {x ≤ t}. VC dim 1 — shatters any 1 point, never 2.',
};

export const halfspaceClass: VCClassSpec = {
  key: 'halfspace-2d',
  label: 'Halfspaces in ℝ²',
  vcDim: 3,
  inputDim: 2,
  witnessPoints: [[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]] as const,
  description: 'Classifiers {w · x ≤ b}. VC dim 3 — shatters any 3 non-collinear points; Radon blocks n = 4.',
};

export const rectangleClass: VCClassSpec = {
  key: 'rectangle-2d',
  label: 'Axis-aligned rectangles in ℝ²',
  vcDim: 4,
  inputDim: 2,
  witnessPoints: [
    [0.5, 0.2],
    [0.8, 0.5],
    [0.5, 0.8],
    [0.2, 0.5],
  ] as const,
  description: 'Classifiers {(x₁, x₂) : a₁ ≤ x₁ ≤ b₁, a₂ ≤ x₂ ≤ b₂}. VC dim 4 — 4-diamond shatters; n = 5 blocked by inner-point containment.',
};

export const vcClasses: readonly VCClassSpec[] = [
  halflineClass,
  halfspaceClass,
  rectangleClass,
] as const;
