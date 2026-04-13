/**
 * multivariate-distributions-data.ts — Presets and helper data for Topic 8.
 */

/** Covariance matrix presets for MultivariateNormalExplorer. */
export const mvnPresets = [
  { name: 'Spherical (ρ = 0)', sigma1: 1, sigma2: 1, rho: 0 },
  { name: 'Positive (ρ = 0.8)', sigma1: 1, sigma2: 1, rho: 0.8 },
  { name: 'Negative (ρ = −0.7)', sigma1: 1, sigma2: 1, rho: -0.7 },
  { name: 'Stretched (σ₁ = 3, σ₂ = 0.5)', sigma1: 3, sigma2: 0.5, rho: 0.3 },
];

/** Dirichlet presets for DirichletSimplexExplorer. */
export const dirichletPresets = [
  { name: 'Uniform (1, 1, 1)', alpha: [1, 1, 1] },
  { name: 'Concentrated (10, 10, 10)', alpha: [10, 10, 10] },
  { name: 'Sparse (0.1, 0.1, 0.1)', alpha: [0.1, 0.1, 0.1] },
  { name: 'Asymmetric (5, 1, 1)', alpha: [5, 1, 1] },
];

/** Marginal distribution options for CopulaExplorer. */
export const copulaMarginals = [
  { name: 'Exponential(1)', key: 'exponential' as const },
  { name: 'Normal(0, 1)', key: 'normal' as const },
  { name: 'Beta(2, 5)', key: 'beta' as const },
  { name: 'Uniform(0, 1)', key: 'uniform' as const },
];

export type MarginalKey = typeof copulaMarginals[number]['key'];

/**
 * Ternary coordinate mapping: 2-simplex → 2D Cartesian.
 * Maps (p₁, p₂, p₃) with Σpⱼ = 1 to an equilateral triangle.
 * Vertex labels: p₁ (bottom-left), p₂ (bottom-right), p₃ (top).
 */
export function ternaryCoords(p: [number, number, number]): { x: number; y: number } {
  return {
    x: p[1] + 0.5 * p[2],
    y: (Math.sqrt(3) / 2) * p[2],
  };
}

/** Simplex triangle vertices in 2D for drawing the equilateral triangle. */
export const simplexVertices = [
  { x: 0, y: 0, label: 'p₁' },
  { x: 1, y: 0, label: 'p₂' },
  { x: 0.5, y: Math.sqrt(3) / 2, label: 'p₃' },
];
