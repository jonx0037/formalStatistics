/**
 * likelihood-ratio-tests-data.ts — Presets for Topic 18's three required
 * interactive components (plus one optional). Follows Topic 17's pattern
 * (hypothesis-testing-data.ts): typed preset interfaces, pure-data exports,
 * and a local references object whose keys match the citations used in the
 * topic's MDX.
 *
 * Brief: §7.1 (handoff brief). The schema here adapts the brief's prescription
 * to match the existing Topic 17 data-file conventions:
 *   • preset arrays rather than plain objects (keeps downstream D3 loops clean)
 *   • a `topic18References` object keyed by citation id, mirroring `topic17References`
 *   • typed presets that share the `AsymptoticTestFamily` union from testing.ts
 */

import type { AsymptoticTestFamily } from '@/components/viz/shared/testing';

// ── Component 1: NPLemmaVisualizer ──────────────────────────────────────────

export interface NPLemmaPreset {
  id: string;
  name: string;
  family: AsymptoticTestFamily;
  theta0: number;
  theta1: number;
  sigma?: number;
  n: number;
  alpha: number;
  /** Slider range for θ₁ (the scenario-specific alternative). */
  theta1Range: [number, number];
  theta1Step: number;
  description: string;
}

export const npLemmaPresets: NPLemmaPreset[] = [
  {
    id: 'normal-vs-normal',
    name: 'Normal, μ₀=0 vs μ₁=1, σ=1, n=25',
    family: 'normal-mean-known-sigma',
    theta0: 0,
    theta1: 1,
    sigma: 1,
    n: 25,
    alpha: 0.05,
    theta1Range: [0.1, 3],
    theta1Step: 0.05,
    description:
      'The canonical NP setting — two Normal densities with the same variance but shifted means. The LR is monotone in x̄, so the NP rejection region is a right-tail of x̄ with threshold θ₀ + z_{1−α}σ/√n.',
  },
  {
    id: 'bernoulli-vs-bernoulli',
    name: 'Bernoulli, p₀=0.3 vs p₁=0.5, n=50',
    family: 'bernoulli',
    theta0: 0.3,
    theta1: 0.5,
    n: 50,
    alpha: 0.05,
    theta1Range: [0.05, 0.95],
    theta1Step: 0.01,
    description:
      'Discrete analogue. The LR takes only n + 1 distinct values (the number of successes ranges over {0, 1, …, n}), so the NP boundary is an integer threshold on ΣXᵢ. Exact size is usually < α — the conservative behavior of §18.4.',
  },
  {
    id: 'exponential-vs-exponential',
    name: 'Exponential rate, λ₀=1 vs λ₁=0.5, n=25',
    family: 'exponential',
    theta0: 1.0,
    theta1: 0.5,
    n: 25,
    alpha: 0.05,
    theta1Range: [0.1, 3],
    theta1Step: 0.05,
    description:
      'Continuous heavy-tailed alternative. Note the direction: a smaller rate (λ₁ = 0.5) means longer expected waits, so the LR increases in ΣXᵢ. NP rejects for large ΣXᵢ (long observed total).',
  },
];

// ── Component 2: KarlinRubinUMP ─────────────────────────────────────────────

export interface KarlinRubinPreset {
  id: string;
  name: string;
  family: AsymptoticTestFamily;
  theta0: number;
  theta1Default: number;
  theta1Range: [number, number];
  sigma?: number;
  n: number;
  alpha: number;
  /** T(x) name for the display panel. */
  Tform: string;
  /** Monotonicity direction of the LR in T: 'increasing' means reject for large T. */
  mlrDirection: 'increasing-in-T' | 'decreasing-in-T';
  description: string;
}

export const karlinRubinPresets: KarlinRubinPreset[] = [
  {
    id: 'bernoulli',
    name: 'Bernoulli H₀: p ≤ 0.5, T = ΣXᵢ',
    family: 'bernoulli',
    theta0: 0.5,
    theta1Default: 0.7,
    theta1Range: [0.05, 0.95],
    n: 50,
    alpha: 0.05,
    Tform: 'ΣXᵢ',
    mlrDirection: 'increasing-in-T',
    description:
      'The pedagogical payoff of §18.3: once MLR in ΣXᵢ is established, the binomial exact test is UMP for every θ₁ > 0.5 simultaneously. A single test dominates every alternative.',
  },
  {
    id: 'normal-mean',
    name: 'Normal mean (σ=1), H₀: μ ≤ 0, T = x̄',
    family: 'normal-mean-known-sigma',
    theta0: 0,
    theta1Default: 0.5,
    theta1Range: [-2, 2],
    sigma: 1,
    n: 25,
    alpha: 0.05,
    Tform: 'x̄',
    mlrDirection: 'increasing-in-T',
    description:
      'One-sided z-test is UMP under the Karlin-Rubin construction — the cleanest continuous example of MLR.',
  },
  {
    id: 'poisson',
    name: 'Poisson rate H₀: λ ≤ 2, T = ΣXᵢ',
    family: 'poisson',
    theta0: 2,
    theta1Default: 3,
    theta1Range: [0.5, 6],
    n: 30,
    alpha: 0.05,
    Tform: 'ΣXᵢ',
    mlrDirection: 'increasing-in-T',
    description:
      'Exponential-family MLR example — the Poisson rate test is UMP for every λ₁ > 2 via the same exponential-family canonical-form argument.',
  },
  {
    id: 'exponential',
    name: 'Exponential rate H₀: λ ≥ 1, T = ΣXᵢ',
    family: 'exponential',
    theta0: 1,
    theta1Default: 0.7,
    theta1Range: [0.1, 3],
    n: 25,
    alpha: 0.05,
    Tform: 'ΣXᵢ',
    mlrDirection: 'increasing-in-T',
    description:
      'Direction flip: smaller rate means longer waits, so the LR f(x;λ₁)/f(x;λ₀) is increasing in ΣXᵢ when λ₁ < λ₀. UMP rejects "too long a total wait" for the null rate λ₀.',
  },
];

// ── Component 3: WilksConvergence (featured) ────────────────────────────────

export interface WilksConvergencePreset {
  id: string;
  name: string;
  family: AsymptoticTestFamily;
  theta0: number;
  sigma?: number;
  /** Default n for the single-panel histogram. */
  nDefault: number;
  nRange: [number, number];
  /** The fixed four n values used in the right-panel mini-grid. */
  nGrid: [number, number, number, number];
  MDefault: number;
  MRange: [number, number];
  description: string;
}

export const wilksConvergencePresets: WilksConvergencePreset[] = [
  {
    id: 'bernoulli',
    name: 'Bernoulli H₀: p = 0.5',
    family: 'bernoulli',
    theta0: 0.5,
    nDefault: 100,
    nRange: [10, 1000],
    nGrid: [10, 50, 200, 1000],
    MDefault: 2000,
    MRange: [500, 10000],
    description:
      'The cleanest visual: at n = 10 the histogram is lumpy and over-dispersed; by n = 200 it closely hugs χ²₁; by n = 1000 moment match is within 2%.',
  },
  {
    id: 'normal-mean-known-sigma',
    name: 'Normal mean (σ=1) H₀: μ = 0',
    family: 'normal-mean-known-sigma',
    theta0: 0,
    sigma: 1,
    nDefault: 50,
    nRange: [5, 1000],
    nGrid: [10, 50, 200, 1000],
    MDefault: 2000,
    MRange: [500, 10000],
    description:
      'Exact-law special case: −2 log Λₙ = n(x̄−μ₀)²/σ² is distributed as χ²₁ exactly for every n ≥ 1. The histogram hugs the density at every slider position.',
  },
  {
    id: 'normal-mean',
    name: 'Normal mean (σ unknown) H₀: μ = 0',
    family: 'normal-mean',
    theta0: 0,
    nDefault: 30,
    nRange: [5, 500],
    nGrid: [10, 50, 200, 1000],
    MDefault: 2000,
    MRange: [500, 10000],
    description:
      'The σ-unknown case: −2 log Λₙ = n log(1 + n(x̄−μ₀)²/(n·σ̂²_MLE)). Convergence to χ²₁ is asymptotic; at small n the tails are heavier than the limit.',
  },
  {
    id: 'poisson',
    name: 'Poisson rate H₀: λ = 2',
    family: 'poisson',
    theta0: 2,
    nDefault: 50,
    nRange: [5, 1000],
    nGrid: [10, 50, 200, 1000],
    MDefault: 2000,
    MRange: [500, 10000],
    description:
      'Second discrete-family example. At small n the Poisson discreteness is visible as gaps in the −2 log Λ histogram; by n = 100 the gaps have filled in.',
  },
];

// ── Component 4 (optional): ReparamInvariance ──────────────────────────────

export interface ReparamPreset {
  id: string;
  name: string;
  p0: number;
  phatDefault: number;
  phatRange: [number, number];
  nDefault: number;
  nRange: [number, number];
  link: 'logit' | 'probit' | 'log';
  description: string;
}

export const reparamPresets: ReparamPreset[] = [
  {
    id: 'bernoulli-logit',
    name: 'Bernoulli (logit) — p₀ = 0.5, p̂ = 0.3, n = 50',
    p0: 0.5,
    phatDefault: 0.3,
    phatRange: [0.01, 0.99],
    nDefault: 50,
    nRange: [10, 500],
    link: 'logit',
    description:
      "Logit η = log(p/(1−p)) is the canonical GLM link. Wald's p-values differ between the p and η parameterizations; LRT is invariant. §18.8 Thm 6.",
  },
  {
    id: 'bernoulli-probit',
    name: 'Bernoulli (probit) — p₀ = 0.5, p̂ = 0.3, n = 50',
    p0: 0.5,
    phatDefault: 0.3,
    phatRange: [0.01, 0.99],
    nDefault: 50,
    nRange: [10, 500],
    link: 'probit',
    description:
      'Probit η = Φ⁻¹(p). Same Wald-non-invariance phenomenon. LRT is still invariant because it depends only on the attained likelihood.',
  },
  {
    id: 'bernoulli-log',
    name: 'Bernoulli (log) — p₀ = 0.1, p̂ = 0.05, n = 100 (rare-event regime)',
    p0: 0.1,
    phatDefault: 0.05,
    phatRange: [0.001, 0.5],
    nDefault: 100,
    nRange: [20, 1000],
    link: 'log',
    description:
      'Rare-event regime where Wald parameterization-dependence bites hardest: log-odds Wald vs raw-proportion Wald can disagree on significance.',
  },
];

// ── Foundational references cited inline in §18.x ──────────────────────────

/**
 * Matches the Topic 17 pattern (`topic17References`): a frozen object keyed
 * by citation short-id, with `cite` (Chicago 17th short form) and a verified
 * `url`. Rendered inline in the MDX via text templates; the authoritative
 * full-form Chicago entries live in the frontmatter `references:` block.
 */
export const topic18References = {
  neymanPearson1933: {
    key: 'NEY1933',
    cite:
      'Neyman & Pearson (1933), On the Problem of the Most Efficient Tests of Statistical Hypotheses, Phil. Trans. R. Soc. A 231: 289–337',
    url: 'https://doi.org/10.1098/rsta.1933.0009',
  },
  karlinRubin1956: {
    key: 'KAR1956',
    cite:
      'Karlin & Rubin (1956), The Theory of Decision Procedures for Distributions with Monotone Likelihood Ratio, Ann. Math. Statist. 27(2): 272–299',
    url: 'https://doi.org/10.1214/aoms/1177728259',
  },
  karlin1957: {
    key: 'KAR1957',
    cite: 'Karlin (1957), Pólya Type Distributions, II, Ann. Math. Statist. 28(2): 281–308',
    url: 'https://doi.org/10.1214/aoms/1177706960',
  },
  wilks1938: {
    key: 'WIL1938',
    cite:
      'Wilks (1938), The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses, Ann. Math. Statist. 9(1): 60–62',
    url: 'https://projecteuclid.org/euclid.aoms/1177732360',
  },
  rao1948: {
    key: 'RAO1948',
    cite:
      'Rao (1948), Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation, Proc. Camb. Phil. Soc. 44(1): 50–57',
    url: 'https://doi.org/10.1017/S0305004100023987',
  },
  ferguson1967: {
    key: 'FER1967',
    cite: 'Ferguson (1967), Mathematical Statistics: A Decision Theoretic Approach, Academic Press',
    url: 'https://www.worldcat.org/title/mathematical-statistics-a-decision-theoretic-approach/oclc/490902',
  },
  vanderVaart1998: {
    key: 'VAN1998',
    cite: 'van der Vaart (1998), Asymptotic Statistics, Cambridge University Press',
    url: 'https://doi.org/10.1017/CBO9780511802256',
  },
  buse1982: {
    key: 'BUS1982',
    cite:
      'Buse (1982), The Likelihood Ratio, Wald, and Lagrange Multiplier Tests: An Expository Note, Amer. Statist. 36(3a): 153–157',
    url: 'https://doi.org/10.1080/00031305.1982.10482817',
  },
  lehmannRomano2005: {
    key: 'LEH2005',
    cite: 'Lehmann & Romano (2005), Testing Statistical Hypotheses, 3rd ed., Springer',
    url: 'https://doi.org/10.1007/0-387-27605-X',
  },
  casellaBerger2002: {
    key: 'CAS2002',
    cite: 'Casella & Berger (2002), Statistical Inference, 2nd ed., Duxbury',
    url: 'https://www.cengage.com/c/statistical-inference-2e-casella/',
  },
} as const;
