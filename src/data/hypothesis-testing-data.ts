/**
 * hypothesis-testing-data.ts — Presets for Topic 17's five interactive
 * components. Matches the shape defined in the handoff brief §7.
 *
 * Imports nothing from the shared viz modules; this is a pure data file
 * that components read and interpret.
 */

export type TestSide = 'left' | 'right' | 'two';

// ── NullAlternativeSimulator ────────────────────────────────────────────────

export type NullAltFamily =
  | 'normal-mean-known-sigma'
  | 'two-proportion'
  | 'exponential-mean';

export interface NullAlternativePreset {
  id: string;
  name: string;
  family: NullAltFamily;
  nullTheta: number;
  altThetaDefault: number;
  altThetaRange: [number, number];
  altThetaStep: number;
  nDefault: number;
  nRange: [number, number];
  sigmaKnown?: number;
  baselineP?: number;
  alphaDefault: number;
  sides: TestSide[];
  description: string;
}

export const nullAlternativePresets: NullAlternativePreset[] = [
  {
    id: 'normal-z',
    name: 'Normal mean z-test (σ known)',
    family: 'normal-mean-known-sigma',
    nullTheta: 0,
    altThetaDefault: 0.5,
    altThetaRange: [-2, 2],
    altThetaStep: 0.05,
    nDefault: 30,
    nRange: [5, 500],
    sigmaKnown: 1,
    alphaDefault: 0.05,
    sides: ['right', 'two'],
    description:
      'The canonical picture: two Normal sampling distributions of the z-statistic, separated by the standardized effect size δ = √n(μ − μ₀)/σ. The rejection region is a tail under H₀; power is the complementary tail under H_A.',
  },
  {
    id: 'two-proportion',
    name: 'Two-sample proportion z-test (A/B testing)',
    family: 'two-proportion',
    nullTheta: 0, // p_B − p_A under H₀
    altThetaDefault: 0.02,
    altThetaRange: [-0.1, 0.1],
    altThetaStep: 0.005,
    nDefault: 1000,
    nRange: [100, 10000],
    baselineP: 0.1,
    alphaDefault: 0.05,
    sides: ['right', 'two'],
    description:
      'The ML-native application. H₀: p_B = p_A. Sliders control the true lift p_B − p_A and per-arm sample size. This is the diagram every A/B-testing platform is drawing behind the scenes.',
  },
  {
    id: 'exponential-mean',
    name: 'Exponential mean (one-sample)',
    family: 'exponential-mean',
    nullTheta: 1,
    altThetaDefault: 1.3,
    altThetaRange: [0.2, 3],
    altThetaStep: 0.05,
    nDefault: 50,
    nRange: [10, 500],
    alphaDefault: 0.05,
    sides: ['right', 'two'],
    description:
      'Waiting-time data. The sum Σ X_i under Exp(λ) is Gamma(n, λ), so the z-approximation for √n(X̄ − 1/λ)/(1/λ) is reasonable at n ≥ 30; this scenario lets the reader see the mild non-Normality of the null distribution for smaller n.',
  },
];

// ── PowerCurveExplorer ─────────────────────────────────────────────────────

export type PowerCurveScenario =
  | 'z-one-sample'
  | 't-one-sample'
  | 'z-two-proportion'
  | 'binomial-exact';

export interface PowerCurvePreset {
  id: string;
  name: string;
  scenario: PowerCurveScenario;
  defaults: {
    sigma?: number;
    alpha: number;
    n: number;
    baselineP?: number;
    lift?: number;
    p0?: number;
    pTrue?: number;
  };
  closedForm: string;
  sampleSizeFormula: string;
}

export const powerCurvePresets: PowerCurvePreset[] = [
  {
    id: 'z-one-sample',
    name: 'Normal mean z-test (σ known)',
    scenario: 'z-one-sample',
    defaults: { sigma: 1, alpha: 0.05, n: 30 },
    closedForm: 'β(μ) = 1 − Φ(z_α − √n(μ − μ₀)/σ)',
    sampleSizeFormula: 'n = (z_α + z_β)² σ² / δ²',
  },
  {
    id: 't-one-sample',
    name: 'Normal mean t-test (σ unknown)',
    scenario: 't-one-sample',
    defaults: { sigma: 1, alpha: 0.05, n: 30 },
    closedForm: 'β(μ) via noncentral t (JKB series)',
    sampleSizeFormula: 'Numerical inversion of noncentral t CDF',
  },
  {
    id: 'z-two-proportion',
    name: 'Two-sample proportion z-test (A/B testing)',
    scenario: 'z-two-proportion',
    defaults: { baselineP: 0.1, lift: 0.02, alpha: 0.05, n: 1000 },
    closedForm: 'β via pooled-SE under H₀, unpooled under H_A',
    sampleSizeFormula:
      'n_per_arm = (z_α √(2p̄q̄) + z_β √(p_A q_A + p_B q_B))² / δ²',
  },
  {
    id: 'binomial-exact',
    name: 'Binomial exact test (one-sided)',
    scenario: 'binomial-exact',
    defaults: { n: 20, p0: 0.5, pTrue: 0.7, alpha: 0.05 },
    closedForm: 'β = Σ_{k ≥ boundary} Binomial(k; n, pTrue)',
    sampleSizeFormula:
      'Numerical search: smallest n such that exact power ≥ 1 − β at pTrue',
  },
];

// ── PValueDemonstrator ─────────────────────────────────────────────────────

export type PValueFamily = 'normal-z' | 'normal-t' | 'two-proportion' | 'binomial-exact';

export interface PValueScenario {
  id: string;
  name: string;
  family: PValueFamily;
  nullParams: {
    mu?: number;
    sigma?: number;
    p?: number;
    pA?: number;
    pB?: number;
    n?: number;
    nA?: number;
    nB?: number;
  };
  sides: TestSide[];
  description: string;
}

export const pValueScenarios: PValueScenario[] = [
  {
    id: 'normal-z',
    name: 'One-sample z-test (Normal, σ = 1)',
    family: 'normal-z',
    nullParams: { mu: 0, sigma: 1 },
    sides: ['right', 'two'],
    description:
      'Under continuous H₀, the p-value is Uniform(0, 1) — Thm 3. The MC-uniformity mode verifies this empirically.',
  },
  {
    id: 'normal-t',
    name: 'One-sample t-test (Normal, σ unknown)',
    family: 'normal-t',
    nullParams: { mu: 0, sigma: 1 },
    sides: ['right', 'two'],
    description:
      'The t-null is exact via Basu (§16.9 → Thm 5). Under H_A (shift in μ), the t-statistic follows a noncentral t distribution; the MC p-value histogram concentrates toward zero.',
  },
  {
    id: 'two-proportion',
    name: 'Two-sample proportion z-test',
    family: 'two-proportion',
    nullParams: { pA: 0.1, pB: 0.1, nA: 1000, nB: 1000 },
    sides: ['right', 'two'],
    description:
      'A/B test. Under H₀ the pooled-SE z-statistic is asymptotically N(0, 1); the uniformity is also asymptotic.',
  },
  {
    id: 'binomial-exact',
    name: 'Binomial exact (n = 20, p₀ = 0.5)',
    family: 'binomial-exact',
    nullParams: { n: 20, p: 0.5 },
    sides: ['right', 'two'],
    description:
      'Discrete null. Under H₀ the p-value distribution is *not* Uniform(0, 1) — it concentrates on a finite set of values (the discrete p-value set). The MC-uniformity mode shows this discrete structure.',
  },
];

// ── TTestBasuFoundation ────────────────────────────────────────────────────

export interface TTestBasuPreset {
  id: string;
  name: string;
  defaults: { n: number; muTrue: number; mu0: number; sigma: number; M: number };
  description: string;
  basuCallback: string; // URL fragment into sufficient-statistics
}

export const tTestBasuPresets: TTestBasuPreset[] = [
  {
    id: 'n10-h0',
    name: 'One-sample t-test at n = 10 (H₀ true)',
    defaults: { n: 10, muTrue: 0, mu0: 0, sigma: 1, M: 5000 },
    description:
      'The foundational case. Basu gives X̄ ⊥⊥ S²; the t-ratio has t₉ distribution exactly. Heavier tails than Normal visible at small n.',
    basuCallback: '/topics/sufficient-statistics#section-16-9',
  },
  {
    id: 'n30-h0',
    name: 'One-sample t-test at n = 30 (H₀ true)',
    defaults: { n: 30, muTrue: 0, mu0: 0, sigma: 1, M: 5000 },
    description:
      't₂₉ is nearly Normal; the t → Normal asymptotics become visible directly as the heavy tails thin out.',
    basuCallback: '/topics/sufficient-statistics#section-16-9',
  },
  {
    id: 'n10-ha',
    name: 'One-sample t-test at n = 10 (H_A: μ = 0.8)',
    defaults: { n: 10, muTrue: 0.8, mu0: 0, sigma: 1, M: 5000 },
    description:
      'Under H_A the t-statistic shifts right; the sampling distribution is noncentral t. Power visible as shifted mass in the rejection region.',
    basuCallback: '/topics/sufficient-statistics#section-16-9',
  },
];

// ── WaldScoreLRTComparison ─────────────────────────────────────────────────

export type WSLFamily =
  | 'bernoulli'
  | 'normal-mean-known-sigma'
  | 'normal-mean'
  | 'poisson';

export interface WaldScoreLRTPreset {
  id: string;
  name: string;
  family: WSLFamily;
  nullTheta: number;
  defaultAltTheta: number;
  altRange: [number, number];
  sigmaKnown?: number;
  nDefault: number;
  MDefault: number;
  formulas: { wald: string; score: string; lrt: string };
  description: string;
}

export const waldScoreLRTPresets: WaldScoreLRTPreset[] = [
  {
    id: 'bernoulli',
    name: 'Bernoulli H₀: p = 0.5',
    family: 'bernoulli',
    nullTheta: 0.5,
    defaultAltTheta: 0.5,
    altRange: [0.05, 0.95],
    nDefault: 100,
    MDefault: 3000,
    formulas: {
      wald: 'W_n = n(p̂ − p₀)² / [p̂(1 − p̂)]',
      score: 'S_n = n(p̂ − p₀)² / [p₀(1 − p₀)]',
      lrt: '−2 log Λ = 2n [p̂ log(p̂/p₀) + (1 − p̂) log((1 − p̂)/(1 − p₀))]',
    },
    description:
      'Cleanest algebra: the three statistics differ only by the variance-estimate plug-in (p̂ vs p₀). Under H₀, all three converge to χ²₁; divergence under H_A shows finite-sample power differences.',
  },
  {
    id: 'normal-mean-known-sigma',
    name: 'Normal mean (σ known)',
    family: 'normal-mean-known-sigma',
    nullTheta: 0,
    defaultAltTheta: 0,
    altRange: [-2, 2],
    sigmaKnown: 1,
    nDefault: 50,
    MDefault: 3000,
    formulas: {
      wald: 'W_n = n(x̄ − μ₀)² / σ²',
      score: 'S_n = n(x̄ − μ₀)² / σ²   (= Wald)',
      lrt: '−2 log Λ = n(x̄ − μ₀)² / σ²   (= Wald)',
    },
    description:
      'All three collapse to the squared z-statistic — a useful sanity check. "Agree under H₀" is exact, not asymptotic, in this special case.',
  },
  {
    id: 'normal-mean',
    name: 'Normal mean (σ unknown)',
    family: 'normal-mean',
    nullTheta: 0,
    defaultAltTheta: 0,
    altRange: [-2, 2],
    nDefault: 30,
    MDefault: 3000,
    formulas: {
      wald: 'W_n = n(x̄ − μ₀)² / S²',
      score: 'S_n = n(x̄ − μ₀)² / σ̂_MLE²   (uses n⁻¹ Σ(X − x̄)²)',
      lrt: '−2 log Λ = n log(1 + (x̄ − μ₀)² / σ̂_MLE²)',
    },
    description:
      'Here the three diverge mildly in finite samples; Wald uses S² (unbiased), score uses σ̂²_MLE, LRT uses a log form. Asymptotic agreement as n → ∞.',
  },
  {
    id: 'poisson',
    name: 'Poisson H₀: λ = 2',
    family: 'poisson',
    nullTheta: 2,
    defaultAltTheta: 2,
    altRange: [0.5, 5],
    nDefault: 50,
    MDefault: 3000,
    formulas: {
      wald: 'W_n = n(λ̂ − λ₀)² / λ̂',
      score: 'S_n = n(λ̂ − λ₀)² / λ₀',
      lrt: '−2 log Λ = 2n [λ̂ log(λ̂/λ₀) − (λ̂ − λ₀)]',
    },
    description:
      'The other classic exp-family case. Same agreement/divergence structure as Bernoulli.',
  },
];

// ── Foundational references used inline in §17.x ───────────────────────────

export const topic17References = {
  fisher1922: {
    key: 'FIS1922',
    cite: 'Fisher (1922), On the Mathematical Foundations of Theoretical Statistics, Phil. Trans. R. Soc. A 222: 309–368',
    url: 'https://royalsocietypublishing.org/doi/10.1098/rsta.1922.0009',
  },
  neymanPearson1933: {
    key: 'NEY1933',
    cite: 'Neyman & Pearson (1933), On the Problem of the Most Efficient Tests of Statistical Hypotheses, Phil. Trans. R. Soc. A 231: 289–337',
    url: 'https://doi.org/10.1098/rsta.1933.0009',
  },
  student1908: {
    key: 'STU1908',
    cite: "'Student' (Gosset) (1908), The Probable Error of a Mean, Biometrika 6(1): 1–25",
    url: 'https://www.jstor.org/stable/2331554',
  },
  pearson1900: {
    key: 'PEA1900',
    cite: 'Pearson (1900), On the Criterion that a Given System of Deviations from the Probable Is Such that It Can Be Reasonably Supposed to Have Arisen from Random Sampling, Phil. Mag. (5th ser.) 50: 157–175',
    url: 'https://doi.org/10.1080/14786440009463897',
  },
  wald1943: {
    key: 'WAL1943',
    cite: 'Wald (1943), Tests of Statistical Hypotheses Concerning Several Parameters When the Number of Observations is Large, Trans. AMS 54(3): 426–482',
    url: 'https://doi.org/10.1090/S0002-9947-1943-0012401-3',
  },
  rao1948: {
    key: 'RAO1948',
    cite: 'Rao (1948), Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation, Proc. Camb. Phil. Soc. 44(1): 50–57',
    url: 'https://doi.org/10.1017/S0305004100023987',
  },
  wilks1938: {
    key: 'WIL1938',
    cite: 'Wilks (1938), The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses, Ann. Math. Statist. 9(1): 60–62',
    url: 'https://projecteuclid.org/euclid.aoms/1177732360',
  },
  wassersteinLazar2016: {
    key: 'WAS2016',
    cite: "Wasserstein & Lazar (2016), The ASA's Statement on p-Values: Context, Process, and Purpose, Amer. Statist. 70(2): 129–133",
    url: 'https://doi.org/10.1080/00031305.2016.1154108',
  },
  ioannidis2005: {
    key: 'IOA2005',
    cite: 'Ioannidis (2005), Why Most Published Research Findings Are False, PLoS Medicine 2(8): e124',
    url: 'https://doi.org/10.1371/journal.pmed.0020124',
  },
  gelmanLoken2013: {
    key: 'GEL2013',
    cite: 'Gelman & Loken (2013), The Garden of Forking Paths',
    url: 'http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf',
  },
  lehmannRomano2005: {
    key: 'LEH2005',
    cite: 'Lehmann & Romano (2005), Testing Statistical Hypotheses, 3rd ed., Springer',
    url: 'https://link.springer.com/book/10.1007/0-387-27605-X',
  },
  casellaBerger2002: {
    key: 'CAS2002',
    cite: 'Casella & Berger (2002), Statistical Inference, 2nd ed., Duxbury',
    url: 'https://www.cengage.com/c/statistical-inference-2e-casella/',
  },
} as const;
