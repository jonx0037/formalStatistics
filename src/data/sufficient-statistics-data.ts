/**
 * sufficient-statistics-data.ts — Topic 16 presets for the five interactive
 * components (FactorizationExplorer, RaoBlackwellImprover, CompletenessProbe,
 * BasuIndependence, UMVUEComparator) plus the `topic16References` table used
 * for inline citations.
 *
 * Track 4 closing topic — Sufficient Statistics & Rao-Blackwell.
 */
import type { SufficientFamily, TripleComparisonFamily } from '../components/viz/shared/estimation';

// ── §16.3 FactorizationExplorer presets ─────────────────────────────────────

export interface FactorizationPreset {
  name: string;
  family: SufficientFamily;
  defaultParams: Record<string, number>;
  sufficientStatistic: string; // LaTeX without delimiters
  isExpFamily: boolean;
  supportDependsOnTheta?: boolean;
  description: string;
}

export const factorizationPresets: FactorizationPreset[] = [
  {
    name: 'Normal(μ, σ² known)',
    family: 'normal-mu',
    defaultParams: { mu: 0, sigma2: 1 },
    sufficientStatistic: 'T(x) = \\sum_{i=1}^n x_i',
    isExpFamily: true,
    description:
      'The canonical one-parameter exponential family. T is one-dimensional, matching the parameter dimension. The Fisher–Neyman factorization is direct: $g(T;\\mu) \\cdot h(x)$ with $g$ depending on the data only through $\\sum x_i$.',
  },
  {
    name: 'Normal(μ, σ²) — both unknown',
    family: 'normal-mu-sigma',
    defaultParams: { mu: 0, sigma2: 1 },
    sufficientStatistic: 'T(x) = \\bigl(\\sum x_i,\\; \\sum x_i^2\\bigr)',
    isExpFamily: true,
    description:
      'Two-parameter exponential family; T is 2-dimensional, matching the parameter dimension.',
  },
  {
    name: 'Bernoulli(p)',
    family: 'bernoulli',
    defaultParams: { p: 0.3 },
    sufficientStatistic: 'T(x) = \\sum_{i=1}^n x_i',
    isExpFamily: true,
    description:
      'Single-parameter exponential family; T is the number of successes. T | n ~ Binomial(n, p).',
  },
  {
    name: 'Poisson(λ)',
    family: 'poisson',
    defaultParams: { lambda: 2 },
    sufficientStatistic: 'T(x) = \\sum_{i=1}^n x_i',
    isExpFamily: true,
    description:
      'Single-parameter exponential family; T is the total count. T ~ Poisson(nλ).',
  },
  {
    name: 'Exponential(λ)',
    family: 'exponential',
    defaultParams: { lambda: 1 },
    sufficientStatistic: 'T(x) = \\sum_{i=1}^n x_i',
    isExpFamily: true,
    description:
      'Single-parameter exponential family in rate λ. T ~ Gamma(n, λ); the UMVUE of λ is $(n-1)/T$.',
  },
  {
    name: 'Gamma(α known, β = scale)',
    family: 'gamma-scale',
    defaultParams: { alpha: 3, beta: 2 },
    sufficientStatistic: 'T(x) = \\sum_{i=1}^n x_i',
    isExpFamily: true,
    description:
      'Single-parameter exponential family in β when α is known. UMVUE of β is $(n\\alpha - 1)/T$ — fulfills the §16.11 / `method-of-moments.mdx:1062` comparison.',
  },
  {
    name: 'Uniform(0, θ) — NOT an exponential family',
    family: 'uniform-upper',
    defaultParams: { theta: 2 },
    sufficientStatistic: 'T(x) = x_{(n)} = \\max_i x_i',
    isExpFamily: false,
    supportDependsOnTheta: true,
    description:
      'Sufficient statistic exists (the sample maximum), but the support [0, θ] depends on θ — violates PKD regularity (A), and is the paradigm non-exponential-family case of data reduction (§16.10 Example 23).',
  },
];

// ── §16.5 RaoBlackwellImprover presets ──────────────────────────────────────

export interface RaoBlackwellPreset {
  name: string;
  family: SufficientFamily;
  closedForm:
    | 'bernoulli-p'
    | 'bernoulli-variance'
    | 'poisson-zero-prob'
    | 'normal-tail-prob';
  defaultParams: Record<string, number>;
  crudeFormula: string;
  rbFormula: string;
  expectedVarRatio: number;
  description: string;
}

export const raoBlackwellPresets: RaoBlackwellPreset[] = [
  {
    name: "Bernoulli p — crude: X₁  vs  RB'd: X̄",
    family: 'bernoulli',
    closedForm: 'bernoulli-p',
    defaultParams: { p: 0.3, n: 20 },
    crudeFormula: '\\hat{p}_0 = X_1',
    rbFormula: '\\tilde{p} = \\bar{X}',
    expectedVarRatio: 20,
    description:
      'Most visceral demo: $X_1 \\in \\{0,1\\}$ (huge variance), $\\bar X$ concentrates around $p$. Variance ratio $\\mathrm{Var}(X_1)/\\mathrm{Var}(\\bar X) = n$.',
  },
  {
    name: "Poisson zero-probability — crude: 𝟙{X₁ = 0}  vs  RB'd: (1 − 1/n)^{ΣX}",
    family: 'poisson',
    closedForm: 'poisson-zero-prob',
    defaultParams: { lambda: 2, n: 10 },
    crudeFormula: '\\hat{q}_0 = \\mathbf{1}\\{X_1 = 0\\}',
    rbFormula: '\\tilde{q} = (1 - 1/n)^{\\sum X_i}',
    expectedVarRatio: 10,
    description:
      'The textbook UMVUE-of-$e^{-\\lambda}$ demonstration; RB produces a smooth estimator from a 0/1 indicator.',
  },
  {
    name: "Normal tail — crude: 𝟙{X₁ > c}  vs  RB'd: regularized-Beta tail",
    family: 'normal-mu-sigma',
    closedForm: 'normal-tail-prob',
    defaultParams: { mu: 0, sigma2: 1, c: 1, n: 30 },
    crudeFormula: '\\hat{p}_c = \\mathbf{1}\\{X_1 > c\\}',
    rbFormula: '\\tilde{p}_c = I_y\\!\\left(\\tfrac{n-2}{2}, \\tfrac{n-2}{2}\\right),\\; y = \\tfrac{1}{2}\\!\\left(1 - \\tfrac{(c-\\bar X)/S}{\\sqrt{(n-2+(c-\\bar X)^2/S^2)/n}}\\right)',
    expectedVarRatio: 5,
    description:
      'Lehmann-Casella (1998) §2.4 worked example; RB produces a Beta-CDF expression from a 0/1 indicator.',
  },
  {
    name: "Bernoulli variance p(1−p) — crude: X₁(1−X₁) ≡ 0  vs  RB'd: X̄(n−ΣX)/(n−1)",
    family: 'bernoulli',
    closedForm: 'bernoulli-variance',
    defaultParams: { p: 0.3, n: 30 },
    crudeFormula: '\\hat{v}_0 = X_1(1 - X_1) \\equiv 0',
    rbFormula: '\\tilde{v} = \\dfrac{\\bar{X}(n - \\sum X_i)}{n - 1}',
    expectedVarRatio: 15,
    description:
      'Bernoulli-variance UMVUE — crude is identically 0 ($X_1 \\in \\{0,1\\}$ ⇒ $X_1(1-X_1) = 0$ always); RB rescues a completely uninformative crude estimator.',
  },
];

// ── §16.6 CompletenessProbe presets ─────────────────────────────────────────

export interface CompletenessProbePreset {
  name: string;
  /** Family identifier used by `completenessProbe`. The Uniform(θ, θ+1)
   *  incomplete case lives under the separate `'uniform-shift'` identifier
   *  (its witness is computed by `incompletenessWitnessUniform`). */
  family: SufficientFamily | 'uniform-shift';
  isComplete: boolean;
  thetaRange: [number, number];
  /** For the Uniform(θ, θ+1) preset, the witness is the centered range
   *  $g(T) = R - (n-1)/(n+1)$, $R = X_{(n)} - X_{(1)}$.  See §16.6 Example 13. */
  witnessFn?: string;
  description: string;
}

export const completenessProbePresets: CompletenessProbePreset[] = [
  {
    name: 'Bernoulli(p) — complete',
    family: 'bernoulli',
    isComplete: true,
    thetaRange: [0.05, 0.95],
    description:
      'Finite-support discrete family; completeness is a finite linear-algebra argument. Only the constant-zero function gives the flat $\\mathbb{E}_p[g(T)] \\equiv 0$ curve.',
  },
  {
    name: 'Poisson(λ) — complete',
    family: 'poisson',
    isComplete: true,
    thetaRange: [0.5, 5],
    description:
      'Complete via analytic dependence on $\\lambda$ of the power series $\\sum_k e^{-n\\lambda} (n\\lambda)^k g(k) / k!$.',
  },
  {
    name: 'Normal(μ, σ² = 1) — complete',
    family: 'normal-mu',
    isComplete: true,
    thetaRange: [-2, 2],
    description:
      'Complete via Laplace-transform uniqueness in $\\mu$: $\\mathbb{E}_\\mu[g(T)] = 0$ for all $\\mu$ ⇒ the moment generating function of $g(T)$ vanishes ⇒ $g(T) = 0$ a.s.',
  },
  {
    name: 'Uniform(θ, θ+1) — NOT complete',
    family: 'uniform-shift',
    isComplete: false,
    thetaRange: [-1, 2],
    witnessFn: 'g(T) = R - \\dfrac{n-1}{n+1},\\quad R = X_{(n)} - X_{(1)}',
    description:
      'The canonical incomplete family. The minimal sufficient statistic $T = (X_{(1)}, X_{(n)})$ is 2-dimensional but the parameter is 1-dimensional — an immediate signal of incompleteness. The range $R = X_{(n)} - X_{(1)}$ is ancillary (its distribution is free of $\\theta$, since $\\theta$ is purely a location parameter), so the centered range $g(T) = R - (n-1)/(n+1)$ has $\\mathbb{E}_\\theta[g(T)] = 0$ for every $\\theta$ but is not identically zero — the definition of incompleteness.',
  },
];

// ── §16.9 BasuIndependence presets ──────────────────────────────────────────

export interface BasuPreset {
  name: string;
  family: 'normal' | 'exponential-shift';
  defaultParams: Record<string, number>;
  basuHolds: boolean;
  statisticA: string;
  statisticB: string;
  description: string;
}

export const basuPresets: BasuPreset[] = [
  {
    name: 'Normal(μ, σ² = 1): X̄ complete sufficient for μ; S² ancillary for μ',
    family: 'normal',
    defaultParams: { mu: 0, sigma2: 1 },
    basuHolds: true,
    statisticA: '\\bar{X}',
    statisticB: 'S^2',
    description:
      "The t-distribution foundation. Basu's theorem gives $\\bar X \\perp\\!\\!\\!\\perp S^2$, which makes $t = \\sqrt n(\\bar X - \\mu)/S$ have a well-defined distribution (Student's t with $n-1$ degrees of freedom).",
  },
  {
    name: 'Exponential shift X = Exp(1) + μ: X̄ NOT ancillary for μ',
    family: 'exponential-shift',
    defaultParams: { mu: 0 },
    basuHolds: false,
    statisticA: 'X_{(1)}',
    statisticB: '\\bar{X}',
    description:
      "Contrast case. $X_{(1)}$ is complete sufficient for $\\mu$, but $\\bar X$ is NOT ancillary for $\\mu$ — its mean shifts with $\\mu$ via $\\mathbb{E}_\\mu[\\bar X] = \\mu + 1$. Basu's hypotheses therefore fail; expected positive correlation $\\rho \\approx 1/\\sqrt n$. (The plausible-looking $\\bar X - X_{(1)}$ would actually BE ancillary by exponential memorylessness — so it is the wrong contrast.)",
  },
];

// ── §16.11 UMVUEComparator presets — the closing component for Track 4 ─────

export interface UMVUEComparatorPreset {
  name: string;
  family: TripleComparisonFamily;
  trueParams: Record<string, number>;
  defaultN: number;
  umvueFormula: string;
  mleFormula: string;
  momFormula: string;
  description: string;
}

export const umvueComparatorPresets: UMVUEComparatorPreset[] = [
  {
    name: 'Normal σ² (μ unknown) — UMVUE ≠ MLE = MoM',
    family: 'normal-variance-unknown-mu',
    trueParams: { mu: 0, sigma2: 4 },
    defaultN: 30,
    umvueFormula: '\\hat{\\sigma}^2_{\\text{UMVUE}} = \\dfrac{1}{n-1}\\sum_i(X_i - \\bar X)^2 = S^2_{n-1}',
    mleFormula: '\\hat{\\sigma}^2_{\\text{MLE}} = \\dfrac{1}{n}\\sum_i(X_i - \\bar X)^2 = S^2_n',
    momFormula: '\\hat{\\sigma}^2_{\\text{MoM}} = S^2_n \\quad\\text{(same as MLE)}',
    description:
      "The featured 'all three differ' case. UMVUE is unbiased; MLE = MoM share bias $-\\sigma^2/n$. For $n = 30$ and $\\sigma^2 = 4$, MLE bias $\\approx -0.13$.",
  },
  {
    name: 'Gamma scale β (α = 3 known) — UMVUE ≠ MLE = MoM (fulfills §1062)',
    family: 'gamma-scale-with-known-alpha',
    trueParams: { alpha: 3, beta: 2 },
    defaultN: 50,
    umvueFormula: '\\hat{\\beta}_{\\text{UMVUE}} = \\dfrac{n\\alpha - 1}{\\sum_i X_i}',
    mleFormula: '\\hat{\\beta}_{\\text{MLE}} = \\dfrac{\\alpha}{\\bar X}',
    momFormula: '\\hat{\\beta}_{\\text{MoM}} = \\dfrac{\\alpha}{\\bar X} \\quad\\text{(same as MLE)}',
    description:
      "Fulfills the `method-of-moments.mdx:1062` promise. With $\\alpha$ known, the family is exp family in $\\beta$; $T = \\sum X_i$ is complete sufficient and the UMVUE is $(n\\alpha-1)/T$.",
  },
  {
    name: 'Exponential rate λ — UMVUE ≠ MLE = MoM',
    family: 'exponential-rate',
    trueParams: { lambda: 2 },
    defaultN: 30,
    umvueFormula: '\\hat{\\lambda}_{\\text{UMVUE}} = \\dfrac{n-1}{\\sum_i X_i}',
    mleFormula: '\\hat{\\lambda}_{\\text{MLE}} = \\dfrac{n}{\\sum_i X_i} = \\dfrac{1}{\\bar X}',
    momFormula: '\\hat{\\lambda}_{\\text{MoM}} = \\dfrac{1}{\\bar X} \\quad\\text{(same as MLE)}',
    description:
      "The cleanest UMVUE ≠ MLE comparison. The bias-correction factor is $(n-1)/n$ — small for large $n$ but never zero.",
  },
  {
    name: 'Normal μ (σ² known) — triple coincidence',
    family: 'normal-mean-known-sigma',
    trueParams: { mu: 5, sigma2: 2 },
    defaultN: 30,
    umvueFormula: '\\hat{\\mu}_{\\text{UMVUE}} = \\bar X',
    mleFormula: '\\hat{\\mu}_{\\text{MLE}} = \\bar X',
    momFormula: '\\hat{\\mu}_{\\text{MoM}} = \\bar X',
    description:
      'Boundary case: UMVUE = MLE = MoM = $\\bar X$. All three are unbiased, efficient, and identical sample-by-sample.',
  },
  {
    name: 'Poisson λ — triple coincidence',
    family: 'poisson-rate',
    trueParams: { lambda: 3 },
    defaultN: 30,
    umvueFormula: '\\hat{\\lambda}_{\\text{UMVUE}} = \\bar X',
    mleFormula: '\\hat{\\lambda}_{\\text{MLE}} = \\bar X',
    momFormula: '\\hat{\\lambda}_{\\text{MoM}} = \\bar X',
    description:
      'Exp family in natural parameter; all three coincide at $\\bar X$. The triple-coincidence theorem (Thm 9) in its cleanest form.',
  },
];

// ── References used inline in §16.x ─────────────────────────────────────────

export interface CitationEntry {
  key: string;
  cite: string;
  url: string | null;
}

export const topic16References: Record<string, CitationEntry> = {
  fisher1922: {
    key: 'FIS1922',
    cite: "Fisher (1922), 'On the Mathematical Foundations of Theoretical Statistics', Phil. Trans. R. Soc. A 222: 309–368",
    url: 'https://royalsocietypublishing.org/doi/10.1098/rsta.1922.0009',
  },
  halmosSavage: {
    key: 'HAL1949',
    cite: "Halmos & Savage (1949), 'Application of the Radon-Nikodym Theorem to the Theory of Sufficient Statistics', AMS 20(2): 225–241",
    url: 'https://projecteuclid.org/euclid.aoms/1177730032',
  },
  rao1945: {
    key: 'RAO1945',
    cite: "Rao (1945), 'Information and the Accuracy Attainable in the Estimation of Statistical Parameters', Bull. Calcutta Math. Soc. 37: 81–91",
    url: null,
  },
  blackwell1947: {
    key: 'BLA1947',
    cite: "Blackwell (1947), 'Conditional Expectation and Unbiased Sequential Estimation', AMS 18(1): 105–110",
    url: 'https://projecteuclid.org/euclid.aoms/1177730497',
  },
  lehmannScheffe1950: {
    key: 'LEH1950',
    cite: "Lehmann & Scheffé (1950), 'Completeness, Similar Regions, and Unbiased Estimation — Part I', Sankhyā 10(4): 305–340",
    url: 'https://www.jstor.org/stable/25048038',
  },
  basu1955: {
    key: 'BAS1955',
    cite: "Basu (1955), 'On Statistics Independent of a Complete Sufficient Statistic', Sankhyā 15(4): 377–380",
    url: 'https://www.jstor.org/stable/25048259',
  },
  darmois1935: {
    key: 'DAR1935',
    cite: "Darmois (1935), 'Sur les lois de probabilité à estimation exhaustive', C. R. Acad. Sci. Paris 200: 1265–1266",
    url: null,
  },
  koopman1936: {
    key: 'KOO1936',
    cite: "Koopman (1936), 'On Distributions Admitting a Sufficient Statistic', Trans. AMS 39(3): 399–409",
    url: 'https://www.jstor.org/stable/1989758',
  },
  pitman1936: {
    key: 'PIT1936',
    cite: "Pitman (1936), 'Sufficient Statistics and Intrinsic Accuracy', Proc. Camb. Phil. Soc. 32(4): 567–579",
    url: 'https://doi.org/10.1017/S0305004100019307',
  },
  brown1986: {
    key: 'BRO1986',
    cite: 'Brown (1986), Fundamentals of Statistical Exponential Families, IMS Lecture Notes Monograph Series, Vol. 9',
    url: 'https://projecteuclid.org/euclid.lnms/1215466757',
  },
  lehmannCasella1998: {
    key: 'LEH1998',
    cite: 'Lehmann & Casella (1998), Theory of Point Estimation, 2nd ed., Springer',
    url: 'https://link.springer.com/book/10.1007/b98854',
  },
};
