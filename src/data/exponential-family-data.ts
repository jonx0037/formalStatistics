/** Canonical form metadata for each exponential family member. */
export interface ExponentialFamilyMember {
  name: string;
  distribution: string;        // key matching distributions.ts
  type: "discrete" | "continuous";
  dimension: 1 | 2;           // number of natural parameters
  hx: string;                  // h(x) as LaTeX
  eta: string;                 // η(θ) as LaTeX (or vector for dim=2)
  Tx: string;                  // T(x) as LaTeX (or vector for dim=2)
  Aeta: string;                // A(η) as LaTeX
  AprimeEta: string;           // A'(η) = E[T(X)] as LaTeX
  AdoubleprimeEta: string;     // A''(η) = Var(T(X)) as LaTeX
  naturalParameterSpace: string; // domain of η as LaTeX
  originalParams: string[];    // parameter names in original parameterization
  canonicalLink?: string;      // canonical link function for GLM (if applicable)
  conjugatePrior?: string;     // conjugate prior distribution name
  topicSlug: string;           // link to the distribution's topic page
}

export const exponentialFamilyMembers: ExponentialFamilyMember[] = [
  {
    name: "Bernoulli",
    distribution: "Bernoulli",
    type: "discrete",
    dimension: 1,
    hx: "1",
    eta: "\\log\\frac{p}{1-p}",
    Tx: "x",
    Aeta: "\\log(1 + e^\\eta)",
    AprimeEta: "\\frac{e^\\eta}{1+e^\\eta} = p",
    AdoubleprimeEta: "p(1-p)",
    naturalParameterSpace: "\\eta \\in \\mathbb{R}",
    originalParams: ["p"],
    canonicalLink: "\\text{logit}(p) = \\log\\frac{p}{1-p}",
    conjugatePrior: "Beta(\\alpha, \\beta)",
    topicSlug: "discrete-distributions",
  },
  {
    name: "Binomial (fixed n)",
    distribution: "Binomial",
    type: "discrete",
    dimension: 1,
    hx: "\\binom{n}{k}",
    eta: "\\log\\frac{p}{1-p}",
    Tx: "k",
    Aeta: "n\\log(1 + e^\\eta)",
    AprimeEta: "np",
    AdoubleprimeEta: "np(1-p)",
    naturalParameterSpace: "\\eta \\in \\mathbb{R}",
    originalParams: ["p"],
    canonicalLink: "\\text{logit}(p)",
    conjugatePrior: "Beta(\\alpha, \\beta)",
    topicSlug: "discrete-distributions",
  },
  {
    name: "Geometric",
    distribution: "Geometric",
    type: "discrete",
    dimension: 1,
    hx: "1",
    eta: "\\log(1-p)",
    Tx: "k",
    Aeta: "-\\log(1 - e^\\eta)",
    AprimeEta: "1/p",
    AdoubleprimeEta: "(1-p)/p^2",
    naturalParameterSpace: "\\eta < 0",
    originalParams: ["p"],
    topicSlug: "discrete-distributions",
  },
  {
    name: "Negative Binomial (fixed r)",
    distribution: "NegBin",
    type: "discrete",
    dimension: 1,
    hx: "\\binom{k-1}{r-1}",
    eta: "\\log(1-p)",
    Tx: "k",
    Aeta: "-r\\log(1 - e^\\eta)",
    AprimeEta: "r/p",
    AdoubleprimeEta: "r(1-p)/p^2",
    naturalParameterSpace: "\\eta < 0",
    originalParams: ["p"],
    canonicalLink: "\\log(1-p)",
    conjugatePrior: "Beta(\\alpha, \\beta) on p",
    topicSlug: "discrete-distributions",
  },
  {
    name: "Poisson",
    distribution: "Poisson",
    type: "discrete",
    dimension: 1,
    hx: "\\frac{1}{k!}",
    eta: "\\log\\lambda",
    Tx: "k",
    Aeta: "e^\\eta",
    AprimeEta: "\\lambda",
    AdoubleprimeEta: "\\lambda",
    naturalParameterSpace: "\\eta \\in \\mathbb{R}",
    originalParams: ["\\lambda"],
    canonicalLink: "\\log\\lambda",
    conjugatePrior: "Gamma(\\alpha, \\beta)",
    topicSlug: "discrete-distributions",
  },
  {
    name: "Normal",
    distribution: "Normal",
    type: "continuous",
    dimension: 2,
    hx: "\\frac{1}{\\sqrt{2\\pi}}",
    eta: "\\left(\\frac{\\mu}{\\sigma^2},\\, -\\frac{1}{2\\sigma^2}\\right)",
    Tx: "(x,\\, x^2)",
    Aeta: "\\frac{\\mu^2}{2\\sigma^2} + \\log\\sigma",
    AprimeEta: "(\\mu,\\, \\mu^2 + \\sigma^2)",
    AdoubleprimeEta: "\\text{(Fisher information matrix)}",
    naturalParameterSpace: "\\eta_1 \\in \\mathbb{R},\\, \\eta_2 < 0",
    originalParams: ["\\mu", "\\sigma^2"],
    canonicalLink: "g(\\mu) = \\mu \\;\\text{(identity)}",
    conjugatePrior: "Normal-Inverse-Gamma",
    topicSlug: "continuous-distributions",
  },
  {
    name: "Exponential",
    distribution: "Exponential",
    type: "continuous",
    dimension: 1,
    hx: "1 \\;(x \\geq 0)",
    eta: "-\\lambda",
    Tx: "x",
    Aeta: "-\\log(-\\eta)",
    AprimeEta: "1/\\lambda",
    AdoubleprimeEta: "1/\\lambda^2",
    naturalParameterSpace: "\\eta < 0",
    originalParams: ["\\lambda"],
    conjugatePrior: "Gamma(\\alpha, \\beta)",
    topicSlug: "continuous-distributions",
  },
  {
    name: "Gamma",
    distribution: "Gamma",
    type: "continuous",
    dimension: 2,
    hx: "1 \\;(x > 0)",
    eta: "(\\alpha - 1,\\, -\\beta)",
    Tx: "(\\log x,\\, x)",
    Aeta: "\\log\\Gamma(\\eta_1+1) - (\\eta_1+1)\\log(-\\eta_2)",
    AprimeEta: "(\\psi(\\alpha),\\, \\alpha/\\beta)",
    AdoubleprimeEta: "(\\psi'(\\alpha),\\, \\alpha/\\beta^2)",
    naturalParameterSpace: "\\eta_1 > -1,\\, \\eta_2 < 0",
    originalParams: ["\\alpha", "\\beta"],
    canonicalLink: "g(\\mu) = 1/\\mu \\;\\text{(inverse)}",
    conjugatePrior: "see Barndorff-Nielsen",
    topicSlug: "continuous-distributions",
  },
  {
    name: "Beta",
    distribution: "Beta",
    type: "continuous",
    dimension: 2,
    hx: "1 \\;(x \\in (0,1))",
    eta: "(\\alpha - 1,\\, \\beta - 1)",
    Tx: "(\\log x,\\, \\log(1-x))",
    Aeta: "\\log B(\\eta_1+1, \\eta_2+1)",
    AprimeEta: "(\\psi(\\alpha) - \\psi(\\alpha+\\beta),\\, \\psi(\\beta) - \\psi(\\alpha+\\beta))",
    AdoubleprimeEta: "\\text{(digamma matrix)}",
    naturalParameterSpace: "\\eta_1 > -1,\\, \\eta_2 > -1",
    originalParams: ["\\alpha", "\\beta"],
    conjugatePrior: "see \\S7",
    topicSlug: "continuous-distributions",
  },
];

/** Distributions NOT in the exponential family, with reasons. */
export const nonMembers = [
  { name: "Continuous Uniform(a,b)", reason: "Parameter-dependent support [a,b]", topicSlug: "continuous-distributions" },
  { name: "Discrete Uniform(a,b)", reason: "Parameter-dependent support {a,...,b}", topicSlug: "discrete-distributions" },
  { name: "Hypergeometric(N,K,n)", reason: "Parameter-dependent support", topicSlug: "discrete-distributions" },
  { name: "Student's t(\u03BD)", reason: "Cannot factor as \u03B7(\u03BD)\u00B7T(x)", topicSlug: "continuous-distributions" },
  { name: "F(d\u2081,d\u2082)", reason: "Cannot factor as \u03B7\u00B7T", topicSlug: "continuous-distributions" },
];

/** Conjugate prior pairs for ConjugatePriorExplorer. */
export const conjugatePairs = [
  {
    likelihood: "Bernoulli/Binomial",
    prior: "Beta(\u03B1\u2080, \u03B2\u2080)",
    posteriorRule: "Beta(\u03B1\u2080 + k, \u03B2\u2080 + n \u2212 k)",
    pseudoData: "\u03B1\u2080 \u2212 1 pseudo-successes, \u03B2\u2080 \u2212 1 pseudo-failures",
  },
  {
    likelihood: "Poisson",
    prior: "Gamma(\u03B1\u2080, \u03B2\u2080)",
    posteriorRule: "Gamma(\u03B1\u2080 + \u03A3k\u1D62, \u03B2\u2080 + n)",
    pseudoData: "\u03B1\u2080 \u2212 1 pseudo-total-count, \u03B2\u2080 pseudo-observations",
  },
  {
    likelihood: "Normal (known \u03C3\u00B2)",
    prior: "Normal(\u03BC\u2080, \u03C3\u2080\u00B2)",
    posteriorRule: "Normal(precision-weighted mean, 1/(\u03C3\u2080\u207B\u00B2 + n\u03C3\u207B\u00B2))",
    pseudoData: "Prior is worth \u03C3\u00B2/\u03C3\u2080\u00B2 pseudo-observations",
  },
];

/** Parameter slider configurations for each distribution in ExponentialFamilyExplorer. */
export interface ParamConfig {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export const parameterConfigs: Record<string, ParamConfig[]> = {
  Bernoulli: [
    { name: "p", label: "p", min: 0.01, max: 0.99, step: 0.01, default: 0.5 },
  ],
  Binomial: [
    { name: "p", label: "p", min: 0.01, max: 0.99, step: 0.01, default: 0.5 },
    { name: "n", label: "n", min: 1, max: 30, step: 1, default: 10 },
  ],
  Geometric: [
    { name: "p", label: "p", min: 0.01, max: 0.99, step: 0.01, default: 0.3 },
  ],
  NegBin: [
    { name: "p", label: "p", min: 0.01, max: 0.99, step: 0.01, default: 0.5 },
    { name: "r", label: "r", min: 1, max: 20, step: 1, default: 5 },
  ],
  Poisson: [
    { name: "lambda", label: "\u03BB", min: 0.1, max: 20, step: 0.1, default: 5 },
  ],
  Normal: [
    { name: "mu", label: "\u03BC", min: -10, max: 10, step: 0.1, default: 0 },
    { name: "sigma2", label: "\u03C3\u00B2", min: 0.1, max: 10, step: 0.1, default: 1 },
  ],
  Exponential: [
    { name: "lambda", label: "\u03BB", min: 0.1, max: 10, step: 0.1, default: 1 },
  ],
  Gamma: [
    { name: "alpha", label: "\u03B1", min: 0.1, max: 20, step: 0.1, default: 3 },
    { name: "beta", label: "\u03B2", min: 0.1, max: 10, step: 0.1, default: 1 },
  ],
  Beta: [
    { name: "alpha", label: "\u03B1", min: 0.1, max: 20, step: 0.1, default: 2 },
    { name: "beta", label: "\u03B2", min: 0.1, max: 20, step: 0.1, default: 5 },
  ],
};

/**
 * Numerical log-partition functions for one-parameter exponential family members.
 * Used by LogPartitionExplorer and ExponentialFamilyExplorer readout panels.
 */
export const logPartitionFunctions: Record<string, {
  A: (eta: number) => number;
  Aprime: (eta: number) => number;
  Adoubleprime: (eta: number) => number;
  etaDomain: [number, number];
  etaFromParams: (params: Record<string, number>) => number;
  paramsFromEta: (eta: number) => Record<string, number>;
}> = {
  Bernoulli: {
    A: (eta) => Math.log(1 + Math.exp(eta)),
    Aprime: (eta) => 1 / (1 + Math.exp(-eta)),
    Adoubleprime: (eta) => {
      const s = 1 / (1 + Math.exp(-eta));
      return s * (1 - s);
    },
    etaDomain: [-6, 6],
    etaFromParams: (p) => Math.log(p.p / (1 - p.p)),
    paramsFromEta: (eta) => ({ p: 1 / (1 + Math.exp(-eta)) }),
  },
  Geometric: {
    A: (eta) => -Math.log(1 - Math.exp(eta)),
    Aprime: (eta) => 1 / (1 - Math.exp(eta)),
    Adoubleprime: (eta) => Math.exp(eta) / ((1 - Math.exp(eta)) ** 2),
    etaDomain: [-6, -0.01],
    etaFromParams: (p) => Math.log(1 - p.p),
    paramsFromEta: (eta) => ({ p: 1 - Math.exp(eta) }),
  },
  Poisson: {
    A: (eta) => Math.exp(eta),
    Aprime: (eta) => Math.exp(eta),
    Adoubleprime: (eta) => Math.exp(eta),
    etaDomain: [-2, 4],
    etaFromParams: (p) => Math.log(p.lambda),
    paramsFromEta: (eta) => ({ lambda: Math.exp(eta) }),
  },
  Exponential: {
    A: (eta) => -Math.log(-eta),
    Aprime: (eta) => -1 / eta,
    Adoubleprime: (eta) => 1 / (eta * eta),
    etaDomain: [-10, -0.05],
    etaFromParams: (p) => -p.lambda,
    paramsFromEta: (eta) => ({ lambda: -eta }),
  },
};

/** Derivation step data for CanonicalFormConverter. */
export interface DerivationStep {
  description: string;
  latex: string;
  highlight?: "h" | "eta" | "T" | "A";
}

export const derivationSteps: Record<string, DerivationStep[]> = {
  Bernoulli: [
    { description: "Start with the Bernoulli PMF", latex: "p(x \\mid p) = p^x (1-p)^{1-x}" },
    { description: "Take the logarithm", latex: "\\log p(x \\mid p) = x \\log p + (1-x) \\log(1-p)" },
    { description: "Separate terms by x vs. not-x", latex: "= x \\log\\frac{p}{1-p} + \\log(1-p)" },
    { description: "Identify the natural parameter", latex: "\\eta = \\log\\frac{p}{1-p}", highlight: "eta" },
    { description: "Identify the sufficient statistic", latex: "T(x) = x", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\eta) = \\log(1+e^\\eta)", highlight: "A" },
    { description: "Identify h(x)", latex: "h(x) = 1", highlight: "h" },
    { description: "Write in canonical form", latex: "p(x \\mid \\eta) = \\underbrace{1}_{h(x)} \\exp\\!\\bigl(\\underbrace{\\eta}_{\\eta} \\cdot \\underbrace{x}_{T(x)} - \\underbrace{\\log(1+e^\\eta)}_{A(\\eta)}\\bigr)" },
  ],
  Poisson: [
    { description: "Start with the Poisson PMF", latex: "p(k \\mid \\lambda) = \\frac{\\lambda^k e^{-\\lambda}}{k!}" },
    { description: "Take the logarithm", latex: "\\log p(k \\mid \\lambda) = k \\log\\lambda - \\lambda - \\log(k!)" },
    { description: "Identify the natural parameter", latex: "\\eta = \\log\\lambda", highlight: "eta" },
    { description: "Identify the sufficient statistic", latex: "T(k) = k", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\eta) = e^\\eta = \\lambda", highlight: "A" },
    { description: "Identify h(x)", latex: "h(k) = 1/k!", highlight: "h" },
    { description: "Write in canonical form", latex: "p(k \\mid \\eta) = \\underbrace{\\frac{1}{k!}}_{h(k)} \\exp\\!\\bigl(\\underbrace{\\eta}_{\\eta} \\cdot \\underbrace{k}_{T(k)} - \\underbrace{e^\\eta}_{A(\\eta)}\\bigr)" },
  ],
  Normal: [
    { description: "Start with the Normal PDF", latex: "f(x \\mid \\mu, \\sigma^2) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\!\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)" },
    { description: "Expand the exponent", latex: "= \\frac{1}{\\sqrt{2\\pi}} \\exp\\!\\left(\\frac{\\mu}{\\sigma^2} x - \\frac{1}{2\\sigma^2} x^2 - \\frac{\\mu^2}{2\\sigma^2} - \\log\\sigma\\right)" },
    { description: "Identify the natural parameters (vector)", latex: "\\boldsymbol{\\eta} = \\left(\\frac{\\mu}{\\sigma^2},\\; -\\frac{1}{2\\sigma^2}\\right)", highlight: "eta" },
    { description: "Identify the sufficient statistics (vector)", latex: "\\mathbf{T}(x) = (x,\\; x^2)", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\boldsymbol{\\eta}) = \\frac{\\mu^2}{2\\sigma^2} + \\log\\sigma = -\\frac{\\eta_1^2}{4\\eta_2} - \\frac{1}{2}\\log(-2\\eta_2)", highlight: "A" },
    { description: "Identify h(x)", latex: "h(x) = \\frac{1}{\\sqrt{2\\pi}}", highlight: "h" },
    { description: "Write in canonical form", latex: "f(x \\mid \\boldsymbol{\\eta}) = \\frac{1}{\\sqrt{2\\pi}} \\exp\\!\\bigl(\\boldsymbol{\\eta}^\\top \\mathbf{T}(x) - A(\\boldsymbol{\\eta})\\bigr)" },
  ],
  Exponential: [
    { description: "Start with the Exponential PDF", latex: "f(x \\mid \\lambda) = \\lambda e^{-\\lambda x}, \\quad x \\geq 0" },
    { description: "Take the logarithm", latex: "\\log f(x \\mid \\lambda) = \\log\\lambda - \\lambda x" },
    { description: "Identify the natural parameter", latex: "\\eta = -\\lambda", highlight: "eta" },
    { description: "Identify the sufficient statistic", latex: "T(x) = x", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\eta) = -\\log(-\\eta) = -\\log\\lambda", highlight: "A" },
    { description: "Identify h(x)", latex: "h(x) = 1 \\;\\text{for } x \\geq 0", highlight: "h" },
    { description: "Write in canonical form", latex: "f(x \\mid \\eta) = \\underbrace{1}_{h(x)} \\exp\\!\\bigl(\\underbrace{\\eta}_{\\eta} \\cdot \\underbrace{x}_{T(x)} - \\underbrace{(-\\log(-\\eta))}_{A(\\eta)}\\bigr)" },
  ],
  Geometric: [
    { description: "Start with the Geometric PMF", latex: "p(k \\mid p) = p(1-p)^{k-1}, \\quad k = 1, 2, \\ldots" },
    { description: "Take the logarithm", latex: "\\log p(k \\mid p) = \\log p + (k-1)\\log(1-p)" },
    { description: "Rearrange to isolate k", latex: "= k\\log(1-p) + \\log p - \\log(1-p)" },
    { description: "Identify the natural parameter", latex: "\\eta = \\log(1-p)", highlight: "eta" },
    { description: "Identify the sufficient statistic", latex: "T(k) = k", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\eta) = -\\log(1-e^\\eta)", highlight: "A" },
    { description: "Write in canonical form", latex: "p(k \\mid \\eta) = \\exp\\!\\bigl(\\eta k - (-\\log(1-e^\\eta))\\bigr)" },
  ],
  Binomial: [
    { description: "Start with the Binomial PMF (fixed n)", latex: "p(k \\mid n, p) = \\binom{n}{k} p^k (1-p)^{n-k}" },
    { description: "Take the logarithm", latex: "\\log p(k \\mid n, p) = \\log\\binom{n}{k} + k\\log\\frac{p}{1-p} + n\\log(1-p)" },
    { description: "Identify the natural parameter", latex: "\\eta = \\log\\frac{p}{1-p}", highlight: "eta" },
    { description: "Identify the sufficient statistic", latex: "T(k) = k", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\eta) = n\\log(1+e^\\eta)", highlight: "A" },
    { description: "Identify h(k)", latex: "h(k) = \\binom{n}{k}", highlight: "h" },
    { description: "Write in canonical form", latex: "p(k \\mid \\eta) = \\binom{n}{k} \\exp\\!\\bigl(\\eta k - n\\log(1+e^\\eta)\\bigr)" },
  ],
  NegBin: [
    { description: "Start with the Negative Binomial PMF (fixed r)", latex: "p(k \\mid r, p) = \\binom{k-1}{r-1} p^r (1-p)^{k-r}" },
    { description: "Take the logarithm", latex: "\\log p(k \\mid r, p) = \\log\\binom{k-1}{r-1} + r\\log p + (k-r)\\log(1-p)" },
    { description: "Rearrange to isolate k", latex: "= \\log\\binom{k-1}{r-1} + k\\log(1-p) + r\\log\\frac{p}{1-p}" },
    { description: "Identify the natural parameter", latex: "\\eta = \\log(1-p)", highlight: "eta" },
    { description: "Identify the sufficient statistic", latex: "T(k) = k", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\eta) = -r\\log(1-e^\\eta)", highlight: "A" },
    { description: "Write in canonical form", latex: "p(k \\mid \\eta) = \\binom{k-1}{r-1} \\exp\\!\\bigl(\\eta k + r\\log(1-e^\\eta)\\bigr)" },
  ],
  Gamma: [
    { description: "Start with the Gamma PDF", latex: "f(x \\mid \\alpha, \\beta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} x^{\\alpha-1} e^{-\\beta x}, \\quad x > 0" },
    { description: "Take the logarithm", latex: "\\log f(x \\mid \\alpha, \\beta) = (\\alpha-1)\\log x - \\beta x + \\alpha\\log\\beta - \\log\\Gamma(\\alpha)" },
    { description: "Identify the natural parameters (vector)", latex: "\\boldsymbol{\\eta} = (\\alpha-1,\\; -\\beta)", highlight: "eta" },
    { description: "Identify the sufficient statistics (vector)", latex: "\\mathbf{T}(x) = (\\log x,\\; x)", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\boldsymbol{\\eta}) = \\log\\Gamma(\\eta_1+1) - (\\eta_1+1)\\log(-\\eta_2)", highlight: "A" },
    { description: "Identify h(x)", latex: "h(x) = 1 \\;\\text{for } x > 0", highlight: "h" },
    { description: "Write in canonical form", latex: "f(x \\mid \\boldsymbol{\\eta}) = \\exp\\!\\bigl(\\boldsymbol{\\eta}^\\top \\mathbf{T}(x) - A(\\boldsymbol{\\eta})\\bigr)" },
  ],
  Beta: [
    { description: "Start with the Beta PDF", latex: "f(x \\mid \\alpha, \\beta) = \\frac{1}{B(\\alpha,\\beta)} x^{\\alpha-1}(1-x)^{\\beta-1}, \\quad x \\in (0,1)" },
    { description: "Take the logarithm", latex: "\\log f(x \\mid \\alpha, \\beta) = (\\alpha-1)\\log x + (\\beta-1)\\log(1-x) - \\log B(\\alpha,\\beta)" },
    { description: "Identify the natural parameters (vector)", latex: "\\boldsymbol{\\eta} = (\\alpha-1,\\; \\beta-1)", highlight: "eta" },
    { description: "Identify the sufficient statistics (vector)", latex: "\\mathbf{T}(x) = (\\log x,\\; \\log(1-x))", highlight: "T" },
    { description: "Identify the log-partition function", latex: "A(\\boldsymbol{\\eta}) = \\log B(\\eta_1+1, \\eta_2+1)", highlight: "A" },
    { description: "Identify h(x)", latex: "h(x) = 1 \\;\\text{for } x \\in (0,1)", highlight: "h" },
    { description: "Write in canonical form", latex: "f(x \\mid \\boldsymbol{\\eta}) = \\exp\\!\\bigl(\\boldsymbol{\\eta}^\\top \\mathbf{T}(x) - A(\\boldsymbol{\\eta})\\bigr)" },
  ],
};

/** Color scheme for canonical form components (used by all viz components). */
export const canonicalFormColors = {
  h: "#2563eb",    // blue — h(x)
  eta: "#7c3aed",  // purple — η(θ)
  T: "#16a34a",    // green — T(x)
  A: "#d97706",    // amber — A(η)
} as const;
