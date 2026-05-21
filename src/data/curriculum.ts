export interface Track {
  id: string;
  number: number;
  title: string;
  description: string;
  domain: string;
  published: string[];
  planned: string[];
}

export const tracks: Track[] = [
  {
    id: 'foundations-of-probability',
    number: 1,
    title: 'Foundations of Probability',
    description: 'Kolmogorov axioms, conditional probability, random variables, expectation',
    domain: 'foundations-of-probability',
    published: [
      'Sample Spaces, Events & Axioms',
      'Conditional Probability & Independence',
      'Random Variables & Distribution Functions',
      'Expectation, Variance & Moments',
    ],
    planned: [],
  },
  {
    id: 'core-distributions',
    number: 2,
    title: 'Core Distributions & Families',
    description: 'Discrete and continuous distributions, exponential families, multivariate distributions',
    domain: 'core-distributions',
    published: [
      'Discrete Distributions',
      'Continuous Distributions',
      'Exponential Families',
      'Multivariate Distributions',
    ],
    planned: [],
  },
  {
    id: 'convergence-limit-theorems',
    number: 3,
    title: 'Convergence & Limit Theorems',
    description: 'Modes of convergence, law of large numbers, central limit theorem, tail bounds',
    domain: 'convergence-limit-theorems',
    published: [
      'Modes of Convergence',
      'Law of Large Numbers',
      'Central Limit Theorem',
      'Large Deviations & Tail Bounds',
    ],
    planned: [],
  },
  {
    id: 'statistical-estimation',
    number: 4,
    title: 'Statistical Estimation',
    description: 'Bias-variance, maximum likelihood, method of moments, sufficiency',
    domain: 'statistical-estimation',
    published: [
      'Point Estimation & Bias-Variance',
      'Maximum Likelihood Estimation',
      'Method of Moments & M-Estimation',
      'Sufficient Statistics & Rao-Blackwell',
    ],
    planned: [],
  },
  {
    id: 'hypothesis-testing',
    number: 5,
    title: 'Hypothesis Testing & Confidence',
    description: 'Neyman-Pearson paradigm, likelihood ratio tests, confidence intervals, multiple testing',
    domain: 'hypothesis-testing-confidence',
    published: [
      'Hypothesis Testing Framework',
      'Likelihood-Ratio Tests & Neyman-Pearson',
      'Confidence Intervals & Duality',
      'Multiple Testing & False Discovery',
    ],
    planned: [],
  },
  {
    id: 'regression-linear-models',
    number: 6,
    title: 'Regression & Linear Models',
    description: 'Least squares, generalized linear models, regularization, model selection',
    domain: 'regression-linear-models',
    published: [
      'Simple & Multiple Linear Regression',
      'Generalized Linear Models',
      'Regularization & Penalized Estimation',
      'Model Selection & Information Criteria',
    ],
    planned: [],
  },
  {
    id: 'bayesian-statistics',
    number: 7,
    title: 'Bayesian Statistics',
    description: 'Prior selection, MCMC computation, model comparison, hierarchical models',
    domain: 'bayesian-statistics',
    published: [
      'Bayesian Foundations & Prior Selection',
      'Bayesian Computation & MCMC',
      'Bayesian Model Comparison & BMA',
      'Hierarchical & Empirical Bayes',
    ],
    planned: ['Empirical Bayes'],
  },
  {
    id: 'high-dimensional-nonparametric',
    number: 8,
    title: 'High-Dimensional & Nonparametric',
    description: 'Order statistics, kernel density estimation, bootstrap, empirical processes',
    domain: 'high-dimensional-nonparametric',
    published: [
      'Order Statistics & Quantiles',
      'Kernel Density Estimation',
      'The Bootstrap',
      'Empirical Processes & Uniform Convergence',
    ],
    planned: ['Parametric vs Nonparametric Models'],
  },
  {
    id: 'time-series-state-space',
    number: 9,
    title: 'Time-Series & State-Space Methods',
    description: 'Hidden Markov models, state-space inference, and the foundations of time-series statistics',
    domain: 'time-series-state-space',
    published: [],
    planned: ['Hidden Markov Models'],
  },
];
