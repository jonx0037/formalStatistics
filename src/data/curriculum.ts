export interface Track {
  id: string;
  number: number;
  title: string;
  description: string;
  domain: string;
  topicCount: number;
}

export const tracks: Track[] = [
  {
    id: 'foundations-of-probability',
    number: 1,
    title: 'Foundations of Probability',
    description: 'Kolmogorov axioms, conditional probability, random variables, expectation',
    domain: 'foundations-of-probability',
    topicCount: 4,
  },
  {
    id: 'core-distributions',
    number: 2,
    title: 'Core Distributions & Families',
    description: 'Discrete and continuous distributions, exponential families, multivariate distributions',
    domain: 'core-distributions',
    topicCount: 4,
  },
  {
    id: 'convergence-limit-theorems',
    number: 3,
    title: 'Convergence & Limit Theorems',
    description: 'Modes of convergence, law of large numbers, central limit theorem, tail bounds',
    domain: 'convergence-limit-theorems',
    topicCount: 4,
  },
  {
    id: 'statistical-estimation',
    number: 4,
    title: 'Statistical Estimation',
    description: 'Bias-variance, maximum likelihood, method of moments, sufficiency',
    domain: 'statistical-estimation',
    topicCount: 4,
  },
  {
    id: 'hypothesis-testing',
    number: 5,
    title: 'Hypothesis Testing & Confidence',
    description: 'Neyman-Pearson paradigm, likelihood ratio tests, confidence intervals, multiple testing',
    domain: 'hypothesis-testing-confidence',
    topicCount: 4,
  },
  {
    id: 'regression-linear-models',
    number: 6,
    title: 'Regression & Linear Models',
    description: 'Least squares, generalized linear models, regularization, model selection',
    domain: 'regression-linear-models',
    topicCount: 4,
  },
  {
    id: 'bayesian-statistics',
    number: 7,
    title: 'Bayesian Statistics',
    description: 'Prior selection, MCMC computation, model comparison, hierarchical models',
    domain: 'bayesian-statistics',
    topicCount: 4,
  },
  {
    id: 'high-dimensional-nonparametric',
    number: 8,
    title: 'High-Dimensional & Nonparametric',
    description: 'Order statistics, kernel density estimation, bootstrap, empirical processes',
    domain: 'high-dimensional-nonparametric',
    topicCount: 4,
  },
];
