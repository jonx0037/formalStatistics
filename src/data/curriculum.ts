export interface Track {
  id: string;
  number: number;
  title: string;
  description: string;
  topicCount: number;
}

export const tracks: Track[] = [
  {
    id: 'foundations-of-probability',
    number: 1,
    title: 'Foundations of Probability',
    description: 'Kolmogorov axioms, conditional probability, random variables, expectation',
    topicCount: 4,
  },
  {
    id: 'core-distributions',
    number: 2,
    title: 'Core Distributions & Families',
    description: 'Discrete and continuous distributions, exponential families, multivariate distributions',
    topicCount: 4,
  },
  {
    id: 'convergence-limit-theorems',
    number: 3,
    title: 'Convergence & Limit Theorems',
    description: 'Modes of convergence, law of large numbers, central limit theorem, tail bounds',
    topicCount: 4,
  },
  {
    id: 'statistical-estimation',
    number: 4,
    title: 'Statistical Estimation',
    description: 'Bias-variance, maximum likelihood, method of moments, sufficiency',
    topicCount: 4,
  },
  {
    id: 'hypothesis-testing',
    number: 5,
    title: 'Hypothesis Testing & Confidence',
    description: 'Neyman-Pearson paradigm, likelihood ratio tests, confidence intervals, multiple testing',
    topicCount: 4,
  },
  {
    id: 'regression-linear-models',
    number: 6,
    title: 'Regression & Linear Models',
    description: 'Least squares, generalized linear models, regularization, model selection',
    topicCount: 4,
  },
  {
    id: 'bayesian-statistics',
    number: 7,
    title: 'Bayesian Statistics',
    description: 'Prior selection, MCMC computation, model comparison, hierarchical models',
    topicCount: 4,
  },
  {
    id: 'high-dimensional-nonparametric',
    number: 8,
    title: 'High-Dimensional & Nonparametric',
    description: 'Order statistics, kernel density estimation, bootstrap, empirical processes',
    topicCount: 4,
  },
];
