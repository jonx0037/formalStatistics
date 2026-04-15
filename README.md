# formalStatistics

**The probability and statistics foundations behind modern machine learning.**

Deep-dive explainers combining rigorous mathematics, interactive visualizations, and working code. The bridge between [formalCalculus](https://formalcalculus.com) and [formalML](https://formalml.com) — building the probabilistic and inferential machinery that ML assumes you already have.

[www.formalstatistics.com](https://www.formalstatistics.com)

---

## What This Is

formalStatistics is a curated collection of long-form explainers on the probability and statistics foundations that modern ML relies on. Every topic receives a three-pillar treatment:

1. **Rigorous exposition** — Formal definitions, theorems, and proofs presented with full mathematical detail. Every convergence argument is expanded. Every estimator is derived from first principles.
2. **Interactive visualization** — Embedded widgets that let you manipulate parameters and watch the math come alive (e.g., slide a sample size to watch the Central Limit Theorem kick in, drag data points to see a least-squares fit update in real time, animate sequential Bayesian updating as observations arrive).
3. **Working code** — Production-oriented Python implementations you can run immediately, with bridges to NumPy, SciPy, statsmodels, and standard statistical computing libraries.

The site exists because the gap between "I can call `sklearn.fit()`" and "I understand why this estimator is consistent" is wider than it needs to be.

### Relationship to Sister Sites

formalStatistics sits in the middle of a three-site learning path:

```
formalCalculus → formalStatistics → formalML
(calculus & analysis)  (probability & stats)  (ML mathematics)
```

Where formalCalculus covers the calculus and analysis that probability *assumes*, and formalML covers the mathematical machinery *of* machine learning, formalStatistics covers the probabilistic and inferential foundations that connect them. Every topic includes backward links to formalCalculus prerequisites and forward links to the formalML topics it enables.

## Curriculum

32 topics across 8 tracks, progressing from foundational probability through the statistical theory that directly feeds into graduate-level ML.

### Track 1: Foundations of Probability

| Topic | Level | Description |
|-------|-------|-------------|
| Sample Spaces, Events & Axioms | Foundational | Kolmogorov axioms, sigma-algebras for the working statistician, combinatorial probability |
| Conditional Probability & Independence | Foundational | Bayes' theorem, law of total probability, conditional independence — the backbone of graphical models |
| Random Variables & Distributions | Foundational | Measurable functions, PMFs and PDFs, CDFs — the formal bridge from events to numbers |
| Expectation, Variance & Moments | Foundational | Integration against a measure, moment-generating functions, characteristic functions |

### Track 2: Core Distributions & Families

| Topic | Level | Description |
|-------|-------|-------------|
| Discrete Distributions | Foundational | Bernoulli, Binomial, Poisson, Geometric, Negative Binomial — derivations, relationships, and ML appearances |
| Continuous Distributions | Foundational | Normal, Exponential, Gamma, Beta, Uniform — density derivations, transformations, and why the Gaussian is everywhere |
| Exponential Families | Intermediate | Sufficient statistics, natural parameters, log-partition function — the unifying framework for GLMs |
| Multivariate Distributions | Intermediate | Joint, marginal, conditional densities, the multivariate normal, copulas — dependence beyond correlation |

### Track 3: Convergence & Limit Theorems

| Topic | Level | Description |
|-------|-------|-------------|
| Modes of Convergence | Intermediate | Almost sure, in probability, in distribution, in Lp — the hierarchy and when each matters |
| Law of Large Numbers | Intermediate | Weak and strong LLN, Kolmogorov's theorem — why sample averages work |
| Central Limit Theorem | Intermediate | Lindeberg-Lévy, Lindeberg-Feller, Berry-Esseen bound — why normality emerges and how fast |
| Large Deviations & Tail Bounds | Advanced | Markov, Chebyshev, Chernoff, Hoeffding, sub-Gaussian theory — the on-ramp to concentration inequalities |

### Track 4: Statistical Estimation

| Topic | Level | Description |
|-------|-------|-------------|
| Point Estimation & Bias-Variance | Intermediate | Estimators as random variables, bias, variance, MSE decomposition — the framework for evaluating any estimator |
| Maximum Likelihood Estimation | Intermediate | Likelihood function, score, Fisher information, asymptotic normality — the workhorse of parametric inference |
| Method of Moments & M-Estimation | Intermediate | Moment equations, generalized method of moments, Z-estimators — robust alternatives to MLE |
| Sufficient Statistics & the Rao-Blackwell Theorem | Intermediate | Information compression, UMVUE, completeness — why sufficient statistics are optimal |

### Track 5: Hypothesis Testing & Confidence

| Topic | Level | Description |
|-------|-------|-------------|
| Hypothesis Testing Framework | Foundational | Null and alternative, Type I/II errors, power, p-values — the Neyman-Pearson paradigm |
| Likelihood Ratio Tests & Neyman-Pearson | Intermediate | Most powerful tests, the likelihood ratio principle, Wilks' theorem — optimal testing theory |
| Confidence Intervals & Duality | Intermediate | Pivotal quantities, inversion of tests, coverage probability — what confidence actually means |
| Multiple Testing & False Discovery | Intermediate | Bonferroni, Holm, Benjamini-Hochberg FDR — controlling errors when you test thousands of hypotheses |

### Track 6: Regression & Linear Models

| Topic | Level | Description |
|-------|-------|-------------|
| Simple & Multiple Linear Regression | Foundational | Least squares as projection, Gauss-Markov theorem, residual analysis — the geometry of regression |
| Generalized Linear Models | Intermediate | Link functions, exponential family connection, deviance — logistic regression, Poisson regression, and beyond |
| Regularization & Penalized Estimation | Intermediate | Ridge, lasso, elastic net — bias-variance tradeoff as explicit penalization, Bayesian interpretations |
| Model Selection & Information Criteria | Intermediate | AIC, BIC, cross-validation theory, Mallows' Cp — principled approaches to model complexity |

### Track 7: Bayesian Statistics

| Topic | Level | Description |
|-------|-------|-------------|
| Bayesian Foundations & Prior Selection | Intermediate | Prior, likelihood, posterior, conjugacy, Jeffreys priors — the Bayesian machinery from scratch |
| Bayesian Computation | Intermediate | MCMC, Metropolis-Hastings, Gibbs sampling, Hamiltonian Monte Carlo — sampling from posteriors |
| Bayesian Model Comparison | Advanced | Bayes factors, marginal likelihood, posterior predictive checks — comparing models the Bayesian way |
| Hierarchical & Empirical Bayes | Advanced | Multilevel models, shrinkage estimators, James-Stein — borrowing strength across groups |

### Track 8: High-Dimensional & Nonparametric Methods

| Topic | Level | Description |
|-------|-------|-------------|
| Order Statistics & Quantiles | Intermediate | Distribution-free inference, sample quantile asymptotics — the foundation of nonparametric methods |
| Kernel Density Estimation | Intermediate | Bandwidth selection, bias-variance for density estimation, multivariate KDE — nonparametric density learning |
| The Bootstrap | Intermediate | Efron's bootstrap, bootstrap confidence intervals, bootstrap hypothesis tests — resampling as inference |
| Empirical Processes & Uniform Convergence | Advanced | Glivenko-Cantelli, Donsker's theorem, VC dimension — the on-ramp to statistical learning theory |

### Forward Links to formalML

Every track connects forward to specific formalML topics:

| formalStatistics Track | Enables (on formalml.com) |
|----------------------|--------------------------|
| Foundations of Probability | Measure-Theoretic Probability, Shannon Entropy |
| Core Distributions & Families | Bayesian Nonparametrics, Information Geometry |
| Convergence & Limit Theorems | Concentration Inequalities, PAC Learning |
| Statistical Estimation | Information Geometry, Gradient Descent (Fisher information) |
| Hypothesis Testing & Confidence | PAC Learning, Rate-Distortion Theory |
| Regression & Linear Models | Spectral Theorem, PCA, Gradient Descent |
| Bayesian Statistics | Bayesian Nonparametrics, KL Divergence |
| High-Dimensional & Nonparametric | PAC Learning, Concentration Inequalities, MDL |

### Backward Links from formalCalculus

| formalStatistics Track | Requires (on formalcalculus.com) |
|----------------------|--------------------------------|
| Foundations of Probability | Sequences & Limits, Sigma-Algebras & Measures |
| Core Distributions & Families | Improper Integrals & Special Functions, Change of Variables |
| Convergence & Limit Theorems | Uniform Convergence, Lp Spaces |
| Statistical Estimation | The Derivative & Chain Rule, Multiple Integrals & Fubini's Theorem |
| Hypothesis Testing & Confidence | The Riemann Integral & FTC, Series Convergence |
| Regression & Linear Models | Partial Derivatives & the Gradient, The Hessian & Second-Order Analysis |
| Bayesian Statistics | Multiple Integrals & Fubini's Theorem, The Lebesgue Integral |
| High-Dimensional & Nonparametric | Metric Spaces & Topology, Normed & Banach Spaces |

## Tech Stack

| Layer | Tool |
|-------|------|
| Framework | [Astro](https://astro.build) (static site generation) |
| Content | MDX with KaTeX for math rendering |
| Styling | Tailwind CSS |
| Visualizations | React 19 + D3.js (interactive components) |
| Search | [Pagefind](https://pagefind.app) (static search) |
| Package manager | pnpm |
| Hosting | Vercel |

## Project Structure

```
├── src/
│   ├── pages/              # Astro page routes
│   ├── content/
│   │   └── topics/         # MDX topic files
│   ├── components/
│   │   ├── ui/             # Astro UI components (Nav, TopicCard, etc.)
│   │   └── viz/            # React + D3 visualization components
│   │       └── shared/     # Shared types, color scales, hooks, utility modules
│   ├── layouts/            # Page layout templates
│   ├── data/               # Curriculum graph data, sample datasets
│   ├── lib/                # Utility modules
│   └── styles/             # Global CSS, design tokens
├── public/                 # Static assets
├── docs/plans/             # Planning & handoff documents
├── notebooks/              # Research notebooks (Jupyter)
├── astro.config.mjs        # Astro configuration
├── package.json
└── tsconfig.json
```

## Local Development

```bash
# Install dependencies
pnpm install

# Start dev server (localhost:4321)
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Author

**Jonathan Rocha** — Data scientist and researcher. MS Data Science (SMU), MA English (Texas A&M University-Central Texas), BA History (Texas A&M University). Research interests: time-series data mining, topology-aware deep learning.

- GitHub: [@jonx0037](https://github.com/jonx0037)
- Consultancy: [DataSalt LLC](https://datasalt.ai)
- Predecessor: [formalCalculus](https://formalcalculus.com)
- Successor: [formalML](https://formalml.com)

## License

All rights reserved.
