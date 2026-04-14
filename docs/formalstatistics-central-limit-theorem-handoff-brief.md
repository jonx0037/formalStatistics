# Claude Code Handoff Brief: Central Limit Theorem

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)  
**Repo:** `github.com/jonx0037/formalStatistics`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/central-limit-theorem/11_central_limit_theorem.ipynb`

---

## Important: Eleventh Topic — Third in Track 3

This is **topic 11 of 32** and the **third topic in Track 3 (Convergence & Limit Theorems)**. It completes the main asymptotic trilogy: Topic 9 defined the modes of convergence, Topic 10 proved the LLN (qualitative convergence of the sample mean), and Topic 11 gives the CLT (the distributional shape of the fluctuations). The CLT is the single most important theorem in applied statistics — it is the mathematical reason confidence intervals, hypothesis tests, and p-values work.

**Implications:**

1. **All ten preceding topics are published.** Cross-references to `sample-spaces`, `conditional-probability`, `random-variables`, `expectation-moments`, `discrete-distributions`, `continuous-distributions`, `exponential-families`, `multivariate-distributions`, `modes-of-convergence`, and `law-of-large-numbers` use live internal links.
2. **Nine explicit "(coming soon)" promises from earlier topics must be fulfilled.** See §1 below for the complete list with line numbers. These span four published topics (`modes-of-convergence`, `expectation-moments`, `discrete-distributions`, `law-of-large-numbers`).
3. **Track 3 progress.** This is the third of 4 topics in Track 3. Topic 12 (Large Deviations & Tail Bounds) is the only remaining topic in the track. Topic 12 lists `modes-of-convergence` as a direct prerequisite, not this topic — but it references CLT results for rate comparisons.
4. **The difficulty level is "Intermediate"** — the fifth Intermediate topic after Exponential Families, Multivariate Distributions, Modes of Convergence, and Law of Large Numbers. The reader is assumed to be comfortable with convergence modes (Topic 9), the LLN hierarchy (Topic 10), MGFs and their properties (Topic 4), and all named distributions (Topics 5–6).
5. **Topic 9 already has a CLT preview visualization.** The `ConvergenceInDistributionExplorer` component includes a `clt-preview` mode showing the histogram-becomes-bell-curve phenomenon for Exponential, Uniform, and Bernoulli. Topic 10's `ConvergenceRateExplorer` shows the 1/√n CLT envelope. Topic 11 cannot simply repeat these — it must go deeper: prove WHY the CLT works, show WHEN it fails (Lindeberg condition), quantify HOW FAST (Berry–Esseen), and connect to APPLICATIONS (confidence intervals, hypothesis tests, SGD noise).
6. **This topic extends `convergence.ts`** with `cltNormalization` (promised in the JSDoc header) and characteristic function helpers. It also creates a new data module `central-limit-theorem-data.ts`.
7. **Update the references spreadsheet** with hyperlinks for all references cited in this topic.

---

## 1. Objective

Add a new topic page **"Central Limit Theorem"** as the **third topic in the Convergence & Limit Theorems track** on formalstatistics.com.

1. This is **topic 11 of 32**.
2. **Internal prerequisites:**
   - `modes-of-convergence` (direct) — the four modes of convergence, convergence in distribution (Definition 9.6), Lévy's continuity theorem (Remark 2), MGF convergence technique, Slutsky's theorem (Theorem 9.10), delta method (Theorems 9.11–9.12), Poisson limit proof (Theorem 9.13 — the structural template for the CLT proof)
   - `law-of-large-numbers` (direct) — WLLN and SLLN provide the qualitative statement (X̄ₙ → μ) that the CLT refines distributionally; the hierarchy SLLN → LIL → CLT (Remark 10.5)
3. **External prerequisites from formalcalculus.com:**
   - `sequences-limits` — limits, limsup/liminf (for Berry–Esseen bounds and rate analysis)
   - `series-convergence` — Taylor series, power series expansions (for the MGF proof: Taylor expansion of log M(t/σ√n))
   - `differentiation` — Taylor's theorem with remainder (for the MGF proof and the delta method)
4. **Downstream within formalStatistics:**
   - `large-deviations` (indirect) — concentration inequalities provide the exponential-rate tail bounds that complement the CLT's O(1/√n) rate
   - `confidence-intervals` (direct) — the CLT is the mathematical foundation for asymptotic confidence intervals
   - `hypothesis-testing` (direct) — z-tests and asymptotic tests depend on CLT normality
   - `empirical-processes` (indirect) — Donsker's theorem is the functional CLT
5. **Forward links to formalml.com:**
   - `stochastic-gradient-descent` — SGD noise is approximately Normal by the CLT: the mini-batch gradient ∇L̂_B = (1/B)Σᵢ∈B ∇ℓ(θ; xᵢ) is a sum of iid terms, so its fluctuations around the true gradient are Gaussian at scale 1/√B
   - `generalization-bounds` — Rademacher complexity bounds and symmetrization arguments use CLT-type reasoning for the empirical process
   - `bayesian-neural-networks` — the Bernstein–von Mises theorem (the Bayesian CLT) proves that the posterior concentrates around the MLE at rate 1/√n with Gaussian shape, regardless of the prior
   - `confidence-intervals` (within formalStatistics) — the CLT is what makes confidence intervals work: X̄ₙ ± zα/2 · σ/√n covers the true mean with probability ≈ 1 − α
6. This topic **extends** `convergence.ts` and **creates** a new data module `central-limit-theorem-data.ts`.

**Content scope:**

- Why the LLN isn't enough — the shape of fluctuations matters (motivating narrative)
- De Moivre–Laplace theorem: Binomial → Normal (the historical origin and classical special case)
- Classical CLT statement: Lindeberg–Lévy (iid, finite variance)
- Full proof via MGFs: Taylor expansion of log M(t/σ√n), Lévy continuity theorem
- Proof via characteristic functions (the general version — CFs always exist)
- Lindeberg CLT: the "adult" version for independent, non-identically distributed summands
- Berry–Esseen theorem: the convergence rate is O(1/√n), and skewness determines the constant
- Multivariate CLT: statement connecting back to Topic 8 (Multivariate Distributions)
- Delta method revisited: rigorous treatment now that the CLT is proved (connecting back to Topic 9)
- ML connections: confidence intervals, hypothesis tests, SGD noise, Bayesian CLT

**Binding promises fulfilled:**

| Source file | Line | Exact text | What Topic 11 delivers |
|---|---|---|---|
| `modes-of-convergence.mdx` | 576 | "This is exactly the technique we'll use to prove...the Central Limit Theorem (Topic 11, coming soon)" | §4: Full CLT proof via MGFs using the same Lévy continuity technique as the Poisson limit proof |
| `modes-of-convergence.mdx` | 578 | "The characteristic function version avoids this existence requirement, which is why the full CLT proof uses characteristic functions" | §5: CLT proof via characteristic functions — the general version that does not require MGF existence |
| `modes-of-convergence.mdx` | 961 | "By the CLT (which we're previewing — the full proof is Topic 11, coming soon)" | §9: Delta method revisited with the CLT now fully proved, making the delta method examples rigorous |
| `modes-of-convergence.mdx` | 1070 | "The CLT proof is more involved because the MGF of √n(X̄−μ)/σ requires a careful Taylor expansion of log M_{X₁}(t/(σ√n))" | §4: The specific Taylor expansion of log M(t/σ√n) carried out in full detail |
| `modes-of-convergence.mdx` | 1125 | Berry–Esseen theorem mentioned as the convergence rate for CLT | §7: Berry–Esseen theorem stated with the constant C ≤ 0.4748, convergence rate visualization |
| `modes-of-convergence.mdx` | 1201 | "The CLT proves √n(X̄ₙ−μ)/σ →d N(0,1). The proof technique is MGF convergence" | §3: Core CLT statement (Theorem 11.2); §4: MGF proof |
| `expectation-moments.mdx` | 763 | "We'll use this in the proof of the Central Limit Theorem" | §4: MGF uniqueness (Topic 4, Theorem 17) invoked to identify the limiting distribution as N(0,1) |
| `discrete-distributions.mdx` | 302 | "By the Central Limit Theorem...an approximate 95% confidence interval" | §2: De Moivre–Laplace theorem (Binomial → Normal); §10: confidence interval construction |
| `law-of-large-numbers.mdx` | 1031 | "√n(X̄ₙ−μ)/σ →d N(0,1). The fluctuations at scale 1/√n are approximately Normal" | §3: The full CLT statement and proof, completing the distributional complement to the LLN |

---

## 2. MDX File

### Location

```
src/content/topics/central-limit-theorem.mdx
```

The entry `id` will be `central-limit-theorem`. The dynamic route resolves to `/topics/central-limit-theorem`.

### Frontmatter

```yaml
---
title: "Central Limit Theorem"
subtitle: "Why normality emerges from chaos — the shape of fluctuations, the rate of convergence, and why almost all of classical statistics works."
status: "published"
difficulty: "intermediate"
prerequisites:
  - "modes-of-convergence"
  - "law-of-large-numbers"
tags:
  - "probability"
  - "convergence"
  - "central-limit-theorem"
  - "clt"
  - "normal-approximation"
  - "de-moivre-laplace"
  - "lindeberg"
  - "berry-esseen"
  - "characteristic-functions"
  - "delta-method"
  - "confidence-intervals"
domain: "convergence-limit-theorems"
videoId: null
notebookPath: "notebooks/central-limit-theorem/11_central_limit_theorem.ipynb"
githubUrl: "https://github.com/jonx0037/formalStatistics/blob/main/src/content/topics/central-limit-theorem.mdx"
datePublished: 2026-04-15
estimatedReadTime: 45
abstract: "The Law of Large Numbers says the sample mean converges to the population mean. But convergence is only half the story — what shape do the fluctuations take along the way? The Central Limit Theorem answers: regardless of the underlying distribution, the standardized sample mean √n(X̄ₙ − μ)/σ converges in distribution to a standard Normal. This universality is breathtaking — Bernoulli, Exponential, Poisson, Uniform, any distribution with finite variance produces Gaussian fluctuations at the 1/√n scale. The CLT is the mathematical engine behind confidence intervals, hypothesis tests, p-values, and the entire apparatus of frequentist inference. This topic traces the CLT from its historical origin (de Moivre's 1733 approximation of the Binomial by the Normal) through the modern proofs (via moment-generating functions and characteristic functions), the general Lindeberg version for non-identical summands, the Berry–Esseen convergence rate bound, the multivariate extension, and the delta method for nonlinear transformations. The proof via MGFs follows the exact template established in Topic 9 for the Poisson limit theorem: compute the MGF, Taylor-expand the logarithm, show pointwise convergence to the MGF of N(0,1), and invoke Lévy's continuity theorem. The characteristic function proof extends the result to distributions without MGFs. The Lindeberg CLT reveals when the CLT works and when it fails — the key is that no single summand dominates the total variance. Berry–Esseen quantifies how fast the convergence happens, revealing that skewness is the enemy of normality."
formalcalculusPrereqs:
  - topic: "sequences-limits"
    site: "formalcalculus"
    relationship: "Pointwise limits of MGFs and characteristic functions, limsup in Berry–Esseen rate analysis."
  - topic: "series-convergence"
    site: "formalcalculus"
    relationship: "Taylor series and power series expansions are the backbone of the MGF proof: log(1 + x) = x − x²/2 + O(x³) applied with x = O(1/n)."
  - topic: "differentiation"
    site: "formalcalculus"
    relationship: "Taylor's theorem with remainder provides the rigorous bound on the remainder term in the MGF expansion. The delta method requires differentiability of the transformation g."
formalmlConnections:
  - topic: "stochastic-gradient-descent"
    site: "formalml"
    relationship: "The mini-batch gradient is an average of B iid gradient samples. By the CLT, its fluctuations around the true gradient are Gaussian at scale σ/√B, where σ² is the per-sample gradient variance. This Gaussian noise structure motivates the learning rate schedule η ∝ 1/√t and explains why larger batches reduce noise but have diminishing returns."
  - topic: "generalization-bounds"
    site: "formalml"
    relationship: "Rademacher complexity bounds use symmetrization and CLT-type reasoning. The empirical Rademacher complexity R̂ₙ converges to its expectation at rate 1/√n by the CLT, which gives the O(1/√n) generalization bound for bounded loss functions."
  - topic: "bayesian-neural-networks"
    site: "formalml"
    relationship: "The Bernstein–von Mises theorem (the Bayesian CLT) proves that the posterior distribution √n(θ − θ₀) | X₁,...,Xₙ converges in total variation to N(0, I(θ₀)⁻¹) under regularity conditions, where I(θ₀) is the Fisher information. This is why Bayesian credible intervals and frequentist confidence intervals agree asymptotically."
  - topic: "monte-carlo-methods"
    site: "formalml"
    relationship: "The CLT gives the convergence rate of Monte Carlo estimators: the standard error of (1/n)Σf(Xᵢ) is σ_f/√n, leading to the canonical ±1.96σ/√n confidence band for Monte Carlo estimates."
connections:
  - topic: "modes-of-convergence"
    relationship: "Topic 9 defined convergence in distribution, proved Lévy's continuity theorem (via MGFs), Slutsky's theorem, the delta method, and proved the Poisson limit theorem. The CLT proof follows the identical structure as the Poisson limit proof."
  - topic: "law-of-large-numbers"
    relationship: "Topic 10 proved X̄ₙ → μ (qualitative convergence). The CLT refines this: the deviations √n(X̄ₙ − μ) are Gaussian. The LLN says 'it converges'; the CLT says 'the fluctuations are Normal.'"
  - topic: "expectation-moments"
    relationship: "Topic 4 developed MGFs (Definition 15, Theorem 16), MGF uniqueness (Theorem 17), and moment inequalities. The CLT proof uses MGF uniqueness to identify the limit as N(0,1)."
  - topic: "discrete-distributions"
    relationship: "Topic 5 previewed the Normal approximation to the Binomial. The de Moivre–Laplace theorem (§2) makes this rigorous as the first historical CLT."
  - topic: "continuous-distributions"
    relationship: "Topic 6 developed the Normal distribution properties (PDF, CDF, MGF). The CLT establishes why the Normal appears so ubiquitously in nature and statistics."
  - topic: "multivariate-distributions"
    relationship: "Topic 8 developed the multivariate Normal. The multivariate CLT (§8) extends the univariate result to random vectors."
  - topic: "large-deviations"
    relationship: "Topic 12 complements the CLT: the CLT gives the shape (Gaussian) at scale 1/√n, while concentration inequalities give the rate (exponential) at all scales."
references:
  - type: "book"
    title: "Probability: Theory and Examples"
    author: "Rick Durrett"
    year: 2019
    edition: "5th"
    publisher: "Cambridge University Press"
    isbn: "978-1-108-47368-2"
    url: "https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf"
    note: "Chapter 3 covers the CLT in all its forms — Lindeberg–Lévy, Lindeberg–Feller, Berry–Esseen. The standard graduate reference for the characteristic function proof."
  - type: "book"
    title: "Probability and Measure"
    author: "Patrick Billingsley"
    year: 2012
    edition: "Anniversary"
    publisher: "Wiley"
    isbn: "978-1-118-12237-2"
    url: "https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372"
    note: "Chapter 27 covers the CLT with both MGF and characteristic function proofs. The Lindeberg condition treatment is definitive."
  - type: "book"
    title: "Statistical Inference"
    author: "George Casella & Roger L. Berger"
    year: 2002
    edition: "2nd"
    publisher: "Cengage"
    isbn: "978-0-534-24312-8"
    url: "https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/"
    note: "Chapter 5 covers the CLT from a statistics perspective, with emphasis on confidence intervals and the delta method."
  - type: "book"
    title: "All of Statistics"
    author: "Larry Wasserman"
    year: 2004
    publisher: "Springer"
    isbn: "978-0-387-40272-7"
    url: "https://link.springer.com/book/10.1007/978-0-387-21736-9"
    note: "Chapter 5 covers the CLT concisely with statistical applications. Good for the Berry–Esseen theorem and confidence interval construction."
  - type: "book"
    title: "An Introduction to Probability Theory and Its Applications"
    author: "William Feller"
    year: 1971
    edition: "2nd"
    volume: "2"
    publisher: "Wiley"
    isbn: "978-0-471-25709-7"
    url: "https://www.wiley.com/en-us/An+Introduction+to+Probability+Theory+and+Its+Applications%2C+Volume+2%2C+2nd+Edition-p-9780471257097"
    note: "Chapter XV contains the definitive characteristic function proof of the CLT. Feller's treatment of the Lindeberg condition and the Berry–Esseen theorem is still the gold standard."
  - type: "book"
    title: "Pattern Recognition and Machine Learning"
    author: "Christopher M. Bishop"
    year: 2006
    publisher: "Springer"
    isbn: "978-0-387-31073-2"
    url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/"
    note: "Chapter 2 uses CLT results implicitly throughout. The Gaussian distribution's central role in machine learning is a consequence of the CLT."
---
```

### Content Structure

| Section | MDX Heading | Formal Elements | Figure | Interactive Component |
|---|---|---|---|---|
| 1. Overview | "Why the LLN Isn't Enough" | — | `clt-overview.png` | — |
| 2. De Moivre–Laplace | "De Moivre–Laplace: The First CLT" | Theorem 1, Proof 1, Example 1, Remark 1 | `de-moivre-laplace.png` | `DeMoivreLaplaceExplorer` |
| 3. Classical CLT | "The Classical CLT (Lindeberg–Lévy)" | Theorem 2, Example 2, Remark 2 | — | — |
| 4. MGF Proof | "Proof via Moment-Generating Functions" | Proof 2, Example 3, Remark 3 | `clt-mgf-proof.png` | — |
| 5. CF Proof | "Proof via Characteristic Functions" | Definition 1, Theorem 3 (Lévy CF version), Proof 3, Remark 4 | `clt-cf-proof.png` | — |
| 6. Lindeberg CLT | "The Lindeberg CLT" | Definition 2, Theorem 4, Proof 4 (outline), Theorem 5 (Lyapunov), Corollary 1, Example 4, Remark 5 | `clt-lindeberg.png` | `LindebergExplorer` |
| 7. Berry–Esseen | "Berry–Esseen: How Fast Is Convergence?" | Theorem 6, Example 5, Remark 6 | `berry-esseen.png` | `BerryEsseenExplorer` |
| 8. Multivariate CLT | "The Multivariate CLT" | Theorem 7, Example 6, Remark 7 | `multivariate-clt.png` | — |
| 9. Delta Method | "The Delta Method Revisited" | Theorem 8 (restatement with CLT foundation), Example 7, Remark 8 | `delta-method.png` | `DeltaMethodCLTExplorer` |
| 10. ML Connections | "Connections to Machine Learning" | Example 8, Remark 9 | `ml-connections.png` | — |
| 11. Summary | "Summary" | Summary table | — | `CLTExplorer` |

---

## 3. Formal Elements

| Type | Number | Title |
|---|---|---|
| Definition | 1 | Characteristic function φ_X(t) = E[e^{itX}] |
| Definition | 2 | Lindeberg condition |
| Theorem | 1 | De Moivre–Laplace theorem (Binomial → Normal) |
| Theorem | 2 | Classical CLT (Lindeberg–Lévy): iid, finite variance |
| Theorem | 3 | Lévy's continuity theorem (characteristic function version) |
| Theorem | 4 | Lindeberg CLT (independent, non-identical, Lindeberg condition) |
| Theorem | 5 | Lyapunov CLT (sufficient condition for Lindeberg via third moments) |
| Theorem | 6 | Berry–Esseen theorem (convergence rate O(1/√n)) |
| Theorem | 7 | Multivariate CLT |
| Theorem | 8 | Delta method with CLT (restatement of Theorem 9.11 with CLT now proved) |
| Corollary | 1 | Lyapunov condition implies Lindeberg condition |
| Proof | 1 | De Moivre–Laplace (Stirling + limit of standardized Binomial PMF) |
| Proof | 2 | Classical CLT via MGFs (Taylor expand log M(t/σ√n), Lévy continuity) |
| Proof | 3 | Classical CLT via characteristic functions (Taylor expand log φ(t/σ√n)) |
| Proof | 4 | Lindeberg CLT (outline: Lindeberg replacement + Taylor expansion of CF) |
| Example | 1 | Binomial(100, 0.3): PMF vs Normal density, continuity correction |
| Example | 2 | CLT for different underlying distributions: Exponential, Uniform, Bernoulli, Poisson — convergence speed varies |
| Example | 3 | MGF computation for Exponential(1): explicit Taylor expansion of log M(t/√n) |
| Example | 4 | Lindeberg failure: one variable with variance growing faster than the sum (dominance) |
| Example | 5 | Berry–Esseen comparison: Uniform (symmetric, fast convergence) vs Exponential (skewed, slow convergence) |
| Example | 6 | Multivariate CLT for bivariate data: sample mean vector converging to N(μ, Σ/n) |
| Example | 7 | Delta method for log(X̄): confidence interval for log-mean via CLT + delta method |
| Example | 8 | ML applications: confidence intervals for accuracy, SGD noise characterization, Bayesian CLT preview |
| Remark | 1 | Historical note: de Moivre (1733), Laplace (1812), Lindeberg (1922), Lévy (1925), Berry/Esseen (1941/1942) |
| Remark | 2 | The CLT does not require identical distributions — only finite variance and the Lindeberg condition; the iid version is a special case |
| Remark | 3 | Connection to the Poisson limit proof (Topic 9, Theorem 9.13): same structure, different Taylor expansion |
| Remark | 4 | Characteristic functions vs MGFs: CFs always exist (Fourier transform of the distribution), MGFs may not (the integral may diverge). The CF proof is more general; the MGF proof is more intuitive. |
| Remark | 5 | The Lindeberg condition is *necessary and sufficient* (Feller's converse). The Lyapunov condition is sufficient but not necessary — it trades generality for ease of verification. |
| Remark | 6 | Berry–Esseen constant: the best known value is C ≤ 0.4748 (Shevtsova, 2011). The bound is tight for the Bernoulli case. |
| Remark | 7 | Cramér–Wold device: multivariate convergence in distribution reduces to univariate via projections |
| Remark | 8 | The delta method + CLT is the standard tool for constructing confidence intervals for nonlinear functions of parameters |
| Remark | 9 | Bernstein–von Mises theorem: the Bayesian analog of the CLT — the posterior converges to Normal at rate 1/√n regardless of the prior |

**Total: 2 definitions, 8 theorems, 1 corollary, 4 proofs, 8 examples, 9 remarks = 32 formal elements.**

---

## 4. Static Images

**Directory:** `public/images/topics/central-limit-theorem/`

Run the notebook to generate these figures:

| Filename | Notebook Section | Description |
|---|---|---|
| `clt-overview.png` | §1 (Cell 4) | 3-panel: (left) LLN says X̄ₙ → μ but not the shape, (center) CLT says the deviations are Gaussian — histogram of √n(X̄ₙ − μ)/σ vs N(0,1) density for different distributions, (right) convergence speed comparison across distributions |
| `de-moivre-laplace.png` | §2 (Cell 6) | 3-panel: (left) Bin(n, 0.3) PMF bars approaching Normal PDF as n = 10, 50, 200, (center) standardized Binomial CDF vs Φ(x), (right) continuity correction effect |
| `clt-mgf-proof.png` | §4 (Cell 8) | 3-panel: (left) log M(t/σ√n) Taylor expansion residual → 0 as n grows, (center) n · log M(t/σ√n) → t²/2 for different distributions, (right) MGF of √n(X̄ − μ)/σ approaching e^{t²/2} |
| `clt-cf-proof.png` | §5 (Cell 10) | 3-panel: (left) real and imaginary parts of CF for standardized sum, (center) |φₙ(t) − e^{−t²/2}| → 0, (right) CF vs MGF convergence comparison |
| `clt-lindeberg.png` | §6 (Cell 12) | 3-panel: (left) CLT works with mixed distributions when Lindeberg holds, (center) CLT fails when one variable dominates (Lindeberg violated), (right) Lindeberg truncation fraction plot |
| `berry-esseen.png` | §7 (Cell 14) | 3-panel: (left) sup|Fₙ(x) − Φ(x)| for Uniform vs Exponential, (center) convergence rate 1/√n with Berry–Esseen bound, (right) skewness ρ = E[|X|³]/σ³ determines the constant |
| `multivariate-clt.png` | §8 (Cell 16) | 3-panel: (left) bivariate sample means converging to point, (center) standardized sample means forming 2D Gaussian cloud, (right) Mahalanobis distance distribution approaching χ²(2) |
| `delta-method.png` | §9 (Cell 18) | 3-panel: (left) CLT for X̄ (Normal around μ), (center) delta method for log(X̄) (Normal around log(μ) with transformed variance), (right) variance comparison: [g'(μ)]²σ²/n vs direct simulation |
| `ml-connections.png` | §10 (Cell 20) | 3-panel: (left) confidence interval coverage simulation (proportion of CIs containing μ), (center) SGD gradient noise: histogram of mini-batch gradient fluctuations vs Gaussian, (right) Bayesian CLT: posterior width shrinking at 1/√n |

**To generate:**

```bash
cd notebooks/central-limit-theorem/
pip install numpy matplotlib scipy jupyter
jupyter nbconvert --to notebook --execute 11_central_limit_theorem.ipynb --output executed.ipynb
```

The notebook is seeded (`np.random.seed(42)`) for reproducibility.

---

## 5. Interactive Components

### Component 1: CLTExplorer

**File:** `src/components/viz/CLTExplorer.tsx`

**Purpose:** The flagship visualization. The reader selects an underlying distribution and watches the standardized sample mean √n(X̄ₙ − μ)/σ become Normal as n grows. This must go significantly beyond Topic 9's `ConvergenceInDistributionExplorer` CLT preview — more distributions, three simultaneous views (histogram, CDF, QQ plot), and convergence speed tracking.

**Interactions:**
- Dropdown for underlying distribution: Uniform(0,1), Exponential(1), Bernoulli(0.3), Poisson(5), Beta(2,5), Chi-squared(3), t(5), Gamma(2,1), Discrete Uniform({1,...,6}), Custom(user-specified PMF with ≤ 6 outcomes)
- Slider for n (sample size per average: 1 to 200, with snapping at 1, 2, 5, 10, 20, 50, 100, 200)
- Slider for M (number of replications: 500 to 10,000)
- Left panel: histogram of √n(X̄ₙ − μ)/σ from M replications, overlaid with N(0,1) PDF; KDE smoothed curve option
- Center panel: empirical CDF of the M standardized means vs Φ(x); KS statistic Dₙ annotated
- Right panel: QQ plot of standardized means vs N(0,1) quantiles; 45° reference line; departures from linearity indicate non-normality
- Bottom readout: current n, M, KS statistic, skewness of empirical distribution, kurtosis excess
- "Animate" button: smoothly increments n from 1 to 200, showing the distribution converging in real time
- "Compare" toggle: show two distributions side-by-side to compare convergence speeds

**Data:** No pre-computed data; all generated in-browser.

**Uses from `convergence.ts`:** `runningMean`, `sampleSequence`, `normalSample`, `exponentialSample`. New: `cltNormalization`.

**Uses from `distributions.ts`:** `pdfNormal`, `cdfStdNormal`.

### Component 2: DeMoivreLaplaceExplorer

**File:** `src/components/viz/DeMoivreLaplaceExplorer.tsx`

**Purpose:** The classical Binomial → Normal convergence. Binomial PMF bars approach the Normal PDF curve as n grows. This is the most iconic CLT visualization — the histogram becoming a bell curve.

**Interactions:**
- Slider for n (number of trials: 1 to 200)
- Slider for p (success probability: 0.05 to 0.95)
- Left panel: Binomial(n, p) PMF bars centered and scaled to the standardized axis (x − np)/(√(np(1−p))); N(0,1) PDF overlaid
- Right panel: Binomial CDF (step function) vs Φ((x − np)/√(np(1−p))); maximum difference highlighted
- Toggle: "Show continuity correction" — P(a ≤ X ≤ b) ≈ Φ((b + 0.5 − np)/σ) − Φ((a − 0.5 − np)/σ)
- Readout: n, p, np, √(np(1−p)), max |F_Bin − Φ|
- "Animate" button: smoothly increases n from 1 to 200

**Data:** No pre-computed data.

**Uses from `distributions.ts`:** `pdfNormal`, `cdfStdNormal`, `binomialPMF`, `binomialCDF`.

### Component 3: BerryEsseenExplorer

**File:** `src/components/viz/BerryEsseenExplorer.tsx`

**Purpose:** Visualize the convergence rate of the CLT. Show |Fₙ(x) − Φ(x)| vs the Berry–Esseen bound C·ρ/√n, where ρ = E[|X|³]/σ³ is the absolute third moment ratio. The key insight: skewed distributions converge more slowly.

**Interactions:**
- Dropdown for distribution: Uniform(0,1) (symmetric), Normal(0,1) (symmetric, baseline), Bernoulli(0.5) (symmetric), Exponential(1) (skewed), Chi-squared(1) (highly skewed), Bernoulli(0.1) (very skewed)
- Slider for n (1 to 500)
- Left panel: sup_x |Fₙ(x) − Φ(x)| plotted over n (from simulation of M = 5000 replications); Berry–Esseen upper bound C·ρ/√n overlaid (dashed)
- Right panel: |Fₙ(x) − Φ(x)| as a function of x for the current n; shows where the maximum deviation occurs (typically in the tails for skewed distributions)
- Compare mode: overlay two distributions to see how skewness affects convergence speed
- Readout: ρ (third moment ratio), Berry–Esseen bound, empirical sup deviation, ratio of empirical to bound

**Data:** No pre-computed data.

**Uses from `convergence.ts`:** `cltNormalization`, `sampleSequence`.
**Uses from `distributions.ts`:** `cdfStdNormal`.

### Component 4: LindebergExplorer

**File:** `src/components/viz/LindebergExplorer.tsx`

**Purpose:** Show that the CLT works for independent, non-identically distributed random variables when the Lindeberg condition holds, and fails when it doesn't. The key visual: one variable dominating the sum breaks the CLT.

**Interactions:**
- Preset configurations:
  - "Mixed distributions" — X₁ ~ N(0,1), X₂ ~ Exp(1)−1, X₃ ~ Uniform(−√3, √3), all with variance 1. Lindeberg holds → CLT works.
  - "Increasing variance" — Xᵢ ~ N(0, σᵢ²) with σᵢ² = i. Lindeberg holds (no single variable dominates) → CLT works.
  - "One dominant variable" — X₁ ~ N(0, n²), X₂,...,Xₙ ~ N(0, 1). X₁ dominates → Lindeberg fails → CLT fails.
  - "Custom" — user specifies k ≤ 5 distributions with parameters
- Slider for n (total number of summands: 5 to 200)
- Left panel: histogram of the standardized sum Sₙ/√(Var(Sₙ)) from M = 5000 replications; N(0,1) overlaid
- Center panel: Lindeberg fraction L(ε) = (1/sₙ²) Σ E[Xₖ² · 1(|Xₖ| > ε·sₙ)] plotted vs ε
- Right panel: variance share plot — bar chart showing Var(Xₖ)/Var(Sₙ) for each k; the Lindeberg condition essentially says no single bar dominates
- Readout: Lindeberg condition status (satisfied/violated), max variance share, KS statistic vs N(0,1)

**Data:** No pre-computed data.

**Uses from `convergence.ts`:** `cltNormalization`, `sampleSequence`, `normalSample`.
**Uses from `distributions.ts`:** `cdfStdNormal`.

### Component 5: DeltaMethodCLTExplorer

**File:** `src/components/viz/DeltaMethodCLTExplorer.tsx`

**Purpose:** Now that the CLT is proved, the delta method (Topic 9, Theorems 9.11–9.12) is fully rigorous. This component shows how nonlinear transformations of the sample mean inherit CLT normality with the variance scaled by [g'(μ)]².

**Interactions:**
- Dropdown for underlying distribution: Normal(5, 4), Exponential(1), Gamma(3, 1), Uniform(2, 8)
- Dropdown for transformation g: log(x), √x, 1/x, x², eˣ
- Slider for n (10 to 1000)
- Left panel: histogram of √n(X̄ₙ − μ) with N(0, σ²) overlay (the CLT)
- Center panel: histogram of √n(g(X̄ₙ) − g(μ)) with N(0, [g'(μ)]²σ²) overlay (the delta method)
- Right panel: asymptotic variance comparison — direct simulation variance of g(X̄ₙ) vs the delta method prediction [g'(μ)]²σ²/n, plotted over n
- Readout: μ, g(μ), g'(μ), σ², [g'(μ)]²σ², agreement between simulation and theory

**Data:** Reuses delta method transformation presets from `modes-of-convergence-data.ts`.

**Uses from `convergence.ts`:** `cltNormalization`, `runningMean`, `sampleSequence`.
**Uses from `distributions.ts`:** `pdfNormal`.

### Implementation Notes

- Same architecture as Topics 1–10: React + Tailwind, `client:visible` directive
- SVG/D3.js for all charts and visualizations
- Use CSS custom properties from `viz/shared/colorScales.ts`
- All components responsive (test at 375px width)
- No server-side computation — everything runs in the browser
- The `CLTExplorer` with M = 10,000 replications at n = 200 requires efficient generation — use typed arrays and avoid intermediate allocations
- The `LindebergExplorer` variance share plot should use stacked bars with color coding matching the distribution types
- The `DeltaMethodCLTExplorer` can share code with Topic 9's `DeltaMethodExplorer` — consider extracting shared hooks into `viz/shared/hooks/useDeltaMethod.ts`

---

## 6. Shared Utility Module

### Extensions to: `src/components/viz/shared/convergence.ts`

The JSDoc header in `convergence.ts` already promises `cltNormalization` for Topic 11. Add:

```typescript
// ── CLT Normalization ──────────────────────────────────────────────────

/**
 * Compute the CLT-standardized sample mean: √n(X̄ₙ − μ) / σ.
 * Given a set of samples, compute the sample mean, then standardize.
 * @param samples - array of iid samples
 * @param mu - true population mean
 * @param sigma - true population standard deviation (σ, not σ²)
 * @returns the standardized value √n(X̄ₙ − μ)/σ
 */
export function cltNormalization(
  samples: number[], mu: number, sigma: number
): number;

/**
 * Generate M replications of the CLT-standardized sample mean.
 * For each replication: draw n samples from the sampler, compute √n(X̄ − μ)/σ.
 * @param sampler - function that generates one sample (e.g., () => exponentialSample(1))
 * @param n - sample size per replication
 * @param M - number of replications
 * @param mu - true mean
 * @param sigma - true standard deviation
 * @param rng - optional seeded RNG
 * @returns an array of M standardized sample means
 */
export function cltReplications(
  sampler: (rng?: () => number) => number,
  n: number, M: number, mu: number, sigma: number,
  rng?: () => number
): number[];

// ── Characteristic Function Helpers ────────────────────────────────────

/**
 * Evaluate the empirical characteristic function at point t.
 * φ̂ₙ(t) = (1/n) Σ exp(i·t·Xⱼ).
 * Returns [real part, imaginary part].
 * @param samples - array of observations
 * @param t - evaluation point
 */
export function empiricalCF(
  samples: number[], t: number
): [number, number];

/**
 * Berry–Esseen bound: C · ρ / √n.
 * @param rho - E[|X − μ|³] / σ³ (absolute third moment ratio)
 * @param n - sample size
 * @param C - Berry–Esseen constant (default 0.4748, Shevtsova 2011)
 */
export function berryEsseenBound(
  rho: number, n: number, C?: number
): number;

// ── Additional Samplers ────────────────────────────────────────────────

/**
 * Generate a Poisson(λ) sample via Knuth's algorithm.
 * @param lambda - rate parameter
 * @param rng - optional seeded RNG
 */
export function poissonSample(
  lambda: number, rng?: () => number
): number;

/**
 * Generate a Chi-squared(k) sample as the sum of k squared Normals.
 * @param k - degrees of freedom
 * @param rng - optional seeded RNG
 */
export function chiSquaredSample(
  k: number, rng?: () => number
): number;

/**
 * Generate a Gamma(α, β) sample via rejection (Marsaglia–Tsang).
 * @param alpha - shape parameter
 * @param beta - rate parameter (1/scale)
 * @param rng - optional seeded RNG
 */
export function gammaSample(
  alpha: number, beta: number, rng?: () => number
): number;

/**
 * Generate a Beta(a, b) sample via Gamma ratio.
 * @param a - first shape parameter
 * @param b - second shape parameter
 * @param rng - optional seeded RNG
 */
export function betaSample(
  a: number, b: number, rng?: () => number
): number;
```

**Design contract:** Same as Topics 9–10 — all functions pure (except samplers using RNG), documented with JSDoc, no side effects. The `cltReplications` function is the workhorse for the CLTExplorer and BerryEsseenExplorer — it must be efficient for M = 10,000 replications. Topics 12+ will be further extended with `confidenceIntervalCoverage` and additional concentration-bound functions.

---

## 7. Topic Data Module

### New Module: `src/data/central-limit-theorem-data.ts`

```typescript
/** Distribution presets for CLTExplorer. */
export const cltDistributions = [
  {
    id: "uniform",
    name: "Uniform(0, 1)",
    mu: 0.5,
    sigma: Math.sqrt(1/12),
    skewness: 0,
    thirdMomentRatio: 1.8, // E[|X−μ|³]/σ³ for Uniform
    description: "Symmetric, bounded. Fast convergence — the CLT works well even at n = 5.",
  },
  {
    id: "exponential",
    name: "Exponential(1)",
    mu: 1,
    sigma: 1,
    skewness: 2,
    thirdMomentRatio: 6, // E[|X−μ|³]/σ³ for Exp(1)
    description: "Skewed right. Moderate convergence speed — visible non-normality at n = 10, good by n = 50.",
  },
  {
    id: "bernoulli",
    name: "Bernoulli(0.3)",
    mu: 0.3,
    sigma: Math.sqrt(0.21),
    skewness: 0.873, // (1−2p)/√(p(1−p))
    thirdMomentRatio: 2.34,
    description: "Discrete and asymmetric. The de Moivre–Laplace theorem is this CLT applied to sums of Bernoullis.",
  },
  {
    id: "poisson",
    name: "Poisson(5)",
    mu: 5,
    sigma: Math.sqrt(5),
    skewness: 1 / Math.sqrt(5),
    thirdMomentRatio: 1.45,
    description: "Discrete count data. Moderate skewness — converges reasonably fast.",
  },
  {
    id: "beta",
    name: "Beta(2, 5)",
    mu: 2/7,
    sigma: Math.sqrt(10 / (49 * 8)),
    skewness: 0.596,
    thirdMomentRatio: 1.9,
    description: "Bounded, right-skewed. Common in Bayesian posterior analysis.",
  },
  {
    id: "chi-squared",
    name: "Chi-squared(3)",
    mu: 3,
    sigma: Math.sqrt(6),
    skewness: Math.sqrt(8/3),
    thirdMomentRatio: 4.35,
    description: "Highly skewed right. Slow convergence — needs n ≈ 100 for good Normal approximation.",
  },
  {
    id: "t5",
    name: "t(5)",
    mu: 0,
    sigma: Math.sqrt(5/3),
    skewness: 0,
    thirdMomentRatio: 6.0, // heavy tails despite symmetry
    description: "Symmetric but heavy-tailed. Symmetric distributions converge faster (skewness = 0), but the heavy tails inflate the Berry–Esseen constant.",
  },
  {
    id: "gamma",
    name: "Gamma(2, 1)",
    mu: 2,
    sigma: Math.sqrt(2),
    skewness: Math.sqrt(2),
    thirdMomentRatio: 3.77,
    description: "Right-skewed. The shape parameter α controls skewness: larger α → more symmetric → faster CLT convergence.",
  },
  {
    id: "dice",
    name: "Fair Die {1,...,6}",
    mu: 3.5,
    sigma: Math.sqrt(35/12),
    skewness: 0,
    thirdMomentRatio: 1.64,
    description: "The most familiar random variable. Symmetric, so convergence is fast — rolling 30 dice gives a remarkably Normal sum.",
  },
];

/** Lindeberg condition presets for LindebergExplorer. */
export const lindebergPresets = [
  {
    id: "mixed-equal-var",
    name: "Mixed Distributions (equal variance)",
    description: "X₁ ~ N(0,1), X₂ ~ Exp(1)−1, X₃ ~ Uniform(−√3, √3). All variance 1. Lindeberg holds.",
    lindebergHolds: true,
    distributions: [
      { name: "Normal(0,1)", variance: 1 },
      { name: "Exponential(1)−1", variance: 1 },
      { name: "Uniform(−√3, √3)", variance: 1 },
    ],
  },
  {
    id: "increasing-var",
    name: "Increasing Variance",
    description: "Xᵢ ~ N(0, i). Variance grows linearly. Lindeberg holds: max σᵢ²/sₙ² = n/(n(n+1)/2) → 0.",
    lindebergHolds: true,
    distributions: "dynamic", // generated based on n
  },
  {
    id: "one-dominant",
    name: "One Dominant Variable",
    description: "X₁ ~ N(0, n²), X₂,...,Xₙ ~ N(0, 1). X₁ has variance n² while the rest have variance 1. Var(X₁)/sₙ² = n²/(n²+n−1) → 1. Lindeberg fails.",
    lindebergHolds: false,
    distributions: "dynamic",
  },
  {
    id: "bernoulli-varying-p",
    name: "Bernoulli with Varying p",
    description: "Xᵢ ~ Bernoulli(pᵢ) with pᵢ = 1/(i+1). Different success probabilities. Lindeberg holds as long as no single pᵢ(1−pᵢ) dominates.",
    lindebergHolds: true,
    distributions: "dynamic",
  },
];

/** Berry–Esseen comparison presets. */
export const berryEsseenPresets = [
  {
    id: "uniform",
    name: "Uniform(0,1)",
    rho: 1.8,
    skewness: 0,
    convergenceSpeed: "fast",
    color: "#059669",
  },
  {
    id: "normal",
    name: "Normal(0,1)",
    rho: 2.0,
    skewness: 0,
    convergenceSpeed: "fast",
    color: "#2563eb",
  },
  {
    id: "bernoulli-half",
    name: "Bernoulli(0.5)",
    rho: 1.0,
    skewness: 0,
    convergenceSpeed: "fast",
    color: "#7c3aed",
  },
  {
    id: "exponential",
    name: "Exponential(1)",
    rho: 6.0,
    skewness: 2,
    convergenceSpeed: "slow",
    color: "#d97706",
  },
  {
    id: "chi-squared-1",
    name: "Chi-squared(1)",
    rho: 8.0,
    skewness: 2 * Math.sqrt(2),
    convergenceSpeed: "very slow",
    color: "#dc2626",
  },
  {
    id: "bernoulli-01",
    name: "Bernoulli(0.1)",
    rho: 3.16,
    skewness: 2.67,
    convergenceSpeed: "slow",
    color: "#ea580c",
  },
];

/** Theorem progression for the summary diagram. */
export const cltTheorems = [
  {
    id: "de-moivre-laplace",
    name: "De Moivre–Laplace",
    number: 1,
    assumption: "Sₙ ~ Bin(n, p)",
    conclusion: "(Sₙ − np)/√(np(1−p)) →d N(0,1)",
    proofTool: "Stirling's approximation",
    year: 1733,
  },
  {
    id: "clt-lindeberg-levy",
    name: "CLT (Lindeberg–Lévy)",
    number: 2,
    assumption: "iid, σ² < ∞",
    conclusion: "√n(X̄ₙ − μ)/σ →d N(0,1)",
    proofTool: "MGF Taylor expansion + Lévy continuity",
    year: 1922,
  },
  {
    id: "levy-cf",
    name: "Lévy Continuity (CF version)",
    number: 3,
    assumption: "φₙ(t) → φ(t) pointwise, φ continuous at 0",
    conclusion: "Xₙ →d X",
    proofTool: "Characteristic functions",
    year: 1925,
  },
  {
    id: "lindeberg-clt",
    name: "Lindeberg CLT",
    number: 4,
    assumption: "independent, Lindeberg condition",
    conclusion: "Sₙ/sₙ →d N(0,1)",
    proofTool: "Lindeberg replacement + CF Taylor",
    year: 1922,
  },
  {
    id: "lyapunov-clt",
    name: "Lyapunov CLT",
    number: 5,
    assumption: "independent, Σ E[|Xₖ|^{2+δ}]/sₙ^{2+δ} → 0",
    conclusion: "Sₙ/sₙ →d N(0,1)",
    proofTool: "Lyapunov condition ⟹ Lindeberg",
    year: 1901,
  },
  {
    id: "berry-esseen",
    name: "Berry–Esseen",
    number: 6,
    assumption: "iid, E[|X|³] < ∞",
    conclusion: "sup|Fₙ(x) − Φ(x)| ≤ Cρ/√n",
    proofTool: "Fourier analysis of CFs",
    year: 1941,
  },
  {
    id: "multivariate-clt",
    name: "Multivariate CLT",
    number: 7,
    assumption: "iid vectors, Σ finite",
    conclusion: "√n(X̄ₙ − μ) →d N(0, Σ)",
    proofTool: "Cramér–Wold + univariate CLT",
    year: "—",
  },
  {
    id: "delta-method-clt",
    name: "Delta Method (CLT version)",
    number: 8,
    assumption: "√n(X̄ₙ − μ) →d N(0,σ²), g differentiable",
    conclusion: "√n(g(X̄ₙ) − g(μ)) →d N(0, [g'(μ)]²σ²)",
    proofTool: "Taylor + Slutsky",
    year: "—",
  },
];
```

---

## 8. Curriculum Graph & Site Metadata Updates

### Update: `src/data/curriculum-graph.json`

Change `central-limit-theorem` from `"planned"` to `"published"`. Add the prerequisite edges:

```json
{
  "id": "central-limit-theorem",
  "status": "published",
  "prerequisites": ["modes-of-convergence", "law-of-large-numbers"]
}
```

### Update: `src/data/curriculum.ts`

Mark Track 3, Topic 3 as published. Track 3 now has 3 of 4 topics published.

### Update: `modes-of-convergence.mdx`

Replace "(coming soon)" references to the CLT:

```mdx
<!-- Before: line 576 -->
This is exactly the technique we'll use to prove the Poisson limit theorem (Section 9.9) and the Central Limit Theorem (Topic 11, coming soon).

<!-- After -->
This is exactly the technique we'll use to prove the Poisson limit theorem (Section 9.9) and the [Central Limit Theorem](/topics/central-limit-theorem).
```

```mdx
<!-- Before: line 578 -->
The characteristic function version avoids this existence requirement, which is why the full CLT proof uses characteristic functions.

<!-- After -->
The characteristic function version avoids this existence requirement, which is why the full [CLT proof](/topics/central-limit-theorem) uses characteristic functions for the most general version.
```

```mdx
<!-- Before: line 961 -->
By the CLT (which we're previewing — the full proof is Topic 11, coming soon):

<!-- After -->
By the [CLT](/topics/central-limit-theorem) (proved in full in Topic 11):
```

```mdx
<!-- Before: line 1070 -->
This proof serves as a prototype for how the Central Limit Theorem will be proved in Topic 11 (coming soon). The technique is the same: compute the MGF of the sequence, show it converges pointwise to the MGF of the target distribution, and invoke Levy's continuity theorem. The CLT proof is more involved because the MGF of √n(X̄ₙ − μ)/σ requires a careful Taylor expansion of log M_{X₁}(t/(σ√n)), but the structure is identical.

<!-- After -->
This proof is the prototype for the [Central Limit Theorem proof](/topics/central-limit-theorem). The technique is identical: compute the MGF, show pointwise convergence to the target MGF, and invoke Lévy's continuity theorem. The CLT proof is more involved because the MGF of √n(X̄ₙ − μ)/σ requires a careful Taylor expansion of log M_{X₁}(t/(σ√n)), but the structure established here carries over directly.
```

```mdx
<!-- Before: line 1125 -->
The Berry-Esseen theorem gives the rate for convergence in distribution in the CLT: sup_x |F_{Z̄ₙ}(x) − Φ(x)| ≤ C · E[|X₁|³] / (σ³√n), where C ≤ 0.4748.

<!-- After -->
The [Berry–Esseen theorem](/topics/central-limit-theorem) gives the rate for convergence in distribution in the CLT: sup_x |F_{Z̄ₙ}(x) − Φ(x)| ≤ C · E[|X₁|³] / (σ³√n), where C ≤ 0.4748.
```

```mdx
<!-- Before: line 1201 -->
- **Central Limit Theorem** (Topic 11, coming soon): The CLT proves √n(X̄ₙ − μ)/σ →d N(0,1) (convergence in distribution). The proof technique is MGF convergence—the same one we used for the Poisson limit theorem in Section 9.9.

<!-- After -->
- [Central Limit Theorem](/topics/central-limit-theorem): The CLT proves $\sqrt{n}(\bar{X}_n - \mu)/\sigma \xrightarrow{d} N(0,1)$ (convergence in distribution). The proof via MGF convergence follows the same technique we used for the Poisson limit theorem in Section 9.9. The Lindeberg CLT extends the result to non-identical summands, and the Berry–Esseen theorem quantifies the convergence rate.
```

### Update: `law-of-large-numbers.mdx`

Replace "(coming soon)" references to the CLT:

```mdx
<!-- Before: line 1031 (approx, in the "What Comes Next" section) -->
The Central Limit Theorem (Topic 11, coming soon) answers this: √n(X̄ₙ − μ)/σ →d N(0,1).

<!-- After -->
The [Central Limit Theorem](/topics/central-limit-theorem) answers this: $\sqrt{n}(\bar{X}_n - \mu)/\sigma \xrightarrow{d} N(0,1)$.
```

```mdx
<!-- Before: in Remark 5, line 1181 (approx) -->
**Central Limit Theorem** (Topic 11, coming soon): √n(X̄ₙ − μ)/σ →d N(0, 1).

<!-- After -->
[**Central Limit Theorem**](/topics/central-limit-theorem) (Topic 11): $\sqrt{n}(\bar{X}_n - \mu)/\sigma \xrightarrow{d} N(0,1)$.
```

*(Apply to all instances of "Topic 11, coming soon" in law-of-large-numbers.mdx — there are approximately 4–5 instances.)*

### Update: `expectation-moments.mdx`

Replace "(coming soon)" reference:

```mdx
<!-- Before: line 763 -->
We'll use this in the proof of the Central Limit Theorem (Modes of Convergence + CLT, coming soon).

<!-- After -->
We'll use this in the proof of the [Central Limit Theorem](/topics/central-limit-theorem) — MGF uniqueness is the final step that identifies the limiting distribution as N(0,1).
```

### Update: `discrete-distributions.mdx`

Replace "(coming soon)" reference:

```mdx
<!-- Before: line 302 -->
By the Central Limit Theorem (see [Modes of Convergence](/topics/modes-of-convergence) for the framework and [Central Limit Theorem](/topics/central-limit-theorem) for the full proof), an approximate 95% confidence interval is

<!-- After -->
By the [Central Limit Theorem](/topics/central-limit-theorem), an approximate 95% confidence interval is
```

### Update: References Spreadsheet

Add the following new references with hyperlinks:

| ID | Author | Title | Year | Used In |
|---|---|---|---|---|
| FEL1971 | William Feller | An Introduction to Probability Theory and Its Applications, Vol. 2 | 1971 | central-limit-theorem |

Update existing entries to include `central-limit-theorem` in the "Used In Topics" column:
- DUR2019 (Durrett) — add `central-limit-theorem`
- BIL2012 (Billingsley) — add `central-limit-theorem`
- CAS2002 (Casella & Berger) — add `central-limit-theorem`
- WAS2004 (Wasserman) — add `central-limit-theorem`
- BIS2006 (Bishop) — add `central-limit-theorem`

---

## 9. Verification Checklist

### Content

- [ ] `central-limit-theorem.mdx` renders at `/topics/central-limit-theorem` with no build errors
- [ ] All 9 static images present in `public/images/topics/central-limit-theorem/`
- [ ] KaTeX renders correctly: all display equations, inline math, and theorem blocks
- [ ] All 2 definitions, 8 theorems, 1 corollary, 4 proofs, 8 examples, 9 remarks render in styled blocks
- [ ] All 4 proofs expand fully with no hand-waving
- [ ] Internal links to `modes-of-convergence` and `law-of-large-numbers` work (prerequisites)
- [ ] Internal links to `expectation-moments`, `discrete-distributions`, `continuous-distributions`, `multivariate-distributions` work (referenced topics)
- [ ] formalcalculus.com links (`sequences-limits`, `series-convergence`, `differentiation`) open in new tab with external badge
- [ ] formalml.com links (`stochastic-gradient-descent`, `generalization-bounds`, `bayesian-neural-networks`, `monte-carlo-methods`) open in new tab with forward-reference badge
- [ ] Forward references to Topic 12 (`large-deviations`) and later topics (`confidence-intervals`, `hypothesis-testing`, `empirical-processes`) use plain text + "(coming soon)"
- [ ] `modes-of-convergence.mdx` updated: six "(coming soon)" references to CLT replaced with live links
- [ ] `law-of-large-numbers.mdx` updated: four–five "(coming soon)" references to CLT replaced with live links
- [ ] `expectation-moments.mdx` updated: "(coming soon)" reference to CLT replaced with live link
- [ ] `discrete-distributions.mdx` updated: CLT reference updated with live link
- [ ] References spreadsheet updated with Feller (1971) and existing references updated with `central-limit-theorem` in "Used In" column

### Interactive Components

- [ ] `CLTExplorer` all 9+ distribution presets generate correct convergence behavior; n slider from 1 to 200 without lag; histogram, CDF, and QQ panels update consistently; animate mode works smoothly; compare toggle works
- [ ] `DeMoivreLaplaceExplorer` Binomial PMF bars align with Normal PDF as n grows; continuity correction toggle visibly improves approximation; p slider updates all panels; animate mode works
- [ ] `BerryEsseenExplorer` Berry–Esseen bound is an upper bound on the empirical sup deviation for all presets; skewed distributions show visibly slower convergence than symmetric; compare mode works
- [ ] `LindebergExplorer` "Mixed distributions" and "Increasing variance" presets show CLT convergence; "One dominant variable" preset shows CLT failure; Lindeberg fraction plot correctly identifies condition status; variance share bars render correctly
- [ ] `DeltaMethodCLTExplorer` delta method variance prediction matches simulation for all transformation/distribution combinations; handles g'(μ) = 0 case correctly (second-order); n slider shows improving agreement between simulation and theory

### Infrastructure

- [ ] `convergence.ts` extensions compile with no TypeScript errors
- [ ] `central-limit-theorem-data.ts` compiles
- [ ] Track 3 shows 3/4 topics published in curriculum UI
- [ ] Pagefind indexes the topic on rebuild
- [ ] `pnpm build` succeeds with zero errors
- [ ] Site deploys to Vercel successfully
- [ ] Mobile responsive (all 5 interactive components at 375px width)
- [ ] "Intermediate" difficulty badge styled correctly

---

## 10. Build Order

1. **Extend `src/components/viz/shared/convergence.ts`** with `cltNormalization`, `cltReplications`, `empiricalCF`, `berryEsseenBound`, `poissonSample`, `chiSquaredSample`, `gammaSample`, `betaSample`. Write console log tests: `cltNormalization` on 100 N(0,1) samples should give a value near 0; `cltReplications` with Exponential(1) at n = 100 should produce approximately N(0,1)-distributed values; `berryEsseenBound` should decrease as n grows; `empiricalCF` on N(0,1) samples should have real part ≈ e^{−t²/2} and imaginary part ≈ 0.
2. **Create `central-limit-theorem-data.ts`** with distribution presets, Lindeberg presets, Berry–Esseen presets, and theorem progression.
3. **Copy notebook figures** to `public/images/topics/central-limit-theorem/`.
4. **Create `central-limit-theorem.mdx`** with full frontmatter and all markdown/LaTeX content. Include all formal element blocks (definitions, theorems, proofs, examples, remarks). No interactive components yet.
5. **Build `CLTExplorer.tsx`** — the flagship: multi-distribution CLT convergence with histogram + CDF + QQ plot.
6. **Build `DeMoivreLaplaceExplorer.tsx`** — Binomial PMF bars approaching Normal PDF with continuity correction.
7. **Build `BerryEsseenExplorer.tsx`** — convergence rate comparison with Berry–Esseen bound.
8. **Build `LindebergExplorer.tsx`** — Lindeberg condition: CLT success vs failure with variance share visualization.
9. **Build `DeltaMethodCLTExplorer.tsx`** — delta method with CLT foundation; can share hooks with Topic 9's DeltaMethodExplorer.
10. **Embed all 5 components** in the MDX with `client:visible`.
11. **Update `curriculum-graph.json`** and `curriculum.ts` — mark `central-limit-theorem` as published; Track 3 has 3/4.
12. **Update `modes-of-convergence.mdx`** — replace six "(coming soon)" references to CLT with live links.
13. **Update `law-of-large-numbers.mdx`** — replace four–five "(coming soon)" references to CLT with live links.
14. **Update `expectation-moments.mdx`** — replace "(coming soon)" reference to CLT with live link.
15. **Update `discrete-distributions.mdx`** — update CLT reference with live link.
16. **Update the references spreadsheet** — add Feller (1971), update existing references with `central-limit-theorem` in "Used In" column.
17. Run the full verification checklist (§9).
18. `pnpm build` — verify zero errors.
19. Deploy to Vercel.

---

## Appendix A: KaTeX Constraints

Same as Topics 1–10:

- **No `\begin{aligned}` blocks with `&` markers.** Multi-line derivations use separate `$$...$$` blocks with prose glue between lines. This topic has the MGF proof (multi-step Taylor expansion) and the CF proof — both require chains of equalities. Use separate display blocks with prose connectors like "Substituting:" and "Therefore:" between them.
- **No `\begin{array}{c|cccc}` tables.** Use HTML tables or markdown tables for tabular content. The summary table at the end and the theorem comparison table should use markdown tables.
- Inline math uses `$...$`. Display math uses `$$...$$` on its own line.
- Test all LaTeX rendering in the dev server before committing.
- **Notation:** Use `\xrightarrow{d}` for convergence in distribution. If KaTeX does not support `\xrightarrow{d}`, use `\overset{d}{\to}`. The notation must match the conventions established in Topics 9–10.
- **Complex number notation:** The CF proof uses $i = \sqrt{-1}$. Use `\mathrm{i}` or just `i` consistently. The exponential $e^{itx}$ should render correctly in KaTeX.
- **Characteristic function:** Use `\varphi` for the CF: $\varphi_X(t) = E[e^{itX}]$. This distinguishes it from the Normal CDF $\Phi(x)$.

---

## Appendix B: Design Decisions

1. **De Moivre–Laplace before the general CLT.** The historical order is pedagogically superior: the reader sees the specific case (Binomial → Normal) before the general result. This makes the general CLT feel like a natural extension rather than an abstract theorem. It also fulfills the promise from `discrete-distributions.mdx:302` immediately.

2. **Two proofs: MGF first, then CF.** Topic 9 established the MGF proof technique with the Poisson limit theorem and explicitly promised: "the CLT proof follows the same structure." The MGF proof is more intuitive (real-valued functions, Taylor expansion is a calculus exercise) and fulfills the binding promise. The CF proof is included because: (a) Topic 9, line 578, explicitly promised it, (b) it's the more general version (CFs always exist), and (c) it's the proof used in graduate probability courses. The MGF proof goes first because the reader has seen the technique before.

3. **Lindeberg CLT, not just Lindeberg–Lévy.** The iid version is the starting point, but the Lindeberg version reveals the *necessary and sufficient* condition for the CLT to hold. This is pedagogically important because it shows *when the CLT fails* — which is as important as knowing when it works. The Lindeberg condition (no single variable dominates the sum) is also the key insight for understanding why mini-batch SGD noise is approximately Gaussian (each sample's gradient contribution is small relative to the batch total).

4. **Berry–Esseen: statement + visualization, not proof.** The Berry–Esseen proof is a technical tour de force using Fourier analysis of characteristic functions — it's beyond our scope. But the *statement* and the *visualization* are essential: they quantify how fast the CLT kicks in and reveal that skewness is the primary factor slowing convergence. This is practically important for sample-size calculations.

5. **Five interactive components — matching Topics 9–10.** The CLTExplorer is the flagship and goes in the Summary section (bottom of the page) rather than §3 because: (a) the reader has already seen the CLT preview in Topic 9, so placing it up front would feel repetitive, and (b) the summary position lets the reader experiment with the CLT after understanding the theory, proofs, and limitations. The DeMoivreLaplaceExplorer, BerryEsseenExplorer, LindebergExplorer, and DeltaMethodCLTExplorer each serve a specific section.

6. **Delta method revisited, not re-proved.** Topic 9 proved the delta method (Theorems 9.11–9.12). Topic 11 restates it as Theorem 11.8 with the CLT now providing the rigorous foundation for the "√n(X̄ₙ − μ) →d N(0, σ²)" hypothesis. The delta method section focuses on *applications* (confidence intervals for nonlinear functions) rather than proving the result.

7. **Multivariate CLT stated, not proved.** The multivariate CLT follows from the Cramér–Wold device (reduce to univariate via projections) and the univariate CLT. The Cramér–Wold reduction is explained, but the full proof is a technicality. The statement connects to Topic 8 (Multivariate Distributions) and sets up the Bernstein–von Mises theorem reference in ML Connections.

8. **The CLTExplorer goes deeper than Topic 9's preview.** The CLT preview in `ConvergenceInDistributionExplorer` has 3 distributions and shows histogram + CDF. The `CLTExplorer` has 9+ distributions, shows histograms, CDFs, and QQ plots, tracks convergence speed via the KS statistic, and supports side-by-side comparisons. The QQ plot is the key addition — it reveals non-normality more clearly than histograms.

---

*Brief version: v1 | Created: 2026-04-13 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/central-limit-theorem/11_central_limit_theorem.ipynb`*
