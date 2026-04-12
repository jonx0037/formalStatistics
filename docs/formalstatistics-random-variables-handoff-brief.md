# Claude Code Handoff Brief: Random Variables & Distribution Functions

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)  
**Repo:** `github.com/jonx0037/formalStatistics`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/random-variables/03_random_variables_distribution_functions.ipynb`

---

## Important: Third Topic — Infrastructure Exists

This is **topic 3 of 32** and builds directly on Topics 1–2. The site scaffolding, layout templates, shared utility modules, and curriculum graph already exist. This topic requires:

1. A new MDX file, five new interactive components, a new shared module (`distributions.ts`), a new data module, and extensions to the curriculum graph.
2. Updates to `curriculum-graph.json` and `curriculum.ts` to mark this topic as published.
3. Cross-references back to `sample-spaces` and `conditional-probability` as internal prerequisites, and forward to Topics 4+ as "(coming soon)."
4. **Update the references spreadsheet** with hyperlinks for all references cited in this topic.

---

## 1. Objective

Add a new topic page **"Random Variables & Distribution Functions"** as the **third topic in the Foundations of Probability track** on formalstatistics.com.

1. This is **topic 3 of 32**.
2. **Internal prerequisites:**
   - `sample-spaces` — the probability space (Ω, F, P), sigma-algebras (measurability), and the Kolmogorov axioms
   - `conditional-probability` — conditional probability P(A|B), the multiplication rule, the chain rule, and independence of events
3. **External prerequisites from formalcalculus.com:**
   - `sequences-limits` — CDF limit properties ($\lim_{x \to -\infty} F(x) = 0$, $\lim_{x \to \infty} F(x) = 1$) and continuity of probability (used in the CDF proof)
   - `change-of-variables` — the Jacobian and the change of variables formula for transformations of random variables (Theorem 7)
4. **Downstream within formalStatistics:**
   - `expectation-variance` (direct) — expectation is integration against the distribution; variance is the second central moment. Both require the PMF/PDF/CDF framework defined here.
   - `discrete-distributions` (direct) — Bernoulli, Binomial, Poisson, Geometric, Negative Binomial are specific instances of discrete random variables with named PMFs.
   - `continuous-distributions` (direct) — Normal, Exponential, Gamma, Beta, Uniform are specific instances of continuous random variables with named PDFs.
   - `exponential-families` (indirect) — exponential family parameterization uses the PDF/PMF as the starting point.
   - `bayesian-foundations` (indirect) — posterior computation requires conditional distributions and Bayes' theorem for densities.
5. **Forward links to formalml.com:**
   - `information-geometry` — Fisher information is defined via the score function, which is the derivative of the log-likelihood — a function of the PDF/PMF.
   - `normalizing-flows` — normalizing flows learn invertible transformations of simple distributions; the change of variables formula (Theorem 7) is the mathematical backbone.
   - `variational-inference` — variational methods approximate posterior distributions; the KL divergence between densities requires the conditional PDF framework.
6. This topic **creates** a new shared utility module `distributions.ts` at `src/components/viz/shared/distributions.ts`. This module provides PMF/PDF/CDF evaluation functions and will be extended by Topics 4–8.

**Content scope:**

- Random variables as measurable functions: definition, motivation, examples
- Random vectors
- Discrete random variables: PMFs, support, Bernoulli and Binomial previews
- Continuous random variables: PDFs, the density-is-not-a-probability clarification, Uniform and Normal examples
- The CDF as the universal descriptor: definition, properties (non-decreasing, right-continuous, limits), connecting PDF/CDF via FTC
- Joint distributions: joint PMF, joint PDF, joint CDF
- Marginal distributions: summing/integrating out
- Conditional distributions: conditional PMF, conditional PDF, chain rule for densities, Bayes' theorem for densities
- Independence of random variables: factorization characterization
- Transformations: CDF method, change of variables formula (Jacobian), inverse CDF transform
- Expectation preview (definitions only, no proofs — those belong to Topic 4)
- ML connections: feature vectors as random vectors, generative vs discriminative as joint vs conditional, softmax as PMF, loss as transformation, inverse CDF transform/normalizing flows

---

## 2. MDX File

### Location

```
src/content/topics/random-variables.mdx
```

The entry `id` will be `random-variables`. The dynamic route resolves to `/topics/random-variables`.

### Frontmatter

```yaml
---
title: "Random Variables & Distribution Functions"
subtitle: "The bridge from events to numbers — PMFs, PDFs, CDFs, and the distribution machinery that makes statistical computation possible"
status: "published"
difficulty: "foundational"
prerequisites:
  - "sample-spaces"
  - "conditional-probability"
tags:
  - "probability"
  - "random-variables"
  - "pmf"
  - "pdf"
  - "cdf"
  - "joint-distribution"
  - "marginal-distribution"
  - "conditional-distribution"
  - "transformation"
  - "measurability"
domain: "foundations-of-probability"
videoId: null
notebookPath: "notebooks/random-variables/03_random_variables_distribution_functions.ipynb"
githubUrl: "https://github.com/jonx0037/formalStatistics/blob/main/src/content/topics/random-variables.mdx"
datePublished: 2026-04-11
estimatedReadTime: 50
abstract: "A random variable X : Ω → ℝ is a measurable function that translates abstract outcomes into numbers, and once we have numbers, we can compute. Discrete random variables are described by probability mass functions p_X(x) = P(X = x), where each value is a genuine probability. Continuous random variables are described by probability density functions f_X(x), where the density is NOT a probability — only integrals of the density give probabilities. The cumulative distribution function F_X(x) = P(X ≤ x) is a universal descriptor that applies to every random variable. Joint distributions f_{X,Y}(x,y) capture the full dependence between random variables; marginals are obtained by integrating out, and conditional distributions f_{X|Y}(x|y) = f_{X,Y}(x,y)/f_Y(y) extend conditional probability from events to random variables. Independence means the joint factors: f_{X,Y} = f_X · f_Y. For transformations Y = g(X), the CDF method and the change of variables formula with its Jacobian determine the new distribution — machinery that powers normalizing flows in generative modeling. The inverse CDF transform shows that Uniform(0,1) random variables can generate samples from any distribution."
formalcalculusPrereqs:
  - topic: "sequences-limits"
    site: "formalcalculus"
    relationship: "CDF properties (Theorem 1) use limits at infinity and continuity of probability from Topic 1. The proof that F(x) → 0 as x → -∞ and F(x) → 1 as x → ∞ requires the convergence framework from formalCalculus."
  - topic: "change-of-variables"
    site: "formalcalculus"
    relationship: "The change of variables formula for PDFs (Theorem 7) uses the Jacobian and the substitution rule for integrals. The multidimensional version uses the Jacobian determinant."
formalmlConnections:
  - topic: "normalizing-flows"
    site: "formalml"
    relationship: "Normalizing flows learn invertible transformations of simple distributions. The change of variables formula f_Y(y) = f_X(g⁻¹(y)) · |det J_{g⁻¹}(y)| is the mathematical backbone — each flow layer applies this formula."
  - topic: "information-geometry"
    site: "formalml"
    relationship: "Fisher information is defined via the score function ∂/∂θ log f(x; θ), which requires the PDF parameterization. The statistical manifold is the space of distributions."
  - topic: "variational-inference"
    site: "formalml"
    relationship: "Variational inference approximates posterior distributions by minimizing KL divergence between density functions. The conditional PDF framework enables the ELBO decomposition."
connections:
  - topic: "sample-spaces"
    relationship: "Random variables are measurable functions on the probability space (Ω, F, P) from Topic 1. The sigma-algebra F determines which functions qualify as random variables."
  - topic: "conditional-probability"
    relationship: "Conditional distributions f_{X|Y}(x|y) = f_{X,Y}(x,y)/f_Y(y) extend conditional probability P(A|B) = P(A∩B)/P(B) from events to random variables. The chain rule for densities mirrors the multiplication rule."
  - topic: "expectation-variance"
    relationship: "Expectation is integration against the distribution: E[X] = ∫ x f_X(x) dx (continuous) or Σ x p_X(x) (discrete). The PMF/PDF/CDF framework defined here is the prerequisite."
  - topic: "discrete-distributions"
    relationship: "Bernoulli, Binomial, Poisson, etc., are specific named PMFs. This topic defines what a PMF is; that topic catalogs the important ones."
  - topic: "continuous-distributions"
    relationship: "Normal, Exponential, Gamma, Beta, etc., are specific named PDFs. This topic defines what a PDF is; that topic catalogs the important ones."
references:
  - type: "book"
    title: "Probability and Measure"
    author: "Patrick Billingsley"
    year: 2012
    edition: "Anniversary Edition"
    publisher: "Wiley"
    isbn: "978-1-118-12237-0"
    url: "https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372"
    note: "Chapter 5 covers random variables and distribution functions with full measure-theoretic rigor."
  - type: "book"
    title: "Probability: Theory and Examples"
    author: "Rick Durrett"
    year: 2019
    edition: "5th"
    publisher: "Cambridge University Press"
    isbn: "978-1-108-47368-2"
    url: "https://services.math.duke.edu/~rtd/PTE/pte.html"
    note: "Chapter 1 on distribution functions and random variables."
  - type: "book"
    title: "Probability and Random Processes"
    author: "Geoffrey Grimmett & David Stirzaker"
    year: 2020
    edition: "4th"
    publisher: "Oxford University Press"
    isbn: "978-0-19-884759-1"
    url: "https://global.oup.com/academic/product/probability-and-random-processes-9780198847595"
    note: "Chapter 3 on discrete and continuous random variables with excellent worked examples."
  - type: "book"
    title: "All of Statistics"
    author: "Larry Wasserman"
    year: 2004
    publisher: "Springer"
    isbn: "978-0-387-40272-7"
    url: "https://link.springer.com/book/10.1007/978-0-387-21736-9"
    note: "Chapters 2–3 cover random variables, distributions, and conditional distributions concisely."
  - type: "book"
    title: "Statistical Inference"
    author: "George Casella & Roger L. Berger"
    year: 2002
    edition: "2nd"
    publisher: "Duxbury"
    isbn: "978-0-534-24312-8"
    url: "https://www.cengage.com/c/statistical-inference-2e-casella/"
    note: "Chapter 1–2 on probability and random variables. Excellent treatment of transformations."
  - type: "book"
    title: "Pattern Recognition and Machine Learning"
    author: "Christopher M. Bishop"
    year: 2006
    publisher: "Springer"
    isbn: "978-0-387-31073-2"
    url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/"
    note: "Chapter 2 on probability distributions and their ML applications."
  - type: "book"
    title: "Deep Learning"
    author: "Ian Goodfellow, Yoshua Bengio & Aaron Courville"
    year: 2016
    publisher: "MIT Press"
    isbn: "978-0-262-03561-3"
    url: "https://www.deeplearningbook.org/"
    note: "Chapter 3 covers probability and information theory for deep learning."
  - type: "article"
    title: "Normalizing Flows: An Introduction and Review of Current Methods"
    author: "Ivan Kobyzev, Simon J.D. Prince & Marcus A. Brubaker"
    year: 2021
    journal: "IEEE TPAMI"
    volume: "43(11)"
    pages: "3964-3979"
    url: "https://doi.org/10.1109/TPAMI.2020.2992934"
    note: "Survey of normalizing flows — the change of variables formula as the engine of generative modeling."
---
```

### Content Structure

| Section | MDX Heading | Formal Elements | Figure | Interactive Component |
|---|---|---|---|---|
| 1. Random Variables | "Random Variables: The Bridge from Events to Numbers" | Definitions 1–2 (Random Variable, Random Vector), Remark 1, Examples 1–2 | `random-variable-mappings.png` | `RandomVariableMappingExplorer` |
| 2. Discrete Random Variables | "Discrete Random Variables and PMFs" | Definitions 3–5 (Discrete RV, PMF, Support), Examples 3–4 | `pmf-examples.png` | `PMFPDFExplorer` (discrete mode) |
| 3. Continuous Random Variables | "Continuous Random Variables and PDFs" | Definition 6 (Continuous RV/PDF), Remark 2, Examples 5–6 | `pdf-examples.png` | `PMFPDFExplorer` (continuous mode) |
| 4. The CDF | "The Cumulative Distribution Function" | Definition 7 (CDF), Theorem 1 (CDF Properties), Remark 3, Proof 1 | `cdf-comparison.png` | `CDFExplorer` |
| 5. Joint Distributions | "Joint Distributions" | Definitions 8–10 (Joint PMF, PDF, CDF), Examples 7–8 | `joint-distributions.png` | `JointDistributionExplorer` |
| 6. Marginal Distributions | "Marginal Distributions" | Definitions 11–12 (Marginal PMF, PDF), Theorem 2, Remark 4, Proof 2 | `joint-with-marginals.png` | — |
| 7. Conditional Distributions | "Conditional Distributions" | Definitions 13–14 (Cond. PMF, PDF), Theorem 3 (Chain Rule for Densities), Example 9, Proof 3 | `conditional-distribution-slice.png` | — |
| 8. Independence of RVs | "Independence of Random Variables" | Definition 15, Theorems 4–5, Example 10, Proofs 4–5 | `independence-vs-dependence.png` | — |
| 9. Transformations | "Transformations of Random Variables" | Theorems 6–7 (CDF Method, Change of Variables), Remark 5, Examples 11–12, Proofs 6–7 | `transformation-cdf-method.png` | `TransformationExplorer` |
| 10. Expectation Preview | "Expectation (Preview)" | Definition 16 (preview only) | — | — |
| 11. ML Connections | "Connections to ML" | Theorem 8 (Inverse CDF Transform), ML table, Proof 8 | `inverse-cdf-transform.png` | — |
| 12. Summary | "Summary" | Summary table, references | — | — |

---

## 3. Formal Element Inventory

| Type | # | Title |
|------|---|-------|
| Definition | 1 | Random Variable |
| Definition | 2 | Random Vector |
| Definition | 3 | Discrete Random Variable |
| Definition | 4 | Probability Mass Function (PMF) |
| Definition | 5 | Support |
| Definition | 6 | Continuous Random Variable / PDF |
| Definition | 7 | Cumulative Distribution Function (CDF) |
| Definition | 8 | Joint PMF |
| Definition | 9 | Joint PDF |
| Definition | 10 | Joint CDF |
| Definition | 11 | Marginal PMF |
| Definition | 12 | Marginal PDF |
| Definition | 13 | Conditional PMF |
| Definition | 14 | Conditional PDF |
| Definition | 15 | Independence of Random Variables |
| Definition | 16 | Expectation (preview) |
| Theorem | 1 | Properties of the CDF |
| Theorem | 2 | Marginals from the Joint |
| Theorem | 3 | Chain Rule for Densities |
| Theorem | 4 | Independence ⟺ Joint Factors |
| Theorem | 5 | Independence ⟹ Conditional = Marginal |
| Theorem | 6 | CDF Method |
| Theorem | 7 | Change of Variables for PDFs |
| Theorem | 8 | Universality of the Uniform (Inverse CDF Transform) |
| Example | 1 | Die roll as a random variable |
| Example | 2 | Sum of two dice |
| Example | 3 | Bernoulli random variable |
| Example | 4 | Binomial random variable |
| Example | 5 | Uniform distribution on [0,1] |
| Example | 6 | Standard normal distribution |
| Example | 7 | Joint distribution of two dice |
| Example | 8 | Standard bivariate normal |
| Example | 9 | Conditional distribution of bivariate normal |
| Example | 10 | Independent vs dependent normals |
| Example | 11 | Linear transformation |
| Example | 12 | Log transformation (lognormal) |
| Remark | 1 | Why measurability? |
| Remark | 2 | The density is NOT a probability |
| Remark | 3 | Using the CDF to compute probabilities |
| Remark | 4 | Joint determines marginals, not vice versa |
| Remark | 5 | The Jacobian in higher dimensions |
| Proof | 1 | CDF Properties (Theorem 1) |
| Proof | 2 | Marginals from the Joint (Theorem 2) |
| Proof | 3 | Chain Rule for Densities (Theorem 3) |
| Proof | 4 | Independence ⟺ Joint Factors (Theorem 4) |
| Proof | 5 | Independence ⟹ Conditional = Marginal (Theorem 5) |
| Proof | 6 | CDF Method (Theorem 6) |
| Proof | 7 | Change of Variables (Theorem 7) |
| Proof | 8 | Inverse CDF Transform (Theorem 8) |

**Totals:** 16 definitions, 8 theorems, 12 examples, 5 remarks, 8 proofs = 49 formal elements. This topic is larger than Topics 1–2 because it introduces the central objects (random variables, distributions) that all subsequent topics build on.

---

## 4. Static Images

**Directory:** `public/images/topics/random-variables/`

Run the notebook to generate these figures:

| Filename | Notebook Section | Description |
|---|---|---|
| `random-variable-mappings.png` | §1 (Cell 4) | 3-panel: identity mapping, indicator mapping, sum-of-two-dice mapping — arrows from Ω to ℝ |
| `pmf-examples.png` | §2 (Cell 6) | 3-panel: Bernoulli, Binomial, Geometric PMF bar charts |
| `pdf-examples.png` | §3 (Cell 8) | 3-panel: Uniform with shaded area, Normal with shaded area, Uniform(0,0.5) showing density > 1 |
| `cdf-comparison.png` | §4 (Cell 10) | 3-panel: discrete step CDF (Binomial), smooth CDF (Normal), CDF→probability interval method |
| `joint-distributions.png` | §5 (Cell 12) | 3-panel: two-dice heatmap, bivariate normal ρ=0, bivariate normal ρ=0.7 |
| `joint-with-marginals.png` | §6 (Cell 14) | Joint PDF contour with marginal densities on axes (scatter-matrix style layout) |
| `conditional-distribution-slice.png` | §7 (Cell 16) | 3-panel: joint with slice line, conditional PDF (normalized slice), multiple conditionals for different y |
| `independence-vs-dependence.png` | §8 (Cell 18) | 3-panel: independent contours (circular), dependent contours (elliptical), samples from both |
| `transformation-cdf-method.png` | §9 (Cell 20) | 3-panel: PDF of X, CDF method for Y=X², resulting χ²(1) PDF |
| `inverse-cdf-transform.png` | §11 (Cell 23) | 3-panel: Uniform samples, inverse CDF mapping, resulting Normal samples |

**To generate:**

```bash
cd notebooks/random-variables/
pip install numpy matplotlib scipy jupyter
jupyter nbconvert --to notebook --execute 03_random_variables_distribution_functions.ipynb --output executed.ipynb
```

The notebook is seeded (`np.random.seed(42)`) for reproducibility.

---

## 5. Interactive Components

### Component 1: RandomVariableMappingExplorer

**File:** `src/components/viz/RandomVariableMappingExplorer.tsx`

**Purpose:** Visualize $X : \Omega \to \mathbb{R}$ as a literal function mapping outcomes to numbers.

**Interactions:**
- Dropdown to select experiment: Single Die, Two Coins, Two Dice, Custom
- Left panel: Ω displayed as an ellipse with labeled outcomes
- Right panel: ℝ displayed as a number line or ellipse with target values
- Arrows from each outcome to its value under X
- Dropdown to select the random variable mapping: identity, sum, indicator (even), max, squared
- For "Two Dice": 6×6 grid in Ω, arrows to sum values on ℝ; multiple arrows converging to the same value show the pre-image
- Readout: for each value $x$ in the range, show $X^{-1}(\{x\})$ and $P(X = x)$
- Color-code arrows by target value

**Data:** No pre-computed data — all small finite sets.

**Uses from `distributions.ts`:** None directly — this is a pure mapping visualizer.

### Component 2: PMFPDFExplorer

**File:** `src/components/viz/PMFPDFExplorer.tsx`

**Purpose:** Interactive toggle between PMF (discrete) and PDF (continuous) with parameter sliders, emphasizing the area interpretation for PDFs.

**Interactions:**
- **Mode toggle:** Discrete / Continuous
- **Discrete mode:**
  - Distribution selector: Bernoulli, Binomial, Geometric, Poisson (preview)
  - Parameter sliders (e.g., $n$ and $p$ for Binomial)
  - Bar chart of PMF with values labeled on each bar
  - Click a bar → readout shows $P(X = x) = p_X(x)$ (the value IS the probability)
  - Highlight: "Each bar height IS a probability"
- **Continuous mode:**
  - Distribution selector: Uniform(a,b), Normal(μ,σ²), Exponential(λ)
  - Parameter sliders
  - Curve of PDF
  - Click-and-drag to select an interval [a,b] → shaded area highlights and readout shows $P(a \leq X \leq b) = \int_a^b f(x)\,dx$ (numerically computed)
  - Warning banner when $f(x) > 1$: "Density can exceed 1 — it's not a probability!"
  - Highlight: "The AREA is the probability, not the height."

**Data:** No pre-computed data — uses `distributions.ts` functions.

**Uses from `distributions.ts`:** `pmfBernoulli`, `pmfBinomial`, `pmfGeometric`, `pmfPoisson`, `pdfUniform`, `pdfNormal`, `pdfExponential`, `cdfNormal`, `cdfUniform`, `cdfExponential`.

### Component 3: CDFExplorer

**File:** `src/components/viz/CDFExplorer.tsx`

**Purpose:** Side-by-side discrete CDF (step function) and continuous CDF (smooth curve) with interactive $P(X \leq x)$ query.

**Interactions:**
- Left panel: Discrete CDF (Binomial by default)
- Right panel: Continuous CDF (Normal by default)
- Both panels: vertical slider/drag line at $x$ → readout shows $F(x) = P(X \leq x)$
- Distribution selectors and parameter sliders for both panels
- Toggle: "Show PDF/PMF alongside" — draws the PMF bars or PDF curve in a small subplot above/below the CDF
- Interval query mode: user selects $a$ and $b$ → highlights $F(b) - F(a)$ on the CDF and the corresponding area under the PDF/PMF
- CDF properties annotations: "non-decreasing ✓", "right-continuous ✓", "$F(-\infty)=0$, $F(\infty)=1$ ✓"

**Data:** No pre-computed data.

**Uses from `distributions.ts`:** CDF functions for all available distributions.

### Component 4: JointDistributionExplorer

**File:** `src/components/viz/JointDistributionExplorer.tsx`

**Purpose:** 2D joint distribution visualization with marginals on axes and conditional distribution slicing.

**Interactions:**
- **Mode toggle:** Discrete (two dice) / Continuous (bivariate normal)
- **Discrete mode:**
  - 6×6 heatmap of joint PMF for two dice
  - Click a cell → readout shows $p_{X,Y}(x,y)$
  - Toggle: "Show marginals" — bar charts on top and right axes
  - Toggle: "Show conditional" — select a column (condition on $Y = y$) → highlight that column, show the conditional PMF $p_{X|Y}(x \mid y)$ as a normalized bar chart
- **Continuous mode:**
  - Bivariate normal contour plot
  - Slider for $\rho$ from $-0.95$ to $0.95$
  - Marginal PDFs on top and right axes (always shown)
  - Drag a horizontal line (condition on $Y = y$) → the conditional $f_{X|Y}(x \mid y)$ displayed as a curve below the contour
  - Independence indicator: "Independent ($\rho = 0$)" or "Dependent ($\rho \neq 0$)" with factorization check
- Animated transition when $\rho$ changes — contours morph from circular to elliptical

**Data:** No pre-computed data.

**Uses from `distributions.ts`:** `pdfBivariateNormal`, `conditionalNormalParams`, `pdfNormal`.

### Component 5: TransformationExplorer

**File:** `src/components/viz/TransformationExplorer.tsx`

**Purpose:** Visualize $Y = g(X)$: pick a distribution for $X$ and a transformation $g$, watch the CDF method produce the distribution of $Y$.

**Interactions:**
- Source distribution selector: Normal(0,1), Uniform(0,1), Exponential(1)
- Transformation selector: $Y = X^2$, $Y = e^X$, $Y = |X|$, $Y = aX + b$ (with sliders for $a, b$), $Y = F^{-1}(X)$ (inverse CDF mode)
- Three-panel display:
  1. PDF of $X$ with the function $g(x)$ curve overlaid or shown in a small inset
  2. CDF method animation: $F_Y(y) = P(g(X) \leq y)$ shown step-by-step
  3. PDF of $Y$ (computed numerically or analytically)
- For the inverse CDF mode ($X \sim \text{Uniform}(0,1)$, $Y = F^{-1}(X)$):
  - Target distribution selector: Normal, Exponential, Beta
  - Shows the CDF of the target, horizontal lines from $U$ to $F^{-1}(U)$, histogram of generated samples
- Monte Carlo overlay: draw $n$ samples from $X$, transform them, overlay histogram on the theoretical PDF of $Y$
- Sample size slider: $n = 100$ to $n = 10{,}000$

**Data:** No pre-computed data.

**Uses from `distributions.ts`:** PDF, CDF, and quantile functions for all available distributions.

### Implementation Notes

- Same architecture as Topics 1–2: React + Tailwind, `client:visible` directive
- SVG for mapping diagrams; D3.js for all charts and distributions
- Use CSS custom properties from `viz/shared/colorScales.ts`
- All components responsive (test at 375px width)
- No server-side computation
- For numerical integration in the browser (PDF area computation), use the trapezoidal rule on a fine grid — no external library needed

---

## 6. Shared Utility Module: `distributions.ts`

### New Module: `src/components/viz/shared/distributions.ts`

This module provides PMF/PDF/CDF/quantile functions for common distributions. It is separate from `probability.ts` (which handles set operations, conditional probability, and independence). It will be extended by Topics 4–8.

```typescript
// ── Types ───────────────────────────────────────────────────────────────────

/** Parameters for a distribution. */
export interface DistributionParams {
  [key: string]: number;
}

/** A distribution descriptor. */
export interface Distribution {
  name: string;
  type: 'discrete' | 'continuous';
  params: DistributionParams;
  support: [number, number]; // [min, max] (Infinity allowed)
}

// ── Discrete PMFs ───────────────────────────────────────────────────────────

/** Bernoulli PMF: P(X = k) = p^k (1-p)^(1-k), k ∈ {0,1}. */
export function pmfBernoulli(k: number, p: number): number;

/** Binomial PMF: P(X = k) = C(n,k) p^k (1-p)^(n-k). */
export function pmfBinomial(k: number, n: number, p: number): number;

/** Geometric PMF: P(X = k) = (1-p)^(k-1) p, k = 1,2,... */
export function pmfGeometric(k: number, p: number): number;

/** Poisson PMF: P(X = k) = e^(-λ) λ^k / k! */
export function pmfPoisson(k: number, lambda: number): number;

// ── Continuous PDFs ─────────────────────────────────────────────────────────

/** Uniform PDF on [a, b]. */
export function pdfUniform(x: number, a: number, b: number): number;

/** Normal PDF with mean μ and variance σ². */
export function pdfNormal(x: number, mu: number, sigma2: number): number;

/** Exponential PDF with rate λ. */
export function pdfExponential(x: number, lambda: number): number;

/** Standard normal PDF (convenience). */
export function pdfStdNormal(x: number): number;

/** Bivariate normal PDF with correlation ρ (standard marginals). */
export function pdfBivariateNormal(
  x: number, y: number,
  muX: number, muY: number,
  sigmaX: number, sigmaY: number,
  rho: number
): number;

// ── CDFs ────────────────────────────────────────────────────────────────────

/** Binomial CDF: P(X ≤ k). */
export function cdfBinomial(k: number, n: number, p: number): number;

/** Normal CDF using the error function approximation. */
export function cdfNormal(x: number, mu: number, sigma2: number): number;

/** Standard normal CDF Φ(x). */
export function cdfStdNormal(x: number): number;

/** Uniform CDF on [a, b]. */
export function cdfUniform(x: number, a: number, b: number): number;

/** Exponential CDF. */
export function cdfExponential(x: number, lambda: number): number;

// ── Quantile (Inverse CDF) Functions ────────────────────────────────────────

/** Standard normal quantile Φ⁻¹(p) — Rational approximation. */
export function quantileStdNormal(p: number): number;

/** Normal quantile. */
export function quantileNormal(p: number, mu: number, sigma2: number): number;

/** Exponential quantile: F⁻¹(p) = -ln(1-p)/λ. */
export function quantileExponential(p: number, lambda: number): number;

/** Uniform quantile. */
export function quantileUniform(p: number, a: number, b: number): number;

// ── Conditional Normal ──────────────────────────────────────────────────────

/** Conditional parameters of X|Y=y for bivariate normal. */
export function conditionalNormalParams(
  muX: number, muY: number,
  sigmaX: number, sigmaY: number,
  rho: number,
  yGiven: number
): { condMean: number; condVar: number };

// ── Numerical Integration ───────────────────────────────────────────────────

/** Trapezoidal rule integration of f over [a, b] with n steps. */
export function trapezoidalIntegral(
  f: (x: number) => number,
  a: number, b: number,
  n?: number
): number;

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Error function approximation (for normal CDF). */
export function erf(x: number): number;

/** Log-gamma function (for binomial coefficient computation). */
export function lnGamma(x: number): number;

/** Binomial coefficient C(n,k) via log-gamma. */
export function binomialCoeff(n: number, k: number): number;
```

**Design contract:** Same as `probability.ts` — all functions pure, deterministic, documented with JSDoc. The `cdfNormal` implementation should use the error function approximation (Abramowitz & Stegun or Horner form) accurate to ~10⁻⁷. The `quantileStdNormal` should use the rational approximation from Peter Acklam or the Beasley-Springer-Moro algorithm.

---

## 7. Topic Data Module

### New Module: `src/data/random-variables-data.ts`

```typescript
/** Preset random variable mappings for RandomVariableMappingExplorer. */
export const rvMappingPresets = [
  {
    name: "Die Roll (Identity)",
    omega: ["1", "2", "3", "4", "5", "6"],
    mappings: {
      "Identity X(ω) = ω": { "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6 },
      "Squared X(ω) = ω²": { "1": 1, "2": 4, "3": 9, "4": 16, "5": 25, "6": 36 },
      "Even Indicator": { "1": 0, "2": 1, "3": 0, "4": 1, "5": 0, "6": 1 },
    },
  },
  {
    name: "Two Coins",
    omega: ["HH", "HT", "TH", "TT"],
    mappings: {
      "# Heads": { "HH": 2, "HT": 1, "TH": 1, "TT": 0 },
      "First Coin (H=1)": { "HH": 1, "HT": 1, "TH": 0, "TT": 0 },
      "Match Indicator": { "HH": 1, "HT": 0, "TH": 0, "TT": 1 },
    },
  },
];

/** Distribution presets for PMFPDFExplorer. */
export const discreteDistributionPresets = [
  { name: "Bernoulli", params: { p: 0.5 }, paramRanges: { p: [0.01, 0.99] } },
  { name: "Binomial", params: { n: 10, p: 0.3 }, paramRanges: { n: [1, 30], p: [0.01, 0.99] } },
  { name: "Geometric", params: { p: 0.3 }, paramRanges: { p: [0.05, 0.95] } },
  { name: "Poisson", params: { lambda: 5 }, paramRanges: { lambda: [0.5, 20] } },
];

export const continuousDistributionPresets = [
  { name: "Uniform", params: { a: 0, b: 1 }, paramRanges: { a: [-5, 5], b: [-5, 10] } },
  { name: "Normal", params: { mu: 0, sigma2: 1 }, paramRanges: { mu: [-5, 5], sigma2: [0.1, 10] } },
  { name: "Exponential", params: { lambda: 1 }, paramRanges: { lambda: [0.1, 5] } },
];

/** Transformation presets for TransformationExplorer. */
export const transformationPresets = [
  { name: "Y = X²", source: "Normal(0,1)", formula: "x² → χ²(1)", gFunc: (x: number) => x * x },
  { name: "Y = eˣ", source: "Normal(0,1)", formula: "eˣ → Lognormal", gFunc: (x: number) => Math.exp(x) },
  { name: "Y = |X|", source: "Normal(0,1)", formula: "|X| → Half-Normal", gFunc: (x: number) => Math.abs(x) },
  { name: "Y = aX + b", source: "Normal(0,1)", formula: "Linear → Normal(b, a²)", gFunc: (x: number, a: number, b: number) => a * x + b },
];

/** Bivariate normal presets for JointDistributionExplorer. */
export const bivariateNormalPresets = [
  { name: "Independent (ρ = 0)", rho: 0 },
  { name: "Moderate positive (ρ = 0.5)", rho: 0.5 },
  { name: "Strong positive (ρ = 0.85)", rho: 0.85 },
  { name: "Negative (ρ = −0.6)", rho: -0.6 },
];
```

---

## 8. Curriculum Graph & Site Metadata Updates

### Update: `src/data/curriculum-graph.json`

Change `random-variables` from `"planned"` to `"published"`. Add prerequisite edges:

```json
{
  "id": "random-variables",
  "status": "published",
  "prerequisites": ["sample-spaces", "conditional-probability"]
}
```

### Update: `src/data/curriculum.ts`

Mark Track 1, Topic 3 as published.

### Update: `conditional-probability.mdx`

Replace all "(coming soon)" references to random variables with live links:

```mdx
<!-- Before -->
Random Variables & Distributions (coming soon)

<!-- After -->
[Random Variables & Distribution Functions](/topics/random-variables)
```

Also update the §7 conditional independence remark and the §9 "What's Next" section to link to the live topic.

### Update: `sample-spaces.mdx`

Check for any remaining "(coming soon)" references to random variables and replace with live links.

### Update: References Spreadsheet

Add all 8 references from this topic's frontmatter to the references spreadsheet, including hyperlinks (the `url` field in each reference entry). Ensure no duplicate entries — Billingsley, Durrett, Grimmett & Stirzaker, Wasserman, and Bishop appear in Topics 1–2 as well; update those rows if URLs were previously missing, and add new rows for Casella & Berger, Goodfellow et al., and Kobyzev et al.

---

## 9. Verification Checklist

### Content

- [ ] `random-variables.mdx` renders at `/topics/random-variables` with no build errors
- [ ] All 10 static images present in `public/images/topics/random-variables/`
- [ ] KaTeX renders correctly: all display equations, inline math, and theorem blocks
- [ ] All 16 definitions, 8 theorems, 12 examples, 5 remarks render in styled blocks
- [ ] All 8 proofs expand fully with no hand-waving
- [ ] Internal links to `sample-spaces` and `conditional-probability` work
- [ ] formalcalculus.com links (`sequences-limits`, `change-of-variables`) open in new tab with external badge
- [ ] formalml.com links (`normalizing-flows`, `information-geometry`, `variational-inference`) open in new tab with forward-reference badge
- [ ] Forward references to Topic 4 use plain text + "(coming soon)"
- [ ] `conditional-probability.mdx` updated: "(coming soon)" for random variables replaced with live link
- [ ] References spreadsheet updated with all 8 references and hyperlinks

### Interactive Components

- [ ] `RandomVariableMappingExplorer` dropdown switches presets; arrows map correctly; pre-image and PMF readout update
- [ ] `PMFPDFExplorer` discrete mode shows PMF bars with probability labels; continuous mode shows PDF curve with area computation; density > 1 warning appears correctly
- [ ] `CDFExplorer` step function and smooth curve render correctly; interactive query line updates $F(x)$ readout; interval query mode computes $F(b) - F(a)$
- [ ] `JointDistributionExplorer` discrete heatmap and continuous contour work; marginals display; conditioning slider produces correct conditional distribution; ρ slider morphs contours
- [ ] `TransformationExplorer` all transformations produce correct output distributions; Monte Carlo histogram converges; inverse CDF mode works

### Infrastructure

- [ ] `distributions.ts` compiles with no TypeScript errors
- [ ] `random-variables-data.ts` compiles
- [ ] Curriculum graph updated — `random-variables` now published
- [ ] Pagefind indexes the new topic on rebuild
- [ ] `pnpm build` succeeds with zero errors
- [ ] Mobile responsive (all 5 interactive components at 375px width)

---

## 10. Build Order

1. **Create `src/components/viz/shared/distributions.ts`** with all functions from §6. Implement the normal CDF via an error-function approximation and the normal quantile via a rational approximation. Write console log tests to verify PMF sums to 1, PDF integrates to 1, CDF limits, and quantile-CDF round-trips.
2. **Create `src/data/random-variables-data.ts`** with all presets from §7.
3. **Copy notebook figures** to `public/images/topics/random-variables/`.
4. **Create `random-variables.mdx`** with full frontmatter and all markdown/LaTeX content. Include all formal element blocks (definitions, theorems, proofs, examples, remarks). No interactive components yet.
5. **Build `RandomVariableMappingExplorer.tsx`** — the mapping visualizer with arrows from Ω to ℝ.
6. **Build `PMFPDFExplorer.tsx`** — the dual-mode PMF/PDF explorer with area interpretation.
7. **Build `CDFExplorer.tsx`** — side-by-side discrete and continuous CDFs with interactive query.
8. **Build `JointDistributionExplorer.tsx`** — the flagship: joint distribution with marginals and conditional slicing.
9. **Build `TransformationExplorer.tsx`** — transformation visualizer with CDF method and inverse CDF.
10. **Embed all 5 components** in the MDX with `client:visible`.
11. **Update `curriculum-graph.json`** and `curriculum.ts` — mark `random-variables` as published.
12. **Update `conditional-probability.mdx`** and `sample-spaces.mdx` — replace "(coming soon)" with live links.
13. **Update the references spreadsheet** — add new references with hyperlinks, update existing entries if URLs were missing.
14. Run the full verification checklist (§9).
15. `pnpm build` — verify zero errors.
16. Deploy to Vercel.

---

## Appendix A: KaTeX Constraints

Same as Topics 1–2:

- **No `\begin{aligned}` blocks with `&` markers.** Multi-line derivations use separate `$$...$$` blocks with prose glue between lines.
- **No `\begin{array}{c|cccc}` tables.** Use HTML tables or markdown tables for tabular content.
- Inline math uses `$...$`. Display math uses `$$...$$` on its own line.
- Test all LaTeX rendering in the dev server before committing.

---

## Appendix B: Design Decisions

1. **Title: "Random Variables & Distribution Functions"** (not "Random Variables & Distributions") to distinguish from Topics 5–6 (Discrete Distributions, Continuous Distributions), which catalog specific distribution families. This topic defines the *framework* — what PMFs, PDFs, and CDFs are — while Topics 5–6 fill in the *catalog*.

2. **Interleaved discrete/continuous treatment.** PMF → PDF → CDF rather than "all discrete, then all continuous." This mirrors how practitioners think (parallel constructions) and lets the CDF serve as the unifying bridge immediately.

3. **New `distributions.ts` module** rather than extending `probability.ts`. The existing module handles set operations, conditional probability, and independence — concepts that are conceptually different from distribution function evaluation. Separation of concerns keeps both modules focused and prevents `probability.ts` from growing unwieldy.

4. **Largest formal element count (49) of any topic so far.** This is justified because random variables and distributions are the central objects of probability theory. Every subsequent topic builds on these definitions. The Topic 1–2 counts (29 each) were appropriate for their narrower scope.

5. **The `JointDistributionExplorer` is the flagship component.** It combines joint distribution visualization with marginals (Topic 3's core contribution) and conditional distribution slicing (which connects back to Topic 2). The ρ slider for the bivariate normal makes independence vs. dependence viscerally clear.

6. **Inverse CDF transform in ML Connections, not in Transformations.** The universality of the uniform is a theorem about transformations, but its primary payoff is computational (Monte Carlo sampling), and its primary modern application is normalizing flows. Placing it in §11 (ML Connections) makes the application front and center.

7. **Expectation preview is deliberately minimal.** Two paragraphs, one definition, no proofs. The reader sees where Topic 4 is headed, but all derivations (linearity, variance decomposition, law of total expectation) belong to that topic.

---

*Brief version: v1 | Created: 2026-04-11 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/random-variables/03_random_variables_distribution_functions.ipynb`*
