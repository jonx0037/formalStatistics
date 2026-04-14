# CLAUDE.md — formalStatistics

## Project Overview

formalStatistics is a static site of long-form probability and statistics explainers for ML practitioners, grad students, and researchers. It fills the gap between standard undergraduate statistics courses and the probabilistic and inferential foundations assumed by [formalML](https://formalml.com) — providing the rigorous probability theory, statistical inference, and stochastic process machinery that modern machine learning relies on. Every topic gets three pillars: rigorous math, interactive visualization, and working code.

Live site: https://formalstatistics.com
Predecessor site: https://formalcalculus.com
Successor site: https://formalml.com

## Tech Stack

- **Framework:** Astro 6 (static site generation)
- **UI:** React 19 (interactive components only — Astro handles static markup)
- **Content:** MDX with remark-math + rehype-katex for LaTeX rendering
- **Styling:** Tailwind CSS 4
- **Visualizations:** D3.js 7 (via React components in `src/components/viz/`)
- **Search:** Pagefind (runs post-build)
- **Package manager:** pnpm (not npm — no package-lock.json)
- **Deploy:** Vercel

## Commands

```bash
pnpm dev        # Dev server at localhost:4321
pnpm build      # Production build (runs pagefind post-build)
pnpm preview    # Preview production build
```

## Project Structure

```
src/
├── pages/              # Astro routes (topics use [...slug].astro)
├── content/topics/     # MDX topic files (the content)
├── components/
│   ├── ui/             # Astro structural components (Nav, TopicCard, TheoremBlock, etc.)
│   └── viz/            # React + D3 interactive visualizations
│       └── shared/     # Shared hooks, types, color scales, utility modules
├── data/               # Curriculum graph, sample datasets
├── layouts/            # Page layout templates
├── lib/                # Utility modules
└── styles/             # Global CSS, design tokens

docs/plans/             # Planning & handoff documents
notebooks/              # Research notebooks (Jupyter, not tracked in git)
public/images/          # Static images organized by topic
```

## Content Conventions

### Mathematical exposition style

- **Geometric-first:** Introduce concepts visually and concretely before measure-theoretic or algebraic machinery. Probability is deeply geometric — random variables as functions, distributions as shapes, expectations as centers of mass. Exploit this relentlessly.
- **Foundational topics:** Zero measure theory on first pass — stop at intuitive probability and counting arguments, then build rigor in dedicated sections.
- **Intermediate topics:** Measure-theoretic formalism after intuitive foundations are established.
- **Proofs:** Expand fully with every step in expectation calculations, inequality chains, and convergence arguments — never "it can be shown." Statistics proofs are where students learn to reason about uncertainty; cutting corners here is unacceptable.
- **Examples:** Concrete, motivating examples before every definition. Use ML-relevant examples wherever possible (bias-variance decomposition, likelihood maximization, posterior computation, hypothesis tests on model performance, bootstrap for confidence intervals on validation metrics).
- **Bridge backward:** Every topic should reference the specific formalCalculus prerequisites it relies on — not re-teaching calculus, but linking to exactly where the reader can find the required background (e.g., "this expectation integral requires Lebesgue integration — see formalCalculus: The Lebesgue Integral").
- **Bridge forward:** Every topic should include a "Connections to ML" section or callout boxes that explain exactly where these statistics appear in machine learning, with explicit links to formalml.com topics where applicable.

### Difficulty calibration

formalStatistics serves readers who have the calculus foundations (from formalCalculus or equivalent) but need rigorous probability and statistics. Calibrate accordingly:

- **Foundational:** Assumes comfort with limits, integrals, and series (formalCalculus Tracks 1–2 or equivalent). Builds probability from combinatorics and basic measure theory. Introduces random variables, distributions, and expectation with full rigor but careful scaffolding.
- **Intermediate:** Assumes comfort with multivariate calculus and basic probability (formalCalculus Tracks 3–4 and formalStatistics foundational topics). Covers joint distributions, convergence theorems, estimation theory, and hypothesis testing.
- **Advanced:** Assumes the full probability and single/multivariate calculus toolkit. Covers measure-theoretic probability, stochastic processes, high-dimensional statistics, and Bayesian nonparametrics — the direct on-ramps to formalml.com topics.

### MDX topic file structure

Each topic in `src/content/topics/` is an MDX file with YAML frontmatter defining:
- title, description, domain, difficulty, prerequisites, references
- `formalcalculusPrereqs` — array of formalcalculus.com topic slugs, this topic requires
- `formalmlConnections` — array of formalml.com topic slugs, this topic feeds into
- Interactive viz components are imported and embedded inline

### Visualization components

- All viz components live in `src/components/viz/`
- Use D3.js via the `useD3` hook in `viz/shared/useD3.ts`
- Use `useResizeObserver` for responsive sizing
- Shared color scales in `viz/shared/colorScales.ts`
- Shared types in `viz/shared/types.ts`
- Shared statistics utilities in track-specific modules (e.g., `viz/shared/distributions.ts`, `viz/shared/sampling.ts`, `viz/shared/estimation.ts`)
- Use `.style()` for CSS custom properties in D3 SVG elements (not `.attr("style", ...)`)

### Statistics-specific visualization conventions

Statistics visualizations have unique requirements:

- **Distribution visualizers:** PDF/CDF side-by-side with parameter sliders. Overlay multiple distributions for comparison. Support log-scale axes. Show the area interpretation of probability (shaded regions under curves).
- **Sampling visualizers:** Animated draws from distributions with running histograms that converge to the true density. Sample size slider from n=1 to n=10000+. Show individual samples and aggregate behavior simultaneously.
- **Convergence visualizers:** Law of Large Numbers — show sample mean trajectories converging. Central Limit Theorem — show the sampling distribution of the mean normalizing as n grows. Support multiple underlying distributions to show universality.
- **Inference visualizers:** Confidence interval coverage simulators — generate many intervals, highlight which ones capture the true parameter. P-value simulators under the null and alternative. Power curves as a function of sample size and effect size.
- **Bayesian visualizers:** Prior → Likelihood → Posterior animation. Sequential updating occurs as data arrives, one observation at a time. Prior sensitivity — toggle between priors and watch the posterior shift.
- **Regression visualizers:** Least squares with draggable data points. Residual plots updating in real time. Ridge/lasso penalty paths with coefficient trajectories.
- **Hypothesis test visualizers:** Test statistic distributions under H₀ and H₁. Rejection region shading. Type I/II error areas with adjustable α.
- **Bootstrap visualizers:** Animated resampling with replacement. Bootstrap distribution building up sample by sample. Comparison to the theoretical sampling distribution.

### Curriculum graph

- Topic metadata and prerequisite DAG defined in `src/data/curriculum-graph.json`
- Track definitions in `src/data/curriculum.ts`
- When adding a new topic, update both files and add cross-links in related topics
- Prerequisite links may reference formalcalculus.com topics (external prerequisites, marked distinctly in the DAG)

### Relationship to sister sites

formalStatistics sits between formalCalculus and formalML in the learning path:

```
formalCalculus → formalStatistics → formalML
(calculus)       (probability/stats)  (ML math)
```

- **formalCalculus (prerequisite):** formalStatistics topics may reference formalcalculus.com topics as prerequisites — external links with a visual indicator (e.g., ← formalCalculus badge). The reader is expected to have covered the relevant calculus before arriving here.
- **formalML (successor):** formalStatistics topics can reference formalml.com topics as "where this leads" — external links with a visual indicator (e.g., → formalML badge).
- **No circular dependencies:** formalStatistics should never assume knowledge from formalml.com topics. It *may* assume knowledge of topics from formalcalculus.com.
- The three sites share a tech stack and editorial voice but are independent codebases and deployments.

### Citations and references

- Every MDX topic file MUST end with a rendered `### References` section as the last content in the file, preceded by a `---` horizontal rule.
- The references section lists every entry from the topic's frontmatter `references:` array, in the same order, as a numbered Markdown list.
- Format for books: `N. Author(s). (Year). [*Title*](URL) (Edition). Publisher.`
- Format for articles: `N. Author(s). (Year). [*Title*](URL). *Journal*, Volume(Issue), Pages.`
- Every frontmatter reference entry MUST include a `url:` field with a working hyperlink.
- This section is non-negotiable. A topic without visible citations does not ship.

## Code Style

- TypeScript throughout (Astro + React)
- Functional React components with hooks
- No class components
- Prefer named exports
- D3 selections scoped to component refs — no global DOM manipulation

## Do NOT

- Use npm or generate package-lock.json
- Commit .vscode/, .DS_Store, or firebase-debug.log
- Create draft files outside src/content/topics/ — drafts live as unpublished MDX
- Skip intuitive probability before measure-theoretic formalism
- Write one-line proof sketches — expand or omit
- Assume the reader already knows statistics rigorously — that's what this site teaches
- Assume knowledge from formalml.com topics — only formalcalculus.com topics may be prerequisites
- Link to formalml.com topics as prerequisites — only as forward references

## Editorial Voice

- **Tone:** Informed peer, not lecturer. Think "a sharp colleague explaining something at a whiteboard" — conversational enough to use contractions and the occasional aside, but precise enough that no claim is hand-wavy. The prose should read well *as prose*, not just as a vehicle for equations.
- **Pronouns:** Default to "we" as the collaborative mathematical "we" (we define, we observe, we can now see that…). Use "you" sparingly and only for direct reader instructions — "you can verify this by…" or "try adjusting the sample size slider to see…". Avoid passive voice for derivations; if someone is doing the math, say who.
- **Assumed reader knowledge:** The reader has a solid calculus foundation — limits, derivatives, integrals, series, and some measure theory (from formalCalculus or equivalent). They may have taken an undergraduate statistics course and can compute a mean, variance, or t-test mechanically, but they don't yet understand *why* those procedures work or when they break down. They've heard of maximum likelihood but may not have derived an estimator from scratch. Meet them where they are and build from there.
- **Jargon and notation:** Introduce notation explicitly on first use in every topic — even standard stuff like $\mathbb{E}[X]$, $\text{Var}(X)$, or $X \sim \mathcal{N}(\mu, \sigma^2)$. Never let a symbol appear without a plain-English gloss nearby. Jargon is fine once defined, but prefer the concrete name over the abstract one when both exist (say "the average squared distance from the mean" before saying "the variance").
- **Attitude toward the reader:** Respect without flattery. Don't say "simply," "obviously," or "it's easy to see." If something is genuinely straightforward, the exposition will make that self-evident. If something is hard, say so — "this is where most textbooks lose people, so let's slow down" is more useful than pretending it's trivial.
- **ML motivation:** Every topic should make clear *why* an ML practitioner needs this. Not as an afterthought or appendix, but woven into the exposition. "This is why your cross-validation estimate has variance" is better than "this theorem has applications in model selection."
