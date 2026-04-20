# Claude Chat Brief-Drafting Session — Topic 25: Bayesian Foundations & Prior Selection

I'm drafting the **handoff brief and Jupyter notebook** for **Topic 25** of [formalstatistics.com](https://www.formalstatistics.com). The output of this session goes downstream to a Claude Code agent that will implement the MDX, components, and cross-references against the brief — so the brief needs to be authoritative enough that downstream decisions are mechanical.

---

## Who you're talking to

I'm Jonathan Rocha — founder of DataSalt LLC (AI/ML consultancy), SMU MSDS student, author of *Applied NLP for Finance*. Background in humanities (TAMU English MA, History BA) and financial NLP. I'm writing **formalStatistics** as the rigorous middle book between [formalCalculus](https://formalcalculus.com) (calculus foundations, shipped) and [formalML](https://formalml.com) (machine-learning math, in progress).

**How I want to work:**
- Concise, intellectually dense responses. No trailing summaries.
- For design decisions: present the function signature or section heading and let me write the body — don't draft full prose, I'll just rewrite.
- For tradeoffs: surface them as a recommendation with the main alternative, in 2–3 sentences. Don't decide unilaterally.
- I like to swing big — choose the ambitious path when there's a tie.
- For ambiguous or multi-step asks, ask for clarification before committing to a direction.

**My math voice:** geometric-first. Visual intuition before algebraic machinery. Probability is geometric — random variables as functions, distributions as shapes, expectations as centers of mass. Exploit it.

---

## Project context

**formalStatistics** is a 32-topic static site teaching probability and statistics rigorously, organized into 8 tracks of 4 topics each. Stack: Astro 6 + React 19 + MDX + KaTeX + D3.js, deployed to Vercel.

**Three-pillar pedagogy** (every topic):
1. **Rigorous math** — full proofs, no "it can be shown."
2. **Interactive visualization** — D3-based components readers manipulate.
3. **Working code** — the testing/estimation utilities components depend on.

**Sister-site relationship:**
- [formalCalculus](https://formalcalculus.com) — *prerequisites*. We can cite calculus topics as live `ExternalLink`s.
- [formalML](https://formalml.com) — *successors*. We point forward to ML topics for "where this leads," but never assume they're prerequisite knowledge.

**Editorial voice** (locked for the project):
- "Informed peer at a whiteboard," not lecturer.
- Default to "we" (collaborative mathematical we); use "you" sparingly for direct reader instructions.
- Introduce notation explicitly on first use, every topic — even standard symbols.
- Respect without flattery: avoid "simply," "obviously," "it's easy to see."
- Every topic ties back to ML motivation explicitly — "this is why your cross-validation estimate has variance" beats "this theorem has applications in model selection."

---

## Curriculum state at session start

- **24 of 32** topics shipped across the 8 tracks.
- **Track 7 (Bayesian Statistics)** status: **0 of 4 published — Topic 25 opens Track 7**.
- **Most recent ship:** Topic 24: Model Selection & Information Criteria (Track 6 closer) — see https://github.com/jonx0037/formalStatistics/pull/28.
- **Topic 25** is **1st of 4 (track opener)**.
- **Primary predecessor:** Topic 24: Model Selection & Information Criteria (slug `model-selection-and-information-criteria`). §24.4 develops BIC as the Laplace approximation to the Bayesian marginal likelihood — that derivation is the explicit on-ramp into Track 7. §24.10 Rem 23 (BMA full pointer), Rem 26 (DIC/WAIC/PSIS-LOO), and Rem 31 (Track 7 BIC → marginal likelihood → BMA → MCMC) are all forward-promises that Topic 25 opens the door for.
- **Other shipped topics this one consumes:** Topic 4 (Conditional Probability — Bayes' theorem itself, prior × likelihood / evidence framing); Topic 7 (Exponential Families — conjugate-prior structure is canonical exponential-family construction); Topic 8 (Multivariate Distributions — multivariate Normal as prior/posterior; Wishart prior on Σ); Topic 14 (Maximum Likelihood — MLE is the frequentist counterpart of the posterior mode under a flat prior; Topic 14 Thm 6 asymptotic normality is the engine for Bernstein–von Mises); Topic 24 (Model Selection — §24.4's BIC-Laplace derivation generalizes to the posterior-Laplace approximation that Topic 25 develops in full).
- **Existing notebooks worth studying for figure / MC / preset patterns:** `notebooks/conditional-probability/02_conditional_probability_independence.ipynb` (Bayes-theorem visualization patterns: `bayes-theorem-flow.png`, `medical-testing.png` — direct prior models for Topic 25's prior/posterior figures); `notebooks/maximum-likelihood/14_maximum_likelihood.ipynb` (likelihood surfaces, MLE asymptotic normality figure `mle-asymptotic-normality.png` — the frequentist counterpart Topic 25 contrasts against Bernstein–von Mises); `notebooks/exponential-families/07_exponential_families.ipynb` (`conjugate-priors.png` is the canonical conjugate-prior chart — Topic 25 either reuses or extends it); `notebooks/multivariate-distributions/08_multivariate_distributions.ipynb` (MVN density patterns in `multivariate-normal-deep-dive.png`, `conditional-mvn.png` — directly applicable to the multivariate Normal-Normal example and MVN posteriors); `notebooks/model-selection/24_model_selection.ipynb` (the cell that produced `24-bic-laplace.png` is the Laplace-approximation toy figure pattern Topic 25 will mirror at full posterior depth). These are the prior notebooks whose cell structure, palette choices, MC simulation conventions, and verification helpers are most directly transferrable to Topic 25's notebook. Cite them by path so the Claude Chat session can `Read` them directly to study before drafting §4.

This topic **opens Track 7**. Tracks 1–6 are entirely frequentist (probability foundations, distributions, convergence, estimation, testing, regression); Track 7 introduces the Bayesian alternative formalism that runs in parallel. Topic 25's job is to lay the conceptual groundwork (prior, likelihood, posterior, posterior predictive, conjugacy, prior choice, Bernstein–von Mises bridge to frequentist asymptotics), so the next three topics can build computation (Topic 26 MCMC), model comparison (Topic 27 marginal likelihood / Bayes factors / BMA), and hierarchical structure (Topic 28 partial pooling, empirical Bayes) on top.

---

## Topic to develop — Topic 25: Bayesian Foundations & Prior Selection

**Working slug:** `bayesian-foundations-and-prior-selection` (decision to confirm — see "Decisions to lock" below). Note: `curriculum-graph.json` currently has the entry with `id: "bayesian-foundations"` and `url: "/topics/bayesian-foundations"` (provisional). Per the Topic 18/19/23/24 "X & Y" precedent, expanding to `bayesian-foundations-and-prior-selection` is the natural choice — but worth a deliberate decision because it touches every Track 7 forward-pointer that gets written from now on.

**One-paragraph thesis** (subject to revision in the conversation):

Topic 25 is the entry point to Bayesian statistics: the rigorous treatment of probability as a coherent system for updating belief in light of data. The exposition starts from Bayes' theorem applied to a parameter $\theta$ with prior density $\pi(\theta)$, develops the four primitives every Bayesian computation produces — **prior, likelihood, posterior, posterior predictive** — and addresses prior choice as a substantive modeling decision rather than an afterthought. The featured arc is **conjugacy** for the canonical exponential-family models (Beta-Binomial, Normal-Normal with known variance, Normal-Normal-Inverse-Gamma with unknown variance, Gamma-Poisson, Dirichlet-Multinomial, Wishart-Normal): closed-form posterior families that make Bayesian inference computationally tractable on the textbook examples. The deepest result is **Bernstein–von Mises** — under regularity, the posterior $p(\theta \mid \mathbf{y})$ converges in total variation to $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}^{-1}/n)$, so Bayesian credible intervals and frequentist Wald intervals coincide asymptotically. This is the bridge that connects Topics 13–24 (frequentist machinery) to the Bayesian alternative. Topic 25 closes by previewing the full Track 7 program: MCMC (Topic 26), marginal-likelihood-based model comparison + BMA (Topic 27), hierarchical priors and partial pooling (Topic 28).

**Initial featured-theorem guess:** **Bernstein–von Mises** — it's the deepest result, the explicit bridge to frequentist asymptotics, and the answer to every "but does the prior matter as $n \to \infty$?" question the reader will bring. Strong second candidate: a **conjugacy theorem** (the exponential-family general form, with Beta-Binomial / Normal-Normal / Gamma-Poisson as canonical instances) — featured if we decide the topic should center on construction rather than asymptotics. My instinct says Bernstein–von Mises, with a full conjugacy *theorem-with-derivation* (not "stated only") in the section before; let's discuss.

**Initial scope boundaries** (deferrals to lock early):
- **MCMC algorithms** (Metropolis–Hastings, Gibbs, NUTS, HMC) → defer entirely to Topic 26 (Bayesian Computation). Topic 25 mentions MCMC by name only when the conjugate-prior framework breaks down — one paragraph in §X.10.
- **BMA, Bayes factors, marginal-likelihood computation** (nested sampling, bridge sampling, thermodynamic integration) → defer to Topic 27 (Bayesian Model Comparison). Topic 25 cites BMA in the §X.10 forward map, no derivation.
- **Hierarchical priors, partial pooling, empirical Bayes** → defer to Topic 28 (Hierarchical Bayes). Topic 25 contrasts "subjective prior" vs "hierarchical prior" in one remark to set up Topic 28.
- **Variational methods** → defer to formalml (Topic 24's S1 cleanup retargeted the EM-mixture variational pointer there). Topic 25 mentions VI by name only in the §X.10 forward map.
- **Bayesian decision theory** (loss functions, posterior expected loss, Bayes risk) → defer to formalml or a later track. Topic 25 mentions decision theory only as the formal home of "posterior mean as point estimator under squared-error loss."
- **Reference priors, intrinsic priors, Berger-Pericchi machinery, objective Bayes formal frameworks** → mention by name in the §X.10 forward map, defer full treatment.
- **Improper-prior pathologies and marginalization paradoxes** → state the Stone-Dawid-Zidek paradox in one remark; defer the formal study to Topic 27 + formalml.
- **Predictive checks, posterior predictive p-values, model criticism** → preview in §X.10 only; defer to Topic 27 (Bayesian model comparison includes posterior predictive checks).

The temptation is to make Topic 25 the catch-all "intro to Bayesian statistics" that covers everything in 14k words. Resist — Track 7 has three more topics to absorb the rest.

---

## What I want from this conversation

**Primary deliverable:** a complete handoff brief in markdown, structured per the established 10-section template (see "Brief structure" below). This brief becomes `docs/formalstatistics-bayesian-foundations-and-prior-selection-handoff-brief.md` in the repo and is the single source of truth for the downstream Claude Code implementation session.

**Secondary deliverable:** a Jupyter notebook outline (cells specified, full code if time permits) that produces the brief's figure manifest. Lives at `notebooks/bayesian-foundations-and-prior-selection/25_bayesian_foundations.ipynb`.

**Optional tertiary deliverable:** a scaffold document (`docs/topic-25-scaffold.md`) — a quick-reference companion to the brief with locked decision rationale and citation shortlist. Useful but not strictly required.

**By session end, I should be able to:**
1. Save the brief to `docs/`.
2. Save the notebook (or its outline) to `notebooks/bayesian-foundations-and-prior-selection/`.
3. Hand both to a Claude Code session via the companion starter-prompt template, with no remaining design decisions.

---

## Forward-promise harvest

Before drafting any new content, **find every place in the shipped MDX where this topic was promised**. These are non-negotiable contracts — the brief's §1.3 lists them with verbatim quotes, and the implementation must fulfill each.

**To harvest** (we'll do this together at the start of the conversation):

```bash
# Pattern 1: explicit "(coming soon, Topic 25)" markers
grep -rn 'coming soon, Topic 25' src/content/topics/

# Pattern 2: name-based mentions (Bayesian foundations / prior selection / Track 7)
grep -rn 'Topic 25\|bayesian-foundations\|prior selection\|conjugate prior\|Bernstein-von Mises\|posterior\|Track 7' src/content/topics/

# Pattern 3: forward-pointer hyperlinks to the provisional slug
grep -rn '/topics/bayesian-foundations' src/content/topics/

# Pattern 4: "Track 7 (Bayesian Foundations, coming soon)" — the literal forward-promise wording Topic 14, 22, 23, and 24 all use
grep -rn 'Track 7 (Bayesian' src/content/topics/
```

Expected count: at least 15 unique forward-promises (Topic 24 alone planted Rem 7 BMA flag, Rem 8 priors-on-model-space pointer, Rem 23 BMA full pointer, Rem 26 DIC/WAIC/PSIS-LOO pointer, Rem 31 Track 7 on-ramp paragraph; predecessors Topics 14, 17, 21, 22, 23 all carry "Track 7" or "Bayesian Foundations, coming soon" forward-pointers as well). Each entry in §1.3 of the brief should be a row in a table with columns: Source location | Verbatim text | Topic 25 deliverable.

**Note for this session specifically:** because Topic 25 opens a new track AND has been forward-promised heavily by Track 6, the harvest table will be unusually long. Budget 20–30 minutes for the grep + verbatim-extraction step.

---

## Decisions to lock explicitly during this session

The brief is a contract. Every entry in this list needs an answer before the conversation is done; defer none of them to the Claude Code session.

| Decision | Default if no opinion | Notes |
|---|---|---|
| **Slug** | `bayesian-foundations-and-prior-selection` | Topic 18/19/23/24 set the precedent of expanding to `slug-and-X` when there's a "& X" half. The provisional `bayesian-foundations` is in `curriculum-graph.json:33` — needs a deliberate update either way. Track 7's other three topic slugs (`bayesian-computation`, `bayesian-model-comparison`, `hierarchical-bayes`) should also be reviewed for "& X" expansion in this session even if they aren't being implemented yet, so the forward-pointers Topic 25 emits use the locked slugs. |
| **Difficulty** | `intermediate` | Track 5+ is intermediate by default. Bayesian Foundations is genuinely no harder than MLE asymptotics — accessible to anyone who absorbed Track 4. |
| **Read time / word count** | 60–65 min / 10–11K words | Track openers tend to run slightly long because they set notation for an entire track. Topic 21 (Track 6 opener) was 60-min/10.5K. Topic 25 is comparable. |
| **Number of sections** | 10 | One section per major construction + a §25.1 motivation + a §25.10 forward-look closer. |
| **Number of full proofs** | 3–4 | Pick the theorems where the *proof* carries pedagogical weight, not just the statement. Strong candidates: (1) conjugacy for the canonical exponential family (general form), (2) Bernstein–von Mises (sketch-quality if full version is too long; brief should specify), (3) posterior consistency under regularity (Doob's theorem stated; sketch-quality). |
| **Number of theorems (proofed + stated)** | 7–9 | Track openers carry more stated theorems than mid-track topics because they introduce notation. |
| **Number of examples** | 14–18 | Conjugate-pair examples (Beta-Binomial, Normal-Normal, Gamma-Poisson, Dirichlet-Multinomial, Wishart-Normal) alone fill 5–6 examples. Plus prior-elicitation, prior-sensitivity, and asymptotic-comparison examples. |
| **Number of remarks** | 22–28 | Remarks carry the connective tissue: scope boundaries, alternative interpretations (subjective vs objective Bayes), ML-context callouts (variational inference, conjugate-prior shortcuts in modern deep learning). |
| **Number of figures** | 9–11 | The featured figure is the Bernstein–von Mises convergence (or whichever featured theorem we lock) — the visual where prior/posterior/MLE-Normal overlay as $n$ grows. |
| **Number of interactive components** | 3 required + 1 optional | Locked across recent topics. Likely candidates: a **PriorPosteriorExplorer** (slide hyperparameter, watch posterior update with each new data point), a **ConjugatePairBrowser** (pick a likelihood, see the conjugate prior + posterior closed form), and a **BernsteinVonMisesAnimator** (sample size slider, posterior overlaid with $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}^{-1}/n)$). The optional fourth could be a **PriorSensitivityComparator** for Topic 25's prior-elicitation worked example. |
| **Featured theorem** | Bernstein–von Mises (proposed) | The single result the topic is *about*. Drives the featured figure and the featured component. Alternative: the conjugacy theorem for the exponential family — more constructive, easier to teach, less deep. |
| **Featured component** | BernsteinVonMisesAnimator (proposed, paired with the featured theorem) | The flagship interactive. Goes in the section housing the featured theorem. |
| **Citations** | (open) | New entries needed: Bernardo–Smith *Bayesian Theory* (1994); Gelman et al. *Bayesian Data Analysis* (2013, 3rd); Robert *The Bayesian Choice* (2007, 2nd); Jeffreys *Theory of Probability* (1961, 3rd) for the prior-selection historical foundation; Lindley *Understanding Uncertainty* (2014) for the subjective-Bayes anchor; van der Vaart *Asymptotic Statistics* §10 for the Bernstein–von Mises proof; Diaconis–Freedman 1986 for Doob-type consistency caveats. Reused entries: Casella–Berger §7.2, §10.4 (decision-theoretic Bayes); Lehmann–Casella §4 (Bayes estimation); Hastie-Tibshirani-Friedman §8.3 (MAP shrinkage). |
| **Scope boundaries** | (open) | What this topic does NOT cover — see "Initial scope boundaries" above. Each boundary becomes a one-paragraph forward-pointing remark in §25.10. |

---

## Brief structure (the 10-section contract)

The handoff brief follows this structure, established by Topics 17, 18, and 19. Don't deviate without flagging — Claude Code expects these section numbers.

```
§ 1   Important: 25th Topic — Position in Track (Track 7 opener)
  §1.1   Scope boundary (what this topic covers; what it does NOT)
  §1.2   Forward-promise fulfillment table (Source | Verbatim text | Deliverable)
§ 2   Frontmatter
  §2.1   Full YAML spec (every reference with verified URL)
  §2.2   Length target rationale
§ 3   Content Structure
  §3.1   Section map (table: § | Title | Formal elements | Figure | Component)
  §3.2   Proof specifications (full text of every full proof, ready to render)
  §3.3   Notation conventions (introducing the Bayesian notation used throughout
         Track 7: π(θ), p(θ|y), p(y_new|y), m(y), Bayes factor, etc.)
§ 4   Static Figures & Notebook Structure
  §4.1   Figure manifest (table: # | File | Section | Dimensions | Description)
  §4.2   Notebook structure (cell-by-cell)
§ 5   Interactive Components
  §5.1...§5.N   Per-component spec (purpose, interactions, data, dependencies, mobile)
  §5.last       Implementation notes (architecture, palette, performance)
§ 6   Shared Module — likely creating a new `bayes.ts` (Track 7 leaf module)
  §6.1   Full new-function manifest (TypeScript signatures with JSDoc)
         Likely surface: posterior() for conjugate pairs (dispatch on family),
         posteriorPredictive(), credibleInterval(), bayesFactorLaplace(),
         posteriorSample() (rejection / inverse-CDF for conjugate cases —
         MCMC defers to Topic 26).
  §6.2   New test-harness entries (numeric IDs, expected values, tolerances)
§ 7   Preset Data Module — `src/data/bayesian-foundations-data.ts` (new)
  §7.1   Full TypeScript content (preset hyperparameters for each component;
         the canonical Beta-Binomial, Normal-Normal, Gamma-Poisson examples
         the worked-examples section walks through)
§ 8   Cross-Reference Updates
  §8.1   Predecessor MDX edits (table: # | Source | Find | Replace)
         Especially Topic 24 §24.4 Rems 7, 8, 10 + §24.10 Rem 23, 26, 31 —
         all the Track-7 pointers Topic 25 now activates.
  §8.2   Curriculum-graph.json edits (slug change if locked + status flip
         to "published"; expand prerequisites; preserve `id: "bayesian-foundations"`
         per Topic 23/24 edge-stability precedent)
  §8.3   curriculum.ts edits (track status — Track 7 opens here)
  §8.4   References spreadsheet edits (rows to add + rows to update)
  §8.5   Sitemap / RSS verification
§ 9   Verification Checklist
  Content / Proofs / Forward-promise / Components / Cross-refs / References / Build
§ 10  Build Order
  Numbered steps the Claude Code agent executes sequentially.

Appendix A: Style Guidelines (KaTeX, notation, figure callouts, cross-refs)
Appendix B: Design Decisions (locked decisions with rationale — prevents re-litigation)
```

**Critical brief requirements:**
- Every reference URL must resolve. Verify before locking.
- Every full proof in §3.2 must be ready to render in MDX — no "TBD" or "[insert algebra here]."
- The §3.1 section map's per-section formal-element counts must add up to the totals stated in §1.
- The §6.2 test entries must have specific expected values with tolerances; downstream Claude Code uses them as a regression suite.
- §8.1's find/replace strings must be verbatim copy-paste-ready, including markdown syntax.
- **Track-7 opener requirement (special):** §3.3 (notation conventions) needs to lock the Bayesian notation that ALL of Track 7 will inherit — π(θ) for prior, p(θ|y) for posterior, p(y_new|y) for posterior predictive, m(y) for marginal likelihood, K(θ ‖ θ') for KL. Get this right once and Topics 26–28 inherit cleanly. Wrong notation here cascades.

---

## Notebook structure (the cell contract)

**Before drafting the cell-by-cell outline below, read the prior notebooks listed in "Existing notebooks worth studying" above.** Look at: cell granularity (one figure per cell vs grouped), palette setup conventions, how MC simulations are seeded and looped, and how the verification helper at the end exposes ground-truth values for the §6.2 test pins. Mirror the pattern that fits the topic best — don't reinvent.

The notebook produces the figure manifest from §4.1. Cell structure mirrors Topics 17–19:

```
Cell 1   Imports + seed
         numpy, scipy.stats, matplotlib.pyplot, np.random.default_rng(42)
         OUTPUT_DIR (relative to notebook CWD; figures saved with savefig)

Cell 2   Palette + matplotlib defaults
         Reuse Track 5/6 palette where possible; introduce a Bayesian-specific
         "prior=blue, likelihood=amber, posterior=purple" trio that Topics 26–28
         will inherit. rcParams: dpi 150, savefig.bbox 'tight', spines top/right off.

Cell 3   Figure 1 (the §25.1 motivation figure — likely a sequential prior→posterior
         update on the Beta-Binomial coin-toss example, with the prior fading and
         the posterior sharpening over n=1, 5, 20, 100 observations)
Cell 4   Figure 2 (often the §25.2 featured-theorem figure — Bernstein–von Mises
         convergence, posterior + Normal-MLE approximation overlaid as n grows)
...
Cell N+2 Figure N
Cell N+3 Verification helper (computes ground-truth values cited in §6.2 tests:
         posterior parameters for each conjugate pair, credible-interval bounds,
         Bayes factor reference values)
Cell N+4 Markdown summary (figure list + numerical values for cross-checking the MDX)
```

**Notebook conventions:**
- Every figure: `dpi=150, bbox_inches='tight', pad_inches=0.15`.
- Filename pattern: `25-{kebab-description}.png`.
- Output directory written to `public/images/topics/bayesian-foundations-and-prior-selection/` (relative path — drops files in `notebooks/bayesian-foundations-and-prior-selection/public/...` if you run from the notebook directory; manual move-to-public is part of the Claude Code build step).
- `np.random.default_rng(42)` at module top (NOT legacy `np.random.seed(42)` — Topic 24's nested-Poisson example used `default_rng` and the codebase has standardized there; legacy global seed is for Topic 17–22 backward compatibility).
- Every figure title is short and self-explanatory; the alt-text in the MDX repeats the substantive content for accessibility.
- KaTeX vs matplotlib mathtext gotcha: `\iff` works in KaTeX but NOT in matplotlib — use `\Leftrightarrow` in figure titles.
- For Track 7 specifically, decide on the prior/posterior color convention in Cell 2 — it gets reused across Topics 26–28.

---

## Style locks (don't waste cycles re-deriving these)

**KaTeX constraints:**
- No `\begin{aligned}` blocks. Multi-line derivations use separate `$$...$$` blocks with prose connectors.
- No `\begin{array}` tables. Use markdown tables.
- Inline math `$...$`; display math `$$...$$` on its own line.

**MDX constraints (these have bitten us):**
- Curly braces `{...}` in MDX body text or image alt-text are parsed as JSX expressions. If the content includes Greek letters, subscript Unicode (₀, ₁, ₂), or Unicode minus (`−`), the JSX parser fails. Reword without curly braces in alt-text and prose; use math `$\{...\}$` if you need set notation.
- Cross-site links: use the `<ExternalLink>` Astro component for formalCalculus / formalML, not raw markdown links.
- Section anchors follow the convention `#section-25-X` (used by predecessors' forward-pointers, even though no rehype-slug plugin is registered; this is technical debt to be addressed separately).
- **JSX comment gotcha (new in Topic 24):** `{/* … */}` JSX comments in the MDX import block (between the frontmatter and the first markdown heading) cause `MDXError: Unexpected BlockStatement in code: only import/exports are supported`. JSX comments are fine in the body, NOT in the import block.
- **`<ProofBlock>` accepts `number?` and `title?` props** as of Topic 24 (PR #28, commit 67948f4). Use `<ProofBlock number={N} title="...">` for consistency with TheoremBlock / DefinitionBlock / ExampleBlock / RemarkBlock — all of which already carry these props.

**Citations (Chicago 17th, Notes & Bibliography):**
- Books: `N. Author(s). (Year). [*Title*](URL) (Edition). Publisher.`
- Articles: `N. Author(s). (Year). [*Title*](URL). *Journal*, Volume(Issue), Pages.`
- Every frontmatter `references:` entry must have a `url:` field with a working hyperlink (verify each).
- The MDX must end with a rendered `### References` section listing every entry from frontmatter as a numbered markdown list, preceded by a `---` horizontal rule. Non-negotiable.

**Components:**
- `client:visible` directive (not `client:load`).
- React + Tailwind + SVG/D3 only. No Chart.js, Plotly, Recharts.
- Mobile-responsive at 375px width.
- Use the Track palette from `viz/shared/colorScales.ts` or the topic-specific palette established in the notebook. Topic 25 likely needs a new `bayesianColors` palette in `colorScales.ts` mirrored to CSS vars in `global.css` (Topic 22 / Topic 24 GLM and IC palettes set the precedent — `--color-prior`, `--color-likelihood`, `--color-posterior` is the natural Track-7 trio).

**End-of-proof citation style:**
- `∎ — using [list of specific theorems/papers]` — uniformly applied across every full proof.

---

## Scope-boundary discipline

Every topic has a temptation to over-cover. The brief's §1.1 makes scope explicit by enumerating both *what's in* and *what's out*. The "what's out" list is just as important — each deferral becomes a one-paragraph forward-pointing remark in §X.10 and is the only acceptable response to "shouldn't we cover Y?".

**Default deferrals to consider:**
- **Modern extensions** (post-2010 literature) → usually defer to formalml.com or to a later track. Rationale: these depend on infrastructure not yet built (e.g., e-processes need martingale theory; conformal needs exchangeability machinery).
- **Vector-θ generalizations** → state the result, cite Lehmann-Romano 2005 Ch. 7 / Ch. 12 / Ch. 8 (depending on topic), don't derive. Most main ideas are scalar-clean.
- **Bayesian counterparts** → contrast briefly, defer full treatment to Track 7. (For Topic 25 itself, this is reversed: it IS the Bayesian counterpart; the frequentist counterparts in Topics 13–24 are cited as the baseline being extended.)
- **Nonparametric / distribution-free analogs** → defer to Track 8.
- **Computational / algorithmic details** → cite the algorithm, don't re-derive convergence proofs. For Topic 25 specifically: MCMC, VI, importance sampling all defer to Topic 26 / formalml.

The bar for inclusion: *Does this result substantially change how the reader thinks about the topic's main object, or is it a downstream specialization that can be safely cited?* Specializations defer; reframing results stay.

**Topic-25-specific scope discipline:** Bayesian foundations is the topic where it's hardest to draw the line. Be especially strict about:
- Don't drift into Bayesian decision theory (formalml).
- Don't develop the philosophy debate (subjective vs objective Bayes vs frequentist) past a one-remark contrast — Lindley 2014 is cited for the philosophical anchor; the math content is what stays.
- Don't include any MCMC algorithm even by name in §§25.2–25.9 — Topic 26 is the dedicated home and the §25.10 forward map is the only place MCMC appears.

---

## Output format

**Brief:** Single markdown file. Reply with the complete brief in one go (or in clearly-marked chunks if length-limited), copy-paste-ready to save as `docs/formalstatistics-bayesian-foundations-and-prior-selection-handoff-brief.md`.

**Notebook:** Either a complete `.ipynb` JSON or — preferred — a cell-by-cell outline I can paste into a Jupyter session and fill in iteratively. The latter is usually faster because you can flag the algorithmic decisions for me to make rather than guessing.

**Scaffold (optional):** A short companion markdown — typically 200–400 lines — restating the locked decisions with one-line rationale each. Useful for the Claude Code session as a quick-reference second window.

---

## When the conversation is done

By the end of this session, I should have:

- [ ] **Brief** saved at `docs/formalstatistics-bayesian-foundations-and-prior-selection-handoff-brief.md` with all 10 sections + Appendix A + Appendix B
- [ ] **Notebook** saved at `notebooks/bayesian-foundations-and-prior-selection/25_bayesian_foundations.ipynb` (or its outline)
- [ ] **Slug** locked (and `curriculum-graph.json:33` prepped for the change from `bayesian-foundations` → `bayesian-foundations-and-prior-selection`, preserving the existing `id: "bayesian-foundations"` for edge stability per Topic 23/24 precedent)
- [ ] **Featured theorem and featured component** named explicitly
- [ ] **Forward-promise table** complete with verbatim quotes from grep — expect 15+ entries given Topic 24 alone planted multiple Track-7 pointers
- [ ] **Citation list** complete with all URLs verified — Bernardo–Smith, Gelman BDA, Robert, Jeffreys, Lindley, van der Vaart §10, Diaconis–Freedman 1986 (new) plus Casella–Berger, Lehmann–Casella, HTF (reused-with-update)
- [ ] **Scope boundaries** named with deferral pointers — especially the four other Track-7 topics still planned
- [ ] **Track-7 notation lock** — π(θ), p(θ|y), p(y_new|y), m(y), K(θ ‖ θ') frozen here; Topics 26–28 inherit
- [ ] **Track-7 color palette lock** — `--color-prior`, `--color-likelihood`, `--color-posterior` CSS vars proposed, mirrored to `bayesianColors` in `colorScales.ts`
- [ ] No "TBD," "open question," or "decide later" markers anywhere in the brief

Once those are in place, I open a fresh Claude Code session with the [Claude Code starter prompt template](claude-code-starter-prompt-template.md) (filled in for Topic 25), and the implementation runs mostly mechanically against the brief.

---

## If we get stuck mid-session

- **On scope:** ask "what's the smallest version of this topic that fulfills every forward-promise from prior topics?" That's the floor; everything else is a matter of taste.
- **On a proof:** if a proof you want to include is longer than ~20 MDX lines, either it needs to become a stated-only theorem with a citation, or the topic is over-scoped. Bernstein–von Mises in particular is famous for full proofs running 5+ pages — sketch-quality is the right call here unless we want to expand the topic past 11K words.
- **On a component:** if you can't sketch the interaction in 4 bullet points (controls, what updates, what the readout says, what the mobile fallback is), the component is under-specified — discuss before locking.
- **On a numerical value for a test:** verify against scipy / R / Wolfram before locking; "approximately X" doesn't survive Claude Code's regression test.
- **On Track-7-wide infrastructure decisions** (the Bayesian color palette, the canonical notation for π(θ), the shared module name `bayes.ts` vs `bayesian.ts` vs splitting): these are load-bearing across four topics — flag back to me explicitly. Wrong choice here recurs three more times.

When genuinely stuck (the question isn't in any prior brief and isn't obvious from the published MDX), flag the decision back to me explicitly rather than picking a default. Brief decisions are load-bearing; silent defaults compound.

---

*Starter prompt version: v1 · Topic 25: Bayesian Foundations & Prior Selection*
*Companion: [claude-code-starter-prompt-template.md](claude-code-starter-prompt-template.md) (downstream)*
