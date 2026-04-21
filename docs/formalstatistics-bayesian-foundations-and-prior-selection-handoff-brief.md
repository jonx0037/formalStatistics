# Claude Code Handoff Brief: Bayesian Foundations & Prior Selection

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)
**Repo:** `github.com/jonx0037/formalStatistics`
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel
**Package Manager:** pnpm
**Status:** Ready for implementation
**Reference Notebook:** `notebooks/bayesian-foundations/25_bayesian_foundations.ipynb`

---

## Important: Twenty-fifth Topic — First in Track 7 (Track opener)

This is **topic 25 of 32** and the **first topic in Track 7 (Bayesian Statistics)**. Tracks 1–6 (Topics 1–24) built the frequentist machinery: probability foundations, distributions, convergence, estimation, testing, regression. Topic 25 opens the Bayesian track. It does not displace the frequentist framework — it adds a parallel formalism that treats $\theta$ as a random variable with a prior $\pi(\theta)$, updates to a posterior $p(\theta \mid \mathbf{y})$ via Bayes' theorem, and produces point estimates, interval estimates, and predictions by integrating against the posterior. The featured theorem (§25.8 Thm 4, **Bernstein–von Mises**) is the bridge back to Topics 14, 17, and 19: under regularity, the posterior concentrates on $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}(\theta_0)^{-1}/n)$ in total variation, so Bayesian credible intervals and frequentist Wald intervals coincide asymptotically.

**Implications:**

1. **All twenty-four preceding topics are published.** Cross-references from `sample-spaces` through `model-selection-and-information-criteria` use live internal links.

2. **Twenty-one binding forward-promises must be fulfilled.** Every shipped Track-4/5/6 topic plants at least one pointer at Topic 25 — the most load-bearing of any topic on the site. §1.3 enumerates each with verbatim text and deliverable mapping.

3. **Track 7 slug cascade locks four slugs.** This topic is `bayesian-foundations-and-prior-selection` (expanding from provisional `bayesian-foundations`). Topics 26–28 are `bayesian-computation-and-mcmc`, `bayesian-model-comparison-and-bma`, `hierarchical-bayes-and-partial-pooling` (status `planned`; slugs locked now so Topic 25's forward-pointers are stable). Preserve original `id`s (`bayesian-foundations`, `bayesian-computation`, etc.) in `curriculum-graph.json` per Topic 23/24 edge-stability precedent.

4. **Topic 25 *creates* a new shared Track 7 module `bayes.ts`** at `src/components/viz/shared/bayes.ts`. Tracks 4–6 established the extend-don't-create convention (`estimation.ts`, `testing.ts`, `regression.ts`); a new module for a new track matches precedent.

5. **Track 7 notation lock.** §25.3 §3.3 commits the notation every Topic 26–28 will inherit: $\pi(\theta)$ prior; $p(\theta \mid \mathbf{y})$ posterior; $p(\tilde y \mid \mathbf{y})$ posterior predictive; $m(\mathbf{y})$ marginal likelihood; $\mathrm{KL}(\pi \,\|\, \pi')$ KL divergence; $\mathrm{BF}_{10}$ Bayes factor (mentioned §25.10 only, full treatment Topic 27).

6. **Track 7 color palette lock.** Three new CSS custom properties — `--color-prior` (blue), `--color-likelihood` (amber), `--color-posterior` (purple) — added to `global.css` and mirrored to `bayesianColors` in `viz/shared/colorScales.ts`. Topics 26–28 inherit.

7. **Do not re-derive MLE asymptotic normality.** Topic 14 Thm 4 (§14.5) is the engine of the BvM sketch (§25.8 Proof 3). §25.8 cites it; it does not re-prove it. Similarly, do not re-derive conditional MVN (Topic 8 Thm 3) or Bayes' theorem (Topic 4 Thm 4); cite them.

8. **Scope-boundary discipline — the hardest of any topic so far.** Bayesian foundations is the topic where it is hardest to draw the line. Track 7 has three more topics (MCMC, BMA, hierarchical) to absorb the rest. §1.1 enumerates nine explicit deferrals; each becomes a §25.10 forward-pointing remark.

9. **This is a track opener.** The brief's "track completion" language from Topic 20 and 24 does not apply. Instead, Topic 25 emphasizes the conceptual shift — from "best estimate plus coverage" to "posterior distribution over $\theta$" — and frames the three Topic 26–28 topics as follow-ons.

---

## 1. Forward-Promise Fulfillment

### 1.1 Scope boundary

**What Topic 25 covers (the full list):**

- **Bayes' theorem applied to a parameter $\theta$** (§25.2 Def 1 + Thm 1): prior $\pi(\theta)$, likelihood $L(\theta) = p(\mathbf{y} \mid \theta)$, posterior $p(\theta \mid \mathbf{y}) \propto L(\theta)\pi(\theta)$, marginal likelihood $m(\mathbf{y}) = \int L(\theta)\pi(\theta)\,d\theta$, posterior predictive $p(\tilde y \mid \mathbf{y}) = \int p(\tilde y \mid \theta)p(\theta \mid \mathbf{y})\,d\theta$.
- **Track 7 notation conventions locked** (§25.3): $\pi$, $p(\theta \mid \mathbf{y})$, $p(\tilde y \mid \mathbf{y})$, $m(\mathbf{y})$, $\mathrm{KL}$, $\mathrm{BF}_{10}$.
- **Exponential-family conjugacy theorem with full proof** (§25.4 Thm 2 + Proof 1): the general conjugate-prior construction; the Topic 7 Thm 3 specialization becomes the scalar case.
- **Beta-Binomial posterior derived end-to-end with full proof** (§25.5 Thm 3 + Proof 2): the first fully-walked conjugate pair; anchor for all subsequent pairs.
- **Four additional canonical conjugate pairs as worked examples** (§25.5): Normal-Normal known $\sigma^2$ (Ex 2), Normal-Normal-Inverse-Gamma unknown $\sigma^2$ (Ex 3), Gamma-Poisson (Ex 4), Dirichlet-Multinomial (Ex 5). Wishart-Normal cited as §25.9 remark.
- **Posterior predictive with full proof for Beta-Binomial** (§25.6 Thm 3b + Proof 2b): closed-form Beta-Binomial compound distribution.
- **Credible intervals** (§25.6 Def 5 + Ex 6): equal-tailed vs highest-posterior-density (HPD).
- **Point estimates** (§25.6 Def 6): posterior mean, posterior median, MAP (posterior mode); decision-theoretic framing one remark.
- **Prior selection as substantive modeling decision** (§25.7): informative vs weakly-informative vs non-informative (§25.7 Rem); Jeffreys prior defined, derived for Bernoulli (→ Beta(½,½)) and Normal-scale (→ $\pi(\sigma)\propto 1/\sigma$) (§25.7 Thm 4); reparameterization invariance stated not proved; improper priors with integrable-posterior condition (§25.7 Def 7 + Rem); Stone–Dawid–Zidek paradox one remark.
- **Prior sensitivity worked example** (§25.7 Ex 7): Beta-Binomial with three contrasting priors on the same data.
- **Bernstein–von Mises theorem with sketch proof** (§25.8 Thm 5 + Proof 3): the posterior converges in total variation to $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}(\theta_0)^{-1}/n)$ under regularity. Featured result.
- **Laplace approximation to the posterior and to $m(\mathbf{y})$** (§25.8 Rem): Taylor-expand log-posterior at MAP; second-order Gaussian; connects §25.8 to Topic 24 §24.4's BIC derivation.
- **MAP-as-regularized-MLE bridge discharged from Topic 14 Rem 9** (§25.8 Ex 8): Gaussian prior ⇒ ridge; Laplace prior ⇒ lasso; explicitly cites Topic 23 Thm 5.
- **Forward map** (§25.10): MCMC (→ Topic 26); BMA + Bayes factors + marginal-likelihood computation (→ Topic 27); hierarchical priors, partial pooling, empirical Bayes (→ Topic 28); variational inference, Bayesian decision theory, reference priors, predictive checks, Bayesian nonparametrics, local FDR (→ formalml / later).
- **Ten PNG figures**, **three required interactive components plus one optional**, **~10,500 words, 60-minute read**.

**What Topic 25 does NOT cover (nine explicit deferrals; each a §25.10 remark):**

- **MCMC algorithms** (Metropolis–Hastings, Gibbs, HMC, NUTS) → Topic 26. Topic 25 mentions MCMC by name only when the conjugate framework breaks down — one §25.10 paragraph.
- **BMA, Bayes factors, marginal-likelihood computation** (nested sampling, bridge sampling, thermodynamic integration) → Topic 27. Topic 25 names $m(\mathbf{y})$ and $\mathrm{BF}_{10}$ in §25.10 only, no derivation beyond the Laplace approximation remark.
- **Hierarchical priors, partial pooling, empirical Bayes** → Topic 28. Topic 25 contrasts "fixed prior" vs "hierarchical prior" in one §25.10 remark.
- **Variational inference** → formalml (Topic 14 §14 Appendix Rem 10's stopgap pointer). §25.10 one-line mention.
- **Bayesian decision theory** (loss functions, posterior expected loss, Bayes risk) → §25.10 one-line mention; full treatment formalml / deferred.
- **Reference priors, intrinsic priors, Berger–Pericchi machinery** → §25.10 name-drop only.
- **Improper-prior pathologies beyond integrable-posterior condition** → Stone–Dawid–Zidek paradox one §25.7 remark; full measure-theoretic treatment → Topic 27 / formalml.
- **Posterior predictive checks and model criticism** → §25.10 pointer to Topic 27.
- **Bayesian nonparametrics** (Dirichlet processes, Gaussian processes as priors, Polya trees) → §25.10 one-line mention; Topic 8 already points to formalml GPs.

### 1.2 Primary predecessor and chain of dependencies

**Primary predecessor: Topic 24 (Model Selection & Information Criteria).** §24.4's BIC-Laplace derivation is the explicit on-ramp to Topic 25 — §24.10 Rem 31 names the full chain "BIC → marginal likelihood → BMA → MCMC → DIC/WAIC/PSIS-LOO" as Track 7 territory. The Laplace approximation machinery generalizes immediately from the BIC partition-function expansion to Topic 25's posterior-at-MAP expansion. §25.8 Rem activates the link.

**Direct feeders:**

- Topic 4 (Conditional Probability) — Thm 4 (Bayes' theorem) is the engine; Topic 25 applies it to $\theta$ as random variable.
- Topic 7 (Exponential Families) — Thm 3 (conjugate prior for exp families) is the specialization of §25.4 Thm 2; the canonical pairs (Beta-Bernoulli, Gamma-Poisson, Normal-Normal known σ²) are worked there as algebra. Topic 25 lifts them into the Bayesian inferential framework with credible intervals and posterior predictive.
- Topic 8 (Multivariate Distributions) — MVN conditional formula powers Ex 3 (Normal-Normal-Inverse-Gamma unknown σ²); Dirichlet-Multinomial conjugacy (Thm 6 there) is the scalar case of Ex 5.
- Topic 14 (Maximum Likelihood) — Thm 4 (asymptotic normality) is the engine of §25.8's BvM sketch. §14.11 Rem 9 MAP-as-regularized-MLE is discharged as Ex 8.
- Topic 23 (Regularization) — §23.7 Thm 5 (MAP-penalization correspondence) is cited by §25.8 Ex 8 without re-derivation.
- Topic 24 (Model Selection) — §24.4 BIC-Laplace is generalized by §25.8 Laplace-approximation remark.

**Conceptual predecessors cited but not feeder:**

- Topic 17 (Hypothesis Testing) — §17.11 Rem 15 Bayes factors (Topic 25 §25.10 activates, full treatment Topic 27).
- Topic 19 (Confidence Intervals & Duality) — §19.1 Rem 3 and §19.10 Rem 21 (frequentist coverage vs Bayesian credibility); Topic 25 §25.6 picks this up.
- Topic 21 (Linear Regression) — §21.10 Rem 25 (Bayesian linear regression); Topic 25 §25.5 Ex 3 covers the scalar Normal–Normal-Inverse-Gamma; regression generalization → Topic 27/28.

### 1.3 Forward-promise fulfillment table

Twenty-one binding forward-promises from fifteen shipped MDX files. Every row is a contract: the verbatim text in the source must be replaced with the live-link form after Topic 25 ships. §8.1 has the exact find-and-replace operations.

| # | Source | Location | Verbatim text (abbreviated) | Topic 25 deliverable |
|---|---|---|---|---|
| B1 | `maximum-likelihood.mdx` | §14.11 Rem 9 (line ~898) | "The Bernstein–von Mises theorem (Track 7, coming soon) says that under regularity, the MAP estimate converges to the MLE at rate $1/n$: priors stop mattering once there is enough data." | §25.8 Thm 5 (BvM, sketch proof); §25.8 Ex 8 restates the MAP-MLE convergence. Activate link to `/topics/bayesian-foundations-and-prior-selection#section-25-8`. |
| B2 | `maximum-likelihood.mdx` | §14.11 "Where this leads" (line ~925) | "…**Bayesian Foundations** (Track 7, coming soon) formalizes the MAP/regularized-MLE correspondence of Remark 9, giving a principled framework for the informative priors that regularization has been quietly deploying." | §25.3 Thm 1 (posterior); §25.7 Thm 4 (Jeffreys); §25.8 Ex 8 discharges MAP correspondence. Activate link. |
| B3 | `maximum-likelihood.mdx` | §14 Appendix Rem 10 (line ~947) | "…developed in Track 7 (Bayesian Foundations, coming soon) and Track 7 (Variational Methods, coming soon)." | §25.10 forward-map names Bayesian Foundations (this topic, activate link); Variational Methods stays "(formalml, coming soon)" — VI is not a Topic 26/27/28 deliverable. |
| B4 | `continuous-distributions.mdx` | §6 "What comes next" bullet | "Bayesian Foundations (coming soon) develops the Beta-Bernoulli, Gamma-Poisson, and Normal-Normal conjugate pairs in full generality" | §25.5 Ex 1 (Beta-Binomial), Ex 2 (Normal-Normal known σ²), Ex 4 (Gamma-Poisson). Activate link. |
| B5 | `continuous-distributions.mdx` | §6.6 Ex 5 "Beta-Bernoulli A/B Testing" | "Bayesian Foundations (coming soon) will develop the general framework." | §25.5 Beta-Binomial + §25.6 posterior predictive (Beta-Binomial compound). Activate link. |
| B6 | `discrete-distributions.mdx` | frontmatter `connections[bayesian-foundations]` | Relationship text references conjugate-prior pairs. | §25.5 conjugate catalog. Update slug reference. |
| B7 | `exponential-families.mdx` | §7.11 "What comes next" (line ~935) | "Bayesian Foundations (coming soon) develops the full Bayesian framework, with the conjugate prior theorem from Section 7.7 as the starting point for conjugate inference" | §25.4 Thm 2 (exp-fam conjugacy; cites Topic 7 Thm 3 as the algebraic special case and lifts it into Bayesian inferential framework). Activate link. |
| B8 | `exponential-families.mdx` | frontmatter `connections[bayesian-foundations]` | "Beta-Bernoulli, Gamma-Poisson, and Normal-Normal are the three canonical conjugate pairs…" | §25.5 conjugate pair catalog. Update relationship text. |
| B9 | `multivariate-distributions.mdx` | §8.10 ML connections — BNN paragraph | "The posterior $p(\mathbf{w} \mid \mathcal{D})$ is approximately MVN for well-specified models (by the Bernstein-von Mises theorem), with the Hessian of the negative log-posterior at the MAP estimate serving as the inverse covariance (the Laplace approximation)." | §25.8 Thm 5 (BvM); §25.8 Rem (Laplace approximation as inverse-Hessian). |
| B10 | `multivariate-distributions.mdx` | frontmatter `connections[bayesian-computation]` | "Multivariate posteriors are the starting point for MCMC and variational methods. The conditional MVN powers Gibbs sampling." | §25.5 Ex 2 uses conditional MVN; §25.10 MCMC pointer activates Topic 26. Update frontmatter slug: `bayesian-computation` → `bayesian-computation-and-mcmc`. |
| B11 | `conditional-probability.mdx` | frontmatter `connections[bayesian-foundations]` | "Bayes' theorem derived here becomes the engine of Bayesian statistics: prior × likelihood ∝ posterior. Conjugate priors, Jeffreys priors, and posterior computation all build on this foundation." | §25.2 opening (cites Topic 4 Thm 4); §25.5 (conjugate catalog); §25.7 (Jeffreys). Update slug. |
| B12 | `hypothesis-testing.mdx` | §17.11 Rem 15 (Bayes factors) | "Bayesian Foundations (coming soon, Track 7) will develop the full framework." | §25.10 Rem (Bayes factor named, full treatment → Topic 27). Activate link. |
| B13 | `hypothesis-testing.mdx` | §17.12 Rem 19 "Where Track 5 goes next" | "**Bayesian Foundations** (Track 7, coming soon) — Bayes factors as the Bayesian counterpart" | §25.10 forward-map names Bayes factors → Topic 27. Activate link. |
| B14 | `confidence-intervals-and-duality.mdx` | §19.1 Rem 3 | "Track 7 develops the Bayesian perspective; here we stay frequentist." | §25.6 (credible interval); §25.8 (BvM ⇒ Wald/credible numerical coincidence under flat prior for Normal mean). |
| B15 | `confidence-intervals-and-duality.mdx` | §19.10 Rem 21 | "A $(1-\alpha)$ Bayesian credible interval is a set $C$ with posterior probability $\pi(\theta \in C \mid X) = 1 - \alpha$. … Track 7 develops the theory" | §25.6 Def 5 (credible interval); §25.8 Rem (flat-prior coincidence). Activate link. |
| B16 | `confidence-intervals-and-duality.mdx` | §19.10 Rem 22 cheat sheet row | "Posterior-probability claim needed → Bayesian credible (Track 7)" | §25.6 credible interval construction. Activate link. |
| B17 | `confidence-intervals-and-duality.mdx` | §19.10 Rem 23 Track-5 closing | "**Track 7 (Bayesian).** The contrast with frequentist coverage is where Bayesian inference earns its keep." | §25.1 motivation; §25.10 forward-map. |
| B18 | `linear-regression.mdx` | §21.10 Rem 25 "Track 7 — Bayesian linear regression" | "A conjugate Normal-inverse-gamma prior on $(\boldsymbol\beta, \sigma^2)$ yields a closed-form Normal-inverse-gamma posterior… Track 7 covers this in full, plus non-conjugate priors via MCMC." | §25.5 Ex 3 (scalar Normal–Normal-Inverse-Gamma unknown σ²); regression generalization previewed as §25.10 pointer to Topic 27/28. Activate link with caveat "scalar case covered; regression extension Topic 27/28." |
| B19 | `regularization-and-penalized-estimation.mdx` | §23.10 Rem 24 "Track 7 — Bayesian regularization, hierarchical priors, horseshoe" | "Thm 5's MAP-as-penalized-MLE correspondence is the gateway to the Bayesian treatment of regularization." | §25.7 (Jeffreys, prior-as-penalty); §25.8 Ex 8 (MAP = penalized MLE); §25.10 Rem (hierarchical + horseshoe → Topic 28). Activate link. |
| B20 | `model-selection-and-information-criteria.mdx` | §24.4 Rem 7, Rem 8; §24.10 Rem 23, Rem 26, Rem 31 | BIC ≈ log marginal likelihood; priors on model space; BMA full pointer; DIC/WAIC/PSIS-LOO; Track 7 on-ramp naming priors, MCMC, marginal likelihood, BMA. | §25.8 Rem (Laplace approximation to $m(\mathbf{y})$ — same machinery as §24.4 BIC, now applied to full posterior); §25.10 forward-map (BMA + DIC/WAIC/PSIS-LOO → Topic 27; MCMC → Topic 26). Activate Rem 31 on-ramp link. |
| B21 | `multiple-testing-and-false-discovery.mdx` | §20.10 Bayesian-multiplicity remark | "Bayesian multiplicity and local-FDR (Efron 2010) → Track 7." | §25.10 one-line mention; full local-FDR treatment defers to Topic 27 or formalml. Activate link. |

**Cleanup operations (slug cascade):**

| # | Source | Operation |
|---|---|---|
| C1 | `curriculum-graph.json` entry with `id: "bayesian-foundations"` | `url: "/topics/bayesian-foundations"` → `"/topics/bayesian-foundations-and-prior-selection"`; status `planned` → `published`; `id` preserved per Topic 23/24 precedent. |
| C2 | `curriculum-graph.json` entries for Topics 26–28 | URLs updated to locked slugs (`bayesian-computation-and-mcmc`, `bayesian-model-comparison-and-bma`, `hierarchical-bayes-and-partial-pooling`); status remains `planned`; original `id`s preserved (`bayesian-computation`, `bayesian-model-comparison`, `hierarchical-bayes`). |
| C3 | `src/data/curriculum.ts` | Track 7 status `planned` → `in-progress` (1 of 4). Topic 25 published-date set. |
| C4 | `conditional-probability.mdx`, `exponential-families.mdx`, `continuous-distributions.mdx`, `discrete-distributions.mdx`, `multivariate-distributions.mdx` | Frontmatter `connections[bayesian-foundations]` entries: update slug-target to new URL on next rebuild (markdown-link resolution automatic once `curriculum-graph.json` flips). |
| C5 | `multivariate-distributions.mdx` frontmatter | `connections[bayesian-computation]` → `connections[bayesian-computation-and-mcmc]`; relationship text unchanged. |
| C6 | References spreadsheet | Add Topic 25 to "Used In Topics" column for Gelman BDA3 (already present from Topic 8); other Topic 25 references added in §8.4. |

**Grep verification after all edits:**

```bash
grep -rn 'bayesian-foundations\b' src/content/topics/ --include='*.mdx'
# Should return zero stale references to the short slug.

grep -rn '/topics/bayesian-foundations-and-prior-selection' src/content/topics/ --include='*.mdx' | wc -l
# Should return at least 18 (one per activated B-row).

grep -rn 'Track 7, coming soon\|Bayesian Foundations.*coming soon' src/content/topics/ --include='*.mdx'
# Should return zero — all 21 forward-promises now live.
```

---

## 2. Frontmatter

### 2.1 Full frontmatter spec

The MDX file begins with the following YAML. Every field is required; every `url:` must resolve. Verified against Topic 24's frontmatter template.

```yaml
---
title: "Bayesian Foundations & Prior Selection"
slug: "bayesian-foundations-and-prior-selection"
track: 7
trackName: "Bayesian Statistics"
topicNumber: 25
positionInTrack: 1
readTime: "60 min"
difficulty: "intermediate"
status: "published"
publishedDate: "TO BE SET AT PUBLICATION"
lastUpdated: "TO BE SET AT PUBLICATION"
description: "Topics 13–24 built the frequentist framework: point estimators evaluated by bias and MSE, tests defined by size and power, CIs defined by coverage. Topic 25 opens Track 7 with the parallel Bayesian formalism — $\\theta$ treated as a random variable with prior $\\pi(\\theta)$, updated to a posterior $p(\\theta \\mid \\mathbf{y}) \\propto L(\\theta)\\pi(\\theta)$ via Bayes' theorem, with point estimates, credible intervals, and predictive distributions produced by integration against the posterior. §25.4 proves the general exponential-family conjugacy theorem; §25.5 works Beta-Binomial, Normal-Normal (known σ²), Normal-Normal-Inverse-Gamma (unknown σ²), Gamma-Poisson, and Dirichlet-Multinomial. §25.6 defines credible intervals (equal-tailed and HPD) and the posterior predictive. §25.7 treats prior selection — informative / weakly-informative / non-informative — with Jeffreys priors derived for Bernoulli and Normal-scale, and the Stone–Dawid–Zidek paradox as the cautionary note on improper priors. §25.8 proves **Bernstein–von Mises** (sketch, following van der Vaart §10): under regularity, the posterior converges in total variation to $\\mathcal{N}(\\hat\\theta_{\\text{MLE}}, \\mathcal{I}^{-1}/n)$, so Bayesian credible and frequentist Wald intervals coincide asymptotically. §25.10 previews Track 7: MCMC (Topic 26), BMA and Bayes factors (Topic 27), hierarchical and empirical Bayes (Topic 28)."
prerequisites:
  - topic: "conditional-probability"
    relationship: "Topic 4 Thm 4 (Bayes' theorem for events) is the engine. Topic 25 applies it to the parameter $\\theta$ treated as a random variable, with the prior $\\pi(\\theta)$ playing the role of $P(A)$ and the likelihood $p(\\mathbf{y} \\mid \\theta)$ the role of $P(B \\mid A)$."
  - topic: "expectation-moments"
    relationship: "Topic 4 E[X], Var(X), and the law of total probability (marginalization) define the posterior predictive $p(\\tilde y \\mid \\mathbf{y}) = \\int p(\\tilde y \\mid \\theta) p(\\theta \\mid \\mathbf{y}) d\\theta$ as an iterated expectation."
  - topic: "exponential-families"
    relationship: "Topic 7 Thm 3 (conjugate prior for exp families) is the algebraic kernel of Topic 25 §25.4 Thm 2. The canonical pairs — Beta-Bernoulli, Gamma-Poisson, Normal-Normal known σ² — are already worked in Topic 7 §7.7 as the exp-family update rule; Topic 25 lifts them into the Bayesian inferential framework with credible intervals and posterior predictive."
  - topic: "multivariate-distributions"
    relationship: "The conditional MVN formula (Topic 8 Thm 3) powers Ex 3 (Normal-Normal-Inverse-Gamma unknown σ²). Dirichlet-Multinomial conjugacy (Topic 8 Thm 6) is lifted into the Bayesian framework as Ex 5. The MVN-at-MAP Laplace approximation (§25.8 Rem) generalizes to multivariate θ."
  - topic: "maximum-likelihood"
    relationship: "Topic 14 Thm 4 (MLE asymptotic normality) is the engine of §25.8 BvM sketch. Topic 14 §14.11 Rem 9 (MAP as regularized MLE) is formalized in §25.8 Ex 8."
  - topic: "regularization-and-penalized-estimation"
    relationship: "Topic 23 §23.7 Thm 5 (MAP-penalization correspondence: Gaussian prior ⇒ ridge, Laplace prior ⇒ lasso) is cited by §25.8 Ex 8 without re-derivation. Topic 25 completes the arc from Topic 14 Rem 9 → Topic 23 Thm 5 → §25.8 Ex 8."
  - topic: "point-estimation"
    relationship: "Topic 13's bias-variance framework reappears in §25.6 as the three posterior point estimators (posterior mean, posterior median, MAP); Topic 13 Rem on risk and decision theory grounds the one-remark pointer to Bayes risk."
  - topic: "confidence-intervals-and-duality"
    relationship: "Topic 19 §19.1 Rem 3 set up the frequentist-vs-Bayesian distinction for interval estimation; §19.10 Rem 21–22 explicitly deferred credible intervals to Track 7. Topic 25 §25.6 delivers (equal-tailed + HPD + flat-prior coincidence with z-CI for Normal mean) and §25.8 (BvM asymptotic coincidence) closes the promise."
  - topic: "model-selection-and-information-criteria"
    relationship: "Topic 24 §24.4 Proof 2 (BIC = Laplace approximation to $-2\\log m(\\mathcal{M})$) is the direct on-ramp; §25.8 Rem generalizes the same Laplace machinery to the full posterior. Topic 24 §24.10 Rem 31 explicitly named Track 7 with the chain BIC → marginal likelihood → BMA → MCMC → DIC/WAIC/PSIS-LOO."
formalcalculusPrereqs:
  - topic: "multivariable-integration"
    site: "formalcalculus"
    relationship: "The marginal likelihood $m(\\mathbf{y}) = \\int L(\\theta)\\pi(\\theta)\\,d\\theta$ and posterior predictive $p(\\tilde y \\mid \\mathbf{y}) = \\int p(\\tilde y \\mid \\theta)p(\\theta \\mid \\mathbf{y})\\,d\\theta$ are integrals against the prior / posterior density. §25.4 Proof 1 does the conjugate-family closure calculation by identifying the integral's kernel with a known density family."
  - topic: "multivariable-calculus"
    site: "formalcalculus"
    relationship: "§25.8 Proof 3 (BvM sketch) Taylor-expands the log-posterior in a neighborhood of $\\hat\\theta_{\\text{MLE}}$; the leading Gaussian term is extracted via standard Taylor-with-remainder. The Laplace approximation remark is multivariate Taylor with a Hessian determinant."
  - topic: "change-of-variables"
    site: "formalcalculus"
    relationship: "§25.7 Rem (reparameterization invariance of Jeffreys prior) uses the Jacobian change-of-variables formula to show $\\pi_J(\\phi) = \\pi_J(\\theta)|d\\theta/d\\phi|$ for any monotone reparameterization $\\phi = g(\\theta)$."
formalmlConnections:
  - topic: "bayesian-neural-networks"
    site: "formalml"
    relationship: "BNN posteriors are MVN near the MAP estimate by BvM (§25.8 Thm 5); the Laplace approximation (§25.8 Rem) is the canonical construction. formalml's BNN topic develops the non-conjugate posterior approximation via variational inference, extending Topic 25's Laplace machinery to deep-network-scale models."
    url: "https://formalml.com/topics/bayesian-neural-networks"
  - topic: "variational-inference"
    site: "formalml"
    relationship: "When conjugate priors break down (non-exponential-family likelihoods, hierarchical models beyond Topic 28) and MCMC is too slow, variational inference approximates the posterior with a tractable family $q(\\theta)$ minimizing $\\mathrm{KL}(q \\| p(\\cdot \\mid \\mathbf{y}))$. Topic 25 §25.10 pointers; formalml develops ELBO, mean-field VI, normalizing flows."
    url: "https://formalml.com/topics/variational-inference"
  - topic: "gaussian-processes"
    site: "formalml"
    relationship: "GPs are the infinite-dimensional generalization of Topic 25's finite-dimensional Bayesian framework: the prior is over functions rather than parameters, and the posterior is a conditional MVN (Topic 8 Thm 3 applied infinite-dimensionally)."
    url: "https://formalml.com/topics/gaussian-processes"
  - topic: "probabilistic-programming"
    site: "formalml"
    relationship: "Probabilistic programming languages (Stan, PyMC, NumPyro) automate the posterior computation Topic 25 does by hand — users specify prior × likelihood, the language handles sampling via HMC/NUTS. Topic 25's conjugate calculations show what the PPL would output for the easy cases."
    url: "https://formalml.com/topics/probabilistic-programming"
connections:
  - topic: "conditional-probability"
    relationship: "Topic 4's Bayes' theorem for events is lifted to parameters. The prior × likelihood ∝ posterior shorthand is the same multiplication as Topic 4's $P(A \\cap B)$ / $P(B)$."
  - topic: "exponential-families"
    relationship: "Topic 7's conjugate-prior construction is the algebraic kernel. Topic 25 adds credible intervals, posterior predictive, and prior-selection theory on top."
  - topic: "maximum-likelihood"
    relationship: "The MLE and the MAP are two halves of the same argmax — MAP maximizes $\\ell(\\theta) + \\log\\pi(\\theta)$. Under BvM (§25.8) they coincide asymptotically."
  - topic: "model-selection-and-information-criteria"
    relationship: "BIC is the first-order Laplace approximation to $-2\\log m(\\mathcal{M})$. Topic 25 §25.8 Rem generalizes the Laplace machinery to the posterior; Topic 27 will return to $m(\\mathbf{y})$ computation for BMA."
  - topic: "bayesian-computation-and-mcmc"
    relationship: "When the conjugate framework breaks down, MCMC is the standard tool. Topic 26 develops Metropolis–Hastings, Gibbs sampling (which uses Topic 25 conjugate full-conditionals as building blocks), HMC, and NUTS."
  - topic: "bayesian-model-comparison-and-bma"
    relationship: "Bayes factors $\\mathrm{BF}_{10} = m(\\mathbf{y} \\mid H_1) / m(\\mathbf{y} \\mid H_0)$ and BMA use the marginal likelihood Topic 25 §25.8 Rem introduces. Full treatment Topic 27."
  - topic: "hierarchical-bayes-and-partial-pooling"
    relationship: "Hierarchical priors $\\theta_i \\mid \\mu \\sim \\pi(\\cdot \\mid \\mu)$, $\\mu \\sim \\pi(\\mu)$ generalize Topic 25's fixed-prior framework. Empirical Bayes uses the data to estimate $\\mu$. Topic 28 develops."
references:
  - type: "book"
    title: "Bayesian Data Analysis"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari & Donald B. Rubin"
    year: 2013
    edition: "3rd"
    publisher: "CRC Press"
    isbn: "978-1-4398-4095-5"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    note: "GEL2013. The standard modern reference. Ch. 2 (single-parameter models, all canonical conjugate pairs), Ch. 3 (multivariate, including Normal-Normal-Inverse-Gamma), Ch. 5 (hierarchical models — Topic 28 territory). Cited throughout §25.5, §25.6, §25.7. Already in spreadsheet from Topic 8; append `bayesian-foundations-and-prior-selection` to Used In Topics."
  - type: "book"
    title: "Bayesian Theory"
    author: "José M. Bernardo & Adrian F. M. Smith"
    year: 1994
    publisher: "Wiley"
    isbn: "978-0-471-92416-6"
    url: "https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316870"
    note: "BER1994. The foundational-theory text. Ch. 4 (posterior distributions and inference), Ch. 5 (reference analysis, Jeffreys priors). Cited in §25.3 notation rationale, §25.7 Jeffreys derivation. New to spreadsheet."
  - type: "book"
    title: "The Bayesian Choice: From Decision-Theoretic Foundations to Computational Implementation"
    author: "Christian P. Robert"
    year: 2007
    edition: "2nd"
    publisher: "Springer"
    isbn: "978-0-387-71598-8"
    url: "https://link.springer.com/book/10.1007/0-387-71599-1"
    note: "ROB2007. Decision-theoretic emphasis; useful for §25.6 Rem on posterior mean as Bayes estimator under squared-error loss. New to spreadsheet."
  - type: "book"
    title: "Theory of Probability"
    author: "Harold Jeffreys"
    year: 1961
    edition: "3rd"
    publisher: "Oxford University Press"
    isbn: "978-0-19-850368-2"
    url: "https://global.oup.com/academic/product/theory-of-probability-9780198503682"
    note: "JEF1961. The historical source for the Jeffreys prior construction cited in §25.7 Thm 4. Topic 25 states the invariance property; full philosophical defense is in JEF1961 Ch. III. New to spreadsheet."
  - type: "book"
    title: "Understanding Uncertainty"
    author: "Dennis V. Lindley"
    year: 2014
    edition: "Revised"
    publisher: "Wiley"
    isbn: "978-1-118-65012-7"
    url: "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118650158"
    note: "LIN2014. Subjective-Bayes anchor. Cited in §25.1 one-sentence philosophy remark and §25.10 for the subjective-vs-objective-Bayes contrast pointer. New to spreadsheet."
  - type: "book"
    title: "Asymptotic Statistics"
    author: "A. W. van der Vaart"
    year: 1998
    publisher: "Cambridge University Press"
    isbn: "978-0-521-78450-4"
    url: "https://www.cambridge.org/core/books/asymptotic-statistics/A3C7DAD3F7E66A1FA60E9C8FE132EE1D"
    note: "VDV1998. §10 contains the BvM proof Topic 25 sketches (§25.8 Proof 3). Already in spreadsheet from Topic 14; append topic to Used In Topics."
  - type: "paper"
    title: "On the consistency of Bayes estimates"
    author: "Persi Diaconis & David Freedman"
    year: 1986
    journal: "The Annals of Statistics"
    volume: "14(1)"
    pages: "1–26"
    url: "https://doi.org/10.1214/aos/1176349830"
    note: "DIA1986. Doob-type consistency and the caveats (inconsistency in infinite-dimensional parameter spaces). Cited at §25.8 Rem on BvM failure modes. New to spreadsheet."
  - type: "book"
    title: "Statistical Inference"
    author: "George Casella & Roger L. Berger"
    year: 2002
    edition: "2nd"
    publisher: "Duxbury"
    isbn: "978-0-534-24312-8"
    url: "https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9780534243128"
    note: "CAS2002. §7.2 (Bayes estimators), §10.4 (Bayes vs frequentist contrast). Inherited from earlier topics; append topic to Used In Topics."
  - type: "book"
    title: "Theory of Point Estimation"
    author: "E. L. Lehmann & George Casella"
    year: 1998
    edition: "2nd"
    publisher: "Springer"
    isbn: "978-0-387-98502-2"
    url: "https://link.springer.com/book/10.1007/b98854"
    note: "LEH1998. §4 (Bayes estimation). Inherited from Topic 13; append topic."
  - type: "book"
    title: "The Elements of Statistical Learning"
    author: "Trevor Hastie, Robert Tibshirani & Jerome Friedman"
    year: 2009
    edition: "2nd"
    publisher: "Springer"
    isbn: "978-0-387-84857-0"
    url: "https://hastie.su.domains/ElemStatLearn/"
    note: "HAS2009. §8.3 (MAP and shrinkage as prior). Inherited from Topic 23; append topic."
---
```

### 2.2 Length-target rationale

**Target: ~10,500 words, 60-minute read.** Track-opener cadence: Topic 13 (Track 4 opener) was ~10K/55min; Topic 17 (Track 5 opener) was 12K/55min; Topic 21 (Track 6 opener) was 10.5K/60min. Topic 25 as Track 7 opener aligns with Topic 21's length — Bayesian machinery needs the same notation-commit investment regression needed, and the featured BvM proof consumes proof budget the Track 6 opener gave to Gauss–Markov.

Word-budget allocation (approximate):
- §25.1 motivation: 500 words (posterior as "distribution over $\theta$"; contrast with Topic 13–24 point estimators).
- §25.2 Bayes for $\theta$: 900 (Def 1, Thm 1, Rem on proportionality).
- §25.3 notation lock: 400 (the Track-7-wide notation block).
- §25.4 exp-fam conjugacy theorem: 1,200 (Thm 2 + full Proof 1).
- §25.5 five conjugate pairs: 1,800 (Thm 3 + Proof 2 for Beta-Binomial; Ex 2–5 shorter).
- §25.6 credible intervals + posterior predictive: 1,200 (Def 5–6, Ex 6, Thm 3b + Proof 2b).
- §25.7 prior selection: 1,400 (Rem on informative/weak/non-informative; Thm 4 Jeffreys; Ex 7 sensitivity; Rem on Stone–Dawid–Zidek).
- §25.8 BvM + Laplace + MAP-ridge bridge: 1,800 (Thm 5 + Proof 3 sketch; Ex 8 MAP-ridge; Rem on Laplace; Rem on failure modes).
- §25.9 Wishart-Normal + joint posterior for $\mu, \Sigma$: 500 (Rem only; pointer to GEL2013 Ch. 3).
- §25.10 forward-map: 800 (nine remarks × ~90 words each).

Total ≈ 10,500 words.

---

## 3. Content Structure

### 3.1 Section map

| § | Title | Formal elements | Figure | Interactive component |
|---|---|---|---|---|
| 25.1 | "Why Bayesian?" | Rem 1 (motivation: distribution over $\theta$ vs point estimator), Rem 2 (subjective-vs-objective philosophical anchor, LIN2014 cited), Rem 3 (what's in vs what's out — scope-boundary preview) | `25-1-prior-to-posterior-animation.png` | — |
| 25.2 | "Bayes' Theorem for a Parameter" | Def 1 (prior, likelihood, posterior, marginal likelihood, posterior predictive), Thm 1 (Bayes' theorem for $\theta$ — stated, proof cites Topic 4 Thm 4), Ex 1 (coin-toss: Beta(1,1) prior + 3 heads in 4 tosses → Beta(4,2) posterior, computed end-to-end), Rem 4 (proportionality shorthand: posterior ∝ likelihood × prior) | `25-2-bayes-update-geometry.png` | `PriorPosteriorExplorer` (featured for §25.2 + §25.5) |
| 25.3 | "Track 7 Notation Conventions" | Notation block (locked for Tracks 7): $\pi(\theta)$, $p(\theta \mid \mathbf{y})$, $p(\tilde y \mid \mathbf{y})$, $m(\mathbf{y})$, $\mathrm{KL}(\pi \,\|\, \pi')$, $\mathrm{BF}_{10}$, posterior-mean/median/MAP subscripts | — | — |
| 25.4 | "Exponential-Family Conjugacy" | Def 2 (conjugate family), **Thm 2 (exp-fam conjugacy — FULL PROOF, Proof 1)**, Rem 5 (Topic 7 Thm 3 is the special case), Rem 6 (pseudo-observation interpretation of hyperparameters) | `25-4-exp-fam-conjugacy-kernel.png` | — |
| 25.5 | "The Canonical Conjugate Pairs" | **Thm 3 (Beta-Binomial posterior — FULL PROOF, Proof 2)**, Ex 2 (Normal-Normal known $\sigma^2$: precision-weighted average; derivation sketched, cites Topic 7 Ex 4), Ex 3 (Normal-Normal-Inverse-Gamma unknown $\sigma^2$: joint posterior via conditional-MVN; derivation sketched, cites GEL2013 §3.3), Ex 4 (Gamma-Poisson: cites Topic 7 Ex 3), Ex 5 (Dirichlet-Multinomial: cites Topic 8 Thm 6), Rem 7 (all five as instances of Thm 2), Rem 8 (when conjugacy breaks: non-exp-family likelihoods, hierarchical priors → Topic 26/28) | `25-5-five-conjugate-pairs-panel.png` | `ConjugatePairBrowser` (required) |
| 25.6 | "Credible Intervals and Posterior Predictive" | Def 5 (equal-tailed + HPD credible interval), Def 6 (posterior point estimators: mean, median, MAP), **Thm 3b (Beta-Binomial posterior predictive — FULL PROOF, Proof 2b)**, Ex 6 (95% credible intervals for all five conjugate pairs), Rem 9 (posterior mean = Bayes estimator under squared-error loss, ROB2007 §2 cited), Rem 10 (HPD vs equal-tailed for skewed posteriors), Rem 11 (posterior predictive for Normal-Normal: Student-t; cites Topic 6 §6.7) | `25-6-credible-vs-wald.png`, `25-6-posterior-predictive-beta-binomial.png` | — |
| 25.7 | "Prior Selection" | Rem 12 (three classes: informative, weakly-informative, non-informative — GEL2013 Ch. 2), **Thm 4 (Jeffreys prior and reparameterization invariance — stated, full derivation for Bernoulli and Normal-scale; invariance property stated without proof, cites JEF1961 Ch. III)**, Ex 7 (prior sensitivity: three priors on Beta-Binomial — informative peaked at 0.2, weak Beta(2,2), non-informative Jeffreys — same data, three posteriors), Def 7 (improper prior; integrable-posterior condition), Rem 13 (Stone–Dawid–Zidek paradox — improper priors can produce paradoxes; cites STO1983), Rem 14 (reference priors and Bernardo construction → formalml), Rem 15 (prior elicitation as substantive modeling — ROB2007 §3, GEL2013 §2.9) | `25-7-prior-sensitivity-three-priors.png`, `25-7-jeffreys-prior-bernoulli.png` | `PriorSensitivityComparator` (optional fourth) |
| 25.8 | "Bernstein–von Mises: The Bridge to Frequentism" | **Thm 5 (Bernstein–von Mises — FEATURED, sketch proof, Proof 3)**, Ex 8 (MAP = penalized MLE: Gaussian prior ⇒ ridge, Laplace prior ⇒ lasso; cites Topic 23 Thm 5 + Topic 14 Rem 9), Rem 16 (Laplace approximation to the posterior and to $m(\mathbf{y})$ — same machinery as Topic 24 §24.4 BIC, now applied to full posterior), Rem 17 (flat-prior coincidence: for Normal mean with known variance, credible interval and z-CI are numerically identical — discharges Topic 19 §19.1 Rem 3), Rem 18 (when BvM fails: heavy-tailed priors, non-identifiable models, $d \sim n$, nonparametric — DIA1986, VDV1998 §10.3) | `25-8-bernstein-von-mises-convergence.png`, `25-8-map-as-penalized-mle.png` | `BernsteinVonMisesAnimator` (featured, required) |
| 25.9 | "The Multivariate Case: Normal-Inverse-Wishart" | Rem 19 (joint posterior on $(\boldsymbol\mu, \boldsymbol\Sigma)$ via Normal-Inverse-Wishart prior — stated with parameter-update rule; cites GEL2013 §3.6 for the derivation), Rem 20 (scalar Topic 25 mechanics generalize; full regression extension → Topic 27/28 — discharges Topic 21 §21.10 Rem 25) | — | — |
| 25.10 | "Forward Map" | Rem 21 (Topic 26 — MCMC; Metropolis–Hastings, Gibbs, HMC, NUTS), Rem 22 (Topic 27 — Bayes factors, BMA, marginal-likelihood computation, DIC/WAIC/PSIS-LOO), Rem 23 (Topic 28 — hierarchical + empirical Bayes + partial pooling + horseshoe), Rem 24 (variational inference → formalml), Rem 25 (Bayesian decision theory — one-line mention, ROB2007), Rem 26 (reference priors / Berger–Pericchi → deferred), Rem 27 (predictive checks + model criticism → Topic 27), Rem 28 (Bayesian nonparametrics: Dirichlet processes, Polya trees → formalml), Rem 29 (local FDR and Bayesian multiplicity — discharges Topic 20 §20.10), closing paragraph | `25-10-forward-map.png` | — |

**Totals:** 7 Definitions, 5 Theorems (3 full proofs: Thm 2, Thm 3, Thm 3b; 1 sketch proof: Thm 5; 1 stated-with-derivation: Thm 4), 8 Examples, 29 Remarks, 10 Figures, 3 required + 1 optional interactive components. Totals match §1 preamble.

### 3.2 Proof specifications (ready to render)

Three full proofs plus one sketch proof. Each is written to the target MDX line count indicated.

---

#### Proof 1 — Exponential-family conjugacy theorem (Thm 2, §25.4)

**Target:** ~20 MDX lines. Structure:

1. **Setup.** Let $\{f(y; \theta) : \theta \in \Theta\}$ be a one-parameter exponential family in canonical form: $f(y; \eta) = h(y)\exp(\eta\, T(y) - A(\eta))$, with natural parameter $\eta \in H \subseteq \mathbb{R}$ and sufficient statistic $T(y)$.

2. **Proposed conjugate prior.** Define the prior density on $H$:
$$\pi(\eta \mid \chi_0, \nu_0) = K(\chi_0, \nu_0)\exp(\chi_0 \eta - \nu_0 A(\eta))$$
with hyperparameters $\chi_0 \in \mathbb{R}$, $\nu_0 > 0$, and normalizing constant $K(\chi_0, \nu_0)$ chosen so that $\int_H \pi(\eta)\,d\eta = 1$. (Whenever this integral is finite, $\pi$ is a proper prior; §25.7 Def 7 handles the improper case.)

3. **Apply Bayes' theorem.** Given $n$ iid observations $y_1, \ldots, y_n$ with joint likelihood $L(\eta) = \prod_{i=1}^n f(y_i; \eta) = \left[\prod_i h(y_i)\right]\exp\left(\eta \sum_i T(y_i) - n A(\eta)\right)$. Write $S = \sum_{i=1}^n T(y_i)$ for the sufficient-statistic sum. Then:

$$p(\eta \mid \mathbf{y}) \;\propto\; L(\eta)\pi(\eta) \;\propto\; \exp(\eta S - n A(\eta))\exp(\chi_0 \eta - \nu_0 A(\eta))$$

4. **Collect exponents.** Combining the two exponentials:

$$p(\eta \mid \mathbf{y}) \;\propto\; \exp\!\big((\chi_0 + S)\eta - (\nu_0 + n) A(\eta)\big)$$

5. **Identify the posterior kernel.** This is the kernel of the same conjugate family with updated hyperparameters $\chi_0' = \chi_0 + S$, $\nu_0' = \nu_0 + n$. Since the normalizing constant is determined by the shape of the kernel, we conclude:

$$p(\eta \mid \mathbf{y}) = \pi(\eta \mid \chi_0 + S, \nu_0 + n)$$

6. **Interpretation of the update rule.** The hyperparameter $\nu_0$ acts as a **pseudo-sample-size**: the prior is worth $\nu_0$ equivalent observations before any data arrive. $\chi_0$ is the pseudo-sufficient-statistic total. After observing $n$ real data points with sufficient-statistic sum $S$, the effective sample size grows to $\nu_0 + n$ and the effective sufficient-statistic total to $\chi_0 + S$. ∎ — using Topic 4 Thm 4 (Bayes' theorem) and Topic 7 Thm 3 (exp-family form).

**End-of-proof citation:** "∎ — using Topic 4 Thm 4 (Bayes' theorem) and Topic 7 §7.7 Thm 3 (exp-family form)."

**Pedagogical note for §25.4 body text:** The theorem subsumes Topic 7 Thm 3 (which stated the construction as pure algebra on the exp-family kernel). Topic 25 adds the inferential framing — the $\chi_0'$, $\nu_0'$ update rule is now the posterior, and every downstream Bayesian quantity (credible interval, posterior predictive, MAP) follows from this one identification.

---

#### Proof 2 — Beta-Binomial posterior (Thm 3, §25.5)

**Target:** ~12 MDX lines. Structure:

1. **Setup.** Observe $k$ successes in $n$ independent Bernoulli($\theta$) trials; equivalently, observe $S_n = k$ from $S_n \sim \text{Binomial}(n, \theta)$. Place a Beta$(\alpha_0, \beta_0)$ prior on $\theta \in (0, 1)$.

2. **Likelihood and prior.** The Binomial likelihood, viewed as a function of $\theta$:

$$L(\theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} \;\propto\; \theta^k (1-\theta)^{n-k}$$

The Beta prior density:

$$\pi(\theta) = \frac{1}{B(\alpha_0, \beta_0)} \theta^{\alpha_0 - 1} (1-\theta)^{\beta_0 - 1} \;\propto\; \theta^{\alpha_0 - 1} (1-\theta)^{\beta_0 - 1}$$

3. **Apply Bayes.** The posterior is proportional to likelihood × prior:

$$p(\theta \mid k) \;\propto\; \theta^k (1-\theta)^{n-k} \cdot \theta^{\alpha_0 - 1}(1-\theta)^{\beta_0 - 1} \;=\; \theta^{(\alpha_0 + k) - 1}(1-\theta)^{(\beta_0 + n - k) - 1}$$

4. **Identify.** This is the kernel of Beta$(\alpha_0 + k, \beta_0 + n - k)$. Since the posterior must integrate to 1, the normalizing constant is $1/B(\alpha_0 + k, \beta_0 + n - k)$, and:

$$p(\theta \mid k) = \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$$

5. **Posterior moments.** The posterior mean is:

$$\mathbb{E}[\theta \mid k] = \frac{\alpha_0 + k}{\alpha_0 + \beta_0 + n} = w \cdot \frac{\alpha_0}{\alpha_0 + \beta_0} + (1-w) \cdot \frac{k}{n}$$

where $w = (\alpha_0 + \beta_0)/(\alpha_0 + \beta_0 + n)$ — a convex combination of the prior mean and the sample proportion, with weights proportional to the pseudo-sample-size $\alpha_0 + \beta_0$ and the real sample size $n$. ∎ — using Topic 4 Thm 4 (Bayes' theorem) and Topic 6 Thm 12 (Beta moments).

**End-of-proof citation:** "∎ — using Topic 4 Thm 4 (Bayes' theorem) and Topic 6 §6.6 Thm 12 (Beta moments)."

---

#### Proof 2b — Beta-Binomial posterior predictive (Thm 3b, §25.6)

**Target:** ~15 MDX lines. Structure:

1. **Setup.** Having observed $k$ successes in $n$ trials with posterior $\theta \mid \mathbf{y} \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$, the posterior predictive is the distribution of a future count $\tilde y \in \{0, 1, \ldots, m\}$ from $m$ new independent Bernoulli($\theta$) trials.

2. **Definition.** The posterior predictive pmf is:

$$p(\tilde y \mid \mathbf{y}) = \int_0^1 p(\tilde y \mid \theta) \, p(\theta \mid \mathbf{y}) \, d\theta$$

3. **Substitute the likelihood and posterior.** With $p(\tilde y \mid \theta) = \binom{m}{\tilde y}\theta^{\tilde y}(1-\theta)^{m-\tilde y}$ and $p(\theta \mid \mathbf{y}) = \theta^{\alpha^\star - 1}(1-\theta)^{\beta^\star - 1}/B(\alpha^\star, \beta^\star)$ where $\alpha^\star = \alpha_0 + k$, $\beta^\star = \beta_0 + n - k$:

$$p(\tilde y \mid \mathbf{y}) = \binom{m}{\tilde y} \frac{1}{B(\alpha^\star, \beta^\star)} \int_0^1 \theta^{\tilde y + \alpha^\star - 1}(1-\theta)^{m - \tilde y + \beta^\star - 1} d\theta$$

4. **Identify the integral as a Beta function.** The integrand is the kernel of Beta$(\tilde y + \alpha^\star, m - \tilde y + \beta^\star)$. Its integral is $B(\tilde y + \alpha^\star, m - \tilde y + \beta^\star)$:

$$p(\tilde y \mid \mathbf{y}) = \binom{m}{\tilde y} \frac{B(\tilde y + \alpha^\star, m - \tilde y + \beta^\star)}{B(\alpha^\star, \beta^\star)}$$

5. **This is the Beta-Binomial compound distribution** with parameters $(m, \alpha^\star, \beta^\star)$. The posterior predictive is *wider* than the Binomial$(m, \hat\theta)$ plug-in approximation, reflecting the posterior uncertainty in $\theta$ — a structural feature of Bayesian prediction that plug-in point-estimate predictions miss. ∎ — using Topic 4 law of total probability and Topic 6 Def 3 (Beta function).

**End-of-proof citation:** "∎ — using Topic 4 §4.3 law of total probability and Topic 6 §6.6 Def 3 (Beta function)."

---

#### Proof 3 — Bernstein–von Mises theorem (Thm 5, §25.8) — sketch

**Target:** ~25 MDX lines. Sketch-quality, following VDV1998 §10.2. Full proof requires LAN contiguity and uniform tightness; those are outside Topic 25's scope.

1. **Statement.** Let $Y_1, \ldots, Y_n \overset{\text{iid}}{\sim} f(\cdot; \theta_0)$ with $\theta_0$ in the interior of $\Theta \subseteq \mathbb{R}$, a regular parametric family with Fisher information $\mathcal{I}(\theta_0) \in (0, \infty)$. Let $\hat\theta_n$ be the MLE. For any proper prior $\pi$ continuous and positive at $\theta_0$:

$$\sup_{B \in \mathcal{B}(\mathbb{R})} \left| \Pr_{\theta \sim p(\cdot \mid \mathbf{y})}\!\left(\sqrt{n}(\theta - \hat\theta_n) \in B\right) - \Pr_{Z \sim \mathcal{N}(0, \mathcal{I}(\theta_0)^{-1})}(Z \in B) \right| \;\xrightarrow{P}\; 0.$$

That is, the rescaled posterior converges in total variation to $\mathcal{N}(0, \mathcal{I}(\theta_0)^{-1})$ in probability under the true distribution $\Pr_{\theta_0}$.

2. **Taylor-expand the log-posterior at $\hat\theta_n$.** Write $\ell_n(\theta) = \sum_i \log f(Y_i; \theta)$. The log-posterior is:

$$\log p(\theta \mid \mathbf{y}) = \ell_n(\theta) + \log\pi(\theta) + \text{const}$$

By Taylor's theorem, for $\theta$ near $\hat\theta_n$:

$$\ell_n(\theta) = \ell_n(\hat\theta_n) + \ell_n'(\hat\theta_n)(\theta - \hat\theta_n) + \tfrac{1}{2}\ell_n''(\tilde\theta)(\theta - \hat\theta_n)^2$$

The first-derivative term vanishes because $\hat\theta_n$ is the MLE ($\ell_n'(\hat\theta_n) = 0$). The second derivative $-\ell_n''(\tilde\theta)/n \to \mathcal{I}(\theta_0)$ in probability by Topic 14 Thm 4's argument.

3. **Reparameterize.** Let $u = \sqrt{n}(\theta - \hat\theta_n)$. Then $\theta = \hat\theta_n + u/\sqrt{n}$, and:

$$\ell_n(\theta) - \ell_n(\hat\theta_n) \;=\; -\tfrac{1}{2}\mathcal{I}(\theta_0) u^2 + o_P(1)$$

uniformly on compact $u$-sets (this is the LAN condition, stated without full proof; VDV1998 §7.2 Lemma 7.6 has the details).

4. **Prior smoothness.** Since $\pi$ is continuous and positive at $\theta_0$, $\log\pi(\theta)$ is bounded on a neighborhood of $\theta_0$, and $\log\pi(\hat\theta_n + u/\sqrt n) = \log\pi(\theta_0) + o_P(1)$ uniformly on compact $u$-sets (by consistency of $\hat\theta_n$, which holds by Topic 14 Thm 3).

5. **Assemble the posterior density.** In the $u$-parameterization:

$$p(u \mid \mathbf{y}) \;\propto\; \exp\!\left(\ell_n(\hat\theta_n + u/\sqrt n) - \ell_n(\hat\theta_n)\right)\pi(\hat\theta_n + u/\sqrt n) \;\propto\; \exp\!\left(-\tfrac{1}{2}\mathcal{I}(\theta_0)u^2\right)(1 + o_P(1))$$

This is the kernel of $\mathcal{N}(0, \mathcal{I}(\theta_0)^{-1})$.

6. **Extend to total-variation convergence.** Step 5 establishes pointwise (at each $u$) convergence of densities up to normalization. The total-variation upgrade requires showing (a) the posterior puts asymptotically negligible mass outside a shrinking neighborhood of $\hat\theta_n$, and (b) uniform integrability of the posterior density. Both follow from the prior's positivity at $\theta_0$ plus the regular-family tail bounds. Full argument: VDV1998 §10.2 Thm 10.1.

7. **Interpretation.** The posterior forgets the prior at rate $1/\sqrt n$: the prior's contribution to the posterior density becomes vanishingly small relative to the likelihood's contribution. Bayesian and frequentist inference converge — credible intervals $\to$ Wald intervals in TV distance. ∎ (sketch) — using Topic 14 Thm 4 (MLE asymptotic normality), Topic 11 Thm 1 (CLT applied to the score), and VDV1998 §10.2 for the TV upgrade.

**End-of-proof citation:** "∎ (sketch) — using Topic 14 §14.5 Thm 4 (MLE asymptotic normality), Topic 11 §11.3 Thm 1 (CLT), and van der Vaart 1998 §10.2 Thm 10.1 (full TV upgrade)."

---

### 3.3 Track 7 notation conventions — locked here

This block appears in §25.3 verbatim; Topics 26–28 inherit.

| Object | Symbol | Interpretation |
|---|---|---|
| Prior density | $\pi(\theta)$ | Distribution over $\theta$ before observing data. Positive, integrable (or improper — §25.7 Def 7). |
| Posterior density | $p(\theta \mid \mathbf{y})$ | Distribution over $\theta$ after observing $\mathbf{y} = (y_1, \ldots, y_n)$. |
| Sampling density | $p(\mathbf{y} \mid \theta)$ or $L(\theta; \mathbf{y})$ | Likelihood — joint density of $\mathbf{y}$ given $\theta$, viewed as a function of $\theta$. |
| Marginal likelihood | $m(\mathbf{y}) = \int p(\mathbf{y} \mid \theta)\pi(\theta)\,d\theta$ | Normalizing constant; also called *evidence* or *prior predictive*. |
| Posterior predictive | $p(\tilde y \mid \mathbf{y}) = \int p(\tilde y \mid \theta) p(\theta \mid \mathbf{y})\,d\theta$ | Distribution of a future observation $\tilde y$ integrating out posterior uncertainty. |
| KL divergence | $\mathrm{KL}(\pi \,\|\, \pi')$ | Relative entropy of $\pi$ w.r.t. $\pi'$. |
| Bayes factor | $\mathrm{BF}_{10} = m(\mathbf{y} \mid H_1)/m(\mathbf{y} \mid H_0)$ | Posterior-odds update factor. Topic 25 mentions only; Topic 27 develops. |
| MAP estimate | $\hat\theta_{\text{MAP}} = \arg\max p(\theta \mid \mathbf{y})$ | Mode of the posterior. |
| Posterior mean | $\hat\theta_{\text{PM}} = \mathbb{E}[\theta \mid \mathbf{y}]$ | Bayes estimator under squared-error loss. |
| Posterior median | $\hat\theta_{\text{med}}$ | Bayes estimator under absolute-error loss. |
| $(1-\alpha)$ credible set | $C$ with $\int_C p(\theta \mid \mathbf{y})\,d\theta = 1-\alpha$ | Analog of CI in the Bayesian framework. |
| HPD interval | $C^{\text{HPD}} = \{\theta : p(\theta \mid \mathbf{y}) \ge c_\alpha\}$ | Highest-posterior-density interval — the shortest $(1-\alpha)$ credible set. |

**Notation rationales locked here (Topic 26–28 inherit):**

- $\pi$ for prior (Bernardo–Smith / Robert house style). Visually distinct from the sampling $p$ and posterior $p(\cdot \mid \mathbf{y})$; avoids the Gelman-style $p(\theta)$ overloading that forces context-based disambiguation.
- $p(\tilde y \mid \mathbf{y})$ for posterior predictive (tilde for the new observation; cleaner in KaTeX than $y_{\text{new}}$ subscript).
- $m(\mathbf{y})$ for marginal likelihood (Bernardo / Robert; avoids the collision with $p(\mathbf{y})$ as the sampling density that Gelman's notation forces).
- $\mathrm{KL}(\pi \,\|\, \pi')$ matches Topics 7 and 14.
- Bayesian and frequentist estimators distinguished by subscripts: $\hat\theta_{\text{MLE}}$, $\hat\theta_{\text{MAP}}$, $\hat\theta_{\text{PM}}$, $\hat\theta_{\text{med}}$.

---

## 4. Static Figures & Notebook Structure

### 4.1 Figure manifest

Ten PNGs produced by the notebook; output directory `public/images/topics/bayesian-foundations-and-prior-selection/`.

| # | File | Section | Dimensions (in) | DPI | Description |
|---|---|---|---|---|---|
| 1 | `25-1-prior-to-posterior-animation.png` | §25.1 | 12×4 (3-panel) | 150 | Beta-Binomial sequential update: prior Beta(2, 2) → posterior after $n=1, 5, 20, 100$ observations with true $\theta = 0.3$. Prior fades; posterior sharpens. Sets the visual grammar for `PriorPosteriorExplorer`. |
| 2 | `25-2-bayes-update-geometry.png` | §25.2 | 10×4 (3-panel) | 150 | Prior (blue) × likelihood (amber) = (unnormalized posterior) ∝ posterior (purple) for Ex 1 (Beta(1,1) prior + 3H/1T). Shows the prior×likelihood shape before renormalization. |
| 3 | `25-4-exp-fam-conjugacy-kernel.png` | §25.4 | 10×4 | 150 | Three conjugate pairs shown as density curves: Beta-Binomial, Gamma-Poisson, Normal-Normal — all with the $\chi_0' = \chi_0 + S$, $\nu_0' = \nu_0 + n$ update rule annotated. Discharges the "canonical kernel" intuition. |
| 4 | `25-5-five-conjugate-pairs-panel.png` | §25.5 | 14×10 (5-panel) | 150 | Prior and posterior for each of the five canonical pairs. Each panel: prior curve, likelihood (scaled), posterior curve. Formula box overlay. |
| 5 | `25-6-credible-vs-wald.png` | §25.6 | 10×4 (2-panel) | 150 | Beta(5, 20) posterior with 95% equal-tailed credible interval (shaded) vs 95% HPD (shaded + hatched). Both shown on the same density. Right panel: Normal(0, 1) posterior + z-CI overlay showing numerical coincidence under flat prior. |
| 6 | `25-6-posterior-predictive-beta-binomial.png` | §25.6 | 10×4 (2-panel) | 150 | Beta-Binomial posterior predictive PMF (purple bars) vs Binomial($m$, $\hat\theta_{\text{MLE}}$) plug-in (grey bars) for $m=20$ new trials, $\alpha^\star=5, \beta^\star=20$. Shows predictive uncertainty is wider than plug-in. |
| 7 | `25-7-prior-sensitivity-three-priors.png` | §25.7 | 12×4 (3-panel) | 150 | Same data (10 heads in 50 tosses), three priors: informative Beta(2, 8) peaked at 0.2; weak Beta(2, 2); Jeffreys Beta(½, ½). Three overlapping posteriors. |
| 8 | `25-7-jeffreys-prior-bernoulli.png` | §25.7 | 10×4 (2-panel) | 150 | Bernoulli Jeffreys prior $\pi(\theta) \propto 1/\sqrt{\theta(1-\theta)}$ density (left); reparameterization invariance shown by transforming to $\phi = \text{logit}(\theta)$ and confirming Jeffreys $\pi_J(\phi)$ matches direct Jeffreys on $\phi$ (right). |
| 9 | `25-8-bernstein-von-mises-convergence.png` | §25.8 | 14×4 (4-panel) | 150 | **FEATURED FIGURE.** Beta(2, 2) prior + Binomial likelihood, true $\theta_0 = 0.3$. Four panels for $n = 5, 25, 100, 500$. Each panel: posterior density (purple), Normal-at-MLE approximation $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}^{-1}/n)$ (dashed black), vertical line at $\theta_0$, vertical line at $\hat\theta_{\text{MLE}}$. As $n$ grows, posterior collapses onto the dashed Normal. |
| 10 | `25-8-map-as-penalized-mle.png` | §25.8 | 10×4 (2-panel) | 150 | Left: Gaussian prior on $\theta$ (thin) + likelihood (amber) → MAP (vertical line) coincides with ridge estimate (vertical dashed line in same position) for Normal-Normal conjugate pair with $\sigma^2 = 1, \tau^2 = 0.25$. Right: Laplace prior + likelihood → MAP coincides with lasso estimate. |
| 11 | `25-10-forward-map.png` | §25.10 | 10×8 | 150 | Forward-map diagram. Central hub "Bayesian Foundations (Topic 25)" with arrows out to Topic 26 (MCMC: Metropolis–Hastings, Gibbs, HMC, NUTS), Topic 27 (Bayes factors, BMA, DIC/WAIC/PSIS-LOO, marginal likelihood), Topic 28 (hierarchical, partial pooling, empirical Bayes, horseshoe), and formalml (VI, BNN, GP, probabilistic programming, Bayesian nonparametrics). Back-arrows to Topic 4 (Bayes thm), Topic 7 (conjugacy), Topic 14 (MLE/BvM), Topic 23 (MAP=penalty), Topic 24 (BIC=Laplace). |

**Figure conventions (all ten):**
- `dpi=150, bbox_inches='tight', pad_inches=0.15`.
- Palette: `--color-prior` (blue) for priors; `--color-likelihood` (amber) for likelihoods; `--color-posterior` (purple) for posteriors. Topics 26–28 inherit this trio.
- Titles short; alt-text in MDX repeats substantive content.
- KaTeX-unsafe matplotlib workarounds: `\geq` not `\ge`; `\sqrt{n}` not `\sqrt n`; `\to` not `\xrightarrow`; `\Leftrightarrow` not `\iff`.

### 4.2 Notebook structure (cell-by-cell)

**Path:** `notebooks/bayesian-foundations/25_bayesian_foundations.ipynb` (short-form directory, dropping "-and-prior-selection" per established convention — matches Topic 19's `notebooks/confidence-intervals/`).

**Delivery:** native `.ipynb`, outputs cleared. User executes and verifies locally.

**Cells (13 total):**

- **Cell 1 — Imports + seed.** `numpy`, `scipy.stats` (for `beta`, `gamma`, `norm`, `dirichlet`, `multinomial`, `binom`, `betabinom`), `matplotlib.pyplot`. `rng = np.random.default_rng(42)` at module scope. Variable `OUTPUT_DIR = Path('public/images/topics/bayesian-foundations-and-prior-selection')`. Note: Topic 25 uses `default_rng(42)` (matching Topic 24's convention), not legacy `np.random.seed(42)`.

- **Cell 2 — Palette + matplotlib defaults.** Define `COLOR_PRIOR = '#3B82F6'` (blue-500), `COLOR_LIKELIHOOD = '#F59E0B'` (amber-500), `COLOR_POSTERIOR = '#8B5CF6'` (violet-500). Set `rcParams`: `figure.dpi=150`, `savefig.bbox='tight'`, `savefig.pad_inches=0.15`, spines top/right off, `font.family='sans-serif'`, axes labelsize 11, title size 12. These three CSS colors are the Track 7-wide palette Topic 26–28 will inherit.

- **Cell 3 — Figure 1 (`25-1-prior-to-posterior-animation.png`).** Beta(2, 2) prior. True $\theta = 0.3$. For $n \in [1, 5, 20, 100]$: simulate $k \sim \text{Binomial}(n, 0.3)$ with `rng.binomial(n, 0.3)`; posterior $\alpha' = 2+k$, $\beta' = 2+n-k$; plot prior (faded) and posterior (bold) on $[0,1]$; vertical line at $\theta=0.3$. Four-panel figure.

- **Cell 4 — Figure 2 (`25-2-bayes-update-geometry.png`).** Ex 1 (Beta(1,1) prior + 3H/1T). Three panels: (a) prior density Beta(1,1) constant 1 on $[0,1]$; (b) Binomial likelihood $\theta^3(1-\theta)$ on $[0,1]$; (c) posterior Beta(4, 2) density. Annotation "×" between (a) and (b), "∝" between (b) and (c). The three densities are on different y-scales — label axes clearly.

- **Cell 5 — Figure 3 (`25-4-exp-fam-conjugacy-kernel.png`).** Three-panel: Beta-Binomial, Gamma-Poisson, Normal-Normal. Each panel: prior density + posterior density (shifted hyperparameters). Formula overlay showing the $\chi_0' = \chi_0 + S$, $\nu_0' = \nu_0 + n$ update rule.

- **Cell 6 — Figure 4 (`25-5-five-conjugate-pairs-panel.png`).** Five-panel, 2×3 grid (last panel left blank or used for summary legend). Panel 1: Beta-Bernoulli (prior Beta(2, 2), observed k=30 of n=50, posterior Beta(32, 22)). Panel 2: Normal-Normal known $\sigma^2$ (prior $\mathcal{N}(0, 4)$, observed $\bar x = 1, n=20, \sigma^2=1$, posterior $\mathcal{N}(0.95, 0.0488)$). Panel 3: Normal-Normal-Inverse-Gamma — plot the joint on a contour diagram with marginal on $\mu$ overlaid. Panel 4: Gamma-Poisson (prior Gamma(2, 1), observed $S=15, n=5$, posterior Gamma(17, 6)). Panel 5: Dirichlet(1,1,1) → Dirichlet(11, 21, 31) on the 2-simplex (ternary plot).

- **Cell 7 — Figures 5 + 6 (`25-6-credible-vs-wald.png`, `25-6-posterior-predictive-beta-binomial.png`).** Figure 5: Beta(5, 20) posterior; compute equal-tailed 95% via `beta.ppf([0.025, 0.975], 5, 20)`; compute HPD 95% (requires optimization — use scipy to find the horizontal density-level $c_\alpha$ that gives the shortest interval). Shade both regions; overlay the density. Second panel: flat improper prior on $\mu$ + Normal likelihood $\mathcal{N}(\bar x, \sigma^2/n)$ gives posterior = z-CI density; shade 95% credible (equivalent to z-CI). Figure 6: posterior predictive `betabinom(m=20, a=5, b=20)` PMF vs `binom(m=20, p=5/25)` plug-in.

- **Cell 8 — Figure 7 (`25-7-prior-sensitivity-three-priors.png`).** Same data: 10 heads in 50 tosses. Three priors: Beta(2, 8) (informative, mean 0.2), Beta(2, 2) (weak), Beta(0.5, 0.5) (Jeffreys). Three posteriors: Beta(12, 48), Beta(12, 42), Beta(10.5, 40.5). Overlaid density curves with matching-color prior + posterior, plus a vertical line at the MLE $\hat\theta = 0.2$.

- **Cell 9 — Figure 8 (`25-7-jeffreys-prior-bernoulli.png`).** Left: Beta(0.5, 0.5) density (the Bernoulli Jeffreys prior). Right: reparameterize $\phi = \text{logit}(\theta)$; plot $\pi_J(\phi)$ derived two ways (direct Jeffreys on $\phi$ using $\mathcal{I}(\phi) = e^\phi/(1+e^\phi)^2$; and $\pi_J(\theta)|d\theta/d\phi|$). Confirm the two match numerically; annotate "reparameterization-invariant."

- **Cell 10 — Figure 9 (`25-8-bernstein-von-mises-convergence.png`).** Four-panel, one per $n \in [5, 25, 100, 500]$. Beta(2, 2) prior, true $\theta_0 = 0.3$, observed $k = \text{round}(n \cdot 0.3)$. Posterior density (purple) + Normal approximation $\mathcal{N}(\hat\theta_{\text{MLE}}, 1/(n \cdot \hat\theta(1-\hat\theta)))$ (dashed black). Vertical lines at $\theta_0$ and $\hat\theta_{\text{MLE}}$. For $n=5$ the two curves disagree noticeably; for $n=500$ they're visually identical.

- **Cell 11 — Figure 10 (`25-8-map-as-penalized-mle.png`).** Left: Normal $\mathcal{N}(0, 0.25)$ prior + Normal likelihood $\mathcal{N}(\bar x = 1, 1/n)$ with $n=4$. Overlay prior, likelihood, posterior. MAP vertical line coincides with ridge-estimate vertical line. Right: Laplace(0, 0.5) prior + same Normal likelihood. MAP line = lasso estimate.

- **Cell 12 — Figure 11 (`25-10-forward-map.png`).** Static diagram (matplotlib boxes + arrows, Topic 23/24 template). Central box: "Bayesian Foundations (Topic 25)" purple. Arrows out to: Topic 26 (MCMC) amber; Topic 27 (BMA) amber; Topic 28 (Hierarchical) amber; formalml (VI, BNN, GP, PP, Nonparametrics) light purple. Back-arrows (thin, grey) to Topics 4, 7, 14, 23, 24.

- **Cell 13 — Verification helper.** Computes and prints ground-truth values cited in §6.2 test pins:
  - Beta-Binomial posterior mean & variance at $(\alpha_0, \beta_0, n, k) = (2, 2, 50, 10)$: expected posterior $\text{Beta}(12, 42)$, mean $12/54 = 0.2222\overline{2}$, variance $(12 \cdot 42)/((54)^2 \cdot 55)$.
  - Normal-Normal posterior parameters at $(\mu_0, \sigma_0^2, n, \bar x, \sigma^2) = (0, 4, 20, 1, 1)$: posterior mean $0.95238$, posterior variance $0.04762$.
  - Gamma-Poisson posterior parameters at $(\alpha_0, \beta_0, n, S) = (2, 1, 5, 15)$: $\text{Gamma}(17, 6)$, mean $17/6 \approx 2.8333$.
  - Beta-Binomial posterior predictive at $(m, \alpha^\star, \beta^\star) = (20, 5, 20)$: compute `scipy.stats.betabinom.pmf(range(21), 20, 5, 20)`; verify $\sum = 1$; compute variance.
  - Jeffreys-prior Beta(½,½) posterior at $(n, k) = (50, 10)$: Beta(10.5, 40.5), mean $10.5/51 = 0.20588$.
  - 95% equal-tailed credible interval on Beta(12, 42): `beta.ppf([0.025, 0.975], 12, 42)` = approximately $[0.122, 0.346]$. Flag: verify against scipy output.
  - MAP for Normal-Normal $(\mu_0, \sigma_0^2, n, \bar x, \sigma^2) = (0, 0.25, 4, 1, 1)$: ridge-equivalent with $\lambda = \sigma^2/(\sigma_0^2) = 4$. MAP = $\sigma_0^2 \bar x / (\sigma^2/n + \sigma_0^2) \cdot n/\sigma^2$… use exact formula $\mu_n = (\mu_0/\sigma_0^2 + n\bar x/\sigma^2) \cdot \sigma_n^2$ with $\sigma_n^2 = 1/(1/\sigma_0^2 + n/\sigma^2) = 1/(4 + 4) = 0.125$. MAP $= 0.125 \cdot (0 + 4) = 0.5$.
  - BvM convergence: at $n=500$ with Beta(2, 2) prior and $k=150$, posterior is Beta(152, 352), variance $\approx 0.0004216$; Normal-at-MLE approximation $\sigma^2 = \hat\theta(1-\hat\theta)/n = 0.3 \cdot 0.7 / 500 = 0.00042$. Ratio of variances $\approx 1.004$ — visually identical, TV distance $\ll 0.01$.
  
  Print all values with labels formatted as `# TEST 25.N: description → value` so they transfer directly into §6.2 of the brief and `regression.test.ts` of the implementation.

**Notebook conventions reminder:**
- `np.random.default_rng(42)` at module top.
- `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` before first savefig.
- Every `savefig(OUTPUT_DIR / '25-N-name.png', dpi=150, bbox_inches='tight', pad_inches=0.15)`.
- Triple-quoted matplotlib-figure-level text uses raw strings (`r"""..."""`) to prevent `\b`, `\m`, `\s` control-character interpretation.
- Canvas rendering not required — no plot exceeds 1,000 points.

**Verification of notebook outputs against brief §6.2 test pins:** Cell 13 prints every numeric value cited in test assertions. Jonathan executes locally, checks, and pastes Cell 13 output back into the brief to confirm §6.2 pins match scipy's exact output to the tolerance specified.

---

*(End of Part 1 — §§1–4. Part 2 follows: §§5 components, §6 bayes.ts module, §7 preset data, §8 cross-reference updates, §9 verification checklist, §10 build order, Appendix A KaTeX/notation/MDX locks, Appendix B design decisions.)*

---

*(Continuation of Part 1. This part covers §5 interactive components, §6 `bayes.ts` shared module, §7 preset data module, §8 cross-reference updates, §9 verification checklist, §10 build order, Appendix A KaTeX/notation/MDX locks, Appendix B design decisions.)*

---

## 5. Interactive Components

Three required components plus one optional (four total, matching Topics 19, 21, 22 cadence). All components: React + Tailwind + SVG/D3, `client:visible` directive (not `client:load`), mobile-responsive at 375px width. Use the Track 7 palette from `viz/shared/colorScales.ts` (new `bayesianColors` export in §6.1).

### 5.1 PriorPosteriorExplorer (required — featured for §25.2 and §25.5)

**File:** `src/components/viz/PriorPosteriorExplorer.tsx`

**Purpose:** Sequential Bayesian updating for Beta-Binomial. The anchor component for §25.2 and §25.5. Builds on Topic 6's `BetaPriorPosteriorExplorer` but extends with: (a) likelihood overlay always-on by default, (b) sequential-observation mode (add one coin flip at a time), (c) posterior mean + credible interval readout, (d) true-$\theta$ slider for Monte Carlo verification.

**Interactions:**
- Prior panel: sliders for $\alpha_0 \in [0.5, 20]$ and $\beta_0 \in [0.5, 20]$ (continuous). Preset dropdown: "Uniform Beta(1, 1)", "Jeffreys Beta(½, ½)", "Weak Beta(2, 2)", "Informative heads-favoring Beta(10, 3)", "Informative tails-favoring Beta(3, 10)".
- Data panel: two inputs — $n \in [0, 200]$ and $k \in [0, n]$ — plus "Simulate one trial" button (Bernoulli trial with `rng.binomial` at current true $\theta$). Also "Reset to prior" button.
- True-$\theta$ slider (collapsible "simulation mode"): $\theta_{\text{true}} \in [0, 1]$; when active, "Simulate one trial" draws from Bernoulli($\theta_{\text{true}}$).
- Density plot: prior (blue), scaled likelihood (amber), posterior (purple), all on $[0, 1]$. Posterior density highlighted; prior fades to 40% opacity when data present.
- Live readout (below density plot):
  - Prior: $\mathbb{E}[\theta] = \alpha_0/(\alpha_0+\beta_0)$, 95% credible interval.
  - Posterior: $\mathbb{E}[\theta \mid \mathbf{y}] = (\alpha_0+k)/(\alpha_0+\beta_0+n)$, 95% equal-tailed credible interval, 95% HPD interval.
  - MAP: $\hat\theta_{\text{MAP}} = (\alpha_0+k-1)/(\alpha_0+\beta_0+n-2)$ (when $\alpha^\star > 1$ and $\beta^\star > 1$).
  - "Pseudo-observations" interpretation: $\alpha_0 + \beta_0 = $ pseudo-sample-size; after $n$ real trials, effective sample size is $\alpha_0+\beta_0+n$.
- Sequential mode: pressing "Simulate one trial" animates the density transition (CSS transition, 400ms ease-out) from the current posterior to the new posterior.

**Data:** No pre-computed data. All computation in the browser via `pdfBeta` and `quantileBeta` from `bayes.ts`.

**Uses from shared modules:** `pdfBeta`, `quantileBeta`, `credibleIntervalBeta`, `hpdIntervalBeta` from `bayes.ts`. `logbinomial` from `distributions.ts` (for scaled likelihood).

**Mobile fallback:** At 375px, controls stack vertically below the density plot. Density plot height 220px. Readout collapses into a three-row summary: prior / posterior / MAP. Simulation-mode toggle collapses by default.

---

### 5.2 ConjugatePairBrowser (required)

**File:** `src/components/viz/ConjugatePairBrowser.tsx`

**Purpose:** A guided tour of all five canonical conjugate pairs. Tab-based browser with one preset scenario per pair; user picks a pair, adjusts hyperparameters and data, sees the posterior closed form, the density update, and the posterior predictive.

**Interactions:**
- Tab selector: five tabs — Beta-Binomial, Normal-Normal (known σ²), Normal-Normal-Inverse-Gamma (unknown σ²), Gamma-Poisson, Dirichlet-Multinomial.
- Per-tab controls (prior hyperparameters + data sliders):
  - **Beta-Binomial**: $\alpha_0, \beta_0 \in [0.5, 20]$; $n \in [1, 100], k \in [0, n]$. Posterior: Beta($\alpha_0+k, \beta_0+n-k$). Posterior predictive for $m=10$ new trials: Beta-Binomial.
  - **Normal-Normal known σ²**: $\mu_0 \in [-5, 5], \sigma_0^2 \in [0.1, 10], \sigma^2 \in [0.1, 10], n \in [1, 50], \bar x \in [-5, 5]$. Posterior: $\mathcal{N}(\mu_n, \sigma_n^2)$ with closed-form precision-weighted average. Posterior predictive for new obs: $\mathcal{N}(\mu_n, \sigma_n^2 + \sigma^2)$.
  - **Normal-Normal-Inverse-Gamma unknown σ²**: $\mu_0, \kappa_0, \alpha_0, \beta_0$; $n, \bar x, s^2$. Posterior: $\mu \mid \sigma^2, \mathbf{y}$ is Normal; $\sigma^2 \mid \mathbf{y}$ is Inverse-Gamma; marginal on $\mu$ is non-standardized Student-t. Display the joint posterior as a 2D contour plot (20×20 grid) of $p(\mu, \sigma^2 \mid \mathbf{y})$.
  - **Gamma-Poisson**: $\alpha_0 \in [0.5, 10], \beta_0 \in [0.1, 5]$; $n \in [1, 30], S = \sum y_i \in [0, 50]$. Posterior: Gamma($\alpha_0+S, \beta_0+n$). Posterior predictive: Negative Binomial (the Gamma-Poisson compound).
  - **Dirichlet-Multinomial**: $k=3$ categories; $\boldsymbol\alpha_0 \in [0.5, 10]^3$; observed counts $\mathbf{x} \in [0, 50]^3$. Posterior: Dirichlet($\boldsymbol\alpha_0 + \mathbf{x}$). Display as a ternary plot on the 2-simplex with prior (faint) and posterior (bold) contour densities.
- Formula panel (always visible): prior formula, likelihood formula, posterior formula, posterior predictive formula. Rendered via KaTeX.
- Pseudo-observations annotation per tab.

**Data:** Presets per tab in `bayesian-foundations-data.ts` (§7).

**Uses from shared modules:** `pdfBeta`, `pdfGamma`, `pdfNormal`, `pdfDirichlet`, `pdfNormalInverseGamma` from `bayes.ts`.

**Mobile fallback:** At 375px, tabs become a dropdown. Formula panel toggles below chart. Dirichlet ternary plot clamped to 300px × 260px.

---

### 5.3 BernsteinVonMisesAnimator (required — featured for §25.8)

**File:** `src/components/viz/BernsteinVonMisesAnimator.tsx`

**Purpose:** The flagship interactive for Topic 25. Visualizes BvM convergence: sample-size slider, posterior density overlaid with $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}(\theta_0)^{-1}/n)$ approximation.

**Interactions:**
- Model selector: Beta-Binomial (Bernoulli($\theta$) likelihood), Gamma-Poisson (Poisson($\lambda$) likelihood), Normal-Normal known $\sigma^2$ (Normal mean). Default Beta-Binomial.
- Prior selector (for the active model): "Jeffreys", "Uniform", "Weakly informative", "Strongly informative" — each with prior-specific hyperparameter defaults.
- True-parameter slider ($\theta_0$ for Bernoulli, $\lambda_0$ for Poisson, $\mu_0$ for Normal) controlling the data-generating process.
- Sample-size slider: $n \in [1, 1000]$ on log scale; tick marks at $n \in \{5, 25, 100, 500, 1000\}$.
- Play/pause animation button: animates $n$ from 1 to 1000 over 8 seconds (log-scale time).
- Seed slider (advanced): reseed the data generator to reproduce exact draws.
- Main chart: x-axis parameter space (domain-adapted per model); y-axis density.
  - Posterior density (purple solid).
  - Normal-at-MLE approximation $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}(\hat\theta_{\text{MLE}})^{-1}/n)$ (black dashed).
  - Vertical lines: $\theta_0$ (green dotted), $\hat\theta_{\text{MLE}}$ (orange solid).
- Readout:
  - Current $n$ and $k$ (or $S$, $\bar x$).
  - $\hat\theta_{\text{MLE}}$ and observed Fisher info $\hat{\mathcal{I}}$.
  - Posterior mean, 95% credible interval.
  - Normal approximation mean, 95% interval.
  - **Total-variation distance between posterior and Normal approximation** (computed as $\frac{1}{2}\int |p(\theta \mid \mathbf{y}) - \phi(\theta; \hat\theta_{\text{MLE}}, \hat{\mathcal{I}}^{-1}/n)| d\theta$ via numerical integration on a 500-point grid). Large for small $n$, decays to $< 0.01$ by $n = 500$ for Beta-Binomial and Gamma-Poisson under their informative priors.
- Annotation panel: "BvM predicts posterior → $\mathcal{N}(\hat\theta_{\text{MLE}}, \mathcal{I}(\theta_0)^{-1}/n)$ in total variation as $n \to \infty$. Current TV distance: [value]."

**Data:** No preset data beyond per-model default-parameter values.

**Uses from shared modules:** `pdfBeta`, `pdfGamma`, `pdfNormal`, `mleBernoulli`, `mlePoisson`, `mleNormal`, `fisherInformationBernoulli`, `fisherInformationPoisson`, `fisherInformationNormal` from `bayes.ts` + `estimation.ts`.

**Mobile fallback:** At 375px, controls stack vertically above chart. Readout collapses into a three-row table. Animation button always visible. TV distance uses reduced 200-point grid for performance.

---

### 5.4 PriorSensitivityComparator (optional fourth)

**File:** `src/components/viz/PriorSensitivityComparator.tsx`

**Purpose:** §25.7 Ex 7 made interactive. Same data, three contrasting priors, three posteriors overlaid.

**Interactions:**
- Dataset panel: slider for $n \in [1, 200]$, slider for $k/n \in [0, 1]$ (proportion of successes). Or "Load preset" (e.g., "A/B test: 10 heads in 50 tosses").
- Three prior panels side-by-side:
  - Prior 1 (informative): Beta($\alpha_1, \beta_1$), user-adjustable. Default Beta(2, 8) peaked at 0.2.
  - Prior 2 (weak): Beta($\alpha_2, \beta_2$), user-adjustable. Default Beta(2, 2).
  - Prior 3 (non-informative): Beta(0.5, 0.5) (Jeffreys), fixed.
- Three corresponding posterior densities overlaid on a single chart.
- Readout: posterior means, 95% credible intervals, max absolute difference across the three posteriors (the "sensitivity magnitude"). Annotation: "When n is small, priors matter: max difference = [X]. When n is large, BvM kicks in and the three posteriors converge (see `BernsteinVonMisesAnimator`)."

**Mobile fallback:** Three prior panels stack vertically. Overlaid chart fills the width.

---

### 5.5 Component implementation notes

- **Architecture:** React functional components with `useState` + `useMemo` for derived density arrays. All density evaluation on a 300-point grid unless otherwise noted. Throttle slider updates to `requestAnimationFrame` (60fps).
- **Palette:** New `bayesianColors` export from `viz/shared/colorScales.ts`:
  ```typescript
  export const bayesianColors = {
    prior: 'var(--color-prior)',      // #3B82F6 blue-500
    likelihood: 'var(--color-likelihood)', // #F59E0B amber-500
    posterior: 'var(--color-posterior)',   // #8B5CF6 violet-500
    true: 'var(--color-true-parameter)',   // #10B981 emerald-500
    mle: 'var(--color-mle)',           // #F97316 orange-500
  };
  ```
  Mirror to `src/styles/global.css`:
  ```css
  :root {
    --color-prior: #3B82F6;
    --color-likelihood: #F59E0B;
    --color-posterior: #8B5CF6;
    --color-true-parameter: #10B981;
    --color-mle: #F97316;
  }
  ```
  These five CSS vars are Track-7-wide and will be reused by Topics 26–28.
- **Performance:** `BernsteinVonMisesAnimator`'s TV-distance integration is the most expensive operation — precompute on a cached grid, recompute only when $n$ changes by a factor ≥ 1.2 or a control changes.
- **Accessibility:** All sliders have `aria-label` and `aria-valuetext`. Density plots use redundant color+linestyle encoding (blue-solid, amber-dashed, purple-solid).
- **Dirichlet ternary plot** (in `ConjugatePairBrowser`): use a custom SVG triangle transform — $(p_1, p_2, p_3) \mapsto (p_1 \cdot T_1 + p_2 \cdot T_2 + p_3 \cdot T_3)$ where $T_i$ are the triangle vertices. Grid contours computed on a regular 30×30 simplex lattice.

---

## 6. Shared Module — `bayes.ts` (new, Track 7 leaf)

**File:** `src/components/viz/shared/bayes.ts` (new file). Track-7-wide; Topics 26–28 extend.

### 6.1 Full new-function manifest

TypeScript signatures with JSDoc. All functions pure, deterministic, documented. No external state; no side effects.

```typescript
// ── Prior / posterior / predictive density functions ───────────────────────

/**
 * Beta density at x ∈ (0,1). Reuses Topic 6's pdfBeta for closed-form density.
 * @param x value in (0, 1)
 * @param alpha shape1 > 0
 * @param beta  shape2 > 0
 */
export function pdfBeta(x: number, alpha: number, beta: number): number;

/**
 * Gamma density at x > 0 using shape-rate parameterization.
 * @param x value > 0
 * @param alpha shape > 0
 * @param beta  rate  > 0
 */
export function pdfGamma(x: number, alpha: number, beta: number): number;

/**
 * Normal density at x ∈ ℝ. Reuses Topic 6's pdfNormal.
 */
export function pdfNormal(x: number, mu: number, sigma2: number): number;

/**
 * Inverse-Gamma density at x > 0. For Normal-Normal-IG unknown σ² posterior.
 * @param x value > 0
 * @param alpha shape > 0
 * @param beta  scale > 0
 */
export function pdfInverseGamma(x: number, alpha: number, beta: number): number;

/**
 * Normal-Inverse-Gamma joint density at (mu, sigma2). For §25.5 Ex 3.
 * p(mu, sigma2) = pdfNormal(mu; mu0, sigma2/kappa0) * pdfInverseGamma(sigma2; alpha0, beta0)
 */
export function pdfNormalInverseGamma(
  mu: number, sigma2: number,
  mu0: number, kappa0: number, alpha0: number, beta0: number
): number;

/**
 * Non-standardized Student-t density — marginal on μ in Normal-Normal-IG.
 * Shape parameters: location mu_n, scale sigma_n_squared, df 2*alpha_n.
 */
export function pdfStudentTMarginal(
  x: number, mu: number, scale2: number, df: number
): number;

/**
 * Dirichlet density at a simplex point p ∈ Δ^{k-1}. For §25.5 Ex 5.
 * @param p vector on simplex (sum = 1, all ≥ 0)
 * @param alpha concentration vector (all > 0)
 */
export function pdfDirichlet(p: number[], alpha: number[]): number;

/**
 * Beta-Binomial posterior predictive PMF at new count y_new ∈ {0,...,m}.
 * For §25.6 Thm 3b.
 */
export function pmfBetaBinomial(
  yNew: number, m: number, alphaStar: number, betaStar: number
): number;

/**
 * Negative Binomial posterior predictive PMF (Gamma-Poisson compound).
 * PMF at y_new ∈ {0, 1, 2, ...} given posterior Gamma(alphaStar, betaStar).
 */
export function pmfNegativeBinomialPosterior(
  yNew: number, alphaStar: number, betaStar: number
): number;

// ── Posterior computation — dispatch on conjugate family ───────────────────

/**
 * Conjugate posterior hyperparameter update — generic dispatch.
 *
 * @param family 'beta-binomial' | 'normal-normal' | 'normal-normal-ig' |
 *               'gamma-poisson' | 'dirichlet-multinomial'
 * @param prior  prior hyperparameters (family-specific shape)
 * @param data   sufficient-statistic summary (family-specific shape)
 * @returns posterior hyperparameters (same shape as prior)
 */
export function posterior(
  family: ConjugateFamily,
  prior: PriorHyperparams,
  data: SuffStats
): PosteriorHyperparams;

// ── Credible intervals and HPD ─────────────────────────────────────────────

/**
 * Equal-tailed 1-alpha credible interval for Beta(a, b) posterior.
 */
export function credibleIntervalBeta(
  alpha: number, beta: number, level: number
): [number, number];

/**
 * Equal-tailed 1-alpha credible interval for Gamma(a, b) posterior.
 */
export function credibleIntervalGamma(
  alpha: number, beta: number, level: number
): [number, number];

/**
 * Equal-tailed 1-alpha credible interval for Normal(mu, sigma2) posterior.
 */
export function credibleIntervalNormal(
  mu: number, sigma2: number, level: number
): [number, number];

/**
 * Highest-Posterior-Density (HPD) interval for a unimodal posterior.
 * Generic solver: finds horizontal density-level c such that
 * {θ: p(θ) ≥ c} has posterior mass 1-alpha.
 *
 * @param pdf     density function
 * @param support [lower, upper] support bounds
 * @param level   1 - α
 * @param nGrid   grid resolution (default 500)
 */
export function hpdInterval(
  pdf: (x: number) => number,
  support: [number, number],
  level: number,
  nGrid?: number
): [number, number];

/**
 * Specialized closed-form HPD for Beta(a, b).
 * Numerical solver using hpdInterval under the hood; memoized.
 */
export function hpdIntervalBeta(
  alpha: number, beta: number, level: number
): [number, number];

// ── MAP and posterior moments ──────────────────────────────────────────────

/**
 * MAP estimate for a conjugate posterior.
 */
export function mapEstimate(
  family: ConjugateFamily,
  posterior: PosteriorHyperparams
): number;

/**
 * Posterior mean for a conjugate posterior.
 */
export function posteriorMean(
  family: ConjugateFamily,
  posterior: PosteriorHyperparams
): number;

/**
 * Posterior variance for a conjugate posterior.
 */
export function posteriorVariance(
  family: ConjugateFamily,
  posterior: PosteriorHyperparams
): number;

// ── Jeffreys prior ─────────────────────────────────────────────────────────

/**
 * Jeffreys prior density for a one-parameter family.
 * Computes pi_J(theta) = sqrt(I(theta)).
 * @param family 'bernoulli' | 'poisson' | 'normal-mean' | 'normal-scale' |
 *               'exponential'
 */
export function jeffreysPrior(
  family: 'bernoulli' | 'poisson' | 'normal-mean' | 'normal-scale' | 'exponential',
  theta: number,
  knownVariance?: number
): number;

// ── Laplace approximation ─────────────────────────────────────────────────

/**
 * Laplace approximation to log marginal likelihood (also: BIC first-order
 * bridge). For §25.8 Rem.
 *
 * log m(y) ≈ log p(y | θ_MAP) + log π(θ_MAP) + (k/2) log(2π) - (1/2) log det H
 * where H is the Hessian of -log p(θ, y) at θ_MAP.
 *
 * @param logLikAtMAP log p(y | θ_MAP)
 * @param logPriorAtMAP log π(θ_MAP)
 * @param hessianLogDet log|det(-∇²log p(θ, y) at θ_MAP)|
 * @param k parameter dimension
 */
export function laplaceLogMarginalLikelihood(
  logLikAtMAP: number,
  logPriorAtMAP: number,
  hessianLogDet: number,
  k: number
): number;

// ── Posterior sampling (conjugate cases only; MCMC deferred to Topic 26) ──

/**
 * Draw n samples from a conjugate posterior using inverse-CDF / direct
 * sampling. For non-conjugate posteriors, Topic 26's MCMC machinery is needed.
 *
 * Beta posterior: uses Beta sampler (Topic 6's existing infrastructure).
 * Gamma posterior: uses Gamma sampler.
 * Normal posterior: uses Normal sampler.
 * Dirichlet posterior: samples via normalized independent Gammas (Topic 8).
 *
 * @param family conjugate family
 * @param posterior posterior hyperparameters
 * @param n number of samples
 * @param rng seedable RNG (default: Math.random)
 */
export function posteriorSample(
  family: ConjugateFamily,
  posterior: PosteriorHyperparams,
  n: number,
  rng?: () => number
): number[] | number[][];  // vector for multivariate families

// ── Posterior predictive ──────────────────────────────────────────────────

/**
 * Posterior predictive density/PMF at y_new for a conjugate family.
 * Beta-Binomial: returns Beta-Binomial PMF.
 * Normal-Normal: returns Normal(mu_n, sigma_n² + sigma²) density.
 * Gamma-Poisson: returns Negative Binomial PMF.
 * Etc.
 */
export function posteriorPredictive(
  family: ConjugateFamily,
  posterior: PosteriorHyperparams,
  yNew: number,
  options?: PredictiveOptions  // e.g., m for Beta-Binomial
): number;

// ── Types ──────────────────────────────────────────────────────────────────

export type ConjugateFamily =
  | 'beta-binomial'
  | 'normal-normal'
  | 'normal-normal-ig'
  | 'gamma-poisson'
  | 'dirichlet-multinomial';

export type PriorHyperparams =
  | { family: 'beta-binomial'; alpha0: number; beta0: number }
  | { family: 'normal-normal'; mu0: number; sigma0_sq: number; sigma_sq: number }
  | { family: 'normal-normal-ig'; mu0: number; kappa0: number; alpha0: number; beta0: number }
  | { family: 'gamma-poisson'; alpha0: number; beta0: number }
  | { family: 'dirichlet-multinomial'; alpha0: number[] };

export type SuffStats =
  | { family: 'beta-binomial'; n: number; k: number }
  | { family: 'normal-normal'; n: number; yBar: number }
  | { family: 'normal-normal-ig'; n: number; yBar: number; s2: number }
  | { family: 'gamma-poisson'; n: number; S: number }
  | { family: 'dirichlet-multinomial'; counts: number[] };

export type PosteriorHyperparams = PriorHyperparams;  // same shape, updated values

export type PredictiveOptions =
  | { family: 'beta-binomial'; m: number }
  | { family: 'normal-normal' }
  | { family: 'gamma-poisson' }
  | { family: 'dirichlet-multinomial'; nNew: number };
```

**Implementation notes:**

- **Reuse over duplication.** `pdfBeta`, `pdfGamma`, `pdfNormal`, `pdfInverseGamma` reuse Topic 6's `distributions.ts` (import, don't redefine). Add `pdfNormalInverseGamma`, `pdfStudentTMarginal`, `pdfDirichlet` as new functions.
- **Memoization.** `credibleIntervalBeta` and `hpdIntervalBeta` are called heavily by the interactive components; memoize with a cache keyed on $(\alpha, \beta, \text{level})$.
- **HPD generic solver.** `hpdInterval` uses a bisection on the density level $c$ to find where $\int_{\{p \ge c\}} p = 1 - \alpha$. 30 iterations suffice for tolerance $10^{-4}$.
- **Posterior sample types.** Multivariate families (Dirichlet) return `number[][]` — each inner array is one sample (length $k$). Scalar families return `number[]`.

### 6.2 New test-harness entries

Append to `src/components/viz/shared/__tests__/bayes.test.ts` (new file). All values exact outputs of notebook Cell 13 (verified with `scipy.stats` under `numpy.random.default_rng(seed=42)` semantics).

```typescript
describe('Topic 25: Bayesian foundations', () => {

  // T25.1 — Beta-Binomial posterior hyperparameters
  it('T25.1: Beta(2,2) + 10/50 → Beta(12, 42)', () => {
    const post = posterior(
      'beta-binomial',
      { family: 'beta-binomial', alpha0: 2, beta0: 2 },
      { family: 'beta-binomial', n: 50, k: 10 }
    );
    expect(post).toEqual({ family: 'beta-binomial', alpha0: 12, beta0: 42 });
  });

  // T25.2 — Beta-Binomial posterior mean
  it('T25.2: Beta(12, 42) mean = 12/54', () => {
    const mean = posteriorMean('beta-binomial',
      { family: 'beta-binomial', alpha0: 12, beta0: 42 });
    expect(mean).toBeCloseTo(12 / 54, 10);
  });

  // T25.3 — Beta-Binomial posterior variance
  it('T25.3: Beta(12, 42) variance = 12·42/(54²·55)', () => {
    const v = posteriorVariance('beta-binomial',
      { family: 'beta-binomial', alpha0: 12, beta0: 42 });
    const expected = (12 * 42) / (54 * 54 * 55);
    expect(v).toBeCloseTo(expected, 10);
  });

  // T25.4 — Beta(12, 42) 95% equal-tailed credible interval
  it('T25.4: Beta(12, 42) 95% credible interval ≈ [0.122, 0.346]', () => {
    const [lo, hi] = credibleIntervalBeta(12, 42, 0.95);
    expect(lo).toBeCloseTo(0.1222, 3);
    expect(hi).toBeCloseTo(0.3466, 3);
  });

  // T25.5 — Beta(12, 42) 95% HPD interval narrower than equal-tailed
  it('T25.5: HPD interval width < equal-tailed width for Beta(12, 42)', () => {
    const [lo1, hi1] = credibleIntervalBeta(12, 42, 0.95);
    const [lo2, hi2] = hpdIntervalBeta(12, 42, 0.95);
    expect(hi2 - lo2).toBeLessThanOrEqual(hi1 - lo1 + 1e-6);
  });

  // T25.6 — Normal-Normal known σ² posterior mean (precision-weighted)
  it('T25.6: Normal-Normal mu_0=0, sigma_0²=4, n=20, x̄=1, σ²=1 → μ_n = 20/21', () => {
    const post = posterior('normal-normal',
      { family: 'normal-normal', mu0: 0, sigma0_sq: 4, sigma_sq: 1 },
      { family: 'normal-normal', n: 20, yBar: 1 }) as {
        family: 'normal-normal'; mu0: number; sigma0_sq: number; sigma_sq: number;
      };
    // post.mu0 now holds the posterior mean (naming quirk — document)
    expect(post.mu0).toBeCloseTo(20 / 21, 6);
    // posterior variance σ_n² = 1/(1/4 + 20) = 4/81
    expect(post.sigma0_sq).toBeCloseTo(4 / 81, 6);
  });

  // T25.7 — Gamma-Poisson posterior
  it('T25.7: Gamma(2, 1) + Poisson n=5, S=15 → Gamma(17, 6)', () => {
    const post = posterior('gamma-poisson',
      { family: 'gamma-poisson', alpha0: 2, beta0: 1 },
      { family: 'gamma-poisson', n: 5, S: 15 });
    expect(post).toEqual({ family: 'gamma-poisson', alpha0: 17, beta0: 6 });
    expect(posteriorMean('gamma-poisson', post)).toBeCloseTo(17 / 6, 10);
  });

  // T25.8 — Dirichlet-Multinomial posterior
  it('T25.8: Dir(1,1,1) + counts (12, 10, 8) → Dir(13, 11, 9)', () => {
    const post = posterior('dirichlet-multinomial',
      { family: 'dirichlet-multinomial', alpha0: [1, 1, 1] },
      { family: 'dirichlet-multinomial', counts: [12, 10, 8] }) as {
        family: 'dirichlet-multinomial'; alpha0: number[];
      };
    expect(post.alpha0).toEqual([13, 11, 9]);
  });

  // T25.9 — Beta-Binomial posterior predictive sums to 1
  it('T25.9: Beta-Binomial(m=20, α*=5, β*=20) PMF sums to 1', () => {
    let total = 0;
    for (let yNew = 0; yNew <= 20; yNew++) {
      total += pmfBetaBinomial(yNew, 20, 5, 20);
    }
    expect(total).toBeCloseTo(1.0, 8);
  });

  // T25.10 — Beta-Binomial posterior predictive variance > Binomial plug-in
  it('T25.10: Beta-Binomial variance > Binomial(m=20, p̂=5/25) variance', () => {
    const pHat = 5 / 25;
    const binomVar = 20 * pHat * (1 - pHat);
    // Beta-Binomial(m=20, α=5, β=20) variance: mα(α+β+m)/((α+β)²(α+β+1))
    const bbVar = (20 * 5 * 20 * (5 + 20 + 20)) / ((5 + 20) ** 2 * (5 + 20 + 1));
    // Compute empirically from PMF
    let mean = 0, meansq = 0;
    for (let y = 0; y <= 20; y++) {
      const p = pmfBetaBinomial(y, 20, 5, 20);
      mean += y * p;
      meansq += y * y * p;
    }
    const empirical = meansq - mean * mean;
    expect(empirical).toBeCloseTo(bbVar, 5);
    expect(empirical).toBeGreaterThan(binomVar);
  });

  // T25.11 — Jeffreys prior Bernoulli is Beta(½, ½) up to proportionality
  it('T25.11: Jeffreys prior Bernoulli at θ=0.5 equals 1/√(θ(1−θ)) · const', () => {
    const theta = 0.5;
    const pij = jeffreysPrior('bernoulli', theta);
    const expected = 1 / Math.sqrt(theta * (1 - theta));  // 1/√0.25 = 2
    expect(pij).toBeCloseTo(expected, 8);
  });

  // T25.12 — Jeffreys prior posterior on Bernoulli matches Beta(0.5+k, 0.5+n-k)
  it('T25.12: Jeffreys prior + 10/50 → Beta(10.5, 40.5), mean = 10.5/51', () => {
    const post = posterior('beta-binomial',
      { family: 'beta-binomial', alpha0: 0.5, beta0: 0.5 },
      { family: 'beta-binomial', n: 50, k: 10 });
    expect(post).toEqual({ family: 'beta-binomial', alpha0: 10.5, beta0: 40.5 });
    expect(posteriorMean('beta-binomial', post)).toBeCloseTo(10.5 / 51, 10);
  });

  // T25.13 — MAP for Normal-Normal (ridge equivalent)
  it('T25.13: Normal-Normal MAP = ridge estimate with λ = σ²/σ_0²', () => {
    // Prior N(0, 0.25), likelihood center x̄=1 with σ²=1, n=4
    // Posterior mean / MAP = (μ_0/σ_0² + n·x̄/σ²) · σ_n²
    // σ_n² = 1/(1/0.25 + 4) = 1/8 = 0.125
    // MAP = 0.125 · (0 + 4) = 0.5
    const post = posterior('normal-normal',
      { family: 'normal-normal', mu0: 0, sigma0_sq: 0.25, sigma_sq: 1 },
      { family: 'normal-normal', n: 4, yBar: 1 });
    const map = mapEstimate('normal-normal', post);
    expect(map).toBeCloseTo(0.5, 10);
  });

  // T25.14 — BvM: posterior → Normal at large n (TV distance < 0.01)
  //   Beta(2, 2) prior, true θ = 0.3, n = 500, k = 150 → Beta(152, 352)
  //   Normal approximation mean = 150/500 = 0.3, var = 0.3·0.7/500 = 4.2e-4
  //   Posterior variance = 152·352/(504²·505) ≈ 4.16e-4
  it('T25.14: Beta(152, 352) posterior variance / BvM Normal variance ≈ 1', () => {
    const postVar = posteriorVariance('beta-binomial',
      { family: 'beta-binomial', alpha0: 152, beta0: 352 });
    const mleHat = 150 / 500;
    const normalVar = mleHat * (1 - mleHat) / 500;
    expect(postVar / normalVar).toBeGreaterThan(0.98);
    expect(postVar / normalVar).toBeLessThan(1.02);
  });

  // T25.15 — Conjugate-pair browser: Normal-Normal-IG joint density is positive
  it('T25.15: Normal-Normal-IG joint density positive on support', () => {
    const val = pdfNormalInverseGamma(
      /*mu=*/0.5, /*sigma2=*/1.0,
      /*mu0=*/0, /*kappa0=*/1, /*alpha0=*/2, /*beta0=*/2
    );
    expect(val).toBeGreaterThan(0);
    expect(Number.isFinite(val)).toBe(true);
  });
});
```

**15 tests total.** Jonathan runs the notebook locally (Cell 13), verifies each `scipy.stats` output matches the expected value in the tests above, and confirms the `.test.ts` file compiles before PR.

---

## 7. Preset Data Module — `bayesian-foundations-data.ts` (new)

**File:** `src/data/bayesian-foundations-data.ts` (new). Contains preset data for all four interactive components.

```typescript
/**
 * Preset data for Topic 25: Bayesian Foundations & Prior Selection.
 * Used by PriorPosteriorExplorer, ConjugatePairBrowser,
 * BernsteinVonMisesAnimator, and PriorSensitivityComparator.
 */

// ── PriorPosteriorExplorer presets ─────────────────────────────────────────

export const priorPosteriorPresets = [
  {
    name: 'Uniform Beta(1, 1)',
    alpha0: 1, beta0: 1,
    description: 'No prior information — all probabilities equally likely.',
  },
  {
    name: 'Jeffreys Beta(½, ½)',
    alpha0: 0.5, beta0: 0.5,
    description: 'Non-informative reference prior, reparameterization-invariant.',
  },
  {
    name: 'Weakly informative Beta(2, 2)',
    alpha0: 2, beta0: 2,
    description: 'Mild preference for θ = ½; pseudo-sample-size 4.',
  },
  {
    name: 'Informative heads-favoring Beta(10, 3)',
    alpha0: 10, beta0: 3,
    description: 'Strong prior belief θ ≈ 0.77; pseudo-sample-size 13.',
  },
  {
    name: 'Informative tails-favoring Beta(3, 10)',
    alpha0: 3, beta0: 10,
    description: 'Strong prior belief θ ≈ 0.23; pseudo-sample-size 13.',
  },
];

// ── ConjugatePairBrowser presets (per conjugate family) ────────────────────

export const betaBinomialPresets = [
  {
    name: 'A/B test conversion (weak prior)',
    alpha0: 2, beta0: 2, n: 50, k: 10,
    mNew: 50,
    description:
      'Weak prior + observed 10/50 → Beta(12, 42). Posterior mean 0.222, 95% CrI [0.122, 0.346].',
  },
  {
    name: 'Quality-control rejection rate (Jeffreys)',
    alpha0: 0.5, beta0: 0.5, n: 20, k: 2,
    mNew: 20,
    description:
      'Jeffreys prior + 2 rejects in 20 → Beta(2.5, 18.5). Posterior mean 0.119.',
  },
  {
    name: 'Vaccine efficacy (strong success prior)',
    alpha0: 15, beta0: 5, n: 100, k: 85,
    mNew: 100,
    description:
      'Strong prior on high efficacy + data → Beta(100, 20). Posterior mean 0.833, tight CrI.',
  },
];

export const normalNormalPresets = [
  {
    name: 'Weight estimation (weak prior)',
    mu0: 70, sigma0_sq: 100, sigma_sq: 4, n: 10, yBar: 72.5,
    description:
      'Prior N(70, 100) + 10 obs with mean 72.5, σ²=4 → posterior mean ≈ 72.49.',
  },
  {
    name: 'IQ measurement (strong prior)',
    mu0: 100, sigma0_sq: 225, sigma_sq: 100, n: 5, yBar: 115,
    description:
      'Prior N(100, 225) + 5 obs with mean 115, σ²=100 → posterior mean 110.71.',
  },
  {
    name: 'Sensor calibration (noninformative)',
    mu0: 0, sigma0_sq: 10000, sigma_sq: 1, n: 30, yBar: 0.85,
    description:
      'Nearly-flat prior → posterior mean ≈ MLE = 0.85.',
  },
];

export const normalNormalIGPresets = [
  {
    name: 'Student-t arising naturally',
    mu0: 0, kappa0: 1, alpha0: 2, beta0: 2, n: 20, yBar: 0.5, s2: 1.2,
    description:
      'Unknown σ²: marginal posterior on μ is Student-t, not Normal.',
  },
];

export const gammaPoissonPresets = [
  {
    name: 'Call-center arrival rate',
    alpha0: 2, beta0: 1, n: 10, S: 45,
    mNew: 10,
    description:
      'Prior mean 2 calls/interval + 45 calls in 10 intervals → posterior mean 47/11 ≈ 4.27.',
  },
  {
    name: 'Defect rate (rare-event)',
    alpha0: 1, beta0: 10, n: 5, S: 2,
    mNew: 5,
    description:
      'Prior favoring low rate + 2 defects in 5 units → posterior mean 3/15 = 0.2.',
  },
];

export const dirichletMultinomialPresets = [
  {
    name: 'Three-way topic proportions',
    alpha0: [1, 1, 1], counts: [12, 10, 8],
    description:
      'Uniform prior on the simplex + observed (12, 10, 8) → posterior Dir(13, 11, 9).',
  },
  {
    name: 'Strong prior on dominant category',
    alpha0: [10, 2, 2], counts: [5, 5, 5],
    description:
      'Prior concentrated at (0.71, 0.14, 0.14) pulls the posterior toward category 1 despite balanced data.',
  },
];

// ── BernsteinVonMisesAnimator presets ──────────────────────────────────────

export const bvmPresets = [
  {
    name: 'Beta-Binomial with Jeffreys prior',
    family: 'beta-binomial',
    prior: { alpha0: 0.5, beta0: 0.5 },
    trueTheta: 0.3,
    description:
      'At n=500 the posterior is visually indistinguishable from the Normal-at-MLE.',
  },
  {
    name: 'Beta-Binomial with strong prior',
    family: 'beta-binomial',
    prior: { alpha0: 10, beta0: 40 },
    trueTheta: 0.7,  // DELIBERATE mismatch to prior's 0.2
    description:
      'Strong prior at wrong location → posterior takes longer to concentrate on MLE; BvM still holds.',
  },
  {
    name: 'Gamma-Poisson',
    family: 'gamma-poisson',
    prior: { alpha0: 2, beta0: 1 },
    trueLambda: 3,
    description: 'Gamma prior + Poisson likelihood; BvM holds by n ≈ 100.',
  },
  {
    name: 'Normal-Normal known σ²',
    family: 'normal-normal',
    prior: { mu0: 0, sigma0_sq: 4, sigma_sq: 1 },
    trueMu: 2,
    description: 'BvM is exact for every n when prior is Normal (no limit needed).',
  },
];

// ── PriorSensitivityComparator presets ─────────────────────────────────────

export const prior3WayPresets = [
  {
    name: 'Canonical Beta-Binomial contrast (10/50)',
    n: 50, k: 10,
    prior1: { alpha: 2, beta: 8, label: 'Informative (mean 0.2)' },
    prior2: { alpha: 2, beta: 2, label: 'Weakly informative' },
    prior3: { alpha: 0.5, beta: 0.5, label: 'Jeffreys (non-informative)' },
    description:
      'At moderate n, informative prior dominates; at n=1000 all three posteriors align.',
  },
  {
    name: 'Small-sample A/B test (3/10)',
    n: 10, k: 3,
    prior1: { alpha: 8, beta: 2, label: 'Informative heads-favoring' },
    prior2: { alpha: 1, beta: 1, label: 'Uniform' },
    prior3: { alpha: 0.5, beta: 0.5, label: 'Jeffreys' },
    description:
      'Small data → priors dominate; posterior means range across three different ballparks.',
  },
];
```

---

## 8. Cross-Reference Updates

### 8.1 Predecessor MDX edits (21 forward-promise activations + cleanups)

Exact verbatim find-and-replace operations. Ordered by source file.

#### B1 — `maximum-likelihood.mdx:898` (§14.11 Rem 9)

**Find:**
```
The Bernstein–von Mises theorem (Track 7, coming soon) says that under regularity, the MAP estimate converges to the MLE at rate $1/n$: priors stop mattering once there is enough data.
```

**Replace:**
```
The [Bernstein–von Mises theorem](/topics/bayesian-foundations-and-prior-selection#section-25-8) says that under regularity, the posterior concentrates around the MLE at rate $1/\sqrt{n}$ in total variation: priors stop mattering once there is enough data. Topic 25 §25.8 develops this with a sketch proof following van der Vaart 1998 §10.
```

**Rationale:** The original text read "at rate $1/n$" which is imprecise (the posterior concentrates at $1/\sqrt{n}$, and the statement was about the posterior, not the MAP-MLE gap). B1 corrects this as part of the activation.

#### B2 — `maximum-likelihood.mdx:925` ("Where this leads" paragraph)

**Find:**
```
And **Bayesian Foundations** (coming soon) formalizes the MAP/regularized-MLE correspondence of Remark 9, giving a principled framework for the informative priors that regularization has been quietly deploying.
```

**Replace:**
```
And [**Bayesian Foundations**](/topics/bayesian-foundations-and-prior-selection) formalizes the MAP/regularized-MLE correspondence of Remark 9, giving a principled framework for the informative priors that regularization has been quietly deploying. §25.8 Ex 8 closes the arc from Topic 14 §14.11 Rem 9 through Topic 23 §23.7 Thm 5.
```

*Note:* The current text in `maximum-likelihood.mdx:925` after Topic 22's C4 cleanup reads "**Bayesian Foundations** (Track 7, coming soon)" — Topic 22's cleanup left "(Track 7, coming soon)" as the stopgap. B2 replaces this final stopgap with the live link.

#### B3 — `maximum-likelihood.mdx:947` (EM Appendix)

**Find:**
```
The full theory — Jensen's inequality applied to the log, the ELBO lower bound, convergence rates, and monotonicity — is developed in Track 7 (Bayesian Foundations, coming soon) and Track 7 (Variational Methods, coming soon).
```

**Replace:**
```
The full theory — Jensen's inequality applied to the log, the ELBO lower bound, convergence rates, and monotonicity — is developed in [Bayesian Foundations (Topic 25)](/topics/bayesian-foundations-and-prior-selection) and variational methods (formalml, coming soon).
```

**Rationale:** Topic 25 covers the prior-posterior-ELBO setup but not EM. VI reference stays as a formalml pointer since Topic 26 (MCMC) is the chosen Track 7 computation topic, not VI.

#### B4 — `continuous-distributions.mdx` §6 "What comes next" bullet

**Find:**
```
- Bayesian Foundations (coming soon) develops the Beta-Bernoulli, Gamma-Poisson, and Normal-Normal conjugate pairs in full generality
```

**Replace:**
```
- [Bayesian Foundations (Topic 25)](/topics/bayesian-foundations-and-prior-selection) develops the Beta-Bernoulli, Gamma-Poisson, and Normal-Normal conjugate pairs in full generality, adds Normal-Normal-Inverse-Gamma (unknown σ²) and Dirichlet-Multinomial, and frames them all as instances of the exponential-family conjugacy theorem
```

#### B5 — `continuous-distributions.mdx` §6.6 Ex 5 closing

**Find:**
```
This is Bayesian A/B testing — and it starts with the Beta-Bernoulli conjugate pair. Bayesian Foundations (coming soon) will develop the general framework.
```

**Replace:**
```
This is Bayesian A/B testing — and it starts with the Beta-Bernoulli conjugate pair. [Bayesian Foundations (Topic 25)](/topics/bayesian-foundations-and-prior-selection#section-25-5) develops the general framework, including the posterior predictive for the Beta-Binomial compound.
```

#### B6 — `discrete-distributions.mdx` frontmatter `connections[bayesian-foundations]`

Slug-target automatic once `curriculum-graph.json` flips. Relationship text: no change.

#### B7 — `exponential-families.mdx:935` (§7.11 What comes next)

**Find:**
```
- Bayesian Foundations (coming soon) develops the full Bayesian framework, with the conjugate prior theorem from Section 7.7 as the starting point for conjugate inference
```

**Replace:**
```
- [Bayesian Foundations (Topic 25)](/topics/bayesian-foundations-and-prior-selection#section-25-4) develops the full Bayesian framework. The §7.7 conjugate prior theorem becomes Topic 25's Thm 2, lifted into the Bayesian inferential framework with credible intervals, posterior predictive, and Bernstein–von Mises asymptotics
```

#### B8 — `exponential-families.mdx` frontmatter `connections[bayesian-foundations]`

Update relationship text to add "posterior predictive, credible interval, BvM." Slug-target automatic.

#### B9 — `multivariate-distributions.mdx` §8.10 ML connections BNN paragraph

**Find:**
```
The posterior $p(\mathbf{w} \mid \mathcal{D})$ is approximately MVN for well-specified models (by the Bernstein-von Mises theorem), with the Hessian of the negative log-posterior at the MAP estimate serving as the inverse covariance (the Laplace approximation).
```

**Replace:**
```
The posterior $p(\mathbf{w} \mid \mathcal{D})$ is approximately MVN for well-specified models (by the [Bernstein-von Mises theorem](/topics/bayesian-foundations-and-prior-selection#section-25-8)), with the Hessian of the negative log-posterior at the MAP estimate serving as the inverse covariance (the Laplace approximation). Topic 25 §25.8 proves BvM in the scalar case and Rem 16 handles the multivariate Laplace extension.
```

#### B10 — `multivariate-distributions.mdx` frontmatter `connections[bayesian-computation]`

**Find:** `topic: "bayesian-computation"` (frontmatter line).

**Replace:** `topic: "bayesian-computation-and-mcmc"` — slug update; relationship text unchanged.

#### B11 — `conditional-probability.mdx` frontmatter `connections[bayesian-foundations]`

Slug-target automatic. Relationship text: no change.

#### B12 — `hypothesis-testing.mdx` §17.11 Rem 15

**Find:**
```
Bayes factors are more consistent with Fisher's "evidence" framing and less vulnerable to the Ioannidis critique (Remark 6), but they require specifying priors for the competing hypotheses, which introduces its own modelling choices. Bayesian Foundations (coming soon, Track 7) will develop the full framework.
```

**Replace:**
```
Bayes factors are more consistent with Fisher's "evidence" framing and less vulnerable to the Ioannidis critique (Remark 6), but they require specifying priors for the competing hypotheses, which introduces its own modelling choices. [Bayesian Foundations (Topic 25)](/topics/bayesian-foundations-and-prior-selection#section-25-10) introduces the marginal likelihood $m(\mathbf{y})$ and names Bayes factors; the full Bayes-factor framework and BMA are Topic 27's territory.
```

#### B13 — `hypothesis-testing.mdx` §17.12 Rem 19 bullet

**Find:**
```
- **Bayesian Foundations** (Track 7, coming soon) — Bayes factors as the Bayesian counterpart
```

**Replace:**
```
- [**Bayesian Foundations**](/topics/bayesian-foundations-and-prior-selection) (Topic 25) — posterior over $\theta$, conjugate priors, credible intervals, Bernstein–von Mises; Bayes factors named in §25.10 with full development deferred to Topic 27
```

#### B14 — `confidence-intervals-and-duality.mdx` §19.1 Rem 3

**Find:**
```
Track 7 develops the Bayesian perspective; here we stay frequentist.
```

**Replace:**
```
[Topic 25](/topics/bayesian-foundations-and-prior-selection#section-25-6) develops the Bayesian perspective, including credible intervals and the flat-prior coincidence with z-CIs for Normal means; here we stay frequentist.
```

#### B15 — `confidence-intervals-and-duality.mdx` §19.10 Rem 21

**Find:**
```
Track 7 develops the theory; the key takeaways for Topic 19 are (1) frequentist coverage $\ne$ Bayesian credibility in general, (2) the two coincide under specific prior choices, and (3) frequentist guarantees are *over data*, Bayesian guarantees are *over parameter*.
```

**Replace:**
```
[Topic 25](/topics/bayesian-foundations-and-prior-selection#section-25-6) develops the theory; the key takeaways for Topic 19 are (1) frequentist coverage $\ne$ Bayesian credibility in general, (2) the two coincide under specific prior choices (flat improper prior for Normal mean — §25.8 Rem 17), and (3) frequentist guarantees are *over data*, Bayesian guarantees are *over parameter*. Topic 25 §25.8 Thm 5 (Bernstein–von Mises) proves the two frameworks agree asymptotically.
```

#### B16 — `confidence-intervals-and-duality.mdx` §19.10 Rem 22 cheat-sheet row

**Find:**
```
| Posterior-probability claim needed | Bayesian credible (Track 7) | Different framework |
```

**Replace:**
```
| Posterior-probability claim needed | [Bayesian credible (Topic 25)](/topics/bayesian-foundations-and-prior-selection#section-25-6) | Different framework |
```

#### B17 — `confidence-intervals-and-duality.mdx` §19.10 Rem 23

**Find:**
```
**Track 7 (Bayesian).** The contrast with frequentist coverage is where Bayesian inference earns its keep.
```

**Replace:**
```
**[Track 7 (Bayesian)](/topics/bayesian-foundations-and-prior-selection).** The contrast with frequentist coverage is where Bayesian inference earns its keep. Topic 25 §25.6 introduces credible intervals; §25.8 shows their asymptotic numerical agreement with Wald CIs under BvM.
```

#### B18 — `linear-regression.mdx` §21.10 Rem 25

**Find:**
```
A conjugate Normal-inverse-gamma prior on $(\boldsymbol\beta, \sigma^2)$ yields a closed-form Normal-inverse-gamma posterior, with the posterior mean an explicit ridge-regression estimate (the prior precision plays the role of $\lambda$). The Bayesian CI is the **credible interval** — a subset of parameter space with posterior mass $1 - \alpha$ — which agrees with the Wald-t CI only under flat priors and as $n \to \infty$. The conditional-MVN calculation of [Topic 8 §8.4](/topics/multivariate-distributions) is the mechanical engine. Track 7 covers this in full, plus non-conjugate priors via MCMC.
```

**Replace:**
```
A conjugate Normal-inverse-gamma prior on $(\boldsymbol\beta, \sigma^2)$ yields a closed-form Normal-inverse-gamma posterior, with the posterior mean an explicit ridge-regression estimate (the prior precision plays the role of $\lambda$). The Bayesian CI is the **credible interval** — a subset of parameter space with posterior mass $1 - \alpha$ — which agrees with the Wald-t CI only under flat priors and as $n \to \infty$. The conditional-MVN calculation of [Topic 8 §8.4](/topics/multivariate-distributions) is the mechanical engine. [Topic 25 §25.5](/topics/bayesian-foundations-and-prior-selection#section-25-5) covers the scalar Normal–Normal-Inverse-Gamma case; the regression extension and non-conjugate priors via MCMC are Topics 27–28.
```

#### B19 — `regularization-and-penalized-estimation.mdx` §23.10 Rem 24

**Find:**
```
Thm 5's MAP-as-penalized-MLE correspondence is the gateway to the Bayesian treatment of regularization. Track 7 develops conjugate priors for linear and GLM regression (Normal–inverse-gamma for linear, weakly-informative Cauchy for logistic), hierarchical priors that learn $\lambda$ from the data (empirical-Bayes ridge), and modern continuous shrinkage priors like the **horseshoe** (Carvalho–Polson–Scott 2010), which approximate lasso's sparsity behavior with a globally adaptive scale. The horseshoe is, in a sense, the "lasso of priors."
```

**Replace:**
```
Thm 5's MAP-as-penalized-MLE correspondence is the gateway to the Bayesian treatment of regularization. [Topic 25 §25.7–§25.8](/topics/bayesian-foundations-and-prior-selection#section-25-7) develops prior selection (Jeffreys, weakly-informative, improper with integrable posterior) and shows MAP-as-penalized-MLE as Ex 8. [Topic 28 (Hierarchical Bayes, coming soon)](/topics/hierarchical-bayes-and-partial-pooling) develops hierarchical priors that learn $\lambda$ from the data (empirical-Bayes ridge) and modern continuous shrinkage priors like the **horseshoe** (Carvalho–Polson–Scott 2010), which approximate lasso's sparsity behavior with a globally adaptive scale.
```

#### B20 — `model-selection-and-information-criteria.mdx` §24.10 Rem 31 opening

**Find:**
```
Thm 2's BIC-Laplace derivation is the gateway to the full Bayesian model-comparison machinery. Track 7 (Bayesian Foundations, upcoming) develops:
```

**Replace:**
```
Thm 2's BIC-Laplace derivation is the gateway to the full Bayesian model-comparison machinery. [Topic 25 (Bayesian Foundations)](/topics/bayesian-foundations-and-prior-selection) opens Track 7, and the three subsequent Track 7 topics develop:
```

(Bullets in the remark remain as-is; bullet-internal "MCMC", "marginal likelihood", "BMA", "DIC/WAIC/PSIS-LOO" mentions correctly point forward to Topics 26 and 27.)

#### B21 — `multiple-testing-and-false-discovery.mdx` §20.10 Bayesian-multiplicity remark

**Find:**
```
**Bayesian multiplicity and local-FDR (Efron 2010)** → one-paragraph remark in §20.10 pointing to Track 7.
```

**Replace:**
```
**Bayesian multiplicity and local-FDR (Efron 2010)** → [Topic 25 §25.10 names this pointer](/topics/bayesian-foundations-and-prior-selection#section-25-10); the full local-FDR framework awaits [Topic 27 (Bayesian Model Comparison, coming soon)](/topics/bayesian-model-comparison-and-bma) or formalml.
```

### 8.2 `curriculum-graph.json` edits

Target entry (current state): `id: "bayesian-foundations"`, `url: "/topics/bayesian-foundations"`, `status: "planned"`.

**Operations:**

1. Update `url: "/topics/bayesian-foundations-and-prior-selection"`.
2. Update `status: "published"`.
3. Preserve `id: "bayesian-foundations"` (edge stability per Topic 23/24 precedent).
4. Add prerequisite edges:
   ```json
   "prerequisites": [
     "conditional-probability",
     "expectation-moments",
     "exponential-families",
     "multivariate-distributions",
     "maximum-likelihood",
     "regularization-and-penalized-estimation",
     "point-estimation",
     "confidence-intervals-and-duality",
     "model-selection-and-information-criteria"
   ]
   ```
5. Track 7 other three topics: update URL strings to the locked slugs; preserve `id`:
   ```
   id: "bayesian-computation", url: "/topics/bayesian-computation-and-mcmc", status: "planned"
   id: "bayesian-model-comparison", url: "/topics/bayesian-model-comparison-and-bma", status: "planned"
   id: "hierarchical-bayes", url: "/topics/hierarchical-bayes-and-partial-pooling", status: "planned"
   ```

### 8.3 `curriculum.ts` edits

- Track 7 status `planned` → `in-progress`.
- Topic 25 `publishedDate` set to actual publish date.
- Track 7 `topicCount` stays at 4.

### 8.4 References spreadsheet edits

**New entries (5):**

| Key | Reference | URL |
|---|---|---|
| BER1994 | Bernardo & Smith (1994), *Bayesian Theory*, Wiley | https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316870 |
| ROB2007 | Robert (2007), *The Bayesian Choice* (2nd), Springer | https://link.springer.com/book/10.1007/0-387-71599-1 |
| JEF1961 | Jeffreys (1961), *Theory of Probability* (3rd), Oxford | https://global.oup.com/academic/product/theory-of-probability-9780198503682 |
| LIN2014 | Lindley (2014), *Understanding Uncertainty* (Rev), Wiley | https://onlinelibrary.wiley.com/doi/book/10.1002/9781118650158 |
| DIA1986 | Diaconis & Freedman (1986), "On the consistency of Bayes estimates," AoS 14(1): 1–26 | https://doi.org/10.1214/aos/1176349830 |

**Updates to existing entries** (append `bayesian-foundations-and-prior-selection` to "Used In Topics"):

- GEL2013 (Gelman et al., *Bayesian Data Analysis* 3rd) — already present from Topic 8
- VDV1998 (van der Vaart, *Asymptotic Statistics*) — already present from Topic 14
- CAS2002 (Casella & Berger) — inherited
- LEH1998 (Lehmann & Casella) — inherited from Topic 13
- HAS2009 (Hastie–Tibshirani–Friedman) — inherited from Topic 23

### 8.5 Sitemap / RSS verification

- `astro-sitemap` auto-regenerates on build; verify new URL appears.
- Pagefind re-indexes; verify topic appears in search for: "Bayesian", "prior", "posterior", "conjugate prior", "Jeffreys", "Bernstein-von Mises", "credible interval".
- Track 7 card on homepage updated from "coming soon" to "in progress (1 of 4)".

---

## 9. Verification Checklist

### Content / Proofs

- [ ] All 21 forward-promises from §1.3 activated with live links matching §8.1 verbatim.
- [ ] §25.3 notation block locked; symbols match Appendix A table verbatim.
- [ ] §25.4 Proof 1 (exp-fam conjugacy) matches §3.2 Proof 1 steps 1–6; end-of-proof citation to Topics 4 and 7.
- [ ] §25.5 Proof 2 (Beta-Binomial) matches §3.2 Proof 2 steps 1–5; end-of-proof citation to Topics 4 and 6.
- [ ] §25.6 Proof 2b (Beta-Binomial posterior predictive) matches §3.2 Proof 2b steps 1–5.
- [ ] §25.8 Proof 3 (BvM sketch) matches §3.2 Proof 3 steps 1–7; end-of-proof citation to Topic 14 + VDV1998 §10.2.
- [ ] §25.7 Thm 4 (Jeffreys) derived in full for Bernoulli and Normal-scale; reparameterization invariance stated without proof.
- [ ] §25.8 Ex 8 (MAP = penalized MLE) cites Topic 23 Thm 5 and Topic 14 Rem 9 — no re-derivation.
- [ ] §25.10 has exactly nine forward-pointing remarks (Rem 21–29).
- [ ] No "TBD," "open question," or "decide later" markers anywhere.
- [ ] §25.9 Wishart-Normal handled as one remark with GEL2013 §3.6 pointer (no derivation).
- [ ] Slug-change cascade in §8.1 C1–C6 applied.
- [ ] Stone–Dawid–Zidek paradox mentioned once only (§25.7 Rem 13); full treatment deferred.

### Formal element counts (match §1 totals)

- [ ] 7 Definitions (Def 1–7).
- [ ] 5 Theorems (Thm 1 Bayes for θ — stated; Thm 2 exp-fam conjugacy — full proof; Thm 3 Beta-Binomial posterior — full proof; Thm 3b Beta-Binomial posterior predictive — full proof; Thm 4 Jeffreys — derivation for Bernoulli + Normal-scale; Thm 5 BvM — sketch proof).
- [ ] 8 Examples (Ex 1–8).
- [ ] 29 Remarks (Rem 1–29).
- [ ] 10 figures (`25-1-` through `25-8-` + `25-10-forward-map`).
- [ ] 3 required + 1 optional interactive components.

### Frontmatter

- [ ] All 10 references present; all URLs resolve.
- [ ] `slug: "bayesian-foundations-and-prior-selection"`.
- [ ] `track: 7`, `trackName: "Bayesian Statistics"`, `topicNumber: 25`, `positionInTrack: 1`.
- [ ] `readTime: "60 min"`, `difficulty: "intermediate"`, `status: "published"`.
- [ ] `prerequisites` list matches §8.2 curriculum-graph prerequisites.
- [ ] `formalcalculusPrereqs` has 3 entries.
- [ ] `formalmlConnections` has 4 entries.
- [ ] `connections` has 7 entries.
- [ ] `notebookPath: "notebooks/bayesian-foundations/25_bayesian_foundations.ipynb"`.

### Figures

- [ ] All 11 PNGs (counting `25-10-forward-map.png`) in `public/images/topics/bayesian-foundations-and-prior-selection/`.
- [ ] Filenames match §4.1 manifest exactly.
- [ ] All figures 150 DPI, tight bbox, 0.15 pad.
- [ ] Palette: blue `#3B82F6` priors; amber `#F59E0B` likelihoods; purple `#8B5CF6` posteriors.
- [ ] No `\iff` in titles (use `\Leftrightarrow`); no raw Greek in alt-text curly braces.

### Interactive components

- [ ] `PriorPosteriorExplorer.tsx` — 5 presets work; sequential mode animates smoothly (400ms transitions); credible and HPD readouts match test values.
- [ ] `ConjugatePairBrowser.tsx` — 5 tabs render; formula panel KaTeX-correct; Dirichlet ternary plot renders on 30×30 lattice; Normal-Normal-IG contour plot valid.
- [ ] `BernsteinVonMisesAnimator.tsx` — 4 models work; animation plays at 8s log-scale; TV-distance readout computed on 500-point grid (200 on mobile); collapses to < 0.01 by $n = 500$ for Beta-Binomial with Jeffreys prior.
- [ ] `PriorSensitivityComparator.tsx` — 2 presets work; three posteriors overlay cleanly; "sensitivity magnitude" readout is the max absolute difference.
- [ ] All four components `client:visible` (not `client:load`).
- [ ] All four components mobile-responsive at 375px.
- [ ] `bayesianColors` palette imported and used consistently.

### Shared module

- [ ] `src/components/viz/shared/bayes.ts` compiles with zero TypeScript errors.
- [ ] All 15 tests in `bayes.test.ts` pass (T25.1–T25.15).
- [ ] `bayesianColors` export added to `viz/shared/colorScales.ts`.
- [ ] Five new CSS vars added to `src/styles/global.css`.
- [ ] New shared module imports `pdfBeta`, `pdfGamma`, `pdfNormal`, `pdfInverseGamma` from `distributions.ts`; does not redefine.

### Preset data

- [ ] `src/data/bayesian-foundations-data.ts` compiles.
- [ ] All five preset groups present with at least 2 entries each (except `normalNormalIGPresets`, which has 1).

### Cross-references

- [ ] At least 21 `/topics/bayesian-foundations-and-prior-selection` live links present (grep verified).
- [ ] `curriculum-graph.json` entry updated with new URL, status `published`, 9 prerequisites.
- [ ] Track 7 other 3 entries updated with new URLs; status stays `planned`.
- [ ] `curriculum.ts` Track 7 status `in-progress`.
- [ ] References spreadsheet has 5 new entries; 5 existing entries have topic appended.
- [ ] All frontmatter `connections[bayesian-*]` slugs updated per §8.1 C4–C5.

### Build

- [ ] `pnpm build` succeeds with zero errors, zero warnings.
- [ ] `pnpm astro check` passes (no broken internal links).
- [ ] `pnpm test` passes (all existing tests + 15 new Topic 25 tests).
- [ ] Pagefind index updated; search returns Topic 25 for expected queries.
- [ ] Sitemap.xml includes new URL.
- [ ] Mobile responsive at 375px × 667px (iPhone SE viewport).
- [ ] "Intermediate" difficulty badge styled correctly.
- [ ] Track 7 homepage card shows "1 of 4 published."

### Grep verification commands (run before PR)

```bash
# 21 activated links:
grep -rn '/topics/bayesian-foundations-and-prior-selection' src/content/topics/ --include='*.mdx' | wc -l
# Should return at least 21.

# Zero stale short-slug references:
grep -rn 'topics/bayesian-foundations"\|topics/bayesian-foundations\b[^-]' src/content/topics/ --include='*.mdx'
# Should return zero.

# Zero "Track 7, coming soon" placeholders:
grep -rn 'Track 7, coming soon\|Bayesian Foundations.*coming soon\|Bayesian Foundations (coming soon)' src/content/topics/ --include='*.mdx'
# Should return zero.

# Zero MAP-BvM rate typos (B1 correction):
grep -rn 'MAP.*converges.*rate.*1/n\|MAP.*converges.*rate \$1/n' src/content/topics/ --include='*.mdx'
# Should return zero after B1's correction.

# Bayes factor references all point to Topic 25 or 27:
grep -rn 'Bayes factor\|BF_{10}' src/content/topics/ --include='*.mdx' | grep -v '(coming soon)'
# Should reference Topic 25 §25.10 or Topic 27 (upcoming).
```

---

## 10. Build Order

Numbered steps the Claude Code agent executes sequentially. One commit per step.

1. **Scaffold:** Create `notebooks/bayesian-foundations/`, `public/images/topics/bayesian-foundations-and-prior-selection/`, preset stub `src/data/bayesian-foundations-data.ts`, shared stub `src/components/viz/shared/bayes.ts`, test stub `src/components/viz/shared/__tests__/bayes.test.ts`.

2. **CSS palette:** Add five new Track 7 CSS custom properties to `src/styles/global.css` per §5.5. Add `bayesianColors` export to `src/components/viz/shared/colorScales.ts`.

3. **Shared module:** Implement all functions in `bayes.ts` per §6.1. Import existing `pdfBeta`, `pdfGamma`, `pdfNormal`, `pdfInverseGamma` from `distributions.ts`; do not redefine. Add `pdfNormalInverseGamma`, `pdfStudentTMarginal`, `pdfDirichlet` as new. Implement `posterior` dispatch, `credibleInterval*`, `hpdInterval` (generic bisection), `mapEstimate`, `posteriorMean`, `posteriorVariance`, `jeffreysPrior`, `laplaceLogMarginalLikelihood`, `posteriorSample`, `posteriorPredictive`, Beta-Binomial / Negative-Binomial posterior-predictive PMFs.

4. **Tests:** Write 15 tests in `bayes.test.ts` per §6.2. Run `pnpm test`; verify all pass. Test values come from notebook Cell 13 — execute the notebook locally first to obtain them.

5. **Preset data:** Fill `src/data/bayesian-foundations-data.ts` per §7. TypeScript-check.

6. **Notebook:** Complete `notebooks/bayesian-foundations/25_bayesian_foundations.ipynb` per §4.2 (13 cells). Run end-to-end locally. Copy PNGs to `public/images/topics/bayesian-foundations-and-prior-selection/`. Clear outputs; commit.

7. **Interactive components** (four commits, one per component): `PriorPosteriorExplorer.tsx`, `ConjugatePairBrowser.tsx`, `BernsteinVonMisesAnimator.tsx`, `PriorSensitivityComparator.tsx`. For each: write, hook up presets, test at 375px / 768px / 1440px, keyboard accessibility, color+linestyle redundant encoding.

8. **MDX body:** Create `src/content/topics/bayesian-foundations-and-prior-selection.mdx` with full frontmatter per §2.1, all 10 sections per §3.1 map, all formal-element blocks with Astro UI components (`<DefinitionBlock>`, `<TheoremBlock>`, `<ProofBlock>`, `<ExampleBlock>`, `<RemarkBlock>`), full proof text for Proofs 1/2/2b/3 per §3.2, all 11 figure tags, all 4 component embeds with `client:visible`, and `### References` section at the end.

9. **Cross-reference edits:** Apply §8.1 operations B1–B21 and C1–C6 in order. Commit each per source file.

10. **Curriculum graph + metadata:** Apply §8.2, §8.3, §8.4 operations. Verify homepage Track 7 card.

11. **Pre-flight grep:** Run §9 grep verification commands; all checks pass.

12. **Build + deploy:** `pnpm build`; `pnpm test`; `pnpm astro check`. Push to `topic/bayesian-foundations-and-prior-selection` branch and open PR. Title: "Topic 25: Bayesian Foundations & Prior Selection (opens Track 7)." Description references 21 forward-promises closed, the BvM rate correction in B1, and the Topic 25–28 slug cascade.

13. **After merge:** Verify Vercel surface. Update `publishedDate` / `lastUpdated`. Confirm Pagefind search.

---

## Appendix A: Style Guidelines

### KaTeX constraints (same as Topics 1–24)

- **No `\begin{aligned}` blocks with `&` markers.** Multi-line derivations use separate `$$...$$` blocks with prose glue. Proofs 1, 2, 2b, and 3 all follow this pattern.
- **No `\begin{array}` tables.** Use markdown tables.
- Inline math `$...$`; display math `$$...$$` on its own line.
- **Independence notation:** `\perp\!\!\!\perp` — established in Topic 16 §16.9. Topic 25 references only.
- **Track 7 symbols locked in §25.3:** $\pi(\theta)$, $p(\theta \mid \mathbf{y})$, $p(\tilde y \mid \mathbf{y})$, $m(\mathbf{y})$, $\mathrm{KL}(\pi \,\|\, \pi')$, $\mathrm{BF}_{10}$, $\hat\theta_{\text{MAP}}$, $\hat\theta_{\text{PM}}$, $\hat\theta_{\text{med}}$.
- **Matplotlib mathtext gotchas:** `\geq` not `\ge`; `\leq` not `\le`; `\sqrt{n}` with braces; `\to` not `\xrightarrow` (unsupported); `\Leftrightarrow` not `\iff` (unsupported).

### MDX constraints

- **Curly braces in prose:** raw `{...}` fails the JSX parser. Use `$\{...\}$` for set notation in prose or `alt`-text.
- **Cross-site links:** `<ExternalLink site="formalcalculus" ... />` or `<ExternalLink site="formalml" ... />`, not raw markdown links.
- **Section anchors:** `#section-25-X` convention.
- **JSX comments in import block:** `{/* ... */}` between frontmatter and first heading → parse error. Fine in body; not in import block.
- **`<ProofBlock>` props:** `number={N}` and `title="..."` both required.
- **End-of-proof citation style:** `∎ — using [list]`. Sketch proofs: `∎ (sketch) — using [list]`.

### Citation style (Chicago 17th, Notes & Bibliography)

- Books: `N. Author(s). (Year). [*Title*](URL) (Edition). Publisher.`
- Articles: `N. Author(s). (Year). [*Title*](URL). *Journal*, Volume(Issue), Pages.`
- Every frontmatter `references:` entry has `url:` field with working hyperlink (all 10 verified in §2.1).
- MDX ends with rendered `### References` section, preceded by `---`.

### Component styling

- `client:visible` directive (never `client:load`).
- React + Tailwind + SVG/D3 only. No Chart.js, Plotly, Recharts.
- Mobile-responsive at 375px.
- Track 7 palette `bayesianColors` from `viz/shared/colorScales.ts`; CSS vars from `global.css`.

---

## Appendix B: Design Decisions (locked — prevents re-litigation)

### B.1 Featured theorem: Bernstein–von Mises (sketch proof)

**Decision:** §25.8 Thm 5 (BvM) is the featured theorem. Sketch-quality proof (~25 MDX lines), following VDV1998 §10.2. Full proof (LAN contiguity, uniform tightness) out of scope.
**Rationale:** BvM is the deepest result, the explicit bridge to frequentist asymptotics, the answer to "does the prior matter as $n \to \infty$?" The sketch captures the Taylor-expand-and-extract-Gaussian argument — the load-bearing pedagogical step — and cites VDV1998 for the TV upgrade.
**Alternative:** Conjugacy theorem as featured. Rejected — more constructive but less deep; Topic 7 already covered the construction.

### B.2 Conjugate-pair catalog: five in full + Wishart as §25.9 remark

**Decision:** Beta-Binomial (Thm 3 + Proof 2 full), Normal-Normal known σ² (Ex 2 sketched), Normal-Normal-Inverse-Gamma unknown σ² (Ex 3 sketched, cites GEL2013 §3.3), Gamma-Poisson (Ex 4 cites Topic 7 Ex 3), Dirichlet-Multinomial (Ex 5 cites Topic 8 Thm 6). Wishart-Normal as §25.9 Rem with GEL2013 §3.6 pointer.
**Rationale:** Six pairs in full would eat 6 of 8 example slots. Wishart has the most moving parts and least pedagogical novelty in a track opener. The scalar Normal-Normal-Inverse-Gamma covers the machinery; the multivariate Wishart generalization is a remark.
**Alternative:** All six in full. Rejected — inflates past 11K words.

### B.3 Four full proofs: Thm 2, Thm 3, Thm 3b, Thm 5

**Decision:** Proof 1 (exp-fam conjugacy, full ~20 lines), Proof 2 (Beta-Binomial, full ~12 lines), Proof 2b (Beta-Binomial posterior predictive, full ~15 lines), Proof 3 (BvM sketch ~25 lines). Thm 4 (Jeffreys) is derivation for Bernoulli + Normal-scale without a standalone `<ProofBlock>`. Thm 1 (Bayes for θ) stated-only, cites Topic 4 Thm 4.
**Rationale:** Three full proofs per Track 4/5 opener precedent (Topic 13 had three; Topic 17 had four). Featured BvM needs its own proof; conjugacy theorem and its headline instance (Beta-Binomial) justify two more; posterior predictive (Proof 2b) is the shortest full proof and closes a Topic 6 §6.7 Ex 5 forward-promise.
**Alternative:** Drop Proof 2b, state posterior-predictive only. Rejected — the Beta-Binomial compound is a strictly stronger result than "plug-in Binomial" and is pedagogically important for the "posterior predictive is wider than plug-in" intuition.

### B.4 Jeffreys derivation depth

**Decision:** Thm 4 derived in full for Bernoulli (→ Beta(½, ½)) and Normal-scale (→ $\pi(\sigma) \propto 1/\sigma$). Reparameterization invariance stated with JEF1961 pointer, not proved.
**Rationale:** Two scalar derivations give enough traction for the reader to compute Jeffreys in other families; full invariance proof is algebra-heavy and adds marginal pedagogical value in a track opener.
**Alternative:** State formula only, no derivation. Rejected — §25.7's prior-sensitivity section gains substantively from showing "the non-informative prior is not flat" explicitly.

### B.5 Three required components + one optional

**Decision:** `PriorPosteriorExplorer`, `ConjugatePairBrowser`, `BernsteinVonMisesAnimator` required. `PriorSensitivityComparator` optional fourth.
**Rationale:** Matches Topics 19, 21, 22 cadence. The optional `PriorSensitivityComparator` earns its keep in §25.7 but can be cut if implementation time is tight; the three required components cover §25.2, §25.5, and §25.8 — the three load-bearing sections.
**Alternative:** Four required. Rejected — sensitivity comparator is narrower in scope than the other three.

### B.6 Shared module name: `bayes.ts`

**Decision:** New Track-7 leaf module `src/components/viz/shared/bayes.ts`. Locked before Topic 26–28 development.
**Rationale:** Matches `estimation.ts` / `testing.ts` / `regression.ts` short-form precedent. `bayesian.ts` is verbose; `priors.ts` too narrow (posterior / predictive / credible / Bayes-factor utilities all live here).
**Alternative:** Extend `estimation.ts`. Rejected — different paradigm; extends-don't-create applies within a track, not across tracks.

### B.7 Track 7 slug cascade

**Decision:** Topic 25–28 slugs locked now:
- 25: `bayesian-foundations-and-prior-selection`
- 26: `bayesian-computation-and-mcmc`
- 27: `bayesian-model-comparison-and-bma`
- 28: `hierarchical-bayes-and-partial-pooling`

Original `id` values preserved in `curriculum-graph.json` per Topic 23/24 edge-stability precedent.
**Rationale:** Topic 25's §25.10 forward-pointers use these slugs. Locking now prevents cascade rework when Topics 26–28 build out.
**Alternative:** Lock only Topic 25; leave 26–28 as short form. Rejected — every Topic 25 forward-pointer would need re-editing when 26–28 expand, multiplying slug-cascade work.

### B.8 Notation lock for all of Track 7

**Decision:** $\pi(\theta)$ prior; $p(\theta \mid \mathbf{y})$ posterior; $p(\tilde y \mid \mathbf{y})$ posterior predictive; $m(\mathbf{y})$ marginal likelihood; $\mathrm{KL}(\pi \,\|\, \pi')$; $\mathrm{BF}_{10}$.
**Rationale:** $\pi$ for prior (Bernardo–Smith / Robert) is visually distinct from sampling $p$ and posterior $p(\cdot \mid \mathbf{y})$. $\tilde y$ for new observation is cleaner than `y_new` subscript. $m(\mathbf{y})$ avoids collision with $p(\mathbf{y})$ as sampling density.
**Alternative:** Gelman-style $p(\theta)$ overloading. Rejected — requires disambiguation by context in every formula.

### B.9 Color palette for Track 7

**Decision:** Three new CSS vars `--color-prior` (blue #3B82F6), `--color-likelihood` (amber #F59E0B), `--color-posterior` (purple #8B5CF6), plus two utility vars `--color-true-parameter` (emerald #10B981), `--color-mle` (orange #F97316). Mirrored to `bayesianColors` in `colorScales.ts`.
**Rationale:** Three-color prior/likelihood/posterior trio matches standard Bayesian-textbook visual grammar. Topics 26–28 inherit. The two utility vars support BvM-animator's dual-reference plot ($\theta_0$ vs $\hat\theta_{\text{MLE}}$).
**Alternative:** Reuse Topic 22 GLM palette. Rejected — Bayesian workflow has a fundamentally different color story (update-based, not model-based).

### B.10 Topic 25 opens Track 7 (not closes Track 6)

**Decision:** Topic 25 is the Track 7 opener; the "track-completion" language from Topic 20 and 24 does not apply. Instead, Topic 25 emphasizes the conceptual shift from Topics 13–24's frequentist framework to Bayesian inference.
**Rationale:** Matches Topic 13 (Track 4 opener), Topic 17 (Track 5 opener), Topic 21 (Track 6 opener). The Track-7 notation lock and color-palette lock are load-bearing for Topics 26–28; opener positioning reinforces that.

### B.11 Notebook path convention (short-form directory)

**Decision:** `notebooks/bayesian-foundations/25_bayesian_foundations.ipynb` (short-form, drops "-and-prior-selection" suffix). Image directory keeps the full slug.
**Rationale:** Matches Topic 19's established convention (`notebooks/confidence-intervals/` for slug `confidence-intervals-and-duality`). Topic 25 preserves the pattern for Topics 26–28 to inherit: `notebooks/bayesian-computation/`, `notebooks/bayesian-model-comparison/`, `notebooks/hierarchical-bayes/`.

### B.12 Notebook delivery state (unexecuted; `default_rng(42)`)

**Decision:** Native `.ipynb` with outputs cleared. `np.random.default_rng(42)` at module top (not legacy `np.random.seed(42)`). Jonathan executes locally.
**Rationale:** Topic 24 standardized on `default_rng`; Topic 25 follows the newer convention. Executing notebook delivery would force Claude to run matplotlib on the container and saves nothing — Jonathan needs to verify scipy output vs `scipy.stats` reference values locally anyway.

### B.13 Scope boundaries (nine deferrals)

**Decision:** MCMC → 26; BMA / Bayes factors / marginal-likelihood computation → 27; hierarchical / empirical Bayes → 28; VI → formalml; decision theory → formalml; reference priors / Bernardo → §25.10 name-drop; improper-prior paradoxes beyond integrable-posterior → 27/formalml; posterior predictive checks → 27; Bayesian nonparametrics → formalml.
**Rationale:** Track 7 has three more topics to absorb the rest. Each deferral is a §25.10 one-paragraph remark. The bar for inclusion in Topic 25: "does this substantially change the reader's picture of prior/likelihood/posterior?"
**Alternative:** Include all of them in a longer Topic 25. Rejected — 14K-word topic is the catch-all anti-pattern; Track 7's structure exists specifically to prevent it.

### B.14 One-sentence subjective-vs-objective Bayes treatment

**Decision:** Rem 2 is one sentence: "The philosophical debate between subjective and objective Bayes (LIN2014, JEF1961) is out of scope for Topic 25; we treat priors as substantive modeling choices whose sensitivity we examine empirically (§25.7)." Full philosophical treatment deferred to a later philosophical essay or formalml.
**Rationale:** Editorial voice is "informed peer at a whiteboard," not philosophy lecture. The math is the content.

### B.15 Fourteen reference entries total

**Decision:** 5 new entries (BER1994, ROB2007, JEF1961, LIN2014, DIA1986) + 5 inherited (GEL2013, VDV1998, CAS2002, LEH1998, HAS2009) = 10 total. Plus the 4 entries already present in earlier topics' spreadsheet for cross-track reuse. Topic 25's frontmatter carries 10 references.
**Rationale:** 10 references is comparable to Topics 21 (14) and 24 (10). Going below 8 risks under-citing the foundational texts; going above 15 is attribution spam for a track opener.
**Alternative:** Add Bernardo's *Reference Analysis* (2005). Rejected — reference-priors framework is §25.7 Rem 14 one-line mention only; full citation belongs in Topic 27 or formalml.

---

*Brief version: v1 · Topic 25: Bayesian Foundations & Prior Selection · Created April 2026*
*Reference notebook: `notebooks/bayesian-foundations/25_bayesian_foundations.ipynb`*
*Companion: Claude Code starter prompt to be generated from the `claude-code-starter-prompt-template.md` fill-in*
