# Claude Code Handoff Brief: Confidence Intervals & Duality

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)
**Repo:** `github.com/jonx0037/formalStatistics`
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel
**Package Manager:** pnpm
**Status:** Ready for implementation
**Reference Notebook:** `notebooks/confidence-intervals-and-duality/19_confidence_intervals.ipynb`

---

## Important: Nineteenth Topic — Third in Track 5

This is **topic 19 of 32** and the **third topic in Track 5 (Hypothesis Testing & Confidence)**. Track 5 opened with Topic 17 (Hypothesis Testing Framework), which built the scaffolding; Topic 18 (Likelihood-Ratio Tests & Neyman-Pearson) delivered the optimality theory and named its forward debts. Topic 19 is the collect: **every family of level-$\alpha$ tests is a $(1-\alpha)$ confidence procedure, and every confidence procedure is a family of tests.** The featured theorem is the duality itself (§19.2 Thm 1). The pedagogical arc runs from the frequentist-semantics clarification of §19.1 through test-inversion constructions (pivotal, Wald, Score, LRT, Wilson, Clopper–Pearson, profile likelihood) to coverage diagnostics and the one-sided / equivalence extensions. Topic 19 also fulfills four explicit forward-promises that Topics 17–18 made — three from Topic 17 §17.10 and §17.12, and three from Topic 18 §18.5 Remark 10, §18.8 Remark 16, and §18.10 Remark 22.

**Implications:**

1. **All eighteen preceding topics are published.** Cross-references from `sample-spaces` through `likelihood-ratio-tests-and-np` use live internal links.

2. **Seven explicit forward-promises must be fulfilled.** Four from `hypothesis-testing.mdx` (§17.10 Rem 12 / Ex 17 / Rem 13; §17.12 Rem 19) and three from `likelihood-ratio-tests-and-np.mdx` (§18.5 Rem 10; §18.8 Rem 16; §18.10 Rem 22). See §1 below for the complete list with verbatim text and replacement locations.

3. **Topic 19 *extends* — does not create — the shared Track 5 module `testing.ts`.** Topic 17 seeded `testing.ts`; Topic 18 extended it with the Neyman-Pearson LR evaluator, UMP boundary finder, non-central χ², Wilks simulator, and local-power computer. Topic 19 adds the four CI constructors (Wald, Score, LRT inversions for scalar θ), the Wilson and Clopper–Pearson closed/exact forms for binomial p, the profile-likelihood CI engine, the TOST procedure, and a coverage simulator. See §6 for the full manifest.

4. **Do not re-derive Wilks' theorem.** Topic 18 Thm 4 (§18.6) is the asymptotic engine that powers Proof 4 (profile-likelihood CI coverage). §19.7 cites Wilks; it does not re-prove it.

5. **Do not re-derive the score test's null distribution.** Topic 18 §18.7 Thm 5 establishes that $S_n \xrightarrow{d} \chi^2_1$ under $H_0$ with regularity. §19.5 uses this directly to get the Wilson interval's asymptotic coverage — the proof is the *inversion* of that null distribution, not a re-derivation of it.

6. **Do not re-derive MLE asymptotic normality.** Topic 14 Thm 14.3 powers the Wald CI (§19.4). §19.4 cites it; it does not re-prove it.

7. **Scalar θ only, with a controlled vector-θ preview in §19.7.** Every proof is for scalar θ ∈ ℝ. The profile-likelihood section in §19.7 previews the vector case (θ ∈ ℝ, ψ ∈ ℝ^{k-1} nuisance) with a worked Normal-mean-unknown-variance example and one figure, but does not venture into simultaneous CIs for vector-valued functionals — those belong to Topic 20.

8. **Every topic after Topic 19 inherits the CI scaffolding.** Topic 20 (Multiple Testing) uses the test-CI duality directly — simultaneous CIs are just test-family-inversions with FWER/FDR control. Track 6 (Regression) builds CIs for GLM coefficients on top of the Wald / LRT machinery established here. Track 7 (Bayesian) contrasts coverage with posterior credibility; the §19.1 clarification ("coverage ≠ posterior probability") is the explicit bridge.

### 1. Forward-Promise Fulfillment

### 1.1 Scope boundary

**What Topic 19 covers (the full list):**

- The test-CI duality theorem with full proof (§19.2 Thm 1 + Proof 1).
- Pivotal-quantity constructions (§19.3): z, t, χ², F — all as direct inversions of Topic 17 §17.6–§17.8 tests.
- Wald / Score / LRT confidence intervals (§19.4) — inversions of Topic 18 §18.7's asymptotic trio.
- The Wilson interval with full proof (§19.5 Thm 2 + Proof 2) — score-test inversion for binomial $p$.
- The Clopper–Pearson exact interval with full proof (§19.6 Thm 3 + Proof 3) — via the beta–binomial duality.
- The profile-likelihood confidence interval with asymptotic-coverage proof (§19.7 Thm 4 + Proof 4).
- Coverage diagnostics (§19.8): actual vs nominal, anti-conservative warnings, small-$n$ divergence.
- One-sided confidence intervals and the TOST equivalence-testing framework (§19.9).
- Ten PNG figures, four interactive components (three required + one optional), ~9,500 words.

**What Topic 19 does NOT cover:**

- **Bootstrap confidence intervals** → one-paragraph remark in §19.10 pointing to Track 8. Percentile, BCa, and studentized bootstrap intervals are the natural next extension but require a fully nonparametric treatment.
- **Bayesian credible intervals** → one-paragraph remark in §19.10 pointing to Track 7. The frequentist-vs-Bayesian distinction is set up in §19.1; the full Bayesian theory (posterior credibility, HPD intervals, Lindley's paradox) is Track 7's territory.
- **Simultaneous / joint confidence regions** → one-paragraph remark in §19.10 pointing to Topic 20. Bonferroni, Scheffé, Tukey, and Working–Hotelling bands require multiple-testing adjustments that belong to Topic 20.
- **Fieller's theorem / ratio-of-parameters CIs** → not mentioned. Niche; adds substantial complexity without pedagogical payoff at this difficulty level.
- **Vector-θ simultaneous CIs, Hotelling's T², confidence ellipsoids** → stated only in the §19.7 vector preview; cited to Lehmann–Romano 2005 Ch. 8 and nothing more.
- **Sequential confidence intervals, always-valid inference, confidence sequences** → one-line pointer in §19.10 to the modern extensions (Howard/Ramdas/McAuliffe 2021) and to A/B-testing platforms; no treatment.

### 1.2 Forward-promise fulfillment (7 items from Topics 17–18)

These are the live "Topic 19 will..." pointers currently in the published MDX. Each must be fulfilled by Topic 19's content, and — where the MDX reads "Topic 19" as plain text or points to the placeholder slug `/topics/confidence-intervals` — Claude Code must replace with a live internal link to `/topics/confidence-intervals-and-duality` after Topic 19 ships. See §8 for the exact MDX updates.

| # | Source location | Verbatim text | Topic 19 deliverable |
|---|---|---|---|
| 1 | `hypothesis-testing.mdx` §17.10 Rem 12 | "Inverting a z-test gives the z-interval; inverting a t-test gives the t-interval; inverting an LRT gives the profile-likelihood interval; inverting a score test gives the score interval. Topic 19 develops all four constructions." | §19.2 Thm 1 + Proof 1 (the duality in full); §19.3 (z/t as pivotal inversions); §19.4 (Wald/Score/LRT inversions); §19.7 (profile-likelihood CI). |
| 2 | `hypothesis-testing.mdx` §17.10 Ex 17 | "The inversion also works for the Wald, score, and LRT tests in §17.9, giving three different (1 − α) intervals for a general parameter. Topic 19 develops the full set." | §19.4 — three CIs (Wald, Score, LRT) with explicit Bernoulli, Normal, and Poisson specializations. |
| 3 | `hypothesis-testing.mdx` §17.10 Rem 13 | "The Wilson interval, based on the score test, is preferred. Full theory, with coverage diagnostics and comparisons, is Confidence Intervals (coming soon, Topic 19)." | §19.5 Thm 2 + Proof 2 (Wilson derivation); §19.8 (coverage diagnostics: Wald vs Wilson vs Clopper–Pearson). |
| 4 | `hypothesis-testing.mdx` §17.12 Rem 19 | "Confidence Intervals (coming soon, Topic 19) — the duality previewed in §17.10 becomes the construction. Pivotal quantities, Wald / score / LRT intervals, coverage diagnostics, the Wilson interval for binomial proportions (which fixes the Wald boundary problem from Remark 13)." | §19.2 (duality); §19.3 (pivotal); §19.4 (Wald/Score/LRT); §19.5 (Wilson as Wald-boundary fix); §19.8 (coverage). |
| 5 | `likelihood-ratio-tests-and-np.mdx` §18.5 Rem 10 | "Profile likelihood is a CI-construction tool first (Topic 19) and a testing artifact second." | §19.7 Thm 4 + Proof 4 (profile-likelihood CI with asymptotic-coverage proof citing §18.6). |
| 6 | `likelihood-ratio-tests-and-np.mdx` §18.8 Rem 16 | "For A/B tests with rare-event outcomes (conversion rates of 0.1% or smaller), the Wald test's boundary fragility motivates the Wilson interval in Topic 19." | §19.5 Thm 2 + Proof 2 (Wilson closed form); §19.8 Ex 7 (MC at $n = 20$, $p = 0.05$: Wald under-covers, Wilson nominal). |
| 7 | `likelihood-ratio-tests-and-np.mdx` §18.10 Rem 22 | "Every family of level-α tests indexed by the null value θ₀ defines a (1−α)-confidence set. This test-CI duality is the organizing principle of Topic 19." | §19.2 Thm 1 (duality theorem, featured) + Proof 1; §19.4 (all three asymptotic CIs as test inversions); §19.7 (profile-likelihood CI as LRT inversion). |

---

## 2. Frontmatter

### 2.1 Full frontmatter spec

The MDX file begins with the following YAML. Every field is required; every `url:` must resolve.

```yaml
---
title: "Confidence Intervals & Duality"
slug: "confidence-intervals-and-duality"
track: 5
trackName: "Hypothesis Testing & Confidence"
topicNumber: 19
positionInTrack: 3
readTime: "55 min"
difficulty: "intermediate"
status: "published"
publishedDate: "TO BE SET AT PUBLICATION"
lastUpdated: "TO BE SET AT PUBLICATION"
description: "Readers who have completed Topic 17's hypothesis-testing framework and Topic 18's optimality theory, and who are ready for the dual perspective: every family of level-α tests indexed by the null value θ₀ is a (1−α) confidence procedure, and every confidence procedure is a family of tests. The exposition constructs the z, t, χ², F, Wald, Score, LRT, Wilson, Clopper–Pearson, and profile-likelihood intervals as a single pattern — test inversion — then calibrates them against actual coverage and contrasts them with the Bayesian posterior-probability semantics that look superficially similar but mean something different. Scope is scalar θ throughout; a brief vector-θ preview appears in the profile-likelihood section."
prerequisites:
  - topic: "hypothesis-testing"
    relationship: "Topic 17 built the test framework and previewed the CI duality in §17.10. Topic 19 delivers the dual construction in full: the z, t, and χ² tests of §17.6–§17.8 invert into the corresponding CIs of §19.3."
  - topic: "likelihood-ratio-tests-and-np"
    relationship: "Topic 18 gave the Wald / Score / LRT trio with asymptotic χ²₁ null distribution (§18.7 Thm 5). Topic 19 §19.4 inverts each of the three into a confidence interval; §19.5 derives the Wilson interval as the score-test inversion for binomial p; §19.7's profile-likelihood CI coverage proof cites Wilks' theorem (§18.6 Thm 4)."
  - topic: "maximum-likelihood"
    relationship: "Topic 14 Thm 14.3 (MLE asymptotic normality) is the engine of the Wald CI (§19.4): θ̂ ± z_{α/2} / √(n I(θ̂)) is the direct consequence. Topic 14 §14.6 (first-order condition) powers the profile-likelihood construction in §19.7."
  - topic: "point-estimation"
    relationship: "Topic 13 Thm 13.9 (Cramér-Rao lower bound) reappears in §19.4 Rem 4: the CRLB on estimator variance is the asymptotic lower bound on Wald CI width. Standard-error computation from Topic 13 §13.7 is the concrete ingredient."
  - topic: "sufficient-statistics"
    relationship: "Topic 16's Basu independence (Thm 7) underlies the exact t-CI in §19.3 — the same independence that made the t-test exact in Topic 17 §17.7 makes the t-CI exact here. The dualization is one line."
  - topic: "continuous-distributions"
    relationship: "The beta, t, χ², and F quantile functions from Topic 6 are the computational engines of §19.3 (pivotal CIs) and §19.6 (Clopper–Pearson via inverse-beta)."
  - topic: "discrete-distributions"
    relationship: "The binomial CDF from Topic 5 and its relationship to the incomplete beta function are the basis of the Clopper–Pearson construction in §19.6."
formalcalculusPrereqs:
  - topic: "integration"
    site: "formalcalculus"
    relationship: "Coverage probability P_θ(θ ∈ C(X)) is an integral of the sampling density over the event {θ ∈ C(X)}. The coverage diagnostic in §19.8 computes this integral numerically for binomial p."
  - topic: "sequences-limits"
    site: "formalcalculus"
    relationship: "Asymptotic coverage — the statement lim_{n→∞} P_θ(θ ∈ C_n(X)) = 1 − α — is a sequences-limits statement. The Wald-under-coverage-in-small-samples result in §19.5 is its finite-sample companion."
  - topic: "differentiation"
    site: "formalcalculus"
    relationship: "The profile likelihood L_P(θ) = sup_ψ L(θ, ψ) is a composition of an implicit function (ψ̂(θ) solves the nuisance score equation) and the joint likelihood. Differentiation under the sup is the Danskin envelope theorem — cited in §19.7 Rem 15."
formalmlConnections:
  - topic: "ab-testing-platforms"
    site: "formalml"
    relationship: "Every deployed A/B-test platform reports a confidence interval on the treatment effect alongside the p-value. The Wilson interval is the industry default for rare-event conversion metrics (§19.5 Ex 5); bootstrap and always-valid confidence sequences are the two major extensions used in production."
  - topic: "generalized-linear-models"
    site: "formalml"
    relationship: "GLM coefficient inference uses the Wald / LRT CI trio of §19.4 at the scale of every deployed logistic, Poisson, and gamma regression. The LRT-based deviance-difference CI is the numerical default in R's confint() on glm objects; Wald is the default in statsmodels."
  - topic: "statistical-learning-theory"
    site: "formalml"
    relationship: "PAC-Bayes generalization bounds are uniform (1−δ) confidence statements on the hypothesis class. The coverage-calibration theme of §19.8 generalizes to PAC-bound tightness diagnostics in the learning-theory literature."
  - topic: "causal-inference"
    site: "formalml"
    relationship: "Instrumental-variables regression, doubly-robust estimation, and difference-in-differences all report confidence intervals on the average treatment effect. The Wald CI is most common; sandwich variance estimators produce sandwich-Wald intervals. TOST (§19.9) is the equivalence-testing framework for noninferiority trials in biostatistics."
references:
  - type: "paper"
    title: "Probable Inference, the Law of Succession, and Statistical Inference"
    author: "Edwin B. Wilson"
    year: 1927
    journal: "Journal of the American Statistical Association"
    volume: "22(158)"
    pages: "209–212"
    url: "https://doi.org/10.1080/01621459.1927.10502953"
    note: "Cited as WIL1927. §19.5 Thm 2 attribution — the score-test-inverted binomial CI. The modern textbook 'Wilson interval' is this paper's construction."
  - type: "paper"
    title: "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial"
    author: "Charles J. Clopper & Egon S. Pearson"
    year: 1934
    journal: "Biometrika"
    volume: "26(4)"
    pages: "404–413"
    url: "https://doi.org/10.1093/biomet/26.4.404"
    note: "Cited as CLO1934. §19.6 Thm 3 attribution — the exact beta-quantile binomial CI via beta–binomial duality. Conservative but exact."
  - type: "paper"
    title: "Interval Estimation for a Binomial Proportion"
    author: "Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta"
    year: 2001
    journal: "Statistical Science"
    volume: "16(2)"
    pages: "101–133"
    url: "https://doi.org/10.1214/ss/1009213286"
    note: "Cited as BRO2001. §19.5 Rem 8 and §19.8 — the paper that quantitatively established Wald's coverage pathology and promoted Wilson / Agresti–Coull / Jeffreys as practical alternatives. Primary source for the actual-vs-nominal coverage curves."
  - type: "paper"
    title: "Approximate Is Better than 'Exact' for Interval Estimation of Binomial Proportions"
    author: "Alan Agresti & Brent A. Coull"
    year: 1998
    journal: "The American Statistician"
    volume: "52(2)"
    pages: "119–126"
    url: "https://doi.org/10.1080/00031305.1998.10480550"
    note: "Cited as AGR1998. §19.5 Rem 9 — the 'add 2 successes, add 2 failures' practical approximation to Wilson. Pedagogical anchor for 'approximate beats exact' framing."
  - type: "paper"
    title: "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability"
    author: "Donald J. Schuirmann"
    year: 1987
    journal: "Journal of Pharmacokinetics and Biopharmaceutics"
    volume: "15(6)"
    pages: "657–680"
    url: "https://doi.org/10.1007/BF01068419"
    note: "Cited as SCH1987. §19.9 Thm 5 attribution — the Two One-Sided Tests (TOST) procedure for equivalence testing. The FDA bioequivalence 80/125 rule is its canonical application."
  - id: "LEH2005"
    text: "Lehmann, Erich L., and Joseph P. Romano. 2005. Testing Statistical Hypotheses. 3rd ed. Springer Texts in Statistics. New York: Springer."
    url: "https://doi.org/10.1007/0-387-27605-X"
    usedIn: "Already cited in Topics 17–18. In Topic 19: §19.2 Rem 2 (duality in vector θ); §19.4 Rem 5 (three asymptotic CIs' coverage); §19.7 Rem 14 (profile likelihood Ch. 12.4); §19.10 Rem 19 (scope-boundary pointers)."
  - id: "CAS2002"
    text: "Casella, George, and Roger L. Berger. 2002. Statistical Inference. 2nd ed. Pacific Grove, CA: Duxbury."
    url: "https://www.cengage.com/c/statistical-inference-2e-casella"
    usedIn: "Already cited in Topics 17–18. In Topic 19: §19.3 Ex 1 (t-CI) and §19.6 Ex 4 (Clopper–Pearson) worked-example alignment."
  - id: "WIL1938"
    text: "Wilks, Samuel S. 1938. \"The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses.\" Annals of Mathematical Statistics 9 (1): 60–62."
    url: "https://doi.org/10.1214/aoms/1177732360"
    usedIn: "Already cited in Topic 18. In Topic 19: §19.4 (LRT CI asymptotic coverage); §19.7 Thm 4 + Proof 4 (profile-likelihood CI coverage cites Wilks directly)."
  - id: "NEY1937"
    text: "Neyman, Jerzy. 1937. \"Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability.\" Philosophical Transactions of the Royal Society A 236 (767): 333–380."
    url: "https://doi.org/10.1098/rsta.1937.0005"
    usedIn: "§19.1 Rem 1 + §19.2 Rem 1 — the original formulation of the confidence-interval concept as distinct from fiducial / Bayesian posterior probability. Primary historical source for the 'coverage ≠ posterior probability' clarification."
---
```

### 2.2 Length target

Target: **55-minute read** — 10 sections, 4 full proofs, three required interactive components (plus one optional), ~9,500 words. Topic 19 is slightly shorter than Topic 18 (60 min / 10K words) because four of its five main constructions (z-CI, t-CI, Wald-CI, LRT-CI) are one-step inversions of tests already derived in Topics 17–18 — the original machinery is concentrated in §19.5 (Wilson), §19.6 (Clopper–Pearson), and §19.7 (profile likelihood). The §19.1 frequentist-semantics clarification and §19.8 coverage diagnostics each get roughly the weight of a derivation section, keeping total content density consistent with Track 5 precedent.

---

## 3. Content Structure

### 3.1 Section Map

| § | Title | Formal elements | Figure | Interactive component |
|---|-------|----------------|--------|---------------------|
| 19.1 | "What a Confidence Interval Is (and Is Not)" | Definitions 1–2 (confidence procedure; coverage), Remarks 1–3 (Neyman 1937 semantics; the "95% probability" trap; frequentist vs Bayesian contrast) | `19-ci-motivation.png` | — |
| 19.2 | "The Test–CI Duality Theorem" | Theorem 1 (test–CI duality — **featured**), **Proof 1**, Examples 1–2 (z-test inversion → z-CI; t-test inversion → t-CI), Remarks 4–5 (duality as organizing principle; vector-θ extension cited to LEH2005) | `19-duality-diagram.png` | `CITestDualityVisualizer` |
| 19.3 | "Pivotal Quantities" | Definition 3 (pivot), Examples 3–5 (z-pivot for Normal mean; t-pivot with Basu; χ²-pivot for Normal variance; F-pivot for variance ratio), Remark 6 (pivots are rare; inversion of asymptotic tests is the general tool) | `19-pivotal-construction.png` | — |
| 19.4 | "Wald, Score, LRT Confidence Intervals" | Definitions 4–6 (Wald CI; Score CI; LRT CI as test-inversions), Theorem 2 (asymptotic coverage of all three — **stated**, cites §18.7 Thm 5 and §18.6 Thm 4), Examples 6–7 (Bernoulli all three worked; Poisson rate all three worked), Remarks 7–8 (CRLB as CI-width envelope; small-$n$ divergence) | `19-wald-score-lrt-intervals.png`, `19-wald-boundary-pathology.png` | — |
| 19.5 | "The Wilson Interval" | Theorem 3 (Wilson interval via score-test inversion), **Proof 2**, Example 8 (Bernoulli $\hat p = 0$ — Wald gives a point, Wilson gives $[0, 3.8/n]$), Remarks 9–10 (BRO2001 coverage quantification; AGR1998 plus-4 approximation) | — | — |
| 19.6 | "Clopper–Pearson Exact Intervals" | Theorem 4 (Clopper–Pearson via beta–binomial), **Proof 3**, Example 9 ($n = 20$, $x = 3$: CP $[0.032, 0.379]$ vs Wald $[0.014, 0.286]$), Remarks 11–12 (conservatism: inverting a discrete test over-covers; CP as FDA default for binomial confidence) | `19-clopper-pearson-construction.png` | `BinomialCoverageComparison` |
| 19.7 | "Profile Likelihood Confidence Intervals" | Definition 7 (profile likelihood), Theorem 5 (profile-likelihood CI asymptotic coverage), **Proof 4**, Examples 10–11 (Normal mean with unknown variance — profile recovers t-CI; Gamma shape with unknown rate), Remarks 13–15 (Wilks-based coverage; vector-θ via Ch-12.4 LEH2005; envelope-theorem differentiation) | `19-profile-likelihood.png` | `ProfileLikelihoodExplorer` |
| 19.8 | "Coverage Diagnostics: Actual vs Nominal" | Definition 8 (actual coverage), Theorem 6 (Wald under-covers asymptotically at boundary — stated), Example 12 (MC at $n = 20, 100, 500$ across $p \in [0.01, 0.99]$), Remarks 16–17 (anti-conservative vs conservative CIs; coverage as an identification tool for misspecification) | `19-binomial-coverage-curves.png` | — |
| 19.9 | "One-Sided CIs and TOST Equivalence Testing" | Definition 9 (one-sided CI), Theorem 7 (TOST equivalence procedure — stated with Schuirmann attribution), Examples 13–14 (one-sided upper bound on a toxicity rate; FDA bioequivalence 80/125 log-scale), Remark 18 (noninferiority vs equivalence) | `19-tost-preview.png` | `ReparamCIInvariance` (optional) |
| 19.10 | "Limitations & Forward Look" | Remarks 19–23 (scope boundary; bootstrap CIs → Track 8; Bayesian credible intervals → Track 7; simultaneous CIs → Topic 20; cheat sheet; Track 6 + formalml.com pointers) | `19-forward-map.png` | — |

**Total: 9 Definitions + 7 Theorems + 4 Full Proofs + 14 Examples + 23 Remarks = 57 formal elements.**

### 3.2 Proof specifications

Four full proofs — matching Topic 18's "expand fully or omit" convention. Each is written without `\begin{aligned}` blocks; multi-line derivations use separate `$$…$$` blocks with prose glue. End-of-proof citation style is `∎ — using [...]`, uniformly applied.

#### Proof 1 — Theorem 1 (Test–CI Duality), §19.2

**Length target:** ~10 MDX lines. Indicator-function argument.

*Statement.* Fix $\alpha \in (0, 1)$ and a parametric family $\{P_\theta : \theta \in \Theta\}$. Suppose that for every $\theta_0 \in \Theta$ we have a level-$\alpha$ test $\phi_{\theta_0}(X) \in \{0, 1\}$ of $H_0: \theta = \theta_0$ (reject iff $\phi_{\theta_0}(X) = 1$), so that $P_{\theta_0}(\phi_{\theta_0}(X) = 1) \le \alpha$. Define

$$C(X) = \{\theta_0 \in \Theta : \phi_{\theta_0}(X) = 0\}.$$

Then $C(X)$ is a $(1 - \alpha)$ confidence set for $\theta$, meaning $P_\theta(\theta \in C(X)) \ge 1 - \alpha$ for every $\theta \in \Theta$. Conversely, given a $(1-\alpha)$ confidence set $C(X)$, the collection $\{\phi_{\theta_0}(X) = \mathbf{1}\{\theta_0 \notin C(X)\} : \theta_0 \in \Theta\}$ is a family of level-$\alpha$ tests.

*Proof.*

**Step 1 — Forward direction (tests → CI).** Fix $\theta \in \Theta$. The event $\{\theta \in C(X)\}$ is by construction the event $\{\phi_\theta(X) = 0\}$, which is the complement of the rejection event for the level-$\alpha$ test of $H_0: \theta = \theta$ (the null value coinciding with the true value).

$$P_\theta(\theta \in C(X)) = P_\theta(\phi_\theta(X) = 0) = 1 - P_\theta(\phi_\theta(X) = 1) \ge 1 - \alpha.$$

The inequality is the size constraint of the test at the true parameter, applied to $\theta_0 = \theta$. Since $\theta$ was arbitrary, the coverage $\ge 1 - \alpha$ holds uniformly.

**Step 2 — Converse direction (CI → tests).** Fix $\theta_0 \in \Theta$ and define $\phi_{\theta_0}(X) = \mathbf{1}\{\theta_0 \notin C(X)\}$. Under $H_0: \theta = \theta_0$, the size is

$$P_{\theta_0}(\phi_{\theta_0}(X) = 1) = P_{\theta_0}(\theta_0 \notin C(X)) = 1 - P_{\theta_0}(\theta_0 \in C(X)) \le 1 - (1 - \alpha) = \alpha,$$

where the inequality is the $(1 - \alpha)$ coverage of $C$ at $\theta = \theta_0$. Hence the family $\{\phi_{\theta_0}\}$ consists of level-$\alpha$ tests.

∎ — using the size constraint in each direction; see NEY1937 for the original formulation.

#### Proof 2 — Theorem 3 (Wilson Interval), §19.5

**Length target:** ~15 MDX lines. Closed-form quadratic inversion.

*Statement.* Let $X_1, \ldots, X_n$ be iid Bernoulli$(p)$ with $\hat p_n = \bar X_n$. The asymptotic level-$\alpha$ score test of $H_0: p = p_0$ rejects iff $|Z_n(p_0)| > z_{\alpha/2}$, where $Z_n(p_0) = \sqrt n\,(\hat p_n - p_0)/\sqrt{p_0(1-p_0)}$. The test-inversion CI is the **Wilson interval**

$$C_{\text{Wilson}}(X) = \frac{\hat p_n + \frac{z^2}{2n} \pm z\,\sqrt{\frac{\hat p_n(1-\hat p_n)}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n},$$

where $z = z_{\alpha/2}$.

*Proof.*

**Step 1 — Set up the inversion.** By the duality theorem (Proof 1), $p_0 \in C(X)$ iff the score test fails to reject at $p_0$:

$$(\hat p_n - p_0)^2 \le z^2\,\frac{p_0(1 - p_0)}{n}.$$

**Step 2 — Rearrange as a quadratic in $p_0$.** Expanding both sides and collecting terms in $p_0$:

$$\hat p_n^2 - 2 \hat p_n p_0 + p_0^2 \le \frac{z^2 p_0}{n} - \frac{z^2 p_0^2}{n}.$$

Grouping into the quadratic inequality $A p_0^2 + B p_0 + C \le 0$ with

$$A = 1 + \frac{z^2}{n}, \qquad B = -\!\left(2\hat p_n + \frac{z^2}{n}\right), \qquad C = \hat p_n^2.$$

**Step 3 — Solve.** The coefficient $A > 0$, so the inequality $A p_0^2 + B p_0 + C \le 0$ defines the interval $[p_-, p_+]$ between the roots of the corresponding quadratic equation. By the quadratic formula,

$$p_\pm = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A} = \frac{2\hat p_n + z^2/n \pm \sqrt{(2\hat p_n + z^2/n)^2 - 4(1 + z^2/n)\hat p_n^2}}{2(1 + z^2/n)}.$$

**Step 4 — Simplify the discriminant.** Expanding $(2\hat p_n + z^2/n)^2 - 4(1 + z^2/n)\hat p_n^2$ and cancelling the $4\hat p_n^2$ terms gives $\frac{4 z^2 \hat p_n}{n} + \frac{z^4}{n^2} - \frac{4 z^2 \hat p_n^2}{n} = \frac{4 z^2 \hat p_n(1 - \hat p_n)}{n} + \frac{z^4}{n^2}$. Factoring out $4z^2/n^2$ from under the square root:

$$\sqrt{\cdot} = \frac{2z}{n}\sqrt{n \hat p_n(1 - \hat p_n) + \tfrac{z^2}{4}} = 2z\sqrt{\tfrac{\hat p_n(1 - \hat p_n)}{n} + \tfrac{z^2}{4n^2}}.$$

**Step 5 — Assemble.** Substituting back and dividing numerator and denominator by $2$:

$$p_\pm = \frac{\hat p_n + z^2/(2n) \pm z\sqrt{\hat p_n(1-\hat p_n)/n + z^2/(4n^2)}}{1 + z^2/n}.$$

This is the stated Wilson interval.

∎ — using the duality theorem (Thm 1), the score test's null distribution (Topic 18 §18.7 Thm 5), and elementary quadratic-formula algebra. See WIL1927 for the original derivation.

#### Proof 3 — Theorem 4 (Clopper–Pearson Interval), §19.6

**Length target:** ~20 MDX lines. Beta–binomial duality via incomplete-beta regularization.

*Statement.* Let $X \sim \text{Binomial}(n, p)$ with observed value $x \in \{0, 1, \ldots, n\}$. The **Clopper–Pearson** $(1 - \alpha)$ confidence interval for $p$ is

$$C_{\text{CP}}(x) = [p_L, p_U], \qquad p_L = B^{-1}(\alpha/2;\, x, n - x + 1), \qquad p_U = B^{-1}(1 - \alpha/2;\, x + 1, n - x),$$

where $B^{-1}(q; a, b)$ is the $q$-quantile of the Beta$(a, b)$ distribution (with the conventions $p_L = 0$ when $x = 0$ and $p_U = 1$ when $x = n$). Coverage $\ge 1 - \alpha$ for every $p \in [0, 1]$.

*Proof.*

**Step 1 — Exact two-sided test inversion.** The exact two-sided test of $H_0: p = p_0$ at level $\alpha$ (Topic 17 §17.6 Ex 11) fails to reject at $p_0$ iff both

$$P_{p_0}(X \ge x) \ge \alpha/2 \qquad \text{(lower tail bound)}, \qquad P_{p_0}(X \le x) \ge \alpha/2 \qquad \text{(upper tail bound)}.$$

By the duality theorem (Proof 1), the non-rejection set is the CI $[p_L, p_U]$: $p_L$ is the largest $p_0$ satisfying $P_{p_0}(X \ge x) = \alpha/2$ (equality by continuity of the binomial CDF in $p$), and $p_U$ is the smallest $p_0$ satisfying $P_{p_0}(X \le x) = \alpha/2$.

**Step 2 — Beta–binomial identity.** The key identity — provable by repeated integration by parts on the regularized incomplete beta function — is

$$P_p(X \le k) = \sum_{j=0}^k \binom{n}{j} p^j(1-p)^{n-j} = I_{1-p}(n - k, k + 1),$$

where $I_x(a, b) = B(x; a, b) / B(a, b)$ is the regularized incomplete beta function (i.e., the Beta$(a, b)$ CDF evaluated at $x$). Equivalently, $P_p(X \ge k) = I_p(k, n - k + 1)$.

**Step 3 — Solve for $p_L$ via beta quantile.** Setting $P_{p_L}(X \ge x) = \alpha/2$ and applying the identity with $k = x$:

$$I_{p_L}(x, n - x + 1) = \alpha/2 \qquad \Longleftrightarrow \qquad p_L = I^{-1}(\alpha/2; x, n - x + 1) = B^{-1}(\alpha/2; x, n - x + 1).$$

The second equality identifies the inverse regularized incomplete beta with the inverse CDF of the Beta$(x, n - x + 1)$ distribution.

**Step 4 — Solve for $p_U$.** Symmetrically, setting $P_{p_U}(X \le x) = \alpha/2$ and using $P_p(X \le x) = 1 - I_p(x + 1, n - x)$:

$$1 - I_{p_U}(x + 1, n - x) = \alpha/2 \qquad \Longleftrightarrow \qquad I_{p_U}(x + 1, n - x) = 1 - \alpha/2,$$

hence $p_U = B^{-1}(1 - \alpha/2; x + 1, n - x)$.

**Step 5 — Coverage bound.** Because the binomial is discrete, the exact tail probabilities $P_p(X \ge x)$ and $P_p(X \le x)$ are step functions of $p$ with jumps at the $n + 1$ possible values of $X$. Enforcing $\alpha/2$-size at equality in Step 1 means the test's actual size is $\le \alpha$ (over-controlled at most $p_0$); by Step 2 of Proof 1, the CI's coverage is $\ge 1 - \alpha$. Equality is attained only at boundary points where the discrete CDF achieves $\alpha/2$ exactly — hence the CI is *exact* but generically conservative, with actual coverage strictly exceeding nominal at most $p$.

∎ — using the duality theorem (Thm 1), the beta–binomial identity (classical; see CLO1934 for the original presentation in this context), and discreteness of the binomial CDF.

#### Proof 4 — Theorem 5 (Profile-Likelihood CI Coverage), §19.7

**Length target:** ~10 MDX lines. Wilks-driven; cites §18.6.

*Statement.* Let $\{P_{\theta, \psi} : \theta \in \Theta \subset \mathbb{R}, \psi \in \Psi \subset \mathbb{R}^{k-1}\}$ be a regular parametric family with scalar parameter of interest $\theta$ and nuisance $\psi$. Define the profile log-likelihood $\ell_P(\theta) = \sup_\psi \ell(\theta, \psi)$, the profile MLE $\hat\theta_n = \arg\max_\theta \ell_P(\theta)$, and the **profile-likelihood confidence interval**

$$C_P(X) = \{\theta : 2\,[\ell_P(\hat\theta_n) - \ell_P(\theta)] \le \chi^2_{1, 1 - \alpha}\}.$$

Under Wilks' regularity conditions, $P_{(\theta_0, \psi_0)}(\theta_0 \in C_P(X)) \to 1 - \alpha$ as $n \to \infty$.

*Proof.*

**Step 1 — Profile LRT statistic.** Fix the true parameter $(\theta_0, \psi_0)$. The test-inversion construction of $C_P(X)$ is precisely the inversion of the generalized LRT (Topic 18 §18.5 Def 6) for the composite null $H_0: \theta = \theta_0$ with $\psi$ nuisance:

$$-2 \log \Lambda_n(\theta_0) = 2\,\big[\ell(\hat\theta_n, \hat\psi_n) - \sup_\psi \ell(\theta_0, \psi)\big] = 2\,[\ell_P(\hat\theta_n) - \ell_P(\theta_0)].$$

The equality uses the definition of the profile: $\sup_\psi \ell(\theta_0, \psi) = \ell_P(\theta_0)$ and $\ell(\hat\theta_n, \hat\psi_n) = \ell_P(\hat\theta_n)$ by construction.

**Step 2 — Wilks' asymptotic null distribution.** By Wilks' theorem ([Topic 18 §18.6 Thm 4](/topics/likelihood-ratio-tests-and-np#section-18-6)), under the regular-parametric assumptions and with scalar $\theta$ restricted under $H_0$,

$$-2\log\Lambda_n(\theta_0) \xrightarrow{d} \chi^2_1 \qquad \text{under } (\theta_0, \psi_0).$$

**Step 3 — Coverage by inversion.** The event $\{\theta_0 \in C_P(X)\}$ is the event $\{-2\log\Lambda_n(\theta_0) \le \chi^2_{1, 1-\alpha}\}$ — the non-rejection event for the LRT at $\theta_0$. By Step 2 and the continuous mapping theorem,

$$P_{(\theta_0, \psi_0)}(\theta_0 \in C_P(X)) = P_{(\theta_0, \psi_0)}(-2\log\Lambda_n(\theta_0) \le \chi^2_{1,1-\alpha}) \to P(\chi^2_1 \le \chi^2_{1, 1-\alpha}) = 1 - \alpha.$$

∎ — using the duality theorem (Thm 1), Wilks' theorem (Topic 18 §18.6), and continuous mapping (Topic 9).

### 3.3 Notation conventions (extending Topics 17–18)

- **Confidence set / interval:** $C(X)$ or $C_n(X)$ when the sample-size dependence is salient.
- **Coverage:** $P_\theta(\theta \in C(X))$. Lower case "coverage" = nominal $1 - \alpha$; phrase "actual coverage" = the population value $P_\theta(\theta \in C(X))$.
- **Quantiles:** $z_{\alpha/2}$ = upper-$\alpha/2$ standard-normal quantile; $t_{n-1, \alpha/2}$ = upper-$\alpha/2$ t-quantile with $n-1$ df; $\chi^2_{k, 1-\alpha}$ = $(1-\alpha)$-quantile of $\chi^2_k$; $F_{k_1, k_2, \alpha/2}$ = upper-$\alpha/2$ F-quantile. Matches Topic 17.
- **Pivot:** $Q(X, \theta)$ — a function of data and parameter whose distribution does not depend on $\theta$.
- **Profile log-likelihood:** $\ell_P(\theta) = \sup_\psi \ell(\theta, \psi)$. Never abbreviated.
- **Wilson-interval constants:** $z = z_{\alpha/2}$; $\hat p_n = \bar X_n$; used locally to §19.5.
- **Clopper–Pearson endpoints:** $p_L, p_U$; Beta quantile $B^{-1}(q; a, b)$.
- **One-sided CIs:** $C_L(X) = (-\infty, U(X)]$ and $C_U(X) = [L(X), \infty)$ (upper-bound and lower-bound procedures, respectively).
- **TOST:** equivalence margin $\delta > 0$; the procedure rejects the non-equivalence null $H_0: |\theta - \theta_0| \ge \delta$ iff both one-sided tests of $\theta \le \theta_0 - \delta$ and $\theta \ge \theta_0 + \delta$ reject at level $\alpha$.

---

## 4. Static Figures & Notebook Structure

### 4.1 Figure manifest (10 PNGs)

| # | File | Section | Dimensions | Description |
|---|------|---------|-----------|-------------|
| 1 | `19-ci-motivation.png` | §19.1 | 1400 × 500 | 2-panel. Left: twenty 95% CIs computed from twenty independent samples of $n = 30$ from $\mathcal{N}(\mu, 1)$ with $\mu = 0$; approximately one of the twenty fails to cover $\mu$. The caption makes the frequentist semantics visible — coverage is about the procedure's long-run behavior, not the posterior probability of any one interval. Right: conceptual split — frequentist coverage vs Bayesian posterior credibility rendered as two distinct diagrams, sharing the same observed interval but with different probabilistic objects shaded. |
| 2 | `19-duality-diagram.png` | §19.2 | 1400 × 600 | **Featured figure.** The two-plane diagram. Left plane: $(\theta_0, x)$ axes with shaded non-rejection region $\{(\theta_0, x) : \phi_{\theta_0}(x) = 0\}$. Horizontal slicing at fixed $x$ yields the CI $C(x)$; vertical slicing at fixed $\theta_0$ yields the acceptance region of the test of $H_0: \theta = \theta_0$. Annotations explicitly label both slicings. |
| 3 | `19-pivotal-construction.png` | §19.3 | 1400 × 400 | 3-panel. z-pivot: standard-Normal density with shaded ±$z_{\alpha/2}$ tails; below, the algebraic inversion $\bar X \pm z_{\alpha/2}\sigma/\sqrt n$. t-pivot: $t_{n-1}$ density for $n = 10$, same inversion yielding $\bar X \pm t_{n-1, \alpha/2} S/\sqrt n$. χ²-pivot: $\chi^2_{n-1}$ density, quantile pair, CI for $\sigma^2$. |
| 4 | `19-wald-score-lrt-intervals.png` | §19.4 | 1400 × 500 | 2-panel. Left: log-likelihood $\ell(p)$ for Bernoulli with observed $\hat p = 0.3$, $n = 50$. The Wald CI is constructed from the quadratic approximation (gold curve); the LRT CI is the set where $\ell(p) \ge \ell(\hat p) - \chi^2_{1, 0.95}/2$ (green shading). The Score CI is the set where $|U(p)/\sqrt{nI(p)}| \le z_{0.025}$. All three CIs shown as horizontal bars at the bottom with numerical endpoints. Right: same for Poisson rate $\lambda$ with observed $\hat\lambda = 2$, $n = 30$. |
| 5 | `19-wald-boundary-pathology.png` | §19.4 / §19.5 | 1400 × 500 | 2-panel. Left: actual coverage of the Wald CI for binomial $p$ across $p \in [0.01, 0.99]$, $n = 30$ — the sawtooth under-coverage at moderate $p$, dropping to $\approx 0$ as $p \to 0$ or $p \to 1$. Right: at $\hat p = 0$, the Wald CI is $[0, 0]$ (a point); Wilson gives a proper interval $[0, 3.8/n]$; Clopper–Pearson gives $[0, 1 - (\alpha/2)^{1/n}]$. Three CI bars overlaid with endpoints annotated. |
| 6 | `19-binomial-coverage-curves.png` | §19.8 | 1400 × 600 | **Coverage diagnostic figure.** 3-panel grid, rows for $n \in \{20, 100, 500\}$, each row showing actual coverage of Wald, Wilson, Clopper–Pearson across $p \in [0.005, 0.995]$. Horizontal line at 0.95. Wald is the anti-conservative (actual < nominal at moderate $p$, catastrophic at boundary); Wilson oscillates around 0.95; Clopper–Pearson is always $\ge 0.95$ but often substantially higher. |
| 7 | `19-clopper-pearson-construction.png` | §19.6 | 1400 × 500 | 2-panel. Left: the discrete binomial CDF $P_{p}(X \le x)$ as a step function of $p$ for fixed $x = 3$, $n = 20$; horizontal line at $\alpha/2 = 0.025$; the crossing point is $p_U$. Similarly for $P_p(X \ge x)$ and $p_L$. Right: the beta-distribution CDF that achieves the same crossing — visualizes the beta–binomial identity directly. |
| 8 | `19-profile-likelihood.png` | §19.7 | 1400 × 500 | 2-panel. Left: joint log-likelihood surface $\ell(\mu, \sigma)$ for Normal sample of $n = 20$; the profile $\ell_P(\mu) = \sup_\sigma \ell(\mu, \sigma)$ traced as a red curve along the $\mu$-axis; the $-\chi^2_{1, 0.95}/2$ threshold band shaded, delineating the profile CI. Right: profile likelihood for Gamma shape $\alpha$ with unknown rate $\beta$; same band-shading; the resulting CI endpoints numerically annotated. |
| 9 | `19-tost-preview.png` | §19.9 | 1400 × 400 | 2-panel. Left: two one-sided tests at level $\alpha$ — the rejection regions for $H_0^L: \theta \le \theta_0 - \delta$ (left tail) and $H_0^U: \theta \ge \theta_0 + \delta$ (right tail), both needed to reject the equivalence null. Right: the 90% CI $[L, U]$ vs the equivalence margin $[\theta_0 - \delta, \theta_0 + \delta]$; equivalence established iff $[L, U] \subset [\theta_0 - \delta, \theta_0 + \delta]$. FDA bioequivalence 80/125 annotation on log-scale. |
| 10 | `19-forward-map.png` | §19.10 | 1400 × 500 | Summary schematic. Central box: "Topic 19: test–CI duality." Arrows outward to Track 6 (regression CIs), Track 7 (Bayesian credible intervals — contrast), Track 8 (bootstrap CIs — extension), Topic 20 (simultaneous CIs). Each arrow is labeled with the pedagogical relationship. |

### 4.2 Notebook structure

The notebook `19_confidence_intervals.ipynb` has 13 code cells plus a trailing markdown cell, matching Topic 18's structure:

1. **Imports + seed.** `numpy`, `scipy.stats` (binom, beta, chi2, norm, t, f, gamma), `matplotlib.pyplot`. `np.random.seed(42)`.
2. **Palette + matplotlib defaults.** Reuse the Topic 18 palette verbatim: amber `#D97706` (Wald), purple `#7C3AED` (Score / Wilson), green `#10B981` (LRT / profile), grey `#475569` (reference). Matplotlib rcParams: title size 14, axis label size 12, tick size 10, figure DPI 150, `bbox_inches='tight'` on save.
3. **Figure 1** (`19-ci-motivation.png`) — ~50 lines. Twenty simulated 95% z-CIs plus a conceptual frequentist-vs-Bayesian panel.
4. **Figure 2** (`19-duality-diagram.png`) — ~80 lines. **Featured figure.** Two-plane duality diagram with shaded acceptance region and slicing annotations.
5. **Figure 3** (`19-pivotal-construction.png`) — ~60 lines. z/t/χ² pivotal-inversion panels.
6. **Figure 4** (`19-wald-score-lrt-intervals.png`) — ~80 lines. Log-likelihood with three CIs overlaid for Bernoulli and Poisson.
7. **Figure 5** (`19-wald-boundary-pathology.png`) — ~70 lines. Wald coverage sawtooth + boundary-case three-CI comparison.
8. **Figure 6** (`19-binomial-coverage-curves.png`) — ~100 lines. 3-row coverage grid, Wald vs Wilson vs Clopper–Pearson across $n \in \{20, 100, 500\}$.
9. **Figure 7** (`19-clopper-pearson-construction.png`) — ~60 lines. Binomial CDF step function + beta CDF equivalence.
10. **Figure 8** (`19-profile-likelihood.png`) — ~80 lines. Joint log-likelihood surface + profile curve + band-shaded CI.
11. **Figure 9** (`19-tost-preview.png`) — ~50 lines. TOST two-sided rejection regions + CI-contained-in-margin.
12. **Figure 10** (`19-forward-map.png`) — ~50 lines. Summary schematic with outbound arrows.
13. **Coverage-simulation helper** — ~40 lines. Seeded MC coverage runner for binomial p: takes CI function + true $p$ + $n$ + $M$, returns actual coverage. Used to generate Figure 6's data. Verification: Wald actual coverage at $(p, n) = (0.1, 50)$ matches BRO2001 Table 1.
14. **Markdown cell** — "All figures saved to `public/images/topics/confidence-intervals-and-duality/`. Run the cells above to regenerate."

All figures saved with `plt.savefig(figname, dpi=150, bbox_inches='tight', pad_inches=0.15)`.

---

## 5. Interactive Components

Four components total: three required, one optional. All follow the Topic 17–18 patterns: React + Tailwind, `client:visible` directive (**not** `client:load`), SVG/D3.js only, CSS custom properties from `viz/shared/colorScales.ts`, responsive to 375px width, client-side computation only.

### Component 1: CITestDualityVisualizer

**File:** `src/components/viz/CITestDualityVisualizer.tsx`

**Purpose:** The featured component. Make the test–CI duality visually concrete. The reader sees the two-plane diagram with the shaded non-rejection region; dragging a vertical line changes $\theta_0$ and reveals the acceptance region of the test of $H_0: \theta = \theta_0$; dragging a horizontal line changes the observed data $x$ and reveals the CI $C(x)$. The orthogonal slicing is the core pedagogical moment of §19.2.

**Interactions:**
- Dropdown: scenario (Normal mean with $\sigma$ known — the cleanest z-test case; Bernoulli proportion — discrete; Poisson rate). Default: Normal, $\sigma = 1$.
- Slider: sample size $n$ (5 to 200).
- Slider: $\alpha$ (0.01 to 0.20).
- Draggable horizontal line: observed statistic value $t(x)$ (e.g., $\bar x$ for Normal). As dragged, the shaded CI $C(x)$ updates live in the right panel.
- Draggable vertical line: null value $\theta_0$. As dragged, the test's acceptance region at $\theta_0$ updates in the top panel.
- Left panel: the two-plane diagram — $(\theta_0, t(x))$ axes, shaded non-rejection region, both draggable lines overlaid with their induced slicings.
- Right panel: the induced CI $C(x) = [L(x), U(x)]$ shown as a horizontal bar along the $\theta_0$-axis, with numerical endpoints.
- Top panel: the induced acceptance region $A(\theta_0) = [L_T(\theta_0), U_T(\theta_0)]$ shown as a horizontal bar along the $t(x)$-axis.
- Bottom readout: "At $t(x) = \cdots$, CI is $[\cdots, \cdots]$. At $\theta_0 = \cdots$, acceptance region is $[\cdots, \cdots]$."

**Data:** `dualityPresets` from `confidence-intervals-data.ts`.

**Uses from `testing.ts`:** `standardNormalCDF`, `standardNormalInvCDF`, `binomialCDF`, `poissonCDF`.

**Mobile behavior:** left panel full-width on top, right panel below, top panel collapses to inline annotation on the left panel.

### Component 2: BinomialCoverageComparison

**File:** `src/components/viz/BinomialCoverageComparison.tsx`

**Purpose:** Make the coverage-diagnostic argument of §19.8 directly interactive. The reader chooses $n$ and a CI procedure (Wald, Wilson, Clopper–Pearson, or "all three") and watches the actual-coverage curve traced across $p \in [0.005, 0.995]$, computed exactly (not Monte Carlo) from the binomial CDF.

**Interactions:**
- Slider: $n$ (5 to 1000, log-scale).
- Radio: which CI procedure to display (Wald, Wilson, Clopper–Pearson, All three).
- Slider: $\alpha$ (0.01 to 0.20); default 0.05.
- Left panel: actual-coverage curves across $p$ on the x-axis, coverage on the y-axis. Horizontal dashed line at nominal $1 - \alpha$. The curves are colored: Wald (amber), Wilson (purple), Clopper–Pearson (green).
- Right panel: at the currently-hovered $p$, a bar-readout of each CI's endpoints for a single MC draw (new draw button). Shows the CI in interval form — visually anchoring "what does coverage miss look like?"
- Bottom readout: "At $p = \cdots, n = \cdots$: Wald actual coverage = $\cdots$; Wilson = $\cdots$; Clopper–Pearson = $\cdots$. Wald deviation from nominal = $\cdots$ (under-covering / over-covering)."

**Data:** `coveragePresets` from `confidence-intervals-data.ts`.

**Uses from `testing.ts`:** `waldCIBinomial`, `wilsonInterval`, `clopperPearsonInterval`, `binomialCDF`, `actualCoverageBinomial` (all new in §6).

**Mobile behavior:** coverage curves on top (full-width); per-draw CI bars below.

### Component 3: ProfileLikelihoodExplorer

**File:** `src/components/viz/ProfileLikelihoodExplorer.tsx`

**Purpose:** Make the profile-likelihood construction tangible. The reader sees the joint log-likelihood surface $\ell(\theta, \psi)$ as a contour plot; the profile $\ell_P(\theta) = \sup_\psi \ell(\theta, \psi)$ is drawn along the $\theta$-axis as the ridge line of the contour; the $\{\theta : 2[\ell_P(\hat\theta) - \ell_P(\theta)] \le \chi^2_{1,1-\alpha}\}$ band shades the profile CI.

**Interactions:**
- Dropdown: scenario. (a) Normal mean $\mu$ with unknown variance $\sigma^2$ (profile recovers exact t-CI — pedagogical anchor); (b) Gamma shape $\alpha$ with unknown rate $\beta$. Default: Normal mean.
- Slider: sample size $n$ (10 to 200).
- Slider: observed MLE $\hat\theta$ (within reasonable range).
- Slider: $\alpha$ (0.01 to 0.20).
- Left panel: contour plot of $\ell(\theta, \psi)$ with the profile curve traced as a ridge in red; the conditional MLE $\hat\psi(\theta)$ is the $\psi$-coordinate where the ridge crosses each vertical $\theta$-slice.
- Right panel: profile $\ell_P(\theta)$ plotted alone; horizontal threshold line at $\ell_P(\hat\theta) - \chi^2_{1, 1-\alpha}/2$; shaded band between the two intersections is the profile CI.
- Below right panel: the analogous Wald CI (quadratic approximation to profile at $\hat\theta$) overlaid for comparison. When the profile is non-quadratic, the two CIs differ — a direct visual of the §19.4 Rem 8 finite-sample divergence story.
- Bottom readout: profile CI $[\cdots, \cdots]$; Wald CI $[\cdots, \cdots]$; for Normal: "Exact t-CI is $[\cdots, \cdots]$ — profile recovers it up to the $\chi^2_1$ vs $t_{n-1}^2$ asymptotic substitution."

**Data:** `profilePresets` from `confidence-intervals-data.ts`.

**Uses from `testing.ts`:** `profileLikelihoodCI`, `logLikelihoodNormal2D`, `logLikelihoodGamma2D`, `chiSquaredInvCDF`, `studentTInvCDF`.

**Mobile behavior:** contour plot full-width on top; profile plot below; Wald comparison overlay toggleable.

**Performance note:** contour evaluation on a 100×100 grid. Memoize with `useMemo` keyed on `(scenario, n, θ̂, α)`. Debounce slider updates at 150ms.

### Component 4 (optional): ReparamCIInvariance

**File:** `src/components/viz/ReparamCIInvariance.tsx`

**Purpose:** The CI-specific analog of Topic 18's `ReparamInvariance`. Demonstrate that the Wald CI endpoints depend on parameterization (the back-transformed Wald CI on $\eta = \logit p$ is *not* equal to the Wald CI on $p$), while the LRT CI endpoints are invariant (Thm 6 of Topic 18, dualized). This is the CI companion to the Topic 18 optional component.

**Interactions:**
- Slider: observed $\hat p$ (binomial sample proportion).
- Slider: sample size $n$.
- Dropdown: reparameterization (logit, log, probit). Default: logit.
- Left panel: Wald CI computed in the $p$-parameterization (directly), and Wald CI computed in the $\eta$-parameterization then back-transformed to $p$-space. Two intervals are shown side-by-side with numerical endpoints; the difference is highlighted.
- Right panel: LRT CI computed in each parameterization and back-transformed — both yield the same $p$-space interval (within numerical tolerance), confirming invariance.
- Bottom readout: "At $\hat p = \cdots, n = \cdots$: Wald CI in $p$-space is $[\cdots, \cdots]$; Wald CI in $\eta$-space back-transformed is $[\cdots, \cdots]$. Difference = $\cdots$. LRT CIs agree to $\le 10^{-6}$."

**Data:** `reparamCIPresets` from `confidence-intervals-data.ts`.

**Uses from `testing.ts`:** `waldCIBinomial`, `waldCILogit`, `lrtCIBinomial`, `chiSquaredInvCDF`.

**Mobile behavior:** left panel on top, right panel below.

**Decision:** build only if time permits. Topic 19 is acceptable with three components.

### Implementation Notes

- **Architecture:** same as Topics 1–18. React + Tailwind, `client:visible` directive.
- **SVG/D3.js for all charts.** No Chart.js, Plotly, Recharts, or other libraries.
- **CSS custom properties** from `viz/shared/colorScales.ts` for the Track 5 palette (amber Wald, purple Score/Wilson, green LRT/profile).
- **Responsive:** all components tested at 375px width.
- **Client-side computation only.**
- **`CITestDualityVisualizer`** is the showpiece; polish the slicing animations; use lightweight D3 for the draggable lines.
- **`BinomialCoverageComparison`** uses *exact* coverage via the binomial CDF, not MC — this is faster, more accurate, and more pedagogically correct (exhibits the sawtooth jumps that MC obscures).
- **`ProfileLikelihoodExplorer`** is the heaviest component; memoize the contour grid; debounce sliders.

---

## 6. Shared Module (`testing.ts` extension)

`testing.ts` is the Track 5 shared module, created in Topic 17 and extended in Topics 18 and 19. The Topic 17–18 baseline includes null-distribution densities/CDFs/quantiles (z, t, χ², F, non-central χ²), the full Wald/Score/LRT trio for Bernoulli/Normal/Poisson, the binomial exact test and UMP boundary finder, the NP likelihood-ratio evaluator, and the Wilks Monte Carlo simulator. Topic 19 adds the CI-construction layer: Wald/Score/LRT inversions, Wilson and Clopper–Pearson for binomial, profile likelihood, TOST, and a coverage simulator.

### 6.1 Full new-function manifest

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// TOPIC 19 EXTENSIONS TO testing.ts
// ═══════════════════════════════════════════════════════════════════════════

// ── Pivotal confidence intervals ────────────────────────────────────────────

/** z-CI for Normal mean with known σ: x̄ ± z_{α/2} σ / √n. */
export function zCINormalMean(
  data: number[], sigmaKnown: number, alpha: number
): { lower: number; upper: number };

/** t-CI for Normal mean with unknown σ: x̄ ± t_{n-1, α/2} S / √n. */
export function tCINormalMean(
  data: number[], alpha: number
): { lower: number; upper: number };

/** χ²-CI for Normal variance (μ unknown):
 *  [(n-1)S² / χ²_{n-1, 1-α/2},  (n-1)S² / χ²_{n-1, α/2}]. */
export function chiSquaredCINormalVariance(
  data: number[], alpha: number
): { lower: number; upper: number };

/** F-CI for ratio of Normal variances (two independent samples). */
export function fCIVarianceRatio(
  data1: number[], data2: number[], alpha: number
): { lower: number; upper: number };

// ── Wald / Score / LRT CIs (scalar θ) ───────────────────────────────────────

/** Wald CI: θ̂ ± z_{α/2} / √(n I(θ̂)).
 *  Families: 'bernoulli', 'normal-mean', 'normal-variance', 'poisson', 'exponential'.
 *  Returns {lower, upper} clipped to the parameter space if applicable. */
export function waldCI(
  family: AsymptoticTestFamily,
  data: number[], alpha: number, knownParam?: number
): { lower: number; upper: number };

/** Score CI: {θ : |S_n(θ)| ≤ z_{α/2}} — typically requires numerical root-finding.
 *  For binomial, returns the Wilson interval in closed form (faster path). */
export function scoreCI(
  family: AsymptoticTestFamily,
  data: number[], alpha: number, knownParam?: number
): { lower: number; upper: number };

/** LRT CI: {θ : -2 log Λ_n(θ) ≤ χ²_{1, 1-α}} — solved by bisection on the profile.
 *  For regular parametric families with scalar θ and no nuisance, returns the
 *  inversion of the Topic 18 generalized LR. */
export function lrtCI(
  family: AsymptoticTestFamily,
  data: number[], alpha: number, knownParam?: number
): { lower: number; upper: number };

// ── Wilson (closed form) ────────────────────────────────────────────────────

/** Wilson interval for binomial p — the score-test inversion of §19.5.
 *  x is the number of successes out of n trials; returns the full closed-form
 *  formula with z = z_{α/2}. Handles x = 0 and x = n without degeneracy. */
export function wilsonInterval(
  x: number, n: number, alpha: number
): { lower: number; upper: number };

/** Agresti–Coull "plus-4" approximation to Wilson: add 2 successes and 2 failures,
 *  then apply Wald. Simpler than Wilson but close in coverage (§19.5 Rem 9). */
export function agrestiCoullInterval(
  x: number, n: number, alpha: number
): { lower: number; upper: number };

// ── Clopper–Pearson (exact) ─────────────────────────────────────────────────

/** Clopper–Pearson exact binomial CI via beta quantile.
 *  p_L = B^{-1}(α/2; x, n-x+1),  p_U = B^{-1}(1-α/2; x+1, n-x).
 *  Boundary handling: p_L = 0 when x = 0; p_U = 1 when x = n. */
export function clopperPearsonInterval(
  x: number, n: number, alpha: number
): { lower: number; upper: number };

/** Generic exact CI for discrete one-parameter families via CDF-step inversion.
 *  Used by Poisson (gamma–Poisson duality). Returns exact but conservative CI. */
export function exactDiscreteCI(
  family: 'binomial' | 'poisson',
  observedStatistic: number, sampleSize: number, alpha: number
): { lower: number; upper: number };

// ── Profile likelihood ──────────────────────────────────────────────────────

/** Profile-likelihood CI for a scalar parameter of interest θ in a model with
 *  nuisance ψ. `logLikelihood(theta, psi)` returns ℓ; `psiGrid(theta)` yields the
 *  search grid for ψ at each θ (or `psiOptimize` does numerical profiling).
 *  Returns the bisection-found endpoints of the set
 *    {θ : 2[ℓ_P(θ̂) − ℓ_P(θ)] ≤ χ²_{1, 1-α}}. */
export function profileLikelihoodCI(
  logLikelihood: (theta: number, psi: number | number[]) => number,
  psiOptimize: (theta: number) => number | number[],
  thetaHat: number,
  thetaSearchRange: [number, number],
  alpha: number
): { lower: number; upper: number; profileCurve: Array<[number, number]> };

/** Helper: profile log-likelihood evaluator for common models. */
export function logLikelihoodNormal2D(mu: number, sigma: number, data: number[]): number;
export function logLikelihoodGamma2D(shape: number, rate: number, data: number[]): number;
export function profileNuisanceOptimizerNormal(mu: number, data: number[]): number;
export function profileNuisanceOptimizerGamma(shape: number, data: number[]): number;

// ── TOST equivalence procedure ──────────────────────────────────────────────

/** Two One-Sided Tests (TOST) for equivalence.
 *  Rejects the non-equivalence null H_0: |θ - θ_0| ≥ δ iff both one-sided tests
 *  of θ ≤ θ_0 - δ and θ ≥ θ_0 + δ reject at level α.
 *  Returns {rejectLow, rejectHigh, equivalence} where equivalence = rejectLow && rejectHigh. */
export function tostTest(
  data: number[],
  theta0: number, delta: number, alpha: number,
  family: 'normal-mean' | 'bernoulli' | 'poisson'
): { rejectLow: boolean; rejectHigh: boolean; equivalence: boolean;
     pLow: number; pHigh: number };

/** TOST CI interpretation: the 1-2α two-sided CI for θ, which is equivalent to
 *  the TOST procedure at level α when compared against the equivalence margin. */
export function tostConfidenceInterval(
  data: number[], alpha: number, family: 'normal-mean' | 'bernoulli' | 'poisson'
): { lower: number; upper: number };

// ── Coverage diagnostics ────────────────────────────────────────────────────

/** Exact actual coverage of a binomial CI procedure, computed by summing over
 *  all (n+1) possible outcomes x ∈ {0, 1, ..., n} weighted by P_p(X = x).
 *  Returns coverage as a function of p across the supplied grid. */
export function actualCoverageBinomial(
  ciProcedure: (x: number, n: number, alpha: number) => { lower: number; upper: number },
  n: number, alpha: number, pGrid: number[]
): number[];

/** Monte Carlo coverage simulator for continuous-parameter CIs.
 *  Seeds RNG with `seed` (default 42 for reproducibility). */
export function coverageSimulator(
  ciProcedure: (data: number[], alpha: number) => { lower: number; upper: number },
  sampler: (trueParam: number, n: number) => number[],
  trueParam: number, n: number, alpha: number, M: number, seed?: number
): { actualCoverage: number; meanWidth: number };

// ── Type extensions ─────────────────────────────────────────────────────────

/** Asymptotic test families supported in Topic 19 CIs — extends Topic 18. */
export type AsymptoticTestFamily =
  | 'bernoulli' | 'normal-mean' | 'normal-variance' | 'poisson' | 'exponential';

/** Standard CI result shape. */
export interface ConfidenceInterval {
  lower: number;
  upper: number;
  alpha: number;
  procedure: string;  // "wald", "wilson", "clopper-pearson", "profile", etc.
}
```

### 6.2 Testing the module

Add to `src/components/viz/shared/testing.test.ts` (existing Topic 17–18 test harness):

- **Test 19A — Wilson closed form.** At $n = 20, x = 5, \alpha = 0.05$: Wilson returns $[0.110, 0.471]$ (matches BRO2001 Table 4 to 3 decimals).
- **Test 19B — Clopper–Pearson closed form.** At $n = 20, x = 5, \alpha = 0.05$: CP returns $[0.087, 0.491]$ (matches R `binom.test` to 3 decimals).
- **Test 19C — Wald CI boundary.** At $n = 20, x = 0, \alpha = 0.05$: Wald returns $[0, 0]$ (degenerate); Wilson returns $[0, 0.161]$; CP returns $[0, 0.168]$.
- **Test 19D — Actual coverage.** At $n = 20, p = 0.1, \alpha = 0.05$: Wald actual coverage $\approx 0.878$ (under-covers); Wilson $\approx 0.961$; CP $\approx 0.997$. Matches BRO2001 Table 1.
- **Test 19E — Profile recovers t-CI.** For Normal mean with unknown variance at $n = 30$, the profile CI matches $\bar X \pm t_{29, 0.025} S/\sqrt{30}$ to within the $\chi^2_1$-vs-$t_{29}^2$ asymptotic gap (< 0.01 in endpoint at $n = 30$).
- **Test 19F — Duality self-consistency.** Invert the score test of $H_0: p = p_0$ at 20 grid values of $p_0$; the resulting non-rejection set exactly equals the Wilson interval from `wilsonInterval(x, n, α)` to within floating-point tolerance.
- **Test 19G — TOST at FDA margin.** For Normal $\bar X = 0.05, S = 0.10, n = 20$, $\theta_0 = 0, \delta = 0.20, \alpha = 0.05$: TOST concludes equivalence (both one-sided tests reject).

All tests seeded where stochastic (seed 42); deterministic tests (Wilson, CP, Wald closed forms) are exact.

---

## 7. Preset Data Module

### Create: `src/data/confidence-intervals-data.ts`

Three preset collections power the four interactive components.

```typescript
export const dualityPresets = [
  {
    name: "Normal mean (σ = 1)",
    family: "normal-mean",
    n: 30,
    alpha: 0.05,
    trueParam: 0,
    observedStatistic: 0.25,  // x̄
  },
  {
    name: "Bernoulli proportion",
    family: "bernoulli",
    n: 50,
    alpha: 0.05,
    trueParam: 0.4,
    observedStatistic: 0.36,  // p̂
  },
  {
    name: "Poisson rate",
    family: "poisson",
    n: 40,
    alpha: 0.05,
    trueParam: 2.0,
    observedStatistic: 1.9,  // λ̂
  },
];

export const coveragePresets = [
  { name: "n = 20 (small)", n: 20, alpha: 0.05 },
  { name: "n = 100 (moderate)", n: 100, alpha: 0.05 },
  { name: "n = 500 (large)", n: 500, alpha: 0.05 },
];

export const profilePresets = [
  {
    name: "Normal mean, σ unknown",
    scenario: "normal-2d",
    n: 20,
    observedThetaHat: 0.5,   // x̄
    observedNuisance: 1.2,    // S
    alpha: 0.05,
    thetaGrid: [-1, 2, 100],  // [lo, hi, npoints]
    psiGrid: [0.1, 3, 50],
  },
  {
    name: "Gamma shape, rate unknown",
    scenario: "gamma-2d",
    n: 30,
    observedThetaHat: 2.5,   // shape MLE
    observedNuisance: 1.8,    // rate conditional MLE at shape = 2.5
    alpha: 0.05,
    thetaGrid: [0.5, 6, 100],
    psiGrid: [0.1, 5, 50],
  },
];

export const reparamCIPresets = [
  {
    name: "Bernoulli p → logit η",
    pHat: 0.3, n: 50, alpha: 0.05, reparam: "logit",
  },
  {
    name: "Bernoulli p → log p (rare event)",
    pHat: 0.05, n: 100, alpha: 0.05, reparam: "log",
  },
  {
    name: "Bernoulli p → probit",
    pHat: 0.35, n: 40, alpha: 0.05, reparam: "probit",
  },
];
```

All new preset data lives in `confidence-intervals-data.ts`. No preset updates are needed to prior topics' data modules.

---

## 8. Cross-Reference Updates

### 8.1 Updates to `hypothesis-testing.mdx` (Topic 17)

Four live "Topic 19" pointers that need to become live links. Verify each with grep before editing.

| # | Source MDX location | Find (approximate; verify with grep) | Replace with |
|---|---|---|---|
| 1 | §17.10 Rem 12 | "Topic 19 develops all four constructions." | "[Topic 19 develops all four constructions.](/topics/confidence-intervals-and-duality#section-19-2)" |
| 2 | §17.10 Ex 17 | "Topic 19 develops the full set." | "[Topic 19 develops the full set.](/topics/confidence-intervals-and-duality#section-19-4)" |
| 3 | §17.10 Rem 13 | "Full theory, with coverage diagnostics and comparisons, is **Confidence Intervals (coming soon, Topic 19)**." | "Full theory, with coverage diagnostics and comparisons, is [Confidence Intervals & Duality (Topic 19)](/topics/confidence-intervals-and-duality)." |
| 4 | §17.12 Rem 19 | "**Confidence Intervals (coming soon, Topic 19)** — the duality previewed in §17.10 becomes the construction. Pivotal quantities, Wald / score / LRT intervals, coverage diagnostics, the Wilson interval for binomial proportions (which fixes the Wald boundary problem from Remark 13)." | "[Confidence Intervals & Duality (Topic 19)](/topics/confidence-intervals-and-duality) — the duality previewed in §17.10 becomes the construction. Pivotal quantities, Wald / score / LRT intervals, coverage diagnostics, the Wilson interval for binomial proportions (which fixes the Wald boundary problem from Remark 13)." |

### 8.2 Updates to `likelihood-ratio-tests-and-np.mdx` (Topic 18)

**Important:** Topic 18 already used the placeholder slug `/topics/confidence-intervals`. These links must be updated to `/topics/confidence-intervals-and-duality`. There are three affected locations.

| # | Source MDX location | Find | Replace with |
|---|---|---|---|
| 1 | §18.5 Rem 10 | `[Topic 19](/topics/confidence-intervals) (forthcoming)` | `[Topic 19](/topics/confidence-intervals-and-duality#section-19-7)` |
| 2 | §18.8 Rem 16 | "the Wilson interval in Topic 19" (plain text, no link) | "the [Wilson interval in Topic 19](/topics/confidence-intervals-and-duality#section-19-5)" |
| 3 | §18.10 Rem 22 | `[Topic 19 (forthcoming)](/topics/confidence-intervals)` | `[Topic 19](/topics/confidence-intervals-and-duality)` |

**Grep commands to verify complete coverage:**

```bash
# Should return exactly the four Topic-17 references and three Topic-18 references after the update
grep -rn '/topics/confidence-intervals-and-duality' --include='*.mdx' src/content/topics/

# Should return zero remaining references to the placeholder slug
grep -rn '/topics/confidence-intervals[^-]' --include='*.mdx' src/content/topics/

# Should return zero remaining "(coming soon, Topic 19)" markers
grep -rn 'coming soon, Topic 19\|Topic 19.*coming soon' --include='*.mdx' src/content/topics/
```

All three grep outputs should match their expected counts after the edits.

### 8.3 Updates to `curriculum-graph.json`

Mark `confidence-intervals-and-duality` as published; add prerequisite edges.

```json
{
  "id": "confidence-intervals-and-duality",
  "title": "Confidence Intervals & Duality",
  "track": 5,
  "trackName": "Hypothesis Testing & Confidence",
  "topicNumber": 19,
  "status": "published",
  "prerequisites": [
    "hypothesis-testing",
    "likelihood-ratio-tests-and-np",
    "maximum-likelihood",
    "point-estimation",
    "sufficient-statistics",
    "continuous-distributions",
    "discrete-distributions"
  ]
}
```

### 8.4 Updates to `curriculum.ts`

Track 5 status: `3 of 4 published` (was `2 of 4` after Topic 18).

### 8.5 References spreadsheet updates

**Add 5 new entries** to `formalstatisticscitations.xlsx`:

| Short key | Citation | URL | Used In Topics |
|---|---|---|---|
| WIL1927 | Wilson, Edwin B. 1927. "Probable Inference, the Law of Succession, and Statistical Inference." JASA 22(158): 209–212. | https://doi.org/10.1080/01621459.1927.10502953 | confidence-intervals-and-duality |
| CLO1934 | Clopper, C. J., and E. S. Pearson. 1934. "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial." Biometrika 26(4): 404–413. | https://doi.org/10.1093/biomet/26.4.404 | confidence-intervals-and-duality |
| BRO2001 | Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. 2001. "Interval Estimation for a Binomial Proportion." Statistical Science 16(2): 101–133. | https://doi.org/10.1214/ss/1009213286 | confidence-intervals-and-duality |
| AGR1998 | Agresti, Alan, and Brent A. Coull. 1998. "Approximate Is Better than 'Exact' for Interval Estimation of Binomial Proportions." The American Statistician 52(2): 119–126. | https://doi.org/10.1080/00031305.1998.10480550 | confidence-intervals-and-duality |
| SCH1987 | Schuirmann, Donald J. 1987. "A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability." J. Pharmacokinetics and Biopharmaceutics 15(6): 657–680. | https://doi.org/10.1007/BF01068419 | confidence-intervals-and-duality |
| NEY1937 | Neyman, Jerzy. 1937. "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability." Phil. Trans. Roy. Soc. A 236(767): 333–380. | https://doi.org/10.1098/rsta.1937.0005 | confidence-intervals-and-duality |

**Update 4 existing entries** — append `confidence-intervals-and-duality` to the "Used In Topics" column:

- `WIL1938` (was: `hypothesis-testing, likelihood-ratio-tests-and-np`) → `hypothesis-testing, likelihood-ratio-tests-and-np, confidence-intervals-and-duality`
- `LEH2005` (was: `hypothesis-testing, likelihood-ratio-tests-and-np`) → `hypothesis-testing, likelihood-ratio-tests-and-np, confidence-intervals-and-duality`
- `CAS2002` (was: `[prior], hypothesis-testing, likelihood-ratio-tests-and-np`) → `[prior], hypothesis-testing, likelihood-ratio-tests-and-np, confidence-intervals-and-duality`
- `BAS1955` (if present from Topic 16) — confirm still referenced in §19.3 via t-CI exact derivation; append if not already.

Before editing the XLSX, verify the existing schema with:

```bash
python -c "import openpyxl; wb = openpyxl.load_workbook('formalstatisticscitations.xlsx'); ws = wb.active; print([c.value for c in ws[1]])"
```

Expected header row: `Short key | Citation | URL | Used In Topics` (or similar). Preserve the exact column layout.

### 8.6 Sitemap / RSS

Verify `sitemap.xml` and `rss.xml` include the new `/topics/confidence-intervals-and-duality` URL. These are auto-generated on build; verify after deploy.

---

## 9. Verification Checklist

### Content

- [ ] `confidence-intervals-and-duality.mdx` renders at `/topics/confidence-intervals-and-duality` with no build errors.
- [ ] All 10 static images present in `public/images/topics/confidence-intervals-and-duality/`.
- [ ] KaTeX renders correctly: all display equations, inline math, theorem blocks.
- [ ] All 9 definitions, 7 theorems, 14 examples, 23 remarks render in styled blocks.
- [ ] All 4 full proofs expand completely with no hand-waving (duality, Wilson, Clopper–Pearson, profile).
- [ ] Proofs use separate `$$...$$` blocks with prose connectors — **NO** `\begin{aligned}` blocks and **NO** `\begin{array}` tables anywhere.
- [ ] All section headings use the `19.X` prefix (§19.1 through §19.10).
- [ ] Notation consistent with Topics 17–18: $\hat\theta_n$ (MLE), $I(\theta)$ (Fisher info), $W_n, S_n, -2\log\Lambda_n$ (test statistics), $z_{\alpha/2}, t_{n-1,\alpha/2}, \chi^2_{k, 1-\alpha}$ (quantiles).
- [ ] Citation style is **end-of-proof** (`∎ — using [...]`), applied uniformly across all 4 proofs.
- [ ] §19.1 hammers the "95% probability" trap explicitly via a dedicated RemarkBlock with frequentist-vs-Bayesian semantics contrasted.

### Proof-specific checks

- [ ] **Proof 1 (duality)**: 2-step bidirectional argument (tests → CI; CI → tests); size-constraint inequality in each direction; final `∎ — using NEY1937`.
- [ ] **Proof 2 (Wilson)**: 5-step structure (set up inversion → quadratic rearrangement → solve → simplify discriminant → assemble); final `∎ — using duality, score null dist (Topic 18 §18.7 Thm 5), WIL1927`.
- [ ] **Proof 3 (Clopper–Pearson)**: 5-step structure (exact test inversion → beta–binomial identity → solve for $p_L$ → solve for $p_U$ → coverage bound via discreteness); final `∎ — using duality, beta–binomial identity, CLO1934`.
- [ ] **Proof 4 (profile-likelihood coverage)**: 3-step structure (profile LRT statistic → Wilks' asymptotic null → coverage by inversion); cites [Topic 18 §18.6 Thm 4](/topics/likelihood-ratio-tests-and-np#section-18-6) directly; final `∎ — using duality, Wilks (Topic 18 §18.6), continuous mapping (Topic 9)`.

### Forward-promise fulfillment

- [ ] §19.2 delivers the test–CI duality theorem with full proof — fulfills Topic 17 §17.10 Rem 12 + Topic 18 §18.10 Rem 22 forward promises (#1, #4, #7).
- [ ] §19.3 delivers z/t/χ²/F pivotal CIs — fulfills Topic 17 §17.10 Ex 17 forward promise (#2).
- [ ] §19.4 delivers Wald/Score/LRT CIs with explicit Bernoulli, Normal, Poisson specializations — fulfills Topic 17 §17.12 Rem 19 (#4).
- [ ] §19.5 delivers Wilson interval with full proof — fulfills Topic 17 §17.10 Rem 13 and Topic 18 §18.8 Rem 16 (#3, #6).
- [ ] §19.6 delivers Clopper–Pearson exact interval with full proof — beyond-forward-promise deliverable.
- [ ] §19.7 delivers profile-likelihood CI with asymptotic-coverage proof citing §18.6 — fulfills Topic 18 §18.5 Rem 10 (#5).
- [ ] §19.8 delivers coverage diagnostics — supports Topic 17 §17.10 Rem 13 (#3).
- [ ] §19.9 delivers one-sided CIs and TOST — supplemental Track-5 enrichment; previews Topic 20.

### Cross-reference updates

- [ ] `hypothesis-testing.mdx` §17.10 Rem 12 / Ex 17 / Rem 13 now links to `/topics/confidence-intervals-and-duality` at the appropriate sub-anchors.
- [ ] `hypothesis-testing.mdx` §17.12 Rem 19 replaces "(coming soon, Topic 19)" with live link.
- [ ] `likelihood-ratio-tests-and-np.mdx` §18.5 Rem 10 URL updated from `/topics/confidence-intervals` to `/topics/confidence-intervals-and-duality#section-19-7`.
- [ ] `likelihood-ratio-tests-and-np.mdx` §18.8 Rem 16 adds live link to Wilson.
- [ ] `likelihood-ratio-tests-and-np.mdx` §18.10 Rem 22 URL updated from `/topics/confidence-intervals` to `/topics/confidence-intervals-and-duality`.
- [ ] Grep verifies zero remaining references to placeholder slug `/topics/confidence-intervals` (without the `-and-duality` suffix).
- [ ] Grep verifies zero remaining "(coming soon, Topic 19)" markers.

### References spreadsheet

- [ ] 6 new references added: WIL1927, CLO1934, BRO2001, AGR1998, SCH1987, NEY1937. Each with Chicago 17th ed. text and verified URL.
- [ ] 3 existing references updated in "Used In Topics" column: WIL1938, LEH2005, CAS2002.

### Interactive Components

- [ ] `CITestDualityVisualizer` — draggable horizontal line reveals CI; draggable vertical line reveals acceptance region; both update live; all 3 scenario presets work; mobile responsive at 375px.
- [ ] `BinomialCoverageComparison` — all three CI procedures displayed; exact coverage curves computed (not MC); horizontal nominal line at 0.95; per-draw CI bars for single-draw inspection; all 3 $n$-presets work.
- [ ] `ProfileLikelihoodExplorer` — joint log-likelihood contour renders; profile curve traces ridge; $\chi^2_1$-threshold band shades; Wald comparison overlay toggleable; Normal preset recovers t-CI (within asymptotic gap); Gamma preset shows non-symmetric profile.
- [ ] `ReparamCIInvariance` (if built) — all 3 reparameterizations work; Wald CI differences visible; LRT CIs agree to floating-point tolerance.

### Infrastructure

- [ ] Extended `testing.ts` compiles with no TypeScript errors.
- [ ] `confidence-intervals-data.ts` compiles.
- [ ] `testing.test.ts` passes all 7 new tests (19A–19G).
- [ ] Track 5 shows 3/4 topics published in curriculum UI.
- [ ] Pagefind indexes the topic on rebuild.
- [ ] `pnpm build` succeeds with zero errors.
- [ ] Site deploys to Vercel successfully.
- [ ] Mobile responsive (all interactive components at 375px width).
- [ ] "Intermediate" difficulty badge styled correctly.

---

## 10. Build Order

1. **Generate the notebook** `notebooks/confidence-intervals-and-duality/19_confidence_intervals.ipynb` and execute all 13 cells to produce the 10 PNG figures. Verify outputs visually; save to `public/images/topics/confidence-intervals-and-duality/`.
2. **Extend `src/components/viz/shared/testing.ts`** with all functions from §6. Add JSDoc comments. Compile and run existing Topic 17–18 tests to verify no regression.
3. **Write `src/components/viz/shared/testing.test.ts`** — add tests 19A through 19G. Confirm all pass.
4. **Create `src/data/confidence-intervals-data.ts`** with the three preset collections from §7.
5. **Build `CITestDualityVisualizer.tsx`** (Component 1) — the featured component. Test with all 3 scenario presets. Verify draggable slicing works on desktop and touch.
6. **Build `BinomialCoverageComparison.tsx`** (Component 2). Verify the exact coverage computation against the BRO2001 Table 1 reference values.
7. **Build `ProfileLikelihoodExplorer.tsx`** (Component 3). Verify Normal preset recovers t-CI.
8. **Optionally build `ReparamCIInvariance.tsx`** (Component 4) if time permits.
9. **Write `src/content/topics/confidence-intervals-and-duality.mdx`** following the §3 structure. Use the import manifest:

    ```mdx
    import DefinitionBlock from '../../components/ui/DefinitionBlock.astro';
    import TheoremBlock from '../../components/ui/TheoremBlock.astro';
    import ProofBlock from '../../components/ui/ProofBlock.astro';
    import ExampleBlock from '../../components/ui/ExampleBlock.astro';
    import RemarkBlock from '../../components/ui/RemarkBlock.astro';
    import ExternalLink from '../../components/ui/ExternalLink.astro';

    import CITestDualityVisualizer from '../../components/viz/CITestDualityVisualizer.tsx';
    import BinomialCoverageComparison from '../../components/viz/BinomialCoverageComparison.tsx';
    import ProfileLikelihoodExplorer from '../../components/viz/ProfileLikelihoodExplorer.tsx';
    // Optional:
    // import ReparamCIInvariance from '../../components/viz/ReparamCIInvariance.tsx';
    ```

10. **Apply all cross-reference updates** from §8.1, §8.2 (to Topics 17 and 18 MDX). Run the grep-verification commands.
11. **Update `src/data/curriculum-graph.json`** and `curriculum.ts` per §8.3, §8.4.
12. **Update `formalstatisticscitations.xlsx`** per §8.5 — add 6 new rows, update 3 existing rows.
13. **Local dev verification.** `pnpm dev`, navigate to `/topics/confidence-intervals-and-duality`, walk through every section; verify KaTeX renders; verify all four interactive components work; verify links to Topics 17–18 back-navigate correctly.
14. **Commit, push, deploy.** Confirm Vercel deploy succeeds. Run production-URL smoke test.
15. **Post-deploy validation.** Verify Pagefind has indexed the topic; verify sitemap.xml and rss.xml include the new URL; verify the Topic 18 links now resolve correctly.

---

## Appendix A: Style Guidelines

Topic 19 inherits all Topic 17–18 style conventions without modification. The critical reminders:

### KaTeX

- **No `\begin{aligned}` blocks.** Multi-line derivations use separate `$$…$$` blocks with prose connectors. Proof 2 has 5 separate display blocks connected by prose; Proof 3 has 5 separate display blocks with explicit step labels.
- **No `\begin{array}` tables.** Use Markdown tables. The cheat sheet in §19.10 Rem 22 is a Markdown table; the coverage readouts in §19.8 Ex 12 are Markdown tables.
- **Inline math** uses `$...$`. **Display math** uses `$$...$$` on its own line.
- **Test all LaTeX in the dev server before committing.**

### Notation

- **Confidence set / interval:** $C(X)$, $C_n(X)$.
- **CI endpoints:** $L(X), U(X)$; for binomial specifics, $p_L, p_U$.
- **Quantiles:** $z_{\alpha/2}, t_{n-1,\alpha/2}, \chi^2_{k,1-\alpha}, F_{k_1,k_2,\alpha/2}$ — matches Topics 17–18.
- **Profile log-likelihood:** $\ell_P(\theta)$. Not abbreviated.
- **Coverage:** $P_\theta(\theta \in C(X))$.
- **TOST:** equivalence margin $\delta$.
- **Reparameterization:** $\eta = g(\theta)$ when general; $\eta = \logit p$ for binomial specifics.

### Figure callouts

Inline figure references use the Markdown image syntax with square-bracket alt text that doubles as the figure caption:

```mdx
![Figure 2 — two-plane duality diagram with shaded non-rejection region.](/images/topics/confidence-intervals-and-duality/19-duality-diagram.png)
```

The featured figure (Figure 2) appears immediately after Theorem 1 statement in §19.2, before Proof 1.

### Cross-references

- Topic 17: `[Topic 17](/topics/hypothesis-testing)`, with sub-anchors `#section-17-X`.
- Topic 18: `[Topic 18](/topics/likelihood-ratio-tests-and-np)`, with sub-anchors `#section-18-X`.
- Topics 13, 14, 16 (Track 4): live links using the existing conventions.
- formalcalculus.com and formalml.com: `<ExternalLink>` component with `site="formalcalculus"` or `site="formalml"`.

---

## Appendix B: Design Decisions

1. **Ten sections (locked via opening decision list, Decision 10).** Ten sections match Topic 18's rhythm, keep the profile-likelihood section (§19.7) breathing room, and preserve §19.10 as a dedicated forward-look. Compressing to 8 would force merging §19.8 (coverage diagnostics) into §19.5 (Wilson) — blurring the "construction" arc of §19.2–§19.7 with the "calibration" arc of §19.8 — and would drop the §19.9 TOST treatment that gives §19.9 its pedagogical identity.

2. **Intermediate difficulty, 55-min read (locked via Decision 1).** Four of the five main constructions (z-CI, t-CI, Wald, LRT) are one-step inversions of Topic 17–18 tests. Original machinery concentrates in §19.5 (Wilson), §19.6 (Clopper–Pearson), §19.7 (profile likelihood). The frequentist-semantics clarification in §19.1 and the coverage-diagnostic treatment in §19.8 each get roughly a derivation-section's weight. Total target 9,500 words vs Topic 18's 10K.

3. **Four full proofs (locked via Decision 2).** Duality (10 lines), Wilson (15 lines), Clopper–Pearson (20 lines), profile (10 lines). Total ~55 MDX lines of proof machinery — matches Topic 18's proof density.

4. **Three required + one optional interactive component (locked via Decision 3).** Matches Topic 18's 3+1 pattern. Required: `CITestDualityVisualizer` (featured, for §19.2); `BinomialCoverageComparison` (for §19.8); `ProfileLikelihoodExplorer` (for §19.7). Optional: `ReparamCIInvariance` (for §19.9, CI-companion to Topic 18's `ReparamInvariance`).

5. **Scalar θ only with vector-θ preview in §19.7 (locked via Decision 5).** Every proof is for scalar θ. The profile-likelihood section previews the vector case with a worked Normal-mean-unknown-variance example and one figure; simultaneous CIs and Hotelling's T² are deferred to Topic 20 / cited to LEH2005 Ch. 8.

6. **TOST as one-theorem + one-example + one-remark (locked via Decision 4).** Schuirmann 1987 is attributed in Thm 7. The FDA bioequivalence 80/125 rule is the worked example on a log scale. Noninferiority vs. equivalence distinction is the remark. No full proof — the TOST construction is direct from the one-sided test machinery of §19.9.

7. **`testing.ts` extension, not a new module (locked via Decision 7).** The ~150-line extension adds CI-construction machinery (Wald/Score/LRT CI, Wilson, Clopper–Pearson, profile, TOST, coverage simulator) to the existing Topic 17–18 `testing.ts`. Single-module-per-track pattern continues.

8. **Five new references, four existing references extended (locked via Decision 6).** New: WIL1927, CLO1934, BRO2001, AGR1998, SCH1987, NEY1937. Extended: WIL1938, LEH2005, CAS2002 usage. Chicago 17th ed. format with verified URLs throughout.

9. **Slug `confidence-intervals-and-duality` (locked via Jonathan's explicit ruling).** Matches Topic 18's `-and-` compound pattern. Requires patching three live Topic 18 links that point to the placeholder `/topics/confidence-intervals` — see §8.2. Ensures consistent URL conventions across the track.

10. **Featured figure = `19-duality-diagram.png` (two-plane slicing diagram).** This is the visual analog of Topic 18's `18-wilks-convergence.png` — the single figure that delivers ~30% of the topic's pedagogical payoff on its own. The CI-test correspondence is literally geometric in the two-plane picture: horizontal slicing at fixed data gives the CI; vertical slicing at fixed null value gives the test. `CITestDualityVisualizer` animates this.

11. **§19.1 hammers the "95% probability" trap hard.** Three remarks (Rem 1 Neyman 1937 origin; Rem 2 the trap itself; Rem 3 frequentist-vs-Bayesian contrast) + Figure 1 with the frequentist / Bayesian conceptual split. This is the topic's #1 pedagogical anchor, and it gets upfront weight rather than being buried.

12. **Wilson's proof uses the quadratic-formula closed-form derivation (5 steps).** The alternative — treating Wilson as a numerical root-find of the score statistic — is correct but less pedagogical. The closed form is the reason Wilson is fast and default-recommended; the proof should make that explicit.

13. **Clopper–Pearson proof uses the beta–binomial identity as its centerpiece.** The identity $P_p(X \le k) = I_{1-p}(n - k, k + 1)$ is the master piece of discrete-test inversion. Using it explicitly (rather than hand-waving "the exact test inverts to a beta quantile") gives §19.6 its theoretical backbone and connects back to Topic 6's beta CDF.

14. **Coverage diagnostics use *exact* computation, not Monte Carlo.** For binomial coverage, the exact actual coverage is computable by summing over the $n+1$ possible outcomes. This is faster, more accurate, and visually more honest than MC — it shows the sawtooth pattern caused by discreteness, which MC would smooth. `BinomialCoverageComparison` uses the exact computation.

15. **End-of-proof citation style (`∎ — using [...]`), applied uniformly.** Matches Topics 17–18. All four proofs end with the same visual marker naming the specific tools consumed.

---

*Brief version: v1 | Created: 2026-04-18 | Author: Jonathan Rocha*
*Reference notebook: `notebooks/confidence-intervals-and-duality/19_confidence_intervals.ipynb`*
