# Claude Code Handoff Brief: Likelihood-Ratio Tests & Neyman-Pearson

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)
**Repo:** `github.com/jonx0037/formalStatistics`
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel
**Package Manager:** pnpm
**Status:** Ready for implementation
**Reference Notebook:** `notebooks/likelihood-ratio-tests/18_likelihood_ratio_tests.ipynb`

---

## Important: Eighteenth Topic — Second in Track 5

This is **topic 18 of 32** and the **second topic in Track 5 (Hypothesis Testing & Confidence)**. Track 5 opened with Topic 17 (Hypothesis Testing Framework), which built the scaffolding — null vs. alternative, test statistics, size and power, p-values, z / t / χ² — and left the optimality theory as Topic 18's territory. Topic 18 is the payoff: the Neyman-Pearson lemma for simple-vs-simple tests, the Karlin-Rubin theorem for one-sided composite tests, Wilks' theorem for the composite likelihood-ratio test, and the first-order equivalence of the Wald, Score, and LRT asymptotic trio. Topic 18 also fulfills seven explicit forward-promises that Topic 17 made, each of which appears as a "Topic 18 will..." pointer in the published MDX.

**Implications:**

1. **All seventeen preceding topics are published.** Cross-references from `sample-spaces` through `hypothesis-testing` use live internal links.

2. **Seven explicit Topic 17 forward-promises must be fulfilled.** Each appears as a "Topic 18 will..." pointer in the published `hypothesis-testing.mdx`. See §1 below for the complete list with line numbers and verbatim text.

3. **Topic 18 *extends* — does not create — the shared Track 5 module `testing.ts`.** Topic 17 seeded `testing.ts` with size/power helpers, analytic null distributions for z/t/χ², the binomial exact test, the Monte Carlo p-value runner, and Wald/Score/LRT closed forms for Bernoulli, Normal, and Poisson. Topic 18 adds the Neyman-Pearson likelihood-ratio evaluator, the UMP boundary finder for MLR families, the non-central χ² distribution for local power, the Wilks Monte Carlo simulator, and a local-power computer. See §6 for the complete new-function manifest.

4. **Do not re-derive MLE asymptotic normality.** Topic 14 Thm 14.3 is the engine of §18.6's Wilks proof. §18.6 cites it; it does not re-prove it. The analogous pedagogical hygiene applies to Topic 17 Thm 7 (Wald case derived inline) and Topic 9's continuous mapping + Slutsky.

5. **Do not re-derive the Neyman-Pearson lemma in §18.3.** §18.2 proves it in full; §18.3 cites Theorem 1 and uses it as a black-box engine in the Karlin-Rubin proof.

6. **Scalar θ only.** Every proof is written for scalar θ ∈ ℝ. Vector-θ generalizations (k-df Wilks, vector-valued UMP, Hunt-Stein, UMP unbiased) are stated with citations to Lehmann-Romano 2005 and nothing more. This scope discipline is load-bearing — it keeps the topic's length and rigor register consistent with Topics 14 and 16.

7. **One new Topic 17 cross-reference will be added by Claude Code:** §17.9 Thm 7's statement "(Wilks 1938; full proof in Topic 18)" in Def 15 needs to have "(Topic 18)" replaced with a live link to `/topics/likelihood-ratio-tests-and-np#section-18-6`. See §8 below.

---

## 1. Objective & Scope

### 1.1 Objective

Deliver the optimality theory for classical hypothesis testing. Topic 17 established the framework — this topic answers the question "which level-α test should we use?" through four complementary optimality results:

1. **Neyman-Pearson lemma** (§18.2) — for simple-vs-simple hypotheses, the likelihood-ratio test is most powerful. This is the finite-sample optimality result, provable from indicator-function algebra alone.

2. **Karlin-Rubin theorem** (§18.3) — for one-sided composite hypotheses in families with monotone likelihood ratio (MLR), the NP construction yields a single uniformly most powerful (UMP) test. MLR is the structural property that makes "one test is MP at every alternative" possible.

3. **Wilks' theorem** (§18.6) — for composite hypotheses in regular parametric families, −2 log Λ_n converges in distribution to χ²_k, where k is the number of parameter restrictions. This is the asymptotic optimality result and the technical high point of the topic.

4. **Three-tests equivalence** (§18.7) — the Wald, Score, and likelihood-ratio statistics agree to first order under the null. They differ in finite samples in specific, predictable ways (§18.8), and those differences matter in practice.

The topic also quantifies the finite-sample divergence of the three tests (§18.8), proves the reparameterization invariance of the LRT (§18.8 Thm 6), and characterizes local power via non-central χ² (§18.9) — tying the topic back to Topic 13's CRLB as the envelope for achievable local power.

### 1.2 Scope Boundaries

What Topic 18 covers:
- Scalar θ ∈ ℝ throughout. Vector-θ results stated only, with LEH2005 pointers.
- Four full proofs (NP lemma, Karlin-Rubin, Wilks, three-tests equivalence).
- Three required interactive components (NPLemmaVisualizer, KarlinRubinUMP, WilksConvergence) plus one optional (ReparamInvariance).
- Ten numbered sections (§18.1–§18.10).
- 10 PNG figures.

What Topic 18 does NOT cover:
- UMP unbiased tests → stated, cited to LEH2005 Ch. 4. Not proved.
- Invariance and similar tests → stated, cited to LEH2005 Ch. 6. Not proved.
- Profile likelihood → single remark preview in §18.5. Full treatment is Topic 19.
- Confidence intervals → one-paragraph remark in §18.10 pointing forward. Full treatment is Topic 19.
- Multiple testing / FDR / Bonferroni → one-paragraph remark in §18.10. Full treatment is Topic 20.
- Multi-parameter UMP, Hunt-Stein theorem, boundary cases (Chernoff 1954), non-regular MLEs → cited only.
- LAN / Le Cam / empirical-process treatment of Wilks → one remark pointing to VAN1998 §16.

### 1.3 Forward-promise fulfillment (7 items from Topic 17)

These are the "Topic 18 will..." pointers currently live in `hypothesis-testing.mdx`. Each must be fulfilled by Topic 18's content and — where the MDX pointer reads "Topic 18" as plain text — Claude Code must replace with a live internal link after Topic 18 ships. See §8 for the exact MDX updates.

| # | Source location | Verbatim Topic 17 text | Topic 18 deliverable |
|---|---|---|---|
| 1 | `hypothesis-testing.mdx` §17.2 Definition 2 + §17.9 | Simple vs composite distinction; promise that the NP lemma covers the simple-vs-simple optimality question | §18.2 Theorem 1 + Proof 1 (NP lemma in full) |
| 2 | `hypothesis-testing.mdx` §17.4 Theorem 2 + proof sketch | "Power is non-decreasing in effect size for one-sided tests in exp families — stated, proved in Topic 18 via MLR" | §18.3 Theorem 2 + Proof 2 (Karlin-Rubin); §18.4 Example 4 (exp families have MLR) |
| 3 | `hypothesis-testing.mdx` §17.9 Theorem 7 + Def 15 | "Wilks' regularity conditions, −2 log Λ_n →_d χ²_k where k is the number of restricted parameters (Wilks 1938; full proof in Topic 18)" | §18.6 Theorem 4 + Proof 3 (Wilks in full) |
| 4 | `hypothesis-testing.mdx` §17.4 Remark 5 | CRLB as power-envelope preview | §18.9 Theorem 7 + Remark 18 (non-central χ²; I(θ₀) as power envelope) |
| 5 | `hypothesis-testing.mdx` §17.6 Example 11 | Binomial exact test's UMP status — "foreshadows Topic 18's UMP machinery" | §18.4 Example 6 (binomial exact is UMP, via Karlin-Rubin) |
| 6 | `hypothesis-testing.mdx` §17.9 Remark 10 | "Topic 18 treats the finite-sample divergence quantitatively" | §18.8 Example 14 (MC: Wald liberal, Score conservative, LRT nominal at n = 20) |
| 7 | `hypothesis-testing.mdx` §17.9 Remark 11 | "LRT is parameterization-invariant; Wald is not" — concrete example promised | §18.8 Theorem 6 + Example 13 (Bernoulli p vs logit η) |

---

## 2. Frontmatter

### 2.1 Full frontmatter spec

The MDX file begins with the following YAML. Every field is required; every `url:` must resolve.

```yaml
---
title: "Likelihood-Ratio Tests & Neyman-Pearson"
slug: "likelihood-ratio-tests-and-np"
track: 5
trackName: "Hypothesis Testing & Confidence"
topicNumber: 18
positionInTrack: 2
readTime: "60 min"
difficulty: "intermediate"
status: "published"
publishedDate: "TO BE SET AT PUBLICATION"
lastUpdated: "TO BE SET AT PUBLICATION"
description: "Readers who have completed Topic 17's hypothesis-testing framework and Topic 14's MLE asymptotic normality theorem, and who are ready for the optimality layer: the Neyman-Pearson lemma for simple-vs-simple tests, Karlin-Rubin's one-sided UMP construction via monotone likelihood ratio, and Wilks' χ² theorem for the likelihood-ratio statistic. The exposition stays in the classical Taylor-expansion / Slutsky / continuous-mapping register established in Topics 9 and 14 — no empirical-process theory, no Le Cam / LAN machinery, and no new measure-theoretic tools."
prerequisites:
  - topic: "hypothesis-testing"
    relationship: "Topic 17 built the hypothesis-testing framework: null, alternative, size, power, p-values, and the Wald/Score/LRT trio with its asymptotic null distribution stated. Topic 18 delivers the optimality theory on top: NP for simple-vs-simple, Karlin-Rubin for one-sided composite, Wilks for composite χ² limit, and the first-order equivalence of the trio."
  - topic: "maximum-likelihood"
    relationship: "Topic 14 Thm 14.3 (MLE asymptotic normality, √n(θ̂−θ₀) →_d N(0, I(θ₀)^{−1})) is the engine of Wilks' theorem. Topic 14 §14.6 (MLE first-order condition) is Step 3 of the Wilks proof. Topic 14 Thm 14.2 (MLE consistency) is Step 4."
  - topic: "point-estimation"
    relationship: "Topic 13 Thm 13.9 (Cramér-Rao lower bound) reappears in §18.9 as the power envelope: the Fisher information I(θ₀) that bounds estimator variance also bounds the achievable local power of any asymptotically efficient test."
  - topic: "modes-of-convergence"
    relationship: "Topic 9's continuous mapping theorem is Step 7 of the Wilks proof. Slutsky's theorem is Step 8. The O_P / o_P algebra of Topic 9 §9.4 is Step 4's remainder control."
  - topic: "sufficient-statistics"
    relationship: "Topic 16's MLR-via-exponential-family characterization powers §18.4 Example 4 (every exp family has MLR in its natural sufficient statistic). The factorization theorem (Thm 16.1) underlies the 'UMP test depends on data only through T' observation in Proof 2."
  - topic: "exponential-families"
    relationship: "Topic 7's canonical form makes the one-line argument in §18.4 Example 4 possible: the LR f(x;η₂)/f(x;η₁) = exp((η₂−η₁)T(x) − (A(η₂)−A(η₁))) is monotone in T(x) when η₂ > η₁ and T depends on x through the natural sufficient statistic."
  - topic: "central-limit-theorem"
    relationship: "Topic 11 Thm 11.1 is the engine of Topic 14 Thm 14.3, which is in turn the engine of Wilks. §18.9's non-central χ² analysis of local power is a direct CLT corollary applied to the score function under contiguous alternatives."
formalcalculusPrereqs:
  - topic: "differentiation"
    site: "formalcalculus"
    relationship: "The Wilks proof's Step 2 is Taylor's theorem with remainder. Step 3 is the first-order condition for an interior maximum (MLE) expressed as a vanishing derivative."
  - topic: "integration"
    site: "formalcalculus"
    relationship: "The Neyman-Pearson proof integrates the indicator-function inequality (φ − φ')(f₁ − k f₀) ≥ 0 over the sample space. The non-central χ² analysis in §18.9 uses a series-expansion of the CDF."
  - topic: "sequences-limits"
    site: "formalcalculus"
    relationship: "The convergence in distribution of −2 log Λ_n to χ²₁ is a limit statement; the O_P / o_P algebra used in Step 4 of the Wilks proof is rigorous-limit bookkeeping."
formalmlConnections:
  - topic: "ab-testing-platforms"
    site: "formalml"
    relationship: "A/B testing platforms report Wald-style z-tests by default. The reparameterization-invariance argument of §18.8 is why, for rate-based metrics (conversion, CTR), a log-odds-ratio LRT is preferable to a Wald test on raw proportions near zero — a fact that production experimentation systems increasingly encode."
  - topic: "generalized-linear-models"
    site: "formalml"
    relationship: "Wald / Score / LRT tests for GLM coefficients are the direct generalization of §18.7. The score test's 'only fit the null model' advantage is central to stepwise model selection and to the deviance-difference framework for nested models."
  - topic: "statistical-learning-theory"
    site: "formalml"
    relationship: "PAC-Bayes and generalization bounds are uniform-concentration statements whose proofs mirror the indicator-function comparison of Proof 1 (Neyman-Pearson). The 'test is MP ⇔ LR test' duality reappears in the Bayes-decision framing of classification: 0-1 loss + likelihood ratio = Bayes classifier."
  - topic: "model-comparison"
    site: "formalml"
    relationship: "Wilks' theorem is the theoretical backbone of the likelihood-ratio test for nested models — the frequentist answer to 'did the extra parameters help?' The analogous Bayesian answer is the Bayes factor (Topic 22); the ML-native answer is cross-validated held-out likelihood (formalml.com/model-comparison)."
connections:
  - topic: "hypothesis-testing"
    relationship: "Direct continuation. Topic 17 §17.9's inline derivation of the Wald case (via Topic 14 Thm 14.3 + Slutsky + continuous mapping) is the template for Wilks' proof, which extends the same machinery from a linear function of θ̂_n to a quadratic function. Topic 17 Remark 10's statement of asymptotic equivalence is now proved as §18.7 Theorem 5."
  - topic: "maximum-likelihood"
    relationship: "Wilks' proof imports Topic 14 Thm 14.3 at Step 6 and the MLE consistency from Thm 14.2 at Step 4. The classical Taylor-expansion register of Topic 14 is reused without modification — no new analytical machinery."
  - topic: "sufficient-statistics"
    relationship: "Exponential-family MLR is a direct corollary of the factorization theorem. Basu's theorem (Topic 16 Thm 7) does not appear in Topic 18, but its structural role — separating ancillary and sufficient information — parallels the 'UMP test depends on T only' property of Proof 2."
  - topic: "point-estimation"
    relationship: "The Fisher information I(θ₀) plays a dual role: Topic 13 uses it as the CRLB for unbiased estimators; Topic 18 §18.9 uses it as the non-centrality coefficient for local power. These are two faces of the same efficiency bound."
references:
  - id: "NEY1933"
    text: "Neyman, Jerzy, and Egon S. Pearson. 1933. \"On the Problem of the Most Efficient Tests of Statistical Hypotheses.\" Philosophical Transactions of the Royal Society of London, Series A 231: 289–337."
    url: "https://doi.org/10.1098/rsta.1933.0009"
    usedIn: "§18.2 Theorem 1 + Proof 1 attribution — the Neyman-Pearson lemma's original paper."
  - id: "KAR1956"
    text: "Karlin, Samuel, and Herman Rubin. 1956. \"The Theory of Decision Procedures for Distributions with Monotone Likelihood Ratio.\" Annals of Mathematical Statistics 27 (2): 272–99."
    url: "https://doi.org/10.1214/aoms/1177728259"
    usedIn: "§18.3 Theorem 2 attribution — the Karlin-Rubin theorem's original paper. Also cited in §18.4 Examples 6–7 as the source for the binomial and Normal UMP claims."
  - id: "KAR1957"
    text: "Karlin, Samuel. 1957. \"Pólya Type Distributions, II.\" Annals of Mathematical Statistics 28 (2): 281–308."
    url: "https://doi.org/10.1214/aoms/1177706960"
    usedIn: "§18.3 Remark 7 — coinage of the monotone-likelihood-ratio definition."
  - id: "WIL1938"
    text: "Wilks, Samuel S. 1938. \"The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses.\" Annals of Mathematical Statistics 9 (1): 60–62."
    url: "https://doi.org/10.1214/aoms/1177732360"
    usedIn: "§18.6 Theorem 4 + Proof 3 attribution — Wilks' theorem's original paper."
  - id: "RAO1948"
    text: "Rao, C. Radhakrishna. 1948. \"Large Sample Tests of Statistical Hypotheses Concerning Several Parameters with Applications to Problems of Estimation.\" Mathematical Proceedings of the Cambridge Philosophical Society 44 (1): 50–57."
    url: "https://doi.org/10.1017/S0305004100023987"
    usedIn: "§18.7 Theorem 5 — Rao's score test and the three-tests equivalence. Already cited in Topic 17 for the score test; Topic 18 extends the citation to cover asymptotic equivalence."
  - id: "FER1967"
    text: "Ferguson, Thomas S. 1967. Mathematical Statistics: A Decision Theoretic Approach. New York: Academic Press."
    url: "https://www.worldcat.org/title/mathematical-statistics-a-decision-theoretic-approach/oclc/490902"
    usedIn: "§18.1 Remark 1 — decision-theoretic framing; §18.5 Remark 11 — the LRT as a generalized risk procedure."
  - id: "VAN1998"
    text: "van der Vaart, Aad W. 1998. Asymptotic Statistics. Cambridge Series in Statistical and Probabilistic Mathematics 3. Cambridge: Cambridge University Press."
    url: "https://doi.org/10.1017/CBO9780511802256"
    usedIn: "§18.6 Remark 12 — pointer to the LAN / Le Cam rigorous treatment of Wilks in Ch. 16. §18.7 — modern reference for the three-tests equivalence."
  - id: "BUS1982"
    text: "Buse, A. 1982. \"The Likelihood Ratio, Wald, and Lagrange Multiplier Tests: An Expository Note.\" The American Statistician 36 (3a): 153–57."
    url: "https://doi.org/10.1080/00031305.1982.10482817"
    usedIn: "§18.7 Remark 15 — the canonical pedagogical reference for first-order equivalence, pitched at the econometrics-oriented reader."
  - id: "LEH2005"
    text: "Lehmann, Erich L., and Joseph P. Romano. 2005. Testing Statistical Hypotheses. 3rd ed. Springer Texts in Statistics. New York: Springer."
    url: "https://doi.org/10.1007/0-387-27605-X"
    usedIn: "Already cited in Topic 17. In Topic 18: §18.3 Remark 6 (two-sided UMP unbiased); §18.4 Example 8, Remark 8 (scale-invariant t-test UMP; two-sided z-test UMP unbiased); §18.5 Remark 10 (profile likelihood); §18.6 Remarks 13–14 (vector-θ Wilks; non-regular cases); §18.10 Remark 21 (scope-boundary pointers)."
  - id: "CAS2002"
    text: "Casella, George, and Roger L. Berger. 2002. Statistical Inference. 2nd ed. Pacific Grove, CA: Duxbury."
    url: "https://www.cengage.com/c/statistical-inference-2e-casella"
    usedIn: "Already cited in Topic 17. In Topic 18: §18.2 Example 2 alignment; §18.4 Example 6 alignment — textbook anchor for worked examples."
---
```

### 2.2 Length target

Target: **60-minute read** — 10 sections, 4 full proofs, three required interactive components (plus one optional), ~10,000 words. Topic 18 is shorter than Topic 17 (55 min / 12,000 words) because it is proof-heavy and catalog-light — fewer worked test families, more depth per theorem.

---

## 3. Content Structure

### 3.1 Section Map

| § | Title | Formal elements | Figure | Interactive component |
|---|-------|----------------|--------|---------------------|
| 18.1 | "From Framework to Optimality" | Remarks 1–3 (optimality vs validity; four-pillar preview; scope boundary) | `18-optimality-motivation.png` | — |
| 18.2 | "The Neyman-Pearson Lemma" | Definitions 1–2 (MP test; simple-vs-simple LR), Theorem 1 (NP lemma), **Proof 1**, Examples 1–3 (Bernoulli, Normal, Exponential), Remarks 4–5 (randomization; template vs usable test) | `18-neyman-pearson-regions.png` | `NPLemmaVisualizer` |
| 18.3 | "Monotone Likelihood Ratio & Karlin-Rubin" | Definitions 3–4 (MLR; UMP), Theorem 2 (Karlin-Rubin), **Proof 2**, Examples 4–5 (exp family has MLR; Uniform has MLR), Remarks 6–7 (two-sided UMP nonexistence; KAR1957 history) | `18-mlr-illustration.png` | `KarlinRubinUMP` |
| 18.4 | "UMP in Action" | Examples 6–8 (binomial UMP; Normal z-test UMP; Normal t-test UMP-invariant), Remarks 8–9 (two-sided z not UMP; why this motivates §18.5) | `18-karlin-rubin-ump.png`, `18-binomial-ump-exact.png` | — |
| 18.5 | "The Likelihood-Ratio Principle for Composite H₀" | Definitions 5–6 (generalized LR; LRT rejection), Theorem 3 (invariance — stated; proved in §18.8), Examples 9–10 (Normal t-test as LRT; χ² variance test as LRT), Remarks 10–11 (profile likelihood preview; LRT as general-purpose replacement for UMP) | — | — |
| 18.6 | "Wilks' Theorem" | Theorem 4 (Wilks), **Proof 3** (8 steps; ~25–30 MDX lines), Example 11 (Bernoulli: explicit −2 log Λ vs Wald vs Score), Remarks 12–14 (regularity conditions; vector-θ k-df extension; non-regular failure modes) | `18-wilks-convergence.png`, `18-wilks-proof-geometry.png` | `WilksConvergence` |
| 18.7 | "Wald, Score, LRT: First-Order Equivalence" | Theorem 5 (three-tests equivalence), **Proof 4**, Example 12 (Bernoulli MC matches), Remark 15 (Wald vs Score vs LRT pedagogical distinction) | `18-three-tests-histograms.png` | — |
| 18.8 | "Finite-Sample Divergence & Reparameterization Invariance" | Theorem 6 (LRT invariant under reparameterization; Wald not), Examples 13–14 (Bernoulli p vs logit; MC at n=20), Remarks 16–17 (Wald boundary pathology; GLM link-choice invariance) | `18-reparam-invariance.png` | `ReparamInvariance` (optional) |
| 18.9 | "Power Envelope and Local Power" | Definition 7 (non-central χ²), Theorem 7 (local power → χ²₁(h² I)), Example 15 (Bernoulli power at h ∈ {0,1,2,3}), Remarks 18–20 (CRLB as envelope; asymptotic efficiency; ARE pointer) | `18-local-power-envelope.png` | — |
| 18.10 | "Limitations & Forward Look" | Remarks 21–25 (what's not covered; CI duality preview; multiple-testing preview; cheat sheet; Track 6/7/8 forward pointers) | — | — |

**Total: 7 Definitions + 7 Theorems + 4 Full Proofs + 15 Examples + 25 Remarks = 58 formal elements.**

### 3.2 Opening paragraph (§18.1)

Use this verbatim as the opening of §18.1 (Jon approved as Option B):

> Two tests have the same size. Both reject the null 5% of the time when the null is true. One rejects a true effect of size δ = 0.3 with probability 0.4; the other rejects with probability 0.6. Which should we use? Topic 17 built a framework for valid testing but left this question open. This topic answers it: we characterize when a uniformly most powerful test exists (Neyman-Pearson, Karlin-Rubin), construct the likelihood-ratio test as a general-purpose procedure when it doesn't, and prove that the three asymptotic tests — Wald, Score, LRT — agree to first order under the null.

The opening paragraph is followed by a figure (`18-optimality-motivation.png`) and the three Remarks.

### 3.3 Full text of Proof 1 (Neyman-Pearson lemma)

**Setting.** Let X be a random vector with density f(x; θ) (with respect to some common σ-finite measure μ). Test H₀: θ = θ₀ vs H₁: θ = θ₁, both simple. For a test φ(x) ∈ [0, 1] (the probability of rejecting H₀ at observation x), define size E_{θ₀}[φ(X)] and power E_{θ₁}[φ(X)].

**Claim.** The Neyman-Pearson test

$$\varphi^*(x) \;=\; \begin{cases} 1 & \text{if } f(x; \theta_1) > k \cdot f(x; \theta_0) \\ 0 & \text{if } f(x; \theta_1) < k \cdot f(x; \theta_0) \end{cases}$$

(with arbitrary behavior on the boundary, randomized if needed to achieve exact size α) is most powerful level-α among all tests with size ≤ α.

**Proof.** Let φ be any test with E_{θ₀}[φ(X)] ≤ α. We show E_{θ₁}[φ*(X)] ≥ E_{θ₁}[φ(X)].

**Step 1.** Consider the pointwise inequality

$$\bigl(\varphi^*(x) - \varphi(x)\bigr) \cdot \bigl(f(x; \theta_1) - k \cdot f(x; \theta_0)\bigr) \;\geq\; 0 \quad \text{for all } x.$$

To verify: at any x with f(x; θ₁) > k f(x; θ₀), we have φ*(x) = 1 ≥ φ(x), so both factors are ≥ 0. At any x with f(x; θ₁) < k f(x; θ₀), we have φ*(x) = 0 ≤ φ(x), so both factors are ≤ 0. Either way, the product is ≥ 0.

**Step 2.** Integrate the inequality with respect to μ:

$$\int \bigl(\varphi^*(x) - \varphi(x)\bigr)\bigl(f(x; \theta_1) - k f(x; \theta_0)\bigr) \, d\mu(x) \;\geq\; 0.$$

Expanding,

$$\int \varphi^*(x) f(x; \theta_1) \, d\mu - \int \varphi(x) f(x; \theta_1) \, d\mu \;\geq\; k \left( \int \varphi^*(x) f(x; \theta_0) \, d\mu - \int \varphi(x) f(x; \theta_0) \, d\mu \right).$$

In expectation notation,

$$E_{\theta_1}[\varphi^*(X)] - E_{\theta_1}[\varphi(X)] \;\geq\; k \bigl( E_{\theta_0}[\varphi^*(X)] - E_{\theta_0}[\varphi(X)] \bigr).$$

**Step 3.** By hypothesis, φ has size ≤ α, and φ* has size α (chosen that way). Therefore E_{θ₀}[φ*(X)] − E_{θ₀}[φ(X)] ≥ 0, and multiplying by k ≥ 0 preserves the inequality:

$$E_{\theta_1}[\varphi^*(X)] - E_{\theta_1}[\varphi(X)] \;\geq\; 0.$$

Equivalently, the NP test has power at least as large as any other level-α test. ∎ — using NEY1933

### 3.4 Full text of Proof 2 (Karlin-Rubin)

**Setting.** Let {f(·; θ) : θ ∈ Θ ⊆ ℝ} have monotone likelihood ratio in T(X) (Definition 3). Test H₀: θ ≤ θ₀ vs H₁: θ > θ₀. Let c be the critical value such that P_{θ₀}(T(X) > c) = α (with randomization at T = c if T is discrete and the exact size cannot be hit).

**Claim.** The test φ(x) = 1{T(x) > c} is UMP level α.

**Proof.** We prove two things: (i) φ has size ≤ α over the composite null, and (ii) φ is MP at every θ₁ > θ₀.

**Step 1 (size over composite H₀).** By MLR, for any θ < θ₀,

$$\frac{f(x; \theta_0)}{f(x; \theta)} \text{ is non-decreasing in } T(x).$$

This means the NP test of "θ vs θ₀" rejects for large T. The power at θ₀ of such a test — i.e., the size at θ of the NP test rejecting "θ = θ₀" — is higher than any other level-α test at θ₀, in particular it exceeds the size at θ. Thus P_{θ}(T(X) > c) ≤ P_{θ₀}(T(X) > c) = α for every θ < θ₀, and size over H₀: θ ≤ θ₀ is exactly α.

**Step 2 (MP at each θ₁ > θ₀).** Fix θ₁ > θ₀. By MLR,

$$\frac{f(x; \theta_1)}{f(x; \theta_0)} \text{ is non-decreasing in } T(x).$$

There exists a threshold κ such that f(x; θ₁) > κ f(x; θ₀) iff T(x) > c (or the MLR ratio exceeds κ at the same x-values). The Neyman-Pearson test of H₀: θ = θ₀ vs H₁: θ = θ₁ rejects iff f(x; θ₁) > κ f(x; θ₀) — which by MLR is iff T(x) > c. That is: the NP rejection region is exactly {T > c}.

Critically, the rejection region {T > c} does not depend on θ₁. The same test is the MP level-α test at every θ₁ > θ₀ simultaneously. This is the definition of UMP.

Combining Step 1 and Step 2: φ has size α on the composite null and is MP at every alternative. Therefore φ is UMP level α. ∎ — using KAR1956 and Theorem 1 (NP lemma)

### 3.5 Full text of Proof 3 (Wilks' theorem)

**Setting.** Let X₁, …, X_n be iid from {f(·; θ) : θ ∈ Θ ⊆ ℝ} with Θ open. Under H₀: θ = θ₀ with θ₀ in the interior of Θ, and standard regularity — the MLE is consistent and asymptotically normal per Topic 14 Thm 14.3; the Fisher information I(θ) is continuous and positive at θ₀; the third log-density derivative ∂³ log f / ∂θ³ is uniformly bounded in a neighborhood of θ₀ by a function M(x) with E_{θ₀}[M(X)] < ∞ — we prove

$$-2 \log \Lambda_n \;\xrightarrow{d}\; \chi^2_1 \quad \text{under } H_0,$$

where Λ_n = L(θ₀) / L(θ̂_n) and θ̂_n is the MLE.

**Proof.** The argument proceeds in 8 steps.

**Step 1 — Rewrite in log-likelihood form.** By definition of the log-likelihood ℓ(θ) = log L(θ),

$$-2 \log \Lambda_n \;=\; -2 \bigl[ \ell(\theta_0) - \ell(\hat\theta_n) \bigr] \;=\; 2 \bigl[ \ell(\hat\theta_n) - \ell(\theta_0) \bigr].$$

**Step 2 — Taylor expand ℓ(θ₀) around θ̂_n.** By Taylor's theorem with remainder, there exists ξ_n between θ₀ and θ̂_n such that

$$\ell(\theta_0) \;=\; \ell(\hat\theta_n) + \ell'(\hat\theta_n)(\theta_0 - \hat\theta_n) + \tfrac{1}{2}\ell''(\hat\theta_n)(\theta_0 - \hat\theta_n)^2 + R_n,$$

where R_n = (1/6) ℓ‴(ξ_n)(θ₀ − θ̂_n)³.

**Step 3 — First-order term vanishes.** Because θ̂_n is the MLE in the interior of Θ, the first-order condition ℓ′(θ̂_n) = 0 holds exactly (Topic 14 §14.6). Substituting into Step 2,

$$\ell(\theta_0) - \ell(\hat\theta_n) \;=\; \tfrac{1}{2}\ell''(\hat\theta_n)(\theta_0 - \hat\theta_n)^2 + R_n.$$

Therefore

$$-2\log\Lambda_n \;=\; -\ell''(\hat\theta_n)(\hat\theta_n - \theta_0)^2 - 2 R_n.$$

**Step 4 — Remainder control.** We show 2R_n = o_P(1). By the third-derivative hypothesis, there exists a neighborhood U of θ₀ and a function M(x) with E_{θ₀}[M(X)] < ∞ such that |ℓ‴(θ)| ≤ Σᵢ M(X_i) for all θ ∈ U. Since θ̂_n →_P θ₀ (Topic 14 Thm 14.2), we have ξ_n ∈ U with probability going to 1, and on that event

$$|R_n| \;\leq\; \tfrac{1}{6} \Bigl| \sum_{i=1}^n M(X_i) \Bigr| \cdot |\hat\theta_n - \theta_0|^3.$$

By the weak law of large numbers applied to M(X_i), (1/n) Σᵢ M(X_i) →_P E_{θ₀}[M(X)] < ∞. Therefore |Σᵢ M(X_i)| = O_P(n). By Topic 14 Thm 14.3, √n(θ̂_n − θ₀) = O_P(1), so |θ̂_n − θ₀|³ = O_P(n^{−3/2}). Combining,

$$|R_n| \;=\; O_P(n) \cdot O_P(n^{-3/2}) \;=\; O_P(n^{-1/2}) \;=\; o_P(1).$$

Thus 2R_n = o_P(1) as claimed.

**Step 5 — Rescale observed curvature to Fisher information.** Write −ℓ″(θ̂_n) = n · [−(1/n) ℓ″(θ̂_n)]. The bracketed quantity converges in probability to I(θ₀):

$$-\frac{1}{n} \ell''(\hat\theta_n) \;\xrightarrow{P}\; I(\theta_0).$$

This identity — "observed information at the MLE converges to Fisher information at θ₀" — is the same lemma used in the proof of Topic 14 Thm 14.3. It follows from the SLLN applied to the iid summands −∂²_θ log f(X_i; θ), continuity of I(·) at θ₀, and consistency of θ̂_n (Topic 14 §14.7). We cite Topic 14's proof of Thm 14.3 for the argument.

**Figure callout (inline here).** Figure 7 (`18-wilks-proof-geometry.png`) visualizes the geometry of Step 3: the drop ℓ(θ̂_n) − ℓ(θ₀) is exactly the quadratic form ½ · n · I(θ₀) · (θ̂_n − θ₀)² modulo o_P(1) — the log-likelihood looks quadratic in a √n-shrinking neighborhood of its peak, and Wilks' theorem is the χ²₁ limit of that quadratic form. The figure shows ℓ(θ) overlaid with its quadratic approximation at θ̂_n (convex-up parabola) and the horizontal line at ℓ(θ₀); the vertical distance between them is (minus half of) −2 log Λ_n.

**Step 6 — Invoke MLE asymptotic normality.** By Topic 14 Thm 14.3,

$$\sqrt n (\hat\theta_n - \theta_0) \;\xrightarrow{d}\; \mathcal{N}(0, I(\theta_0)^{-1}) \quad \text{under } H_0.$$

Equivalently, √[n I(θ₀)] · (θ̂_n − θ₀) →_d N(0, 1).

**Step 7 — Continuous mapping (square).** Let Z_n = √[n I(θ₀)] · (θ̂_n − θ₀). Then Z_n →_d Z ~ N(0, 1). By the continuous mapping theorem (Topic 9), Z²_n →_d Z² ~ χ²₁. Note that Z²_n = n I(θ₀)(θ̂_n − θ₀)².

**Step 8 — Combine via Slutsky.** From Step 3 with Step 5,

$$-2 \log \Lambda_n \;=\; -\ell''(\hat\theta_n) (\hat\theta_n - \theta_0)^2 - 2 R_n \;=\; n \cdot \Bigl[-\tfrac{1}{n}\ell''(\hat\theta_n)\Bigr] \cdot (\hat\theta_n - \theta_0)^2 - 2 R_n.$$

Rewrite the leading term as Z²_n · [ −(1/n)ℓ″(θ̂_n) / I(θ₀) ]. The ratio converges in probability to 1 by Step 5, and Z²_n →_d χ²₁ by Step 7. By Slutsky's theorem, the product converges in distribution to χ²₁. The remainder o_P(1) does not affect the distributional limit, and we conclude

$$-2 \log \Lambda_n \;\xrightarrow{d}\; \chi^2_1 \quad \text{under } H_0. \qquad \blacksquare$$

— using Topic 14 Thm 14.3, Topic 9 (continuous mapping, Slutsky), WIL1938

### 3.6 Full text of Proof 4 (three-tests equivalence)

**Setting.** Same regularity as Wilks (Proof 3). Under H₀: θ = θ₀, the three asymptotic test statistics are defined as in Topic 17 §17.9:

$$W_n = n (\hat\theta_n - \theta_0)^2 I(\hat\theta_n), \qquad S_n = \frac{U(\theta_0)^2}{n I(\theta_0)}, \qquad -2 \log \Lambda_n = 2[\ell(\hat\theta_n) - \ell(\theta_0)],$$

where U(θ) = ℓ′(θ) is the score function.

**Claim.** Under H₀ and regularity, W_n − (−2 log Λ_n) = O_P(n^{−1/2}) and S_n − (−2 log Λ_n) = O_P(n^{−1/2}). In particular, all three statistics have the same asymptotic distribution χ²₁ under H₀.

**Proof.** We derive a common quadratic expansion for all three and compare.

**Step 1 (LRT).** From Proof 3 Step 8, −2 log Λ_n = n · I(θ₀) · (θ̂_n − θ₀)² + o_P(1). Equivalently:

$$-2 \log \Lambda_n \;=\; n \cdot I(\theta_0) \cdot (\hat\theta_n - \theta_0)^2 + o_P(1).$$

**Step 2 (Wald).** By consistency of θ̂_n and continuity of I(·), I(θ̂_n) = I(θ₀) + o_P(1). Therefore

$$W_n \;=\; n (\hat\theta_n - \theta_0)^2 \bigl[I(\theta_0) + o_P(1)\bigr] \;=\; n I(\theta_0)(\hat\theta_n - \theta_0)^2 + o_P(1),$$

using the fact that n(θ̂_n − θ₀)² = O_P(1) by Topic 14 Thm 14.3.

**Step 3 (Score).** Taylor expand U(θ₀) around θ̂_n. Since U(θ̂_n) = ℓ′(θ̂_n) = 0 (MLE FOC),

$$U(\theta_0) \;=\; U(\hat\theta_n) + U'(\tilde\theta_n)(\theta_0 - \hat\theta_n) \;=\; \ell''(\tilde\theta_n)(\theta_0 - \hat\theta_n)$$

for some θ̃_n between θ₀ and θ̂_n. By the same lemma as Proof 3 Step 5, (1/n)(−ℓ″(θ̃_n)) →_P I(θ₀), so −ℓ″(θ̃_n) = n I(θ₀) + o_P(n). Therefore

$$U(\theta_0) \;=\; -\ell''(\tilde\theta_n)(\hat\theta_n - \theta_0) \;=\; [n I(\theta_0) + o_P(n)](\hat\theta_n - \theta_0).$$

Squaring and dividing by n I(θ₀):

$$S_n \;=\; \frac{U(\theta_0)^2}{n I(\theta_0)} \;=\; n I(\theta_0)(\hat\theta_n - \theta_0)^2 + o_P(1).$$

**Step 4 — Compare.** From Steps 1–3, all three statistics equal n I(θ₀)(θ̂_n − θ₀)² + o_P(1). Pairwise differences are o_P(1), and a finer analysis (tracking the n^{−1/2} terms in each o_P(1)) shows they are O_P(n^{−1/2}). In particular, all three converge in distribution to χ²₁ under H₀. ∎ — using Topic 14 Thm 14.3, Proof 3, RAO1948

### 3.7 Remark on proof style & KaTeX

All four proofs above are written in separate `$$...$$` blocks with prose connectors between them. **Do not use `\begin{aligned}`**. Do not use `\begin{array}`. Every multi-line derivation should be broken into separate display math blocks with a prose sentence between each. Inline math uses `$...$`. See Appendix A for the complete KaTeX constraint list.

---

## 4. Static Images

**Directory:** `public/images/topics/likelihood-ratio-tests-and-np/`

Run the notebook to generate these figures. The notebook is seeded (`np.random.seed(42)`) for reproducibility.

### Figure manifest

| # | Filename | § | Dimensions | Description |
|---|----------|---|-----------|-------------|
| 1 | `18-optimality-motivation.png` | §18.1 | 1400 × 500 | 2-panel. Left: two overlaid Normal-mean power curves at the same size α = 0.05 but different test statistics — one curve strictly dominates the other in the right tail. Right: a vertical line at δ = 0.3 highlighting the 0.4-vs-0.6 power gap from the opening paragraph. Title: "Two level-α tests; one dominates. Which?" |
| 2 | `18-neyman-pearson-regions.png` | §18.2 | 1400 × 500 | 2-panel. Left: densities f(x; θ₀) and f(x; θ₁) (Normal, μ₀ = 0, μ₁ = 1, σ = 1) with the NP rejection region {x : f₁(x) ≥ k f₀(x)} shaded. Right: size and power as shaded tail areas — size in the left panel's right tail (Type I), power as the same region under f₁. |
| 3 | `18-mlr-illustration.png` | §18.3 | 1400 × 1000 | 4-panel grid. Each panel plots log(f(x; θ₂) / f(x; θ₁)) as a function of T(x) for: top-left Bernoulli (T = ΣXᵢ), top-right Normal mean (T = x̄), bottom-left Exponential rate (T = ΣXᵢ), bottom-right Uniform(0, θ) (T = max Xᵢ). Each line is visibly monotone (non-decreasing). |
| 4 | `18-karlin-rubin-ump.png` | §18.3, §18.4 | 1400 × 500 | 2-panel. Left: UMP power curve for the Normal one-sided z-test (H₀: μ ≤ 0, H₁: μ > 0), plotted as β(μ) with μ from −2 to 3, showing β(0) = 0.05 and β(μ) increasing monotonically. Right: the Normal sampling distribution of x̄ at μ = 0 with the UMP rejection region {x̄ > c} shaded — c = 1.645/√n for n = 25. |
| 5 | `18-binomial-ump-exact.png` | §18.4 | 1400 × 500 | 2-panel. Left: exact UMP rejection boundary (integer threshold on ΣXᵢ) vs Normal approximation boundary at n = 20, 50, 100 — showing discrepancy at small n due to discreteness. Right: exact size vs Normal-approximation size at each boundary — exact size is conservative (< 0.05), approximate is biased. |
| 6 | `18-wilks-convergence.png` | §18.6 | 1400 × 1000 | 4-panel grid. Each panel shows an MC histogram of −2 log Λ_n under H₀ for Bernoulli(p₀ = 0.5) at n ∈ {10, 50, 200, 1000}, M = 5000. χ²₁ density overlaid. Convergence is visible as the histogram approaches the density with increasing n. Empirical mean / 95th percentile annotated. |
| 7 | `18-wilks-proof-geometry.png` | §18.6 | 1200 × 600 | Single panel. Log-likelihood ℓ(θ) for Bernoulli(p = 0.5) at n = 100 (realized sample), plotted vs θ ∈ [0.3, 0.7]. Overlaid: the quadratic Taylor approximation ℓ(θ̂_n) + ½ ℓ″(θ̂_n)(θ − θ̂_n)², tangent at θ̂_n. Vertical dashed line at θ₀ = 0.5; horizontal arrows marking the drop ℓ(θ̂_n) − ℓ(θ₀) and the quadratic value at θ₀. Annotated: "−2 log Λ_n = 2 · [drop] ≈ n I(θ₀)(θ̂_n − θ₀)²". |
| 8 | `18-three-tests-histograms.png` | §18.7 | 1400 × 500 | 2-panel. Left: Wald (amber), Score (purple), LRT (green) histograms overlaid under H₀, Bernoulli p₀ = 0.5, n = 100, M = 5000. χ²₁ density overlaid in grey. Right: empirical rejection rate at α = 0.05 for each test — all three ≈ 0.05 within MC error. |
| 9 | `18-reparam-invariance.png` | §18.8 | 1400 × 500 | 2-panel. Left: Wald p-value for Bernoulli H₀: p = 0.5 at observed p̂ = 0.3, n = 50 — computed in the p-parameterization (gives one p-value) and in the logit-η parameterization (gives a different p-value). Numerical values annotated. Right: LRT p-value computed both ways — identical (same numerical value annotated). |
| 10 | `18-local-power-envelope.png` | §18.9 | 1400 × 500 | 2-panel. Left: local power curves β(h) for W_n, S_n, −2 log Λ_n under local alternatives θ_n = θ₀ + h/√n with θ₀ = 0.5 (Bernoulli), h ∈ [0, 4]. Non-central χ²₁(h² I(θ₀)) envelope overlaid — all three tests hug the envelope. Right: same for Normal mean, σ = 1, θ₀ = 0; same qualitative picture. |

### Notebook structure

The notebook `18_likelihood_ratio_tests.ipynb` should have these code cells:

1. **Imports + seed.** `numpy`, `scipy.stats`, `matplotlib`, `seed = 42`.
2. **Palette + matplotlib defaults.** Reuse the Topic 17 palette: amber `#d97706` (Wald), purple `#7c3aed` (Score), green `#10b981` (LRT), grey `#475569` (reference χ²); title size 14, axis label size 12, tick size 10, figure DPI 150, bbox_inches='tight' on save.
3. **Figure 1** (optimality motivation) — ~40 lines.
4. **Figure 2** (NP regions) — ~50 lines.
5. **Figure 3** (MLR illustration, 4 panels) — ~80 lines.
6. **Figure 4** (Karlin-Rubin UMP) — ~50 lines.
7. **Figure 5** (binomial exact UMP vs Normal approximation) — ~60 lines.
8. **Figure 6** (Wilks convergence, 4 panels, MC) — ~80 lines.
9. **Figure 7** (Wilks proof geometry) — ~50 lines. This is the inline-in-proof figure.
10. **Figure 8** (three tests histograms) — ~60 lines.
11. **Figure 9** (reparam invariance) — ~50 lines.
12. **Figure 10** (local power envelope, 2 panels) — ~80 lines.
13. **Markdown cell** — "All figures saved to `public/images/topics/likelihood-ratio-tests-and-np/`. Run the cells above to regenerate."

---

## 5. Interactive Components

Three required components plus one optional. All follow Topic 17's established patterns: React + Tailwind, `client:visible` directive (not `client:load`), SVG/D3.js only, CSS custom properties from `viz/shared/colorScales.ts`, responsive to 375px width.

### Component 1: NPLemmaVisualizer

**File:** `src/components/viz/NPLemmaVisualizer.tsx`

**Purpose:** Make the Neyman-Pearson lemma concrete by visualizing the density-ratio rejection region and showing how the critical value k trades off size vs power.

**Interactions:**
- Dropdown: scenario (Normal-vs-Normal with different means; Bernoulli-vs-Bernoulli; Exponential-vs-Exponential). Default: Normal, μ₀ = 0, μ₁ = 1, σ = 1.
- Slider: θ₁ (alternative parameter, scenario-dependent).
- Slider: critical value k for the likelihood ratio; or equivalently, the threshold c on the sufficient statistic T with an internal conversion.
- Radio: rejection-region display — "on raw x" (shaded regions under f₀ and f₁) vs "on LR surface" (plot of Λ(x) with horizontal line at k, shaded x-values where Λ(x) ≥ k).
- Left panel: the two densities f(x; θ₀) and f(x; θ₁) overlaid with the NP rejection region shaded on both (using translucent amber for the region under f₀ = size, translucent green for the region under f₁ = power).
- Center panel: size (Type I error) and power readouts, updated live as k moves.
- Right panel: the log-likelihood-ratio log Λ(x) plotted as a function of x, with a horizontal line at log k. Shaded x-values are where log Λ(x) ≥ log k.

**Data:** `npLemmaPresets` from `likelihood-ratio-tests-data.ts`.

**Uses from `testing.ts`:** `logLikelihoodRatio`, `npCriticalValue`, `standardNormalCDF`.

**Uses from `distributions.ts` (Topic 2):** `normalPDF`, `bernoulliPMF`, `exponentialPDF`.

**Mobile behavior:** left panel on top, center and right panels stacked below.

### Component 2: KarlinRubinUMP

**File:** `src/components/viz/KarlinRubinUMP.tsx`

**Purpose:** Show MLR in exponential families and the resulting UMP rejection boundary. Make visible that the same rejection region {T > c} works at every alternative — the structural property that defines UMP.

**Interactions:**
- Dropdown: exponential-family scenario (Bernoulli, Normal mean σ known, Poisson rate, Exponential rate). Default: Bernoulli.
- Slider: θ₀ (null value).
- Slider: θ₁ (alternative; slider restricted to θ > θ₀).
- Slider: α (level; default 0.05).
- Slider: n (sample size; default 50).
- Left panel: log-likelihood ratio log[f(x; θ₁)/f(x; θ₀)] plotted as a function of T(x). The line is visibly non-decreasing (monotone), confirming MLR. A horizontal line at the NP critical value; x-values to the right are the NP rejection region.
- Center panel: power function β(θ) for θ ∈ [θ₀ − δ, θ₀ + δ] (δ chosen for visibility), with the UMP boundary c held fixed. The plot shows a monotone increase in θ; the test is UMP because this curve lies above any other level-α test's power curve.
- Right panel: tidy formula display — "UMP rejection region: {T(X) > c}; threshold c solves P_{θ₀}(T > c) = α." For the current scenario, the Tform and exact threshold are shown.

**Data:** `karlinRubinPresets` from `likelihood-ratio-tests-data.ts`.

**Uses from `testing.ts`:** `umpOneSidedBoundary`, `logLikelihoodRatio`, `binomialCDF`, `poissonCDF`, `standardNormalCDF`.

**Mobile behavior:** left panel on top, center and right panels stacked below.

### Component 3: WilksConvergence

**File:** `src/components/viz/WilksConvergence.tsx`

**Purpose:** Visualize the convergence of −2 log Λ_n to χ²_k as n increases. This is the single most important component of Topic 18 — the "Wilks' theorem in pixels" analog of Topic 17's `TTestBasuFoundation`.

**Interactions:**
- Dropdown: scenario (Bernoulli p₀, Normal mean σ known, Normal mean σ unknown, Poisson rate). Default: Bernoulli, p₀ = 0.5.
- Slider: θ₀ (null value).
- Slider: n (sample size, 10 to 1000, log-scale).
- Slider: M (MC replications, 500 to 10000).
- Button: "Re-sample" — re-runs the MC with a new seed.
- Left panel: histogram of −2 log Λ_n across M MC runs under H₀, with χ²₁ density overlaid. As n grows, the histogram visibly approaches the density.
- Center panel: empirical moment readouts (mean, variance, 95th percentile) vs. theoretical χ²₁ values (1, 2, 3.84). Color-coded green when within 5%, amber when within 10%, red otherwise.
- Right panel: 4-panel mini-grid showing the same convergence at n ∈ {10, 50, 200, 1000} simultaneously — a compressed version of `18-wilks-convergence.png`.

**Data:** `wilksConvergencePresets` from `likelihood-ratio-tests-data.ts`.

**Uses from `testing.ts`:** `wilksSimulate`, `chiSquaredPDF`.

**Mobile behavior:** left panel on top, center panel below, right panel (4 mini-panels) at bottom.

**Performance note:** MC at n = 1000, M = 10000 is 10M total samples. Use `useMemo` aggressively. Warn at > 5000 MC replications that the run may take 1–2 seconds.

### Component 4 (optional): ReparamInvariance

**File:** `src/components/viz/ReparamInvariance.tsx`

**Purpose:** Demonstrate Theorem 6 (§18.8) — LRT invariant, Wald not — with a live Bernoulli example.

**Interactions:**
- Slider: p̂ (observed sample proportion).
- Slider: p₀ (null value).
- Slider: n (sample size).
- Dropdown: reparameterization (logit, log, probit).
- Left panel: Wald p-value in the p-parameterization and the η-parameterization, shown as two numerical readouts with the difference highlighted.
- Right panel: LRT p-value in both parameterizations — identical numerical value shown twice, confirming invariance.

**Data:** `reparamPresets` from `likelihood-ratio-tests-data.ts`.

**Uses from `testing.ts`:** `waldStatistic`, `lrtStatistic`, `chiSquaredCDF`.

**Mobile behavior:** left panel on top, right panel below.

**Decision:** build only if time permits. Topic 18 is acceptable with three components; this fourth is a nice-to-have.

### Implementation Notes

- **Architecture:** same as Topics 1–17. React + Tailwind, `client:visible` directive (**not** `client:load`).
- **SVG/D3.js for all charts.** No Chart.js, Plotly, Recharts, or other libraries.
- **CSS custom properties** from `viz/shared/colorScales.ts` for the Topic 17 palette (amber Wald, purple Score, green LRT).
- **Responsive:** all components tested at 375px width.
- **Client-side computation only.** No server-side calls.
- **`NPLemmaVisualizer` must be fast** — closed-form size/power for Normal and Bernoulli, no MC by default.
- **`WilksConvergence` uses MC** — memoize results with `useMemo`; warn at > 5000 replications.

---

## 6. Shared Module (`testing.ts` extension)

`testing.ts` is the Track 5 shared module, created in Topic 17 and extended here. The Topic 17 baseline includes the full Wald/Score/LRT trio for Bernoulli, Normal, and Poisson; the binomial exact test; analytic null distributions for z/t/χ²; a Monte Carlo p-value runner; and closed-form power calculators. Topic 18 adds the optimality machinery.

### 6.1 Full new-function manifest

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// TOPIC 18 EXTENSIONS TO testing.ts
// ═══════════════════════════════════════════════════════════════════════════

// ── Neyman-Pearson & likelihood ratios ──────────────────────────────────────

/** Log likelihood ratio log Λ(x) = log f(x;θ₁) − log f(x;θ₀) for iid samples.
 *  Returns log for numerical stability. Used by NPLemmaVisualizer. */
export function logLikelihoodRatio(
  family: AsymptoticTestFamily,
  data: number[], theta0: number, theta1: number, knownParam?: number
): number;

/** NP critical value: find threshold k such that P_{θ₀}(Λ ≥ k) = α.
 *  For MLR families this is equivalent to a threshold on the sufficient statistic T;
 *  returned as {threshold, Tform, exactSize, onT}. */
export function npCriticalValue(
  family: AsymptoticTestFamily,
  theta0: number, theta1: number, n: number, alpha: number,
  knownParam?: number
): { threshold: number; onT: boolean; Tform: string; exactSize: number };

// ── Karlin-Rubin / MLR one-sided UMP ────────────────────────────────────────

/** UMP one-sided rejection boundary for an MLR family.
 *  Returns the threshold on the sufficient statistic T and the achieved exact size.
 *  For discrete families (Bernoulli, Poisson) exactSize ≤ alpha (conservative). */
export function umpOneSidedBoundary(
  family: AsymptoticTestFamily,
  theta0: number, n: number, alpha: number,
  side: 'left' | 'right', knownParam?: number
): { boundary: number; Tform: string; exactSize: number };

// ── Wilks' theorem: Monte Carlo under H₀ ────────────────────────────────────

/** Simulate −2 log Λ_n under H₀ for Wilks convergence visualization.
 *  Used by WilksConvergence component and §18.6 Example 11.
 *  Reuses lrtStatistic from Topic 17. */
export function wilksSimulate(
  family: AsymptoticTestFamily,
  theta0: number, n: number, M: number,
  knownParam?: number, seed?: number
): number[];

// ── Non-central χ² for local power ──────────────────────────────────────────

/** Non-central χ²_k(λ) CDF via series expansion.
 *  CDF(x; k, λ) = Σ_j e^{−λ/2} (λ/2)^j / j! · ChiSquaredCDF(x; k + 2j).
 *  Truncate series when term < 1e-12. */
export function nonCentralChiSquaredCDF(
  x: number, k: number, lambda: number
): number;

/** Non-central χ²_k(λ) PDF for density overlays. */
export function nonCentralChiSquaredPDF(
  x: number, k: number, lambda: number
): number;

// ── Local-power envelope ────────────────────────────────────────────────────

/** Local power: P(reject) when θ = θ₀ + h/√n, computed from
 *  χ²_1(h² I(θ₀)) at the α quantile of χ²_1.
 *  Returns 1 − nonCentralChiSquaredCDF(chi2Quantile(1−α, 1), 1, h² I(θ₀)). */
export function localPower(
  family: AsymptoticTestFamily,
  theta0: number, h: number, alpha: number, knownParam?: number
): number;
```

### 6.2 Console log tests to add (13 new)

Add to the existing Topic 17 test block in `src/lib/stat/testing.test.ts` or wherever Topic 17 placed its 19 tests:

1. `logLikelihoodRatio('bernoulli', [1,1,1,0,0], 0.3, 0.7)` — closed-form: 3·log(7/3) + 2·log(3/7) ≈ 0.8473; within 1e-6.
2. `logLikelihoodRatio('normal-mean-known-sigma', [0,0,0], 0, 1, 1)` — closed-form: 3·[−½·1²] = −1.5; exact.
3. `npCriticalValue('normal-mean-known-sigma', 0, 1, 25, 0.05, 1)` → `{threshold: 0.329, onT: true, Tform: 'xbar', exactSize: 0.05}`; threshold within 1e-3 of 1.645/√25 ≈ 0.329.
4. `umpOneSidedBoundary('bernoulli', 0.5, 20, 0.05, 'right')` → `{boundary: 15, Tform: 'ΣXᵢ', exactSize: 0.0207}` — matches Topic 17's `binomialExactRejectionBoundary` for cross-module sanity.
5. `umpOneSidedBoundary('poisson', 5, 30, 0.05, 'right')` → `{boundary: matches tabulated value, exactSize ≤ 0.05}`. Use scipy.stats.poisson ppf for the exact tabulated value.
6. `wilksSimulate('bernoulli', 0.5, 100, 2000, undefined, 42)` — empirical mean ∈ [0.9, 1.1]; 95th percentile ∈ [3.6, 4.1] (χ²₁ target: mean = 1, 95% = 3.84).
7. `wilksSimulate('normal-mean', 0, 200, 2000, 1, 42)` — same moment / quantile check.
8. `nonCentralChiSquaredCDF(3.84, 1, 0)` ≈ 0.95 — central case sanity (reduces to χ²₁ CDF).
9. `nonCentralChiSquaredCDF(3.84, 1, 4)` ≈ 0.485 — non-central at λ = 4; verify against scipy.stats.ncx2.cdf(3.84, 1, 4).
10. `nonCentralChiSquaredPDF(1, 1, 0)` ≈ 0.2420 — matches central χ²₁ PDF at x = 1.
11. `localPower('normal-mean-known-sigma', 0, 0, 0.05, 1)` ≈ 0.05 — power at null equals size (h = 0, non-centrality = 0).
12. `localPower('normal-mean-known-sigma', 0, 2, 0.05, 1)` ≈ 0.516 — non-centrality 4, I = 1, standard reference.
13. `localPower('bernoulli', 0.5, 2, 0.05)` ≈ 0.977 — I(0.5) = 4, non-centrality 16.

### 6.3 Implementation notes

- **Non-central χ² series.** The standard series expansion truncates when the Poisson weight e^{−λ/2}(λ/2)^j / j! drops below 1e-12. For λ up to ~50, truncation happens at j ≈ 100–150 terms. For larger λ, switch to a saddlepoint approximation or decline via boundary checks.
- **`wilksSimulate` seed handling.** Use a deterministic seeded PRNG (e.g., `seedrandom` or roll your own Mulberry32). Every component call with the same seed must produce the same MC stream.
- **`umpOneSidedBoundary` for continuous families** returns the `α`-quantile of the null distribution of T. For discrete families, it returns the smallest integer threshold such that the tail probability is ≤ α, and the achieved exact size (which will be < α).
- **No changes to existing Topic 17 functions.** The extension is strictly additive. The existing `waldStatistic`, `scoreStatistic`, `lrtStatistic`, `binomialExactPValue`, `monteCarloPValue`, etc., are unchanged.

---

## 7. Data Files

### 7.1 New file: `src/data/likelihood-ratio-tests-data.ts`

Follows the Topic 17 pattern (`hypothesis-testing-data.ts`). Contains preset scenarios for each interactive component plus the reference list.

```typescript
import type { RefItem, AsymptoticTestFamily } from '@/types';

export const npLemmaPresets = {
  normalVsNormal: {
    family: 'normal-mean-known-sigma' as AsymptoticTestFamily,
    theta0: 0, theta1: 1, sigma: 1, n: 25, alpha: 0.05,
    label: 'Normal, μ₀=0 vs μ₁=1, σ=1, n=25',
  },
  bernoulliVsBernoulli: {
    family: 'bernoulli' as AsymptoticTestFamily,
    theta0: 0.3, theta1: 0.5, n: 50, alpha: 0.05,
    label: 'Bernoulli, p₀=0.3 vs p₁=0.5, n=50',
  },
  exponentialVsExponential: {
    family: 'exponential' as AsymptoticTestFamily,
    theta0: 1.0, theta1: 0.5, n: 25, alpha: 0.05,
    label: 'Exponential, λ₀=1 vs λ₁=0.5, n=25',
  },
};

export const karlinRubinPresets = {
  bernoulli: { family: 'bernoulli', theta0: 0.5, theta1: 0.7, n: 50, alpha: 0.05 },
  normalMean: { family: 'normal-mean-known-sigma', theta0: 0, theta1: 0.5, sigma: 1, n: 25, alpha: 0.05 },
  poisson: { family: 'poisson', theta0: 2, theta1: 3, n: 30, alpha: 0.05 },
  exponential: { family: 'exponential', theta0: 1, theta1: 1.5, n: 25, alpha: 0.05 },
};

export const wilksConvergencePresets = {
  bernoulli: { family: 'bernoulli', theta0: 0.5 },
  normalMeanKnown: { family: 'normal-mean-known-sigma', theta0: 0, sigma: 1 },
  normalMeanUnknown: { family: 'normal-mean', theta0: 0 },
  poisson: { family: 'poisson', theta0: 2 },
};

export const reparamPresets = {
  bernoulliLogit: { p0: 0.5, phat: 0.3, n: 50, link: 'logit' },
  bernoulliProbit: { p0: 0.5, phat: 0.3, n: 50, link: 'probit' },
  bernoulliLog: { p0: 0.1, phat: 0.05, n: 100, link: 'log' },
};

export const topic18References: RefItem[] = [
  { id: 'NEY1933', shortText: 'Neyman & Pearson 1933' },
  { id: 'KAR1956', shortText: 'Karlin & Rubin 1956' },
  { id: 'KAR1957', shortText: 'Karlin 1957' },
  { id: 'WIL1938', shortText: 'Wilks 1938' },
  { id: 'RAO1948', shortText: 'Rao 1948' },
  { id: 'FER1967', shortText: 'Ferguson 1967' },
  { id: 'VAN1998', shortText: 'van der Vaart 1998' },
  { id: 'BUS1982', shortText: 'Buse 1982' },
  { id: 'LEH2005', shortText: 'Lehmann & Romano 2005' },
  { id: 'CAS2002', shortText: 'Casella & Berger 2002' },
];
```

### 7.2 No changes to `hypothesis-testing-data.ts`

Topic 17's data file is untouched. All new preset data lives in `likelihood-ratio-tests-data.ts`.

---

## 8. Cross-Reference Updates

### 8.1 Updates to `hypothesis-testing.mdx` (Topic 17)

These are the live "Topic 18 will..." pointers that need to become live links. Search for each phrase and update.

| Update | Source MDX location | Find (approximate; verify with grep) | Replace with |
|---|---|---|---|
| 1 | §17.2 Def 2 and §17.9 (preview of optimality theory) | "Topic 18" or "Neyman-Pearson" as plain text without a link | Wherever Topic 18 is referenced for NP lemma, add `[Likelihood-Ratio Tests & Neyman-Pearson](/topics/likelihood-ratio-tests-and-np)` |
| 2 | §17.4 Theorem 2 + proof sketch | "stated, proved in Topic 18 via MLR" | "stated, [proved in Topic 18 via MLR](/topics/likelihood-ratio-tests-and-np#section-18-3)" |
| 3 | §17.9 Def 15 | "(Wilks 1938; full proof in Topic 18)" | "(Wilks 1938; [full proof in Topic 18](/topics/likelihood-ratio-tests-and-np#section-18-6))" |
| 4 | §17.9 Thm 7 proof attribution | "Wilks' full proof of the LRT case is Topic 18's territory." | "Wilks' full proof of the LRT case is [Topic 18's territory](/topics/likelihood-ratio-tests-and-np#section-18-6)." |
| 5 | §17.9 Remark 10 | "Topic 18 treats the finite-sample divergence quantitatively." | "[Topic 18](/topics/likelihood-ratio-tests-and-np#section-18-8) treats the finite-sample divergence quantitatively." |
| 6 | §17.9 Remark 11 | "Topic 18 treats." or similar (verify verbatim) | `[Topic 18](/topics/likelihood-ratio-tests-and-np#section-18-8) treats.` |
| 7 | §17.4 Remark 5 | CRLB-as-power-envelope preview | Link the phrase "Topic 18" in the remark body to `/topics/likelihood-ratio-tests-and-np#section-18-9` |
| 8 | §17.6 Example 11 / §17.6 end-of-proof | "foreshadows Topic 18's UMP machinery (the binomial exact test is UMP one-sided for Bernoulli, which Topic 18 will prove via MLR)" | "foreshadows [Topic 18's UMP machinery](/topics/likelihood-ratio-tests-and-np#section-18-4) (the binomial exact test is UMP one-sided for Bernoulli, which Topic 18 proves via MLR)" |

**Grep command to verify complete coverage:**

```bash
grep -rn 'Topic 18\|likelihood-ratio-tests' --include='*.mdx' src/content/topics/ | grep -v '(coming soon)'
```

Should return exactly the updated references (no unmigrated "Topic 18" plain-text pointers). There should be **zero** "(coming soon)" markers pointing to Topic 18 after the update — because Topic 17 did not use "(coming soon)" for Topic 18 references; it used "Topic 18 will..." as plain text.

### 8.2 Updates to `curriculum-graph.json`

Mark `likelihood-ratio-tests-and-np` as published; add prerequisite edges.

```json
{
  "id": "likelihood-ratio-tests-and-np",
  "title": "Likelihood-Ratio Tests & Neyman-Pearson",
  "track": 5,
  "trackName": "Hypothesis Testing & Confidence",
  "topicNumber": 18,
  "status": "published",
  "prerequisites": [
    "hypothesis-testing",
    "maximum-likelihood",
    "point-estimation",
    "modes-of-convergence",
    "sufficient-statistics",
    "exponential-families",
    "central-limit-theorem"
  ]
}
```

### 8.3 Updates to `curriculum.ts`

Track 5 status: `2 of 4 published` (was `1 of 4` after Topic 17).

### 8.4 References spreadsheet updates

**Add 5 new entries** to `formalstatisticscitations.xlsx`:

| Short key | Used In Topics |
|---|---|
| KAR1956 | `likelihood-ratio-tests-and-np` |
| KAR1957 | `likelihood-ratio-tests-and-np` |
| FER1967 | `likelihood-ratio-tests-and-np` |
| VAN1998 | `likelihood-ratio-tests-and-np` |
| BUS1982 | `likelihood-ratio-tests-and-np` |

Each row populated with full Chicago 17th edition text and verified URL per §2.1 references block above.

**Update 5 existing entries** — append `likelihood-ratio-tests-and-np` to the "Used In Topics" column:

- `NEY1933` (was: `hypothesis-testing`) → `hypothesis-testing, likelihood-ratio-tests-and-np`
- `WIL1938` (was: `hypothesis-testing`) → `hypothesis-testing, likelihood-ratio-tests-and-np`
- `RAO1948` (was: `hypothesis-testing`) → `hypothesis-testing, likelihood-ratio-tests-and-np`
- `LEH2005` (was: `hypothesis-testing`) → `hypothesis-testing, likelihood-ratio-tests-and-np`
- `CAS2002` (was: `[prior topics], hypothesis-testing`) → `[prior topics], hypothesis-testing, likelihood-ratio-tests-and-np`

### 8.5 Sitemap / RSS

Verify `sitemap.xml` and `rss.xml` include the new `/topics/likelihood-ratio-tests-and-np` URL. These are auto-generated on build; no manual edits needed, but verify after deploy.

---

## 9. Verification Checklist

### Content

- [ ] `likelihood-ratio-tests-and-np.mdx` renders at `/topics/likelihood-ratio-tests-and-np` with no build errors
- [ ] All 10 static images present in `public/images/topics/likelihood-ratio-tests-and-np/`
- [ ] KaTeX renders correctly: all display equations, inline math, theorem blocks
- [ ] All 7 definitions, 7 theorems, 15 examples, 25 remarks render in styled blocks
- [ ] All 4 full proofs expand completely with no hand-waving (NP lemma, Karlin-Rubin, Wilks, three-tests equivalence)
- [ ] Proofs use separate `$$...$$` blocks with prose connectors — **NO** `\begin{aligned}` blocks and **NO** `\begin{array}` tables anywhere
- [ ] All section headings use the `18.X` prefix (§18.1 through §18.10)
- [ ] Notation: Λ_n for the LR statistic, −2 log Λ_n for the transformed version; θ̂_n for the MLE; I(θ) for Fisher information; W_n, S_n, −2 log Λ_n for Wald/Score/LRT; χ²_k for chi-squared with k degrees of freedom
- [ ] Citation style is **end-of-proof** (`∎ — using [...]`), applied uniformly across all 4 proofs
- [ ] Wilks Proof Step 4 is written out in full (remainder control with uniform third-derivative bound)
- [ ] Wilks Proof Step 5 cites Topic 14 for the observed-information-to-Fisher-information lemma (not re-derived)
- [ ] Figure 7 (`18-wilks-proof-geometry.png`) renders inline between Step 4 and Step 5 of Proof 3

### Proof-specific checks

- [ ] **Proof 1 (NP lemma)**: 3-step structure (pointwise inequality → integrate → rearrange using size constraint); all three display blocks render; final `∎ — using NEY1933`
- [ ] **Proof 2 (Karlin-Rubin)**: 2-step structure (size over composite H₀ → MP at each θ₁ > θ₀); cites Theorem 1 (NP lemma) and MLR (Definition 3); final `∎ — using KAR1956 and Theorem 1`
- [ ] **Proof 3 (Wilks)**: 8-step structure; Steps 4 and 5 specifically called out; figure callout at Step 4/5 boundary; final `∎ — using Topic 14 Thm 14.3, Topic 9, WIL1938`
- [ ] **Proof 4 (three-tests equivalence)**: 4-step structure with all three statistics reduced to n I(θ₀)(θ̂_n − θ₀)² + o_P(1); cites Proof 3; final `∎ — using Topic 14 Thm 14.3, Proof 3, RAO1948`

### Forward-promise fulfillment

- [ ] §18.2 delivers NP lemma with full proof — fulfills Topic 17 §17.2 / §17.9 forward promise #1
- [ ] §18.3 delivers Karlin-Rubin with full proof — fulfills Topic 17 §17.4 Thm 2 forward promise #2
- [ ] §18.6 delivers Wilks' theorem with full proof — fulfills Topic 17 §17.9 Thm 7 / Def 15 forward promise #3
- [ ] §18.9 delivers CRLB-as-power-envelope via non-central χ² — fulfills Topic 17 §17.4 Remark 5 forward promise #4
- [ ] §18.4 Example 6 delivers binomial exact as UMP — fulfills Topic 17 §17.6 Ex 11 forward promise #5
- [ ] §18.8 Example 14 delivers quantitative finite-sample divergence — fulfills Topic 17 §17.9 Remark 10 forward promise #6
- [ ] §18.8 Theorem 6 + Example 13 delivers reparameterization invariance concrete example — fulfills Topic 17 §17.9 Remark 11 forward promise #7

### Cross-reference updates

- [ ] `hypothesis-testing.mdx` §17.2 Def 2 / §17.9 "Topic 18" pointers now link to `/topics/likelihood-ratio-tests-and-np`
- [ ] `hypothesis-testing.mdx` §17.4 Thm 2 / proof sketch now links to `#section-18-3`
- [ ] `hypothesis-testing.mdx` §17.9 Def 15 Wilks pointer now links to `#section-18-6`
- [ ] `hypothesis-testing.mdx` §17.9 Thm 7 "Topic 18's territory" now links to `#section-18-6`
- [ ] `hypothesis-testing.mdx` §17.9 Remark 10 "Topic 18 treats" now links to `#section-18-8`
- [ ] `hypothesis-testing.mdx` §17.9 Remark 11 now links to `#section-18-8`
- [ ] `hypothesis-testing.mdx` §17.4 Remark 5 CRLB-envelope reference now links to `#section-18-9`
- [ ] `hypothesis-testing.mdx` §17.6 Ex 11 binomial-UMP reference now links to `#section-18-4`
- [ ] Grep verifies no plain-text "Topic 18" references remain in Topic 17 MDX

### References spreadsheet

- [ ] 5 new references added: KAR1956, KAR1957, FER1967, VAN1998, BUS1982 — all with verified URLs
- [ ] 5 existing references updated: NEY1933, WIL1938, RAO1948, LEH2005, CAS2002 — `likelihood-ratio-tests-and-np` appended to "Used In Topics"
- [ ] All 10 Topic 18 citations render in the in-topic references section with working `url:` fields

### Interactive Components

- [ ] `NPLemmaVisualizer` renders all 3 scenarios (Normal, Bernoulli, Exponential); density-ratio region shades correctly; size and power readouts update live with k slider
- [ ] `KarlinRubinUMP` renders all 4 exp-family scenarios; MLR line is visibly non-decreasing; UMP boundary c shown correctly on the LR curve
- [ ] `WilksConvergence` renders convergence at n ∈ {10, 50, 200, 1000}; empirical moments match χ²₁ theory within 5% at n ≥ 200
- [ ] (Optional) `ReparamInvariance` renders both Wald (different p-values) and LRT (identical p-values) correctly; toggle between logit, log, probit works

### `testing.ts` extension

- [ ] 7 new functions exported: `logLikelihoodRatio`, `npCriticalValue`, `umpOneSidedBoundary`, `wilksSimulate`, `nonCentralChiSquaredCDF`, `nonCentralChiSquaredPDF`, `localPower`
- [ ] All 13 new console-log tests pass
- [ ] No regressions in existing Topic 17 tests (19 tests from Topic 17 still pass)
- [ ] TypeScript compiles with no new errors

### Curriculum graph

- [ ] `curriculum-graph.json` includes `likelihood-ratio-tests-and-np` as published
- [ ] `curriculum.ts` Track 5 status: "in-progress (2 of 4)"
- [ ] Prerequisite edges render correctly on the curriculum graph UI
- [ ] Topic card displays "60 min" read time, "intermediate" difficulty

### Build and deploy

- [ ] `pnpm build` completes with zero errors and zero warnings
- [ ] Pagefind indexes the new topic (verify by searching "Wilks" on the deployed site)
- [ ] Deployed preview on Vercel renders all sections, figures, and components correctly
- [ ] Mobile viewport (375px) renders all components responsively

---

## 10. Build Order

Numbered steps for Claude Code. Follow in order — later steps depend on earlier ones.

1. **Create branch** `topic-18-lrt-np` from `main`. All changes on this branch.
2. **Copy notebook figures** to `public/images/topics/likelihood-ratio-tests-and-np/`. All 10 PNGs.
3. **Extend `src/lib/stat/testing.ts`** with the 7 new functions per §6. Write console-log tests for all 13 test cases; verify all pass.
4. **Create `src/data/likelihood-ratio-tests-data.ts`** with all presets from §7.1: `npLemmaPresets`, `karlinRubinPresets`, `wilksConvergencePresets`, `reparamPresets`, `topic18References`.
5. **Create `src/content/topics/likelihood-ratio-tests-and-np.mdx`** with full frontmatter (§2.1) and all markdown/LaTeX content. Include all formal-element blocks (7 definitions, 7 theorems, 4 full proofs, 15 examples, 25 remarks). Include all prose for the section opener (§3.2), Proofs 1–4 (§3.3–§3.6), and the cheat sheet / forward look of §18.10. No interactive components embedded yet.
6. **Build `NPLemmaVisualizer.tsx`** — density-ratio regions, draggable k, live size/power. Three scenarios (Normal, Bernoulli, Exponential).
7. **Build `KarlinRubinUMP.tsx`** — MLR line with monotonicity visible; UMP boundary on sufficient statistic T; four exp-family scenarios.
8. **Build `WilksConvergence.tsx`** — the flagship. MC histogram under H₀ with χ²₁ overlay; slider for n; 4-panel mini-grid at fixed n values. Memoize MC runs with `useMemo`.
9. **(Optional) Build `ReparamInvariance.tsx`** — Wald vs LRT p-values under reparameterization. Three link choices (logit, log, probit).
10. **Embed all components** in the MDX with `client:visible` directive (**not** `client:load`).
11. **Update `curriculum-graph.json`** — mark `likelihood-ratio-tests-and-np` as published; add prerequisite edges. Verify Track 5 shows 2/4 published.
12. **Update `curriculum.ts`** — Track 5 status "in-progress (2 of 4)".
13. **Update `hypothesis-testing.mdx`** per §8.1 — replace 8 plain-text "Topic 18" pointers with live anchor links. Run the grep from §8.1 to verify complete coverage.
14. **Update references spreadsheet** (`formalstatisticscitations.xlsx`) per §8.4 — add 5 new entries with full Chicago + URLs; update 5 existing entries' "Used In Topics" column.
15. **Run full verification checklist** (§9). Fix any failures.
16. **`pnpm build`** — verify zero errors. Deploy to Vercel preview. Manual QA on mobile and desktop.
17. **Merge to `main`** and deploy to production.

**Estimated implementation time for Claude Code:** 6–8 hours (notebook figure generation assumed already done by Jonathan).

---

## Appendix A: KaTeX Constraints

Same as Topics 1–17:

- **No `\begin{aligned}` blocks with `&` markers.** Multi-line derivations use separate `$$...$$` blocks with prose glue between lines. Topic 18 has four multi-line proofs: NP lemma's 3-step pointwise-then-integrate argument, Karlin-Rubin's 2-step size + MP argument, Wilks' 8-step Taylor expansion, and the three-tests equivalence's 4-step quadratic-collapse argument. Everyone uses separate display blocks with prose like "Taylor-expanding ℓ around θ̂_n:" and "By Slutsky's theorem applied to the product:" as connectors.
- **No `\begin{array}{c|cccc}` tables.** Use HTML tables or markdown tables for tabular content. The section map in §3.1 and the forward-promise table in §1.3 are both markdown tables. The cheat sheet in §18.10, Remark 24, should be a Markdown table.
- **Inline math** uses `$...$`. **Display math** uses `$$...$$` on its own line.
- **Test all LaTeX rendering in the dev server before committing.**

### Notation conventions (extending Topic 17)

- **LR statistic:** Λ_n for the generalized LR (Topic 17 §17.9 Def 15); −2 log Λ_n for the transformed version (used throughout §18.5–§18.9). In §18.2, use Λ(x) = f(x; θ₁) / f(x; θ₀) for the simple-vs-simple case and flag the notational distinction from the composite Λ_n in the remark following Def 2.
- **MLE:** θ̂_n with explicit subscript n; do not abbreviate to θ̂.
- **Fisher information:** I(θ); observed information −ℓ″(θ) or J_n(θ) when context makes the distinction important.
- **Score function:** U(θ) = ℓ′(θ) for scalar θ. Matches Topic 17 §17.9 Def 14.
- **Test statistics:** W_n (Wald), S_n (Score), −2 log Λ_n (LRT). Matches Topic 17.
- **Null-distribution arrows:** `\xrightarrow{d}` for convergence in distribution, `\xrightarrow{P}` for convergence in probability, `\xrightarrow{a.s.}` for almost-sure.
- **Sample size dependence:** always subscript n when varying (θ̂_n, W_n, etc.).
- **Residual / remainder:** R_n for the Taylor remainder in Proof 3 Step 2.
- **Non-central χ²:** χ²_k(λ) where k = degrees of freedom and λ = non-centrality.
- **Local alternative parameter:** h in θ_n = θ₀ + h/√n.
- **Indicators:** `\mathbf{1}\{A\}` for the indicator of event A.

---

## Appendix B: Design Decisions

1. **Ten sections (locked via Decision 6, scaffold Part 1).** Ten sections give the Wilks proof room to breathe at §18.6 (not buried at §18.8), keep the binomial-UMP payoff visible in a dedicated §18.4, and preserve a dedicated §18.10 for scope-boundary pointers. Compressing to 8 sections would force merging §18.7 (three-tests equivalence) into §18.6 — muddying the post-Wilks debrief arc — and would drop the §18.8 reparameterization-invariance treatment that Topic 17 §17.9 Remark 11 explicitly promised as a Topic 18 deliverable.

2. **Intermediate difficulty (locked via Decision 1).** Every proof in Topic 18 — NP lemma (indicator algebra), Karlin-Rubin (NP + MLR), Wilks (Topic 14 Thm 14.3 + classical Taylor + continuous mapping), three-tests equivalence (Taylor + Slutsky) — uses machinery already established in Topics 9, 14, and 17. Wilks is the longest proof on the site because of quadratic-expansion bookkeeping, not because of new analytical abstraction. Flipping to "advanced" would imply net new analytical registers (LAN, empirical processes, martingales) — none of which appear. Consistency with Topics 9–17's "intermediate" labeling is preserved.

3. **Classical Taylor-expansion Wilks proof (locked via Decision 2).** The 8-step proof (§3.5) uses Topic 14's classical Taylor register without modification. The Le Cam / LAN / empirical-process treatment is cited once in §18.6 Remark 12 via VAN1998 §16. Advantages: (a) the proof consumes existing Topic 14 machinery; (b) it matches the intermediate-difficulty register; (c) the Step-4 remainder-control argument is genuinely pedagogically valuable — the one place in the curriculum where O_P / o_P algebra is used to control a third-derivative remainder in a non-trivial setting.

4. **Scalar θ only (locked via Decision 3).** Every proof is for scalar θ. Vector-θ (k-df Wilks, vector UMP, Hunt-Stein, UMP unbiased) is stated with LEH2005 citations. This scope discipline keeps the topic at ~10,000 words; the vector-θ generalizations would double the length without adding new pedagogical machinery. The vector-θ case is a bookkeeping extension that practitioners can absorb without full proof — §18.6 Remark 13 gives the statement and cites.

5. **Four interactive components (3 required + 1 optional, locked via scaffold Part 1).** Topic 17 had 5 components; Topic 18 has fewer because it is proof-heavy. The required three (NPLemmaVisualizer, KarlinRubinUMP, WilksConvergence) cover the three main optimality results; the optional ReparamInvariance visualizes Theorem 6 if time permits. A fifth component would dilute the WilksConvergence featured-component emphasis.

6. **Profile likelihood: one-remark preview (locked via Decision 4).** §18.5 Remark 10 previews profile likelihood in one paragraph, with the full treatment deferred to Topic 19. Including profile likelihood in full would require a dedicated section (~500 words + a worked example + likely another component) — bloating Topic 18 beyond its 60-minute target. Deferring keeps Topic 19 (confidence intervals) as the natural home: profile likelihood is a CI-formation tool first and a Topic-18-detail second.

7. **`testing.ts` extension, not new module (locked via Decision 5).** The ~150-line extension adds optimality-specific machinery (NP ratio, UMP boundary, Wilks simulator, non-central χ², local power) to the existing Topic 17 `testing.ts`. No new module is created; the single-module-per-track pattern continues. Topics 19 and 20 will further extend `testing.ts` with CI inversion and multiple-testing machinery respectively.

8. **Binomial exact test is UMP as the centerpiece application of Karlin-Rubin (§18.4 Example 6).** This is the pedagogical payoff of the §17.6 Example 11 binomial-exact-test treatment: Topic 17 showed that the binomial exact test has correct size and is a well-defined procedure; Topic 18 shows that it is *the* optimal test under the Neyman-Pearson / Karlin-Rubin framework for Bernoulli one-sided H₀. One PNG figure (`18-binomial-ump-exact.png`) visualizes the exact UMP boundary vs the Normal-approximation boundary at small n, making the discreteness + optimality story concrete.

9. **Wilks convergence as the featured interactive component.** Like Topic 17's `TTestBasuFoundation`, Topic 18 has one component that is the visual centerpiece: `WilksConvergence`. It is the "Wilks' theorem in pixels" moment — readers see the −2 log Λ_n MC histogram under H₀ morph into χ²₁ as n grows, and the empirical moments match theory to within MC error. This component alone delivers 30% of the topic's pedagogical payoff; the other two required components support it.

10. **End-of-proof citation style (`∎ — using [...]`), applied uniformly.** Matches Topic 17. All four proofs end with the same visual marker, naming the specific tools consumed. This gives readers an at-a-glance audit of which upstream results are being used.

11. **References include Neyman-Pearson 1933, Karlin-Rubin 1956, Karlin 1957, Wilks 1938, Rao 1948, Ferguson 1967, van der Vaart 1998, Buse 1982, Lehmann-Romano 2005, Casella-Berger 2002.** The canonical primary sources (NEY1933, KAR1956, WIL1938, RAO1948) are cited at theorem attributions. The modern textbook references (LEH2005, CAS2002, VAN1998) are cited throughout as the authoritative treatments. KAR1957 is cited once for its use of historical terminology. FER1967 is cited for the decision-theoretic framing of §18.1. BUS1982 is cited in §18.7 as the pedagogical anchor for the three-test equivalence.

---

*Brief version: v1 | Created: 2026-04-17 | Author: Jonathan Rocha*
*Reference notebook: `notebooks/likelihood-ratio-tests/18_likelihood_ratio_tests.ipynb`*
*Scaffold reference: `docs/topic-18-scaffold.md`*
