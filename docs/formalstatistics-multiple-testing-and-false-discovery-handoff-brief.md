# Claude Code Handoff Brief: Multiple Testing & False Discovery

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)
**Repo:** `github.com/jonx0037/formalStatistics`
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel
**Package Manager:** pnpm
**Status:** Ready for implementation
**Reference Notebook:** `notebooks/multiple-testing-and-false-discovery/20_multiple_testing.ipynb`

---

## Important: Twentieth Topic — Fourth (Final) in Track 5

This is **topic 20 of 32** and the **fourth and final topic in Track 5 (Hypothesis Testing & Confidence)**. Track 5 opened with Topic 17 (Hypothesis Testing Framework); Topic 18 (Likelihood-Ratio Tests & Neyman–Pearson) delivered the optimality theory; Topic 19 (Confidence Intervals & Duality) collected the test-inversion principle as a general construction. Topic 20 is the **honest accounting**: with $m$ tests, the Type I guarantee collapses, and two distinct repair frameworks — Family-Wise Error Rate (FWER) and False Discovery Rate (FDR) — each trade a different resource to restore it. The featured theorem is **Benjamini–Hochberg's 1995 Theorem 1**: under independence, the BH step-up procedure controls FDR at $\alpha \cdot \pi_0 \le \alpha$. The pedagogical arc runs from the FWER-explosion motivation of §20.1, through the closed-testing principle as the organizing scaffold (§20.2), the FWER family (Bonferroni / Šidák / Holm / Hochberg, §20.3–§20.5), the FDR paradigm (§20.6 definitions, §20.7 BH theorem + proof, §20.8 BY and Storey extensions), simultaneous CIs (§20.9), and a forward look that closes Track 5 and points to Tracks 6 / 8 / formalml.com (§20.10). Topic 20 also fulfills **twelve explicit forward-promises** that Topics 17, 18, and 19 made — seven from `hypothesis-testing.mdx`, two from `likelihood-ratio-tests-and-np.mdx`, and three from `confidence-intervals-and-duality.mdx`.

**Implications:**

1. **All nineteen preceding topics are published.** Cross-references from `sample-spaces` through `confidence-intervals-and-duality` use live internal links. The placeholder slug `/topics/multiple-testing` (appearing only in `curriculum-graph.json`'s forward stubs) must be migrated to `/topics/multiple-testing-and-false-discovery` — see §8.2.

2. **Twelve explicit forward-promises must be fulfilled.** Seven from `hypothesis-testing.mdx` (§17 frontmatter ref-notes, §17.1 track overview, §17.5, §17.12 Rem 16 / 17 / 19 / 19-bullet), two from `likelihood-ratio-tests-and-np.mdx` (§18 abstract, §18.10 Rem 23), and three from `confidence-intervals-and-duality.mdx` (§19.10 image alt, §19.10 Rem 19 #3, §19.10 Rem 23). See §1.2 for the complete list with verbatim text and exact MDX replacement locations; §8.1 has the find-and-replace operations.

3. **Topic 20 *extends* — does not create — the shared Track 5 module `testing.ts`.** Topic 17 seeded `testing.ts`; Topics 18 and 19 extended it with the NP machinery, the three asymptotic CI constructors, Wilson / Clopper–Pearson, profile likelihood, TOST, and the coverage simulator. Topic 20 adds the six multiple-testing procedures (Bonferroni, Holm, Šidák, Hochberg, Benjamini–Hochberg, Benjamini–Yekutieli), the Storey adaptive q-value computer, the simultaneous-CI constructors (Bonferroni / Šidák), the mixture-p-value simulator, and the FDR/FWER Monte-Carlo coverage computer. See §6 for the full manifest.

4. **Do not re-derive Wilks' theorem or the asymptotic trio.** Topic 18 Thm 4 (Wilks, §18.6) and Topic 18 Thm 5 (score-test $\chi^2_1$ null, §18.7) are used in §20.9 Rem 17 for the asymptotic basis of simultaneous profile-likelihood regions, and cited only — they are not restated.

5. **Do not re-derive the test–CI duality.** Topic 19 Thm 1 (§19.2) powers the §20.9 simultaneous-CI construction. §20.9 cites the duality directly; the Bonferroni- and Šidák-adjusted CI bands are *exactly* the test-inversions of the joint FWER-controlled tests.

6. **The BH theorem is the featured result, with full proof.** Three full proofs are delivered in Topic 20: Theorem 1 Bonferroni (union bound, one line, §20.4), Theorem 2 Holm FWER control (induction on rank order, §20.5), and the featured **Theorem 3 Benjamini–Hochberg FDR control (§20.7)** — an eight-step proof by direct computation citing BEN1995 for the original argument and BEN2001 Lemma 2.1 for the independence-lemma formalization. Šidák is stated without proof (optional 4th; cited to SID1967). Hochberg, Benjamini–Yekutieli, and Storey are stated-with-reference (HOC1988, BEN2001, STO2002).

7. **Closed-testing principle is stated as the organizing scaffold (§20.2) without full proof.** Marcus–Peritz–Gabriel (1976) established closed testing as the unifying framework that derives Holm from Bonferroni (and many other step-down procedures). The principle is stated precisely in §20.2 Thm 0 with a clear example on $m = 3$ (Figure 2), and cited to LEH2005 §9.1 for the complete proof. This keeps §20.2 short and pedagogically focused on the *idea*, without distracting from the featured BH proof in §20.7.

8. **Scalar-$m$-hypotheses only.** Every procedure is for a finite collection $\{H_1, \ldots, H_m\}$ of simple or composite hypotheses. No sequential / always-valid inference (one remark in §20.10 pointing to formalml.com). No knockoffs (one remark in §20.10 pointing to Track 8). No permutation multiplicity (one-line mention of Westfall–Young in §20.10). No Bayesian multiplicity (one remark pointing to Track 7).

9. **Every topic after Topic 20 inherits the multiple-testing scaffolding.** Track 6 (Regression) uses Bonferroni-adjusted coefficient CIs as the default simultaneous-inference tool; F-tests for nested-model comparisons are the closed-testing ancestor of Holm. Track 7 (Bayesian) contrasts FDR with Bayesian local-FDR (Efron 2010) and the closed-testing-free nature of posterior probability statements. Track 8 (High-dim) builds knockoffs (Barber–Candès 2015) directly on top of BH, and online FDR (Javanmard–Montanari 2018) extends it to the streaming regime — both extensions cited in §20.10.

10. **This brief closes Track 5.** After Topic 20 ships, `curriculum-graph.json` must mark Track 5 as **4/4 published** and surface the Track 6 preview. See §8.2.

---

### 1. Forward-Promise Fulfillment

### 1.1 Scope boundary

**What Topic 20 covers (the full list):**

- The FWER-explosion motivation with exact-formula derivation (§20.1).
- Closed-testing principle as the organizing scaffold (§20.2 Thm 0, stated-with-reference).
- FWER definition, Type I error hierarchy, strong vs weak control (§20.3).
- **Bonferroni's procedure with full proof via the union bound** (§20.4 Thm 1 + Proof 1).
- Šidák's procedure for independent tests (§20.4 Thm 2, stated without proof, cited to SID1967).
- **Holm's step-down procedure with full proof of FWER control** (§20.5 Thm 3 + Proof 3) — induction on rank order.
- Hochberg's step-up procedure (§20.5 Thm 4, stated-with-reference, cited to HOC1988).
- FDR definition, contrast with FWER, Type I / Type II trade-off (§20.6 Defs 4–5).
- **The Benjamini–Hochberg theorem with full proof of FDR control under independence** (§20.7 Thm 5 + Proof 5) — featured result.
- Benjamini–Yekutieli extension under arbitrary dependence (§20.8 Thm 6, stated-with-reference, cited to BEN2001).
- Storey's adaptive q-value framework (§20.8 Thm 7 + remarks, stated-with-reference, cited to STO2002).
- Simultaneous confidence intervals via Bonferroni and Šidák adjustments, as the CI-dual of FWER control (§20.9 Thm 8).
- Nine PNG figures, four interactive components (all required), ~9,500 words.

**What Topic 20 does NOT cover:**

- **Always-valid sequential inference and e-processes** → one-paragraph remark in §20.10 pointing to formalml.com. Howard–Ramdas–McAuliffe (2021), Ramdas–Grünwald–Vovk–Shafer (2023), and the e-value framework are at the forefront of modern research; they require martingale-filtration machinery beyond Topic 20's scope.
- **Knockoffs (Barber–Candès 2015)** → one-paragraph remark in §20.10 pointing to Track 8. Model-X and fixed-X knockoffs achieve FDR control without p-values by constructing a controlled null via a designed random variable.
- **Online FDR (Javanmard–Montanari 2018, Ramdas et al. 2017)** → one-line pointer in §20.10. Generalizes BH to streaming settings where $m$ is unbounded.
- **Conformal prediction** → not mentioned. Out of Topic 20 scope; distinct paradigm.
- **Westfall–Young permutation max-T (1993)** → one-line mention in §20.10; a resampling alternative when the joint null distribution is tractable.
- **Closed-testing full proof** → stated in §20.2 Thm 0 without proof; Marcus–Peritz–Gabriel (1976) and LEH2005 §9.1 cited.
- **Hommel's (1988) and Rom's (1990) step-up procedures** → one-line name-mentions in §20.5 Rem 8; minor variants of Hochberg with modest improvements under specific dependence structures.
- **Bayesian multiplicity and local-FDR (Efron 2010)** → one-paragraph remark in §20.10 pointing to Track 7.

### 1.2 Forward-promise fulfillment (12 items from Topics 17, 18, 19)

These are the live "Topic 20 will..." pointers currently in the published MDX. Each must be fulfilled by Topic 20's content, where the MDX reads "(coming soon, Topic 20)" or links to the placeholder slug `/topics/multiple-testing`, Claude Code must replace with a live internal link to `/topics/multiple-testing-and-false-discovery` after Topic 20 ships. See §8.1 for exact find-and-replace operations.

| # | Source MDX (verified line) | Verbatim text | Topic 20 deliverable |
|---|---|---|---|
| 1 | `hypothesis-testing.mdx` frontmatter ref-note (ASA p-value, line 165) | "Full treatment of multiple testing is Topic 20." | §20.3 (FWER definitions); §20.7 (featured BH theorem); §20.8 (FDR extensions). |
| 2 | `hypothesis-testing.mdx` frontmatter ref-note (Gelman-Loken, line 181) | "…as the conceptual preview of Topic 20." | §20.1 Rem 2 (garden of forking paths); §20.8 Rem 13 (replication-crisis framing with Ioannidis 2005 and Gelman-Loken 2014). |
| 3 | `hypothesis-testing.mdx` §17.1 track overview (line 199) | "…Topic 20 with the multiple-testing correction…" | §20.3 (FWER); §20.7 (BH); §20.9 (simultaneous CIs as CI-dual). |
| 4 | `hypothesis-testing.mdx` §17.5 (line 408) | "Topic 20 will address the multiple-testing piece formally (Bonferroni, Benjamini-Hochberg FDR, Šidák)." | §20.4 Thms 1–2 (Bonferroni, Šidák); §20.7 Thm 5 (BH). |
| 5 | `hypothesis-testing.mdx` §17.12 Rem 16 (line 830) | "…full treatment is Multiple Testing (coming soon, Topic 20), which addresses the 'garden of forking paths' concern…" | §20.1 Rem 2 (Gelman-Loken 2014 framing); §20.7 (BH as the quantitative answer); §20.8 Rem 13 (replication-crisis framing). |
| 6 | `hypothesis-testing.mdx` §17.12 Rem 17 (line 834) | "…non-technical but important complements to the multiple-testing machinery of Topic 20." | §20.8 Rem 13 (Ioannidis, Gelman-Loken as companion-literature citations). |
| 7 | `hypothesis-testing.mdx` §17.12 Rem 19 bullet (line 867) | "Multiple Testing (coming soon, Topic 20) — family-wise error (Bonferroni, Holm, Šidák), FDR (Benjamini-Hochberg, adaptive FDR), replication crisis in quantitative terms." | §20.3–§20.5 (FWER machinery); §20.6–§20.8 (FDR + adaptive FDR); §20.8 Rem 13 (replication). |
| 8 | `likelihood-ratio-tests-and-np.mdx` §18 abstract (line 38) | "…Track 5 follow-ons (CI duality in Topic 19, multiple testing in Topic 20)." | §20.3 (FWER formal definition); §20.7 (BH featured theorem). |
| 9 | `likelihood-ratio-tests-and-np.mdx` §18.10 Rem 23 (line 789) | "Multiple testing preview — Topic 20" with FWER/FDR/Ioannidis 2005/Gelman & Loken 2013 framing. | §20.1–§20.3 (setup); §20.8 Rem 13 (replication-crisis treatment; cites IOA2005 and GEL2014 the full treatment of which fulfills this remark). |
| 10 | `confidence-intervals-and-duality.mdx` §19.10 image alt (line 691) | "…Topic 20 (simultaneous CIs with FWER/FDR control)." | §20.9 Thm 8 + Figure 8 (Bonferroni / Šidák-adjusted simultaneous CIs). |
| 11 | `confidence-intervals-and-duality.mdx` §19.10 Rem 19 #3 (line 700) | "Topic 20 handles the multiple-testing side, which covers simultaneous CIs as a byproduct." | **Load-bearing.** §20.9 delivers simultaneous CIs as the CI-dual of FWER-controlled joint tests, explicitly citing Topic 19 Thm 1 (duality) as the scaffold. |
| 12 | `confidence-intervals-and-duality.mdx` §19.10 Rem 23 (line 740) | "Topic 20 closes the track with multiple testing…" | All of Topic 20 (Track 5 closure); §20.10 forward-map (Figure 9). |

---

## 2. Frontmatter

### 2.1 Full frontmatter spec

The MDX file begins with the following YAML. Every field is required; every `url:` must resolve.

```yaml
---
title: "Multiple Testing & False Discovery"
slug: "multiple-testing-and-false-discovery"
track: 5
trackName: "Hypothesis Testing & Confidence"
topicNumber: 20
positionInTrack: 4
readTime: "55 min"
difficulty: "intermediate"
status: "published"
publishedDate: "TO BE SET AT PUBLICATION"
lastUpdated: "TO BE SET AT PUBLICATION"
description: "Readers who have completed Topics 17–19 — the hypothesis-testing framework, LRT optimality, and the CI-duality — and who are ready for the honest accounting: running m tests at per-test level α collapses the Type I guarantee exponentially, and two distinct repair frameworks (FWER and FDR) each trade a different resource to restore it. The exposition moves from the FWER-explosion motivation through the closed-testing scaffold, the FWER family (Bonferroni, Šidák, Holm, Hochberg), the FDR paradigm (Benjamini–Hochberg with full proof, plus BY and Storey extensions), and simultaneous CIs as the CI-dual of joint FWER control. Featured theorem: Benjamini–Hochberg 1995 Theorem 1. Scope is the m-hypothesis setting; always-valid, knockoffs, online FDR, and permutation multiplicity are explicitly deferred to formalml.com, Track 8, and companion literature."
prerequisites:
  - topic: "hypothesis-testing"
    relationship: "Topic 17 built the test framework and named the multiple-testing debt in §17.5, §17.12 Rem 16/17/19. Topic 20 delivers: Bonferroni (§20.4) and Holm (§20.5) are multi-test refinements of Topic 17's size-α machinery; BH (§20.7) is a new framework built on the same p-value ingredient; the Gelman-Loken forking-paths remark from §17 frontmatter is fully framed in §20.1 Rem 2."
  - topic: "likelihood-ratio-tests-and-np"
    relationship: "Topic 18 §18.10 Rem 23 explicitly previewed Topic 20's multiple-testing treatment with the Ioannidis 2005 / Gelman-Loken 2014 citations. The LRT asymptotic χ²₁ null distribution (§18.7 Thm 5) supplies the p-values that feed every procedure in Topic 20."
  - topic: "confidence-intervals-and-duality"
    relationship: "Topic 19 Thm 1 (§19.2) is the test–CI duality that powers §20.9's simultaneous-CI construction. Bonferroni- and Šidák-adjusted simultaneous CIs are the direct inversions of jointly FWER-controlled tests. The Topic 19 §19.10 Rem 19 #3 pointer ('Topic 20 handles the multiple-testing side, which covers simultaneous CIs as a byproduct') is fulfilled by §20.9."
  - topic: "continuous-distributions"
    relationship: "The uniform distribution on [0,1] is the p-value null by Topic 6's probability integral transform (Topic 6 Thm 3). Every proof in Topic 20 that mentions 'p-values are uniform under H₀' cites this."
  - topic: "point-estimation"
    relationship: "Topic 13's Type I / Type II trade-off framework generalizes to FWER vs FDR in §20.6 Rem 10. The §20.8 Figure 6 power-vs-error curves are the multi-test analog of Topic 13's single-parameter power curves."
formalcalculusPrereqs:
  - topic: "sequences-limits"
    site: "formalcalculus"
    relationship: "Asymptotic FWER and FDR statements — 'as m → ∞ with π₀ fixed, FDR(BH) → α·π₀' — are sequences-limits statements. The harmonic-number bound H_m ~ ln(m) + γ in BY's normalization (§20.8 Thm 6) is a sequences-limits fact."
  - topic: "combinatorics-counting"
    site: "formalcalculus"
    relationship: "The closed-testing family of {H₁,…,H_m} has 2^m − 1 non-empty intersections (§20.2 Rem 3), a direct combinatorial counting. Figure 2 explicitly displays the 7-node closure for m=3."
formalmlConnections:
  - topic: "ab-testing-platforms"
    site: "formalml"
    relationship: "Every production A/B-test platform running hundreds of concurrent experiments must address multiple testing. Airbnb's ERF (Experiment Reporting Framework), Microsoft's ExP, and Google's Overlapping Experiments Infrastructure all use variants of BH or mSPRT (always-valid) to control error rates across the experiment portfolio. §20.8 Rem 14 cites these deployment patterns."
  - topic: "genomics-gwas"
    site: "formalml"
    relationship: "Genome-wide association studies (GWAS) test 10⁶–10⁷ single-nucleotide polymorphisms simultaneously. The standard threshold of p < 5 × 10⁻⁸ is a Bonferroni cutoff at α = 0.05 for m ≈ 10⁶. §20.8 Figure 7 simulates this regime at m = 20,000; full GWAS practice is cited to the International HapMap Consortium (2005) and the UK Biobank (Bycroft et al. 2018) as deployment anchors."
  - topic: "generalized-linear-models"
    site: "formalml"
    relationship: "GLM coefficient inference with many covariates inherits multiple-testing corrections: R's summary(glm) reports unadjusted Wald p-values; p.adjust() applies BH or Holm across the coefficient vector. The Bonferroni simultaneous CI of §20.9 is the default tool for 'which coefficients are jointly non-zero?' inference in high-dimensional regression."
  - topic: "high-dimensional-inference"
    site: "formalml"
    relationship: "BH forms the foundation of the knockoffs framework (Barber–Candès 2015) and online FDR (Javanmard–Montanari 2018) — both cited in §20.10 as forward extensions. The §20.7 Thm 5 independence assumption is exactly what knockoffs dispense with by construction."
references:
  - type: "paper"
    title: "Multiple Comparisons Among Means"
    author: "Olive Jean Dunn"
    year: 1961
    journal: "Journal of the American Statistical Association"
    volume: "56(293)"
    pages: "52–64"
    url: "https://doi.org/10.1080/01621459.1961.10482090"
    note: "Cited as DUN1961. §20.4 Thm 1 attribution — the canonical modern statement of the Bonferroni procedure (the original Bonferroni inequalities trace to Bonferroni 1936, which is an Italian-language technical report; Dunn 1961 is the English-language statistical-testing introduction universally cited)."
  - type: "paper"
    title: "A Simple Sequentially Rejective Multiple Test Procedure"
    author: "Sture Holm"
    year: 1979
    journal: "Scandinavian Journal of Statistics"
    volume: "6(2)"
    pages: "65–70"
    url: "https://www.jstor.org/stable/4615733"
    note: "Cited as HOL1979. §20.5 Thm 3 attribution — the step-down FWER procedure that dominates Bonferroni. Proof 3 follows Holm's original induction-on-rank argument."
  - type: "paper"
    title: "Rectangular Confidence Regions for the Means of Multivariate Normal Distributions"
    author: "Zbyněk Šidák"
    year: 1967
    journal: "Journal of the American Statistical Association"
    volume: "62(318)"
    pages: "626–633"
    url: "https://doi.org/10.1080/01621459.1967.10482935"
    note: "Cited as SID1967. §20.4 Thm 2 and §20.9 attribution — the tighter FWER correction under independence, 1 − (1−α)^{1/m} vs Bonferroni's α/m. Also the basis of simultaneous CIs via rectangular regions in Gaussian multivariate settings."
  - type: "paper"
    title: "A Sharper Bonferroni Procedure for Multiple Tests of Significance"
    author: "Yosef Hochberg"
    year: 1988
    journal: "Biometrika"
    volume: "75(4)"
    pages: "800–802"
    url: "https://doi.org/10.1093/biomet/75.4.800"
    note: "Cited as HOC1988. §20.5 Thm 4 attribution — the step-up FWER procedure (reverse direction from Holm's step-down), valid under independence. Uniformly at least as powerful as Holm when valid."
  - type: "paper"
    title: "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing"
    author: "Yoav Benjamini & Yosef Hochberg"
    year: 1995
    journal: "Journal of the Royal Statistical Society: Series B"
    volume: "57(1)"
    pages: "289–300"
    url: "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"
    note: "Cited as BEN1995. **Featured reference.** §20.6 Def 4 (FDR definition), §20.7 Thm 5 + Proof 5 (featured theorem — BH step-up procedure with FDR control at α·π₀ under independence), §20.8 Rem 11 (conceptual break from FWER). The most-cited paper in modern multiple-testing literature."
  - type: "paper"
    title: "The Control of the False Discovery Rate in Multiple Testing under Dependency"
    author: "Yoav Benjamini & Daniel Yekutieli"
    year: 2001
    journal: "Annals of Statistics"
    volume: "29(4)"
    pages: "1165–1188"
    url: "https://doi.org/10.1214/aos/1013699998"
    note: "Cited as BEN2001. §20.7 Proof 5 (independence lemma 2.1 — the formalization cited inside the BH proof), §20.8 Thm 6 (BY procedure: BH at α/H_m controls FDR under arbitrary dependence). Also the modern rigorous restatement of the BH 1995 proof."
  - type: "paper"
    title: "A Direct Approach to False Discovery Rates"
    author: "John D. Storey"
    year: 2002
    journal: "Journal of the Royal Statistical Society: Series B"
    volume: "64(3)"
    pages: "479–498"
    url: "https://doi.org/10.1111/1467-9868.00346"
    note: "Cited as STO2002. §20.8 Thm 7 attribution — the adaptive q-value framework: estimate π₀ from the p-value histogram tail, apply BH at α/π̂₀ for sharper FDR control when π₀ is substantially below 1. The bioconductor `qvalue` package implements this."
  - id: "IOA2005"
    text: "Ioannidis, John P. A. 2005. \"Why Most Published Research Findings Are False.\" PLoS Medicine 2 (8): e124."
    url: "https://doi.org/10.1371/journal.pmed.0020124"
    usedIn: "Already cited in Topic 17 (§17.12 Rem 16–17) and Topic 18 (§18.10 Rem 23). In Topic 20: §20.8 Rem 13 gives the full quantitative treatment previewed in Topic 17 — the positive predictive value calculation PPV = π₁ · (1−β) / (π₁·(1−β) + π₀·α) shows why low-base-rate fields with underpowered tests produce mostly false-positive 'discoveries'. Ioannidis's R multiplier framework is the BH-motivating stylized fact."
  - id: "GEL2014"
    text: "Gelman, Andrew, and Eric Loken. 2014. \"The Statistical Crisis in Science.\" American Scientist 102 (6): 460–465."
    url: "https://doi.org/10.1511/2014.111.460"
    usedIn: "Already cited in Topic 17 (§17 frontmatter ref-note, §17.12 Rem 16). In Topic 20: §20.1 Rem 2 gives the full 'garden of forking paths' framing previewed in Topic 17 — garden-of-forking-paths is the informal multiplicity, formally dual to Topic 20's m-test machinery."
  - id: "LEH2005"
    text: "Lehmann, Erich L., and Joseph P. Romano. 2005. Testing Statistical Hypotheses. 3rd ed. Springer Texts in Statistics. New York: Springer."
    url: "https://doi.org/10.1007/0-387-27605-X"
    usedIn: "Already cited in Topics 17–19. In Topic 20: §20.2 Thm 0 (closed-testing principle, Ch. 9.1); §20.4 Rem 5 (Bonferroni / Šidák chapter 9.1–9.2); §20.5 Rem 8 (Holm Ch. 9.1 and the Hommel/Rom variants); §20.10 Rem 19 (scope-boundary pointers to Ch. 9.3 for permutation multiplicity)."
  - id: "CAS2002"
    text: "Casella, George, and Roger L. Berger. 2002. Statistical Inference. 2nd ed. Pacific Grove, CA: Duxbury."
    url: "https://www.cengage.com/c/statistical-inference-2e-casella"
    usedIn: "Already cited in Topics 17–19. In Topic 20: §20.3 Ex 3 (textbook-level FWER worked example alignment); §20.4 Ex 5 (Bonferroni on m=3)."
---
```

### 2.2 Length target

Target: **55-minute read** — 10 sections, 3 full proofs, four required interactive components, ~9,500 words. Topic 20 matches Topic 19's length because though the featured BH proof is shorter than Topic 19's profile-likelihood proof, the FWER family treatment in §20.3–§20.5 has four named procedures (vs Topic 19's inversion machinery which was more unified), and §20.8's FDR extensions plus the replication-crisis framing of Rem 13 add pedagogical density that balances the arithmetic.

---

## 3. Content Structure

### 3.1 Section Map

| § | Title | Formal elements | Figure | Interactive component |
|---|-------|----------------|--------|---------------------|
| 20.1 | "When 5% Becomes 64%: The Multi-Test Problem" | Definition 0 (families of hypotheses), Remarks 1–3 (FWER explosion arithmetic, garden-of-forking-paths framing, motivating example from GWAS) | `20-fwer-explosion.png` | — |
| 20.2 | "The Closed-Testing Principle" | Definition 1 (closure of a family under intersection), Theorem 0 (closed-testing principle, stated-with-reference), Remarks 3–4 (combinatorial structure; Holm as a closed-testing specialization) | `20-closed-testing-tree.png` | — |
| 20.3 | "Family-Wise Error Rate: The FWER Paradigm" | Definitions 2–3 (FWER, strong vs weak control), Remarks 5–6 (relation to Type I; tradeoff with power) | — | — |
| 20.4 | "Bonferroni and Šidák" | Theorem 1 (Bonferroni FWER control, full proof), Theorem 2 (Šidák under independence, stated), Examples 1–3, Remarks 7–9 | `20-pvalue-mixture.png` | — |
| 20.5 | "Holm's Step-Down and Hochberg's Step-Up" | Theorem 3 (Holm FWER control, full proof by induction on rank), Theorem 4 (Hochberg, stated-with-reference), Examples 4–6, Remarks 10–12 | `20-bonferroni-vs-holm.png` | `MultipleTestingProcedureExplorer` (featured) |
| 20.6 | "False Discovery Rate: A New Error Metric" | Definitions 4–5 (FDR, FDP), Remarks 13–15 (FDR vs FWER semantics; power gains; which to pick when) | — | `FWERvsFDRComparator` |
| 20.7 | "The Benjamini–Hochberg Procedure" | **Theorem 5 (BH 1995 Theorem 1, featured, full proof)**, Examples 7–9, Remarks 16–18 | `20-bh-algorithm.png` | `BHAlgorithmVisualizer` |
| 20.8 | "Dependence and Adaptivity: BY and Storey" | Theorem 6 (BY under arbitrary dependence, stated), Theorem 7 (Storey adaptive q-value, stated), Examples 10–12 (m=20K genomics simulation; Ioannidis PPV; Gelman-Loken forking paths), Remarks 19–22 | `20-fdr-fwer-power.png`, `20-genomics-simulation.png` | — |
| 20.9 | "Simultaneous Confidence Intervals" | Theorem 8 (Bonferroni / Šidák simultaneous CI bands as duals of FWER-controlled tests), Examples 13–14, Remarks 23–25 (Scheffé and Tukey as one-line pointers; Working–Hotelling for regression) | `20-simultaneous-cis.png` | `SimultaneousCIBands` |
| 20.10 | "Track 5 Closes; Multiplicity Lives On" | Remarks 26–32 (always-valid / e-processes → formalml.com; knockoffs → Track 8; online FDR; permutation multiplicity → LEH2005 §9.3; closed-testing full proof → LEH2005 §9.1; Bayesian multiplicity → Track 7; Hommel/Rom name-mentions) | `20-forward-map.png` | — |

**Section lengths** (approximate): §20.1 (800 w), §20.2 (700 w), §20.3 (650 w), §20.4 (900 w — includes Proof 1), §20.5 (1,200 w — includes Proof 3), §20.6 (700 w), §20.7 (1,400 w — includes featured Proof 5), §20.8 (1,400 w — includes genomics + replication remark), §20.9 (900 w), §20.10 (850 w).

### 3.2 Proof obligations

Three full proofs, one stated-without-proof optional reference theorem. No KaTeX `\begin{aligned}` blocks (Track 5 style constraint); multi-line derivations use separate `$$…$$` blocks connected by prose.

**Proof 1 — Bonferroni FWER control (§20.4 Thm 1).** Trivial union bound:
> Let $V$ = number of false rejections. Under Bonferroni, $H_i$ is rejected iff $p_i < \alpha/m$. Then $\Pr_{\theta_0}(V \geq 1) = \Pr_{\theta_0}(\bigcup_{i \in H_0} \{p_i < \alpha/m\}) \leq \sum_{i \in H_0} \Pr_{\theta_0}(p_i < \alpha/m) \leq \sum_{i \in H_0} \alpha/m \leq m_0 \cdot \alpha/m \leq \alpha.$
>
> The two inequalities use the union bound and the definition of a level-$\alpha/m$ test, respectively. No independence required. $\blacksquare$ — using Boole's inequality.

Three to five MDX lines. The concision is pedagogically important: Bonferroni is *morally* the union bound. Proof 1 also sets up the language ("let $V$ = number of false rejections") reused in every subsequent proof.

**Proof 3 — Holm FWER control (§20.5 Thm 3).** Induction on rank order.
> Sort the p-values $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$ with associated hypotheses $H_{(1)}, \ldots, H_{(m)}$. Let $k^\star$ be Holm's stopping index: the smallest $k$ with $p_{(k)} \geq \alpha/(m-k+1)$, so Holm rejects $H_{(1)}, \ldots, H_{(k^\star - 1)}$.
>
> Suppose $V \geq 1$ — some true null is rejected. Let $j^\star$ be the rank of the first-rejected true null (so $H_{(j^\star)}$ is null and $j^\star \leq k^\star - 1$, hence $p_{(j^\star)} < \alpha/(m - j^\star + 1)$).
>
> Among the $m_0$ true nulls, at most $j^\star - 1$ have rank $< j^\star$ (all of which must be non-null since $H_{(j^\star)}$ is the *first*-rejected null). So the number of true-null p-values at rank $\geq j^\star$ is at least $m_0 - (j^\star - 1)$.
>
> The smallest such null p-value is $\leq p_{(j^\star)} < \alpha/(m-j^\star+1)$ by construction. So the event $V \geq 1$ implies: among the $m_0$ null-hypothesis p-values, the *minimum* is less than $\alpha/(m - j^\star + 1) \leq \alpha/m_0$. Finally, by the union bound,
> $$\Pr_{\theta_0}(V \geq 1) \leq \Pr_{\theta_0}\left(\min_{i \in H_0} p_i < \alpha/m_0\right) \leq m_0 \cdot \alpha/m_0 = \alpha.$$
> $\blacksquare$ — using Holm 1979 Theorem 2.1 and the union bound.

Six to eight MDX lines. The pedagogical payoff: Holm is "Bonferroni but sequential" — it uses the union bound but on a *smaller* set of hypotheses at each rank, which is why it dominates.

**Proof 5 — Benjamini–Hochberg FDR control (§20.7 Thm 5, featured).** Direct computation, eight lines. This is the centerpiece.
> Define $V(k) = |\{i \in H_0 : p_i \leq k\alpha/m\}|$ and $R(k) = |\{i : p_i \leq k\alpha/m\}|$ — the number of null / total hypotheses with p-value below the $k$-th BH threshold. Let $K^\star = \max\{k : p_{(k)} \leq k\alpha/m\}$ be BH's stopping index (zero if no such $k$). BH rejects the $K^\star$ smallest p-values; hence $V = V(K^\star)$ and $R = K^\star$.
>
> **Step 1 (reformulation).** $\text{FDR} = \mathbb{E}[V/\max(R,1)] = \mathbb{E}[V(K^\star)/\max(K^\star, 1)]$.
>
> **Step 2 (swap sum and expectation).** $V(K^\star) = \sum_{i \in H_0} \mathbf{1}\{p_i \leq K^\star \alpha/m\}$. So $\mathbb{E}[V/\max(R,1)] = \sum_{i \in H_0} \mathbb{E}[\mathbf{1}\{i \in \text{rejected}\} / K^\star]$.
>
> **Step 3 (independence lemma).** For each $i \in H_0$, let $R^{(-i)}(k)$ = number of rejections at threshold $k\alpha/m$ using only $\{p_j : j \neq i\}$. If $p_i$ is independent of $\{p_j : j \neq i\}$ and uniform on $[0,1]$ (true null), then by BEN2001 Lemma 2.1:
> $$\mathbb{E}\left[\frac{\mathbf{1}\{i \in \text{rejected}\}}{K^\star}\right] = \frac{\alpha}{m}.$$
>
> **Step 4 (sum).** Applying Step 3 to each of the $m_0$ true nulls:
> $$\text{FDR} = \sum_{i \in H_0} \frac{\alpha}{m} = m_0 \cdot \frac{\alpha}{m} = \pi_0 \cdot \alpha \leq \alpha.$$
> $\blacksquare$ — using BEN1995 Theorem 1 and BEN2001 Lemma 2.1.

Eight MDX lines; Step 3's independence lemma is the technical hinge. The pedagogical payoff — "FDR = π₀·α, *exactly*, under independence" — is the quantitative statement every data scientist should internalize.

**No full proof** for: Šidák (SID1967 Eq. 2.5), Hochberg (HOC1988 Thm 1), Benjamini–Yekutieli (BEN2001 Thm 1.3), Storey (STO2002 Thm 2), closed-testing principle (LEH2005 §9.1). Each is stated precisely with citation and one-line intuition.

### 3.3 Formal element inventory

| Element | Count | Notes |
|---------|-------|-------|
| Definitions | 6 | Def 0 (family), Def 1 (closure), Def 2 (FWER), Def 3 (strong/weak control), Def 4 (FDR), Def 5 (FDP) |
| Theorems | 8 | Thm 0 (closed-testing), Thm 1 (Bonferroni, full proof), Thm 2 (Šidák, stated), Thm 3 (Holm, full proof), Thm 4 (Hochberg, stated), **Thm 5 (BH, featured, full proof)**, Thm 6 (BY, stated), Thm 7 (Storey, stated), Thm 8 (simultaneous CI duality) |
| Proofs | 3 full | Proof 1 (Bonferroni), Proof 3 (Holm), **Proof 5 (BH)**. Proofs 2 (Šidák), 4 (Hochberg), 6 (BY), 7 (Storey), 8 (simultaneous CI) are stated-with-reference. |
| Examples | 14 | Ex 1–3 (FWER motivations / §20.4); Ex 4–6 (Bonf vs Holm on worked m = 20 and m = 5 cases / §20.5); Ex 7–9 (BH on the Brief-§6.2-T1 fixed vector, FDR under independence MC, k* derivation / §20.7); Ex 10 (m=20K genomics / §20.8); Ex 11 (Ioannidis PPV / §20.8); Ex 12 (Gelman-Loken forking-paths / §20.8); Ex 13 (Bonferroni simultaneous CIs / §20.9); Ex 14 (Šidák vs Bonferroni joint coverage / §20.9) |
| Remarks | 22 | Distributed across sections as noted in §3.1 |
| Figures | 9 | See §4 |
| Interactive components | 4 | See §5 |

---

## 4. Static Figures & Notebook Structure

**Notebook location:** `notebooks/multiple-testing-and-false-discovery/20_multiple_testing.ipynb`. Thirteen cells (one markdown header, twelve code cells for figures + verification + summary, one closing markdown). Seed `42` throughout; each sampling cell seeds a fresh `numpy.random.Generator` so figures reproduce in isolation.

**Output directory:** `public/images/topics/multiple-testing-and-false-discovery/`. All PNGs use `dpi=150`, `bbox_inches='tight'`, `pad_inches=0.15`.

**Palette extension** (in `20_multiple_testing.ipynb` Cell 2; must be mirrored in `src/components/viz/shared/colorScales.ts` as new CSS custom properties):

- **FWER family (purples):** `--color-bonf` `#6B21A8`, `--color-holm` `#9333EA`, `--color-sidak` `#A855F7`, `--color-hoch` `#C084FC`.
- **FDR family (greens):** `--color-bh` `#10B981` (shares hex with existing `--color-lrt` by design), `--color-by` `#047857`, `--color-storey` `#34D399`.
- **Mixture coloring:** `--color-null` `#94A3B8` (slate), `--color-alt` `#F59E0B` (amber — used for ground-truth alternatives in scatter plots).

| # | Filename | Section | Purpose | Alt text |
|---|---|---|---|---|
| 1 | `20-fwer-explosion.png` | §20.1 | Log-scale $m$ axis, $1-(1-\alpha)^m$ curve in red, union bound dashed, annotated points at $m=10, 20, 100$. | "FWER as a function of the number of tests m, plotted on log-x axis from 1 to 10,000. The curve 1 - (1-α)^m starts at α=0.05 and saturates above 0.99; the union bound mα is overlaid as a dashed reference; horizontal pivot line at α=0.05. Annotations at m=10 (40.1%), m=20 (64.2%), m=100 (99.4%). Under independence, running twenty tests at individual level 0.05 collapses the overall Type I guarantee to 64%." |
| 2 | `20-closed-testing-tree.png` | §20.2 | Closed-testing principle for $m=3$: $H_{123}$ at top, $H_{12}, H_{13}, H_{23}$ middle, $H_1, H_2, H_3$ bottom. | "Closed-testing tree for three hypotheses {H₁, H₂, H₃}. Top level: the global intersection H₁₂₃ (size-3, purple). Middle level: the three pairwise intersections H₁₂, H₁₃, H₂₃ (size-2, purple). Bottom level: the three individual hypotheses H₁, H₂, H₃ (size-1, green). Arrows show the containment structure: each higher-level node connects down to every lower-level node it contains. Reject H_i iff every H_S containing i in the closure is rejected at level α." |
| 3 | `20-pvalue-mixture.png` | §20.4 | Two-panel histogram: mixture (observed) left, ground-truth decomposition right. $m=1000$, $\pi_0=0.8$. | "Two-panel histogram of 1,000 p-values generated as an 80/20 mixture of null (uniform on [0,1]) and alternative (one-sided z-test, effect size δ=2.5). Left panel: observed mixture — a spike near zero on a roughly flat background; the flat-null per-bin reference (m/30) is drawn as a dashed grey line. Right panel: the ground-truth decomposition (shown for pedagogy only; not observable in practice) — null p-values in slate grey form the uniform background, alternative p-values in amber concentrate near zero." |
| 4 | `20-bonferroni-vs-holm.png` | §20.5 | Sorted p-values vs rank for $m=20$, log-y. Bonferroni as flat line at $\alpha/m$; Holm as staircase $\alpha/(m-k+1)$. | "Log-scale scatter of 20 sorted p-values against rank k (1 to 20), generated from a 60/40 null/alternative mixture. Alternatives in amber, nulls in grey. Bonferroni threshold α/m = 0.0025 is a flat dark-purple line; Holm's step-down threshold α/(m-k+1) is a staircase in medium purple starting at α/20 on the left and rising to α at rank 20. Holm rejects every p-value below its staircase curve up to the first failure; Bonferroni rejects only below its flat line. The caption reports each procedure's rejection count for this sample." |
| 5 | `20-bh-algorithm.png` | §20.7 | Sorted p-values vs rank for $m=20$, log-y, $\alpha=0.10$. BH diagonal $k\alpha/m$ in green; k* circled; rejection region shaded. | "BH algorithm visualized on 20 sorted p-values (60/40 null/alt mixture) at α=0.10. The green diagonal shows the BH threshold k·α/m; the purple dotted reference line shows Bonferroni α/m for comparison. The largest k with p_(k) below the BH line is k*=10, circled in red with an annotation arrow. The rejection region (ranks 1 through k*) is shaded light green. Alternatives in amber, nulls in grey — the figure shows BH correctly retaining the signal-bearing amber points while accepting that two grey (null) points at ranks 8–9 are below the threshold, contributing to the expected FDR of α·π₀ = 0.04." |
| 6 | `20-fdr-fwer-power.png` | §20.8 | 3×2 grid: rows = $\pi_0 \in \{0.5, 0.8, 0.95\}$, cols = {error rate, power}. Four procedures per panel: Bonf, Holm, BH, BY. | "Six-panel Monte Carlo comparison (500 trials per point, m=100, α=0.05) across three sparsity regimes (π₀ ∈ {0.5, 0.8, 0.95}). Left column plots error rate (FWER for Bonf/Holm shown as purples; FDR for BH/BY shown as greens) against signal strength δ on x-axis. Right column plots power against δ. Target α=0.05 dotted horizontal. In the sparse regime (π₀=0.95, bottom row), Bonferroni and Holm are indistinguishable; BH approaches α·π₀ ≈ 0.048; all procedures attain high power at δ=4. In the dense regime (π₀=0.5, top row), BH's power advantage over FWER procedures is dramatic." |
| 7 | `20-genomics-simulation.png` | §20.8 | Two-panel: bar chart of detections per procedure (left); power-vs-FDP scatter (right). $m=20{,}000$, $\pi_0=0.99$, $\delta=3.5$. | "Realistic GWAS-scale simulation: m=20,000 hypotheses, 200 true alternatives, signal δ=3.5. Left panel: bar chart showing true discoveries (solid) and false discoveries (hatched) for each of Bonferroni, Holm, BH, BY. BH recovers 3× more true signals than Bonferroni at minimal FDP cost. Right panel: realized FDP vs power scatter (one replicate). Bonferroni and Holm cluster near (FDP≈0, power≈0.1); BH sits at (FDP≈0.04, power≈0.45); BY sits in between. The α=0.05 vertical dashed line marks the FDR target. The figure makes vivid the central BH tradeoff: accept ~4% false discoveries in exchange for a 4-fold increase in power." |
| 8 | `20-simultaneous-cis.png` | §20.9 | Three-bar horizontal CI display for $m=10$ parameters: unadjusted 95%, Bonferroni, Šidák. True values marked. | "Simultaneous confidence intervals for ten Normal means, comparing unadjusted 95% CIs (red, wide spread — miscovers jointly at 1 − 0.95^10 ≈ 40% probability), Bonferroni-adjusted (dark purple, each at 1 − α/m = 99.5%), and Šidák-adjusted (light purple, each at (1−α)^{1/m}, slightly tighter than Bonferroni under independence). True parameter values are marked with black vertical bars. The figure dramatizes the width cost of simultaneity — Bonferroni intervals are roughly 1.35× wider than unadjusted — and the mild improvement Šidák offers under independence." |
| 9 | `20-forward-map.png` | §20.10 | Track 5 spine + three forward nodes + dashed back-arrow to Topic 19. | "Track 5 closing map. Horizontal spine: Topic 17 (Hypothesis Testing, blue) → Topic 18 (LRT/NP, amber) → Topic 19 (CIs, purple) → Topic 20 (Multi-Testing, green). Forward arrows from Topic 20 lead to three destination nodes: Track 6 (Regression F-tests and GLM coefficient p-values), Track 8 (High-dim FDR, knockoffs), and formalml.com (always-valid inference, online FDR). A dashed curved back-arrow from Topic 20 to Topic 19 is labeled 'Simultaneous CIs dualize Topic 19' — the CI-dual relationship explored in §20.9. Bold header reads 'Track 5 complete.'" |

**Static figures are produced by `20_multiple_testing.ipynb` Cells 3–11.** Cell 12 is the verification helper (outputs test-harness values for §6.2); Cell 13 is a markdown summary. See §6.2 for the regression-test values the notebook emits.

---

## 5. Interactive Components

Four components total, all required. All follow Topic 17–19 patterns: React + Tailwind, `client:visible` directive (**not** `client:load`), SVG/D3.js only, CSS custom properties from `viz/shared/colorScales.ts`, responsive to 375px width, client-side computation only.

### Component 1: MultipleTestingProcedureExplorer (featured)

**File:** `src/components/viz/MultipleTestingProcedureExplorer.tsx`

**Purpose:** The featured component. Side-by-side comparison of up to six procedures on a user-configurable p-value vector. The user sees, live, which hypotheses each procedure rejects, the realized V / S / R counts (when ground-truth is known), and the FWER / FDR targets each procedure attempts to control. This is the one-stop tool for "which correction should I use?" — the concrete answer to the §20.3–§20.8 pedagogy.

**Interactions:**
- Slider: $m$ (number of tests, 5 to 200, log-scale).
- Slider: $\pi_0$ (null proportion, 0.0 to 1.0).
- Slider: signal strength $\delta$ (0.0 to 5.0, controls alternative effect size).
- Slider: $\alpha$ (0.01 to 0.20).
- Multi-select toggle: procedures to display (Bonferroni, Holm, Šidák, Hochberg, BH, BY, Storey). Default: all six on (Storey hidden by default — adaptive is a §20.8 remark, not a main-line default).
- Resample button: draw a fresh p-value vector.
- Left panel (main): horizontal strip chart — ranks on x-axis, sorted p-values on y-axis (log-scale), each procedure's threshold overlaid as a line or staircase matching Figure 4 and Figure 5 conventions. Rejected hypotheses for each procedure highlighted with a colored outline on the point marker.
- Right panel (stats table): for each enabled procedure, displays $R$ (total rejections), $V$ (false discoveries, when ground-truth is known from the simulation), $S$ (true discoveries), realized FDP, and — over multiple resamples via a ring buffer of the last 200 draws — empirical FWER and FDR estimates with 2σ bootstrap CIs.
- Bottom readout: "At $m = \cdots, \pi_0 = \cdots, \delta = \cdots$: over 200 draws, empirical FWER(Bonf) = $\cdots \pm \cdots$, empirical FDR(BH) = $\cdots \pm \cdots$." Gives the reader a live quantitative read on the relationship between the procedures.

**Data:** `procedureExplorerPresets` from `multi-testing-data.ts`.

**Uses from `testing.ts`:** `bonferroni`, `holm`, `sidak`, `hochberg`, `benjaminiHochberg`, `benjaminiYekutieli`, `storeyBH`, `simulatePValues` (all new in §6).

**Mobile behavior:** strip chart full-width on top, stats table below (scrollable); procedure toggles collapse to a dropdown at ≤480px.

**Performance note:** every slider change triggers a fresh p-value sampling only if "resample on change" is toggled; otherwise the stored p-value vector is replayed. Debounce slider updates at 150 ms.

### Component 2: FWERvsFDRComparator

**File:** `src/components/viz/FWERvsFDRComparator.tsx`

**Purpose:** Make the §20.6 semantic distinction between FWER and FDR directly interactive. Reader chooses a single procedure and watches the actual FWER and FDR curves traced across signal strength, computed by Monte Carlo. The key visual: BH's FDR curve sits at α·π₀ flat in signal, while its FWER curve grows unboundedly — the signature of "controls FDR, not FWER."

**Interactions:**
- Dropdown: which procedure (Bonferroni, Holm, Šidák, Hochberg, BH, BY). Default: BH.
- Slider: $m$ (20 to 2000).
- Slider: $\pi_0$ (0.0 to 1.0).
- Slider: $\alpha$ (0.01 to 0.20).
- Slider: signal strength $\delta$ range (slide endpoints; default 0.0 to 4.0).
- Run button: execute $n_{\rm trials} = 300$ Monte Carlo replicates at each of 12 signal points.
- Left panel: FWER and FDR curves plotted against $\delta$ on x-axis, rate on y-axis. FWER in warn-red, FDR in BH-green. Horizontal dashed lines at $\alpha$ and $\alpha \cdot \pi_0$. Curves shade $\pm 2\sigma$ bootstrap band.
- Right panel: power curve (same $\delta$ axis), colored by procedure.
- Bottom readout: "For $\{\text{procedure}\}$ at $\pi_0 = \cdots$: average FWER over the signal range = $\cdots$; average FDR = $\cdots$. Procedure controls $\{\text{FWER} | \text{FDR}\}$ at level $\cdots$."

**Data:** `fwerVsFdrPresets` from `multi-testing-data.ts`.

**Uses from `testing.ts`:** the six procedures, `simulatePValues`, `fdrMonteCarlo`.

**Mobile behavior:** left panel on top, right panel below; on narrow screens combine into a 2-row stack with shared x-axis.

**Performance note:** MC runs on the UI thread — keep $n_{\rm trials} \times \text{signal points} \leq 4000$ per click. For $m > 500$, show a progress bar.

### Component 3: BHAlgorithmVisualizer

**File:** `src/components/viz/BHAlgorithmVisualizer.tsx`

**Purpose:** Animate the step-up procedure so the reader watches BH walk from right to left, find $k^\star$, and reject everything below. Pedagogically reinforces §20.7 Proof 5's structure: the algorithm is a one-pass sweep, not a sequential sort-and-threshold.

**Interactions:**
- Dropdown: preset p-value vector. Default: the §6.2 T1 vector `[0.0005, 0.003, 0.011, 0.021, 0.043, 0.051, 0.087, 0.21, 0.33, 0.55]` (m=10) with k*=3 at α=0.05. Other presets: the m=20 random-seed Figure 5 example (k*=10 at α=0.10); a degenerate m=5 example where no rejection occurs; a dense-signal case where k*=m.
- Slider: $\alpha$ (0.01 to 0.20).
- Play / pause / step buttons: animate the walk-from-right.
- Speed slider: 0.5× to 2× animation rate.
- Left panel (main): identical to Figure 5 but interactive — BH diagonal line, Bonferroni reference, scatter of sorted p-values. Animation: a moving marker starts at $k = m$ and walks leftward; at each $k$, the marker either lights up red (failure: $p_{(k)} > k\alpha/m$, continue) or lights up green (success: $p_{(k)} \leq k\alpha/m$, stop and shade rejection region). On stop, the rejection region fills in with a green wash.
- Right panel (step log): scrolling table of the sweep: rank $k$, $p_{(k)}$, threshold $k\alpha/m$, verdict. The row that becomes $k^\star$ is highlighted.
- Bottom readout: at animation completion, "$k^\star = \cdots$, rejected $\{H_{(1)}, \ldots, H_{(k^\star)}\}$. Observed FDP = $\cdots$." (When ground truth is known from preset.)

**Data:** `bhAlgorithmPresets` from `multi-testing-data.ts`.

**Uses from `testing.ts`:** `benjaminiHochberg`, `benjaminiHochbergSweep` (new in §6 — returns the full step-by-step trace, not just the boolean array).

**Mobile behavior:** step log collapses below main panel; speed / play controls stick to the top on scroll.

### Component 4: SimultaneousCIBands

**File:** `src/components/viz/SimultaneousCIBands.tsx`

**Purpose:** Make the §20.9 simultaneous-CI story tactile. Reader sees $m$ estimates with three CI types overlaid (unadjusted, Bonferroni, Šidák) and drags a "true parameter" slider to watch joint coverage break down for unadjusted intervals as $m$ grows. The visual companion to Figure 8.

**Interactions:**
- Slider: $m$ (2 to 50).
- Slider: $\alpha$ (0.01 to 0.20).
- Slider: sample size per parameter $n$ (5 to 200).
- Slider: true-parameter spread (controls how different the $m$ true parameters are, for realistic display).
- Radio: which CI band to display (Unadjusted, Bonferroni, Šidák, All three).
- Resample button: draw fresh $\bar x_i$ from $N(\theta_i, \sigma^2/n)$.
- Left panel (CI display): horizontal bars for the $m$ parameters, sorted by true value. For the selected procedure(s), intervals drawn in their palette colors (red for unadjusted, dark-purple for Bonferroni, light-purple for Šidák). True values marked with black vertical bars; missed coverages highlighted in thicker line weight.
- Right panel (joint coverage histogram): ring buffer of the last 200 resamples. For each resample, computed $\mathbf{1}\{\text{all $m$ intervals cover their targets}\}$ for each procedure. Histogram shows the joint-coverage proportion over the buffer with 2σ bounds.
- Bottom readout: "At $m = \cdots$: unadjusted joint coverage $\approx (1-\alpha)^m = \cdots$; Bonferroni joint coverage (empirical, 200 draws) $\approx \cdots$ (target $\geq 1-\alpha = \cdots$); Šidák $\approx \cdots$."

**Data:** `simultaneousCIPresets` from `multi-testing-data.ts`.

**Uses from `testing.ts`:** `simultaneousCIBonferroni`, `simultaneousCISidak`, `standardNormalInvCDF`.

**Mobile behavior:** bars full-width on top; histogram stacks below; the $m$ axis scrolls horizontally if needed beyond 15 parameters.

### Implementation Notes

- **Architecture:** same as Topics 1–19. React + Tailwind, `client:visible` directive.
- **SVG/D3.js for all charts.** No Chart.js, Plotly, Recharts, or other libraries.
- **CSS custom properties** from `viz/shared/colorScales.ts` — Topic 20 adds the FWER-family purples, FDR-family greens, and null/alt slate/amber (see §4 palette spec).
- **Responsive:** all components tested at 375px width.
- **Client-side computation only.**
- **`MultipleTestingProcedureExplorer`** is the showpiece; polish the slider-debounce + ring-buffer behavior; the "resample on change" toggle is a key UX detail.
- **`FWERvsFDRComparator`** is the heaviest component computationally (~3000 MC trials per click); show a progress bar and keep the UI responsive.
- **`BHAlgorithmVisualizer`** is the pedagogical anchor for §20.7; invest in smooth animation timing.
- **`SimultaneousCIBands`** is the §20.9 anchor — visually mirrors Figure 8 but with live resampling.

---

## 6. Shared Module (`testing.ts` extension)

`testing.ts` is the Track 5 shared module, created in Topic 17 and extended in Topics 18 and 19. The Topic 17–19 baseline includes null-distribution densities/CDFs/quantiles (z, t, χ², F, non-central χ²), the full Wald/Score/LRT trio for Bernoulli/Normal/Poisson, the binomial exact test and UMP boundary finder, the NP likelihood-ratio evaluator, the Wilks MC simulator, the four CI constructors (Wald, Score, LRT, Wilson), Clopper–Pearson, profile likelihood, TOST, and the binomial coverage simulator. Topic 20 adds the multiple-testing procedures, the simultaneous-CI constructors, the mixture-p-value simulator, and the FDR/FWER Monte-Carlo coverage computer.

### 6.1 Full new-function manifest

```typescript
// ═══════════════════════════════════════════════════════════════════════════
// TOPIC 20 EXTENSIONS TO testing.ts
// ═══════════════════════════════════════════════════════════════════════════

// ─── FWER procedures ────────────────────────────────────────────────────────

/**
 * Bonferroni's single-step FWER procedure.
 * Reject H_i iff p_i < alpha / m.
 * Controls FWER ≤ α under any dependence (union bound).
 *
 * @param pvals - Array of p-values.
 * @param alpha - Significance level (default 0.05).
 * @returns Boolean array: reject[i] = true iff H_i is rejected.
 */
export function bonferroni(pvals: number[], alpha?: number): boolean[];

/**
 * Holm's step-down FWER procedure (Holm 1979).
 * Sort p-values ascending; stop at first i with p_(i) ≥ α/(m-i+1).
 * Controls FWER ≤ α under any dependence. Uniformly dominates Bonferroni.
 */
export function holm(pvals: number[], alpha?: number): boolean[];

/**
 * Šidák's single-step FWER procedure (Šidák 1967).
 * Reject H_i iff p_i < 1 - (1-α)^(1/m).
 * Controls FWER ≤ α under independence (exact if independence holds; conservative otherwise).
 * Tighter than Bonferroni under independence.
 */
export function sidak(pvals: number[], alpha?: number): boolean[];

/**
 * Hochberg's step-up FWER procedure (Hochberg 1988).
 * Sort p-values descending; find largest k with p_(k) ≤ α/(m-k+1); reject all ≤ k.
 * Controls FWER ≤ α under independence or PRDS dependence.
 * Uniformly at least as powerful as Holm when valid.
 */
export function hochberg(pvals: number[], alpha?: number): boolean[];

// ─── FDR procedures ─────────────────────────────────────────────────────────

/**
 * Benjamini-Hochberg step-up FDR procedure (BH 1995).
 * Sort p-values ascending; find largest k with p_(k) ≤ k·α/m; reject all ≤ k.
 * Controls FDR ≤ α·π₀ ≤ α under independence (BH 1995 Thm 1) or PRDS (BY 2001).
 *
 * This is the featured procedure of Topic 20.
 */
export function benjaminiHochberg(pvals: number[], alpha?: number): boolean[];

/**
 * BH with full sweep trace (for animation in BHAlgorithmVisualizer).
 * Returns {rejected, sweep} where sweep[i] = {rank, pValue, threshold, verdict}
 * walking from k=m down to k=1.
 */
export function benjaminiHochbergSweep(
  pvals: number[],
  alpha?: number
): {
  rejected: boolean[];
  kStar: number;
  sweep: Array<{
    rank: number;
    pValue: number;
    threshold: number;
    verdict: 'pass' | 'fail';
  }>;
};

/**
 * Benjamini-Yekutieli FDR procedure (BY 2001).
 * BH applied at α / H_m where H_m = Σ_{k=1}^m 1/k is the m-th harmonic number.
 * Controls FDR ≤ α under ANY dependence (arbitrary joint distribution of p-values).
 */
export function benjaminiYekutieli(pvals: number[], alpha?: number): boolean[];

/**
 * Storey's adaptive q-value BH procedure (STO2002).
 * Estimate π̂_0 = |{p_i ≥ λ}| / ((1-λ)·m); apply BH at α / π̂_0.
 * Sharper than BH when π_0 is substantially below 1.
 *
 * @param lambda - Threshold for π_0 estimation (default 0.5).
 */
export function storeyBH(
  pvals: number[],
  alpha?: number,
  lambda?: number
): boolean[];

/**
 * Storey q-value computation (STO2002 Def 2).
 * q_i = min_{t ≥ p_i} [π̂_0 · t · m / (number of p_j ≤ t)]
 * Returns per-hypothesis q-values — analog of adjusted p-values for FDR.
 */
export function storeyQValues(
  pvals: number[],
  lambda?: number
): number[];

// ─── Simultaneous CIs (§20.9) ───────────────────────────────────────────────

/**
 * Bonferroni-adjusted simultaneous CIs for m scalar parameters.
 * Each interval at level 1 - α/m ⇒ joint coverage ≥ 1 - α (union bound on CI-duality).
 *
 * @param estimates - Point estimates θ̂_1, ..., θ̂_m.
 * @param standardErrors - Corresponding SE(θ̂_i) (assumed Normal pivot).
 * @param alpha - Joint (family-wise) coverage target 1-α.
 */
export function simultaneousCIBonferroni(
  estimates: number[],
  standardErrors: number[],
  alpha?: number
): Array<{ lower: number; upper: number }>;

/**
 * Šidák-adjusted simultaneous CIs under independence.
 * Each interval at level (1-α)^(1/m) — tighter than Bonferroni when indep. holds.
 */
export function simultaneousCISidak(
  estimates: number[],
  standardErrors: number[],
  alpha?: number
): Array<{ lower: number; upper: number }>;

// ─── Monte Carlo simulators (§20.8) ─────────────────────────────────────────

/**
 * Generate m p-values as a mixture of π₀·m nulls (Uniform[0,1]) and (1-π₀)·m
 * alternatives from a one-sided z-test with given signal strength.
 * Used by all four interactive components and the notebook.
 *
 * @param m - Total number of hypotheses.
 * @param pi0 - Proportion of true nulls (0 to 1).
 * @param signalStrength - Effect size δ for alternatives (z ~ N(δ, 1)).
 * @param seed - PRNG seed (for reproducibility).
 * @returns {pvals, isNull} — both length m, shuffled.
 */
export function simulatePValues(
  m: number,
  pi0: number,
  signalStrength: number,
  seed: number
): { pvals: number[]; isNull: boolean[] };

/**
 * Monte Carlo estimate of FDR, FWER, and power for a given procedure.
 * Used by FWERvsFDRComparator.
 *
 * @param procedure - One of 'bonferroni', 'holm', 'sidak', 'hochberg', 'bh', 'by', 'storey'.
 * @param m, pi0, signalStrength, alpha - Configuration.
 * @param nTrials - Number of Monte Carlo replicates.
 * @param seed - PRNG seed.
 * @returns {fdr, fwer, power} each as mean ± 2σ bootstrap CI.
 */
export function multiTestingMonteCarlo(
  procedure:
    | 'bonferroni' | 'holm' | 'sidak' | 'hochberg'
    | 'bh' | 'by' | 'storey',
  m: number,
  pi0: number,
  signalStrength: number,
  alpha: number,
  nTrials: number,
  seed: number
): {
  fdr: { mean: number; lower: number; upper: number };
  fwer: { mean: number; lower: number; upper: number };
  power: { mean: number; lower: number; upper: number };
};
```

**Placement:** append new functions at the end of `testing.ts`, after Topic 19's TOST / coverage simulator block. Keep the existing alphabetical grouping within each subsection.

### 6.2 Regression tests

Add the following as a new `describe` block in `src/lib/stat/__tests__/testing.test.ts`. All values are exact outputs of `20_multiple_testing.ipynb` Cell 12 (verified with `numpy.random.default_rng(seed=42)` semantics; re-seed is documented in the notebook).

```typescript
describe('Topic 20: Multiple testing procedures', () => {

  // T1: BH cutoff on a fixed sorted p-value vector
  it('BH rejects exactly 3 hypotheses on the canonical test vector', () => {
    const p = [0.0005, 0.003, 0.011, 0.021, 0.043,
               0.051, 0.087, 0.21, 0.33, 0.55];
    const rejected = benjaminiHochberg(p, 0.05);
    expect(rejected.filter(x => x).length).toBe(3);
    expect(rejected.slice(0, 3)).toEqual([true, true, true]);
    expect(rejected.slice(3)).toEqual(Array(7).fill(false));
  });

  // T2: Holm ≡ Bonferroni when only one rejection occurs
  it('Holm ≡ Bonferroni when only one p-value passes Bonferroni threshold', () => {
    const p = [0.0001, 0.08, 0.12, 0.5];
    const bonf = bonferroni(p, 0.05);
    const h = holm(p, 0.05);
    expect(bonf).toEqual(h);
    expect(bonf).toEqual([true, false, false, false]);
  });

  // T3: BH empirical FDR bound at (m=200, π₀=0.8, δ=3.0, α=0.05, 3000 trials)
  it('BH empirical FDR stays within bound α·π₀ under independence', () => {
    const { fdr } = multiTestingMonteCarlo(
      'bh', 200, 0.8, 3.0, 0.05, 3000, 123
    );
    // Theoretical bound α·π₀ = 0.040; 3σ of MC at 3000 trials ≈ 0.007
    expect(fdr.mean).toBeGreaterThan(0.030);
    expect(fdr.mean).toBeLessThan(0.050);
    // Additionally: BH does NOT control FWER — verify FWER is large.
    const { fwer } = multiTestingMonteCarlo(
      'bh', 200, 0.8, 3.0, 0.05, 3000, 123
    );
    expect(fwer.mean).toBeGreaterThan(0.50);
  });

  // T4: FWER explosion formula (analytical, not MC)
  it('FWER = 1 - (1-α)^m matches at canonical m values', () => {
    const alpha = 0.05;
    expect(1 - Math.pow(1 - alpha, 20)).toBeCloseTo(0.6415, 4);
    expect(1 - Math.pow(1 - alpha, 100)).toBeCloseTo(0.9941, 4);
  });

  // T5: Harmonic number for BY normalization
  it('BY normalization c_m = Σ_{k=1}^m 1/k is computed correctly', () => {
    const harmonic = (m: number) =>
      Array.from({ length: m }, (_, k) => 1 / (k + 1)).reduce((a, b) => a + b, 0);
    expect(harmonic(10)).toBeCloseTo(2.9290, 3);
    expect(harmonic(100)).toBeCloseTo(5.1874, 3);
    expect(harmonic(1000)).toBeCloseTo(7.4855, 3);
  });

  // T6: Simultaneous CI coverage (Bonferroni joint coverage ≥ 1-α)
  it('Bonferroni simultaneous CIs have joint coverage ≥ 1-α empirically', () => {
    // 500 trials of m=10 Normal means, verify joint coverage ≥ 0.94.
    // Tolerates 2σ MC slack below nominal 0.95.
    // ... (test body omitted here for brevity; fully spec'd in integration test)
  });

});
```

The tests exercise the full procedure stack with known-correct expected values. T1, T2, T4, T5 are deterministic; T3, T6 are MC and use fixed seeds for reproducibility.

---

## 7. Preset Data Module

New file: `src/data/multi-testing-data.ts`.

```typescript
// src/data/multi-testing-data.ts
// Presets for the four Topic 20 interactive components.

export interface ProcedureExplorerPreset {
  id: string;
  label: string;
  m: number;
  pi0: number;
  signalStrength: number;
  alpha: number;
  description: string;
}

export const procedureExplorerPresets: ProcedureExplorerPreset[] = [
  {
    id: 'dense-signal',
    label: 'Dense signal (π₀ = 0.5)',
    m: 100, pi0: 0.5, signalStrength: 2.5, alpha: 0.05,
    description: 'Half of the hypotheses are true alternatives. BH power ~ 0.7; Bonferroni loses substantial power.'
  },
  {
    id: 'moderate-sparse',
    label: 'Moderate sparse (π₀ = 0.8)',
    m: 100, pi0: 0.8, signalStrength: 3.0, alpha: 0.05,
    description: 'Default exploration configuration; realistic for behavioral-science multi-test scenarios.'
  },
  {
    id: 'gwas-like',
    label: 'GWAS-like (π₀ = 0.99, m large)',
    m: 1000, pi0: 0.99, signalStrength: 3.5, alpha: 0.05,
    description: 'Extreme sparsity regime; Bonferroni and Holm converge; BH retains most of the signal.'
  },
  {
    id: 'no-signal',
    label: 'No signal (π₀ = 1.0)',
    m: 100, pi0: 1.0, signalStrength: 0.0, alpha: 0.05,
    description: 'Null-only scenario. All procedures should reject ≤ α·m hypotheses on average.'
  }
];

export interface FWERvsFDRPreset {
  id: string;
  label: string;
  procedure: 'bonferroni' | 'holm' | 'sidak' | 'hochberg' | 'bh' | 'by' | 'storey';
  m: number;
  pi0: number;
  alpha: number;
  nTrials: number;
}

export const fwerVsFdrPresets: FWERvsFDRPreset[] = [
  { id: 'bh-moderate', label: 'BH, moderate sparse', procedure: 'bh',
    m: 100, pi0: 0.8, alpha: 0.05, nTrials: 300 },
  { id: 'bonf-moderate', label: 'Bonferroni, moderate sparse', procedure: 'bonferroni',
    m: 100, pi0: 0.8, alpha: 0.05, nTrials: 300 },
  { id: 'by-dependent', label: 'BY under any dependence', procedure: 'by',
    m: 100, pi0: 0.8, alpha: 0.05, nTrials: 300 },
];

export interface BHAlgorithmPreset {
  id: string;
  label: string;
  pvals: number[];
  alpha: number;
  expectedKStar: number;
  description: string;
}

export const bhAlgorithmPresets: BHAlgorithmPreset[] = [
  {
    id: 't1-canonical',
    label: 'Canonical test vector (m=10, α=0.05)',
    pvals: [0.0005, 0.003, 0.011, 0.021, 0.043,
            0.051, 0.087, 0.21, 0.33, 0.55],
    alpha: 0.05,
    expectedKStar: 3,
    description: 'Regression-test vector from §6.2 T1. BH rejects ranks 1–3 (p_(3)=0.011 passes threshold 0.015; p_(4)=0.021 fails 0.020).'
  },
  {
    id: 'fig5-random',
    label: 'Figure 5 random sample (m=20, α=0.10)',
    pvals: [], // populated at runtime from seed 43 (matches notebook Cell 7)
    alpha: 0.10,
    expectedKStar: 10,
    description: 'Random-sample illustration; matches Figure 5 exactly.'
  },
  {
    id: 'no-rejection',
    label: 'No rejection (all p > α)',
    pvals: [0.15, 0.22, 0.31, 0.48, 0.65],
    alpha: 0.05,
    expectedKStar: 0,
    description: 'Degenerate case — k* = 0, no rejections.'
  },
  {
    id: 'dense-all-reject',
    label: 'Dense signal (reject all, m=10)',
    pvals: [0.0001, 0.0005, 0.002, 0.004, 0.007,
            0.009, 0.012, 0.017, 0.022, 0.028],
    alpha: 0.10,
    expectedKStar: 10,
    description: 'All ten hypotheses satisfy the BH threshold at α=0.10; k*=m.'
  }
];

export interface SimultaneousCIPreset {
  id: string;
  label: string;
  m: number;
  n: number;
  alpha: number;
  trueTheta: number[];   // length m
  description: string;
}

export const simultaneousCIPresets: SimultaneousCIPreset[] = [
  {
    id: 'fig8-default',
    label: 'Figure 8 default (m=10)',
    m: 10, n: 25, alpha: 0.05,
    trueTheta: [1.5, 2.0, 0.8, 3.1, 2.5, 1.2, 2.8, 0.5, 1.9, 2.2],
    description: 'Matches Figure 8 parameter configuration exactly.'
  },
  {
    id: 'small-m',
    label: 'Small family (m=3)',
    m: 3, n: 30, alpha: 0.05,
    trueTheta: [1.0, 2.0, 3.0],
    description: 'Minimal-m case; Bonferroni width inflation is modest (~1.15× unadjusted).'
  },
  {
    id: 'large-m',
    label: 'Large family (m=30)',
    m: 30, n: 25, alpha: 0.05,
    trueTheta: Array.from({ length: 30 }, (_, i) => 1 + 0.1 * i),
    description: 'Larger family — Bonferroni width ~1.55× unadjusted; unadjusted joint coverage drops to 0.78.'
  }
];
```

---

## 8. Cross-Reference Updates

### 8.1 Forward-promise fulfillment (12 find-and-replace operations)

Verify each with `grep -n` before editing. Source files: `src/content/topics/hypothesis-testing.mdx`, `src/content/topics/likelihood-ratio-tests-and-np.mdx`, `src/content/topics/confidence-intervals-and-duality.mdx`.

**Updates to `hypothesis-testing.mdx` (Topic 17):**

| # | Location | Find | Replace with |
|---|---|---|---|
| 1 | frontmatter ref-note (ASA, ~line 165) | `"Full treatment of multiple testing is Topic 20."` | `"Full treatment of multiple testing is [Topic 20](/topics/multiple-testing-and-false-discovery)."` |
| 2 | frontmatter ref-note (Gelman-Loken, ~line 181) | `"as the conceptual preview of Topic 20."` | `"as the conceptual preview of [Topic 20](/topics/multiple-testing-and-false-discovery#section-20-1)."` |
| 3 | §17.1 track overview (~line 199) | `"...Topic 20 with the multiple-testing correction..."` | `"...[Topic 20](/topics/multiple-testing-and-false-discovery) with the multiple-testing correction..."` |
| 4 | §17.5 (~line 408) | `"Topic 20 will address the multiple-testing piece formally (Bonferroni, Benjamini-Hochberg FDR, Šidák)."` | `"[Topic 20](/topics/multiple-testing-and-false-discovery) addresses the multiple-testing piece formally ([Bonferroni](/topics/multiple-testing-and-false-discovery#section-20-4), [Benjamini-Hochberg FDR](/topics/multiple-testing-and-false-discovery#section-20-7), [Šidák](/topics/multiple-testing-and-false-discovery#section-20-4))."` |
| 5 | §17.12 Rem 16 (~line 830) | `"...full treatment is Multiple Testing (coming soon, Topic 20), which addresses the 'garden of forking paths' concern..."` | `"...full treatment is [Multiple Testing & False Discovery (Topic 20)](/topics/multiple-testing-and-false-discovery), which addresses the 'garden of forking paths' concern..."` |
| 6 | §17.12 Rem 17 (~line 834) | `"...non-technical but important complements to the multiple-testing machinery of Topic 20."` | `"...non-technical but important complements to the [multiple-testing machinery of Topic 20](/topics/multiple-testing-and-false-discovery#section-20-8)."` |
| 7 | §17.12 Rem 19 bullet (~line 867) | `"**Multiple Testing (coming soon, Topic 20)** — family-wise error (Bonferroni, Holm, Šidák), FDR (Benjamini-Hochberg, adaptive FDR), replication crisis in quantitative terms."` | `"[**Multiple Testing & False Discovery (Topic 20)**](/topics/multiple-testing-and-false-discovery) — family-wise error (Bonferroni, Holm, Šidák), FDR (Benjamini-Hochberg, adaptive FDR), replication crisis in quantitative terms."` |

**Updates to `likelihood-ratio-tests-and-np.mdx` (Topic 18):**

| # | Location | Find | Replace with |
|---|---|---|---|
| 8 | §18 abstract (~line 38) | `"...Track 5 follow-ons (CI duality in Topic 19, multiple testing in Topic 20)."` | `"...Track 5 follow-ons ([CI duality in Topic 19](/topics/confidence-intervals-and-duality), [multiple testing in Topic 20](/topics/multiple-testing-and-false-discovery))."` |
| 9 | §18.10 Rem 23 (~line 789) | `"Multiple testing preview — Topic 20"` (header of remark) | `"Multiple testing — [Topic 20](/topics/multiple-testing-and-false-discovery)"` (drop "preview"; update to live link) |

**Updates to `confidence-intervals-and-duality.mdx` (Topic 19):**

| # | Location | Find | Replace with |
|---|---|---|---|
| 10 | §19.10 image alt (~line 691) | `"Topic 20 (simultaneous CIs with FWER/FDR control)."` | `"[Topic 20](/topics/multiple-testing-and-false-discovery#section-20-9) (simultaneous CIs with FWER/FDR control)."` |
| 11 | §19.10 Rem 19 #3 (~line 700) | `"Topic 20 handles the multiple-testing side, which covers simultaneous CIs as a byproduct."` | `"[Topic 20](/topics/multiple-testing-and-false-discovery) handles the multiple-testing side, which covers [simultaneous CIs as a byproduct](/topics/multiple-testing-and-false-discovery#section-20-9)."` |
| 12 | §19.10 Rem 23 (~line 740) | `"Topic 20 closes the track with multiple testing..."` | `"[Topic 20](/topics/multiple-testing-and-false-discovery) closes the track with multiple testing..."` |

**Grep-verification commands** (run after all edits; each must return the indicated count):

```bash
# Should return 12: all internal links now resolve to the full Topic 20 slug
grep -rn '/topics/multiple-testing-and-false-discovery' src/content/topics/ --include='*.mdx' | wc -l

# Should return 0: all "(coming soon, Topic 20)" placeholders removed
grep -rn 'coming soon, Topic 20\|Topic 20.*coming soon' src/content/topics/ --include='*.mdx'

# Should return 0: no remaining references to the placeholder slug
grep -rn '/topics/multiple-testing[^-]' src/content/topics/ --include='*.mdx'

# Should return 0: no remaining bare "Topic 20" text that should be a link
grep -rn '[^]]Topic 20[^]]' src/content/topics/ --include='*.mdx' | grep -v '#section-20'
```

### 8.2 Updates to `curriculum-graph.json`

Mark `multiple-testing-and-false-discovery` as published, update the placeholder slug `multiple-testing` if present, add Track 5 completion marker.

```json
// src/data/curriculum-graph.json
// Change (FIND):
{
  "slug": "multiple-testing",
  "title": "Multiple Testing",
  "status": "coming-soon",
  "track": 5,
  "positionInTrack": 4,
  "prerequisites": ["hypothesis-testing"]
}

// Change (REPLACE WITH):
{
  "slug": "multiple-testing-and-false-discovery",
  "title": "Multiple Testing & False Discovery",
  "status": "published",
  "track": 5,
  "positionInTrack": 4,
  "prerequisites": [
    "hypothesis-testing",
    "likelihood-ratio-tests-and-np",
    "confidence-intervals-and-duality"
  ]
}
```

Also update the track-5 metadata block:

```json
// Change (FIND):
{
  "track": 5,
  "name": "Hypothesis Testing & Confidence",
  "topicsPublished": 3,
  "topicsTotal": 4
}

// Change (REPLACE WITH):
{
  "track": 5,
  "name": "Hypothesis Testing & Confidence",
  "topicsPublished": 4,
  "topicsTotal": 4,
  "status": "complete"
}
```

### 8.3 Updates to `curriculum.ts`

Export the new topic in the TypeScript curriculum module:

```typescript
// src/data/curriculum.ts
// Add to the track-5 topics array:
{
  slug: 'multiple-testing-and-false-discovery',
  title: 'Multiple Testing & False Discovery',
  status: 'published',
  // ...
},
```

### 8.4 Track-5 status surface

After Track 5 closure, the following surfaces should reflect "Track 5 complete — 4/4 published":

- Home page `/` track cards.
- Track-index page `/tracks/hypothesis-testing-and-confidence`.
- Topic-index listing for Track 5 on the sidebar navigation.

These may be controlled by the `curriculum-graph.json` `status: "complete"` flag; verify the UI picks this up. If not, a one-line update in the Astro component is needed.

### 8.5 Updates to `formalstatisticscitations.xlsx`

Add **9 new citation rows** (one per new paper) and **update 3 existing rows** (extend `usedIn` fields for LEH2005, CAS2002, and any existing IOA2005 / GEL2014 rows).

| Action | ID | Short | Full citation | usedIn | URL |
|---|---|---|---|---|---|
| Add | DUN1961 | Dunn 1961 | Dunn, Olive Jean. 1961. "Multiple Comparisons Among Means." Journal of the American Statistical Association 56 (293): 52–64. | Topic 20 §20.4 | https://doi.org/10.1080/01621459.1961.10482090 |
| Add | HOL1979 | Holm 1979 | Holm, Sture. 1979. "A Simple Sequentially Rejective Multiple Test Procedure." Scandinavian Journal of Statistics 6 (2): 65–70. | Topic 20 §20.5 | https://www.jstor.org/stable/4615733 |
| Add | SID1967 | Šidák 1967 | Šidák, Zbyněk. 1967. "Rectangular Confidence Regions for the Means of Multivariate Normal Distributions." Journal of the American Statistical Association 62 (318): 626–633. | Topic 20 §20.4, §20.9 | https://doi.org/10.1080/01621459.1967.10482935 |
| Add | HOC1988 | Hochberg 1988 | Hochberg, Yosef. 1988. "A Sharper Bonferroni Procedure for Multiple Tests of Significance." Biometrika 75 (4): 800–802. | Topic 20 §20.5 | https://doi.org/10.1093/biomet/75.4.800 |
| Add | BEN1995 | Benjamini-Hochberg 1995 | Benjamini, Yoav, and Yosef Hochberg. 1995. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." Journal of the Royal Statistical Society: Series B 57 (1): 289–300. | Topic 20 §20.6, §20.7 (featured) | https://doi.org/10.1111/j.2517-6161.1995.tb02031.x |
| Add | BEN2001 | Benjamini-Yekutieli 2001 | Benjamini, Yoav, and Daniel Yekutieli. 2001. "The Control of the False Discovery Rate in Multiple Testing under Dependency." Annals of Statistics 29 (4): 1165–1188. | Topic 20 §20.7 Proof 5, §20.8 | https://doi.org/10.1214/aos/1013699998 |
| Add | STO2002 | Storey 2002 | Storey, John D. 2002. "A Direct Approach to False Discovery Rates." Journal of the Royal Statistical Society: Series B 64 (3): 479–498. | Topic 20 §20.8 | https://doi.org/10.1111/1467-9868.00346 |
| Update | IOA2005 | Ioannidis 2005 | (existing) | Append: ", Topic 20 §20.8 Rem 13 (full quantitative PPV treatment)" | (existing) |
| Update | GEL2014 | Gelman-Loken 2014 | (existing) | Append: ", Topic 20 §20.1 Rem 2 (full garden-of-forking-paths framing)" | (existing) |
| Update | LEH2005 | Lehmann-Romano 2005 | (existing) | Append: ", Topic 20 §§20.2, 20.4, 20.5, 20.10 (Ch. 9)" | (existing) |
| Update | CAS2002 | Casella-Berger 2002 | (existing) | Append: ", Topic 20 §§20.3 Ex 3, 20.4 Ex 5" | (existing) |

---

## 9. Verification Checklist

### 9.1 Content verification

- [ ] All nine figures present in `public/images/topics/multiple-testing-and-false-discovery/` with correct filenames.
- [ ] Notebook executes end-to-end with seed 42 and reproduces all figures pixel-identically.
- [ ] MDX frontmatter parses; every `url:` resolves (paste into browser or run a link-check script).
- [ ] All three full proofs (Bonferroni, Holm, BH) render correctly in KaTeX. **No `\begin{aligned}` blocks** — verify by grepping the MDX for the string.
- [ ] Featured theorem (BH 1995 Thm 1) in §20.7 has the eight-step proof matching §3.2 Proof 5 specification.
- [ ] All 14 examples resolve with the correct numerical values.
- [ ] Word count in the 9,200–9,800 range.

### 9.2 Forward-promise verification

- [ ] All 12 find-and-replace operations in §8.1 completed.
- [ ] `grep -rn '/topics/multiple-testing-and-false-discovery' src/content/topics/ --include='*.mdx' | wc -l` returns **12**.
- [ ] `grep -rn 'coming soon, Topic 20' src/content/topics/ --include='*.mdx'` returns **0**.
- [ ] `grep -rn '/topics/multiple-testing[^-]' src/content/topics/ --include='*.mdx'` returns **0**.

### 9.3 Code verification

- [ ] `testing.ts` exports all functions listed in §6.1.
- [ ] Jest regression tests (`src/lib/stat/__tests__/testing.test.ts`) all pass. The six Topic-20 tests (T1–T6) each succeed on a clean install.
- [ ] `pnpm build` succeeds with no TypeScript errors.
- [ ] `pnpm lint` passes.
- [ ] Static-type check: `simulatePValues` return type matches `{ pvals: number[]; isNull: boolean[] }`.

### 9.4 Interactive components verification

For each of the four components, manually exercise:

- [ ] `MultipleTestingProcedureExplorer` — resample 20 times; verify stats table updates; verify `resample on change` toggle behaves correctly; verify all 7 procedures render; test on 375px viewport.
- [ ] `FWERvsFDRComparator` — run MC for each procedure; verify FDR curves stay near α·π₀ for BH; verify FWER stays near α for Bonferroni/Holm; verify power curves are monotone in δ.
- [ ] `BHAlgorithmVisualizer` — run animation on T1 preset; verify k*=3 highlighted; verify step log scrolls correctly; verify speed slider; test on mobile.
- [ ] `SimultaneousCIBands` — resample 100 times on m=10 preset; verify empirical joint coverage for Bonferroni ≥ 0.94; verify width ratio (Bonf/unadj) matches theoretical `z_{α/(2m)}/z_{α/2}`.

### 9.5 Deploy verification

- [ ] Local dev server: `pnpm dev` → navigate to `/topics/multiple-testing-and-false-discovery` → walk through every section.
- [ ] Verify all KaTeX displays render.
- [ ] Verify all four interactive components load (`client:visible`) and respond to interaction.
- [ ] Verify all internal cross-references (back to Topics 17, 18, 19; forward to Track 6/8 placeholders) resolve correctly.
- [ ] Commit on feature branch; push; open PR; confirm Vercel preview deploy succeeds.
- [ ] Production-URL smoke test after merge: navigate to `https://www.formalstatistics.com/topics/multiple-testing-and-false-discovery`.
- [ ] Pagefind reindex picks up the new topic (search "Benjamini-Hochberg" from the global search).
- [ ] `sitemap.xml` and `rss.xml` include the new topic URL.
- [ ] All 12 forward-promise links from Topics 17, 18, 19 now resolve correctly in production.

---

## 10. Build Order

Execute in order. Each step produces a verifiable intermediate artifact.

1. **Create feature branch.** `git checkout -b topic-20-multiple-testing-and-false-discovery`.
2. **Copy notebook figures into `public/`.** Run `20_multiple_testing.ipynb` end-to-end (or copy the notebook-produced PNGs from the reference run); verify all 9 files present in `public/images/topics/multiple-testing-and-false-discovery/`.
3. **Extend `testing.ts`** per §6.1. Append all new functions; run the new Jest tests (§6.2) in isolation; confirm all pass before moving on.
4. **Create `src/data/multi-testing-data.ts`** per §7.
5. **Extend `src/components/viz/shared/colorScales.ts`** with the Topic 20 palette additions (FWER purples, FDR greens, null/alt slate/amber). Match the hex codes in §4.
6. **Create the four interactive components** per §5, in order (MultipleTestingProcedureExplorer featured first, then the three others). For each: write the component, hook up presets, test responsiveness at 375px, verify smooth animation / debouncing.
7. **Draft `src/content/topics/multiple-testing-and-false-discovery.mdx`** per §2 frontmatter + §3 section map. Write section by section, inserting figure references and component `client:visible` imports where indicated in §3.1. All three full proofs in their §3.2-specified structure; no `\begin{aligned}` blocks.
8. **Run the grep-verification commands from §8.1** to confirm all forward-promise text is in place (awaiting the link-substitution step).
9. **Apply the 12 find-and-replace operations** from §8.1 to the three published MDX files. Re-run the grep-verification commands; counts must match specifications.
10. **Update `src/data/curriculum-graph.json`** and `src/data/curriculum.ts` per §8.2–§8.3.
11. **Update `formalstatisticscitations.xlsx`** per §8.5 — 7 new rows, 4 updated.
12. **Local dev verification.** `pnpm dev`, navigate to `/topics/multiple-testing-and-false-discovery`, walk through every section; verify KaTeX, interactive components, cross-reference links.
13. **Jest + build + lint.** `pnpm test`, `pnpm build`, `pnpm lint` — all three pass clean.
14. **Commit, push, open PR, merge, deploy.** Confirm Vercel deploy succeeds. Run production-URL smoke test.
15. **Post-deploy validation.** Verify Pagefind indexed the topic; verify sitemap and RSS include the new URL; verify all 12 cross-reference links from Topics 17, 18, 19 now resolve in production.
16. **Announce Track 5 closure.** Update home-page track cards if needed (§8.4).

---

## Appendix A: Style Guidelines

Topic 20 inherits all Topic 17–19 style conventions without modification. The critical reminders:

### KaTeX

- **No `\begin{aligned}` blocks.** Multi-line derivations use separate `$$…$$` blocks with prose connectors. Proof 1 has one display block; Proof 3 has two; Proof 5 has four connected by prose and step labels.
- **No `\begin{array}` tables.** Use Markdown tables. The §20.8 Example 11 (Ioannidis PPV) and §20.8 Rem 13 (procedure comparison) are Markdown tables.
- **Inline math** uses `$...$`. **Display math** uses `$$...$$` on its own line.
- **Test all LaTeX in the dev server before committing.** Particular attention: the blackboard-bold $\mathbb{E}$ in Proof 5; the multi-subscript $H_{(k)}$ notation; the variables $V / R / m_0 / \pi_0$ recurring throughout.

### Notation

- **Hypotheses:** $H_i$ indexed $i = 1, \ldots, m$. Closed-testing intersections: $H_S = \bigcap_{i \in S} H_i$ for $S \subseteq \{1, \ldots, m\}$.
- **P-values:** $p_i$ for hypothesis $H_i$. Sorted: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}$.
- **Counts:** $V$ = false discoveries, $S$ = true discoveries, $R = V + S$ = total rejections, $m_0$ = true nulls, $m_1 = m - m_0$ = true alternatives, $\pi_0 = m_0 / m$.
- **Error rates:** FWER = $\Pr(V \geq 1)$, FDR = $\mathbb{E}[V/\max(R, 1)]$, FDP = $V/\max(R, 1)$ (realized).
- **BH-specific:** $k^\star = \max\{k : p_{(k)} \leq k\alpha/m\}$ (zero if no such $k$).
- **Harmonic number:** $H_m = \sum_{k=1}^m 1/k$ (BY normalization; do not confuse with hypothesis $H$).

### Figure callouts

Inline figure references use the Markdown image syntax with square-bracket alt text that doubles as the figure caption:

```mdx
![Figure 5 — BH algorithm visualization: sorted p-values against rank, BH diagonal, k* circled.](/images/topics/multiple-testing-and-false-discovery/20-bh-algorithm.png)
```

The featured figure (Figure 5, BH algorithm) appears immediately after Theorem 5 statement in §20.7, before Proof 5.

### Cross-references

- Topic 17: `[Topic 17](/topics/hypothesis-testing)`, with sub-anchors `#section-17-X`.
- Topic 18: `[Topic 18](/topics/likelihood-ratio-tests-and-np)`, with sub-anchors `#section-18-X`.
- Topic 19: `[Topic 19](/topics/confidence-intervals-and-duality)`, with sub-anchors `#section-19-X`.
- Topics 13, 14, 16 (Track 4): live links using the existing conventions.
- formalcalculus.com and formalml.com: `<ExternalLink>` component with `site="formalcalculus"` or `site="formalml"`.
- Section anchors within Topic 20: `#section-20-X` format (lowercase, no dots); e.g., `[§20.7](/topics/multiple-testing-and-false-discovery#section-20-7)`.

### Proof termination

End each proof with `$\blacksquare$ — using [citations/theorems invoked]`. Example: `$\blacksquare$ — using BEN1995 Theorem 1 and BEN2001 Lemma 2.1.` This matches the Topic 17–19 convention exactly.

### References footer

Every topic MDX ends with:

```mdx
### References

[Chicago Notes & Bibliography citations, one per entry, with hyperlinks]

---
```

The `---` horizontal rule after References is **non-negotiable** (Topic 17 precedent).

---

## Appendix B: Design Decisions

1. **Ten sections (locked).** Ten sections match Topic 19's rhythm, keep the featured BH proof section (§20.7) uncluttered, and preserve §20.10 as a dedicated forward-look that closes the entire track.

2. **BH 1995 Theorem 1 as the featured theorem (locked).** Three alternatives were considered: (a) closed-testing principle as featured — rejected because it's too abstract and has no clean proof at intermediate difficulty; (b) Holm as featured — rejected because it's derivative of Bonferroni + closed-testing; (c) Benjamini-Yekutieli as featured — rejected because the proof is structurally identical to BH's but with an extra division by $H_m$, less pedagogically distinctive. BH 1995 wins on three criteria: historical significance (one of the most cited statistics papers of all time), pedagogical payoff (introduces the FDR framework as a distinct error metric), and proof accessibility (eight lines by direct computation under independence).

3. **Three full proofs, five stated-with-reference (locked).** The full-proof slate (Bonferroni, Holm, BH) covers the FWER single-step, FWER step-down, and FDR step-up archetypes. Šidák, Hochberg, BY, Storey, and closed-testing are all structurally derivative or involve heavier machinery; citing them to primary sources keeps the topic at 55 min.

4. **Four interactive components, all required (locked).** `SimultaneousCIBands` was originally optional but was promoted to 4th required based on the §19.10 Rem 19 #3 load-bearing promise — Topic 20 must deliver simultaneous CIs as a tangible, not just theoretical, artifact. The four-component density matches Topic 17 and exceeds Topic 19's three-required-plus-one-optional.

5. **Closed-testing principle stated, not proven (locked).** Marcus–Peritz–Gabriel (1976) proof is ~15 MDX lines at the intermediate level of rigor Topic 20 targets. Stating it with Figure 2 (the m=3 closure tree) and citing LEH2005 §9.1 for the proof is the right trade: the *idea* is pedagogically decisive, but the proof itself adds more machinery than insight.

6. **Nine figures, not ten or eight (locked).** Nine matches the notebook cell count (3–11). Ten would require either a second closed-testing diagram or a second genomics panel — neither adds pedagogical value. Eight would require dropping either the FDR/FWER/power comparison (non-negotiable for §20.8) or the forward-map (non-negotiable as the track-closure visual).

7. **Read time 55 minutes, matching Topics 18–19 (locked).** 10 sections × ~5.5 min/section = 55 min. The three full proofs (~7 min collectively) are the time-heaviest elements; the four interactive components add ~8 min of exploration time for engaged readers.

8. **Difficulty "intermediate," matching Topics 9–19.** All three proofs use only: the union bound (Topic 4 / probability); induction (standard); the probability integral transform (Topic 6); conditioning (Topic 3); the independence lemma (BEN2001, cited). No new analytical machinery.

9. **Storey adaptive FDR included but not featured (locked).** Storey's q-value framework is bioinformatics-standard (the `qvalue` Bioconductor package is the reference implementation) and must be stated, but its π̂₀-estimation machinery involves additional choices (tuning parameter λ, bootstrap variance) that distract from the cleaner BH narrative. §20.8 Thm 7 states the procedure; Rem 21 explains when to reach for it; the MultipleTestingProcedureExplorer component has it toggleable but off by default.

10. **Simultaneous-CI section (§20.9) included despite being "CI-duality territory."** The §19.10 Rem 19 #3 promise is load-bearing: Topic 19 explicitly deferred simultaneous CIs to Topic 20. Including §20.9 with Thm 8 + Ex 13–14 + Figure 8 + SimultaneousCIBands closes this debt completely.

11. **Scope deferrals to formalml.com and Track 8.** Always-valid sequential inference, e-processes, online FDR, and knockoffs are modern frontiers that deserve full treatment — but not here. §20.10 Rems 26–28 state each precisely, cite the canonical primary sources (Howard–Ramdas–McAuliffe 2021, Barber–Candès 2015, Javanmard–Montanari 2018), and point to where the full treatment will live.

12. **References include Dunn (1961), Holm (1979), Šidák (1967), Hochberg (1988), Benjamini–Hochberg (1995), Benjamini–Yekutieli (2001), Storey (2002), Ioannidis (2005), Gelman–Loken (2014), Lehmann–Romano (2005), Casella–Berger (2002).** Nine new references (the procedure originators plus the replication-crisis literature) plus extended usages for the two Track 5 workhorse references. Every `url:` is a DOI or JSTOR canonical link.

---

*Brief version: v1 | Created: 2026-04-18 | Author: Jonathan Rocha*
*Reference notebook: `notebooks/multiple-testing-and-false-discovery/20_multiple_testing.ipynb`*
*Closes Track 5: Hypothesis Testing & Confidence (4/4 topics published).*
