# formalStatistics — Citations Audit & Remediation Handoff Brief

**Date:** April 13, 2026
**Author:** Jonathan Rocha
**Type:** Cross-topic audit and remediation
**Priority:** Critical — academic credibility issue

---

## 1. Problem Statement

Eight of ten published MDX topics on formalStatistics.com are **missing rendered `### References` sections** at the bottom of the page. The references exist in YAML frontmatter but are never rendered as visible, reader-facing citations. Additionally, the two topics that *do* have rendered references use inconsistent formatting.

This is a credibility issue. formalStatistics positions itself as a rigorous resource for grad students, ML practitioners, and researchers. Every topic cites 5–8 authoritative sources in its frontmatter, but the reader never sees them. A site claiming academic rigor without visible citations undermines its own authority.

---

## 2. Audit Results

### 2.1 Per-Topic Status

| # | Topic slug | Frontmatter refs | URLs in frontmatter | Rendered `### References`? | Format |
|---|---|---|---|---|---|
| 1 | `sample-spaces` | 5 | ❌ None | ✅ Present | ❌ Plain text (no hyperlinks) |
| 2 | `conditional-probability` | 7 | ✅ All 5 book refs | ✅ Present | ✅ Hyperlinked (gold standard) |
| 3 | `random-variables` | 8 | ✅ All refs | ❌ **MISSING** | — |
| 4 | `expectation-moments` | 8 | ❌ None | ❌ **MISSING** | — |
| 5 | `discrete-distributions` | 8 | ❌ None | ❌ **MISSING** | — |
| 6 | `continuous-distributions` | 8 | ❌ None | ❌ **MISSING** | — |
| 7 | `exponential-families` | 6 | ✅ All refs | ❌ **MISSING** | — |
| 8 | `multivariate-distributions` | 6 | ✅ All refs | ❌ **MISSING** | — |
| 9 | `modes-of-convergence` | 6 | ✅ All refs | ❌ **MISSING** | — |
| 10 | `law-of-large-numbers` | 6 | ✅ All refs | ❌ **MISSING** | — |

### 2.2 Root Cause

The handoff briefs for Topics 1–7 included "references" in their section outline tables (e.g., `| Summary | "Summary" | Summary table, references |`), but the instruction was insufficiently specific — it was treated as "the frontmatter handles references" rather than "render a visible references section." Topics 8–10 dropped the word "references" from their section outlines entirely.

The frontmatter `references:` array was always populated. The data is there. It was never rendered.

### 2.3 Formatting Inconsistencies

Even between the two topics that have rendered references:

- **`sample-spaces`** uses plain text with no hyperlinks and no ISBNs:
  ```
  1. Billingsley, P. (2012). *Probability and Measure* (Anniversary ed.). Wiley.
  ```
  Its frontmatter also lacks `url:` and `isbn:` fields entirely.

- **`conditional-probability`** uses hyperlinked titles:
  ```
  1. Billingsley, P. (2012). [*Probability and Measure*](https://...) (Anniversary ed.). Wiley.
  ```
  Its frontmatter includes `url:` fields.

The **`conditional-probability` format is the gold standard** and should be adopted across all topics.

---

## 3. Remediation Tasks

### 3.1 Phase 1 — Add Missing URLs to Frontmatter

Four topics (`sample-spaces`, `expectation-moments`, `discrete-distributions`, `continuous-distributions`) have frontmatter references without `url:` fields. Add URLs using the citations spreadsheet (`src/data/formalstatisticscitations.xlsx`) and the URL values already present in other topics' frontmatter for the same books. Below is the canonical URL for every recurring reference:

| Reference | URL |
|---|---|
| Billingsley (2012) | `https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372` |
| Durrett (2019) | `https://services.math.duke.edu/~rtd/PTE/pte.html` |
| Grimmett & Stirzaker (2020) | `https://global.oup.com/academic/product/probability-and-random-processes-9780198847595` |
| Wasserman (2004) | `https://link.springer.com/book/10.1007/978-0-387-21736-9` |
| Shalev-Shwartz & Ben-David (2014) | `https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/` |
| Bishop (2006) | `https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/` |
| Casella & Berger (2002) | `https://www.cengage.com/c/statistical-inference-2e-casella/` |
| Goodfellow, Bengio & Courville (2016) | `https://www.deeplearningbook.org/` |
| Kobyzev, Prince & Brubaker (2021) | `https://doi.org/10.1109/TPAMI.2020.2992934` |
| Koller & Friedman (2009) | `https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/` |
| Hastie, Tibshirani & Friedman (2009) | `https://hastie.su.domains/ElemStatLearn/` |
| Blei, Kucukelbir & McAuliffe (2017) | `https://doi.org/10.1080/01621459.2017.1285773` |
| McCullagh & Nelder (1989) | `https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606` |
| Hilbe (2011) | `https://www.cambridge.org/core/books/negative-binomial-regression/3FB8C87B4C120558E979B3C8B62E0523` |
| Gelman et al. (2013) | `http://www.stat.columbia.edu/~gelman/book/` |
| Bickel & Doksum (2015) | `https://www.routledge.com/Mathematical-Statistics-Basic-Ideas-and-Selected-Topics/Bickel-Doksum/p/book/9781498723800` |
| Barndorff-Nielsen (2014) | `https://www.wiley.com/en-us/Information+and+Exponential+Families+in+Statistical+Theory-p-9781118857502` |
| Efron (2022) | `https://www.cambridge.org/us/academic/subjects/statistics-probability/statistical-theory-and-methods/exponential-families-theory-and-practice` |
| DeGroot & Schervish (2012) | `https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistics/P200000006218` |
| Nelsen (2006) | `https://link.springer.com/book/10.1007/0-387-28678-0` |
| Blei, Ng & Jordan (2003) | `https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf` |
| Resnick (2014) | `https://link.springer.com/book/10.1007/978-0-8176-8409-9` |
| Etemadi (1981) | `https://doi.org/10.1007/BF01013465` |

**Action for each of the 4 topics missing URLs:** Add `url:` and `isbn:` fields to every reference entry in the YAML frontmatter, using the canonical values above.

### 3.2 Phase 2 — Add Rendered `### References` Sections

For **all 8 topics missing the section** (`random-variables` through `law-of-large-numbers`), append a `### References` section at the very end of the MDX file (after the Summary / "What's Next" content).

**Format specification (matching `conditional-probability` gold standard):**

For **books:**
```markdown
N. Author(s). (Year). [*Title*](URL) (Edition). Publisher.
```

For **journal articles:**
```markdown
N. Author(s). (Year). [*Title*](URL). *Journal*, Volume(Issue), Pages.
```

For **books without edition:**
```markdown
N. Author(s). (Year). [*Title*](URL). Publisher.
```

**Concrete example from conditional-probability (the gold standard):**
```markdown
### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/pte.html) (5th ed.). Cambridge University Press.
```

**Rules:**
- The `### References` heading goes after a `---` horizontal rule, at the very bottom of the file
- The section is the **last content in the file** — nothing follows it
- Number references sequentially starting from 1
- Include **every** reference listed in that topic's frontmatter `references:` array, in the same order
- Title is italicized and hyperlinked to the `url:` value
- Edition is in parentheses after the title, only if present
- For articles: include journal name (italicized), volume, issue, and pages

### 3.3 Phase 3 — Fix `sample-spaces` References

The existing `### References` section in `sample-spaces.mdx` uses plain text without hyperlinks. After Phase 1 adds URLs to its frontmatter, **replace** the existing references section with the hyperlinked format:

**Current (lines 524–529 of `sample-spaces.mdx`):**
```markdown
### References

1. Billingsley, P. (2012). *Probability and Measure* (Anniversary ed.). Wiley.
2. Durrett, R. (2019). *Probability: Theory and Examples* (5th ed.). Cambridge University Press.
3. Grimmett, G. & Stirzaker, D. (2020). *Probability and Random Processes* (4th ed.). Oxford University Press.
4. Wasserman, L. (2004). *All of Statistics*. Springer.
5. Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.
```

**Replace with:**
```markdown
### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/pte.html) (5th ed.). Cambridge University Press.
3. Grimmett, G. & Stirzaker, D. (2020). [*Probability and Random Processes*](https://global.oup.com/academic/product/probability-and-random-processes-9780198847595) (4th ed.). Oxford University Press.
4. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
5. Shalev-Shwartz, S. & Ben-David, S. (2014). [*Understanding Machine Learning: From Theory to Algorithms*](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). Cambridge University Press.
```

### 3.4 Phase 4 — Verify `conditional-probability` References (No Changes Expected)

`conditional-probability.mdx` is the gold standard. Verify that its rendered references match its frontmatter — no action expected, just confirmation.

---

## 4. Per-Topic Reference Lists to Render

Below is the exact `### References` section to append to each topic. These are derived directly from each topic's YAML frontmatter `references:` array with URLs from the canonical table above.

### 4.1 `random-variables.mdx`

Append after the final paragraph (the "What's Next" section):

```markdown

---

### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/pte.html) (5th ed.). Cambridge University Press.
3. Grimmett, G. & Stirzaker, D. (2020). [*Probability and Random Processes*](https://global.oup.com/academic/product/probability-and-random-processes-9780198847595) (4th ed.). Oxford University Press.
4. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
5. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/) (2nd ed.). Duxbury.
6. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/). Springer.
7. Goodfellow, I., Bengio, Y. & Courville, A. (2016). [*Deep Learning*](https://www.deeplearningbook.org/). MIT Press.
8. Kobyzev, I., Prince, S. J. D. & Brubaker, M. A. (2021). [*Normalizing Flows: An Introduction and Review of Current Methods*](https://doi.org/10.1109/TPAMI.2020.2992934). *IEEE TPAMI*, 43(11), 3964–3979.
```

### 4.2 `expectation-moments.mdx`

Append after the final paragraph:

```markdown

---

### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/pte.html) (5th ed.). Cambridge University Press.
3. Grimmett, G. & Stirzaker, D. (2020). [*Probability and Random Processes*](https://global.oup.com/academic/product/probability-and-random-processes-9780198847595) (4th ed.). Oxford University Press.
4. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
5. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/) (2nd ed.). Duxbury.
6. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/). Springer.
7. Hastie, T., Tibshirani, R. & Friedman, J. (2009). [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) (2nd ed.). Springer.
8. Blei, D. M., Kucukelbir, A. & McAuliffe, J. D. (2017). [*Variational Inference: A Review for Statisticians*](https://doi.org/10.1080/01621459.2017.1285773). *JASA*, 112(518), 859–877.
```

### 4.3 `discrete-distributions.mdx`

```markdown

---

### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/pte.html) (5th ed.). Cambridge University Press.
3. Grimmett, G. & Stirzaker, D. (2020). [*Probability and Random Processes*](https://global.oup.com/academic/product/probability-and-random-processes-9780198847595) (4th ed.). Oxford University Press.
4. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
5. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/) (2nd ed.). Duxbury.
6. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/). Springer.
7. McCullagh, P. & Nelder, J. A. (1989). [*Generalized Linear Models*](https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606) (2nd ed.). Chapman & Hall/CRC.
8. Hilbe, J. M. (2011). [*Negative Binomial Regression*](https://www.cambridge.org/core/books/negative-binomial-regression/3FB8C87B4C120558E979B3C8B62E0523) (2nd ed.). Cambridge University Press.
```

### 4.4 `continuous-distributions.mdx`

```markdown

---

### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/pte.html) (5th ed.). Cambridge University Press.
3. Grimmett, G. & Stirzaker, D. (2020). [*Probability and Random Processes*](https://global.oup.com/academic/product/probability-and-random-processes-9780198847595) (4th ed.). Oxford University Press.
4. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
5. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/) (2nd ed.). Duxbury.
6. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/). Springer.
7. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013). [*Bayesian Data Analysis*](http://www.stat.columbia.edu/~gelman/book/) (3rd ed.). Chapman & Hall/CRC.
8. McCullagh, P. & Nelder, J. A. (1989). [*Generalized Linear Models*](https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606) (2nd ed.). Chapman & Hall/CRC.
```

### 4.5 `exponential-families.mdx`

```markdown

---

### References

1. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/) (2nd ed.). Duxbury.
2. Bickel, P. J. & Doksum, K. A. (2015). [*Mathematical Statistics: Basic Ideas and Selected Topics*](https://www.routledge.com/Mathematical-Statistics-Basic-Ideas-and-Selected-Topics/Bickel-Doksum/p/book/9781498723800) (2nd ed.). Chapman and Hall/CRC.
3. McCullagh, P. & Nelder, J. A. (1989). [*Generalized Linear Models*](https://www.routledge.com/Generalized-Linear-Models/McCullagh-Nelder/p/book/9780412317606) (2nd ed.). Chapman and Hall.
4. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/). Springer.
5. Barndorff-Nielsen, O. E. (2014). [*Information and Exponential Families in Statistical Theory*](https://www.wiley.com/en-us/Information+and+Exponential+Families+in+Statistical+Theory-p-9781118857502) (2nd ed.). Wiley.
6. Efron, B. (2022). [*Exponential Families in Theory and Practice*](https://www.cambridge.org/us/academic/subjects/statistics-probability/statistical-theory-and-methods/exponential-families-theory-and-practice). Cambridge University Press.
```

### 4.6 `multivariate-distributions.mdx`

```markdown

---

### References

1. DeGroot, M. H. & Schervish, M. J. (2012). [*Probability and Statistics*](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistics/P200000006218) (4th ed.). Pearson.
2. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/) (2nd ed.). Cengage.
3. Nelsen, R. B. (2006). [*An Introduction to Copulas*](https://link.springer.com/book/10.1007/0-387-28678-0) (2nd ed.). Springer.
4. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin, D. B. (2013). [*Bayesian Data Analysis*](http://www.stat.columbia.edu/~gelman/book/) (3rd ed.). CRC Press.
5. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/). Springer.
6. Blei, D. M., Ng, A. Y. & Jordan, M. I. (2003). [*Latent Dirichlet Allocation*](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf). *JMLR*, 3, 993–1022.
```

### 4.7 `modes-of-convergence.mdx`

```markdown

---

### References

1. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
2. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf) (5th ed.). Cambridge University Press.
3. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/) (2nd ed.). Cengage.
4. Resnick, S. (2014). [*A Probability Path*](https://link.springer.com/book/10.1007/978-0-8176-8409-9). Birkhäuser.
5. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
6. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/). Springer.
```

### 4.8 `law-of-large-numbers.mdx`

```markdown

---

### References

1. Durrett, R. (2019). [*Probability: Theory and Examples*](https://services.math.duke.edu/~rtd/PTE/PTE5_011119.pdf) (5th ed.). Cambridge University Press.
2. Billingsley, P. (2012). [*Probability and Measure*](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) (Anniversary ed.). Wiley.
3. Casella, G. & Berger, R. L. (2002). [*Statistical Inference*](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/) (2nd ed.). Cengage.
4. Etemadi, N. (1981). [*An elementary proof of the strong law of large numbers*](https://doi.org/10.1007/BF01013465). *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 55, 119–122.
5. Wasserman, L. (2004). [*All of Statistics*](https://link.springer.com/book/10.1007/978-0-387-21736-9). Springer.
6. Bishop, C. M. (2006). [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/). Springer.
```

---

## 5. Long-Term Recommendation: Automated Reference Rendering

Currently, references are duplicated — once as structured YAML in frontmatter, once as hand-written Markdown at the bottom. This is fragile. If a URL or edition changes, both locations must be updated.

**Recommended future improvement:** Create an Astro component (`src/components/ui/References.astro`) that reads the current page's frontmatter `references:` array and renders the formatted reference list automatically. This would:

1. Eliminate duplication between frontmatter and rendered content
2. Ensure formatting consistency across all topics automatically
3. Make it impossible for future topics to ship without visible citations
4. Allow style changes (e.g., adding ISBNs, DOIs, or annotation notes) in one place

**Sketch of the component:**
```astro
---
// src/components/ui/References.astro
const { references } = Astro.props;
---
{references && references.length > 0 && (
  <>
    <hr />
    <h3>References</h3>
    <ol>
      {references.map((ref) => (
        <li>
          {ref.author} ({ref.year}).{' '}
          {ref.url ? (
            <a href={ref.url} target="_blank" rel="noopener noreferrer">
              <em>{ref.title}</em>
            </a>
          ) : (
            <em>{ref.title}</em>
          )}
          {ref.edition && ` (${ref.edition}).`}
          {' '}{ref.publisher}.
        </li>
      ))}
    </ol>
  </>
)}
```

**This component is NOT part of the current remediation.** The current task is the manual fix described in Phases 1–4 above. The component can be built as a follow-up task and retroactively applied to replace the hand-written sections.

---

## 6. Implementation Checklist

### Phase 1 — Frontmatter URL Backfill
- [ ] `sample-spaces.mdx` — add `url:` and `isbn:` to all 5 references
- [ ] `expectation-moments.mdx` — add `url:` to all 8 references
- [ ] `discrete-distributions.mdx` — add `url:` to all 8 references
- [ ] `continuous-distributions.mdx` — add `url:` to all 8 references

### Phase 2 — Append `### References` Sections
- [ ] `random-variables.mdx` — append section (8 references)
- [ ] `expectation-moments.mdx` — append section (8 references)
- [ ] `discrete-distributions.mdx` — append section (8 references)
- [ ] `continuous-distributions.mdx` — append section (8 references)
- [ ] `exponential-families.mdx` — append section (6 references)
- [ ] `multivariate-distributions.mdx` — append section (6 references)
- [ ] `modes-of-convergence.mdx` — append section (6 references)
- [ ] `law-of-large-numbers.mdx` — append section (6 references)

### Phase 3 — Fix Existing
- [ ] `sample-spaces.mdx` — replace plain-text references with hyperlinked format

### Phase 4 — Verify
- [ ] `conditional-probability.mdx` — confirm rendered references match frontmatter (no changes expected)

### Post-Implementation Verification
- [ ] Run `pnpm build` — confirm no MDX parse errors
- [ ] Spot-check 3 topics in browser — confirm `### References` renders at bottom of page
- [ ] Confirm all hyperlinks resolve (no 404s)
- [ ] Confirm reference order matches frontmatter order in each topic
- [ ] Confirm no duplicate references within any single topic

### Process Fix (Prevent Recurrence)
- [ ] Update `CLAUDE.md` — add "Every topic MUST end with a rendered `### References` section" to Content Conventions
- [ ] Update handoff brief template — add explicit `### References` rendering instruction to the section outline specification

---

## 7. Files Modified

| File | Phase | Action |
|---|---|---|
| `src/content/topics/sample-spaces.mdx` | 1, 3 | Add URLs to frontmatter; replace plain-text references with hyperlinked |
| `src/content/topics/conditional-probability.mdx` | 4 | Verify only |
| `src/content/topics/random-variables.mdx` | 2 | Append `### References` section |
| `src/content/topics/expectation-moments.mdx` | 1, 2 | Add URLs to frontmatter; append `### References` section |
| `src/content/topics/discrete-distributions.mdx` | 1, 2 | Add URLs to frontmatter; append `### References` section |
| `src/content/topics/continuous-distributions.mdx` | 1, 2 | Add URLs to frontmatter; append `### References` section |
| `src/content/topics/exponential-families.mdx` | 2 | Append `### References` section |
| `src/content/topics/multivariate-distributions.mdx` | 2 | Append `### References` section |
| `src/content/topics/modes-of-convergence.mdx` | 2 | Append `### References` section |
| `src/content/topics/law-of-large-numbers.mdx` | 2 | Append `### References` section |
| `CLAUDE.md` | Process fix | Add citation rendering convention |

**Note:** All MDX files live in `src/content/topics/` in the deployed codebase. The paths in this handoff brief reference the files by their slug names for clarity.
