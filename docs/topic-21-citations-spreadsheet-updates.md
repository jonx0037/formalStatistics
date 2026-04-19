# Topic 21 — Citations Spreadsheet Updates

**Target file:** `docs/formalstatistics-citations.xlsx`
**Applies to:** Topic 21 — Simple & Multiple Linear Regression

This artifact lists the rows to add or update in the master citations spreadsheet for Topic 21. Paste each new-row entry as a new line in the spreadsheet; for the two existing rows, append `linear-regression` to the existing `Used In Topics` column.

Verify counts against the current spreadsheet before applying — if any "new" citation already exists (e.g., if SEB2003 was already added for Topic 16 exponential-family supplementary material), move it from the **New rows** list to the **Update rows** list.

---

## 7 new rows

Column order follows the existing spreadsheet schema. Adjust as needed if the real columns differ.

### SEB2003 — Seber & Lee (2003)

| Field | Value |
|---|---|
| ID | SEB2003 |
| Type | Book |
| Author(s) | George A. F. Seber, Alan J. Lee |
| Year | 2003 |
| Title | Linear Regression Analysis |
| Edition | 2nd |
| Publisher | Wiley |
| Journal | — |
| Volume / Issue | — |
| Pages | — |
| URL | https://onlinelibrary.wiley.com/doi/book/10.1002/9780471722199 |
| ISBN | 978-0-471-41540-4 |
| Used In Topics | linear-regression |
| Note | Normal-linear-model reference. Ch. 3 Gauss–Markov + hat matrix; Ch. 4 F-test + simultaneous inference; Ch. 10 diagnostics. |

### WEI2005 — Weisberg (2005)

| Field | Value |
|---|---|
| ID | WEI2005 |
| Type | Book |
| Author(s) | Sanford Weisberg |
| Year | 2005 |
| Title | Applied Linear Regression |
| Edition | 3rd |
| Publisher | Wiley |
| Journal | — |
| Volume / Issue | — |
| Pages | — |
| URL | — |
| ISBN | 978-0-471-66379-9 |
| Used In Topics | linear-regression |
| Note | Canonical undergraduate treatment. §5.1 leverage + Cook's distance; §8.1–§8.2 residual-plot diagnostics. |

### GRE2012 — Greene (2012)

| Field | Value |
|---|---|
| ID | GRE2012 |
| Type | Book |
| Author(s) | William H. Greene |
| Year | 2012 |
| Title | Econometric Analysis |
| Edition | 7th |
| Publisher | Pearson |
| Journal | — |
| Volume / Issue | — |
| Pages | — |
| URL | — |
| ISBN | 978-0-13-139538-1 |
| Used In Topics | linear-regression |
| Note | Econometric OLS treatment. Ch. 3–4 standard coverage; §4.9 heteroscedasticity + HC-robust SEs — motivates Topic 22 sandwich estimators. |

### WOR1929 — Working & Hotelling (1929)

| Field | Value |
|---|---|
| ID | WOR1929 |
| Type | Paper |
| Author(s) | Holbrook Working, Harold Hotelling |
| Year | 1929 |
| Title | Applications of the Theory of Error to the Interpretation of Trends |
| Edition | — |
| Publisher | — |
| Journal | Journal of the American Statistical Association |
| Volume / Issue | 24(165A) |
| Pages | 73–85 |
| URL | https://www.jstor.org/stable/2277011 |
| ISBN | — |
| Used In Topics | linear-regression |
| Note | Originating paper for the simultaneous confidence band over regression coefficient trajectories via the F-distribution. §21.8 Rem 16. |

### GAL1886 — Galton (1886)

| Field | Value |
|---|---|
| ID | GAL1886 |
| Type | Paper |
| Author(s) | Francis Galton |
| Year | 1886 |
| Title | Regression Towards Mediocrity in Hereditary Stature |
| Edition | — |
| Publisher | — |
| Journal | Journal of the Anthropological Institute of Great Britain and Ireland |
| Volume / Issue | 15 |
| Pages | 246–263 |
| URL | https://www.jstor.org/stable/2841583 |
| ISBN | — |
| Used In Topics | linear-regression |
| Note | The paper that gave "regression" its name. §21.1 Rem 2 historical note. |

### GAU1823 — Gauss (1823)

| Field | Value |
|---|---|
| ID | GAU1823 |
| Type | Book |
| Author(s) | Carl Friedrich Gauss |
| Year | 1823 |
| Title | Theoria Combinationis Observationum Erroribus Minimis Obnoxiae |
| Edition | — |
| Publisher | Werke, IV |
| Journal | — |
| Volume / Issue | — |
| Pages | 1–108 |
| URL | — |
| ISBN | — |
| Used In Topics | linear-regression |
| Note | Original source of both least-squares and the Gauss–Markov theorem. §21.6 joint attribution. |

### MAR1900 — Markov (1900)

| Field | Value |
|---|---|
| ID | MAR1900 |
| Type | Book |
| Author(s) | Andrey A. Markov |
| Year | 1900 |
| Title | Wahrscheinlichkeitsrechnung |
| Edition | — |
| Publisher | Teubner |
| Journal | — |
| Volume / Issue | — |
| Pages | — |
| URL | — |
| ISBN | — |
| Used In Topics | linear-regression |
| Note | Markov's probabilistic formalization and extension of Gauss's least-squares theorem. Completes the joint Gauss–Markov attribution. §21.6. |

---

## 2 existing rows — append `linear-regression` to Used In Topics

### LEH2005 — Lehmann & Romano (2005)

- **Action:** Append `linear-regression` to the existing `Used In Topics` column.
- **After update** `Used In Topics` should include (in addition to whatever is already there): `hypothesis-testing, likelihood-ratio-tests-and-np, confidence-intervals-and-duality, multiple-testing-and-false-discovery, linear-regression`.
- **Rationale** §21.8 Thm 9 Proof 9 and §21.8 Rem 16 both cite LEH2005 for the F-test and simultaneous-inference treatments.

### CAS2002 — Casella & Berger (2002)

- **Action:** Append `linear-regression` to the existing `Used In Topics` column.
- **After update** `Used In Topics` should include: `hypothesis-testing, likelihood-ratio-tests-and-np, confidence-intervals-and-duality, multiple-testing-and-false-discovery, linear-regression`.
- **Rationale** §21.5 Thms 4–5 brief derivations and §21.6 Gauss–Markov are written against CAS2002 Ch. 11–12.

---

## Verification checklist after spreadsheet edits

- [ ] All 7 new rows visible in the spreadsheet with correct column placement.
- [ ] LEH2005 and CAS2002 "Used In Topics" each contain `linear-regression`.
- [ ] No duplicate IDs in the spreadsheet (grep via Excel filter).
- [ ] Any "new" row whose ID already existed in the spreadsheet was moved to the Update list and its `Used In Topics` column appended, not overwritten.
