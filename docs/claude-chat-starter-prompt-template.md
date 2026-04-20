# Claude Chat Starter Prompt — TEMPLATE

Ask me clarification questions first, and whenever you have them during the process.

## How to use this template

1. **Find-and-replace all `{{PLACEHOLDER}}` tokens** using the legend below.
2. **Fill in the `[INSTRUCTION: ...]` blocks** with topic-specific content.
3. **Delete this top matter** (everything above the `═══` divider) before pasting to Claude Chat.
4. **Paste the result** as the opening message of a fresh Claude Chat conversation. Use the formalStatistics project workspace so Claude can index the codebase.

The template assumes Claude Chat will produce **at least the handoff brief** during the session. The notebook may be drafted in the same session or split into a follow-up — both work.

---

## Placeholder legend

### Topic identifiers (always required)

| Placeholder | Meaning | Example |
|---|---|---|
| `{{TOPIC_NUMBER}}` | Topic index in the 32-topic curriculum | `20` |
| `{{TOPIC_TITLE}}` | Working title (locked or provisional) | `Multiple Testing & False Discovery` |
| `{{PROVISIONAL_SLUG}}` | Best-guess kebab-case slug — actual slug is a session decision | `multiple-testing-and-false-discovery` |
| `{{TRACK_NUMBER}}` | Track index (1–8) | `5` |
| `{{TRACK_NAME}}` | Track human-readable name | `Hypothesis Testing & Confidence` |
| `{{POSITION_IN_TRACK}}` | Ordinal position with total | `4th of 4 (track closer)` |

### Curriculum state at session start

| Placeholder | Meaning | Example |
|---|---|---|
| `{{LAST_SHIPPED_TOPIC}}` | Most recently merged topic | `Topic 19: Confidence Intervals & Duality` |
| `{{LAST_SHIPPED_PR_URL}}` | URL of that PR | `https://github.com/jonx0037/formalStatistics/pull/21` |
| `{{TRACK_PUBLISHED_COUNT}}` | "X of Y" status for the current track | `3 of 4 published` |
| `{{TOTAL_PUBLISHED_COUNT}}` | Total topics shipped across all tracks | `19 of 32` |
| `{{NEXT_TRACK_OPENER}}` | If this topic closes a track, the next track's opener (for forward pointers) | `Track 6 opens with Linear Regression` |

### Predecessor & relevant prior topics

| Placeholder | Meaning | Example |
|---|---|---|
| `{{PRIMARY_PREDECESSOR}}` | Topic this one most directly extends | `Topic 19: Confidence Intervals & Duality` |
| `{{PRIMARY_PREDECESSOR_SLUG}}` | Slug of the primary predecessor | `confidence-intervals-and-duality` |
| `{{OTHER_RELEVANT_TOPICS}}` | Other shipped topics whose machinery this one consumes | `Topic 17 (testing framework), Topic 18 (LRT/optimality)` |
| `{{RELEVANT_NOTEBOOKS}}` | Existing notebooks worth studying for figure/MC patterns | `19_confidence_intervals.ipynb, 18_likelihood_ratio_tests.ipynb` |

### Forward-promise harvest

| Placeholder | Meaning | Example |
|---|---|---|
| `{{FORWARD_PROMISE_GREP}}` | Grep pattern that finds this topic's name in prior MDX files | `'Topic 20\|multiple testing\|false discovery'` |
| `{{N_FORWARD_PROMISES}}` | Count of forward-promises to fulfill (filled in mid-session) | `4` |

### Optional pre-fills

These are not required but tighten the conversation if you already have an opinion:

| Placeholder | Meaning |
|---|---|
| `{{INITIAL_FEATURED_THEOREM_GUESS}}` | Theorem you suspect should be the topic's flagship (subject to discussion) |
| `{{INITIAL_SCOPE_BOUNDARIES}}` | Topics you already plan to defer (e.g., "always-valid sequential, conformal, knockoffs") |

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     Everything above this line is template metadata and should be deleted
     before pasting to Claude Chat. Everything below is the starter prompt.
     ═══════════════════════════════════════════════════════════════════════════ -->

---

# Claude Chat Brief-Drafting Session — Topic {{TOPIC_NUMBER}}: {{TOPIC_TITLE}}

I'm drafting the **handoff brief and Jupyter notebook** for **Topic {{TOPIC_NUMBER}}** of [formalstatistics.com](https://www.formalstatistics.com). The output of this session goes downstream to a Claude Code agent that will implement the MDX, components, and cross-references against the brief — so the brief needs to be authoritative enough that downstream decisions are mechanical.

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

- **{{TOTAL_PUBLISHED_COUNT}}** topics shipped across the 8 tracks.
- **Track {{TRACK_NUMBER}} ({{TRACK_NAME}})** status: {{TRACK_PUBLISHED_COUNT}}.
- **Most recent ship:** {{LAST_SHIPPED_TOPIC}} — see {{LAST_SHIPPED_PR_URL}}.
- **Topic {{TOPIC_NUMBER}}** is **{{POSITION_IN_TRACK}}**.
- **Primary predecessor:** {{PRIMARY_PREDECESSOR}} (slug `{{PRIMARY_PREDECESSOR_SLUG}}`).
- **Other shipped topics this one consumes:** {{OTHER_RELEVANT_TOPICS}}.
- **Existing notebooks worth studying for figure / MC / preset patterns:** {{RELEVANT_NOTEBOOKS}}. These are the prior notebooks whose cell structure, palette choices, MC simulation conventions, and verification helpers are most directly transferrable to Topic {{TOPIC_NUMBER}}'s notebook. Cite them by path so the Claude Chat session can `Read` them directly to study before drafting §4.

<!-- [INSTRUCTION: If this topic closes a track, add a line: "This topic closes Track {{TRACK_NUMBER}}. {{NEXT_TRACK_OPENER}}." If it opens a track, add a line about what came before in the prior track and what infrastructure is being introduced fresh.] -->

---

## Topic to develop — Topic {{TOPIC_NUMBER}}: {{TOPIC_TITLE}}

**Working slug:** `{{PROVISIONAL_SLUG}}` (decision to confirm — see "Decisions to lock" below).

**One-paragraph thesis** (subject to revision in the conversation):

<!-- [INSTRUCTION: Write 3-5 sentences capturing what this topic is, who it's for, what it builds on, and where it leads. This is the seed for the brief's "Important: Nth Topic — Position in Track" preamble. Example for Topic 19: "Topic 19 is the collection: every family of level-α tests is a (1−α) confidence procedure, and every confidence procedure is a family of tests. The exposition constructs the z, t, χ², F, Wald, Score, LRT, Wilson, Clopper-Pearson, and profile-likelihood intervals as a single pattern — test inversion. Scalar θ throughout."] -->

**Initial featured-theorem guess:** {{INITIAL_FEATURED_THEOREM_GUESS}}

**Initial scope boundaries** (deferrals to lock early): {{INITIAL_SCOPE_BOUNDARIES}}

---

## What I want from this conversation

**Primary deliverable:** a complete handoff brief in markdown, structured per the established 10-section template (see "Brief structure" below). This brief becomes `docs/formalstatistics-{{PROVISIONAL_SLUG}}-handoff-brief.md` in the repo and is the single source of truth for the downstream Claude Code implementation session.

**Secondary deliverable:** a Jupyter notebook outline (cells specified, full code if time permits) that produces the brief's figure manifest. Lives at `notebooks/{{PROVISIONAL_SLUG}}/{{TOPIC_NUMBER}}_{topic_module}.ipynb`.

**Optional tertiary deliverable:** a scaffold document (`docs/topic-{{TOPIC_NUMBER}}-scaffold.md`) — a quick-reference companion to the brief with locked decision rationale and citation shortlist. Useful but not strictly required.

**By session end, I should be able to:**
1. Save the brief to `docs/`.
2. Save the notebook (or its outline) to `notebooks/{{PROVISIONAL_SLUG}}/`.
3. Hand both to a Claude Code session via the companion starter-prompt template, with no remaining design decisions.

---

## Forward-promise harvest

Before drafting any new content, **find every place in the shipped MDX where this topic was promised**. These are non-negotiable contracts — the brief's §1.3 lists them with verbatim quotes, and the implementation must fulfill each.

**To harvest** (we'll do this together at the start of the conversation):

```bash
# Pattern 1: explicit "(coming soon, Topic {{TOPIC_NUMBER}})" markers
grep -rn 'coming soon, Topic {{TOPIC_NUMBER}}' src/content/topics/

# Pattern 2: name-based mentions (use this topic's title fragments)
grep -rn {{FORWARD_PROMISE_GREP}} src/content/topics/

# Pattern 3: forward-pointer hyperlinks to the provisional slug
grep -rn '/topics/{{PROVISIONAL_SLUG}}' src/content/topics/
```

Expected count: roughly {{N_FORWARD_PROMISES}} unique forward-promises (refine after grep). Each entry in §1.3 of the brief should be a row in a table with columns: Source location | Verbatim text | Topic {{TOPIC_NUMBER}} deliverable.

---

## Decisions to lock explicitly during this session

The brief is a contract. Every entry in this list needs an answer before the conversation is done; defer none of them to the Claude Code session.

| Decision | Default if no opinion | Notes |
|---|---|---|
| **Slug** | `{{PROVISIONAL_SLUG}}` | Topic 19 set the precedent of expanding `slug` to `slug-and-X` when there's a "& X" half. Worth a deliberate choice — cross-references are painful to migrate later. |
| **Difficulty** | `intermediate` | foundational / intermediate / advanced. Track 5+ is intermediate by default. |
| **Read time / word count** | 55–60 min / 9.5–10K words | Topic 18 was 60-min/10K; Topic 19 was 55-min/9.5K. The trend has been steady. |
| **Number of sections** | 10 | One section per major construction + a §X.1 motivation + a §X.10 forward-look closer. |
| **Number of full proofs** | 3–4 | Pick the theorems where the *proof* carries pedagogical weight, not just the statement. |
| **Number of theorems (proofed + stated)** | 6–8 | Stated-only theorems are fine for results that are downstream consequences or vector-θ extensions. |
| **Number of examples** | 12–16 | At least one ML-relevant example per major construction. |
| **Number of remarks** | 20–25 | Remarks carry the connective tissue: scope boundaries, alternative derivations, ML-context callouts. |
| **Number of figures** | 8–10 | Each figure is a notebook cell. The featured figure is one of these — the visual analog of the featured theorem. |
| **Number of interactive components** | 3 required + 1 optional | Locked across recent topics. Don't expand to 5 unless there's a strong reason. |
| **Featured theorem** | (open) | The single result the topic is *about*. Drives the featured figure and the featured component. |
| **Featured component** | (open) | The flagship interactive. Goes in the section housing the featured theorem. |
| **Citations** | (open) | New entries (full Chicago 17th + verified URL) and reused entries (which prior `Used In Topics` to append to). |
| **Scope boundaries** | (open) | What this topic does NOT cover, with explicit deferral pointers. Each boundary becomes a one-paragraph forward-pointing remark in §X.10. |

---

## Brief structure (the 10-section contract)

The handoff brief follows this structure, established by Topics 17, 18, and 19. Don't deviate without flagging — Claude Code expects these section numbers.

```
§ 1   Important: {{TOPIC_NUMBER}}th Topic — Position in Track
  §1.1   Scope boundary (what this topic covers; what it does NOT)
  §1.2   Forward-promise fulfillment table (Source | Verbatim text | Deliverable)
§ 2   Frontmatter
  §2.1   Full YAML spec (every reference with verified URL)
  §2.2   Length target rationale
§ 3   Content Structure
  §3.1   Section map (table: § | Title | Formal elements | Figure | Component)
  §3.2   Proof specifications (full text of every full proof, ready to render)
  §3.3   Notation conventions (extending predecessors)
§ 4   Static Figures & Notebook Structure
  §4.1   Figure manifest (table: # | File | Section | Dimensions | Description)
  §4.2   Notebook structure (cell-by-cell)
§ 5   Interactive Components
  §5.1...§5.N   Per-component spec (purpose, interactions, data, dependencies, mobile)
  §5.last       Implementation notes (architecture, palette, performance)
§ 6   Shared Module ({{shared_module}}) [extension or creation]
  §6.1   Full new-function manifest (TypeScript signatures with JSDoc)
  §6.2   New test-harness entries (numeric IDs, expected values, tolerances)
§ 7   Preset Data Module
  §7.1   Full TypeScript content (preset arrays for each component)
§ 8   Cross-Reference Updates
  §8.1   Predecessor MDX edits (table: # | Source | Find | Replace)
  §8.2   Curriculum-graph.json edits (node + edges)
  §8.3   curriculum.ts edits (track status)
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

---

## Notebook structure (the cell contract)

**Before drafting the cell-by-cell outline below, read the prior notebooks listed in "Existing notebooks worth studying" above.** Look at: cell granularity (one figure per cell vs grouped), palette setup conventions, how MC simulations are seeded and looped, and how the verification helper at the end exposes ground-truth values for the §6.2 test pins. Mirror the pattern that fits the topic best — don't reinvent.

The notebook produces the figure manifest from §4.1. Cell structure mirrors Topics 17–19:

```
Cell 1   Imports + seed
         numpy, scipy.stats, matplotlib.pyplot, np.random.seed(42)
         OUTPUT_DIR (relative to notebook CWD; figures saved with savefig)

Cell 2   Palette + matplotlib defaults
         Reuse Track 5 palette (amber Wald, purple Score, green LRT, grey ref,
         blue pivot, red warn) — extend per topic if needed.
         rcParams: dpi 150, savefig.bbox 'tight', spines top/right off.

Cell 3   Figure 1 (the §X.1 motivation figure)
Cell 4   Figure 2 (often the §X.2 featured-theorem figure)
...
Cell N+2 Figure N
Cell N+3 Verification helper (computes ground-truth values cited in §6.2 tests)
Cell N+4 Markdown summary (figure list + numerical values for cross-checking the MDX)
```

**Notebook conventions:**
- Every figure: `dpi=150, bbox_inches='tight', pad_inches=0.15`.
- Filename pattern: `{{TOPIC_NUMBER}}-{kebab-description}.png`.
- Output directory written to `public/images/topics/{{PROVISIONAL_SLUG}}/` (relative path — drops files in `notebooks/{{PROVISIONAL_SLUG}}/public/...` if you run from the notebook directory; manual move-to-public is part of the Claude Code build step).
- `np.random.seed(42)` at module top + re-seed inside any cell that does sampling.
- Every figure title is short and self-explanatory; the alt-text in the MDX repeats the substantive content for accessibility.
- KaTeX vs matplotlib mathtext gotcha: `\iff` works in KaTeX but NOT in matplotlib — use `\Leftrightarrow` in figure titles.

---

## Style locks (don't waste cycles re-deriving these)

**KaTeX constraints:**
- No `\begin{aligned}` blocks. Multi-line derivations use separate `$$...$$` blocks with prose connectors.
- No `\begin{array}` tables. Use markdown tables.
- Inline math `$...$`; display math `$$...$$` on its own line.

**MDX constraints (these have bitten us):**
- Curly braces `{...}` in MDX body text or image alt-text are parsed as JSX expressions. If the content includes Greek letters, subscript Unicode (₀, ₁, ₂), or Unicode minus (`−`), the JSX parser fails. Reword without curly braces in alt-text and prose; use math `$\{...\}$` if you need set notation.
- Cross-site links: use the `<ExternalLink>` Astro component for formalCalculus / formalML, not raw markdown links.
- Section anchors follow the convention `#section-{{TOPIC_NUMBER}}-X` (used by predecessors' forward-pointers, even though no rehype-slug plugin is registered; this is technical debt to be addressed separately).

**Citations (Chicago 17th, Notes & Bibliography):**
- Books: `N. Author(s). (Year). [*Title*](URL) (Edition). Publisher.`
- Articles: `N. Author(s). (Year). [*Title*](URL). *Journal*, Volume(Issue), Pages.`
- Every frontmatter `references:` entry must have a `url:` field with a working hyperlink (verify each).
- The MDX must end with a rendered `### References` section listing every entry from frontmatter as a numbered markdown list, preceded by a `---` horizontal rule. Non-negotiable.

**Components:**
- `client:visible` directive (not `client:load`).
- React + Tailwind + SVG/D3 only. No Chart.js, Plotly, Recharts.
- Mobile-responsive at 375px width.
- Use the Track palette from `viz/shared/colorScales.ts` or the topic-specific palette established in the notebook.

**End-of-proof citation style:**
- `∎ — using [list of specific theorems/papers]` — uniformly applied across every full proof.

---

## Scope-boundary discipline

Every topic has a temptation to over-cover. The brief's §1.1 makes scope explicit by enumerating both *what's in* and *what's out*. The "what's out" list is just as important — each deferral becomes a one-paragraph forward-pointing remark in §X.10 and is the only acceptable response to "shouldn't we cover Y?".

**Default deferrals to consider:**
- **Modern extensions** (post-2010 literature) → usually defer to formalml.com or to a later track. Rationale: these depend on infrastructure not yet built (e.g., e-processes need martingale theory; conformal needs exchangeability machinery).
- **Vector-θ generalizations** → state the result, cite Lehmann-Romano 2005 Ch. 7 / Ch. 12 / Ch. 8 (depending on topic), don't derive. Most main ideas are scalar-clean.
- **Bayesian counterparts** → contrast briefly, defer full treatment to Track 7.
- **Nonparametric / distribution-free analogs** → defer to Track 8.
- **Computational / algorithmic details** → cite the algorithm, don't re-derive convergence proofs.

The bar for inclusion: *Does this result substantially change how the reader thinks about the topic's main object, or is it a downstream specialization that can be safely cited?* Specializations defer; reframing results stay.

---

## Output format

**Brief:** Single markdown file. Reply with the complete brief in one go (or in clearly-marked chunks if length-limited), copy-paste-ready to save as `docs/formalstatistics-{{PROVISIONAL_SLUG}}-handoff-brief.md`.

**Notebook:** Either a complete `.ipynb` JSON or — preferred — a cell-by-cell outline I can paste into a Jupyter session and fill in iteratively. The latter is usually faster because you can flag the algorithmic decisions for me to make rather than guessing.

**Scaffold (optional):** A short companion markdown — typically 200–400 lines — restating the locked decisions with one-line rationale each. Useful for the Claude Code session as a quick-reference second window.

---

## When the conversation is done

By the end of this session, I should have:

- [ ] **Brief** saved at `docs/formalstatistics-{{PROVISIONAL_SLUG}}-handoff-brief.md` with all 10 sections + Appendix A + Appendix B
- [ ] **Notebook** saved at `notebooks/{{PROVISIONAL_SLUG}}/{{TOPIC_NUMBER}}_{module}.ipynb` (or its outline)
- [ ] **Slug** locked (and curriculum-graph.json prepped for the change if the provisional slug doesn't match)
- [ ] **Featured theorem and featured component** named explicitly
- [ ] **Forward-promise table** complete with verbatim quotes from grep
- [ ] **Citation list** complete with all URLs verified
- [ ] **Scope boundaries** named with deferral pointers
- [ ] No "TBD," "open question," or "decide later" markers anywhere in the brief

Once those are in place, I open a fresh Claude Code session with the [Claude Code starter prompt template](claude-code-starter-prompt-template.md) (filled in for Topic {{TOPIC_NUMBER}}), and the implementation runs mostly mechanically against the brief.

---

## If we get stuck mid-session

- **On scope:** ask "what's the smallest version of this topic that fulfills every forward-promise from prior topics?" That's the floor; everything else is a matter of taste.
- **On a proof:** if a proof you want to include is longer than ~20 MDX lines, either it needs to become a stated-only theorem with a citation, or the topic is over-scoped.
- **On a component:** if you can't sketch the interaction in 4 bullet points (controls, what updates, what the readout says, what the mobile fallback is), the component is under-specified — discuss before locking.
- **On a numerical value for a test:** verify against scipy / R / Wolfram before locking; "approximately X" doesn't survive Claude Code's regression test.

When genuinely stuck (the question isn't in any prior brief and isn't obvious from the published MDX), flag the decision back to me explicitly rather than picking a default. Brief decisions are load-bearing; silent defaults compound.

---

*Starter prompt version: v1 · Topic {{TOPIC_NUMBER}}: {{TOPIC_TITLE}}*
*Companion: [claude-code-starter-prompt-template.md](claude-code-starter-prompt-template.md) (downstream)*
