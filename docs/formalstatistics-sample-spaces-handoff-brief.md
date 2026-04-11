# Claude Code Handoff Brief: Sample Spaces, Events & Axioms

**Project:** formalStatistics — [formalstatistics.com](https://www.formalstatistics.com)  
**Repo:** `github.com/jonx0037/formalStatistics`  
**Stack:** Astro 6 · React 19 · MDX · Tailwind CSS 4 · D3.js 7 · KaTeX · Vercel  
**Package Manager:** pnpm  
**Status:** Ready for implementation  
**Reference Notebook:** `notebooks/sample-spaces/01_sample_spaces_events_axioms.ipynb`

---

## Important: First Topic on a New Site

This is **topic 1 of 32** and the **first topic published** on formalstatistics.com. The site does not yet exist as a deployed application — this topic requires scaffolding the Astro project, establishing all conventions, and deploying the first page. Every pattern established here (frontmatter schema, component architecture, shared utility modules, curriculum graph structure, color palette, layout templates) will be inherited by the remaining 31 topics.

**Implications:**

1. **No existing topics to reference in-repo.** Cross-references to other formalStatistics topics should use plain text + "(coming soon)" until those topics are published.
2. **The shared utility infrastructure must be designed for extension.** `probability.ts` (the Track 1 shared module) will be extended by Topics 2–4. Keep interfaces clean and functions pure.
3. **The curriculum graph, content schema, and layout templates are created from scratch.** Use the formalCalculus and formalML repos as architectural references — the sites share a tech stack and editorial voice — but formalStatistics is an independent codebase.
4. **External prerequisite links go to formalcalculus.com.** Forward references go to formalml.com. Both open in new tabs.

---

## 1. Objective

Add a new topic page **"Sample Spaces, Events & Axioms"** as the **first topic in the Foundations of Probability track** on formalstatistics.com.

1. This is **topic 1 of 32** and the **first topic published** on formalstatistics.com. No prior topics exist.
2. **Prerequisites:** `sequences-limits` from formalcalculus.com (external). The continuity-of-probability proof uses convergence of sequences — specifically, that a telescoping sum of non-negative terms converges. No other prerequisites.
3. **Downstream within formalStatistics:**
   - `conditional-probability` (direct) — Bayes' theorem, law of total probability, and conditional independence all build on the probability space defined here
   - `random-variables` (indirect) — random variables are measurable functions from (Ω, F) to (ℝ, B(ℝ)); the sigma-algebra is the bridge
   - `expectation-variance` (indirect) — expectation is integration against the probability measure P
4. **Forward links to formalml.com:**
   - `measure-theoretic-probability` — The sigma-algebra and probability measure defined here generalize to abstract measure spaces. The Kolmogorov axioms are the probability specialization of the Lebesgue measure axioms.
   - `shannon-entropy` — Entropy H(P) = -Σ p_i log p_i is defined on probability distributions — i.e., on probability measures over finite sample spaces. The axioms ensure H is well-defined.
5. **External prerequisite links to formalcalculus.com:**
   - `sequences-limits` — Continuity of probability (Theorem 6) uses convergence of partial sums. The reader should be comfortable with ε-N convergence from this formalCalculus topic.
6. This topic **creates** the shared utility module `probability.ts` at `src/components/viz/shared/probability.ts`. This module will be extended by Topics 2–4 in the Foundations of Probability track.

**Content scope:**

- Experiments, outcomes, sample spaces (finite, countably infinite, uncountable)
- Events as subsets of Ω; set operations as logical connectives; De Morgan's laws
- Sigma-algebras: definition, necessity (Vitali set motivation), examples (trivial, discrete, partition, Borel)
- The Kolmogorov axioms: non-negativity, normalization, countable additivity
- Consequences: complement rule, monotonicity, addition rule (inclusion-exclusion for two events), general inclusion-exclusion, Boole's inequality (union bound), continuity of probability
- Combinatorial probability: multiplication principle, permutations, combinations, birthday problem, derangements
- Conditional probability and independence (preview only — full treatment in Topic 2)
- ML connections: probability spaces in ML models, union bound in PAC learning, sigma-algebras and information/filtrations

---

## 2. MDX File

### Location

```
src/content/topics/sample-spaces.mdx
```

The entry `id` will be `sample-spaces`. The dynamic route resolves to `/topics/sample-spaces`.

### Frontmatter

```yaml
---
title: "Sample Spaces, Events & Axioms"
subtitle: "The Kolmogorov axioms — non-negativity, normalization, and countable additivity — define the contract that every valid probability distribution must satisfy"
status: "published"
difficulty: "foundational"
prerequisites: []
tags:
  - "probability"
  - "sample-space"
  - "sigma-algebra"
  - "kolmogorov-axioms"
  - "events"
  - "counting"
  - "inclusion-exclusion"
  - "union-bound"
domain: "foundations"
videoId: null
notebookPath: "notebooks/sample-spaces/01_sample_spaces_events_axioms.ipynb"
githubUrl: "https://github.com/jonx0037/formalStatistics/blob/main/src/content/topics/sample-spaces.mdx"
datePublished: 2026-04-11
estimatedReadTime: 40
abstract: "Probability theory begins with three objects: a sample space Ω (all possible outcomes), a sigma-algebra F (the collection of events we can assign probabilities to), and a probability measure P : F → [0,1] satisfying Kolmogorov's axioms — non-negativity P(A) ≥ 0, normalization P(Ω) = 1, and countable additivity P(∪ Aₙ) = Σ P(Aₙ) for disjoint events. From these three axioms alone, we derive the complement rule P(Aᶜ) = 1 - P(A), monotonicity, the addition rule (inclusion-exclusion), Boole's inequality (the union bound that powers PAC learning), and the continuity of probability — the property that makes limit theorems possible. For finite equally-likely sample spaces, probability reduces to counting: P(A) = |A|/|Ω|, and the tools of combinatorics (permutations, combinations, inclusion-exclusion) become computational engines. The birthday problem and the hat-check problem (derangements) demonstrate the surprising power of these counting methods. Every ML model — from Bernoulli classifiers to diffusion models — implicitly defines a probability space, and the Kolmogorov axioms are the contract ensuring that the model's probability assignments are internally consistent."
formalcalculusPrereqs:
  - topic: "sequences-limits"
    site: "formalcalculus"
    relationship: "Continuity of probability (Theorem 6) uses convergence of partial sums. The proof that P(Aₙ) → P(A) for increasing sequences Aₙ ↑ A relies on the same telescoping-sum argument and ε-N convergence framework developed in formalCalculus Topic 1."
formalmlConnections:
  - topic: "measure-theoretic-probability"
    site: "formalml"
    relationship: "The sigma-algebra and probability measure defined here generalize to abstract measure spaces. The Kolmogorov axioms are the probability specialization of the Lebesgue measure axioms."
  - topic: "shannon-entropy"
    site: "formalml"
    relationship: "Entropy H(P) = -Σ pᵢ log pᵢ is defined on probability distributions over finite sample spaces. The axioms ensure that probability distributions — and therefore entropy — are well-defined."
connections:
  - topic: "conditional-probability"
    relationship: "The probability space (Ω, F, P) defined here is the foundation for conditional probability P(A|B) = P(A∩B)/P(B). Bayes' theorem, the law of total probability, and conditional independence all build directly on the Kolmogorov axioms."
  - topic: "random-variables"
    relationship: "Random variables are measurable functions X : (Ω, F) → (ℝ, B(ℝ)). The sigma-algebra F defined here determines which functions qualify as random variables — measurability is the bridge from events to numbers."
references:
  - type: "book"
    title: "Probability and Measure"
    author: "Patrick Billingsley"
    year: 2012
    edition: "Anniversary Edition"
    publisher: "Wiley"
    note: "The canonical measure-theoretic probability text. Chapters 1-2 cover sample spaces and probability measures with full rigor."
  - type: "book"
    title: "Probability: Theory and Examples"
    author: "Rick Durrett"
    year: 2019
    edition: "5th"
    publisher: "Cambridge University Press"
    note: "Graduate probability with excellent examples. Chapter 1 provides a concise treatment of the axioms."
  - type: "book"
    title: "Probability and Random Processes"
    author: "Geoffrey Grimmett & David Stirzaker"
    year: 2020
    edition: "4th"
    publisher: "Oxford University Press"
    note: "Strong on combinatorial probability and classical examples."
  - type: "book"
    title: "All of Statistics"
    author: "Larry Wasserman"
    year: 2004
    publisher: "Springer"
    note: "Compact reference bridging probability and statistics for ML practitioners."
  - type: "book"
    title: "Understanding Machine Learning: From Theory to Algorithms"
    author: "Shai Shalev-Shwartz & Shai Ben-David"
    year: 2014
    publisher: "Cambridge University Press"
    note: "Chapter 2 uses the union bound as the starting point for PAC learning theory."
---
```

### Content Structure

The MDX body mirrors the notebook's 9 sections. Convert each section to prose with:

- LaTeX via KaTeX (the site supports this through `remark-math` + `rehype-katex`)
- `TheoremBlock` components for formal elements (definitions, theorems, proofs, examples, remarks)
- Static images from the notebook (see §4)
- Interactive components as React islands (see §5)
- Code blocks with Python syntax highlighting

| Section | MDX Heading | Formal Elements | Figure | Interactive Component |
|---|---|---|---|---|
| 1. Experiments, Outcomes & Sample Spaces | Same | Definition 1 (Sample Space), Example gallery | `sample-spaces-gallery.png` | `SampleSpaceExplorer` |
| 2. Events as Sets | Same | Definition 2 (Event), Example 1 (die) | `event-set-operations.png` | `EventSetOperationsExplorer` |
| 3. Sigma-Algebras | Same | Definition 3 (σ-algebra), Examples 2–4, Remarks 1 | `sigma-algebra-examples.png` | `SigmaAlgebraExplorer` |
| 4. Kolmogorov Axioms | Same | Definition 4 (Probability Measure), Example 5, Remark 2 | `kolmogorov-axioms.png` | — (axioms are best as static exposition) |
| 5. Consequences | Same | Theorems 1–6, Corollary 1, Remark 3, all proofs | `continuity-of-probability.png` | `InclusionExclusionExplorer` |
| 6. Combinatorial Probability | Same | Examples 6–7 (Birthday, Derangements) | `birthday-problem.png`, `derangements.png` | `BirthdayProblemExplorer` |
| 7. Conditional Probability Preview | Same | Definitions 5–6 (preview) | — | — |
| 8. ML Connections | "Connections to ML" | ML probability space table | `union-bound-pac.png` | — |
| 9. Summary | Same | Summary table, references | — | — |

---

## 3. Formal Element Inventory

| Type | # | Title |
|------|---|-------|
| Definition | 1 | Sample Space |
| Definition | 2 | Event |
| Definition | 3 | σ-algebra |
| Definition | 4 | Probability Measure (Kolmogorov Axioms) |
| Definition | 5 | Conditional Probability (preview) |
| Definition | 6 | Independence (preview) |
| Theorem | 1 | Complement Rule |
| Theorem | 2 | Monotonicity |
| Theorem | 3 | Addition Rule (Inclusion-Exclusion, n=2) |
| Theorem | 4 | General Inclusion-Exclusion |
| Theorem | 5 | Union Bound (Boole's Inequality) |
| Theorem | 6 | Continuity of Probability |
| Corollary | 1 | P(∅) = 0 |
| Example | 1 | Die roll events and set operations |
| Example | 2 | Trivial and discrete σ-algebras |
| Example | 3 | Partition σ-algebra on {1,2,3,4} |
| Example | 4 | Borel σ-algebra on ℝ |
| Example | 5 | Fair die probability space |
| Example | 6 | Birthday Problem |
| Example | 7 | Hat-Check Problem (Derangements) |
| Remark | 1 | Algebra vs σ-algebra — why countable matters |
| Remark | 2 | Why countable additivity? |
| Remark | 3 | Continuity of probability ⟺ countable additivity |
| Proof | — | 5 proofs (Theorems 1, 2, 3, 5, 6) |

---

## 4. Static Images

**Directory:** `public/images/topics/sample-spaces/`

Run the notebook to generate these figures:

| Filename | Notebook Section | Description |
|---|---|---|
| `sample-spaces-gallery.png` | §1 (Cell 3) | 4-panel: coin flip, die roll, flip-until-heads, uniform [0,1] |
| `event-set-operations.png` | §2 (Cell 4) | 3-panel Venn diagrams: A∪B, A∩B, Aᶜ |
| `sigma-algebra-examples.png` | §3 (Cell 5) | 3-panel: trivial σ-algebra, partition σ-algebra, power set Hasse diagram |
| `kolmogorov-axioms.png` | §4 (Cell 6) | 3-panel: non-negativity bars, P(Ω)=1 circle, countable additivity stacked bar |
| `continuity-of-probability.png` | §5 (Cell 7) | 2-panel: nested circles (Aₙ↑A), convergence curves |
| `birthday-problem.png` | §6 (Cell 8) | 2-panel: P(match) curve with n=23 callout, exact vs Monte Carlo bars |
| `derangements.png` | §6 (Cell 9) | 2-panel: P(derangement) converging to 1/e, inclusion-exclusion terms |
| `union-bound-pac.png` | §8 (Cell 12) | 2-panel: per-hypothesis budget (log-log), union bound vs exact |

**To generate:**

```bash
pip install numpy matplotlib matplotlib-venn jupyter
jupyter nbconvert --to notebook --execute 01_sample_spaces_events_axioms.ipynb --output executed.ipynb
```

The notebook is seeded (`np.random.seed(42)`) for reproducibility.

---

## 5. Interactive Components

### Component 1: SampleSpaceExplorer

**File:** `src/components/viz/SampleSpaceExplorer.tsx`

**Purpose:** Let the reader build sample spaces for different experiments and see the corresponding events and power sets.

**Interactions:**
- Dropdown to select experiment type: Coin Flip, Die Roll, Two Dice, Custom
- Visual display of Ω as labeled dots/icons
- Click outcomes to select a subset → the selected subset is highlighted as an event A
- Readout shows |A|, |Ω|, P(A) = |A|/|Ω| (equally likely model)
- Toggle to show Aᶜ (complement highlighted in red)
- For "Two Dice," display the 6×6 grid of ordered pairs

**Data:** No pre-computed data needed — all computation is client-side with small finite sets.

### Component 2: EventSetOperationsExplorer

**File:** `src/components/viz/EventSetOperationsExplorer.tsx`

**Purpose:** Interactive Venn diagram showing unions, intersections, complements, and De Morgan's laws.

**Interactions:**
- Two overlapping circles A and B within a rectangle Ω
- Toggle buttons: A∪B, A∩B, Aᶜ, Bᶜ, A\B, (A∪B)ᶜ, (A∩B)ᶜ
- Selected operation highlights the corresponding region
- When De Morgan operations are selected ((A∪B)ᶜ and Aᶜ∩Bᶜ), show both simultaneously with visual proof of equality
- Display P(selected region) using the inclusion-exclusion formula
- Sliders for P(A), P(B), P(A∩B) — constrained so 0 ≤ P(A∩B) ≤ min(P(A), P(B))

**Data:** No pre-computed data — all client-side geometry and arithmetic.

### Component 3: SigmaAlgebraExplorer

**File:** `src/components/viz/SigmaAlgebraExplorer.tsx`

**Purpose:** Explore sigma-algebras on a small finite set. The reader selects subsets and the component checks whether the collection forms a valid σ-algebra.

**Interactions:**
- Sample space Ω = {1, 2, 3, 4} displayed as labeled circles
- Checkboxes for each of the 16 subsets of Ω
- Real-time validation: as the reader toggles subsets, the component checks and reports:
  - ✓/✗ Ω ∈ F
  - ✓/✗ Closed under complements
  - ✓/✗ Closed under unions
  - Overall: "Valid σ-algebra" or list of violations
- Preset buttons: "Trivial," "Partition {{1,2},{3,4}}," "Discrete (power set)"
- Visual: selected subsets are shown in a lattice/Hasse diagram style, with valid σ-algebras highlighted green and invalid collections highlighted red

**Data:** No pre-computed data — all subsets of a 4-element set are enumerable client-side.

### Component 4: InclusionExclusionExplorer

**File:** `src/components/viz/InclusionExclusionExplorer.tsx`

**Purpose:** Visualize inclusion-exclusion for 2 and 3 events. Show each term in the formula being added/subtracted.

**Interactions:**
- Toggle between 2-event and 3-event mode
- Venn diagram with 2 or 3 overlapping circles
- Step-through animation: Term 1 (add P(A)), Term 2 (add P(B)), Term 3 (subtract P(A∩B)), etc.
- At each step, the corresponding region highlights and a running total updates
- Sliders for P(A), P(B), P(C) and intersection probabilities
- Final readout: P(A∪B) or P(A∪B∪C) via inclusion-exclusion vs. direct calculation (match check)

**Data:** No pre-computed data.

### Component 5: BirthdayProblemExplorer

**File:** `src/components/viz/BirthdayProblemExplorer.tsx`

**Purpose:** Interactive birthday problem with exact computation and live Monte Carlo simulation.

**Interactions:**
- Slider for group size n (1 to 80)
- Left panel: the exact probability curve P(match) vs n, with a vertical line at the current n and a callout showing the exact value
- Right panel: animated Monte Carlo simulation
  - "Run simulation" button generates a group of n random birthdays and checks for matches
  - Running tally of simulations done and fraction with at least one match
  - The MC estimate converges to the exact value as more simulations run
- Highlight the n=23 crossing of the 50% threshold

**Data:** No pre-computed data — birthday computation is O(n) and MC is fast enough client-side.

### Implementation Notes

- Use the same component architecture patterns as formalCalculus and formalML interactive components
- React + Tailwind, client-side only (`client:visible` directive in MDX)
- SVG for all Venn diagrams and geometric visualizations (via D3.js or direct SVG)
- Use CSS custom properties for colors — reference `viz/shared/colorScales.ts`
- All components must be responsive (test at 375px width)
- No server-side computation — everything runs in the browser

---

## 6. Shared Utility Module

### New Module: `src/components/viz/shared/probability.ts`

This module provides utility functions for probability computations used across Track 1 (Foundations of Probability) components. It will be extended by Topics 2–4.

```typescript
// ── Types ───────────────────────────────────────────────────────────────────

/** A finite sample space represented as a set of string labels. */
export type SampleSpace = Set<string>;

/** An event is a subset of the sample space. */
export type Event = Set<string>;

/** A sigma-algebra is a collection of events. */
export type SigmaAlgebra = Set<string>; // serialized events for Set comparison

// ── Core probability functions ──────────────────────────────────────────────

/** Equally-likely probability: P(A) = |A| / |Ω| */
export function equallyLikelyP(event: Event, omega: SampleSpace): number;

/** Check if a collection of subsets forms a valid sigma-algebra. */
export function isValidSigmaAlgebra(
  omega: SampleSpace,
  collection: Event[]
): { valid: boolean; violations: string[] };

/** Power set of a finite set. */
export function powerSet<T>(s: Set<T>): Set<T>[];

/** Complement of A in Ω. */
export function complement(a: Event, omega: SampleSpace): Event;

/** Union of two events. */
export function union(a: Event, b: Event): Event;

/** Intersection of two events. */
export function intersection(a: Event, b: Event): Event;

/** Set difference A \ B. */
export function setDifference(a: Event, b: Event): Event;

/** Check if two events are disjoint. */
export function areDisjoint(a: Event, b: Event): boolean;

// ── Counting ────────────────────────────────────────────────────────────────

/** n! (factorial). */
export function factorial(n: number): number;

/** P(n, k) = n! / (n-k)! (permutations). */
export function permutations(n: number, k: number): number;

/** C(n, k) = n! / (k!(n-k)!) (combinations / binomial coefficient). */
export function combinations(n: number, k: number): number;

// ── Birthday problem ────────────────────────────────────────────────────────

/** P(at least one shared birthday) for n people, d days. */
export function birthdayProbability(n: number, d?: number): number;

/** Count derangements D(n) using inclusion-exclusion. */
export function derangementCount(n: number): number;

/** P(derangement) = D(n) / n!. */
export function derangementProbability(n: number): number;

// ── Inclusion-exclusion ─────────────────────────────────────────────────────

/** Inclusion-exclusion for 2 events: P(A∪B) = P(A) + P(B) - P(A∩B). */
export function inclusionExclusion2(
  pA: number, pB: number, pAB: number
): number;

/** Inclusion-exclusion for 3 events. */
export function inclusionExclusion3(
  pA: number, pB: number, pC: number,
  pAB: number, pAC: number, pBC: number,
  pABC: number
): number;

// ── Monte Carlo ─────────────────────────────────────────────────────────────

/** Seeded PRNG for reproducible simulations. */
export function seededRandom(seed: number): () => number;

/** Monte Carlo birthday simulation: returns fraction of trials with a match. */
export function mcBirthday(
  n: number, nTrials: number, rng: () => number, d?: number
): number;
```

**Design contract:** All functions are pure (no side effects), deterministic (given the same seed), and documented with JSDoc. The module will be extended with conditional probability helpers in Topic 2.

---

## 7. Topic Data Module

### New Module: `src/data/sample-spaces-data.ts`

```typescript
/** Preset sample spaces for the SampleSpaceExplorer. */
export const sampleSpacePresets = [
  {
    name: "Coin Flip",
    outcomes: ["H", "T"],
    description: "Flip a fair coin",
  },
  {
    name: "Die Roll",
    outcomes: ["1", "2", "3", "4", "5", "6"],
    description: "Roll a six-sided die",
  },
  {
    name: "Two Coins",
    outcomes: ["HH", "HT", "TH", "TT"],
    description: "Flip two coins (ordered)",
  },
  {
    name: "Two Dice",
    outcomes: /* generate all (i,j) pairs */,
    description: "Roll two dice (ordered)",
    layout: "grid", // 6×6 grid display
  },
];

/** Preset sigma-algebras for the SigmaAlgebraExplorer. */
export const sigmaAlgebraPresets = [
  {
    name: "Trivial",
    omega: ["1", "2", "3", "4"],
    events: [[], ["1", "2", "3", "4"]], // ∅ and Ω
  },
  {
    name: "Partition {1,2} | {3,4}",
    omega: ["1", "2", "3", "4"],
    events: [[], ["1", "2"], ["3", "4"], ["1", "2", "3", "4"]],
  },
  {
    name: "Discrete (Power Set)",
    omega: ["1", "2", "3", "4"],
    events: /* all 16 subsets */,
  },
];

/** Named events for the die roll example. */
export const dieEvents = {
  even: { name: "Even", outcomes: ["2", "4", "6"] },
  odd: { name: "Odd", outcomes: ["1", "3", "5"] },
  atLeast5: { name: "≥ 5", outcomes: ["5", "6"] },
  prime: { name: "Prime", outcomes: ["2", "3", "5"] },
};
```

---

## 8. Curriculum Graph & Site Metadata

### Content Schema

**File:** `src/content.config.ts` (create)

Define the content collection schema for topics. This is the site-wide schema — all 32 topics will conform to it. Model it after the formalCalculus schema, but with `formalcalculusPrereqs` and `formalmlConnections` arrays for cross-site references.

### Curriculum Graph

**File:** `src/data/curriculum-graph.json` (create)

Create the full curriculum graph with all 32 topics. Mark `sample-spaces` as `"published"` and all others as `"planned"`. Include the prerequisite DAG (within formalStatistics) and cross-site prerequisite/connection edges.

### Track Definitions

**File:** `src/data/curriculum.ts` (create)

Define all 8 tracks with their topics. Mark Track 1, Topic 1 as published.

### Pages

- **Homepage** (`src/pages/index.astro`): Show site description, latest topics (just "Sample Spaces, Events & Axioms" for now), curriculum overview with 8 tracks.
- **Topics index** (`src/pages/topics/index.astro`): List all topics with status badges (published vs planned).
- **Topic page** (`src/pages/topics/[...slug].astro`): Dynamic route for topic MDX rendering.
- **Paths page** (`src/pages/paths.astro` or similar): Curriculum visualization showing tracks and prerequisites.

---

## 9. Site Scaffolding

Since this is a brand-new project, the implementation begins with scaffolding:

```bash
cd /Users/jonathanrocha/Developer/Sites/formalStatistics
pnpm create astro@latest . -- --template minimal
pnpm add react@19 @astrojs/react tailwindcss@4 @astrojs/tailwind d3@7
pnpm add -D @types/d3 pagefind
```

Then create the directory structure:

```
src/
├── pages/
│   ├── index.astro
│   └── topics/
│       ├── index.astro
│       └── [...slug].astro
├── content/
│   └── topics/
│       └── sample-spaces.mdx
├── components/
│   ├── ui/
│   │   ├── Nav.astro
│   │   ├── Footer.astro
│   │   ├── TopicCard.astro
│   │   ├── TheoremBlock.astro
│   │   ├── ProofBlock.astro
│   │   ├── ExampleBlock.astro
│   │   ├── RemarkBlock.astro
│   │   ├── DefinitionBlock.astro
│   │   └── ExternalLink.astro      # For formalcalculus/formalml links with badges
│   └── viz/
│       ├── shared/
│       │   ├── probability.ts       # Track 1 shared utilities
│       │   ├── colorScales.ts       # Color palette (match notebook C dict)
│       │   ├── types.ts             # Shared TypeScript types
│       │   └── useD3.ts             # D3 hook
│       ├── SampleSpaceExplorer.tsx
│       ├── EventSetOperationsExplorer.tsx
│       ├── SigmaAlgebraExplorer.tsx
│       ├── InclusionExclusionExplorer.tsx
│       └── BirthdayProblemExplorer.tsx
├── data/
│   ├── curriculum-graph.json
│   ├── curriculum.ts
│   └── sample-spaces-data.ts
├── layouts/
│   ├── BaseLayout.astro
│   └── TopicLayout.astro
├── lib/
│   └── utils.ts
└── styles/
    └── global.css
```

### Color Palette

Match the notebook's color dictionary. Define in `colorScales.ts` and as CSS custom properties in `global.css`:

```typescript
export const colors = {
  primary: '#2563eb',    // Blue
  secondary: '#7c3aed',  // Purple
  accent: '#059669',     // Green
  warning: '#d97706',    // Amber
  danger: '#dc2626',     // Red
  muted: '#6b7280',      // Gray
  ltBlue: '#dbeafe',
  ltGreen: '#d1fae5',
  ltRed: '#fee2e2',
  ltPurple: '#ede9fe',
  ltAmber: '#fef3c7',
};
```

---

## 10. Verification Checklist

### Content

- [ ] `sample-spaces.mdx` renders at `/topics/sample-spaces` with no build errors
- [ ] All 8 static images present in `public/images/topics/sample-spaces/`
- [ ] KaTeX renders correctly: all display equations, inline math, and theorem blocks
- [ ] All 6 definitions, 6 theorems, 1 corollary, 7 examples, 3 remarks render in styled blocks
- [ ] All 5 proofs expand fully with no hand-waving
- [ ] formalcalculus.com link (`sequences-limits`) opens in new tab with external badge
- [ ] formalml.com links open in new tab with forward-reference badge
- [ ] Internal forward references to Topics 2–4 use plain text + "(coming soon)"

### Interactive Components

- [ ] `SampleSpaceExplorer` dropdown switches between presets; outcome selection highlights events; P(A) readout updates
- [ ] `EventSetOperationsExplorer` Venn diagram regions highlight correctly for all 7 operations; De Morgan equality visual works
- [ ] `SigmaAlgebraExplorer` real-time validation correctly identifies valid/invalid collections; all 3 presets load
- [ ] `InclusionExclusionExplorer` step-through animation adds/subtracts terms correctly; 2-event and 3-event modes work
- [ ] `BirthdayProblemExplorer` exact curve renders; MC simulation runs and converges; n=23 threshold highlighted

### Infrastructure

- [ ] `probability.ts` compiles with no TypeScript errors
- [ ] `sample-spaces-data.ts` compiles
- [ ] Homepage renders with topic listing
- [ ] Topics index page shows `sample-spaces` as published, remaining 31 as planned
- [ ] Curriculum graph shows correct prerequisite structure
- [ ] Pagefind indexes the topic on rebuild
- [ ] `pnpm build` succeeds with zero errors
- [ ] Site deploys to Vercel successfully
- [ ] Mobile responsive (all 5 interactive components at 375px width)
- [ ] "Foundational" difficulty badge styled correctly

---

## 11. Build Order

1. **Scaffold the Astro project.** Install dependencies, create directory structure, configure `astro.config.mjs` (React integration, Tailwind, MDX with remark-math + rehype-katex).
2. **Create layout templates** (`BaseLayout.astro`, `TopicLayout.astro`) and UI components (`Nav`, `Footer`, `TheoremBlock`, `ProofBlock`, `ExampleBlock`, `RemarkBlock`, `DefinitionBlock`, `ExternalLink`).
3. **Create `src/components/viz/shared/colorScales.ts`** and `types.ts` — the shared design tokens.
4. **Create `src/components/viz/shared/probability.ts`** — the Track 1 shared utility module. Implement all functions from the §6 specification. Write console log tests to verify.
5. **Create `src/data/sample-spaces-data.ts`** — presets for the interactive components.
6. **Create `src/data/curriculum-graph.json`** and `src/data/curriculum.ts`** — the full 32-topic curriculum with `sample-spaces` as the only published topic.
7. **Create `src/content.config.ts`** — the content collection schema.
8. **Create `sample-spaces.mdx`** with full frontmatter and all markdown/LaTeX content. Include all formal element blocks (definitions, theorems, proofs, examples, remarks). No interactive components yet.
9. **Create page routes** (`index.astro`, `topics/index.astro`, `topics/[...slug].astro`, `paths.astro`).
10. Copy notebook figures to `public/images/topics/sample-spaces/` and verify they load.
11. **Build `SampleSpaceExplorer.tsx`** — the entry-point component. Dropdown, outcome selection, P(A) readout.
12. **Build `EventSetOperationsExplorer.tsx`** — Venn diagram with operation toggles.
13. **Build `SigmaAlgebraExplorer.tsx`** — subset selection with real-time σ-algebra validation.
14. **Build `InclusionExclusionExplorer.tsx`** — step-through inclusion-exclusion animation.
15. **Build `BirthdayProblemExplorer.tsx`** — exact computation + MC simulation.
16. Embed all 5 components in the MDX with `client:visible`.
17. Configure Pagefind post-build hook.
18. Run the full verification checklist (§10).
19. `pnpm build` — verify zero errors.
20. Deploy to Vercel. Verify at formalstatistics.com.

---

## Appendix A: KaTeX Constraints

Based on lessons learned from formalCalculus:

- **No `\begin{aligned}` blocks with `&` markers.** Multi-line derivations use separate `$$...$$` blocks with prose glue between lines.
- **No `\begin{array}{c|cccc}` tables.** Use HTML tables or markdown tables for tabular content.
- Inline math uses `$...$`. Display math uses `$$...$$` on its own line.
- Test all LaTeX rendering in the dev server before committing — KaTeX is less forgiving than MathJax.

---

## Appendix B: Design Decisions

1. **Five interactive components** matching the established average from formalCalculus Topic 1. The `BirthdayProblemExplorer` is the flagship — it combines exact computation with live Monte Carlo simulation, giving the reader an immediate taste of the computational approach to probability that will permeate the entire site.

2. **Sigma-algebra as an interactive exploration, not just a definition.** Most probability textbooks define σ-algebras and move on. The `SigmaAlgebraExplorer` lets readers *build* σ-algebras by selecting subsets and watching validation in real time. This makes the closure properties concrete rather than abstract.

3. **Union bound → PAC learning bridge.** The ML connection for this topic is the union bound. The exposition in §8 traces the direct path from Boole's inequality (Theorem 5) to the simplest PAC learning argument. This is the seed that grows into concentration inequalities and VC theory in the later tracks — and into the formal treatment on formalml.com.

4. **Conditional probability as a preview, not a section.** Definitions 5–6 are deliberately minimal. The full treatment of conditional probability, Bayes' theorem, and independence is Topic 2. Including a preview here ensures the reader sees where we're headed, but the proofs and depth belong in the next topic.

5. **Color palette matches across the three sites.** The `C` dictionary in the notebook and the `colorScales.ts` module use the same hex values. This visual continuity across formalCalculus, formalStatistics, and formalML reinforces the sense of a unified curriculum.

---

*Brief version: v1 | Created: 2026-04-11 | Author: Jonathan Rocha*  
*Reference notebook: `notebooks/sample-spaces/01_sample_spaces_events_axioms.ipynb`*
