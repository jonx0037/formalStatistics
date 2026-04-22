import * as d3 from 'd3';

/**
 * Domain color scale — maps track domain strings to distinct colors.
 * Used for curriculum graph and topic cards.
 */
export const domainColorScale = d3
  .scaleOrdinal<string>()
  .domain([
    'foundations-of-probability',
    'core-distributions',
    'convergence-limit-theorems',
    'statistical-estimation',
    'hypothesis-testing-confidence',
    'regression-linear-models',
    'bayesian-statistics',
    'high-dimensional-nonparametric',
  ])
  .range([
    '#2563EB', // blue — probability foundations
    '#059669', // emerald — distributions
    '#7C3AED', // violet — convergence
    '#D97706', // amber — estimation
    '#DC2626', // red — testing
    '#0891B2', // cyan — regression
    '#4F46E5', // indigo — Bayesian
    '#BE185D', // pink — nonparametric
  ]);

/**
 * Distribution visualization colors.
 */
export const distributionColors = {
  pdf: '#2563EB',
  cdf: '#059669',
  area: '#BFDBFE',
  sample: '#DC2626',
} as const;

/**
 * Inference visualization colors.
 */
export const inferenceColors = {
  null: '#6B7280',
  alternative: '#DC2626',
  confidence: '#BBF7D0',
  rejection: '#FECACA',
} as const;

/**
 * Topic 20 — Multiple Testing & False Discovery palette.
 *
 * Four FWER procedures in a purple gradient (deepest = most conservative):
 *   bonf (deepest) → holm → sidak → hoch (lightest).
 *
 * Three FDR procedures in a green gradient (deepest = most conservative):
 *   by (deepest, arbitrary dependence) → bh (independence) → storey (adaptive).
 *
 * `bh` intentionally shares hex #10B981 with the Topic 18 LRT / Topic 19
 * test-CI-duality green — BH is the FDR inheritor of the same visual family.
 *
 * `nullMixture` / `altMixture` are used in the §20.4 mixture decomposition
 * figures (slate for Uniform nulls, amber for concentrated alternatives).
 */
export const multipleTestingColors = {
  bonf: '#6B21A8',
  holm: '#9333EA',
  sidak: '#A855F7',
  hoch: '#C084FC',
  bh: '#10B981',
  by: '#047857',
  storey: '#34D399',
  nullMixture: '#94A3B8',
  altMixture: '#F59E0B',
} as const;

export type MultipleTestingColorKey = keyof typeof multipleTestingColors;

/**
 * Topic 22 — GLM response-family palette. Mirrors notebook Cell 2's `F` dict
 * and the CSS variables `--color-bernoulli` / `--color-poisson` /
 * `--color-gamma` / `--color-normal` defined in `src/styles/global.css`.
 *
 * Used by:
 *   - GLMExplorer (response-family selector indicator)
 *   - IRLSVisualizer (point coloring under Bernoulli "near-separation" preset)
 *   - DevianceTestExplorer (family selector)
 *   - SandwichCoverageSimulator (DGP family indicator)
 */
export const glmFamilyColors = {
  bernoulli: '#db2777', // pink — binary
  poisson: '#2563eb',   // blue — count
  gamma: '#d97706',     // amber — positive-skewed
  normal: '#6b7280',    // grey — baseline / reduction to §21
} as const;

export type GLMFamilyColorKey = keyof typeof glmFamilyColors;

/**
 * Topic 23 — penalty-family palette (per brief §5.5). Used by every Topic 23
 * interactive component (RegularizationPathExplorer, LevelSetExplorer,
 * CoordinateDescentVisualizer, CrossValidationExplorer) and mirrors the
 * notebook's `penalty_colors` dict so static figures and live components
 * share the same color identity for ridge / lasso / elastic-net coefficient
 * paths and level sets.
 *
 *   ridge      teal   — variance-reduction / stabilization
 *   lasso      amber  — sparsity / variable selection
 *   elasticnet mauve  — hybrid (grouping + sparsity)
 */
export const penaltyColors = {
  ridge: '#2a9d8f',
  lasso: '#e9c46a',
  elasticnet: '#b07090',
} as const;

export type PenaltyColorKey = keyof typeof penaltyColors;

/**
 * Topic 24 — information-criteria palette. Five distinct hues for the AIC /
 * AICc / BIC / Cp / LOO-CV overlay used by ICSelector, CVvsICComparator, and
 * the §24.5 Stone-equivalence figure. Mirrors the CSS variables
 * `--color-ic-aic` … `--color-ic-loo` defined in `src/styles/global.css`,
 * matching Topic 22's `glmFamilyColors` ↔ `--color-bernoulli` themed pattern.
 *
 * Hue choices share anchors with the Topic-22 GLM palette: `aic = poisson
 * blue` (default expected-loss criterion), `loo = gamma amber` (empirical-loss
 * criterion).
 *
 * `informationCriteriaLineStyles` provides matching SVG `stroke-dasharray`
 * fallbacks so color-blind viewers can still distinguish criteria via
 * line texture.
 *
 *   aic   blue       — efficient prediction (default)
 *   aicc  indigo     — small-sample correction
 *   bic   violet     — selection consistency
 *   cp    emerald    — Gaussian-linear historical predecessor
 *   loo   amber      — empirical cross-validation
 */
export const informationCriteriaColors = {
  aic:  'var(--color-ic-aic)',
  aicc: 'var(--color-ic-aicc)',
  bic:  'var(--color-ic-bic)',
  cp:   'var(--color-ic-cp)',
  loo:  'var(--color-ic-loo)',
} as const;

export type InformationCriteriaColorKey = keyof typeof informationCriteriaColors;

/**
 * SVG `stroke-dasharray` companion to `informationCriteriaColors`. Pass these
 * values directly to `.attr('stroke-dasharray', ...)` on a d3 selection.
 * `'none'` is the unbroken-line baseline reserved for AIC (the criterion
 * other curves are typically compared against in Stone-equivalence overlays).
 */
export const informationCriteriaLineStyles = {
  aic:  'none',
  aicc: '4 2',
  bic:  '6 3',
  cp:   '2 2',
  loo:  '8 4 2 4',
} as const;

/**
 * Topic 25 — Bayesian palette (Track 7). Three-color prior/likelihood/
 * posterior trio matches the standard Bayesian-textbook visual grammar.
 * Two utility colors support the BvM animator's dual-reference plot
 * (`true` for the data-generating θ₀, `mle` for θ̂_MLE).
 *
 * Inherited by Topics 26–28. Mirrors the CSS variables `--color-prior` …
 * `--color-mle` defined in `src/styles/global.css`, matching Topic 22's
 * `glmFamilyColors` ↔ `--color-bernoulli` themed pattern.
 *
 *   prior      blue      — information before data
 *   likelihood amber     — information from data
 *   posterior  violet    — updated belief after Bayes
 *   true       emerald   — data-generating parameter θ₀ (BvM reference)
 *   mle        orange    — frequentist θ̂_MLE (BvM reference)
 *   shrink     violet-600 — James–Stein + partial-pool shrinkage (Topic 28)
 *   divergence red-600   — HMC divergences in funnel diagnostics (Topic 28)
 */
export const bayesianColors = {
  prior:      'var(--color-prior)',
  likelihood: 'var(--color-likelihood)',
  posterior:  'var(--color-posterior)',
  true:       'var(--color-true-parameter)',
  mle:        'var(--color-mle)',
  shrink:     'var(--color-shrink)',
  divergence: 'var(--color-divergence)',
  chains: [
    'var(--chain-1)',
    'var(--chain-2)',
    'var(--chain-3)',
    'var(--chain-4)',
  ],
} as const;

export type BayesianColorKey = keyof typeof bayesianColors;

/**
 * SVG `stroke-dasharray` companion to `bayesianColors`. Provides redundant
 * color+linestyle encoding so color-blind viewers can still distinguish
 * prior / likelihood / posterior curves. Prior is solid (the baseline);
 * likelihood is dashed (the update engine); posterior is solid but thicker
 * in practice (set via stroke-width at the component level).
 */
export const bayesianLineStyles = {
  prior:      'none',
  likelihood: '6 3',
  posterior:  'none',
  true:       '2 2',
  mle:        'none',
} as const;

/**
 * Topic 26 — four-chain linestyle companion to `bayesianColors.chains`.
 *
 * Gelman–Rubin multi-chain diagnostics require four visually distinct chain
 * lines. Color alone is insufficient for colorblind readers, so every chain
 * pairs a `--chain-N` color with a distinct SVG `stroke-dasharray`. The
 * matplotlib mnemonic is `['-', '--', '-.', ':']` — this array gives the
 * SVG-native equivalents for D3 `.attr('stroke-dasharray', …)`.
 *
 * Index-aligned with `bayesianColors.chains`: chain i uses
 * `bayesianColors.chains[i]` + `chainLineStyles[i]`.
 *
 *   [0] solid    — chain 1 (baseline)
 *   [1] dashed   — chain 2
 *   [2] dashdot  — chain 3
 *   [3] dotted   — chain 4
 */
export const chainLineStyles = [
  'none',        // solid — chain 1
  '6 3',         // dashed — chain 2
  '4 2 1 2',     // dashdot — chain 3
  '2 2',         // dotted — chain 4
] as const;

/**
 * Matplotlib-style chain linestyle mnemonics. Mirrors the notebook's
 * `CHAIN_LINESTYLES = ['-', '--', '-.', ':']` constant (brief §4.1 Cell 4).
 * Kept for parity with the Python figure-generation code and documentation;
 * for SVG/D3 rendering use `chainLineStyles` above.
 */
export const CHAIN_LINESTYLES = ['-', '--', '-.', ':'] as const;
