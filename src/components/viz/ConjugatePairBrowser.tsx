/**
 * ConjugatePairBrowser — §25.5 5-tab browser for all five canonical conjugate
 * pairs. Each tab shows the prior/likelihood/posterior trio for the active
 * family, with a formula panel (rendered as KaTeX-adjacent Unicode) and
 * family-specific chart: 1D density for the four scalar-posterior cases,
 * a 2D contour for Normal-Normal-Inverse-Gamma (20×20 grid of (μ, σ²)), and
 * a ternary plot for Dirichlet-Multinomial (30×30 simplex lattice).
 */
import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfBeta, pdfGamma, pdfNormal, pdfDirichlet, pdfNormalInverseGamma,
  posterior, posteriorMean, credibleIntervalBeta, credibleIntervalNormal,
  type PosteriorHyperparams,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import {
  betaBinomialPresets, normalNormalPresets, normalNormalIGPresets,
  gammaPoissonPresets, dirichletMultinomialPresets,
} from '../../data/bayesian-foundations-data';

type Tab = 'beta-binomial' | 'normal-normal' | 'normal-normal-ig' | 'gamma-poisson' | 'dirichlet-multinomial';

const TAB_LABELS: Record<Tab, string> = {
  'beta-binomial': 'Beta-Binomial',
  'normal-normal': 'Normal-Normal (σ² known)',
  'normal-normal-ig': 'Normal-Normal-IG (σ² unknown)',
  'gamma-poisson': 'Gamma-Poisson',
  'dirichlet-multinomial': 'Dirichlet-Multinomial',
};

export default function ConjugatePairBrowser() {
  const [tab, setTab] = useState<Tab>('beta-binomial');
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const isMobile = (width || 800) < 640;

  return (
    <div ref={ref} className="my-6 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm">
      {/* Tab bar — dropdown on mobile, pills on desktop */}
      {isMobile ? (
        <select
          className="mb-3 w-full rounded border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-2 py-2 text-sm"
          value={tab}
          onChange={e => setTab(e.target.value as Tab)}
          aria-label="Choose conjugate family"
        >
          {(Object.keys(TAB_LABELS) as Tab[]).map(t => (
            <option key={t} value={t}>{TAB_LABELS[t]}</option>
          ))}
        </select>
      ) : (
        <div className="mb-4 flex flex-wrap gap-1 border-b border-[var(--color-border)]">
          {(Object.keys(TAB_LABELS) as Tab[]).map(t => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`rounded-t-md px-3 py-1.5 text-sm transition ${
                tab === t
                  ? 'border border-b-[var(--color-surface)] border-[var(--color-border)] bg-[var(--color-surface)] font-semibold'
                  : 'text-[var(--color-text-muted)] hover:bg-[var(--color-surface-alt)]'
              }`}
            >
              {TAB_LABELS[t]}
            </button>
          ))}
        </div>
      )}

      {tab === 'beta-binomial' && <BetaBinomialTab width={width || 600} />}
      {tab === 'normal-normal' && <NormalNormalTab width={width || 600} />}
      {tab === 'normal-normal-ig' && <NormalNormalIGTab width={width || 600} />}
      {tab === 'gamma-poisson' && <GammaPoissonTab width={width || 600} />}
      {tab === 'dirichlet-multinomial' && <DirichletMultinomialTab width={width || 600} />}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 1: Beta-Binomial
// ═══════════════════════════════════════════════════════════════════════════
function BetaBinomialTab({ width }: { width: number }) {
  const [alpha0, setAlpha0] = useState(2);
  const [beta0, setBeta0] = useState(2);
  const [n, setN] = useState(50);
  const [k, setK] = useState(10);

  const post = posterior(
    'beta-binomial',
    { family: 'beta-binomial', alpha0, beta0 },
    { family: 'beta-binomial', n, k },
  ) as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;

  const N = 300;
  const priorDensity = buildDensity(x => pdfBeta(x, alpha0, beta0), 0, 1, N);
  const postDensity = buildDensity(x => pdfBeta(x, post.alpha0, post.beta0), 0, 1, N);

  const postMean = posteriorMean('beta-binomial', post);
  const cri = credibleIntervalBeta(post.alpha0, post.beta0, 0.95);

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <ControlPanel>
        <Slider label={`α₀: ${alpha0.toFixed(2)}`} min={0.5} max={20} step={0.1} value={alpha0} onChange={setAlpha0} />
        <Slider label={`β₀: ${beta0.toFixed(2)}`} min={0.5} max={20} step={0.1} value={beta0} onChange={setBeta0} />
        <Slider label={`n: ${n}`} min={1} max={100} step={1} value={n} onChange={v => { setN(v); if (k > v) setK(v); }} />
        <Slider label={`k: ${k}`} min={0} max={n} step={1} value={k} onChange={setK} />
        <PresetRow presets={betaBinomialPresets} onPick={p => { setAlpha0(p.alpha0); setBeta0(p.beta0); setN(p.n); setK(p.k); }} />
      </ControlPanel>
      <div>
        <DensityChart width={width - 280} priorDensity={priorDensity} postDensity={postDensity} support={[0, 1]} ariaLabel="Beta-Binomial prior and posterior" />
        <FormulaPanel
          lines={[
            `Prior π(θ) = Beta(α₀=${alpha0.toFixed(1)}, β₀=${beta0.toFixed(1)})`,
            `Likelihood L(θ) ∝ θ^k (1−θ)^{n−k}, k=${k}, n=${n}`,
            `Posterior p(θ | y) = Beta(α₀+k = ${post.alpha0.toFixed(1)}, β₀+n−k = ${post.beta0.toFixed(1)})`,
            `Posterior mean = (α₀+k)/(α₀+β₀+n) = ${postMean.toFixed(4)}`,
            `95% CrI = [${cri[0].toFixed(3)}, ${cri[1].toFixed(3)}]`,
            `Pseudo-sample-size: prior α₀+β₀ = ${(alpha0+beta0).toFixed(1)}, total after data = ${(alpha0+beta0+n).toFixed(1)}`,
          ]}
        />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 2: Normal-Normal (σ² known)
// ═══════════════════════════════════════════════════════════════════════════
function NormalNormalTab({ width }: { width: number }) {
  const [mu0, setMu0] = useState(0);
  const [sigma0Sq, setSigma0Sq] = useState(4);
  const [sigmaSq, setSigmaSq] = useState(1);
  const [n, setN] = useState(20);
  const [yBar, setYBar] = useState(1);

  const post = posterior(
    'normal-normal',
    { family: 'normal-normal', mu0, sigma0_sq: sigma0Sq, sigma_sq: sigmaSq },
    { family: 'normal-normal', n, yBar },
  ) as Extract<PosteriorHyperparams, { family: 'normal-normal' }>;

  const support: [number, number] = [
    Math.min(mu0, post.mu0) - 4 * Math.sqrt(Math.max(sigma0Sq, post.sigma0_sq)),
    Math.max(mu0, post.mu0) + 4 * Math.sqrt(Math.max(sigma0Sq, post.sigma0_sq)),
  ];
  const N = 300;
  const priorDensity = buildDensity(x => pdfNormal(x, mu0, sigma0Sq), support[0], support[1], N);
  const postDensity = buildDensity(x => pdfNormal(x, post.mu0, post.sigma0_sq), support[0], support[1], N);
  const cri = credibleIntervalNormal(post.mu0, post.sigma0_sq, 0.95);
  const predVar = post.sigma0_sq + sigmaSq;

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <ControlPanel>
        <Slider label={`μ₀: ${mu0.toFixed(2)}`} min={-5} max={5} step={0.1} value={mu0} onChange={setMu0} />
        <Slider label={`σ₀²: ${sigma0Sq.toFixed(2)}`} min={0.1} max={10} step={0.1} value={sigma0Sq} onChange={setSigma0Sq} />
        <Slider label={`σ² (likelihood): ${sigmaSq.toFixed(2)}`} min={0.1} max={10} step={0.1} value={sigmaSq} onChange={setSigmaSq} />
        <Slider label={`n: ${n}`} min={1} max={50} step={1} value={n} onChange={setN} />
        <Slider label={`ȳ: ${yBar.toFixed(2)}`} min={-5} max={5} step={0.1} value={yBar} onChange={setYBar} />
        <PresetRow presets={normalNormalPresets} onPick={p => {
          setMu0(p.mu0); setSigma0Sq(p.sigma0_sq); setSigmaSq(p.sigma_sq); setN(p.n); setYBar(p.yBar);
        }} />
      </ControlPanel>
      <div>
        <DensityChart width={width - 280} priorDensity={priorDensity} postDensity={postDensity} support={support} ariaLabel="Normal-Normal prior and posterior on μ" />
        <FormulaPanel
          lines={[
            `Prior π(μ) = N(μ₀=${mu0.toFixed(2)}, σ₀²=${sigma0Sq.toFixed(2)})`,
            `Likelihood ȳ=${yBar.toFixed(2)}, σ²=${sigmaSq.toFixed(2)}, n=${n}`,
            `Posterior p(μ | y) = N(μ_n = ${post.mu0.toFixed(4)}, σ_n² = ${post.sigma0_sq.toFixed(4)})`,
            `Precision-weighted: μ_n = σ_n² (μ₀/σ₀² + n·ȳ/σ²)`,
            `95% CrI = [${cri[0].toFixed(3)}, ${cri[1].toFixed(3)}]`,
            `Posterior-predictive variance = σ_n² + σ² = ${predVar.toFixed(4)}`,
          ]}
        />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 3: Normal-Normal-IG (σ² unknown) — 2D joint-posterior contour plot
// ═══════════════════════════════════════════════════════════════════════════
function NormalNormalIGTab({ width }: { width: number }) {
  const [mu0, setMu0] = useState(0);
  const [kappa0, setKappa0] = useState(1);
  const [alpha0, setAlpha0] = useState(2);
  const [beta0, setBeta0] = useState(2);
  const [n, setN] = useState(20);
  const [yBar, setYBar] = useState(0.5);
  const [s2, setS2] = useState(1.2);

  const post = posterior(
    'normal-normal-ig',
    { family: 'normal-normal-ig', mu0, kappa0, alpha0, beta0 },
    { family: 'normal-normal-ig', n, yBar, s2 },
  ) as Extract<PosteriorHyperparams, { family: 'normal-normal-ig' }>;

  const muMin = post.mu0 - 2;
  const muMax = post.mu0 + 2;
  const sig2Min = 0.1;
  const sig2Max = 3;

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <ControlPanel>
        <Slider label={`μ₀: ${mu0.toFixed(2)}`} min={-3} max={3} step={0.1} value={mu0} onChange={setMu0} />
        <Slider label={`κ₀: ${kappa0.toFixed(2)}`} min={0.1} max={10} step={0.1} value={kappa0} onChange={setKappa0} />
        <Slider label={`α₀: ${alpha0.toFixed(2)}`} min={0.5} max={10} step={0.1} value={alpha0} onChange={setAlpha0} />
        <Slider label={`β₀: ${beta0.toFixed(2)}`} min={0.1} max={10} step={0.1} value={beta0} onChange={setBeta0} />
        <Slider label={`n: ${n}`} min={1} max={50} step={1} value={n} onChange={setN} />
        <Slider label={`ȳ: ${yBar.toFixed(2)}`} min={-3} max={3} step={0.1} value={yBar} onChange={setYBar} />
        <Slider label={`s²: ${s2.toFixed(2)}`} min={0.1} max={5} step={0.1} value={s2} onChange={setS2} />
        <PresetRow presets={normalNormalIGPresets} onPick={p => {
          setMu0(p.mu0); setKappa0(p.kappa0); setAlpha0(p.alpha0); setBeta0(p.beta0);
          setN(p.n); setYBar(p.yBar); setS2(p.s2);
        }} />
      </ControlPanel>
      <div>
        <ContourChart
          width={width - 280}
          xRange={[muMin, muMax]}
          yRange={[sig2Min, sig2Max]}
          density={(mu, sig2) => pdfNormalInverseGamma(mu, sig2, post.mu0, post.kappa0, post.alpha0, post.beta0)}
          xLabel="μ"
          yLabel="σ²"
          ariaLabel="Joint posterior on (μ, σ²) as 20×20 contour"
        />
        <FormulaPanel
          lines={[
            `Prior π(μ, σ²) = N(μ | μ₀, σ²/κ₀) · InvGamma(σ² | α₀, β₀)`,
            `Posterior hyperparameters: μ_n = ${post.mu0.toFixed(4)}, κ_n = ${post.kappa0.toFixed(2)}, α_n = ${post.alpha0.toFixed(2)}, β_n = ${post.beta0.toFixed(3)}`,
            `Marginal on μ is non-standardized Student-t(2α_n, μ_n, β_n/(α_n·κ_n))`,
            `Marginal on σ² is InvGamma(α_n, β_n)`,
            `See GEL2013 §3.3 Eqns 3.3–3.5 for the update rule`,
          ]}
        />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 4: Gamma-Poisson
// ═══════════════════════════════════════════════════════════════════════════
function GammaPoissonTab({ width }: { width: number }) {
  const [alpha0, setAlpha0] = useState(2);
  const [beta0, setBeta0] = useState(1);
  const [n, setN] = useState(10);
  const [S, setS] = useState(45);

  const post = posterior(
    'gamma-poisson',
    { family: 'gamma-poisson', alpha0, beta0 },
    { family: 'gamma-poisson', n, S },
  ) as Extract<PosteriorHyperparams, { family: 'gamma-poisson' }>;

  const support: [number, number] = [0, Math.max(post.alpha0 / post.beta0 + 4 * Math.sqrt(post.alpha0) / post.beta0, 10)];
  const N = 300;
  const priorDensity = buildDensity(x => pdfGamma(x, alpha0, beta0), support[0], support[1], N);
  const postDensity = buildDensity(x => pdfGamma(x, post.alpha0, post.beta0), support[0], support[1], N);

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <ControlPanel>
        <Slider label={`α₀: ${alpha0.toFixed(2)}`} min={0.5} max={10} step={0.1} value={alpha0} onChange={setAlpha0} />
        <Slider label={`β₀ (rate): ${beta0.toFixed(2)}`} min={0.1} max={5} step={0.1} value={beta0} onChange={setBeta0} />
        <Slider label={`n: ${n}`} min={1} max={30} step={1} value={n} onChange={setN} />
        <Slider label={`S = Σyᵢ: ${S}`} min={0} max={100} step={1} value={S} onChange={setS} />
        <PresetRow presets={gammaPoissonPresets} onPick={p => { setAlpha0(p.alpha0); setBeta0(p.beta0); setN(p.n); setS(p.S); }} />
      </ControlPanel>
      <div>
        <DensityChart width={width - 280} priorDensity={priorDensity} postDensity={postDensity} support={support} ariaLabel="Gamma-Poisson prior and posterior on λ" />
        <FormulaPanel
          lines={[
            `Prior π(λ) = Gamma(α₀=${alpha0.toFixed(1)}, β₀=${beta0.toFixed(2)})  (shape-rate)`,
            `Likelihood ∝ λ^S e^{−nλ}, S=${S}, n=${n}`,
            `Posterior p(λ | y) = Gamma(α₀+S = ${post.alpha0.toFixed(1)}, β₀+n = ${post.beta0.toFixed(2)})`,
            `Posterior mean = (α₀+S)/(β₀+n) = ${(post.alpha0/post.beta0).toFixed(4)}`,
            `Posterior-predictive: NegBinom(r=α_n, p=β_n/(β_n+1))`,
          ]}
        />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tab 5: Dirichlet-Multinomial — ternary plot on 30×30 simplex lattice
// ═══════════════════════════════════════════════════════════════════════════
function DirichletMultinomialTab({ width }: { width: number }) {
  const [a1, setA1] = useState(1);
  const [a2, setA2] = useState(1);
  const [a3, setA3] = useState(1);
  const [c1, setC1] = useState(12);
  const [c2, setC2] = useState(10);
  const [c3, setC3] = useState(8);

  const post = posterior(
    'dirichlet-multinomial',
    { family: 'dirichlet-multinomial', alpha0: [a1, a2, a3] },
    { family: 'dirichlet-multinomial', counts: [c1, c2, c3] },
  ) as Extract<PosteriorHyperparams, { family: 'dirichlet-multinomial' }>;

  const postSum = post.alpha0.reduce((a, b) => a + b, 0);
  const postMean = post.alpha0.map(a => a / postSum);

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <ControlPanel>
        <Slider label={`α₀,₁: ${a1.toFixed(1)}`} min={0.5} max={10} step={0.1} value={a1} onChange={setA1} />
        <Slider label={`α₀,₂: ${a2.toFixed(1)}`} min={0.5} max={10} step={0.1} value={a2} onChange={setA2} />
        <Slider label={`α₀,₃: ${a3.toFixed(1)}`} min={0.5} max={10} step={0.1} value={a3} onChange={setA3} />
        <Slider label={`count₁: ${c1}`} min={0} max={30} step={1} value={c1} onChange={setC1} />
        <Slider label={`count₂: ${c2}`} min={0} max={30} step={1} value={c2} onChange={setC2} />
        <Slider label={`count₃: ${c3}`} min={0} max={30} step={1} value={c3} onChange={setC3} />
        <PresetRow presets={dirichletMultinomialPresets} onPick={p => {
          setA1(p.alpha0[0]); setA2(p.alpha0[1]); setA3(p.alpha0[2]);
          setC1(p.counts[0]); setC2(p.counts[1]); setC3(p.counts[2]);
        }} />
      </ControlPanel>
      <div>
        <TernaryChart
          width={width - 280}
          priorAlpha={[a1, a2, a3]}
          postAlpha={post.alpha0 as [number, number, number]}
        />
        <FormulaPanel
          lines={[
            `Prior π(p) = Dir(α₀ = [${a1.toFixed(1)}, ${a2.toFixed(1)}, ${a3.toFixed(1)}])`,
            `Counts = [${c1}, ${c2}, ${c3}], total n = ${c1+c2+c3}`,
            `Posterior p(p | y) = Dir([${post.alpha0.map(a => a.toFixed(1)).join(', ')}])`,
            `Posterior mean = [${postMean.map(m => m.toFixed(3)).join(', ')}]`,
            `Vertices of the simplex correspond to degenerate categorical distributions.`,
          ]}
        />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════

type Point = { x: number; y: number };

function buildDensity(pdf: (x: number) => number, a: number, b: number, N: number): Point[] {
  const pts: Point[] = [];
  const dx = (b - a) / N;
  for (let i = 0; i <= N; i++) {
    const x = a + i * dx;
    pts.push({ x, y: pdf(x) });
  }
  return pts;
}

const MARGIN = { top: 12, right: 16, bottom: 30, left: 40 };

function DensityChart({
  width, priorDensity, postDensity, support, ariaLabel,
}: {
  width: number; priorDensity: Point[]; postDensity: Point[];
  support: [number, number]; ariaLabel: string;
}) {
  const W = Math.max(320, width);
  const H = 260;
  const pW = W - MARGIN.left - MARGIN.right;
  const pH = H - MARGIN.top - MARGIN.bottom;
  const yMax = Math.max(
    ...priorDensity.map(p => p.y).filter(Number.isFinite),
    ...postDensity.map(p => p.y).filter(Number.isFinite),
    0.01,
  );
  const x2px = (x: number) => MARGIN.left + ((x - support[0]) / (support[1] - support[0])) * pW;
  const y2px = (y: number) => MARGIN.top + pH - (y / yMax) * pH;
  const path = (pts: Point[]) => pts
    .filter(p => Number.isFinite(p.y))
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${x2px(p.x).toFixed(2)},${y2px(p.y).toFixed(2)}`)
    .join(' ');

  return (
    <svg width={W} height={H} role="img" aria-label={ariaLabel}>
      <g stroke="var(--color-text-muted)" strokeWidth={1} fontSize={11} fill="var(--color-text-muted)">
        <line x1={MARGIN.left} y1={MARGIN.top + pH} x2={MARGIN.left + pW} y2={MARGIN.top + pH} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + pH} />
        {[0, 0.5, 1].map(f => {
          const x = support[0] + f * (support[1] - support[0]);
          return <text key={f} x={x2px(x)} y={MARGIN.top + pH + 14} textAnchor="middle">{x.toFixed(2)}</text>;
        })}
      </g>
      <path d={path(priorDensity)} fill="none" stroke={bayesianColors.prior} strokeWidth={2} opacity={0.6} />
      <path d={path(postDensity)} fill="none" stroke={bayesianColors.posterior} strokeWidth={2.5} />
      <g fontSize={11} transform={`translate(${MARGIN.left + 8}, ${MARGIN.top + 4})`}>
        <g>
          <line x1={0} y1={6} x2={18} y2={6} stroke={bayesianColors.prior} strokeWidth={2} opacity={0.6} />
          <text x={22} y={9} fill="var(--color-text)">prior</text>
        </g>
        <g transform="translate(70, 0)">
          <line x1={0} y1={6} x2={18} y2={6} stroke={bayesianColors.posterior} strokeWidth={2.5} />
          <text x={22} y={9} fill="var(--color-text)">posterior</text>
        </g>
      </g>
    </svg>
  );
}

function ContourChart({
  width, xRange, yRange, density, xLabel, yLabel, ariaLabel,
}: {
  width: number; xRange: [number, number]; yRange: [number, number];
  density: (x: number, y: number) => number;
  xLabel: string; yLabel: string; ariaLabel: string;
}) {
  const W = Math.max(320, width);
  const H = 300;
  const pW = W - MARGIN.left - MARGIN.right;
  const pH = H - MARGIN.top - MARGIN.bottom;
  const N = 20; // 20×20 grid per brief §5.2

  const grid = useMemo(() => {
    const g: number[][] = [];
    let max = 0;
    for (let i = 0; i < N; i++) {
      const row: number[] = [];
      for (let j = 0; j < N; j++) {
        const x = xRange[0] + (j / (N - 1)) * (xRange[1] - xRange[0]);
        const y = yRange[0] + (i / (N - 1)) * (yRange[1] - yRange[0]);
        const v = density(x, y);
        row.push(v);
        if (v > max) max = v;
      }
      g.push(row);
    }
    return { g, max };
  }, [xRange, yRange, density]);

  const cellW = pW / N;
  const cellH = pH / N;

  return (
    <svg width={W} height={H} role="img" aria-label={ariaLabel}>
      {/* Heatmap cells (coarse but informative on a 20×20 grid) */}
      {grid.g.map((row, i) => row.map((v, j) => {
        const intensity = grid.max > 0 ? v / grid.max : 0;
        return (
          <rect
            key={`${i}-${j}`}
            x={MARGIN.left + j * cellW}
            y={MARGIN.top + (N - 1 - i) * cellH}
            width={cellW + 0.5}
            height={cellH + 0.5}
            fill={bayesianColors.posterior}
            opacity={intensity}
          />
        );
      }))}
      {/* Axes */}
      <g stroke="var(--color-text-muted)" strokeWidth={1} fontSize={11} fill="var(--color-text-muted)">
        <line x1={MARGIN.left} y1={MARGIN.top + pH} x2={MARGIN.left + pW} y2={MARGIN.top + pH} />
        <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + pH} />
        <text x={MARGIN.left + pW / 2} y={MARGIN.top + pH + 20} textAnchor="middle">{xLabel}</text>
        <text x={MARGIN.left - 26} y={MARGIN.top + pH / 2} textAnchor="middle" transform={`rotate(-90 ${MARGIN.left - 26} ${MARGIN.top + pH / 2})`}>{yLabel}</text>
        {[0, 0.5, 1].map(f => {
          const x = xRange[0] + f * (xRange[1] - xRange[0]);
          return <text key={f} x={MARGIN.left + f * pW} y={MARGIN.top + pH + 14} textAnchor="middle">{x.toFixed(2)}</text>;
        })}
      </g>
    </svg>
  );
}

function TernaryChart({
  width, priorAlpha, postAlpha,
}: {
  width: number; priorAlpha: number[]; postAlpha: [number, number, number];
}) {
  const W = Math.max(300, width);
  const H = 300;
  const N = 30; // 30×30 simplex lattice per brief §5.2

  // Triangle vertices (equilateral, pointing up).
  // V1 = top, V2 = bottom-left, V3 = bottom-right.
  const pad = 30;
  const V = {
    x1: W / 2, y1: pad,
    x2: pad, y2: H - pad,
    x3: W - pad, y3: H - pad,
  };

  const simplexToXY = (p1: number, p2: number, p3: number): [number, number] => [
    p1 * V.x1 + p2 * V.x2 + p3 * V.x3,
    p1 * V.y1 + p2 * V.y2 + p3 * V.y3,
  ];

  const cells = useMemo(() => {
    const result: { x: number; y: number; densPrior: number; densPost: number }[] = [];
    let maxPrior = 0, maxPost = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N - i; j++) {
        // Sample a point in the interior of each triangular cell of the lattice.
        const p1 = (i + 1/3) / N;
        const p2 = (j + 1/3) / N;
        const p3 = 1 - p1 - p2;
        if (p3 < 0.005) continue;
        const [x, y] = simplexToXY(p1, p2, p3);
        const dPrior = pdfDirichlet([p1, p2, p3], priorAlpha);
        const dPost = pdfDirichlet([p1, p2, p3], [...postAlpha]);
        if (dPrior > maxPrior) maxPrior = dPrior;
        if (dPost > maxPost) maxPost = dPost;
        result.push({ x, y, densPrior: dPrior, densPost: dPost });
      }
    }
    return { cells: result, maxPrior, maxPost };
  }, [priorAlpha, postAlpha, N]);

  const cellSize = ((V.x3 - V.x2) / N) * 0.9;

  return (
    <svg width={W} height={H} role="img" aria-label="Dirichlet prior and posterior on the 2-simplex">
      {/* Triangle outline */}
      <polygon
        points={`${V.x1},${V.y1} ${V.x2},${V.y2} ${V.x3},${V.y3}`}
        fill="none"
        stroke="var(--color-text-muted)"
        strokeWidth={1.5}
      />
      {/* Vertex labels */}
      <g fontSize={11} fill="var(--color-text-muted)">
        <text x={V.x1} y={V.y1 - 6} textAnchor="middle">p₁ = 1</text>
        <text x={V.x2 - 14} y={V.y2 + 4} textAnchor="end">p₂ = 1</text>
        <text x={V.x3 + 14} y={V.y3 + 4} textAnchor="start">p₃ = 1</text>
      </g>
      {/* Prior density cells (blue, low opacity) */}
      {cells.cells.map((c, idx) => (
        <circle
          key={`prior-${idx}`}
          cx={c.x}
          cy={c.y}
          r={cellSize / 2.5}
          fill={bayesianColors.prior}
          opacity={cells.maxPrior > 0 ? (c.densPrior / cells.maxPrior) * 0.4 : 0}
        />
      ))}
      {/* Posterior density cells (purple, higher opacity) */}
      {cells.cells.map((c, idx) => (
        <circle
          key={`post-${idx}`}
          cx={c.x}
          cy={c.y}
          r={cellSize / 2.5}
          fill={bayesianColors.posterior}
          opacity={cells.maxPost > 0 ? (c.densPost / cells.maxPost) * 0.7 : 0}
        />
      ))}
      {/* Legend */}
      <g fontSize={11} transform={`translate(10, ${H - 10})`}>
        <circle cx={4} cy={-4} r={4} fill={bayesianColors.prior} opacity={0.4} />
        <text x={12} y={0} fill="var(--color-text)">prior</text>
        <circle cx={54} cy={-4} r={4} fill={bayesianColors.posterior} opacity={0.7} />
        <text x={62} y={0} fill="var(--color-text)">posterior</text>
      </g>
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// UI primitives
// ═══════════════════════════════════════════════════════════════════════════

function ControlPanel({ children }: { children: React.ReactNode }) {
  return <div className="flex flex-col gap-2 rounded-lg bg-[var(--color-surface-alt)] p-3">{children}</div>;
}

function Slider({ label, min, max, step, value, onChange }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        aria-label={label}
        aria-valuetext={String(value)}
      />
    </label>
  );
}

function PresetRow<P extends { name: string }>({ presets, onPick }: {
  presets: readonly P[]; onPick: (p: P) => void;
}) {
  return (
    <div className="mt-2 flex flex-wrap gap-1 border-t border-[var(--color-border)] pt-2">
      {presets.map((p, i) => (
        <button
          key={i}
          type="button"
          onClick={() => onPick(p)}
          className="rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-2 py-0.5 text-xs hover:bg-[var(--color-surface-alt)]"
          aria-label={`Load preset: ${p.name}`}
          title={p.name}
        >
          {p.name.length > 24 ? p.name.slice(0, 22) + '…' : p.name}
        </button>
      ))}
    </div>
  );
}

function FormulaPanel({ lines }: { lines: string[] }) {
  return (
    <div className="mt-3 rounded-md bg-[var(--color-surface-alt)] p-3 font-mono text-xs leading-relaxed">
      {lines.map((l, i) => (
        <div key={i} className={i === 0 ? 'font-semibold' : ''}>{l}</div>
      ))}
    </div>
  );
}
