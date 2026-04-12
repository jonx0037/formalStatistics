import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pdfBeta, pdfGamma, pdfNormal } from './shared/distributions';
import { conjugatePairs } from '../../data/exponential-family-data';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;

const TAB_COLORS = ['#2563eb', '#16a34a', '#7c3aed'] as const;
const TAB_LABELS = ['Beta-Bernoulli', 'Gamma-Poisson', 'Normal-Normal'] as const;

export default function ConjugatePriorExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [activeTab, setActiveTab] = useState<0 | 1 | 2>(0);

  // Tab 0: Beta-Bernoulli state
  const [alpha0, setAlpha0] = useState(2);
  const [beta0, setBeta0] = useState(2);
  const [k, setK] = useState(7);
  const [n, setN] = useState(10);

  // Tab 1: Gamma-Poisson state
  const [gammaAlpha0, setGammaAlpha0] = useState(3);
  const [gammaBeta0, setGammaBeta0] = useState(1);
  const [poissonN, setPoissonN] = useState(10);
  const [poissonS, setPoissonS] = useState(30);

  // Tab 2: Normal-Normal state
  const [mu0, setMu0] = useState(0);
  const [sigma0Sq, setSigma0Sq] = useState(4);
  const [sigmaSq, setSigmaSq] = useState(1);
  const [normalN, setNormalN] = useState(5);
  const [xbar, setXbar] = useState(2.5);

  const chartW = Math.max(280, (width || 600) - 16);
  const chartH = 260;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const color = TAB_COLORS[activeTab];

  // ── Tab 0 curves ──────────────────────────────────────────────────────────

  const betaPrior = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = 0.001 + (i / NUM_POINTS) * 0.998;
      data.push({ x, y: pdfBeta(x, alpha0, beta0) });
    }
    return data;
  }, [alpha0, beta0]);

  const betaPosterior = useMemo(() => {
    const pa = alpha0 + k;
    const pb = beta0 + n - k;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = 0.001 + (i / NUM_POINTS) * 0.998;
      data.push({ x, y: pdfBeta(x, pa, pb) });
    }
    return data;
  }, [alpha0, beta0, k, n]);

  // ── Tab 1 curves ──────────────────────────────────────────────────────────

  const gammaXMax = useMemo(() => {
    const priorMean = gammaAlpha0 / gammaBeta0;
    const priorSd = Math.sqrt(gammaAlpha0) / gammaBeta0;
    const postA = gammaAlpha0 + poissonS;
    const postB = gammaBeta0 + poissonN;
    const postMean = postA / postB;
    const postSd = Math.sqrt(postA) / postB;
    return Math.max(priorMean + 4 * priorSd, postMean + 4 * postSd, 1);
  }, [gammaAlpha0, gammaBeta0, poissonN, poissonS]);

  const gammaPrior = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * gammaXMax;
      data.push({ x, y: pdfGamma(x, gammaAlpha0, gammaBeta0) });
    }
    return data;
  }, [gammaAlpha0, gammaBeta0, gammaXMax]);

  const gammaPosterior = useMemo(() => {
    const postA = gammaAlpha0 + poissonS;
    const postB = gammaBeta0 + poissonN;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = (i / NUM_POINTS) * gammaXMax;
      data.push({ x, y: pdfGamma(x, postA, postB) });
    }
    return data;
  }, [gammaAlpha0, gammaBeta0, poissonN, poissonS, gammaXMax]);

  // ── Tab 2 curves ──────────────────────────────────────────────────────────

  const normalPostSigmaSq = useMemo(
    () => 1 / (1 / sigma0Sq + normalN / sigmaSq),
    [sigma0Sq, sigmaSq, normalN],
  );
  const normalPostMu = useMemo(
    () => normalPostSigmaSq * (mu0 / sigma0Sq + normalN * xbar / sigmaSq),
    [normalPostSigmaSq, mu0, sigma0Sq, normalN, xbar, sigmaSq],
  );

  const normalXRange = useMemo(() => {
    const center = (mu0 + normalPostMu) / 2;
    const halfW = Math.max(4 * Math.sqrt(sigma0Sq), 4 * Math.sqrt(normalPostSigmaSq), 2);
    return { lo: center - halfW, hi: center + halfW };
  }, [mu0, normalPostMu, sigma0Sq, normalPostSigmaSq]);

  const normalPrior = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = normalXRange.lo + (i / NUM_POINTS) * (normalXRange.hi - normalXRange.lo);
      data.push({ x, y: pdfNormal(x, mu0, sigma0Sq) });
    }
    return data;
  }, [mu0, sigma0Sq, normalXRange]);

  const normalPosterior = useMemo(() => {
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = normalXRange.lo + (i / NUM_POINTS) * (normalXRange.hi - normalXRange.lo);
      data.push({ x, y: pdfNormal(x, normalPostMu, normalPostSigmaSq) });
    }
    return data;
  }, [normalPostMu, normalPostSigmaSq, normalXRange]);

  // ── Active curves + domain ────────────────────────────────────────────────

  const { prior, posterior, xMin, xMax } = useMemo(() => {
    if (activeTab === 0) return { prior: betaPrior, posterior: betaPosterior, xMin: 0, xMax: 1 };
    if (activeTab === 1) return { prior: gammaPrior, posterior: gammaPosterior, xMin: 0, xMax: gammaXMax };
    return { prior: normalPrior, posterior: normalPosterior, xMin: normalXRange.lo, xMax: normalXRange.hi };
  }, [activeTab, betaPrior, betaPosterior, gammaPrior, gammaPosterior, gammaXMax, normalPrior, normalPosterior, normalXRange]);

  const maxPDF = useMemo(() => {
    const m = Math.max(...prior.map((d) => d.y), ...posterior.map((d) => d.y));
    return Math.max(m, 0.01);
  }, [prior, posterior]);

  const xSpan = xMax - xMin;

  const buildPath = useCallback((data: { x: number; y: number }[]) => {
    return data
      .map((d, i) => {
        const px = MARGIN.left + ((d.x - xMin) / xSpan) * plotW;
        const py = MARGIN.top + plotH - (d.y / (maxPDF * 1.15)) * plotH;
        return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  }, [xMin, xSpan, plotW, plotH, maxPDF]);

  // ── Annotation data ───────────────────────────────────────────────────────

  const annotation = useMemo(() => {
    if (activeTab === 0) {
      const pa = alpha0 + k;
      const pb = beta0 + n - k;
      const priorMean = alpha0 / (alpha0 + beta0);
      const postMean = pa / (pa + pb);
      return {
        specific: `Beta(${alpha0.toFixed(1)} + ${k}, ${beta0.toFixed(1)} + ${n} - ${k}) = Beta(${pa.toFixed(1)}, ${pb.toFixed(1)})`,
        priorMean: priorMean.toFixed(4),
        postMean: postMean.toFixed(4),
        priorLabel: `Beta(${alpha0.toFixed(1)}, ${beta0.toFixed(1)})`,
        postLabel: `Beta(${pa.toFixed(1)}, ${pb.toFixed(1)})`,
      };
    }
    if (activeTab === 1) {
      const postA = gammaAlpha0 + poissonS;
      const postB = gammaBeta0 + poissonN;
      const priorMean = gammaAlpha0 / gammaBeta0;
      const postMean = postA / postB;
      const effN = gammaBeta0;
      return {
        specific: `Gamma(${gammaAlpha0.toFixed(1)} + ${poissonS}, ${gammaBeta0.toFixed(1)} + ${poissonN}) = Gamma(${postA.toFixed(1)}, ${postB.toFixed(1)})`,
        priorMean: priorMean.toFixed(4),
        postMean: postMean.toFixed(4),
        priorLabel: `Gamma(${gammaAlpha0.toFixed(1)}, ${gammaBeta0.toFixed(1)})`,
        postLabel: `Gamma(${postA.toFixed(1)}, ${postB.toFixed(1)})`,
        extra: `Prior effective sample size: ${effN.toFixed(1)} pseudo-observations`,
      };
    }
    return {
      specific: `N(${normalPostMu.toFixed(3)}, ${normalPostSigmaSq.toFixed(3)})`,
      priorMean: mu0.toFixed(4),
      postMean: normalPostMu.toFixed(4),
      priorLabel: `N(${mu0.toFixed(1)}, ${sigma0Sq.toFixed(1)})`,
      postLabel: `N(${normalPostMu.toFixed(3)}, ${normalPostSigmaSq.toFixed(3)})`,
      extra: `Precision-weighted mean: (${mu0}/${sigma0Sq.toFixed(1)} + ${normalN} * ${xbar}/${sigmaSq.toFixed(1)}) * ${normalPostSigmaSq.toFixed(3)}`,
    };
  }, [activeTab, alpha0, beta0, k, n, gammaAlpha0, gammaBeta0, poissonN, poissonS, mu0, sigma0Sq, sigmaSq, normalN, xbar, normalPostMu, normalPostSigmaSq]);

  // ── X-axis ticks ──────────────────────────────────────────────────────────

  const xTicks = useMemo(() => {
    if (activeTab === 0) return [0, 0.25, 0.5, 0.75, 1];
    const nTicks = 5;
    const ticks: number[] = [];
    for (let i = 0; i < nTicks; i++) {
      ticks.push(xMin + (i / (nTicks - 1)) * xSpan);
    }
    return ticks;
  }, [activeTab, xMin, xSpan]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      <div className="text-center text-sm font-semibold mb-3">
        Conjugate Prior Explorer
      </div>

      {/* Tab buttons */}
      <div className="flex flex-wrap gap-1.5 justify-center mb-3">
        {TAB_LABELS.map((label, i) => (
          <button
            key={label}
            onClick={() => setActiveTab(i as 0 | 1 | 2)}
            className={`rounded-md px-3 py-1 text-sm font-medium transition-colors ${
              activeTab === i
                ? 'text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'
            }`}
            style={activeTab === i ? { backgroundColor: TAB_COLORS[i] } : undefined}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        {activeTab === 0 && (
          <>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">a&#x2080;</span>
              <input type="range" min={1} max={20} step={0.5} value={alpha0} onChange={(e) => setAlpha0(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{alpha0.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">b&#x2080;</span>
              <input type="range" min={1} max={20} step={0.5} value={beta0} onChange={(e) => setBeta0(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{beta0.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">n</span>
              <input type="range" min={1} max={50} step={1} value={n} onChange={(e) => { setN(Number(e.target.value)); setK(Math.min(k, Number(e.target.value))); }} className="w-20" />
              <span className="w-8 tabular-nums">{n}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">k</span>
              <input type="range" min={0} max={n} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{k}</span>
            </label>
          </>
        )}
        {activeTab === 1 && (
          <>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">a&#x2080;</span>
              <input type="range" min={0.5} max={20} step={0.5} value={gammaAlpha0} onChange={(e) => setGammaAlpha0(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{gammaAlpha0.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">b&#x2080;</span>
              <input type="range" min={0.1} max={10} step={0.1} value={gammaBeta0} onChange={(e) => setGammaBeta0(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{gammaBeta0.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">n</span>
              <input type="range" min={1} max={50} step={1} value={poissonN} onChange={(e) => setPoissonN(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{poissonN}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">&Sigma;k&#x1D62;</span>
              <input type="range" min={0} max={200} step={1} value={poissonS} onChange={(e) => setPoissonS(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{poissonS}</span>
            </label>
          </>
        )}
        {activeTab === 2 && (
          <>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">&mu;&#x2080;</span>
              <input type="range" min={-10} max={10} step={0.1} value={mu0} onChange={(e) => setMu0(Number(e.target.value))} className="w-20" />
              <span className="w-10 tabular-nums">{mu0.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">&sigma;&#x2080;&sup2;</span>
              <input type="range" min={0.1} max={10} step={0.1} value={sigma0Sq} onChange={(e) => setSigma0Sq(Number(e.target.value))} className="w-20" />
              <span className="w-10 tabular-nums">{sigma0Sq.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">&sigma;&sup2;</span>
              <input type="range" min={0.1} max={10} step={0.1} value={sigmaSq} onChange={(e) => setSigmaSq(Number(e.target.value))} className="w-20" />
              <span className="w-10 tabular-nums">{sigmaSq.toFixed(1)}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">n</span>
              <input type="range" min={1} max={50} step={1} value={normalN} onChange={(e) => setNormalN(Number(e.target.value))} className="w-20" />
              <span className="w-8 tabular-nums">{normalN}</span>
            </label>
            <label className="flex items-center gap-2 text-xs">
              <span className="font-medium">x&#x0304;</span>
              <input type="range" min={-5} max={5} step={0.1} value={xbar} onChange={(e) => setXbar(Number(e.target.value))} className="w-20" />
              <span className="w-10 tabular-nums">{xbar.toFixed(1)}</span>
            </label>
          </>
        )}
      </div>

      {/* Chart */}
      <svg width={chartW} height={chartH} className="block mx-auto">
        {/* Y grid lines */}
        {[0.25, 0.5, 0.75].map((frac) => {
          const v = maxPDF * frac;
          const y = MARGIN.top + plotH - (v / (maxPDF * 1.15)) * plotH;
          return (
            <line
              key={frac}
              x1={MARGIN.left} y1={y}
              x2={chartW - MARGIN.right} y2={y}
              stroke="currentColor" strokeOpacity={0.08}
            />
          );
        })}

        {/* X grid lines */}
        {xTicks.map((v) => {
          const px = MARGIN.left + ((v - xMin) / xSpan) * plotW;
          return (
            <g key={v}>
              <line
                x1={px} y1={MARGIN.top}
                x2={px} y2={MARGIN.top + plotH}
                stroke="currentColor" strokeOpacity={0.08}
              />
              <text x={px} y={chartH - 4} textAnchor="middle" className="fill-current" style={{ fontSize: '8px' }}>
                {activeTab === 0 ? v.toFixed(2) : v.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* Prior shading */}
        <path
          d={buildPath(prior) + `L${(MARGIN.left + plotW).toFixed(1)},${(MARGIN.top + plotH).toFixed(1)}L${MARGIN.left.toFixed(1)},${(MARGIN.top + plotH).toFixed(1)}Z`}
          fill={color} opacity={0.06}
        />

        {/* Prior curve (dashed) */}
        <path d={buildPath(prior)} fill="none" stroke={color} strokeWidth={1.5} strokeDasharray="6,4" opacity={0.7} />

        {/* Posterior shading */}
        <path
          d={buildPath(posterior) + `L${(MARGIN.left + plotW).toFixed(1)},${(MARGIN.top + plotH).toFixed(1)}L${MARGIN.left.toFixed(1)},${(MARGIN.top + plotH).toFixed(1)}Z`}
          fill={color} opacity={0.1}
        />

        {/* Posterior curve (solid) */}
        <path d={buildPath(posterior)} fill="none" stroke={color} strokeWidth={2.5} />
      </svg>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 justify-center mt-2 text-xs">
        <span style={{ color }}><span style={{ opacity: 0.7 }}>---</span> Prior {annotation.priorLabel}</span>
        <span style={{ color }}>&mdash; Posterior {annotation.postLabel}</span>
      </div>

      {/* Annotation block */}
      <div className="mt-3 text-xs max-w-lg mx-auto space-y-1.5">
        <div className="italic opacity-70 text-center">
          General rule: {conjugatePairs[activeTab]
            ? `${conjugatePairs[activeTab].posteriorRule}`
            : ''}
        </div>
        <div className="text-center font-semibold" style={{ color }}>
          {annotation.specific}
        </div>
        <div className="text-center">
          Prior mean: {annotation.priorMean} &rarr; Posterior mean: {annotation.postMean}
        </div>
        {annotation.extra && (
          <div className="text-center italic opacity-70">
            {annotation.extra}
          </div>
        )}
      </div>
    </div>
  );
}
