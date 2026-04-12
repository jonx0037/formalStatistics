import { useState, useMemo, useCallback, useEffect } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pmfBernoulli, pmfBinomial, pmfGeometric, pmfNegativeBinomial, pmfPoisson,
  pdfNormal, pdfExponential, pdfGamma, pdfBeta,
  expectationBernoulli, expectationBinomial, expectationGeometric,
  expectationNegBin, expectationPoisson,
  expectationNormal, expectationExponential, expectationGamma, expectationBeta,
} from './shared/distributions';
import {
  exponentialFamilyMembers,
  parameterConfigs,
  logPartitionFunctions,
  canonicalFormColors,
} from '../../data/exponential-family-data';
import katex from 'katex';

const MARGIN = { top: 10, right: 15, bottom: 30, left: 50 };
const NUM_POINTS = 300;
const CURVE_COLOR = '#2563eb';

// ── Distribution evaluation dispatch ───────────────────────────────────────

function evaluateDistribution(dist: string, x: number, params: Record<string, number>): number {
  switch (dist) {
    case 'Bernoulli': return pmfBernoulli(x, params.p);
    case 'Binomial': return pmfBinomial(x, params.n, params.p);
    case 'Geometric': return pmfGeometric(x, params.p);
    case 'NegBin': return pmfNegativeBinomial(x, params.r, params.p);
    case 'Poisson': return pmfPoisson(x, params.lambda);
    case 'Normal': return pdfNormal(x, params.mu, params.sigma2);
    case 'Exponential': return pdfExponential(x, params.lambda);
    case 'Gamma': return pdfGamma(x, params.alpha, params.beta);
    case 'Beta': return pdfBeta(x, params.alpha, params.beta);
    default: return 0;
  }
}

function computeExpectation(dist: string, params: Record<string, number>): number {
  switch (dist) {
    case 'Bernoulli': return expectationBernoulli(params.p);
    case 'Binomial': return expectationBinomial(params.n, params.p);
    case 'Geometric': return expectationGeometric(params.p);
    case 'NegBin': return expectationNegBin(params.r, params.p);
    case 'Poisson': return expectationPoisson(params.lambda);
    case 'Normal': return expectationNormal(params.mu);
    case 'Exponential': return expectationExponential(params.lambda);
    case 'Gamma': return expectationGamma(params.alpha, params.beta);
    case 'Beta': return expectationBeta(params.alpha, params.beta);
    default: return 0;
  }
}

// ── X-axis domain for each distribution ────────────────────────────────────

function xDomain(dist: string, params: Record<string, number>): [number, number] {
  switch (dist) {
    case 'Bernoulli': return [-0.5, 1.5];
    case 'Binomial': return [-0.5, params.n + 0.5];
    case 'Geometric': {
      const mean = 1 / params.p;
      return [0.5, Math.max(Math.ceil(mean + 4 * Math.sqrt((1 - params.p) / (params.p * params.p))), 10) + 0.5];
    }
    case 'NegBin': {
      const mean = params.r / params.p;
      const sd = Math.sqrt(params.r * (1 - params.p) / (params.p * params.p));
      return [params.r - 0.5, Math.max(Math.ceil(mean + 3.5 * sd), params.r + 10) + 0.5];
    }
    case 'Poisson': {
      const lam = params.lambda;
      return [-0.5, Math.max(Math.ceil(lam + 4 * Math.sqrt(lam)), 10) + 0.5];
    }
    case 'Normal': {
      const sd = Math.sqrt(params.sigma2);
      return [params.mu - 4 * sd, params.mu + 4 * sd];
    }
    case 'Exponential': {
      const mean = 1 / params.lambda;
      return [0, Math.max(mean + 4 * mean, 5)];
    }
    case 'Gamma': {
      const mean = params.alpha / params.beta;
      const sd = Math.sqrt(params.alpha) / params.beta;
      return [0, Math.max(mean + 4 * sd, 5)];
    }
    case 'Beta': return [0, 1];
    default: return [0, 1];
  }
}

// ── Integer support for discrete distributions ─────────────────────────────

function integerRange(dist: string, params: Record<string, number>): number[] {
  const [lo, hi] = xDomain(dist, params);
  const start = Math.max(
    Math.ceil(lo),
    dist === 'NegBin' ? params.r : dist === 'Geometric' ? 1 : 0,
  );
  const end = Math.floor(hi);
  const result: number[] = [];
  for (let k = start; k <= end; k++) {
    result.push(k);
  }
  return result;
}

// ── Inline log-partition for Binomial/NegBin (param-dependent) ─────────────

function computeLogPartition(dist: string, params: Record<string, number>) {
  // Use the data module for distributions that have entries
  if (logPartitionFunctions[dist]) {
    const lpf = logPartitionFunctions[dist];
    const eta = lpf.etaFromParams(params);
    return {
      eta,
      A: lpf.A(eta),
      Aprime: lpf.Aprime(eta),
      Adoubleprime: lpf.Adoubleprime(eta),
    };
  }

  // Binomial (fixed n): eta = log(p/(1-p)), A(eta) = n*log(1+e^eta)
  if (dist === 'Binomial') {
    const { n, p } = params;
    const eta = Math.log(p / (1 - p));
    const expEta = Math.exp(eta);
    const sigmoid = expEta / (1 + expEta);
    return {
      eta,
      A: n * Math.log(1 + expEta),
      Aprime: n * sigmoid,
      Adoubleprime: n * sigmoid * (1 - sigmoid),
    };
  }

  // NegBin (fixed r): eta = log(1-p), A(eta) = r*eta - r*log(1-e^eta)
  if (dist === 'NegBin') {
    const { r, p } = params;
    const eta = Math.log(1 - p);
    return {
      eta,
      A: r * eta - r * Math.log(1 - Math.exp(eta)),
      Aprime: r / p,
      Adoubleprime: r * (1 - p) / (p * p),
    };
  }

  return null;
}

// ── KaTeX render helpers ───────────────────────────────────────────────────
// All LaTeX strings originate from our own data module (exponential-family-data.ts),
// not from user input. The dangerouslySetInnerHTML usage is safe in this context.

function renderKatex(latex: string, displayMode = false): string {
  try {
    return katex.renderToString(latex, {
      displayMode,
      throwOnError: false,
      trust: true,
    });
  } catch {
    return latex;
  }
}

function coloredKatex(latex: string, color: string): string {
  try {
    return katex.renderToString(`\\textcolor{${color}}{${latex}}`, {
      displayMode: false,
      throwOnError: false,
      trust: true,
    });
  } catch {
    return latex;
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export default function ExponentialFamilyExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [distIndex, setDistIndex] = useState(0);
  const [params, setParams] = useState<Record<string, number>>(() => {
    const configs = parameterConfigs[exponentialFamilyMembers[0].distribution];
    const defaults: Record<string, number> = {};
    for (const c of configs) defaults[c.name] = c.default;
    return defaults;
  });

  const member = exponentialFamilyMembers[distIndex];
  const dist = member.distribution;
  const isDiscrete = member.type === 'discrete';
  const configs = parameterConfigs[dist];

  // Reset params when distribution changes
  useEffect(() => {
    const defaults: Record<string, number> = {};
    for (const c of configs) defaults[c.name] = c.default;
    setParams(defaults);
  }, [distIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  // Responsive layout
  const containerW = Math.max(280, (width || 600) - 16);
  const isNarrow = containerW < 624;
  const chartW = isNarrow ? containerW : Math.floor(containerW * 0.52);
  const chartH = 220;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Compute x-domain
  const [xMin, xMax] = useMemo(() => xDomain(dist, params), [dist, params]);
  const xSpan = xMax - xMin;

  // Mean
  const mean = useMemo(() => computeExpectation(dist, params), [dist, params]);

  // Build curve data (continuous) or bar data (discrete)
  const curveData = useMemo(() => {
    if (isDiscrete) return null;
    const data: { x: number; y: number }[] = [];
    for (let i = 0; i <= NUM_POINTS; i++) {
      const x = xMin + (i / NUM_POINTS) * xSpan;
      data.push({ x, y: evaluateDistribution(dist, x, params) });
    }
    return data;
  }, [isDiscrete, dist, params, xMin, xSpan]);

  const barData = useMemo(() => {
    if (!isDiscrete) return null;
    const ks = integerRange(dist, params);
    return ks.map((k) => ({ k, y: evaluateDistribution(dist, k, params) }));
  }, [isDiscrete, dist, params]);

  // Max PDF/PMF for scaling
  const maxY = useMemo(() => {
    if (curveData) return Math.max(...curveData.map((d) => d.y), 0.01);
    if (barData) return Math.max(...barData.map((d) => d.y), 0.01);
    return 0.01;
  }, [curveData, barData]);

  // SVG path builder (matches GammaFamilyExplorer)
  const buildPath = useCallback(
    (data: { x: number; y: number }[]) => {
      return data
        .map((d, i) => {
          const px = MARGIN.left + ((d.x - xMin) / xSpan) * plotW;
          const py = MARGIN.top + plotH - (d.y / (maxY * 1.15)) * plotH;
          return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
        })
        .join(' ');
    },
    [xMin, xSpan, plotW, plotH, maxY],
  );

  // Log-partition readout
  const lpData = useMemo(() => computeLogPartition(dist, params), [dist, params]);

  // Canonical form HTML (all LaTeX from our own static data module)
  const canonicalHTML = useMemo(() => {
    const m = member;
    const hHtml = coloredKatex(m.hx, canonicalFormColors.h);
    const etaHtml = coloredKatex(m.eta, canonicalFormColors.eta);
    const tHtml = coloredKatex(m.Tx, canonicalFormColors.T);
    const aHtml = coloredKatex(m.Aeta, canonicalFormColors.A);

    const isVector = m.dimension === 2;
    const dot = isVector ? '^\\top' : '\\cdot';

    const fxLabel = isDiscrete ? 'p' : 'f';
    const fullEq = renderKatex(
      `${fxLabel}(x \\mid \\theta) = `
      + `\\textcolor{${canonicalFormColors.h}}{h(x)} `
      + `\\exp\\!\\bigl(`
      + `\\textcolor{${canonicalFormColors.eta}}{\\boldsymbol{\\eta}(\\theta)}${dot} `
      + `\\textcolor{${canonicalFormColors.T}}{\\mathbf{T}(x)} - `
      + `\\textcolor{${canonicalFormColors.A}}{A(\\boldsymbol{\\eta})}`
      + `\\bigr)`,
      true,
    );

    return { fullEq, hHtml, etaHtml, tHtml, aHtml };
  }, [member, isDiscrete]);

  // Scale helpers
  const toSvgX = useCallback(
    (x: number) => MARGIN.left + ((x - xMin) / xSpan) * plotW,
    [xMin, xSpan, plotW],
  );
  const toSvgY = useCallback(
    (y: number) => MARGIN.top + plotH - (y / (maxY * 1.15)) * plotH,
    [plotH, maxY],
  );

  // X-axis tick values
  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50];
    let step = 1;
    for (const n of nice) {
      if (xSpan / n <= 8) { step = n; break; }
    }
    const start = Math.ceil(xMin / step) * step;
    for (let v = start; v <= xMax; v += step) {
      ticks.push(Math.round(v * 1000) / 1000);
    }
    return ticks;
  }, [xMin, xMax, xSpan]);

  // Y-axis grid values
  const yGridValues = useMemo(() => [0, maxY * 0.25, maxY * 0.5, maxY * 0.75], [maxY]);

  return (
    <div ref={ref} className="rounded-lg border p-4 my-6" style={{ borderColor: 'var(--color-border)' }}>
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Exponential Family Explorer
      </div>

      {/* Distribution selector */}
      <div className="flex justify-center mb-3">
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={distIndex}
            onChange={(e) => setDistIndex(Number(e.target.value))}
          >
            {exponentialFamilyMembers.map((m, i) => (
              <option key={m.distribution} value={i}>{m.name}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Parameter sliders */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {configs.map((c) => (
          <label key={c.name} className="flex items-center gap-2 text-xs">
            <span className="font-medium">{c.label}</span>
            <input
              type="range"
              min={c.min}
              max={c.max}
              step={c.step}
              value={params[c.name] ?? c.default}
              onChange={(e) => setParams((prev) => ({ ...prev, [c.name]: Number(e.target.value) }))}
              className="w-28"
            />
            <span className="w-10 tabular-nums text-right">
              {(params[c.name] ?? c.default).toFixed(c.step < 1 ? 2 : 0)}
            </span>
          </label>
        ))}
      </div>

      {/* Main two-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* Left panel: PDF/PMF chart */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: chartW }}>
          <svg width={chartW} height={chartH} className="block mx-auto">
            {/* Y grid lines */}
            {yGridValues.map((v, i) => (
              <line
                key={i}
                x1={MARGIN.left}
                y1={toSvgY(v)}
                x2={chartW - MARGIN.right}
                y2={toSvgY(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* X grid lines */}
            {xTicks.map((v, i) => (
              <line
                key={i}
                x1={toSvgX(v)}
                y1={MARGIN.top}
                x2={toSvgX(v)}
                y2={MARGIN.top + plotH}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Discrete: lollipop bars */}
            {barData && barData.map((d) => {
              const cx = toSvgX(d.k);
              const cy = toSvgY(d.y);
              const baseline = toSvgY(0);
              return (
                <g key={d.k}>
                  <line
                    x1={cx}
                    y1={baseline}
                    x2={cx}
                    y2={cy}
                    stroke={CURVE_COLOR}
                    strokeWidth={2}
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={3.5}
                    fill={CURVE_COLOR}
                  />
                </g>
              );
            })}

            {/* Continuous: smooth curve */}
            {curveData && (
              <path
                d={buildPath(curveData)}
                fill="none"
                stroke={CURVE_COLOR}
                strokeWidth={2.5}
              />
            )}

            {/* E[X] marker: downward-pointing triangle on x-axis */}
            {isFinite(mean) && mean >= xMin && mean <= xMax && (
              <polygon
                points={`${toSvgX(mean)},${MARGIN.top + plotH + 2} ${toSvgX(mean) - 5},${MARGIN.top + plotH + 10} ${toSvgX(mean) + 5},${MARGIN.top + plotH + 10}`}
                fill="#DC2626"
              />
            )}

            {/* Axes */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={chartW - MARGIN.right}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* X-axis ticks */}
            {xTicks.map((v, i) => (
              <text
                key={i}
                x={toSvgX(v)}
                y={MARGIN.top + plotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {Number.isInteger(v) ? v : v.toFixed(1)}
              </text>
            ))}

            {/* Y-axis labels */}
            {[maxY * 0.5, maxY].map((v, i) => (
              <text
                key={i}
                x={MARGIN.left - 6}
                y={toSvgY(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {v < 0.01 ? v.toExponential(0) : v.toFixed(2)}
              </text>
            ))}

            {/* Axis labels */}
            <text
              x={MARGIN.left + plotW / 2}
              y={chartH - 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
            >
              x
            </text>
            <text
              x={12}
              y={MARGIN.top + plotH / 2}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
            >
              {isDiscrete ? 'P(X=k)' : 'f(x)'}
            </text>

            {/* E[X] label */}
            {isFinite(mean) && mean >= xMin && mean <= xMax && (
              <text
                x={toSvgX(mean)}
                y={MARGIN.top + plotH + 24}
                textAnchor="middle"
                fontSize={9}
                fill="#DC2626"
                fontWeight={600}
              >
                E[X]
              </text>
            )}
          </svg>
        </div>

        {/* Right panel: canonical form display */}
        <div
          className={`flex-1 min-w-0 rounded-lg border p-3 flex flex-col justify-center ${isNarrow ? 'w-full' : ''}`}
          style={{ borderColor: 'var(--color-border)', minWidth: isNarrow ? undefined : 220 }}
        >
          <div className="text-xs font-semibold mb-2 text-center opacity-70">
            Canonical Form: {member.name}
          </div>

          {/* Full equation — LaTeX sourced from our own static data module */}
          <div
            className="text-center mb-3 overflow-x-auto"
            dangerouslySetInnerHTML={{ __html: canonicalHTML.fullEq }}
          />

          {/* Component breakdown — all LaTeX from exponential-family-data.ts */}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
            <div className="flex items-baseline gap-1">
              <span className="font-semibold" style={{ color: canonicalFormColors.h }}>h(x) =</span>
              <span dangerouslySetInnerHTML={{ __html: canonicalHTML.hHtml }} />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-semibold" style={{ color: canonicalFormColors.eta }}>{'\u03B7(\u03B8) ='}</span>
              <span dangerouslySetInnerHTML={{ __html: canonicalHTML.etaHtml }} />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-semibold" style={{ color: canonicalFormColors.T }}>T(x) =</span>
              <span dangerouslySetInnerHTML={{ __html: canonicalHTML.tHtml }} />
            </div>
            <div className="flex items-baseline gap-1">
              <span className="font-semibold" style={{ color: canonicalFormColors.A }}>{`A(\u03B7) =`}</span>
              <span dangerouslySetInnerHTML={{ __html: canonicalHTML.aHtml }} />
            </div>
          </div>

          {/* Color legend */}
          <div className="flex flex-wrap gap-3 justify-center mt-3 text-[10px] opacity-60">
            <span style={{ color: canonicalFormColors.h }}>{'\u25A0'} h(x)</span>
            <span style={{ color: canonicalFormColors.eta }}>{'\u25A0'} {'\u03B7(\u03B8)'}</span>
            <span style={{ color: canonicalFormColors.T }}>{'\u25A0'} T(x)</span>
            <span style={{ color: canonicalFormColors.A }}>{'\u25A0'} {`A(\u03B7)`}</span>
          </div>
        </div>
      </div>

      {/* Bottom panel: live readout */}
      <div className="mt-3 pt-3 border-t text-xs" style={{ borderColor: 'var(--color-border)' }}>
        {member.dimension === 1 && lpData ? (
          <div className="flex flex-wrap gap-x-6 gap-y-1 justify-center">
            <span>
              <strong>{'\u03B7'}</strong> = {lpData.eta.toFixed(4)}
            </span>
            <span>
              <strong>{`A(\u03B7)`}</strong> = {lpData.A.toFixed(4)}
            </span>
            <span>
              <strong>{`A\u2032(\u03B7)`}</strong> = E[T(X)] = {lpData.Aprime.toFixed(4)}
            </span>
            <span>
              <strong>{`A\u2033(\u03B7)`}</strong> = Var(T(X)) = {lpData.Adoubleprime.toFixed(4)}
            </span>
          </div>
        ) : (
          <div className="text-center opacity-70">
            {/* Vector natural parameter display — LaTeX from our data module */}
            <span dangerouslySetInnerHTML={{
              __html: renderKatex(
                `\\boldsymbol{\\eta} = ${member.eta}`,
                false,
              ),
            }} />
            <span className="mx-3">&mdash;</span>
            <span>See full derivation in sections 7.3 and 7.5.</span>
          </div>
        )}
      </div>
    </div>
  );
}
