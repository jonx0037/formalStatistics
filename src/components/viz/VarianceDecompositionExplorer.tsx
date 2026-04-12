import { useState, useMemo, useCallback, useRef } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pmfBinomial, pdfNormal } from './shared/distributions';
import {
  expectationDiscrete,
  varianceDiscrete,
  expectationDiscreteG,
  expectationContinuous,
  varianceContinuous,
  expectationContinuousG,
  stdDev,
} from './shared/moments';
import type { DiscreteDistribution, ContinuousDistribution } from './shared/moments';
// ── Distribution presets ──────────────────────────────────────────────────

interface SliderDef {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

const FAIR_DIE: DiscreteDistribution = {
  values: [1, 2, 3, 4, 5, 6],
  probabilities: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
};

const LOADED_DIE: DiscreteDistribution = {
  values: [1, 2, 3, 4, 5, 6],
  probabilities: [0.05, 0.05, 0.10, 0.15, 0.25, 0.40],
};

function getBinomialDist(n: number, p: number): DiscreteDistribution {
  const values = Array.from({ length: n + 1 }, (_, i) => i);
  const probabilities = values.map((k) => pmfBinomial(k, n, p));
  return { values, probabilities };
}

const PRESETS: {
  name: string;
  type: 'discrete' | 'continuous';
}[] = [
  { name: 'Fair Die', type: 'discrete' },
  { name: 'Loaded Die', type: 'discrete' },
  { name: 'Binomial(n,p)', type: 'discrete' },
  { name: 'Normal(\u03BC,\u03C3\u00B2)', type: 'continuous' },
];

const BINOMIAL_SLIDERS: SliderDef[] = [
  { key: 'n', label: 'n', min: 1, max: 30, step: 1 },
  { key: 'p', label: 'p', min: 0.01, max: 0.99, step: 0.01 },
];

const NORMAL_SLIDERS: SliderDef[] = [
  { key: 'mu', label: '\u03BC', min: -5, max: 5, step: 0.1 },
  { key: 'sigma2', label: '\u03C3\u00B2', min: 0.1, max: 5, step: 0.1 },
];

// ── Layout ────────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 20, bottom: 30, left: 50 };

// ── Helper: format to 4 decimal places ────────────────────────────────────

function fmt(n: number): string {
  return n.toFixed(4);
}

// ── Component ─────────────────────────────────────────────────────────────

export default function VarianceDecompositionExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  const [presetIdx, setPresetIdx] = useState(0);
  const [binomialParams, setBinomialParams] = useState<Record<string, number>>({ n: 10, p: 0.3 });
  const [normalParams, setNormalParams] = useState<Record<string, number>>({ mu: 0, sigma2: 1 });
  const [showSigmaBand, setShowSigmaBand] = useState(false);

  const svgRef = useRef<SVGSVGElement>(null);
  const svgW = Math.max(width, 320);
  const svgH = 340;
  const plotW = svgW - MARGIN.left - MARGIN.right;
  const plotH = svgH - MARGIN.top - MARGIN.bottom;

  const preset = PRESETS[presetIdx];
  const isDiscrete = preset.type === 'discrete';

  // ── Current sliders for the selected preset ─────────────────────────────

  const currentSliders: SliderDef[] = useMemo(() => {
    if (preset.name === 'Binomial(n,p)') return BINOMIAL_SLIDERS;
    if (preset.name === 'Normal(\u03BC,\u03C3\u00B2)') return NORMAL_SLIDERS;
    return [];
  }, [preset.name]);

  const currentParams = useMemo(() => {
    if (preset.name === 'Binomial(n,p)') return binomialParams;
    if (preset.name === 'Normal(\u03BC,\u03C3\u00B2)') return normalParams;
    return {};
  }, [preset.name, binomialParams, normalParams]);

  const updateParam = useCallback((key: string, value: number) => {
    if (preset.name === 'Binomial(n,p)') {
      setBinomialParams((prev) => ({ ...prev, [key]: key === 'n' ? Math.round(value) : value }));
    } else if (preset.name === 'Normal(\u03BC,\u03C3\u00B2)') {
      setNormalParams((prev) => ({ ...prev, [key]: value }));
    }
  }, [preset.name]);

  // ── Compute distribution data and moments ───────────────────────────────

  const discreteDist: DiscreteDistribution | null = useMemo(() => {
    switch (preset.name) {
      case 'Fair Die': return FAIR_DIE;
      case 'Loaded Die': return LOADED_DIE;
      case 'Binomial(n,p)': return getBinomialDist(binomialParams.n, binomialParams.p);
      default: return null;
    }
  }, [preset.name, binomialParams]);

  const continuousDist: ContinuousDistribution | null = useMemo(() => {
    if (preset.name !== 'Normal(\u03BC,\u03C3\u00B2)') return null;
    const mu = normalParams.mu;
    const sigma2 = normalParams.sigma2;
    const sigma = Math.sqrt(sigma2);
    return {
      pdf: (x: number) => pdfNormal(x, mu, sigma2),
      support: [mu - 4 * sigma, mu + 4 * sigma] as [number, number],
    };
  }, [preset.name, normalParams]);

  // Moments
  const moments = useMemo(() => {
    if (isDiscrete && discreteDist) {
      const mean = expectationDiscrete(discreteDist);
      const eXSq = expectationDiscreteG(discreteDist, (x) => x * x);
      const variance = varianceDiscrete(discreteDist);
      const defVariance = expectationDiscreteG(discreteDist, (x) => (x - mean) * (x - mean));
      const sigma = stdDev(variance);
      // Per-value deviations for the geometric display
      const deviations = discreteDist.values.map((x, i) => ({
        x,
        p: discreteDist.probabilities[i],
        dev: x - mean,
        devSq: (x - mean) * (x - mean),
        weighted: (x - mean) * (x - mean) * discreteDist.probabilities[i],
      }));
      return { mean, eXSq, variance, defVariance, sigma, deviations };
    }
    if (!isDiscrete && continuousDist) {
      const mean = expectationContinuous(continuousDist, 2000);
      const eXSq = expectationContinuousG(continuousDist, (x) => x * x, 2000);
      const variance = varianceContinuous(continuousDist, 2000);
      const defVariance = expectationContinuousG(continuousDist, (x) => (x - mean) * (x - mean), 2000);
      const sigma = stdDev(variance);
      return { mean, eXSq, variance, defVariance, sigma, deviations: [] };
    }
    return { mean: 0, eXSq: 0, variance: 0, defVariance: 0, sigma: 0, deviations: [] };
  }, [isDiscrete, discreteDist, continuousDist]);

  // ── Continuous PDF curve data ───────────────────────────────────────────

  const continuousData = useMemo(() => {
    if (isDiscrete || !continuousDist) return [];
    const [lo, hi] = continuousDist.support;
    const nPoints = 200;
    const step = (hi - lo) / nPoints;
    return Array.from({ length: nPoints + 1 }, (_, i) => {
      const x = lo + i * step;
      return { x, y: continuousDist.pdf(x) };
    });
  }, [isDiscrete, continuousDist]);

  // ── Scales ──────────────────────────────────────────────────────────────

  // X domain
  const [xMin, xMax] = useMemo((): [number, number] => {
    if (isDiscrete && discreteDist) {
      const vals = discreteDist.values;
      const lo = vals[0];
      const hi = vals[vals.length - 1];
      const pad = (hi - lo) * 0.15 || 1;
      return [lo - pad, hi + pad];
    }
    if (!isDiscrete && continuousDist) {
      return continuousDist.support;
    }
    return [0, 1];
  }, [isDiscrete, discreteDist, continuousDist]);

  // Y max
  const yMax = useMemo(() => {
    if (isDiscrete && discreteDist) {
      return Math.max(...discreteDist.probabilities, 0.01) * 1.15;
    }
    if (continuousData.length > 0) {
      return Math.max(...continuousData.map((d) => d.y), 0.01) * 1.15;
    }
    return 1;
  }, [isDiscrete, discreteDist, continuousData]);

  const xScale = useCallback(
    (val: number): number => MARGIN.left + ((val - xMin) / (xMax - xMin)) * plotW,
    [xMin, xMax, plotW],
  );

  const yScale = useCallback(
    (val: number): number => MARGIN.top + plotH - (val / yMax) * plotH,
    [yMax, plotH],
  );

  // ── Discrete bar geometry ───────────────────────────────────────────────

  const barWidth = useMemo(() => {
    if (!isDiscrete || !discreteDist || discreteDist.values.length <= 1) return plotW * 0.4;
    const range = discreteDist.values[discreteDist.values.length - 1] - discreteDist.values[0];
    return Math.max(6, Math.min(36, (plotW / (range + 2)) * 0.6));
  }, [isDiscrete, discreteDist, plotW]);

  // ── Deviation square sizes (scaled so the largest fits visually) ────────

  const maxSquarePx = useMemo(() => {
    return Math.min(plotH * 0.45, plotW * 0.12);
  }, [plotH, plotW]);

  const maxDevSq = useMemo(() => {
    if (moments.deviations.length === 0) return 1;
    return Math.max(...moments.deviations.map((d) => d.devSq), 0.01);
  }, [moments.deviations]);

  // ── Continuous PDF path ─────────────────────────────────────────────────

  const pdfPath = useMemo(() => {
    if (continuousData.length === 0) return '';
    return continuousData
      .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`)
      .join(' ');
  }, [continuousData, xScale, yScale]);

  // ── Sigma band path (shaded area between mu-sigma and mu+sigma) ─────

  const sigmaBandPath = useMemo(() => {
    if (!showSigmaBand || isDiscrete || continuousData.length === 0) return '';
    const lo = moments.mean - moments.sigma;
    const hi = moments.mean + moments.sigma;
    const filtered = continuousData.filter((d) => d.x >= lo && d.x <= hi);
    if (filtered.length < 2) return '';
    const baseline = yScale(0);
    let path = `M${xScale(filtered[0].x).toFixed(1)},${baseline}`;
    for (const d of filtered) {
      path += ` L${xScale(d.x).toFixed(1)},${yScale(d.y).toFixed(1)}`;
    }
    path += ` L${xScale(filtered[filtered.length - 1].x).toFixed(1)},${baseline} Z`;
    return path;
  }, [showSigmaBand, isDiscrete, continuousData, moments.mean, moments.sigma, xScale, yScale]);

  // ── X-axis ticks ────────────────────────────────────────────────────────

  const xTicks = useMemo(() => {
    if (isDiscrete && discreteDist) {
      return discreteDist.values;
    }
    // Generate ~6 ticks for continuous
    const count = 6;
    const range = xMax - xMin;
    const rawStep = range / count;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalized = rawStep / magnitude;
    let niceStep: number;
    if (normalized <= 1.5) niceStep = 1 * magnitude;
    else if (normalized <= 3.5) niceStep = 2 * magnitude;
    else if (normalized <= 7.5) niceStep = 5 * magnitude;
    else niceStep = 10 * magnitude;

    const ticks: number[] = [];
    const start = Math.ceil(xMin / niceStep) * niceStep;
    for (let t = start; t <= xMax; t += niceStep) {
      ticks.push(t);
    }
    return ticks;
  }, [isDiscrete, discreteDist, xMin, xMax]);

  // ── Y-axis ticks ────────────────────────────────────────────────────────

  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const count = 4;
    for (let i = 1; i <= count; i++) {
      const val = (yMax / count) * i;
      ticks.push(val);
    }
    return ticks;
  }, [yMax]);

  // ── Compact layout detection ────────────────────────────────────────────

  const compact = width < 640;

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
      role="figure"
      aria-label="Variance decomposition explorer: visualizes Var(X) = E[(X minus mu) squared] geometrically and shows the computational formula Var(X) = E[X squared] minus (E[X]) squared"
    >
      {/* Controls */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <label htmlFor="variance-dist-select" className="text-sm font-medium">
          Distribution
        </label>
        <select
          id="variance-dist-select"
          value={presetIdx}
          onChange={(e) => setPresetIdx(Number(e.target.value))}
          className="rounded border px-2 py-1 text-sm"
          style={{ borderColor: 'var(--color-border)' }}
        >
          {PRESETS.map((p, i) => (
            <option key={p.name} value={i}>{p.name}</option>
          ))}
        </select>

        <label className="ml-2 flex items-center gap-1.5 text-sm">
          <input
            type="checkbox"
            checked={showSigmaBand}
            onChange={(e) => setShowSigmaBand(e.target.checked)}
            className="rounded"
            aria-label="Show plus or minus sigma band"
          />
          Show &plusmn;&sigma; band
        </label>
      </div>

      {/* Parameter sliders */}
      {currentSliders.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-4">
          {currentSliders.map((s) => (
            <div key={s.key} className="flex items-center gap-2">
              <label htmlFor={`variance-slider-${s.key}`} className="text-sm font-medium" style={{ minWidth: '20px' }}>
                {s.label}
              </label>
              <input
                id={`variance-slider-${s.key}`}
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={currentParams[s.key]}
                onChange={(e) => updateParam(s.key, Number(e.target.value))}
                className="w-28"
                aria-valuemin={s.min}
                aria-valuemax={s.max}
                aria-valuenow={currentParams[s.key]}
              />
              <span className="text-sm font-mono w-12">
                {s.key === 'n' ? currentParams[s.key] : currentParams[s.key].toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Main layout: SVG + side panel */}
      <div className={compact ? 'space-y-4' : 'flex gap-4'}>
        {/* SVG chart */}
        <div className={compact ? '' : 'flex-1 min-w-0'}>
          <svg
            ref={svgRef}
            width={compact ? svgW : svgW - 280}
            height={svgH}
            className="block"
            role="img"
            aria-label={`${preset.name} distribution with variance decomposition overlay`}
          >
            {/* Axes */}
            <line
              x1={MARGIN.left}
              y1={yScale(0)}
              x2={(compact ? svgW : svgW - 280) - MARGIN.right}
              y2={yScale(0)}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={yScale(0)}
              stroke="currentColor"
              strokeOpacity={0.3}
            />

            {/* Y-axis ticks */}
            {yTicks.map((val) => {
              const yPos = yScale(val);
              if (yPos < MARGIN.top) return null;
              return (
                <g key={val}>
                  <line
                    x1={MARGIN.left - 4}
                    y1={yPos}
                    x2={MARGIN.left}
                    y2={yPos}
                    stroke="currentColor"
                    strokeOpacity={0.3}
                  />
                  <text
                    x={MARGIN.left - 8}
                    y={yPos + 4}
                    textAnchor="end"
                    className="fill-current"
                    style={{ fontSize: '10px' }}
                  >
                    {val.toFixed(2)}
                  </text>
                </g>
              );
            })}

            {/* X-axis ticks */}
            {xTicks.map((val) => {
              const xPos = xScale(val);
              if (xPos < MARGIN.left || xPos > (compact ? svgW : svgW - 280) - MARGIN.right) return null;
              return (
                <g key={val}>
                  <line
                    x1={xPos}
                    y1={yScale(0)}
                    x2={xPos}
                    y2={yScale(0) + 4}
                    stroke="currentColor"
                    strokeOpacity={0.3}
                  />
                  <text
                    x={xPos}
                    y={yScale(0) + 16}
                    textAnchor="middle"
                    className="fill-current"
                    style={{ fontSize: '10px' }}
                  >
                    {Number.isInteger(val) ? val : val.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Y-axis label */}
            <text
              x={14}
              y={MARGIN.top + plotH / 2}
              textAnchor="middle"
              className="fill-current"
              style={{ fontSize: '11px' }}
              transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
            >
              {isDiscrete ? 'P(X = x)' : 'f(x)'}
            </text>

            {/* Sigma band for continuous distributions */}
            {showSigmaBand && !isDiscrete && sigmaBandPath && (
              <path
                d={sigmaBandPath}
                style={{ fill: 'var(--color-warning, #F59E0B)' }}
                opacity={0.15}
              >
                <title>Region from {'\u03BC'} - {'\u03C3'} to {'\u03BC'} + {'\u03C3'}</title>
              </path>
            )}

            {/* Continuous: PDF curve */}
            {!isDiscrete && pdfPath && (
              <path
                d={pdfPath}
                fill="none"
                style={{ stroke: 'var(--color-primary, #2563EB)' }}
                strokeWidth={2}
              />
            )}

            {/* Discrete: bars */}
            {isDiscrete && discreteDist && discreteDist.values.map((x, i) => {
              const p = discreteDist.probabilities[i];
              const barX = xScale(x) - barWidth / 2;
              const barY = yScale(p);
              const barH = yScale(0) - barY;
              return (
                <rect
                  key={x}
                  x={barX}
                  y={barY}
                  width={barWidth}
                  height={Math.max(barH, 0)}
                  style={{ fill: 'var(--color-primary, #2563EB)' }}
                  opacity={0.65}
                  stroke="var(--color-primary, #2563EB)"
                  strokeWidth={1}
                >
                  <title>x = {x}, P(X = {x}) = {p.toFixed(4)}</title>
                </rect>
              );
            })}

            {/* Discrete: deviation squares overlay */}
            {isDiscrete && moments.deviations.map((d) => {
              const sidePx = Math.sqrt(d.devSq / maxDevSq) * maxSquarePx;
              if (sidePx < 2) return null;
              const cx = xScale(d.x);
              const baselineY = yScale(0);
              // Position the square sitting on the baseline, centered on the bar
              const sqX = cx - sidePx / 2;
              const sqY = baselineY - sidePx;
              return (
                <rect
                  key={`sq-${d.x}`}
                  x={sqX}
                  y={sqY}
                  width={sidePx}
                  height={sidePx}
                  style={{ fill: 'var(--color-warning, #F59E0B)' }}
                  opacity={0.3}
                  stroke="var(--color-warning, #F59E0B)"
                  strokeWidth={1}
                  strokeOpacity={0.5}
                  className="transition-all duration-300"
                >
                  <title>(x - {'\u03BC'})² = {d.devSq.toFixed(4)}, weighted: {d.weighted.toFixed(4)}</title>
                </rect>
              );
            })}

            {/* Mean line E[X] */}
            <line
              x1={xScale(moments.mean)}
              y1={MARGIN.top}
              x2={xScale(moments.mean)}
              y2={yScale(0)}
              style={{ stroke: 'var(--color-danger, #DC2626)' }}
              strokeWidth={2}
              strokeDasharray="6,3"
            >
              <title>E[X] = {fmt(moments.mean)}</title>
            </line>

            {/* Mean label */}
            <text
              x={xScale(moments.mean)}
              y={MARGIN.top - 5}
              textAnchor="middle"
              style={{ fill: 'var(--color-danger, #DC2626)', fontSize: '11px', fontWeight: 600 }}
            >
              E[X] = {moments.mean.toFixed(2)}
            </text>

            {/* Discrete: deviation lines from bar peaks to mean */}
            {isDiscrete && discreteDist && discreteDist.values.map((x, i) => {
              const p = discreteDist.probabilities[i];
              if (p < 0.001) return null;
              const barTopY = yScale(p);
              return (
                <line
                  key={`dev-${x}`}
                  x1={xScale(x)}
                  y1={barTopY}
                  x2={xScale(moments.mean)}
                  y2={barTopY}
                  style={{ stroke: 'var(--color-warning, #F59E0B)' }}
                  strokeWidth={1.5}
                  strokeDasharray="3,2"
                  opacity={0.7}
                >
                  <title>x - {'\u03BC'} = {(x - moments.mean).toFixed(4)}</title>
                </line>
              );
            })}

            {/* Sigma band markers for discrete */}
            {showSigmaBand && isDiscrete && (
              <>
                <line
                  x1={xScale(moments.mean - moments.sigma)}
                  y1={MARGIN.top}
                  x2={xScale(moments.mean - moments.sigma)}
                  y2={yScale(0)}
                  style={{ stroke: 'var(--color-warning, #F59E0B)' }}
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                  opacity={0.6}
                />
                <line
                  x1={xScale(moments.mean + moments.sigma)}
                  y1={MARGIN.top}
                  x2={xScale(moments.mean + moments.sigma)}
                  y2={yScale(0)}
                  style={{ stroke: 'var(--color-warning, #F59E0B)' }}
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                  opacity={0.6}
                />
                {/* Horizontal bracket */}
                <line
                  x1={xScale(moments.mean - moments.sigma)}
                  y1={yScale(0) + 22}
                  x2={xScale(moments.mean + moments.sigma)}
                  y2={yScale(0) + 22}
                  style={{ stroke: 'var(--color-warning, #F59E0B)' }}
                  strokeWidth={1.5}
                  opacity={0.8}
                />
                <text
                  x={xScale(moments.mean)}
                  y={yScale(0) + 28}
                  textAnchor="middle"
                  style={{ fill: 'var(--color-warning, #F59E0B)', fontSize: '9px' }}
                  dominantBaseline="hanging"
                >
                  {'\u03BC'} {'\u00B1'} {'\u03C3'}
                </text>
              </>
            )}

            {/* Sigma band boundary lines for continuous */}
            {showSigmaBand && !isDiscrete && (
              <>
                <line
                  x1={xScale(moments.mean - moments.sigma)}
                  y1={MARGIN.top}
                  x2={xScale(moments.mean - moments.sigma)}
                  y2={yScale(0)}
                  style={{ stroke: 'var(--color-warning, #F59E0B)' }}
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                  opacity={0.6}
                />
                <line
                  x1={xScale(moments.mean + moments.sigma)}
                  y1={MARGIN.top}
                  x2={xScale(moments.mean + moments.sigma)}
                  y2={yScale(0)}
                  style={{ stroke: 'var(--color-warning, #F59E0B)' }}
                  strokeWidth={1.5}
                  strokeDasharray="4,3"
                  opacity={0.6}
                />
                <text
                  x={xScale(moments.mean)}
                  y={yScale(0) + 22}
                  textAnchor="middle"
                  style={{ fill: 'var(--color-warning, #F59E0B)', fontSize: '9px' }}
                >
                  {'\u03BC'} {'\u00B1'} {'\u03C3'}
                </text>
              </>
            )}
          </svg>
        </div>

        {/* Side panel: formula readout */}
        <div
          className={compact ? 'rounded border p-3' : 'w-[260px] shrink-0 rounded border p-3'}
          style={{ borderColor: 'var(--color-border)' }}
          role="region"
          aria-label="Variance computation formulas"
        >
          <h3 className="mb-3 text-sm font-semibold">Variance Formulas</h3>

          {/* Definition formula */}
          <div className="mb-4">
            <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-muted, #6B7280)' }}>
              Definition (average squared deviation)
            </div>
            <div className="rounded px-2 py-1.5 text-sm font-mono" style={{ backgroundColor: 'var(--color-surface, #F9FAFB)' }}>
              <div className="mb-1">
                Var(X) = E[(X {'\u2212'} {'\u03BC'}){'\u00B2'}]
              </div>
              {isDiscrete && moments.deviations.length > 0 && (
                <div className="text-xs leading-relaxed" style={{ color: 'var(--color-muted, #6B7280)' }}>
                  {moments.deviations.map((d, i) => (
                    <div key={d.x}>
                      {i > 0 && <span>+ </span>}
                      ({d.x} {'\u2212'} {moments.mean.toFixed(2)}){'\u00B2'} {'\u00B7'} {d.p.toFixed(4)}
                    </div>
                  ))}
                </div>
              )}
              <div className="mt-1 font-semibold" style={{ color: 'var(--color-danger, #DC2626)' }}>
                = {fmt(moments.defVariance)}
              </div>
            </div>
          </div>

          {/* Computational formula */}
          <div className="mb-4">
            <div className="mb-1 text-xs font-medium" style={{ color: 'var(--color-muted, #6B7280)' }}>
              Computational formula
            </div>
            <div className="rounded px-2 py-1.5 text-sm font-mono" style={{ backgroundColor: 'var(--color-surface, #F9FAFB)' }}>
              <div className="mb-1">
                Var(X) = E[X{'\u00B2'}] {'\u2212'} (E[X]){'\u00B2'}
              </div>
              <div className="text-xs" style={{ color: 'var(--color-muted, #6B7280)' }}>
                <div>E[X{'\u00B2'}] = {fmt(moments.eXSq)}</div>
                <div>(E[X]){'\u00B2'} = ({fmt(moments.mean)}){'\u00B2'} = {fmt(moments.mean * moments.mean)}</div>
              </div>
              <div className="mt-1 font-semibold" style={{ color: 'var(--color-danger, #DC2626)' }}>
                = {fmt(moments.eXSq)} {'\u2212'} {fmt(moments.mean * moments.mean)} = {fmt(moments.variance)}
              </div>
            </div>
          </div>

          {/* Summary stats */}
          <div className="space-y-1 text-xs font-mono" style={{ color: 'var(--color-muted, #6B7280)' }}>
            <div>E[X] = {fmt(moments.mean)}</div>
            <div>E[X{'\u00B2'}] = {fmt(moments.eXSq)}</div>
            <div>Var(X) = {fmt(moments.variance)}</div>
            <div>{'\u03C3'} = {'\u221A'}Var(X) = {fmt(moments.sigma)}</div>
          </div>

          {/* Match confirmation */}
          <div
            className="mt-3 rounded px-2 py-1.5 text-xs"
            style={{
              backgroundColor: Math.abs(moments.defVariance - moments.variance) < 1e-6
                ? 'var(--color-success-bg, #ECFDF5)'
                : 'var(--color-warning-bg, #FFFBEB)',
              color: Math.abs(moments.defVariance - moments.variance) < 1e-6
                ? 'var(--color-success, #059669)'
                : 'var(--color-warning, #D97706)',
            }}
          >
            {Math.abs(moments.defVariance - moments.variance) < 1e-6
              ? 'Both formulas agree \u2014 same result.'
              : `Small numerical difference: ${Math.abs(moments.defVariance - moments.variance).toExponential(2)} (numerical integration rounding).`
            }
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-4 text-xs" style={{ color: 'var(--color-muted, #6B7280)' }}>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block h-3 w-3 rounded-sm"
            style={{ backgroundColor: 'var(--color-primary, #2563EB)', opacity: 0.65 }}
          />
          PMF / PDF
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4"
            style={{ backgroundColor: 'var(--color-danger, #DC2626)' }}
          />
          E[X] (mean)
        </div>
        {isDiscrete && (
          <>
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4"
                style={{ backgroundColor: 'var(--color-warning, #F59E0B)', opacity: 0.7 }}
              />
              Deviation lines
            </div>
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ backgroundColor: 'var(--color-warning, #F59E0B)', opacity: 0.3 }}
              />
              (x {'\u2212'} {'\u03BC'}){'\u00B2'} squares
            </div>
          </>
        )}
        {showSigmaBand && (
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ backgroundColor: 'var(--color-warning, #F59E0B)', opacity: 0.15 }}
            />
            {'\u03BC'} {'\u00B1'} {'\u03C3'} band
          </div>
        )}
      </div>
    </div>
  );
}
