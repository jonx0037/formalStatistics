import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  runningMean,
  sampleSequence,
  normalSample,
  exponentialSample,
  bernoulliSample,
  tSample,
  paretoSample,
  chebyshevBound,
} from './shared/convergence';
import { llnDistributions } from '../../data/law-of-large-numbers-data';
import type { LLNDistribution } from '../../data/law-of-large-numbers-data';

// ── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 50 };
const MAX_RENDER_POINTS = 500;

const PATH_BASE_COLOR = '#2563eb';
const TARGET_COLOR = '#111827';
const BAND_COLOR = '#BFDBFE';
const CHEBYSHEV_COLOR = '#059669';
const EMPIRICAL_COLOR = '#DC2626';

// ── Independence Modes ───────────────────────────────────────────────────────

type IndependenceMode = 'iid' | 'uncorrelated' | 'correlated';

// ── Sampler Factory ──────────────────────────────────────────────────────────

function getSampler(distId: string): () => number {
  switch (distId) {
    case 'normal':
      return () => normalSample(3, 2);
    case 'exponential':
      return () => exponentialSample(1);
    case 't-light':
      return () => tSample(3);
    case 't-heavy':
      return () => tSample(1.5);
    case 'cauchy':
      return () => tSample(1);
    case 'pareto-finite':
      return () => paretoSample(2);
    case 'pareto-no-mean':
      return () => paretoSample(0.8);
    case 'bernoulli':
      return () => bernoulliSample(0.3);
    default:
      return () => normalSample(0, 1);
  }
}

// ── Path Generation ──────────────────────────────────────────────────────────

interface PathData {
  /** Running mean values for indices 0..N-1 */
  means: number[];
}

function generatePaths(
  dist: LLNDistribution,
  n: number,
  numPaths: number,
  mode: IndependenceMode,
  rho: number,
): PathData[] {
  const paths: PathData[] = [];
  const baseSampler = getSampler(dist.id);
  const mu = dist.mu ?? 0;

  for (let p = 0; p < numPaths; p++) {
    let samples: number[];

    switch (mode) {
      case 'iid': {
        samples = sampleSequence(baseSampler, n);
        break;
      }
      case 'uncorrelated': {
        // X_i = mu + sigma_i * (Y_i - mu), where Y_i comes from the selected
        // base distribution and sigma_i = 1 + 0.5*sin(i/20).
        // Preserves the selected distribution family while introducing
        // time-varying scale without correlation across draws.
        samples = new Array<number>(n);
        for (let i = 0; i < n; i++) {
          const sigmaI = 1 + 0.5 * Math.sin(i / 20);
          samples[i] = mu + sigmaI * (baseSampler() - mu);
        }
        break;
      }
      case 'correlated': {
        // Mean-preserving AR(1):
        // X_i = mu + rho * (X_{i-1} - mu) + eps_i
        // where eps_i = baseSampler() - mu has mean 0.
        // Stationary mean is mu (not mu/(1-rho)).
        samples = new Array<number>(n);
        samples[0] = baseSampler();
        for (let i = 1; i < n; i++) {
          samples[i] = mu + rho * (samples[i - 1] - mu) + (baseSampler() - mu);
        }
        break;
      }
      default: {
        samples = sampleSequence(baseSampler, n);
      }
    }

    paths.push({ means: runningMean(samples) });
  }

  return paths;
}

// ── Deviation Proportion ────────────────────────────────────────────────────

function computeDeviationProportion(
  paths: PathData[],
  target: number,
  epsilon: number,
  N: number,
): number[] {
  const numPaths = paths.length;
  const proportions = new Array<number>(N);

  for (let i = 0; i < N; i++) {
    let outside = 0;
    for (let p = 0; p < numPaths; p++) {
      if (Math.abs(paths[p].means[i] - target) > epsilon) {
        outside++;
      }
    }
    proportions[i] = outside / numPaths;
  }

  return proportions;
}

// ── Downsampling ────────────────────────────────────────────────────────────

function downsample(values: number[], N: number): Array<{ n: number; value: number }> {
  const step = Math.max(1, Math.floor(N / MAX_RENDER_POINTS));
  const points: Array<{ n: number; value: number }> = [];
  for (let i = 0; i < N; i += step) {
    points.push({ n: i + 1, value: values[i] });
  }
  // Always include last point
  if (points.length === 0 || points[points.length - 1].n !== N) {
    points.push({ n: N, value: values[N - 1] });
  }
  return points;
}

// ── Logarithmic Slider Helpers ──────────────────────────────────────────────

function logSliderToValue(position: number, min: number, max: number): number {
  const minLog = Math.log(min);
  const maxLog = Math.log(max);
  return Math.round(Math.exp(minLog + (position / 100) * (maxLog - minLog)));
}

function valueToLogSlider(value: number, min: number, max: number): number {
  const minLog = Math.log(min);
  const maxLog = Math.log(max);
  return ((Math.log(value) - minLog) / (maxLog - minLog)) * 100;
}

// ── Tick Helpers ────────────────────────────────────────────────────────────

function niceTicksLinear(min: number, max: number, maxCount: number): number[] {
  const range = max - min;
  if (range <= 0) return [min];
  const niceSteps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  let step = 1;
  for (const s of niceSteps) {
    if (range / s <= maxCount) { step = s; break; }
  }
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 0.01; v += step) {
    ticks.push(Math.round(v * 10000) / 10000);
  }
  return ticks;
}

function niceTicksLog(min: number, max: number, maxCount: number): number[] {
  const ticks: number[] = [];
  const logMin = Math.log10(Math.max(1, min));
  const logMax = Math.log10(max);
  const decades = logMax - logMin;

  if (decades <= 2) {
    return niceTicksLinear(min, max, maxCount);
  }

  for (let exp = Math.floor(logMin); exp <= Math.ceil(logMax); exp++) {
    const val = Math.pow(10, exp);
    if (val >= min && val <= max) ticks.push(val);
    if (decades <= 4) {
      if (2 * val >= min && 2 * val <= max) ticks.push(2 * val);
      if (5 * val >= min && 5 * val <= max) ticks.push(5 * val);
    }
  }

  return ticks.sort((a, b) => a - b);
}

function formatTick(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)}k`;
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 0.01) return v.toExponential(0);
  return v.toFixed(2);
}

// ── SVG Path Builder ────────────────────────────────────────────────────────

function buildSvgPath(
  points: Array<{ n: number; value: number }>,
  toX: (n: number) => number,
  toY: (v: number) => number,
): string {
  let d = '';
  for (let i = 0; i < points.length; i++) {
    const px = toX(points[i].n);
    const py = toY(points[i].value);
    if (!isFinite(py) || !isFinite(px)) continue;
    d += d === '' ? `M${px},${py}` : `L${px},${py}`;
  }
  return d;
}

// ── Y-domain Computation ────────────────────────────────────────────────────

function computeYDomain(paths: PathData[]): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const path of paths) {
    for (const v of path.means) {
      if (isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  // Fallback if all paths are non-finite
  if (!isFinite(min)) min = -5;
  if (!isFinite(max)) max = 5;
  const pad = Math.max((max - min) * 0.1, 0.2);
  return [min - pad, max + pad];
}

// ── Component ────────────────────────────────────────────────────────────────

export default function WLLNExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls state
  const [distIndex, setDistIndex] = useState(0);
  const [independenceMode, setIndependenceMode] = useState<IndependenceMode>('iid');
  const [rho, setRho] = useState(0.8);
  const [n, setN] = useState(1000);
  const [numPaths, setNumPaths] = useState(10);
  const [epsilon, setEpsilon] = useState(0.5);
  const [seed, setSeed] = useState(0);

  const dist = llnDistributions[distIndex];
  const mu = dist.mu ?? 0;

  // ── Generate paths (memoized) ──────────────────────────────────────────
  const paths = useMemo(
    () => generatePaths(dist, n, numPaths, independenceMode, rho),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dist, n, numPaths, independenceMode, rho, seed],
  );

  // ── Deviation proportions ──────────────────────────────────────────────
  const deviationProportions = useMemo(
    () => computeDeviationProportion(paths, mu, epsilon, n),
    [paths, mu, epsilon, n],
  );

  // ── Chebyshev bound curve ──────────────────────────────────────────────
  const chebyshevCurve = useMemo(() => {
    if (!dist.hasVariance || !isFinite(dist.sigmaSquared)) return null;
    const points: Array<{ n: number; value: number }> = [];
    const step = Math.max(1, Math.floor(n / MAX_RENDER_POINTS));
    for (let i = 0; i < n; i += step) {
      const idx = i + 1;
      points.push({ n: idx, value: chebyshevBound(idx, dist.sigmaSquared, epsilon) });
    }
    if (points.length === 0 || points[points.length - 1].n !== n) {
      points.push({ n, value: chebyshevBound(n, dist.sigmaSquared, epsilon) });
    }
    return points;
  }, [dist.hasVariance, dist.sigmaSquared, n, epsilon]);

  // ── Downsampled data for rendering ─────────────────────────────────────
  const downsampledPaths = useMemo(
    () => paths.map((p) => downsample(p.means, n)),
    [paths, n],
  );

  const downsampledDeviation = useMemo(
    () => downsample(deviationProportions, n),
    [deviationProportions, n],
  );

  // ── Y-domain for left panel ────────────────────────────────────────────
  const [yMin, yMax] = useMemo(() => computeYDomain(paths), [paths]);

  // ── Responsive layout ─────────────────────────────────────────────────
  const containerW = Math.max(280, (width || 700) - 16);
  const isNarrow = containerW < 640;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = 260;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Scale helpers (left panel: sample paths) ───────────────────────────
  const useLogX = n > 200;

  const toSvgXLeft = useCallback(
    (nVal: number) => {
      if (useLogX) {
        const logN = Math.log(Math.max(1, nVal));
        const logMax = Math.log(n);
        return MARGIN.left + (logN / logMax) * plotW;
      }
      return MARGIN.left + ((nVal - 1) / (n - 1)) * plotW;
    },
    [n, plotW, useLogX],
  );

  const toSvgYLeft = useCallback(
    (v: number) => {
      const clamped = Math.max(yMin, Math.min(yMax, v));
      return MARGIN.top + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
    },
    [yMin, yMax, plotH],
  );

  // ── Scale helpers (right panel: probability decay) ─────────────────────
  const toSvgXRight = toSvgXLeft;

  const toSvgYRight = useCallback(
    (p: number) => {
      const clamped = Math.max(0, Math.min(1, p));
      return MARGIN.top + plotH - clamped * plotH;
    },
    [plotH],
  );

  // ── Tick values ────────────────────────────────────────────────────────
  const xTicks = useMemo(() => {
    if (useLogX) return niceTicksLog(1, n, 6);
    return niceTicksLinear(1, n, 6);
  }, [n, useLogX]);

  const yTicksLeft = useMemo(() => niceTicksLinear(yMin, yMax, 5), [yMin, yMax]);
  const yTicksRight = useMemo(() => [0, 0.25, 0.5, 0.75, 1.0], []);

  // ── Epsilon band coordinates ───────────────────────────────────────────
  const bandTop = toSvgYLeft(mu + epsilon);
  const bandBottom = toSvgYLeft(mu - epsilon);
  const bandHeight = Math.max(0, bandBottom - bandTop);

  // ── Path opacity ramp ─────────────────────────────────────────────────
  const pathOpacity = useCallback(
    (i: number) => {
      if (numPaths <= 1) return 0.5;
      return 0.15 + (0.35 * i) / (numPaths - 1);
    },
    [numPaths],
  );

  // ── Status indicator ──────────────────────────────────────────────────
  const statusInfo = useMemo(() => {
    if (dist.llnApplies === 'slln') {
      return { color: '#059669', bg: 'rgba(5, 150, 105, 0.1)', text: 'SLLN applies \u2713' };
    }
    if (dist.llnApplies === 'wlln') {
      return { color: '#D97706', bg: 'rgba(217, 119, 6, 0.1)', text: 'WLLN only (slow)' };
    }
    return { color: '#DC2626', bg: 'rgba(220, 38, 38, 0.1)', text: 'LLN does not apply \u2717' };
  }, [dist.llnApplies]);

  // ── Resample handler ──────────────────────────────────────────────────
  const handleResample = useCallback(() => {
    setSeed((prev) => prev + 1);
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Weak Law of Large Numbers Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        {/* Distribution dropdown */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={distIndex}
            onChange={(e) => setDistIndex(Number(e.target.value))}
          >
            {llnDistributions.map((d, i) => (
              <option key={d.id} value={i}>
                {d.name}
              </option>
            ))}
          </select>
        </label>

        {/* Independence mode toggle */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Mode</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={independenceMode}
            onChange={(e) => setIndependenceMode(e.target.value as IndependenceMode)}
          >
            <option value="iid">iid</option>
            <option value="uncorrelated">uncorrelated (varying &#x03C3;&#x1D62;)</option>
            <option value="correlated">correlated (AR(1))</option>
          </select>
        </label>

        {/* Rho slider (AR(1) mode only) */}
        {independenceMode === 'correlated' && (
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">&rho;</span>
            <input
              type="range"
              min={0}
              max={0.99}
              step={0.01}
              value={rho}
              onChange={(e) => setRho(Number(e.target.value))}
              className="w-20"
            />
            <span className="w-10 tabular-nums text-right">{rho.toFixed(2)}</span>
          </label>
        )}
      </div>

      <div className="flex flex-wrap gap-4 justify-center mb-3">
        {/* N slider (logarithmic) */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={valueToLogSlider(n, 10, 10000)}
            onChange={(e) => setN(logSliderToValue(Number(e.target.value), 10, 10000))}
            className="w-24"
          />
          <span className="w-14 tabular-nums text-right">{n.toLocaleString()}</span>
        </label>

        {/* Paths slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Paths</span>
          <input
            type="range"
            min={1}
            max={30}
            step={1}
            value={numPaths}
            onChange={(e) => setNumPaths(Number(e.target.value))}
            className="w-20"
          />
          <span className="w-6 tabular-nums text-right">{numPaths}</span>
        </label>

        {/* Epsilon slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">&epsilon;</span>
          <input
            type="range"
            min={0.05}
            max={2}
            step={0.05}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            className="w-20"
          />
          <span className="w-10 tabular-nums text-right">{epsilon.toFixed(2)}</span>
        </label>

        {/* Resample button */}
        <button
          className="rounded border px-3 py-1 text-xs font-medium hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          style={{ borderColor: 'var(--color-border)' }}
          onClick={handleResample}
        >
          Resample
        </button>
      </div>

      {/* Distribution description */}
      <div className="text-center text-xs mb-2" style={{ color: 'var(--color-text-muted)' }}>
        {dist.description}
      </div>

      {/* Status indicator */}
      <div className="flex justify-center mb-3">
        <span
          className="inline-block rounded-full px-3 py-0.5 text-xs font-medium"
          style={{ background: statusInfo.bg, color: statusInfo.color }}
        >
          {statusInfo.text}
        </span>
      </div>

      {/* Dual-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* ── Left panel: Sample Paths ────────────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Sample paths of X&#x0304;&#x2099;
          </div>
          <svg width={panelW} height={chartH} className="block mx-auto">
            {/* Y grid lines */}
            {yTicksLeft.map((v, i) => (
              <line
                key={`ygl-${i}`}
                x1={MARGIN.left}
                y1={toSvgYLeft(v)}
                x2={panelW - MARGIN.right}
                y2={toSvgYLeft(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* X grid lines */}
            {xTicks.map((v, i) => (
              <line
                key={`xgl-${i}`}
                x1={toSvgXLeft(v)}
                y1={MARGIN.top}
                x2={toSvgXLeft(v)}
                y2={MARGIN.top + plotH}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Epsilon band */}
            <rect
              x={MARGIN.left}
              y={bandTop}
              width={plotW}
              height={bandHeight}
              fill={BAND_COLOR}
              opacity={0.3}
            />

            {/* Target mu line (dashed) */}
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(mu)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(mu)}
              stroke={TARGET_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,3"
              opacity={0.7}
            />

            {/* Epsilon band edges */}
            <line
              x1={MARGIN.left}
              y1={bandTop}
              x2={panelW - MARGIN.right}
              y2={bandTop}
              stroke={BAND_COLOR}
              strokeWidth={0.8}
              strokeDasharray="3,3"
              opacity={0.5}
            />
            <line
              x1={MARGIN.left}
              y1={bandTop + bandHeight}
              x2={panelW - MARGIN.right}
              y2={bandTop + bandHeight}
              stroke={BAND_COLOR}
              strokeWidth={0.8}
              strokeDasharray="3,3"
              opacity={0.5}
            />

            {/* Sample paths */}
            {downsampledPaths.map((points, i) => {
              const d = buildSvgPath(points, toSvgXLeft, toSvgYLeft);
              if (!d) return null;
              return (
                <path
                  key={`path-${i}`}
                  d={d}
                  fill="none"
                  stroke={PATH_BASE_COLOR}
                  strokeWidth={1.2}
                  opacity={pathOpacity(i)}
                />
              );
            })}

            {/* Band labels */}
            <text
              x={panelW - MARGIN.right - 2}
              y={bandTop - 3}
              textAnchor="end"
              fontSize={8}
              fill={PATH_BASE_COLOR}
              opacity={0.7}
            >
              &#x03BC;+&#x03B5;
            </text>
            <text
              x={panelW - MARGIN.right - 2}
              y={bandTop + bandHeight + 10}
              textAnchor="end"
              fontSize={8}
              fill={PATH_BASE_COLOR}
              opacity={0.7}
            >
              &#x03BC;&minus;&#x03B5;
            </text>

            {/* Target label */}
            <text
              x={MARGIN.left + 4}
              y={toSvgYLeft(mu) - 4}
              textAnchor="start"
              fontSize={8}
              fill={TARGET_COLOR}
              opacity={0.7}
            >
              &#x03BC; = {mu}
            </text>

            {/* Axes */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={panelW - MARGIN.right}
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
                key={`xtl-${i}`}
                x={toSvgXLeft(v)}
                y={MARGIN.top + plotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {formatTick(v)}
              </text>
            ))}

            {/* X-axis label */}
            <text
              x={MARGIN.left + plotW / 2}
              y={chartH - 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              n
            </text>

            {/* Y-axis ticks */}
            {yTicksLeft.map((v, i) => (
              <text
                key={`ytl-${i}`}
                x={MARGIN.left - 6}
                y={toSvgYLeft(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {formatTick(v)}
              </text>
            ))}

            {/* Y-axis label */}
            <text
              x={12}
              y={MARGIN.top + plotH / 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
            >
              X&#x0304;&#x2099;
            </text>
          </svg>
        </div>

        {/* ── Right panel: Probability Decay ─────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            P(|X&#x0304;&#x2099; &minus; &#x03BC;| &gt; &#x03B5;) vs n
          </div>
          <svg width={panelW} height={chartH} className="block mx-auto">
            {/* Y grid lines */}
            {yTicksRight.map((v, i) => (
              <line
                key={`ygr-${i}`}
                x1={MARGIN.left}
                y1={toSvgYRight(v)}
                x2={panelW - MARGIN.right}
                y2={toSvgYRight(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* X grid lines */}
            {xTicks.map((v, i) => (
              <line
                key={`xgr-${i}`}
                x1={toSvgXRight(v)}
                y1={MARGIN.top}
                x2={toSvgXRight(v)}
                y2={MARGIN.top + plotH}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Chebyshev bound (dashed, if finite variance) */}
            {chebyshevCurve && (
              <path
                d={buildSvgPath(chebyshevCurve, toSvgXRight, toSvgYRight)}
                fill="none"
                stroke={CHEBYSHEV_COLOR}
                strokeWidth={1.8}
                strokeDasharray="6,3"
                opacity={0.8}
              />
            )}

            {/* Empirical probability (solid) */}
            <path
              d={buildSvgPath(downsampledDeviation, toSvgXRight, toSvgYRight)}
              fill="none"
              stroke={EMPIRICAL_COLOR}
              strokeWidth={2}
              opacity={0.85}
            />

            {/* Shaded area under empirical curve */}
            {(() => {
              const pts = downsampledDeviation;
              if (pts.length === 0) return null;
              let areaD = `M${toSvgXRight(pts[0].n)},${toSvgYRight(0)}`;
              for (const pt of pts) {
                areaD += `L${toSvgXRight(pt.n)},${toSvgYRight(pt.value)}`;
              }
              areaD += `L${toSvgXRight(pts[pts.length - 1].n)},${toSvgYRight(0)}Z`;
              return <path d={areaD} fill={EMPIRICAL_COLOR} opacity={0.08} />;
            })()}

            {/* Axes */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={panelW - MARGIN.right}
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
                key={`xtr-${i}`}
                x={toSvgXRight(v)}
                y={MARGIN.top + plotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {formatTick(v)}
              </text>
            ))}

            {/* X-axis label */}
            <text
              x={MARGIN.left + plotW / 2}
              y={chartH - 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              n
            </text>

            {/* Y-axis ticks */}
            {yTicksRight.map((v, i) => (
              <text
                key={`ytr-${i}`}
                x={MARGIN.left - 6}
                y={toSvgYRight(v) + 3}
                textAnchor="end"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {v.toFixed(2)}
              </text>
            ))}

            {/* Y-axis label */}
            <text
              x={12}
              y={MARGIN.top + plotH / 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
              transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
            >
              Probability
            </text>

            {/* Legend */}
            <g transform={`translate(${MARGIN.left + 8}, ${MARGIN.top + 8})`}>
              {/* Empirical */}
              <line x1={0} y1={0} x2={18} y2={0} stroke={EMPIRICAL_COLOR} strokeWidth={2} />
              <text x={22} y={3} fontSize={8} fill="currentColor" opacity={0.7}>
                Empirical
              </text>
              {/* Chebyshev */}
              {chebyshevCurve && (
                <>
                  <line
                    x1={0}
                    y1={14}
                    x2={18}
                    y2={14}
                    stroke={CHEBYSHEV_COLOR}
                    strokeWidth={1.8}
                    strokeDasharray="4,2"
                  />
                  <text x={22} y={17} fontSize={8} fill="currentColor" opacity={0.7}>
                    Chebyshev
                  </text>
                </>
              )}
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}
