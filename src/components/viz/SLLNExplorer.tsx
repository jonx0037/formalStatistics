import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  normalSample,
  exponentialSample,
  bernoulliSample,
  tSample,
  paretoSample,
  sampleSequence,
  runningMean,
} from './shared/convergence';
import { llnDistributions } from '../../data/law-of-large-numbers-data';

// ── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 50 };
const MAX_RENDER_POINTS = 500;
const CHART_HEIGHT = 280;

// Colors
const PATH_COLOR = '#2563eb';
const PATH_OPACITY = 0.12;
const MU_LINE_COLOR = '#111827';
const BAND_FILL = '#BFDBFE';
const BAND_OPACITY = 0.25;
const WLLN_COLOR = '#2563eb';
const SLLN_COLOR = '#7C3AED';
const STATUS_GREEN = '#059669';
const STATUS_YELLOW = '#d97706';
const STATUS_RED = '#DC2626';

// ── Sampler Factory ──────────────────────────────────────────────────────────

function getSampler(distId: string, nu?: number): () => number {
  // When the nu slider is used for t-distributions, override with custom nu
  if (nu !== undefined && (distId === 't-light' || distId === 't-heavy' || distId === 'cauchy')) {
    return () => tSample(nu);
  }

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

// ── Logarithmic Slider Helpers ───────────────────────────────────────────────

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

// ── Tick Helpers ─────────────────────────────────────────────────────────────

function niceTicksLinear(min: number, max: number, maxCount: number): number[] {
  const range = max - min;
  if (range <= 0) return [min];
  const niceSteps = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200,
    500, 1000, 2000, 5000,
  ];
  let step = 1;
  for (const s of niceSteps) {
    if (range / s <= maxCount) {
      step = s;
      break;
    }
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

// ── Y-domain Computation ─────────────────────────────────────────────────────

function computeYDomain(paths: number[][], distId: string): [number, number] {
  // For well-behaved distributions, clamp to [-10, 10]
  // For Cauchy and Pareto(0.8), allow wider range
  const wideDists = new Set(['cauchy', 'pareto-no-mean']);
  const clampRange = wideDists.has(distId) ? 100 : 10;

  let min = Infinity;
  let max = -Infinity;
  for (const path of paths) {
    for (const v of path) {
      const clamped = Math.max(-clampRange, Math.min(clampRange, v));
      if (clamped < min) min = clamped;
      if (clamped > max) max = clamped;
    }
  }

  const pad = Math.max((max - min) * 0.08, 0.2);
  return [min - pad, max + pad];
}

// ── Downsampling ─────────────────────────────────────────────────────────────

function downsample(
  values: number[],
  n: number,
): Array<{ n: number; value: number }> {
  const k = Math.max(1, Math.floor(n / MAX_RENDER_POINTS));
  const points: Array<{ n: number; value: number }> = [];
  for (let i = 0; i < n; i += k) {
    points.push({ n: i + 1, value: values[i] });
  }
  // Always include the last point
  if (points.length === 0 || points[points.length - 1].n !== n) {
    points.push({ n, value: values[n - 1] });
  }
  return points;
}

// ── Determine which t-distributions should use custom nu ─────────────────────

function isTDist(distId: string): boolean {
  return distId === 't-light' || distId === 't-heavy' || distId === 'cauchy';
}

// ── Component ────────────────────────────────────────────────────────────────

export default function SLLNExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls state
  const [distIndex, setDistIndex] = useState(0);
  const [n, setN] = useState(5000);
  const [numPaths, setNumPaths] = useState(20);
  const [epsilon, setEpsilon] = useState(0.5);
  const [nu, setNu] = useState(3);
  const [viewMode, setViewMode] = useState<'wlln' | 'slln'>('slln');
  const [seed, setSeed] = useState(0);

  const dist = llnDistributions[distIndex];
  const showNuSlider = isTDist(dist.id);

  // Effective mu: for Cauchy/Pareto(0.8) where mu is undefined, display target is 0
  const mu = dist.mu ?? 0;

  // Effective nu for t-distributions
  const effectiveNu = showNuSlider ? nu : undefined;

  // Derive the status from the current distribution (or from nu if overridden)
  const status = useMemo(() => {
    // If a custom nu is active for a t-distribution, re-derive status
    if (showNuSlider) {
      if (nu <= 1) return 'none' as const; // E[|X|] = infinity for nu <= 1
      if (nu <= 2) return 'slln' as const; // finite mean, infinite variance -> SLLN via Khintchine
      return 'slln' as const; // finite mean and variance -> full SLLN
    }
    return dist.llnApplies;
  }, [dist, showNuSlider, nu]);

  // Status label and color
  const statusConfig = useMemo(() => {
    switch (status) {
      case 'slln':
        return {
          color: STATUS_GREEN,
          label: 'SLLN applies (E[|X|] < \u221E)',
        };
      case 'wlln':
        return {
          color: STATUS_YELLOW,
          label: 'WLLN only (E[X\u00B2] = \u221E)',
        };
      case 'none':
        return {
          color: STATUS_RED,
          label: 'Neither WLLN nor SLLN (E[|X|] = \u221E)',
        };
    }
  }, [status]);

  // Handle distribution change
  const handleDistChange = useCallback((index: number) => {
    setDistIndex(index);
    const d = llnDistributions[index];
    // Set sensible default nu when switching to t-distributions
    if (isTDist(d.id)) {
      if (d.id === 'cauchy') setNu(1);
      else if (d.id === 't-heavy') setNu(1.5);
      else setNu(3);
    }
  }, []);

  // ── Generate all path data ─────────────────────────────────────────────────
  const pathData = useMemo(() => {
    // seed is in the dependency array to trigger resample
    void seed;
    const sampler = getSampler(dist.id, effectiveNu);
    const paths: number[][] = [];
    for (let p = 0; p < numPaths; p++) {
      const samples = sampleSequence(sampler, n);
      paths.push(runningMean(samples));
    }
    return paths;
  }, [dist.id, n, numPaths, effectiveNu, seed]);

  // ── Downsampled paths for rendering ────────────────────────────────────────
  const downsampled = useMemo(() => {
    return pathData.map((path) => downsample(path, n));
  }, [pathData, n]);

  // ── Convergence tracking ───────────────────────────────────────────────────
  const convergenceData = useMemo(() => {
    const k = Math.max(1, Math.floor(n / MAX_RENDER_POINTS));
    const indices: number[] = [];
    for (let i = 0; i < n; i += k) {
      indices.push(i);
    }
    // Always include the last index
    if (indices[indices.length - 1] !== n - 1) {
      indices.push(n - 1);
    }

    const wllnFraction = indices.map((idx) => {
      let count = 0;
      for (const path of pathData) {
        if (Math.abs(path[idx] - mu) <= epsilon) count++;
      }
      return count / numPaths;
    });

    const sllnFraction = indices.map((idx) => {
      let count = 0;
      for (const path of pathData) {
        // A path is "locked in" at index idx if it stays within the ε-band
        // for all m from idx to the end of the path. This is the correct
        // finite-data proxy for a.s. convergence (tail behavior).
        let stayedIn = true;
        for (let j = idx; j < n; j++) {
          if (Math.abs(path[j] - mu) > epsilon) {
            stayedIn = false;
            break;
          }
        }
        if (stayedIn) count++;
      }
      return count / numPaths;
    });

    return { indices, wllnFraction, sllnFraction };
  }, [pathData, epsilon, mu, n, numPaths]);

  // ── Convergence curve data for rendering ───────────────────────────────────
  const wllnPoints = useMemo(
    () =>
      convergenceData.indices.map((idx, i) => ({
        n: idx + 1,
        value: convergenceData.wllnFraction[i],
      })),
    [convergenceData],
  );

  const sllnPoints = useMemo(
    () =>
      convergenceData.indices.map((idx, i) => ({
        n: idx + 1,
        value: convergenceData.sllnFraction[i],
      })),
    [convergenceData],
  );

  // ── Y-domain for left panel ────────────────────────────────────────────────
  const [yMin, yMax] = useMemo(
    () => computeYDomain(pathData, dist.id),
    [pathData, dist.id],
  );

  // ── Responsive layout ─────────────────────────────────────────────────────
  const containerW = Math.max(280, (width || 700) - 16);
  const isNarrow = containerW < 640;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = CHART_HEIGHT;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── X scale (log when n > 200) ─────────────────────────────────────────────
  const useLogX = n > 200;

  const toSvgX = useCallback(
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

  // ── Y scale for left panel (sample paths) ─────────────────────────────────
  const toSvgYLeft = useCallback(
    (v: number) => {
      const clamped = Math.max(yMin, Math.min(yMax, v));
      return MARGIN.top + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
    },
    [yMin, yMax, plotH],
  );

  // ── Y scale for right panel (fractions 0..1) ──────────────────────────────
  const toSvgYRight = useCallback(
    (p: number) => {
      const clamped = Math.max(0, Math.min(1, p));
      return MARGIN.top + plotH - clamped * plotH;
    },
    [plotH],
  );

  // ── D3 line generators ────────────────────────────────────────────────────
  const lineLeft = useMemo(
    () =>
      d3
        .line<{ n: number; value: number }>()
        .x((d) => toSvgX(d.n))
        .y((d) => toSvgYLeft(d.value))
        .defined((d) => isFinite(d.value)),
    [toSvgX, toSvgYLeft],
  );

  const lineRight = useMemo(
    () =>
      d3
        .line<{ n: number; value: number }>()
        .x((d) => toSvgX(d.n))
        .y((d) => toSvgYRight(d.value)),
    [toSvgX, toSvgYRight],
  );

  // ── Tick values ────────────────────────────────────────────────────────────
  const xTicks = useMemo(() => {
    if (useLogX) return niceTicksLog(1, n, 6);
    return niceTicksLinear(1, n, 6);
  }, [n, useLogX]);

  const yTicksLeft = useMemo(
    () => niceTicksLinear(yMin, yMax, 5),
    [yMin, yMax],
  );

  const yTicksRight = useMemo(() => [0, 0.25, 0.5, 0.75, 1.0], []);

  // ── Epsilon band coordinates ───────────────────────────────────────────────
  const bandTop = toSvgYLeft(mu + epsilon);
  const bandBottom = toSvgYLeft(mu - epsilon);
  const bandHeight = Math.max(0, bandBottom - bandTop);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: SLLN Explorer — Every Path Converges
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        {/* Distribution dropdown */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-bg)',
            }}
            value={distIndex}
            onChange={(e) => handleDistChange(Number(e.target.value))}
          >
            {llnDistributions.map((d, i) => (
              <option key={d.id} value={i}>
                {d.name}
              </option>
            ))}
          </select>
        </label>

        {/* Nu slider (t-distributions only) */}
        {showNuSlider && (
          <label className="flex items-center gap-2 text-xs">
            <span className="font-medium">&nu;</span>
            <input
              type="range"
              min={0.5}
              max={10}
              step={0.5}
              value={nu}
              onChange={(e) => setNu(Number(e.target.value))}
              className="w-24"
            />
            <span className="w-8 tabular-nums text-right">{nu.toFixed(1)}</span>
          </label>
        )}

        {/* N slider (logarithmic) */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={valueToLogSlider(n, 100, 50000)}
            onChange={(e) =>
              setN(logSliderToValue(Number(e.target.value), 100, 50000))
            }
            className="w-24"
          />
          <span className="w-14 tabular-nums text-right">
            {n.toLocaleString()}
          </span>
        </label>

        {/* Paths slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Paths</span>
          <input
            type="range"
            min={1}
            max={50}
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
            min={0.1}
            max={2}
            step={0.1}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            className="w-20"
          />
          <span className="w-8 tabular-nums text-right">
            {epsilon.toFixed(1)}
          </span>
        </label>
      </div>

      {/* Second row: view toggle + resample */}
      <div className="flex flex-wrap gap-4 justify-center mb-3">
        {/* WLLN / SLLN view toggle */}
        <div className="flex items-center gap-1 text-xs">
          <button
            className="rounded-l px-3 py-1 border text-xs font-medium transition-colors"
            style={{
              borderColor: 'var(--color-border)',
              background:
                viewMode === 'wlln'
                  ? WLLN_COLOR
                  : 'var(--color-bg)',
              color: viewMode === 'wlln' ? '#fff' : 'inherit',
            }}
            onClick={() => setViewMode('wlln')}
          >
            WLLN view
          </button>
          <button
            className="rounded-r px-3 py-1 border text-xs font-medium transition-colors"
            style={{
              borderColor: 'var(--color-border)',
              background:
                viewMode === 'slln'
                  ? SLLN_COLOR
                  : 'var(--color-bg)',
              color: viewMode === 'slln' ? '#fff' : 'inherit',
            }}
            onClick={() => setViewMode('slln')}
          >
            SLLN view
          </button>
        </div>

        {/* Resample button */}
        <button
          className="rounded border px-3 py-1 text-xs font-medium transition-colors hover:opacity-80"
          style={{
            borderColor: 'var(--color-border)',
            background: 'var(--color-bg)',
          }}
          onClick={() => setSeed((s) => s + 1)}
        >
          Resample
        </button>
      </div>

      {/* Description */}
      <div
        className="text-center text-xs mb-3"
        style={{ color: 'var(--color-text-muted)' }}
      >
        {dist.description}
      </div>

      {/* Status indicator */}
      <div className="flex justify-center items-center gap-2 mb-4">
        <span
          style={{
            display: 'inline-block',
            width: 10,
            height: 10,
            borderRadius: '50%',
            backgroundColor: statusConfig.color,
          }}
        />
        <span className="text-xs font-medium" style={{ color: statusConfig.color }}>
          {statusConfig.label}
        </span>
      </div>

      {/* Dual-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* ── Left Panel: Sample Paths of X-bar_n ─────────────────────────── */}
        <div
          className={isNarrow ? 'w-full' : ''}
          style={isNarrow ? {} : { width: panelW }}
        >
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
                x1={toSvgX(v)}
                y1={MARGIN.top}
                x2={toSvgX(v)}
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
              fill={BAND_FILL}
              opacity={BAND_OPACITY}
            />

            {/* Band upper/lower edges */}
            <line
              x1={MARGIN.left}
              y1={bandTop}
              x2={panelW - MARGIN.right}
              y2={bandTop}
              stroke={WLLN_COLOR}
              strokeWidth={0.7}
              strokeDasharray="3,3"
              opacity={0.35}
            />
            <line
              x1={MARGIN.left}
              y1={bandTop + bandHeight}
              x2={panelW - MARGIN.right}
              y2={bandTop + bandHeight}
              stroke={WLLN_COLOR}
              strokeWidth={0.7}
              strokeDasharray="3,3"
              opacity={0.35}
            />

            {/* Target mu line (dashed) */}
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(mu)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(mu)}
              stroke={MU_LINE_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,3"
              opacity={0.6}
            />

            {/* Sample paths */}
            {downsampled.map((points, i) => {
              const dStr = lineLeft(points);
              if (!dStr) return null;
              return (
                <path
                  key={`p-${i}`}
                  d={dStr}
                  fill="none"
                  stroke={PATH_COLOR}
                  strokeWidth={1}
                  opacity={PATH_OPACITY}
                />
              );
            })}

            {/* Band labels */}
            <text
              x={panelW - MARGIN.right - 2}
              y={bandTop - 3}
              textAnchor="end"
              fontSize={8}
              fill={WLLN_COLOR}
              opacity={0.6}
            >
              &mu; + &epsilon;
            </text>
            <text
              x={panelW - MARGIN.right - 2}
              y={bandTop + bandHeight + 10}
              textAnchor="end"
              fontSize={8}
              fill={WLLN_COLOR}
              opacity={0.6}
            >
              &mu; &minus; &epsilon;
            </text>

            {/* Mu label */}
            <text
              x={MARGIN.left + 4}
              y={toSvgYLeft(mu) - 4}
              textAnchor="start"
              fontSize={9}
              fill={MU_LINE_COLOR}
              opacity={0.7}
              fontWeight={500}
            >
              &mu; = {mu}
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
                x={toSvgX(v)}
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

        {/* ── Right Panel: Convergence Tracker ────────────────────────────── */}
        <div
          className={isNarrow ? 'w-full' : ''}
          style={isNarrow ? {} : { width: panelW }}
        >
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Convergence Tracker
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
                x1={toSvgX(v)}
                y1={MARGIN.top}
                x2={toSvgX(v)}
                y2={MARGIN.top + plotH}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* WLLN fraction curve (always shown) */}
            <path
              d={lineRight(wllnPoints) ?? ''}
              fill="none"
              stroke={WLLN_COLOR}
              strokeWidth={2}
              opacity={0.85}
            />

            {/* SLLN fraction curve (always shown) */}
            <path
              d={lineRight(sllnPoints) ?? ''}
              fill="none"
              stroke={SLLN_COLOR}
              strokeWidth={2}
              opacity={0.85}
            />

            {/* Shaded area under the active curve */}
            {(() => {
              const activePoints =
                viewMode === 'wlln' ? wllnPoints : sllnPoints;
              const activeColor =
                viewMode === 'wlln' ? WLLN_COLOR : SLLN_COLOR;
              const areaGen = d3
                .area<{ n: number; value: number }>()
                .x((d) => toSvgX(d.n))
                .y0(toSvgYRight(0))
                .y1((d) => toSvgYRight(d.value));
              const aStr = areaGen(activePoints);
              return aStr ? (
                <path d={aStr} fill={activeColor} opacity={0.08} />
              ) : null;
            })()}

            {/* Highlight which curve is selected */}
            <path
              d={
                lineRight(
                  viewMode === 'wlln' ? wllnPoints : sllnPoints,
                ) ?? ''
              }
              fill="none"
              stroke={viewMode === 'wlln' ? WLLN_COLOR : SLLN_COLOR}
              strokeWidth={3}
              opacity={0.3}
            />

            {/* Legend */}
            <g transform={`translate(${MARGIN.left + 8}, ${MARGIN.top + 12})`}>
              <line
                x1={0}
                y1={0}
                x2={14}
                y2={0}
                stroke={WLLN_COLOR}
                strokeWidth={2}
              />
              <text
                x={18}
                y={3}
                fontSize={8}
                fill="currentColor"
                opacity={0.7}
              >
                WLLN: within band at n
              </text>
              <line
                x1={0}
                y1={14}
                x2={14}
                y2={14}
                stroke={SLLN_COLOR}
                strokeWidth={2}
              />
              <text
                x={18}
                y={17}
                fontSize={8}
                fill="currentColor"
                opacity={0.7}
              >
                SLLN: locked in band
              </text>
            </g>

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
                x={toSvgX(v)}
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
              fraction of paths
            </text>
          </svg>
        </div>
      </div>

      {/* Annotation below charts */}
      <div
        className="text-center text-xs mt-3"
        style={{ color: 'var(--color-text-muted)' }}
      >
        {status === 'slln' && (
          <span>
            SLLN curve tracks paths that <em>entered the band and never left</em>{' '}
            — it always sits below the WLLN curve but catches up as n grows.
          </span>
        )}
        {status === 'wlln' && (
          <span>
            Only WLLN applies: the fraction within the band grows, but individual
            paths may re-exit.
          </span>
        )}
        {status === 'none' && (
          <span>
            Neither law applies: paths wander without settling. Try switching to
            a distribution with a finite mean to see convergence.
          </span>
        )}
      </div>
    </div>
  );
}
