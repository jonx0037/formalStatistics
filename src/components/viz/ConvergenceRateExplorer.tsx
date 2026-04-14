import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  sampleSequence,
  normalSample,
  uniformSample,
  bernoulliSample,
  exponentialSample,
  runningMean,
  chebyshevBound,
  hoeffdingBound,
  lilEnvelope,
} from './shared/convergence';
import { ratePresets } from '../../data/law-of-large-numbers-data';

// ── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 55 };
const CHART_H = 260;
const MAX_RENDER_POINTS = 300;

// Colors per ratePresets and spec
const CHEBYSHEV_COLOR = '#059669';  // emerald
const HOEFFDING_COLOR = '#d97706';  // amber
const CLT_COLOR = '#2563eb';        // blue (used for CLT and LIL labels)
const EMPIRICAL_COLOR = '#DC2626';  // red
const PATH_COLOR = '#6B7280';       // gray
const EPSILON_BAND_COLOR = '#E5E7EB';

// ── Distribution Parameters ─────────────────────────────────────────────────

interface DistParams {
  sampler: () => number;
  mu: number;
  sigma: number;
  a?: number;
  b?: number;
}

const distParams: Record<string, DistParams> = {
  uniform: {
    sampler: () => uniformSample(0, 1),
    mu: 0.5,
    sigma: Math.sqrt(1 / 12),
    a: 0,
    b: 1,
  },
  normal: {
    sampler: () => normalSample(0, 1),
    mu: 0,
    sigma: 1,
  },
  bernoulli: {
    sampler: () => bernoulliSample(0.5),
    mu: 0.5,
    sigma: 0.5,
    a: 0,
    b: 1,
  },
  exponential: {
    sampler: () => exponentialSample(1),
    mu: 1,
    sigma: 1,
  },
};

const distNames: Record<string, string> = {
  uniform: 'Uniform(0,1)',
  normal: 'Normal(0,1)',
  bernoulli: 'Bernoulli(0.5)',
  exponential: 'Exponential(1)',
};

const distKeys = Object.keys(distParams);

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

// ── Epsilon Slider Helper ───────────────────────────────────────────────────

function epsSliderToValue(position: number): number {
  // Map 0..100 to 0.01..1 on a log scale for fine control at small epsilon
  const minLog = Math.log(0.01);
  const maxLog = Math.log(1);
  return Math.exp(minLog + (position / 100) * (maxLog - minLog));
}

function valueToEpsSlider(value: number): number {
  const minLog = Math.log(0.01);
  const maxLog = Math.log(1);
  return ((Math.log(value) - minLog) / (maxLog - minLog)) * 100;
}

// ── Downsampling ────────────────────────────────────────────────────────────

function downsamplePath(
  values: number[],
  N: number,
): Array<{ n: number; value: number }> {
  const step = Math.max(1, Math.floor(N / MAX_RENDER_POINTS));
  const points: Array<{ n: number; value: number }> = [];
  for (let i = 0; i < N; i += step) {
    points.push({ n: i + 1, value: values[i] });
  }
  if (points.length === 0 || points[points.length - 1].n !== N) {
    points.push({ n: N, value: values[N - 1] });
  }
  return points;
}

// ── Tick Helpers ────────────────────────────────────────────────────────────

function niceTicksLinear(min: number, max: number, maxCount: number): number[] {
  const range = max - min;
  if (range <= 0) return [min];
  const niceSteps = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10,
    20, 50, 100, 200, 500, 1000, 2000, 5000,
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
    ticks.push(Math.round(v * 100000) / 100000);
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
  if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(0);
  if (Math.abs(v) < 0.1) return v.toFixed(2);
  return v.toFixed(1);
}

function formatBound(v: number | null): string {
  if (v === null) return 'N/A';
  if (v >= 1) return '1.000';
  if (v < 0.0001) return v.toExponential(2);
  return v.toFixed(4);
}

// ── Empirical Bound Estimation ──────────────────────────────────────────────

/**
 * Estimate P(|X̄_n - μ| > ε) from M Monte Carlo replications.
 * Uses a fixed seed-like approach by running the simulation inline.
 */
function estimateExceedance(
  sampler: () => number,
  mu: number,
  epsilon: number,
  ni: number,
  M: number,
): number {
  let count = 0;
  for (let r = 0; r < M; r++) {
    const samples = sampleSequence(sampler, ni);
    const mean = samples.reduce((a, b) => a + b, 0) / ni;
    if (Math.abs(mean - mu) > epsilon) count++;
  }
  return count / M;
}

// ── Component ───────────────────────────────────────────────────────────────

export default function ConvergenceRateExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls
  const [dist, setDist] = useState('uniform');
  const [n, setN] = useState(1000);
  const [epsilon, setEpsilon] = useState(0.1);
  const [numPaths, setNumPaths] = useState(5);
  const [mode, setMode] = useState<'fixed-eps' | 'fixed-n'>('fixed-eps');
  const [seed, setSeed] = useState(0);

  const params = distParams[dist];

  // ── Left Panel: Paths of (X̄_n - μ) ──────────────────────────────────────

  const pathData = useMemo(() => {
    const { sampler, mu } = params;
    const paths: number[][] = [];
    for (let p = 0; p < numPaths; p++) {
      const samples = sampleSequence(sampler, n);
      const means = runningMean(samples);
      paths.push(means.map((m) => m - mu));
    }
    return paths;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dist, n, numPaths, seed]);

  const downsampledPaths = useMemo(
    () => pathData.map((p) => downsamplePath(p, n)),
    [pathData, n],
  );

  // ── Left Panel: Envelopes ────────────────────────────────────────────────

  const envelopes = useMemo(() => {
    const { sigma } = params;
    const k = Math.max(1, Math.floor(n / MAX_RENDER_POINTS));
    const indices: number[] = [];
    for (let i = 0; i < Math.ceil(n / k); i++) {
      indices.push(Math.min(i * k + 1, n));
    }
    // Always include the last index
    if (indices[indices.length - 1] !== n) indices.push(n);

    return indices.map((ni) => ({
      n: ni,
      clt: sigma / Math.sqrt(ni),
      lil: lilEnvelope(ni, sigma),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dist, n]);

  // ── Right Panel: Probability Bounds ──────────────────────────────────────

  const boundsData = useMemo(() => {
    const { sigma, a, b, sampler, mu } = params;
    const sigSq = sigma * sigma;
    const isBounded = a !== undefined && b !== undefined;

    if (mode === 'fixed-eps') {
      // x-axis = n, y-axis = P(|X̄_n - μ| > ε)
      // Use ~20 n-values on a log scale
      const numPoints = 20;
      const nValues: number[] = [];
      for (let i = 0; i < numPoints; i++) {
        const logVal =
          1 + (i * (Math.log10(n) - 1)) / (numPoints - 1);
        nValues.push(Math.max(10, Math.round(Math.pow(10, logVal))));
      }
      // Deduplicate
      const unique = [...new Set(nValues)].sort((x, y) => x - y);

      return unique.map((ni) => ({
        x: ni,
        chebyshev: chebyshevBound(ni, sigSq, epsilon),
        hoeffding:
          isBounded ? hoeffdingBound(ni, a!, b!, epsilon) : null,
        empirical: estimateExceedance(sampler, mu, epsilon, ni, 500),
      }));
    } else {
      // Fixed n mode: x-axis = ε, y-axis = P(|X̄_n - μ| > ε)
      const numPoints = 25;
      const epsValues: number[] = [];
      for (let i = 0; i < numPoints; i++) {
        const logVal =
          Math.log(0.01) +
          (i * (Math.log(1) - Math.log(0.01))) / (numPoints - 1);
        epsValues.push(Math.exp(logVal));
      }

      return epsValues.map((eps) => ({
        x: eps,
        chebyshev: chebyshevBound(n, sigSq, eps),
        hoeffding:
          isBounded ? hoeffdingBound(n, a!, b!, eps) : null,
        empirical: estimateExceedance(sampler, mu, eps, n, 500),
      }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dist, n, epsilon, mode, seed]);

  // ── Y-domain for left panel (symmetric) ─────────────────────────────────

  const [yMinLeft, yMaxLeft] = useMemo(() => {
    let absMax = 0;
    for (const path of pathData) {
      for (const v of path) {
        const av = Math.abs(v);
        if (isFinite(av) && av > absMax) absMax = av;
      }
    }
    // Include the envelope bounds
    const { sigma } = params;
    const cltMax = sigma / Math.sqrt(1);
    absMax = Math.max(absMax, cltMax, epsilon);
    const pad = absMax * 1.15;
    return [-pad, pad];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathData, dist, epsilon]);

  // ── Responsive Layout ───────────────────────────────────────────────────

  const containerW = Math.max(280, (width || 700) - 16);
  const isNarrow = containerW < 624;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = isNarrow ? 220 : CHART_H;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Scale helpers (left panel) ──────────────────────────────────────────

  const useLogX = n > 200;

  const toSvgXLeft = useCallback(
    (nv: number) => {
      if (useLogX) {
        const logN = Math.log(Math.max(1, nv));
        const logMax = Math.log(n);
        return MARGIN.left + (logN / logMax) * plotW;
      }
      return MARGIN.left + ((nv - 1) / (n - 1)) * plotW;
    },
    [n, plotW, useLogX],
  );

  const toSvgYLeft = useCallback(
    (v: number) => {
      const clamped = Math.max(yMinLeft, Math.min(yMaxLeft, v));
      return (
        MARGIN.top +
        plotH -
        ((clamped - yMinLeft) / (yMaxLeft - yMinLeft)) * plotH
      );
    },
    [yMinLeft, yMaxLeft, plotH],
  );

  // ── Scale helpers (right panel) ─────────────────────────────────────────

  const rightXIsLog = mode === 'fixed-eps' ? n > 200 : true;
  const rightXMin = boundsData.length > 0 ? boundsData[0].x : 1;
  const rightXMax =
    boundsData.length > 0 ? boundsData[boundsData.length - 1].x : n;

  const toSvgXRight = useCallback(
    (xv: number) => {
      if (rightXIsLog) {
        const logX = Math.log(Math.max(Number.EPSILON, xv));
        const logMin = Math.log(Math.max(Number.EPSILON, rightXMin));
        const logMax = Math.log(rightXMax);
        const t = logMax > logMin ? (logX - logMin) / (logMax - logMin) : 0;
        return MARGIN.left + t * plotW;
      }
      const t =
        rightXMax > rightXMin
          ? (xv - rightXMin) / (rightXMax - rightXMin)
          : 0;
      return MARGIN.left + t * plotW;
    },
    [rightXIsLog, rightXMin, rightXMax, plotW],
  );

  const toSvgYRight = useCallback(
    (p: number) => {
      const clamped = Math.max(0, Math.min(1, p));
      return MARGIN.top + plotH - clamped * plotH;
    },
    [plotH],
  );

  // ── D3 line generators ────────────────────────────────────────────────

  const lineLeft = useMemo(
    () =>
      d3
        .line<{ n: number; value: number }>()
        .x((d) => toSvgXLeft(d.n))
        .y((d) => toSvgYLeft(d.value))
        .defined((d) => isFinite(d.value)),
    [toSvgXLeft, toSvgYLeft],
  );

  // ── Tick values ───────────────────────────────────────────────────────

  const xTicksLeft = useMemo(() => {
    if (useLogX) return niceTicksLog(1, n, 6);
    return niceTicksLinear(1, n, 6);
  }, [n, useLogX]);

  const yTicksLeft = useMemo(
    () => niceTicksLinear(yMinLeft, yMaxLeft, 5),
    [yMinLeft, yMaxLeft],
  );

  const yTicksRight = useMemo(() => [0, 0.25, 0.5, 0.75, 1.0], []);

  const xTicksRight = useMemo(() => {
    if (boundsData.length === 0) return [];
    if (mode === 'fixed-eps') {
      return niceTicksLog(rightXMin, rightXMax, 5);
    }
    // Fixed-n mode: epsilon on log scale
    return [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0].filter(
      (v) => v >= rightXMin * 0.9 && v <= rightXMax * 1.1,
    );
  }, [boundsData, mode, rightXMin, rightXMax]);

  // ── Envelope path data for left panel ──────────────────────────────────

  const cltUpperPath = useMemo(() => {
    return envelopes
      .filter((e) => isFinite(e.clt))
      .map((e) => ({ n: e.n, value: e.clt }));
  }, [envelopes]);

  const cltLowerPath = useMemo(() => {
    return envelopes
      .filter((e) => isFinite(e.clt))
      .map((e) => ({ n: e.n, value: -e.clt }));
  }, [envelopes]);

  const lilUpperPath = useMemo(() => {
    return envelopes
      .filter((e) => isFinite(e.lil) && e.n >= 3)
      .map((e) => ({ n: e.n, value: e.lil }));
  }, [envelopes]);

  const lilLowerPath = useMemo(() => {
    return envelopes
      .filter((e) => isFinite(e.lil) && e.n >= 3)
      .map((e) => ({ n: e.n, value: -e.lil }));
  }, [envelopes]);

  // ── Annotation values for bottom bar ───────────────────────────────────

  const annotationValues = useMemo(() => {
    const { sigma, a, b } = params;
    const sigSq = sigma * sigma;
    const isBounded = a !== undefined && b !== undefined;
    return {
      sigma: sigma.toFixed(4),
      chebyshev: formatBound(chebyshevBound(n, sigSq, epsilon)),
      hoeffding: isBounded
        ? formatBound(hoeffdingBound(n, a!, b!, epsilon))
        : 'N/A (unbounded)',
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dist, n, epsilon]);

  // ── Resample handler ──────────────────────────────────────────────────

  const handleResample = useCallback(() => {
    setSeed((s) => s + 1);
  }, []);

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Convergence Rate Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {/* Distribution dropdown */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-bg)',
            }}
            value={dist}
            onChange={(e) => setDist(e.target.value)}
          >
            {distKeys.map((k) => (
              <option key={k} value={k}>
                {distNames[k]}
              </option>
            ))}
          </select>
        </label>

        {/* n slider (log scale) */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={valueToLogSlider(n, 10, 10000)}
            onChange={(e) =>
              setN(logSliderToValue(Number(e.target.value), 10, 10000))
            }
            className="w-24"
          />
          <span className="w-14 tabular-nums text-right">
            {n.toLocaleString()}
          </span>
        </label>

        {/* Epsilon slider (log scale) */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">&epsilon;</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={valueToEpsSlider(epsilon)}
            onChange={(e) =>
              setEpsilon(
                Math.round(epsSliderToValue(Number(e.target.value)) * 1000) /
                  1000,
              )
            }
            className="w-20"
          />
          <span className="w-10 tabular-nums text-right">
            {epsilon.toFixed(epsilon < 0.1 ? 3 : 2)}
          </span>
        </label>

        {/* Paths slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Paths</span>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={numPaths}
            onChange={(e) => setNumPaths(Number(e.target.value))}
            className="w-16"
          />
          <span className="w-6 tabular-nums text-right">{numPaths}</span>
        </label>

        {/* Mode toggle */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Right panel</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{
              borderColor: 'var(--color-border)',
              background: 'var(--color-bg)',
            }}
            value={mode}
            onChange={(e) =>
              setMode(e.target.value as 'fixed-eps' | 'fixed-n')
            }
          >
            <option value="fixed-eps">Fixed &epsilon; (vary n)</option>
            <option value="fixed-n">Fixed n (vary &epsilon;)</option>
          </select>
        </label>

        {/* Resample button */}
        <button
          className="rounded border px-3 py-1 text-xs font-medium transition-colors hover:bg-gray-100 dark:hover:bg-gray-800"
          style={{ borderColor: 'var(--color-border)' }}
          onClick={handleResample}
        >
          Resample
        </button>
      </div>

      {/* Dual-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* ── Left Panel: Sample Paths with Envelopes ─────────────────────── */}
        <div
          className={isNarrow ? 'w-full' : ''}
          style={isNarrow ? {} : { width: panelW }}
        >
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Sample paths of X&#x0304;&#x2099; &minus; &mu;
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
            {xTicksLeft.map((v, i) => (
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

            {/* Epsilon band (fixed tolerance) */}
            {(() => {
              const epsTop = toSvgYLeft(epsilon);
              const epsBot = toSvgYLeft(-epsilon);
              const h = Math.max(0, epsBot - epsTop);
              return (
                <rect
                  x={MARGIN.left}
                  y={epsTop}
                  width={plotW}
                  height={h}
                  fill={EPSILON_BAND_COLOR}
                  opacity={0.3}
                />
              );
            })()}

            {/* Epsilon band edges (dotted emerald) */}
            {[epsilon, -epsilon].map((ev, i) => (
              <line
                key={`eps-edge-${i}`}
                x1={MARGIN.left}
                y1={toSvgYLeft(ev)}
                x2={panelW - MARGIN.right}
                y2={toSvgYLeft(ev)}
                stroke={CHEBYSHEV_COLOR}
                strokeWidth={1}
                strokeDasharray="2,4"
                opacity={0.7}
              />
            ))}

            {/* Zero line */}
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(0)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(0)}
              stroke={PATH_COLOR}
              strokeWidth={1}
              strokeDasharray="6,3"
              opacity={0.4}
            />

            {/* CLT envelope: ±σ/√n (dashed blue, lighter) */}
            {(() => {
              const dUpper = lineLeft(cltUpperPath);
              const dLower = lineLeft(cltLowerPath);
              return (
                <>
                  {dUpper && (
                    <path
                      d={dUpper}
                      fill="none"
                      stroke={CLT_COLOR}
                      strokeWidth={1.2}
                      strokeDasharray="6,3"
                      opacity={0.4}
                    />
                  )}
                  {dLower && (
                    <path
                      d={dLower}
                      fill="none"
                      stroke={CLT_COLOR}
                      strokeWidth={1.2}
                      strokeDasharray="6,3"
                      opacity={0.4}
                    />
                  )}
                </>
              );
            })()}

            {/* LIL envelope: ±σ√(2 ln ln n / n) (solid amber, starts at n=3) */}
            {(() => {
              const dUpper = lineLeft(lilUpperPath);
              const dLower = lineLeft(lilLowerPath);
              return (
                <>
                  {dUpper && (
                    <path
                      d={dUpper}
                      fill="none"
                      stroke={HOEFFDING_COLOR}
                      strokeWidth={1.5}
                      opacity={0.7}
                    />
                  )}
                  {dLower && (
                    <path
                      d={dLower}
                      fill="none"
                      stroke={HOEFFDING_COLOR}
                      strokeWidth={1.5}
                      opacity={0.7}
                    />
                  )}
                </>
              );
            })()}

            {/* Sample paths (gray, low opacity) */}
            {downsampledPaths.map((points, i) => {
              const dStr = lineLeft(points);
              if (!dStr) return null;
              return (
                <path
                  key={`path-${i}`}
                  d={dStr}
                  fill="none"
                  stroke={PATH_COLOR}
                  strokeWidth={1}
                  opacity={0.2}
                />
              );
            })}

            {/* Envelope legend labels (right edge) */}
            {(() => {
              const labelX = panelW - MARGIN.right - 2;
              const items: Array<{
                label: string;
                color: string;
                yVal: number;
                dash: boolean;
              }> = [];

              // CLT label
              if (cltUpperPath.length > 0) {
                const lastClt =
                  cltUpperPath[cltUpperPath.length - 1].value;
                items.push({
                  label: '\u00B1\u03C3/\u221An',
                  color: CLT_COLOR,
                  yVal: lastClt,
                  dash: true,
                });
              }

              // LIL label
              if (lilUpperPath.length > 0) {
                const lastLil =
                  lilUpperPath[lilUpperPath.length - 1].value;
                items.push({
                  label: 'LIL',
                  color: HOEFFDING_COLOR,
                  yVal: lastLil,
                  dash: false,
                });
              }

              // Epsilon label
              items.push({
                label: '\u00B1\u03B5',
                color: CHEBYSHEV_COLOR,
                yVal: epsilon,
                dash: true,
              });

              return items.map((item, i) => (
                <text
                  key={`env-label-${i}`}
                  x={labelX}
                  y={toSvgYLeft(item.yVal) - 3}
                  textAnchor="end"
                  fontSize={8}
                  fill={item.color}
                  opacity={0.8}
                >
                  {item.label}
                </text>
              ));
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
            {xTicksLeft.map((v, i) => (
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
              X&#x0304;&#x2099; &minus; &mu;
            </text>
          </svg>
        </div>

        {/* ── Right Panel: Probability Bounds ─────────────────────────────── */}
        <div
          className={isNarrow ? 'w-full' : ''}
          style={isNarrow ? {} : { width: panelW }}
        >
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            {mode === 'fixed-eps'
              ? `Probability bounds (fixed \u03B5 = ${epsilon.toFixed(epsilon < 0.1 ? 3 : 2)})`
              : `Probability bounds (fixed n = ${n.toLocaleString()})`}
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
            {xTicksRight.map((v, i) => (
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

            {/* Chebyshev bound (emerald dashed) */}
            {(() => {
              const pts = boundsData.map((d) => ({
                x: toSvgXRight(d.x),
                y: toSvgYRight(d.chebyshev),
              }));
              const pathStr = pts
                .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`)
                .join('');
              return (
                <path
                  d={pathStr}
                  fill="none"
                  stroke={CHEBYSHEV_COLOR}
                  strokeWidth={2}
                  strokeDasharray="6,3"
                  opacity={0.85}
                />
              );
            })()}

            {/* Hoeffding bound (amber dashed, bounded distributions only) */}
            {(() => {
              const pts = boundsData
                .filter((d) => d.hoeffding !== null)
                .map((d) => ({
                  x: toSvgXRight(d.x),
                  y: toSvgYRight(d.hoeffding!),
                }));
              if (pts.length === 0) return null;
              const pathStr = pts
                .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`)
                .join('');
              return (
                <path
                  d={pathStr}
                  fill="none"
                  stroke={HOEFFDING_COLOR}
                  strokeWidth={2}
                  strokeDasharray="6,3"
                  opacity={0.85}
                />
              );
            })()}

            {/* Empirical estimate (red solid) */}
            {(() => {
              const pts = boundsData.map((d) => ({
                x: toSvgXRight(d.x),
                y: toSvgYRight(d.empirical),
              }));
              const pathStr = pts
                .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`)
                .join('');
              return (
                <path
                  d={pathStr}
                  fill="none"
                  stroke={EMPIRICAL_COLOR}
                  strokeWidth={2}
                  opacity={0.85}
                />
              );
            })()}

            {/* Legend */}
            {(() => {
              const isBounded =
                params.a !== undefined && params.b !== undefined;
              const items: Array<{
                label: string;
                color: string;
                dash: string;
              }> = [
                {
                  label: 'Chebyshev',
                  color: CHEBYSHEV_COLOR,
                  dash: '6,3',
                },
              ];
              if (isBounded) {
                items.push({
                  label: 'Hoeffding',
                  color: HOEFFDING_COLOR,
                  dash: '6,3',
                });
              }
              items.push({
                label: 'Empirical',
                color: EMPIRICAL_COLOR,
                dash: '0',
              });

              return items.map((item, i) => {
                const lx = MARGIN.left + 8;
                const ly = MARGIN.top + 12 + i * 14;
                return (
                  <g key={`legend-${i}`}>
                    <line
                      x1={lx}
                      y1={ly}
                      x2={lx + 18}
                      y2={ly}
                      stroke={item.color}
                      strokeWidth={2}
                      strokeDasharray={
                        item.dash === '0' ? undefined : item.dash
                      }
                      opacity={0.85}
                    />
                    <text
                      x={lx + 22}
                      y={ly + 3}
                      fontSize={8}
                      fill="currentColor"
                      opacity={0.7}
                    >
                      {item.label}
                    </text>
                  </g>
                );
              });
            })()}

            {/* "Unbounded" note for Normal/Exponential */}
            {params.a === undefined && (
              <text
                x={MARGIN.left + plotW / 2}
                y={MARGIN.top + plotH - 8}
                textAnchor="middle"
                fontSize={8}
                fontStyle="italic"
                fill="currentColor"
                opacity={0.45}
              >
                Hoeffding N/A (unbounded distribution)
              </text>
            )}

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
            {xTicksRight.map((v, i) => (
              <text
                key={`xtr-${i}`}
                x={toSvgXRight(v)}
                y={MARGIN.top + plotH + 14}
                textAnchor="middle"
                fontSize={9}
                fill="currentColor"
                opacity={0.6}
              >
                {mode === 'fixed-eps' ? formatTick(v) : v.toFixed(2)}
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
              {mode === 'fixed-eps' ? 'n' : '\u03B5'}
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
              P(|X&#x0304;&#x2099; &minus; &mu;| &gt; &epsilon;)
            </text>
          </svg>
        </div>
      </div>

      {/* Annotation bar */}
      <div
        className="text-center text-xs mt-3"
        style={{ color: 'var(--color-text-muted)' }}
      >
        Distribution: {distNames[dist]} | &sigma; ={' '}
        {annotationValues.sigma} | Chebyshev bound:{' '}
        {annotationValues.chebyshev} | Hoeffding bound:{' '}
        {annotationValues.hoeffding}
      </div>

      {/* Color key for envelopes on left panel */}
      <div className="flex flex-wrap gap-4 justify-center mt-2">
        {ratePresets.map((preset) => (
          <span key={preset.name} className="flex items-center gap-1 text-xs">
            <span
              className="inline-block w-3 h-0.5 rounded"
              style={{ background: preset.color }}
            />
            <span style={{ color: 'var(--color-text-muted)' }}>
              {preset.name}: {preset.formula} ({preset.type})
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
