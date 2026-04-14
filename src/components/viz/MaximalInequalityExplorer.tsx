import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { sampleSequence, normalSample } from './shared/convergence';

// ── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 45 };
const CHART_HEIGHT = 240;
const NUM_HISTOGRAM_REPS = 1000;
const NUM_PROB_REPS = 500;
const HISTOGRAM_BINS = 30;

// ── Colors ───────────────────────────────────────────────────────────────────

const PATH_COLOR = '#6B7280';
const MAX_MARKER_COLOR = '#DC2626';
const SN_MARKER_COLOR = '#059669';
const EPSILON_BAND_FILL = '#FEF3C7';
const HIST_MAX_COLOR = '#2563eb';
const HIST_SN_COLOR = '#059669';
const BOUND_LINE_COLOR = '#111827';

// ── Utility: partial sums ────────────────────────────────────────────────────

function partialSums(n: number): number[] {
  const samples = sampleSequence(() => normalSample(0, 1), n);
  const sums = new Array<number>(n);
  sums[0] = samples[0];
  for (let i = 1; i < n; i++) {
    sums[i] = sums[i - 1] + samples[i];
  }
  return sums;
}

// ── Utility: compute max|S_k| index and value ────────────────────────────────

function maxAbsPartialSum(sums: number[]): { index: number; value: number } {
  let bestIdx = 0;
  let bestVal = Math.abs(sums[0]);
  for (let i = 1; i < sums.length; i++) {
    const absVal = Math.abs(sums[i]);
    if (absVal > bestVal) {
      bestVal = absVal;
      bestIdx = i;
    }
  }
  return { index: bestIdx, value: bestVal };
}

// ── Utility: histogram binning ───────────────────────────────────────────────

interface HistogramBin {
  x0: number;
  x1: number;
  count: number;
}

function buildHistogram(
  values: number[],
  binCount: number,
  domainMin: number,
  domainMax: number,
): HistogramBin[] {
  const bins: HistogramBin[] = [];
  const step = (domainMax - domainMin) / binCount;
  for (let i = 0; i < binCount; i++) {
    bins.push({ x0: domainMin + i * step, x1: domainMin + (i + 1) * step, count: 0 });
  }
  for (const v of values) {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((v - domainMin) / step)));
    bins[idx].count++;
  }
  return bins;
}

// ── Utility: nice tick generation ────────────────────────────────────────────

function niceTicks(min: number, max: number, maxCount: number): number[] {
  const range = max - min;
  if (range <= 0) return [min];
  const niceSteps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100];
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

function formatTick(v: number): string {
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

// ── Component ────────────────────────────────────────────────────────────────

export default function MaximalInequalityExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls
  const [n, setN] = useState(100);
  const [epsilon, setEpsilon] = useState(10);
  const [seed, setSeed] = useState(0);

  // ── Responsive layout ──────────────────────────────────────────────────────
  const containerW = Math.max(280, (width || 900) - 16);
  const isNarrow = containerW < 720;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 24) / 3);
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = CHART_HEIGHT - MARGIN.top - MARGIN.bottom;

  // ── Left panel: single partial sum path ────────────────────────────────────

  const pathData = useMemo(() => {
    // seed is a dependency to force re-generation
    void seed;
    const sums = partialSums(n);
    const maxInfo = maxAbsPartialSum(sums);
    return { sums, maxInfo };
  }, [n, seed]);

  const absSn = Math.abs(pathData.sums[n - 1]);

  // Y-domain for left panel
  const leftYDomain = useMemo(() => {
    const vals = pathData.sums;
    let min = Infinity;
    let max = -Infinity;
    for (const v of vals) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    // Ensure epsilon bands are visible
    min = Math.min(min, -epsilon - 2);
    max = Math.max(max, epsilon + 2);
    const pad = (max - min) * 0.05;
    return [min - pad, max + pad] as [number, number];
  }, [pathData.sums, epsilon]);

  // Scale helpers for left panel
  const toSvgXLeft = useCallback(
    (k: number) => MARGIN.left + ((k - 1) / Math.max(1, n - 1)) * plotW,
    [n, plotW],
  );

  const toSvgYLeft = useCallback(
    (v: number) => {
      const [yMin, yMax] = leftYDomain;
      const clamped = Math.max(yMin, Math.min(yMax, v));
      return MARGIN.top + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
    },
    [leftYDomain, plotH],
  );

  // Build path string for left panel
  const leftPathD = useMemo(() => {
    const points: string[] = [];
    for (let k = 0; k < n; k++) {
      const x = toSvgXLeft(k + 1);
      const y = toSvgYLeft(pathData.sums[k]);
      points.push(`${k === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.join(' ');
  }, [pathData.sums, n, toSvgXLeft, toSvgYLeft]);

  // Left panel ticks
  const leftXTicks = useMemo(() => niceTicks(1, n, 5), [n]);
  const leftYTicks = useMemo(() => niceTicks(leftYDomain[0], leftYDomain[1], 5), [leftYDomain]);

  // ── Center panel: histogram data ───────────────────────────────────────────

  const histogramData = useMemo(() => {
    void seed;
    const maxVals: number[] = [];
    const snVals: number[] = [];
    for (let r = 0; r < NUM_HISTOGRAM_REPS; r++) {
      const sums = partialSums(n);
      const maxAbs = maxAbsPartialSum(sums).value;
      maxVals.push(maxAbs);
      snVals.push(Math.abs(sums[n - 1]));
    }
    return { maxVals, snVals };
  }, [n, seed]);

  // Histogram domain and bins
  const histDomain = useMemo(() => {
    const allVals = [...histogramData.maxVals, ...histogramData.snVals];
    const max = Math.max(...allVals);
    return [0, max * 1.05] as [number, number];
  }, [histogramData]);

  const histBinsMax = useMemo(
    () => buildHistogram(histogramData.maxVals, HISTOGRAM_BINS, histDomain[0], histDomain[1]),
    [histogramData.maxVals, histDomain],
  );

  const histBinsSn = useMemo(
    () => buildHistogram(histogramData.snVals, HISTOGRAM_BINS, histDomain[0], histDomain[1]),
    [histogramData.snVals, histDomain],
  );

  const histMaxCount = useMemo(() => {
    let max = 0;
    for (const b of histBinsMax) max = Math.max(max, b.count);
    for (const b of histBinsSn) max = Math.max(max, b.count);
    return max;
  }, [histBinsMax, histBinsSn]);

  // Fractions exceeding epsilon
  const fracMaxExceed = useMemo(
    () => histogramData.maxVals.filter((v) => v > epsilon).length / NUM_HISTOGRAM_REPS,
    [histogramData.maxVals, epsilon],
  );
  const fracSnExceed = useMemo(
    () => histogramData.snVals.filter((v) => v > epsilon).length / NUM_HISTOGRAM_REPS,
    [histogramData.snVals, epsilon],
  );

  // Center panel scale helpers
  const toSvgXCenter = useCallback(
    (v: number) => MARGIN.left + ((v - histDomain[0]) / (histDomain[1] - histDomain[0])) * plotW,
    [histDomain, plotW],
  );

  const toSvgYCenter = useCallback(
    (count: number) => MARGIN.top + plotH - (count / Math.max(1, histMaxCount)) * plotH,
    [histMaxCount, plotH],
  );

  const centerXTicks = useMemo(() => niceTicks(histDomain[0], histDomain[1], 5), [histDomain]);

  // ── Right panel: probability vs n ──────────────────────────────────────────

  const probData = useMemo(() => {
    void seed;
    const nValues: number[] = [];
    const pMax: number[] = [];
    const pSn: number[] = [];
    const bound: number[] = [];

    const step = 10;
    for (let nVal = step; nVal <= n; nVal += step) {
      nValues.push(nVal);

      let countMax = 0;
      let countSn = 0;
      for (let r = 0; r < NUM_PROB_REPS; r++) {
        const sums = partialSums(nVal);
        const maxAbs = maxAbsPartialSum(sums).value;
        const snAbs = Math.abs(sums[nVal - 1]);
        if (maxAbs > epsilon) countMax++;
        if (snAbs > epsilon) countSn++;
      }

      pMax.push(countMax / NUM_PROB_REPS);
      pSn.push(countSn / NUM_PROB_REPS);
      // Kolmogorov bound: Var(S_n) / epsilon^2 = n * 1 / epsilon^2
      // (each X_i ~ N(0,1) so Var(X_i) = 1, Var(S_n) = n)
      bound.push(Math.min(1, nVal / (epsilon * epsilon)));
    }

    return { nValues, pMax, pSn, bound };
  }, [n, epsilon, seed]);

  // Right panel scale helpers
  const toSvgXRight = useCallback(
    (nVal: number) => {
      const nMax = Math.max(10, probData.nValues[probData.nValues.length - 1] ?? n);
      return MARGIN.left + ((nVal - 10) / Math.max(1, nMax - 10)) * plotW;
    },
    [probData.nValues, n, plotW],
  );

  const toSvgYRight = useCallback(
    (p: number) => {
      const clamped = Math.max(0, Math.min(1, p));
      return MARGIN.top + plotH - clamped * plotH;
    },
    [plotH],
  );

  const rightXTicks = useMemo(() => {
    const maxN = probData.nValues[probData.nValues.length - 1] ?? n;
    return niceTicks(10, maxN, 5);
  }, [probData.nValues, n]);

  const rightYTicks = useMemo(() => [0, 0.25, 0.5, 0.75, 1.0], []);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Kolmogorov's Maximal Inequality
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {/* n slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={10}
            max={500}
            step={10}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-24"
          />
          <span className="w-8 tabular-nums text-right">{n}</span>
        </label>

        {/* epsilon slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">&epsilon;</span>
          <input
            type="range"
            min={1}
            max={30}
            step={1}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            className="w-24"
          />
          <span className="w-8 tabular-nums text-right">{epsilon}</span>
        </label>

        {/* New path button */}
        <button
          className="rounded border px-3 py-1 text-xs font-medium hover:opacity-80 transition-opacity"
          style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
          onClick={() => setSeed((s) => s + 1)}
        >
          New path
        </button>
      </div>

      {/* Three-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-3'}>
        {/* ── Left panel: Single Partial Sum Path ──────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Partial sum path S&#x2081;, ..., S&#x2099;
          </div>
          <svg width={panelW} height={CHART_HEIGHT} className="block mx-auto">
            {/* Y grid lines */}
            {leftYTicks.map((v, i) => (
              <line
                key={`lyg-${i}`}
                x1={MARGIN.left}
                y1={toSvgYLeft(v)}
                x2={panelW - MARGIN.right}
                y2={toSvgYLeft(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Epsilon bands */}
            <rect
              x={MARGIN.left}
              y={toSvgYLeft(epsilon)}
              width={plotW}
              height={Math.max(0, toSvgYLeft(-epsilon) - toSvgYLeft(epsilon))}
              fill={EPSILON_BAND_FILL}
              opacity={0.6}
            />
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(epsilon)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(epsilon)}
              stroke="#F59E0B"
              strokeWidth={1}
              strokeDasharray="4,3"
              opacity={0.7}
            />
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(-epsilon)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(-epsilon)}
              stroke="#F59E0B"
              strokeWidth={1}
              strokeDasharray="4,3"
              opacity={0.7}
            />

            {/* Band labels */}
            <text
              x={panelW - MARGIN.right - 2}
              y={toSvgYLeft(epsilon) - 3}
              textAnchor="end"
              fontSize={8}
              fill="#D97706"
              opacity={0.8}
            >
              +&epsilon;
            </text>
            <text
              x={panelW - MARGIN.right - 2}
              y={toSvgYLeft(-epsilon) + 10}
              textAnchor="end"
              fontSize={8}
              fill="#D97706"
              opacity={0.8}
            >
              -&epsilon;
            </text>

            {/* Zero line */}
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(0)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(0)}
              stroke="currentColor"
              strokeWidth={0.5}
              strokeOpacity={0.2}
            />

            {/* Path */}
            <path
              d={leftPathD}
              fill="none"
              stroke={PATH_COLOR}
              strokeWidth={1.2}
              opacity={0.7}
            />

            {/* max|S_k| marker (k*) */}
            <circle
              cx={toSvgXLeft(pathData.maxInfo.index + 1)}
              cy={toSvgYLeft(pathData.sums[pathData.maxInfo.index])}
              r={5}
              fill="none"
              stroke={MAX_MARKER_COLOR}
              strokeWidth={2}
            />

            {/* |S_n| marker */}
            <rect
              x={toSvgXLeft(n) - 4}
              y={toSvgYLeft(pathData.sums[n - 1]) - 4}
              width={8}
              height={8}
              fill={SN_MARKER_COLOR}
              opacity={0.8}
              rx={1}
            />

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
            {leftXTicks.map((v, i) => (
              <text
                key={`lxt-${i}`}
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
              y={CHART_HEIGHT - 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              k
            </text>

            {/* Y-axis ticks */}
            {leftYTicks.map((v, i) => (
              <text
                key={`lyt-${i}`}
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

            {/* Annotations */}
            <text
              x={MARGIN.left + 4}
              y={MARGIN.top + 12}
              fontSize={9}
              fill={MAX_MARKER_COLOR}
              fontWeight="600"
            >
              max|S&#x2096;| = {pathData.maxInfo.value.toFixed(2)}
            </text>
            <text
              x={MARGIN.left + 4}
              y={MARGIN.top + 24}
              fontSize={9}
              fill={SN_MARKER_COLOR}
              fontWeight="600"
            >
              |S&#x2099;| = {absSn.toFixed(2)}
            </text>
          </svg>
        </div>

        {/* ── Center panel: Histogram ──────────────────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Histogram ({NUM_HISTOGRAM_REPS} replications)
          </div>
          <svg width={panelW} height={CHART_HEIGHT} className="block mx-auto">
            {/* max|S_k| histogram bars */}
            {histBinsMax.map((bin, i) => {
              const x0 = toSvgXCenter(bin.x0);
              const x1 = toSvgXCenter(bin.x1);
              const y = toSvgYCenter(bin.count);
              const barH = MARGIN.top + plotH - y;
              return (
                <rect
                  key={`hm-${i}`}
                  x={x0}
                  y={y}
                  width={Math.max(0, x1 - x0 - 0.5)}
                  height={Math.max(0, barH)}
                  fill={HIST_MAX_COLOR}
                  opacity={0.5}
                />
              );
            })}

            {/* |S_n| histogram bars */}
            {histBinsSn.map((bin, i) => {
              const x0 = toSvgXCenter(bin.x0);
              const x1 = toSvgXCenter(bin.x1);
              const y = toSvgYCenter(bin.count);
              const barH = MARGIN.top + plotH - y;
              return (
                <rect
                  key={`hs-${i}`}
                  x={x0}
                  y={y}
                  width={Math.max(0, x1 - x0 - 0.5)}
                  height={Math.max(0, barH)}
                  fill={HIST_SN_COLOR}
                  opacity={0.5}
                />
              );
            })}

            {/* Epsilon vertical line */}
            <line
              x1={toSvgXCenter(epsilon)}
              y1={MARGIN.top}
              x2={toSvgXCenter(epsilon)}
              y2={MARGIN.top + plotH}
              stroke="#D97706"
              strokeWidth={1.5}
              strokeDasharray="4,3"
            />
            <text
              x={toSvgXCenter(epsilon) + 3}
              y={MARGIN.top + 12}
              fontSize={8}
              fill="#D97706"
              fontWeight="600"
            >
              &epsilon; = {epsilon}
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
            {centerXTicks.map((v, i) => (
              <text
                key={`cxt-${i}`}
                x={toSvgXCenter(v)}
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
              y={CHART_HEIGHT - 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              value
            </text>

            {/* Fraction annotations */}
            <text
              x={panelW - MARGIN.right - 2}
              y={MARGIN.top + plotH - 24}
              textAnchor="end"
              fontSize={8}
              fill={HIST_MAX_COLOR}
              fontWeight="600"
            >
              P(max|S&#x2096;| &gt; &epsilon;) &asymp; {fracMaxExceed.toFixed(3)}
            </text>
            <text
              x={panelW - MARGIN.right - 2}
              y={MARGIN.top + plotH - 12}
              textAnchor="end"
              fontSize={8}
              fill={HIST_SN_COLOR}
              fontWeight="600"
            >
              P(|S&#x2099;| &gt; &epsilon;) &asymp; {fracSnExceed.toFixed(3)}
            </text>
          </svg>

          {/* Legend below histogram */}
          <div className="flex justify-center gap-4 mt-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span className="flex items-center gap-1">
              <svg width="12" height="8">
                <rect width="12" height="8" fill={HIST_MAX_COLOR} opacity={0.5} />
              </svg>
              max|S&#x2096;|
            </span>
            <span className="flex items-center gap-1">
              <svg width="12" height="8">
                <rect width="12" height="8" fill={HIST_SN_COLOR} opacity={0.5} />
              </svg>
              |S&#x2099;|
            </span>
          </div>
        </div>

        {/* ── Right panel: Probability vs n ────────────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Probability vs n
          </div>
          <svg width={panelW} height={CHART_HEIGHT} className="block mx-auto">
            {/* Y grid lines */}
            {rightYTicks.map((v, i) => (
              <line
                key={`ryg-${i}`}
                x1={MARGIN.left}
                y1={toSvgYRight(v)}
                x2={panelW - MARGIN.right}
                y2={toSvgYRight(v)}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Kolmogorov/Chebyshev bound (dashed line) */}
            {probData.nValues.length > 1 && (() => {
              const points: string[] = [];
              for (let i = 0; i < probData.nValues.length; i++) {
                const x = toSvgXRight(probData.nValues[i]);
                const y = toSvgYRight(Math.min(1, probData.bound[i]));
                points.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
              }
              return (
                <path
                  d={points.join(' ')}
                  fill="none"
                  stroke={BOUND_LINE_COLOR}
                  strokeWidth={1.5}
                  strokeDasharray="6,3"
                  opacity={0.6}
                />
              );
            })()}

            {/* P(max|S_k| > epsilon) dots */}
            {probData.nValues.map((nVal, i) => (
              <circle
                key={`pm-${i}`}
                cx={toSvgXRight(nVal)}
                cy={toSvgYRight(probData.pMax[i])}
                r={3}
                fill={HIST_MAX_COLOR}
                opacity={0.7}
              />
            ))}

            {/* P(|S_n| > epsilon) dots */}
            {probData.nValues.map((nVal, i) => (
              <circle
                key={`ps-${i}`}
                cx={toSvgXRight(nVal)}
                cy={toSvgYRight(probData.pSn[i])}
                r={3}
                fill={HIST_SN_COLOR}
                opacity={0.7}
              />
            ))}

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
            {rightXTicks.map((v, i) => (
              <text
                key={`rxt-${i}`}
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
              y={CHART_HEIGHT - 2}
              textAnchor="middle"
              fontSize={9}
              fill="currentColor"
              opacity={0.5}
            >
              n
            </text>

            {/* Y-axis ticks */}
            {rightYTicks.map((v, i) => (
              <text
                key={`ryt-${i}`}
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
              probability
            </text>

            {/* Legend inside panel */}
            <circle cx={MARGIN.left + 8} cy={MARGIN.top + 8} r={3} fill={HIST_MAX_COLOR} opacity={0.7} />
            <text x={MARGIN.left + 14} y={MARGIN.top + 11} fontSize={8} fill={HIST_MAX_COLOR}>
              P(max|S&#x2096;| &gt; &epsilon;)
            </text>
            <circle cx={MARGIN.left + 8} cy={MARGIN.top + 20} r={3} fill={HIST_SN_COLOR} opacity={0.7} />
            <text x={MARGIN.left + 14} y={MARGIN.top + 23} fontSize={8} fill={HIST_SN_COLOR}>
              P(|S&#x2099;| &gt; &epsilon;)
            </text>
            <line
              x1={MARGIN.left + 2}
              y1={MARGIN.top + 32}
              x2={MARGIN.left + 14}
              y2={MARGIN.top + 32}
              stroke={BOUND_LINE_COLOR}
              strokeWidth={1.5}
              strokeDasharray="4,2"
              opacity={0.6}
            />
            <text x={MARGIN.left + 18} y={MARGIN.top + 35} fontSize={8} fill={BOUND_LINE_COLOR} opacity={0.6}>
              n / &epsilon;&sup2;
            </text>
          </svg>
        </div>
      </div>

      {/* Summary caption */}
      <div
        className="text-center text-xs mt-3"
        style={{ color: 'var(--color-text-muted)' }}
      >
        Kolmogorov's maximal inequality bounds P(max&#x2096;&#x2264;&#x2099; |S&#x2096;| &gt; &epsilon;) &le; Var(S&#x2099;)/&epsilon;&sup2; &mdash;
        the same Chebyshev bound that controls |S&#x2099;| alone also controls the running maximum.
      </div>
    </div>
  );
}
