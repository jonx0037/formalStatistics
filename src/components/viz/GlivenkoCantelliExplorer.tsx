import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  sampleSequence,
  normalSample,
  exponentialSample,
  uniformSample,
  empiricalCDFPoints,
  runningKSStatistic,
  dkwBound,
} from './shared/convergence';
import { cdfStdNormal, cdfExponential } from './shared/distributions';

// ── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 10, right: 15, bottom: 35, left: 50 };
const CHART_HEIGHT = 260;
const MAX_N = 2000;

// Colors
const TRUE_CDF_COLOR = '#111827';
const ECDF_COLOR = '#2563eb';
const MAX_DEV_COLOR = '#DC2626';
const DKW_COLOR = '#059669';
const REF_COLOR = '#9CA3AF';
const FAMILY_OPACITY = 0.08;

// Distribution configs
type DistKey = 'normal' | 'exponential' | 'uniform';

interface DistConfig {
  label: string;
  xRange: [number, number];
}

const DIST_CONFIGS: Record<DistKey, DistConfig> = {
  normal: { label: 'Normal(0, 1)', xRange: [-4, 4] },
  exponential: { label: 'Exponential(1)', xRange: [0, 6] },
  uniform: { label: 'Uniform(0, 1)', xRange: [-0.5, 1.5] },
};

// ── Sampling & CDF Helpers ──────────────────────────────────────────────────

function getSampler(dist: DistKey): () => number {
  switch (dist) {
    case 'normal':
      return () => normalSample(0, 1);
    case 'exponential':
      return () => exponentialSample(1);
    case 'uniform':
      return () => uniformSample(0, 1);
  }
}

function getCDF(dist: DistKey): (x: number) => number {
  switch (dist) {
    case 'normal':
      return cdfStdNormal;
    case 'exponential':
      return (x: number) => cdfExponential(x, 1);
    case 'uniform':
      return (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
  }
}

// ── ECDF Step Function Path ─────────────────────────────────────────────────

function buildStepPath(
  points: Array<{ x: number; Fn: number }>,
  toSvgX: (x: number) => number,
  toSvgY: (y: number) => number,
): string {
  if (points.length === 0) return '';
  const parts: string[] = [];
  for (let i = 0; i < points.length; i++) {
    const px = toSvgX(points[i].x);
    const py = toSvgY(points[i].Fn);
    if (i === 0) {
      parts.push(`M${px.toFixed(1)},${py.toFixed(1)}`);
    } else {
      const prevY = toSvgY(points[i - 1].Fn);
      parts.push(`L${px.toFixed(1)},${prevY.toFixed(1)}`);
      parts.push(`L${px.toFixed(1)},${py.toFixed(1)}`);
    }
  }
  return parts.join(' ');
}

// ── Supremum Point Finder ───────────────────────────────────────────────────

/**
 * Find the x-value and |F_n(x) - F(x)| at the supremum of the KS statistic.
 * Returns { x, fnVal, fxVal, diff } for annotation.
 */
function findSupremumPoint(
  samples: number[],
  cdf: (x: number) => number,
): { x: number; fnVal: number; fxVal: number; diff: number } {
  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  let maxDiff = 0;
  let bestX = 0;
  let bestFn = 0;
  let bestFx = 0;

  for (let i = 0; i < n; i++) {
    const Fx = cdf(sorted[i]);
    const ecdfRight = (i + 1) / n;
    const ecdfLeft = i / n;

    const diffRight = Math.abs(ecdfRight - Fx);
    const diffLeft = Math.abs(ecdfLeft - Fx);

    if (diffRight > maxDiff) {
      maxDiff = diffRight;
      bestX = sorted[i];
      bestFn = ecdfRight;
      bestFx = Fx;
    }
    if (diffLeft > maxDiff) {
      maxDiff = diffLeft;
      bestX = sorted[i];
      bestFn = ecdfLeft;
      bestFx = Fx;
    }
  }

  return { x: bestX, fnVal: bestFn, fxVal: bestFx, diff: maxDiff };
}

// ── Tick Helpers ────────────────────────────────────────────────────────────

function niceTicksLinear(min: number, max: number, maxCount: number): number[] {
  const range = max - min;
  if (range <= 0) return [min];
  const niceSteps = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
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
  if (v >= 1000) return `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)}k`;
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) < 0.01) return v.toExponential(0);
  return v.toFixed(2);
}

// ── Component ───────────────────────────────────────────────────────────────

export default function GlivenkoCantelliExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // ── State ──────────────────────────────────────────────────────────────
  const [distribution, setDistribution] = useState<DistKey>('normal');
  const [n, setN] = useState(50);
  const [alpha, setAlpha] = useState(0.05);
  const [showFamily, setShowFamily] = useState(false);
  const [animating, setAnimating] = useState(false);
  // Seed counter to force re-sampling when distribution changes
  const [seed, setSeed] = useState(0);

  const animRef = useRef<number | null>(null);

  // ── Generate all 2000 samples upfront ──────────────────────────────────
  const allSamples = useMemo(() => {
    const sampler = getSampler(distribution);
    return sampleSequence(sampler, MAX_N);
    // seed is included to allow re-generation on distribution change
  }, [distribution, seed]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Running KS stats for right panel ───────────────────────────────────
  const ksStats = useMemo(() => {
    const cdf = getCDF(distribution);
    return runningKSStatistic(allSamples, cdf);
  }, [allSamples, distribution]);

  // ── Current ECDF points for left panel ─────────────────────────────────
  const ecdfPoints = useMemo(() => {
    return empiricalCDFPoints(allSamples.slice(0, n));
  }, [allSamples, n]);

  // ── Current KS stat and DKW bound ──────────────────────────────────────
  const currentKS = n > 0 ? ksStats[n - 1] : 0;
  const currentDKW = dkwBound(n, alpha);

  // ── Supremum point for left panel annotation ───────────────────────────
  const supPoint = useMemo(() => {
    const cdf = getCDF(distribution);
    return findSupremumPoint(allSamples.slice(0, n), cdf);
  }, [allSamples, n, distribution]);

  // ── Intermediate ECDFs for "Show CDF family" ───────────────────────────
  const familyECDFs = useMemo(() => {
    if (!showFamily) return [];
    // Show ECDFs at geometrically spaced sample sizes
    const sizes: number[] = [];
    for (let k = 5; k < n; k = Math.floor(k * 1.8)) {
      sizes.push(k);
    }
    return sizes.map((sz) => ({
      n: sz,
      points: empiricalCDFPoints(allSamples.slice(0, sz)),
    }));
  }, [showFamily, allSamples, n]);

  // ── Animation logic ────────────────────────────────────────────────────
  const startAnimation = useCallback(() => {
    setAnimating(true);
    setN(1);
    let currentN = 1;
    const step = () => {
      currentN = Math.min(
        currentN + Math.max(1, Math.floor(currentN * 0.02)),
        MAX_N,
      );
      setN(currentN);
      if (currentN < MAX_N) {
        animRef.current = requestAnimationFrame(step);
      } else {
        setAnimating(false);
      }
    };
    animRef.current = requestAnimationFrame(step);
  }, []);

  const stopAnimation = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setAnimating(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, []);

  // Reset n and re-seed when distribution changes
  const handleDistChange = useCallback(
    (dist: DistKey) => {
      if (animating) stopAnimation();
      setDistribution(dist);
      setN(50);
      setSeed((s) => s + 1);
    },
    [animating, stopAnimation],
  );

  // ── Responsive layout ─────────────────────────────────────────────────
  const containerW = Math.max(280, (width || 700) - 16);
  const isNarrow = containerW < 624;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = CHART_HEIGHT;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Left panel scales (CDF comparison) ─────────────────────────────────
  const xRange = DIST_CONFIGS[distribution].xRange;

  const toSvgXLeft = useCallback(
    (x: number) => {
      return MARGIN.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
    },
    [xRange, plotW],
  );

  const toSvgYLeft = useCallback(
    (y: number) => {
      const clamped = Math.max(0, Math.min(1, y));
      return MARGIN.top + plotH - clamped * plotH;
    },
    [plotH],
  );

  // ── Right panel scales (KS statistic vs n) ────────────────────────────
  // Y domain: 0 to max(1, max KS stat seen so far) with some padding
  const ksYMax = useMemo(() => {
    let maxVal = 0;
    for (let i = 0; i < n; i++) {
      if (ksStats[i] > maxVal) maxVal = ksStats[i];
    }
    return Math.min(1, Math.max(0.3, maxVal * 1.3));
  }, [ksStats, n]);

  const toSvgXRight = useCallback(
    (nVal: number) => {
      return MARGIN.left + ((nVal - 1) / (MAX_N - 1)) * plotW;
    },
    [plotW],
  );

  const toSvgYRight = useCallback(
    (y: number) => {
      const clamped = Math.max(0, Math.min(ksYMax, y));
      return MARGIN.top + plotH - (clamped / ksYMax) * plotH;
    },
    [plotH, ksYMax],
  );

  // ── True CDF path for left panel ───────────────────────────────────────
  const trueCDFPath = useMemo(() => {
    const cdf = getCDF(distribution);
    const numPoints = 200;
    const step = (xRange[1] - xRange[0]) / numPoints;
    const parts: string[] = [];
    for (let i = 0; i <= numPoints; i++) {
      const x = xRange[0] + i * step;
      const y = cdf(x);
      const px = toSvgXLeft(x);
      const py = toSvgYLeft(y);
      parts.push(i === 0 ? `M${px.toFixed(1)},${py.toFixed(1)}` : `L${px.toFixed(1)},${py.toFixed(1)}`);
    }
    return parts.join(' ');
  }, [distribution, xRange, toSvgXLeft, toSvgYLeft]);

  // ── Family ECDF paths ──────────────────────────────────────────────────
  const familyPaths = useMemo(() => {
    return familyECDFs.map((f) => ({
      n: f.n,
      path: buildStepPath(f.points, toSvgXLeft, toSvgYLeft),
    }));
  }, [familyECDFs, toSvgXLeft, toSvgYLeft]);

  // ── Right panel: KS statistic path ─────────────────────────────────────
  const ksPath = useMemo(() => {
    // Downsample for performance when n is large
    const maxRenderPoints = 400;
    const step = Math.max(1, Math.floor(n / maxRenderPoints));
    const parts: string[] = [];
    for (let i = 0; i < n; i += step) {
      const px = toSvgXRight(i + 1);
      const py = toSvgYRight(ksStats[i]);
      parts.push(i === 0 ? `M${px.toFixed(1)},${py.toFixed(1)}` : `L${px.toFixed(1)},${py.toFixed(1)}`);
    }
    // Always include the last point
    if (n > 0) {
      const px = toSvgXRight(n);
      const py = toSvgYRight(ksStats[n - 1]);
      parts.push(`L${px.toFixed(1)},${py.toFixed(1)}`);
    }
    return parts.join(' ');
  }, [n, ksStats, toSvgXRight, toSvgYRight]);

  // ── Right panel: DKW bound curve ───────────────────────────────────────
  const dkwPath = useMemo(() => {
    const numPoints = 200;
    const parts: string[] = [];
    for (let i = 0; i < numPoints; i++) {
      const nVal = 1 + (i / (numPoints - 1)) * (MAX_N - 1);
      const bound = dkwBound(Math.max(1, Math.round(nVal)), alpha);
      const px = toSvgXRight(nVal);
      const py = toSvgYRight(bound);
      parts.push(i === 0 ? `M${px.toFixed(1)},${py.toFixed(1)}` : `L${px.toFixed(1)},${py.toFixed(1)}`);
    }
    return parts.join(' ');
  }, [alpha, toSvgXRight, toSvgYRight]);

  // ── Right panel: 1/sqrt(n) reference curve ─────────────────────────────
  const refPath = useMemo(() => {
    const numPoints = 200;
    const parts: string[] = [];
    for (let i = 0; i < numPoints; i++) {
      const nVal = 1 + (i / (numPoints - 1)) * (MAX_N - 1);
      const val = 1 / Math.sqrt(nVal);
      const px = toSvgXRight(nVal);
      const py = toSvgYRight(val);
      parts.push(i === 0 ? `M${px.toFixed(1)},${py.toFixed(1)}` : `L${px.toFixed(1)},${py.toFixed(1)}`);
    }
    return parts.join(' ');
  }, [toSvgXRight, toSvgYRight]);

  // ── Tick values ────────────────────────────────────────────────────────
  const xTicksLeft = useMemo(
    () => niceTicksLinear(xRange[0], xRange[1], 6),
    [xRange],
  );
  const yTicksLeft = useMemo(() => [0, 0.25, 0.5, 0.75, 1.0], []);

  const xTicksRight = useMemo(() => {
    const ticks: number[] = [];
    const step = MAX_N <= 500 ? 100 : 500;
    for (let v = 0; v <= MAX_N; v += step) {
      if (v > 0) ticks.push(v);
    }
    return ticks;
  }, []);
  const yTicksRight = useMemo(
    () => niceTicksLinear(0, ksYMax, 5),
    [ksYMax],
  );

  // ── Supremum line coordinates for left panel ───────────────────────────
  const supX = toSvgXLeft(supPoint.x);
  const supY1 = toSvgYLeft(Math.min(supPoint.fnVal, supPoint.fxVal));
  const supY2 = toSvgYLeft(Math.max(supPoint.fnVal, supPoint.fxVal));

  // ── Extend ECDF path to right edge of plot ─────────────────────────────
  const ecdfExtendedPath = useMemo(() => {
    if (ecdfPoints.length === 0) return '';
    // Build the step path and extend the last value to the right edge of the x-range
    const mainPath = buildStepPath(ecdfPoints, toSvgXLeft, toSvgYLeft);
    const lastFn = ecdfPoints[ecdfPoints.length - 1].Fn;
    const rightEdgeX = toSvgXLeft(xRange[1]);
    const lastY = toSvgYLeft(lastFn);
    return `${mainPath} L${rightEdgeX.toFixed(1)},${lastY.toFixed(1)}`;
  }, [ecdfPoints, toSvgXLeft, toSvgYLeft, xRange]);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Glivenko--Cantelli Theorem Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {/* Distribution dropdown */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Distribution</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={distribution}
            onChange={(e) => handleDistChange(e.target.value as DistKey)}
          >
            {(Object.keys(DIST_CONFIGS) as DistKey[]).map((key) => (
              <option key={key} value={key}>
                {DIST_CONFIGS[key].label}
              </option>
            ))}
          </select>
        </label>

        {/* n slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">n</span>
          <input
            type="range"
            min={1}
            max={MAX_N}
            step={1}
            value={n}
            onChange={(e) => {
              if (!animating) setN(Number(e.target.value));
            }}
            className="w-28"
            disabled={animating}
          />
          <span className="w-12 tabular-nums text-right">{n}</span>
        </label>

        {/* Alpha slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">&alpha;</span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.01}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-20"
          />
          <span className="w-10 tabular-nums text-right">{alpha.toFixed(2)}</span>
        </label>

        {/* Animate / Stop button */}
        {animating ? (
          <button
            onClick={stopAnimation}
            className="rounded border px-3 py-1 text-xs font-medium"
            style={{
              borderColor: MAX_DEV_COLOR,
              color: MAX_DEV_COLOR,
              background: 'var(--color-bg)',
            }}
          >
            Stop
          </button>
        ) : (
          <button
            onClick={startAnimation}
            className="rounded border px-3 py-1 text-xs font-medium"
            style={{
              borderColor: ECDF_COLOR,
              color: ECDF_COLOR,
              background: 'var(--color-bg)',
            }}
          >
            Animate
          </button>
        )}

        {/* Show CDF family toggle */}
        <label className="flex items-center gap-2 text-xs cursor-pointer">
          <input
            type="checkbox"
            checked={showFamily}
            onChange={(e) => setShowFamily(e.target.checked)}
            className="rounded"
          />
          <span className="font-medium">Show CDF family</span>
        </label>
      </div>

      {/* Dual-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* ── Left Panel: CDF Comparison ─────────────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            CDF Comparison: F(x) vs F&#x2099;(x)
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

            {/* Family ECDFs (translucent, behind main curves) */}
            {familyPaths.map((fp, i) => (
              <path
                key={`family-${i}`}
                d={fp.path}
                fill="none"
                stroke={ECDF_COLOR}
                strokeWidth={1}
                opacity={FAMILY_OPACITY}
              />
            ))}

            {/* True CDF F(x) — solid black */}
            <path
              d={trueCDFPath}
              fill="none"
              stroke={TRUE_CDF_COLOR}
              strokeWidth={2}
            />

            {/* Empirical CDF F_n(x) — colored step function */}
            <path
              d={ecdfExtendedPath}
              fill="none"
              stroke={ECDF_COLOR}
              strokeWidth={1.5}
            />

            {/* Supremum vertical line segment */}
            {supPoint.diff > 0.001 && (
              <>
                <line
                  x1={supX}
                  y1={supY1}
                  x2={supX}
                  y2={supY2}
                  stroke={MAX_DEV_COLOR}
                  strokeWidth={2}
                  strokeDasharray="3,2"
                />
                {/* Small circles at endpoints */}
                <circle cx={supX} cy={supY1} r={2.5} fill={MAX_DEV_COLOR} />
                <circle cx={supX} cy={supY2} r={2.5} fill={MAX_DEV_COLOR} />
              </>
            )}

            {/* X-axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={panelW - MARGIN.right}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            {/* X-axis ticks and labels */}
            {xTicksLeft.map((v, i) => (
              <g key={`xtl-${i}`}>
                <line
                  x1={toSvgXLeft(v)}
                  y1={MARGIN.top + plotH}
                  x2={toSvgXLeft(v)}
                  y2={MARGIN.top + plotH + 4}
                  stroke="currentColor"
                  strokeOpacity={0.4}
                />
                <text
                  x={toSvgXLeft(v)}
                  y={MARGIN.top + plotH + 15}
                  textAnchor="middle"
                  fontSize={9}
                  fill="currentColor"
                  opacity={0.6}
                >
                  {formatTick(v)}
                </text>
              </g>
            ))}
            {/* X-axis label */}
            <text
              x={MARGIN.left + plotW / 2}
              y={MARGIN.top + plotH + 30}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
            >
              x
            </text>

            {/* Y-axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            {/* Y-axis ticks and labels */}
            {yTicksLeft.map((v, i) => (
              <g key={`ytl-${i}`}>
                <line
                  x1={MARGIN.left - 4}
                  y1={toSvgYLeft(v)}
                  x2={MARGIN.left}
                  y2={toSvgYLeft(v)}
                  stroke="currentColor"
                  strokeOpacity={0.4}
                />
                <text
                  x={MARGIN.left - 7}
                  y={toSvgYLeft(v) + 3}
                  textAnchor="end"
                  fontSize={9}
                  fill="currentColor"
                  opacity={0.6}
                >
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Legend */}
            <line
              x1={MARGIN.left + 8}
              y1={MARGIN.top + 10}
              x2={MARGIN.left + 24}
              y2={MARGIN.top + 10}
              stroke={TRUE_CDF_COLOR}
              strokeWidth={2}
            />
            <text
              x={MARGIN.left + 28}
              y={MARGIN.top + 14}
              fontSize={9}
              fill="currentColor"
              opacity={0.7}
            >
              F(x)
            </text>
            <line
              x1={MARGIN.left + 8}
              y1={MARGIN.top + 24}
              x2={MARGIN.left + 24}
              y2={MARGIN.top + 24}
              stroke={ECDF_COLOR}
              strokeWidth={1.5}
            />
            <text
              x={MARGIN.left + 28}
              y={MARGIN.top + 28}
              fontSize={9}
              fill="currentColor"
              opacity={0.7}
            >
              F&#x2099;(x)
            </text>
            {supPoint.diff > 0.001 && (
              <>
                <line
                  x1={MARGIN.left + 8}
                  y1={MARGIN.top + 38}
                  x2={MARGIN.left + 24}
                  y2={MARGIN.top + 38}
                  stroke={MAX_DEV_COLOR}
                  strokeWidth={2}
                  strokeDasharray="3,2"
                />
                <text
                  x={MARGIN.left + 28}
                  y={MARGIN.top + 42}
                  fontSize={9}
                  fill="currentColor"
                  opacity={0.7}
                >
                  sup |F&#x2099; - F|
                </text>
              </>
            )}
          </svg>
        </div>

        {/* ── Right Panel: KS Statistic Tracking ─────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            KS Statistic: D&#x2099; = sup|F&#x2099;(x) - F(x)|
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

            {/* 1/sqrt(n) reference curve — dotted gray */}
            <path
              d={refPath}
              fill="none"
              stroke={REF_COLOR}
              strokeWidth={1}
              strokeDasharray="2,3"
            />

            {/* DKW bound curve — dashed green */}
            <path
              d={dkwPath}
              fill="none"
              stroke={DKW_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,3"
            />

            {/* KS statistic path — solid blue */}
            <path
              d={ksPath}
              fill="none"
              stroke={ECDF_COLOR}
              strokeWidth={1.5}
            />

            {/* Current D_n point */}
            {n > 0 && (
              <circle
                cx={toSvgXRight(n)}
                cy={toSvgYRight(currentKS)}
                r={3}
                fill={ECDF_COLOR}
              />
            )}

            {/* Current D_n annotation */}
            {n > 0 && (
              <text
                x={Math.min(toSvgXRight(n) + 6, panelW - MARGIN.right - 50)}
                y={Math.max(toSvgYRight(currentKS) - 6, MARGIN.top + 12)}
                fontSize={9}
                fill={ECDF_COLOR}
                fontWeight="600"
              >
                D&#x2099; = {currentKS.toFixed(4)}
              </text>
            )}

            {/* X-axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top + plotH}
              x2={panelW - MARGIN.right}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            {/* X-axis ticks and labels */}
            {xTicksRight.map((v, i) => (
              <g key={`xtr-${i}`}>
                <line
                  x1={toSvgXRight(v)}
                  y1={MARGIN.top + plotH}
                  x2={toSvgXRight(v)}
                  y2={MARGIN.top + plotH + 4}
                  stroke="currentColor"
                  strokeOpacity={0.4}
                />
                <text
                  x={toSvgXRight(v)}
                  y={MARGIN.top + plotH + 15}
                  textAnchor="middle"
                  fontSize={9}
                  fill="currentColor"
                  opacity={0.6}
                >
                  {formatTick(v)}
                </text>
              </g>
            ))}
            {/* X-axis label */}
            <text
              x={MARGIN.left + plotW / 2}
              y={MARGIN.top + plotH + 30}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              opacity={0.5}
            >
              n
            </text>

            {/* Y-axis */}
            <line
              x1={MARGIN.left}
              y1={MARGIN.top}
              x2={MARGIN.left}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              strokeOpacity={0.3}
            />
            {/* Y-axis ticks and labels */}
            {yTicksRight.map((v, i) => (
              <g key={`ytr-${i}`}>
                <line
                  x1={MARGIN.left - 4}
                  y1={toSvgYRight(v)}
                  x2={MARGIN.left}
                  y2={toSvgYRight(v)}
                  stroke="currentColor"
                  strokeOpacity={0.4}
                />
                <text
                  x={MARGIN.left - 7}
                  y={toSvgYRight(v) + 3}
                  textAnchor="end"
                  fontSize={9}
                  fill="currentColor"
                  opacity={0.6}
                >
                  {v.toFixed(2)}
                </text>
              </g>
            ))}

            {/* Legend */}
            <line
              x1={panelW - MARGIN.right - 100}
              y1={MARGIN.top + 10}
              x2={panelW - MARGIN.right - 84}
              y2={MARGIN.top + 10}
              stroke={ECDF_COLOR}
              strokeWidth={1.5}
            />
            <text
              x={panelW - MARGIN.right - 80}
              y={MARGIN.top + 14}
              fontSize={9}
              fill="currentColor"
              opacity={0.7}
            >
              D&#x2099;
            </text>

            <line
              x1={panelW - MARGIN.right - 100}
              y1={MARGIN.top + 24}
              x2={panelW - MARGIN.right - 84}
              y2={MARGIN.top + 24}
              stroke={DKW_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,3"
            />
            <text
              x={panelW - MARGIN.right - 80}
              y={MARGIN.top + 28}
              fontSize={9}
              fill="currentColor"
              opacity={0.7}
            >
              DKW bound
            </text>

            <line
              x1={panelW - MARGIN.right - 100}
              y1={MARGIN.top + 38}
              x2={panelW - MARGIN.right - 84}
              y2={MARGIN.top + 38}
              stroke={REF_COLOR}
              strokeWidth={1}
              strokeDasharray="2,3"
            />
            <text
              x={panelW - MARGIN.right - 80}
              y={MARGIN.top + 42}
              fontSize={9}
              fill="currentColor"
              opacity={0.7}
            >
              {'1/\u221An'}
            </text>
          </svg>
        </div>
      </div>

      {/* Readout below charts */}
      <div
        className="text-center text-xs font-mono mt-3"
        style={{ color: 'var(--color-text-muted)' }}
      >
        n = {n} &nbsp;|&nbsp; D&#x2099; = {currentKS.toFixed(4)} &nbsp;|&nbsp; DKW bound = {currentDKW.toFixed(4)}
      </div>
    </div>
  );
}
