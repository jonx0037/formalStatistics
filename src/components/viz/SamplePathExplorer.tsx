import { useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { runningMean, sampleSequence, typewriterSequence, normalSample } from './shared/convergence';
import { samplePathPresets } from '../../data/modes-of-convergence-data';
import type { SamplePathPreset } from '../../data/modes-of-convergence-data';

// ── Constants ────────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 20, bottom: 30, left: 45 };
const MAX_RENDER_POINTS = 500;

const PATH_COLORS = d3.schemeTableau10;
const BAND_COLOR = '#3B82F6';     // blue-500
const TARGET_COLOR = '#6B7280';   // gray-500
const DEVIATION_COLOR = '#DC2626'; // red-600

// ── Path Generation ──────────────────────────────────────────────────────────

interface PathData {
  /** Full-resolution values for each term index 0..N-1 */
  values: number[];
}

function generatePaths(preset: SamplePathPreset, numPaths: number, N: number): PathData[] {
  const paths: PathData[] = [];

  for (let p = 0; p < numPaths; p++) {
    let values: number[];

    switch (preset.id) {
      case 'sample-mean': {
        // Generate N iid N(0,1) samples, then compute running mean
        const samples = sampleSequence(() => normalSample(0, 1), N);
        values = runningMean(samples);
        break;
      }
      case 'z-over-n': {
        // Generate one Z ~ N(0,1), path[i] = Z / (i+1)
        const z = normalSample(0, 1);
        values = new Array<number>(N);
        for (let i = 0; i < N; i++) {
          values[i] = z / (i + 1);
        }
        break;
      }
      case 'typewriter': {
        // Generate one U ~ Uniform(0,1), call typewriterSequence
        const u = Math.random();
        values = typewriterSequence(u, N);
        break;
      }
      case 'escape': {
        // X_n = (n+1) * 1(U < 1/(n+1)), one U per path
        const u = Math.random();
        values = new Array<number>(N);
        for (let i = 0; i < N; i++) {
          values[i] = u < 1 / (i + 1) ? (i + 1) : 0;
        }
        break;
      }
      default: {
        values = new Array<number>(N).fill(0);
      }
    }

    paths.push({ values });
  }

  return paths;
}

// ── Deviation Proportion ─────────────────────────────────────────────────────

/**
 * At each term index n, compute the fraction of paths whose value
 * falls outside the ε-band around the target.
 */
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
      if (Math.abs(paths[p].values[i] - target) > epsilon) {
        outside++;
      }
    }
    proportions[i] = outside / numPaths;
  }

  return proportions;
}

// ── Downsampling ─────────────────────────────────────────────────────────────

function downsample(values: number[], N: number): Array<{ n: number; value: number }> {
  const step = Math.max(1, Math.floor(N / MAX_RENDER_POINTS));
  const points: Array<{ n: number; value: number }> = [];
  for (let i = 0; i < N; i += step) {
    points.push({ n: i + 1, value: values[i] });
  }
  // Always include the last point
  if (points.length === 0 || points[points.length - 1].n !== N) {
    points.push({ n: N, value: values[N - 1] });
  }
  return points;
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

// ── Y-axis Domain Computation ────────────────────────────────────────────────

function computeYDomain(
  paths: PathData[],
  preset: SamplePathPreset,
): [number, number] {
  if (preset.id === 'typewriter') {
    return [-0.15, 1.3];
  }

  if (preset.id === 'escape') {
    // Clip to a sensible maximum to keep the chart readable
    return [-1, 20];
  }

  // For sample-mean and z-over-n, compute from actual data
  let min = Infinity;
  let max = -Infinity;
  for (const path of paths) {
    for (const v of path.values) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const pad = Math.max((max - min) * 0.1, 0.05);
  return [min - pad, max + pad];
}

// ── Tick Helpers ─────────────────────────────────────────────────────────────

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
  // For log-scaled x-axis (term index n), produce ticks at powers of 10 and nice intermediates
  const ticks: number[] = [];
  const logMin = Math.log10(Math.max(1, min));
  const logMax = Math.log10(max);
  const decades = logMax - logMin;

  if (decades <= 2) {
    // Few decades: use linear-style ticks
    return niceTicksLinear(min, max, maxCount);
  }

  // Powers of 10
  for (let exp = Math.floor(logMin); exp <= Math.ceil(logMax); exp++) {
    const val = Math.pow(10, exp);
    if (val >= min && val <= max) ticks.push(val);
    // Add 2x and 5x intermediates if room
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

// ── Component ────────────────────────────────────────────────────────────────

export default function SamplePathExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();

  // Controls state
  const [presetIndex, setPresetIndex] = useState(0);
  const [seqLength, setSeqLength] = useState(500);
  const [numPaths, setNumPaths] = useState(20);
  const [epsilon, setEpsilon] = useState(samplePathPresets[0].defaultEpsilon);

  const preset = samplePathPresets[presetIndex];
  const N = seqLength;

  // Reset epsilon when preset changes
  const handlePresetChange = useCallback((index: number) => {
    setPresetIndex(index);
    setEpsilon(samplePathPresets[index].defaultEpsilon);
  }, []);

  // ── Generate all path data (memoized on preset, numPaths, N) ────────────
  const paths = useMemo(
    () => generatePaths(preset, numPaths, N),
    [preset, numPaths, N], // eslint-disable-line react-hooks/exhaustive-deps
  );

  // ── Deviation proportions (depends on epsilon too) ──────────────────────
  const deviationProportions = useMemo(
    () => computeDeviationProportion(paths, preset.target, epsilon, N),
    [paths, preset.target, epsilon, N],
  );

  // ── Downsampled path data for rendering ─────────────────────────────────
  const downsampledPaths = useMemo(
    () => paths.map((p) => downsample(p.values, N)),
    [paths, N],
  );

  const downsampledDeviation = useMemo(
    () => downsample(deviationProportions, N),
    [deviationProportions, N],
  );

  // ── Y-domain for left panel ─────────────────────────────────────────────
  const [yMin, yMax] = useMemo(() => computeYDomain(paths, preset), [paths, preset]);

  // ── Responsive layout ───────────────────────────────────────────────────
  const containerW = Math.max(280, (width || 700) - 16);
  const isNarrow = containerW < 624;
  const panelW = isNarrow ? containerW : Math.floor((containerW - 16) / 2);
  const chartH = isNarrow ? 220 : 260;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Scale helpers (left panel: sample paths) ────────────────────────────
  // Use log scale for x when N > 200 (helps see early convergence behavior)
  const useLogX = N > 200;

  const toSvgXLeft = useCallback(
    (n: number) => {
      if (useLogX) {
        const logN = Math.log(Math.max(1, n));
        const logMax = Math.log(N);
        return MARGIN.left + (logN / logMax) * plotW;
      }
      return MARGIN.left + ((n - 1) / (N - 1)) * plotW;
    },
    [N, plotW, useLogX],
  );

  const toSvgYLeft = useCallback(
    (v: number) => {
      const clamped = Math.max(yMin, Math.min(yMax, v));
      return MARGIN.top + plotH - ((clamped - yMin) / (yMax - yMin)) * plotH;
    },
    [yMin, yMax, plotH],
  );

  // ── Scale helpers (right panel: deviation proportion) ───────────────────
  const toSvgXRight = toSvgXLeft; // Same x-scale

  const toSvgYRight = useCallback(
    (p: number) => {
      const clamped = Math.max(0, Math.min(1, p));
      return MARGIN.top + plotH - clamped * plotH;
    },
    [plotH],
  );

  // ── D3 line generators ─────────────────────────────────────────────────
  const lineLeft = useMemo(
    () =>
      d3
        .line<{ n: number; value: number }>()
        .x((d) => toSvgXLeft(d.n))
        .y((d) => toSvgYLeft(d.value))
        .defined((d) => isFinite(d.value)),
    [toSvgXLeft, toSvgYLeft],
  );

  const lineRight = useMemo(
    () =>
      d3
        .line<{ n: number; value: number }>()
        .x((d) => toSvgXRight(d.n))
        .y((d) => toSvgYRight(d.value)),
    [toSvgXRight, toSvgYRight],
  );

  // ── Tick values ─────────────────────────────────────────────────────────
  const xTicks = useMemo(() => {
    if (useLogX) return niceTicksLog(1, N, 6);
    return niceTicksLinear(1, N, 6);
  }, [N, useLogX]);

  const yTicksLeft = useMemo(() => niceTicksLinear(yMin, yMax, 5), [yMin, yMax]);

  const yTicksRight = useMemo(() => [0, 0.25, 0.5, 0.75, 1.0], []);

  // ── Annotation for right panel ──────────────────────────────────────────
  const rightAnnotation = useMemo(() => {
    if (preset.convergenceMode === 'as') {
      return 'All paths eventually enter the band';
    }
    if (preset.id === 'typewriter') {
      return 'Some paths always re-exit the band';
    }
    if (preset.id === 'escape') {
      return 'Proportion \u2192 0, but E[X\u2099] = 1 always';
    }
    return null;
  }, [preset]);

  // ── Epsilon band rectangle coordinates ──────────────────────────────────
  const bandTop = toSvgYLeft(preset.target + epsilon);
  const bandBottom = toSvgYLeft(preset.target - epsilon);
  const bandHeight = Math.max(0, bandBottom - bandTop);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div
      ref={ref}
      className="rounded-lg border p-4 my-6"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Title */}
      <div className="text-center text-sm font-semibold mb-3">
        Interactive: Sample Path Explorer
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 justify-center mb-4">
        {/* Preset dropdown */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Sequence</span>
          <select
            className="rounded border px-2 py-1 text-xs"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            value={presetIndex}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
          >
            {samplePathPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>

        {/* N slider (logarithmic) */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">N</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={valueToLogSlider(seqLength, 10, 10000)}
            onChange={(e) => setSeqLength(logSliderToValue(Number(e.target.value), 10, 10000))}
            className="w-24"
          />
          <span className="w-14 tabular-nums text-right">{seqLength.toLocaleString()}</span>
        </label>

        {/* Paths slider */}
        <label className="flex items-center gap-2 text-xs">
          <span className="font-medium">Paths</span>
          <input
            type="range"
            min={5}
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
            min={0.01}
            max={1}
            step={0.01}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            className="w-20"
          />
          <span className="w-10 tabular-nums text-right">{epsilon.toFixed(2)}</span>
        </label>
      </div>

      {/* Preset description */}
      <div className="text-center text-xs mb-3" style={{ color: 'var(--color-text-muted)' }}>
        {preset.description}
      </div>

      {/* Dual-panel area */}
      <div className={isNarrow ? 'flex flex-col gap-4' : 'flex gap-4'}>
        {/* ── Left panel: Sample Paths ────────────────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Sample paths X&#x2099;
          </div>
          <svg width={panelW} height={chartH} className="block mx-auto">
            {/* Y grid lines */}
            {yTicksLeft.map((v, i) => (
              <line
                key={`yg-${i}`}
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
                key={`xg-${i}`}
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
              opacity={0.1}
            />

            {/* Target line (dashed) */}
            <line
              x1={MARGIN.left}
              y1={toSvgYLeft(preset.target)}
              x2={panelW - MARGIN.right}
              y2={toSvgYLeft(preset.target)}
              stroke={TARGET_COLOR}
              strokeWidth={1.5}
              strokeDasharray="6,3"
              opacity={0.7}
            />

            {/* Epsilon band upper and lower edges */}
            <line
              x1={MARGIN.left}
              y1={bandTop}
              x2={panelW - MARGIN.right}
              y2={bandTop}
              stroke={BAND_COLOR}
              strokeWidth={0.8}
              strokeDasharray="3,3"
              opacity={0.4}
            />
            <line
              x1={MARGIN.left}
              y1={bandTop + bandHeight}
              x2={panelW - MARGIN.right}
              y2={bandTop + bandHeight}
              stroke={BAND_COLOR}
              strokeWidth={0.8}
              strokeDasharray="3,3"
              opacity={0.4}
            />

            {/* Sample paths */}
            {downsampledPaths.map((points, i) => {
              const dStr = lineLeft(points);
              if (!dStr) return null;
              return (
                <path
                  key={`path-${i}`}
                  d={dStr}
                  fill="none"
                  stroke={PATH_COLORS[i % PATH_COLORS.length]}
                  strokeWidth={1.2}
                  opacity={0.4}
                />
              );
            })}

            {/* Band labels: +ε, -ε */}
            <text
              x={panelW - MARGIN.right - 2}
              y={bandTop - 3}
              textAnchor="end"
              fontSize={8}
              fill={BAND_COLOR}
              opacity={0.7}
            >
              +&epsilon;
            </text>
            <text
              x={panelW - MARGIN.right - 2}
              y={bandTop + bandHeight + 10}
              textAnchor="end"
              fontSize={8}
              fill={BAND_COLOR}
              opacity={0.7}
            >
              -&epsilon;
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
                key={`xt-${i}`}
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
                key={`yt-${i}`}
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
              value
            </text>
          </svg>
        </div>

        {/* ── Right panel: Deviation Tracking ────────────────────────────── */}
        <div className={isNarrow ? 'w-full' : ''} style={isNarrow ? {} : { width: panelW }}>
          <div
            className="text-xs font-medium text-center mb-1"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Proportion outside &epsilon;-band
          </div>
          <svg width={panelW} height={chartH} className="block mx-auto">
            {/* Y grid lines */}
            {yTicksRight.map((v, i) => (
              <line
                key={`yg-${i}`}
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
                key={`xg-${i}`}
                x1={toSvgXRight(v)}
                y1={MARGIN.top}
                x2={toSvgXRight(v)}
                y2={MARGIN.top + plotH}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
            ))}

            {/* Deviation proportion line */}
            <path
              d={lineRight(downsampledDeviation) ?? ''}
              fill="none"
              stroke={DEVIATION_COLOR}
              strokeWidth={2}
              opacity={0.85}
            />

            {/* Shaded area under deviation curve */}
            {(() => {
              const areaPath = d3
                .area<{ n: number; value: number }>()
                .x((d) => toSvgXRight(d.n))
                .y0(toSvgYRight(0))
                .y1((d) => toSvgYRight(d.value));
              const aStr = areaPath(downsampledDeviation);
              return aStr ? (
                <path d={aStr} fill={DEVIATION_COLOR} opacity={0.08} />
              ) : null;
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
                key={`xt-${i}`}
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
                key={`yt-${i}`}
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
              P(|X&#x2099; &minus; target| &gt; &epsilon;)
            </text>

            {/* Annotation */}
            {rightAnnotation && (
              <text
                x={MARGIN.left + plotW / 2}
                y={MARGIN.top + 14}
                textAnchor="middle"
                fontSize={9}
                fontStyle="italic"
                fill="currentColor"
                opacity={0.55}
              >
                {rightAnnotation}
              </text>
            )}
          </svg>
        </div>
      </div>

      {/* Convergence mode badge */}
      <div className="flex justify-center mt-3">
        <span
          className="inline-block rounded-full px-3 py-0.5 text-xs font-medium"
          style={{
            background:
              preset.convergenceMode === 'as'
                ? 'rgba(124, 58, 237, 0.12)'
                : 'rgba(217, 119, 6, 0.12)',
            color:
              preset.convergenceMode === 'as' ? '#7C3AED' : '#D97706',
          }}
        >
          {preset.convergenceMode === 'as'
            ? 'Almost sure convergence'
            : 'Convergence in probability only'}
        </span>
      </div>
    </div>
  );
}
