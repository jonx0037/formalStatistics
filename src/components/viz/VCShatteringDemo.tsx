/**
 * VCShatteringDemo — Topic 32 §32.4 component.
 *
 * Makes VC dimension tangible by letting the reader drag a finite point set
 * and seeing (a) whether the current function class shatters it, (b) the
 * shatter coefficient / Sauer-Shelah-Vapnik upper bound, and (c) the
 * growth function $(n+1)^V$ alongside the actual combinatorial sum.
 *
 * Two panels:
 *
 *   Panel 1 (shatter canvas) — the point set drawn on an axis appropriate
 *     for the selected class: a 1D number-line for halflines, a 2D plane
 *     for halfspaces and rectangles. Points are draggable; after each drag
 *     the component re-checks shatterability via `vcShatterCheck` (§18).
 *     A badge at the top shows "shatters" or "does not shatter".
 *
 *   Panel 2 (growth-function plot) — two curves of $n \mapsto s_F(n)$:
 *     the Sauer-Shelah combinatorial bound $\sum_{k=0}^V \binom{n}{k}$ and
 *     the polynomial upper bound $(n+1)^V$. The two bracket the actual
 *     shatter coefficient, and Sauer's lemma guarantees the combinatorial
 *     bound is sharp. Current-sample-size marker overlays.
 *
 * Complexity cap: n ≤ 8 (the "Show all 2ⁿ labellings" toggle would enumerate
 * up to 256 labellings; beyond 8 the grid becomes unreadable).
 */
import { useMemo, useState } from 'react';

import {
  growthFunctionSauerBound,
  vcShatterCheck,
  type VCClassKind,
} from './shared/nonparametric';
import {
  vcClasses,
  type VCClassKey,
  type VCClassSpec,
} from '../../data/empirical-processes-data';
import { empiricalProcessColors as C } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

const MAX_POINTS = 8;
const CANVAS_WIDTH = 420;
const CANVAS_HEIGHT = 280;
const GROWTH_WIDTH = 320;
const GROWTH_HEIGHT = 240;
const MARGIN = { top: 20, right: 16, bottom: 36, left: 48 };
const GROWTH_N_MAX = 30;

const PADDING = 0.08; // fraction of axis range to pad

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function invertLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = rMax - rMin || 1;
  return (p: number) => dMin + ((p - rMin) / span) * (dMax - dMin);
}

/**
 * Witness-point initialisation per class.
 * Positions are in data-space coordinates (the spec's natural units) and
 * later mapped to pixel coordinates by the SVG scales.
 */
function initialPoints(spec: VCClassSpec): Array<number | [number, number]> {
  return spec.witnessPoints.map((p) =>
    Array.isArray(p) ? ([p[0], p[1]] as [number, number]) : (p as number),
  );
}

function pointDataToShatterInput(
  pts: Array<number | [number, number]>,
): Array<number | readonly number[]> {
  return pts.map((p) => (Array.isArray(p) ? p : p));
}

/**
 * Suggest a placement for a newly-added point. 1D → shifted right; 2D →
 * clustered near the centroid of existing points plus a small offset so
 * successive +clicks don't overlap.
 */
function suggestNewPoint(
  spec: VCClassSpec,
  existing: Array<number | [number, number]>,
): number | [number, number] {
  if (spec.inputDim === 1) {
    const vals = existing as number[];
    const last = vals.length > 0 ? vals[vals.length - 1] : 0.3;
    return Math.min(0.95, last + 0.1);
  }
  const pts = existing as [number, number][];
  if (pts.length === 0) return [0.5, 0.5];
  let cx = 0;
  let cy = 0;
  for (const [x, y] of pts) {
    cx += x;
    cy += y;
  }
  cx /= pts.length;
  cy /= pts.length;
  // Small deterministic offset using a hash of the current count
  const k = pts.length;
  const dx = 0.08 * Math.cos(k * 1.3);
  const dy = 0.08 * Math.sin(k * 1.3);
  return [
    Math.max(0.05, Math.min(0.95, cx + dx)),
    Math.max(0.05, Math.min(0.95, cy + dy)),
  ];
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function VCShatteringDemo() {
  const [classKey, setClassKey] = useState<VCClassKey>('halfspace-2d');
  const [points, setPoints] = useState<Array<number | [number, number]>>(() =>
    initialPoints(vcClasses.find((c) => c.key === 'halfspace-2d')!),
  );
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);

  const spec = useMemo(
    () => vcClasses.find((c) => c.key === classKey)!,
    [classKey],
  );

  const shatters = useMemo(
    () =>
      vcShatterCheck(
        classKey as VCClassKind,
        pointDataToShatterInput(points) as Array<number | readonly number[]>,
      ),
    [classKey, points],
  );

  const n = points.length;
  const sauer = useMemo(
    () => growthFunctionSauerBound(spec.vcDim, n),
    [spec.vcDim, n],
  );
  const polyBound = useMemo(() => Math.pow(n + 1, spec.vcDim), [n, spec.vcDim]);
  const maxBound = Math.pow(2, n);

  // Switching classes resets to that class's witness configuration.
  const selectClass = (key: VCClassKey): void => {
    const next = vcClasses.find((c) => c.key === key)!;
    setClassKey(key);
    setPoints(initialPoints(next));
  };

  const addPoint = (): void => {
    if (n >= MAX_POINTS) return;
    setPoints((prev) => [...prev, suggestNewPoint(spec, prev)]);
  };

  const removePoint = (): void => {
    if (n <= 1) return;
    setPoints((prev) => prev.slice(0, -1));
  };

  // ── Canvas scales ─────────────────────────────────────────────────────────

  const plotW = CANVAS_WIDTH - MARGIN.left - MARGIN.right;
  const plotH = CANVAS_HEIGHT - MARGIN.top - MARGIN.bottom;

  // 1D uses the canvas centre line; 2D uses the whole canvas
  const xScale = scaleLinear(-PADDING, 1 + PADDING, MARGIN.left, MARGIN.left + plotW);
  const yScale =
    spec.inputDim === 1
      ? () => MARGIN.top + plotH * 0.5
      : scaleLinear(-PADDING, 1 + PADDING, MARGIN.top + plotH, MARGIN.top);

  const xInv = invertLinear(-PADDING, 1 + PADDING, MARGIN.left, MARGIN.left + plotW);
  const yInv =
    spec.inputDim === 1
      ? () => 0
      : invertLinear(-PADDING, 1 + PADDING, MARGIN.top + plotH, MARGIN.top);

  // ── Drag handlers ─────────────────────────────────────────────────────────

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>): void => {
    if (draggingIdx === null) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    // viewBox maps to CANVAS_WIDTH × CANVAS_HEIGHT; scale pixel → data coords
    const px = ((e.clientX - rect.left) / rect.width) * CANVAS_WIDTH;
    const py = ((e.clientY - rect.top) / rect.height) * CANVAS_HEIGHT;
    const xData = Math.max(-PADDING, Math.min(1 + PADDING, xInv(px)));
    const yData = Math.max(-PADDING, Math.min(1 + PADDING, (yInv as (p: number) => number)(py)));
    setPoints((prev) => {
      const next = prev.slice();
      if (spec.inputDim === 1) next[draggingIdx] = xData;
      else next[draggingIdx] = [xData, yData];
      return next;
    });
  };

  const onPointerUp = (): void => setDraggingIdx(null);

  // ── Growth-function plot scales ──────────────────────────────────────────

  const growthPlotW = GROWTH_WIDTH - MARGIN.left - MARGIN.right;
  const growthPlotH = GROWTH_HEIGHT - MARGIN.top - MARGIN.bottom;
  // Log scale on y, linear on x.
  const yMaxGrowth = Math.pow(2, GROWTH_N_MAX);
  const yLog = (v: number): number =>
    MARGIN.top + growthPlotH -
    (growthPlotH * (Math.log(Math.max(v, 1)) / Math.log(yMaxGrowth)));
  const xGrowth = scaleLinear(0, GROWTH_N_MAX, MARGIN.left, MARGIN.left + growthPlotW);

  // Build the three curves for the current class: Sauer sum, polynomial
  // bound $(n+1)^V$, and $2^n$ "all-labellings" upper bound.
  const curves = useMemo(() => {
    const sauerPts: Array<[number, number]> = [];
    const polyPts: Array<[number, number]> = [];
    const fullPts: Array<[number, number]> = [];
    for (let k = 0; k <= GROWTH_N_MAX; k++) {
      sauerPts.push([k, growthFunctionSauerBound(spec.vcDim, k)]);
      polyPts.push([k, Math.pow(k + 1, spec.vcDim)]);
      fullPts.push([k, Math.pow(2, k)]);
    }
    return { sauerPts, polyPts, fullPts };
  }, [spec.vcDim]);

  const toPath = (pts: readonly (readonly [number, number])[]): string => {
    if (pts.length === 0) return '';
    let d = `M ${xGrowth(pts[0][0]).toFixed(1)} ${yLog(pts[0][1]).toFixed(1)}`;
    for (let i = 1; i < pts.length; i++) {
      d += ` L ${xGrowth(pts[i][0]).toFixed(1)} ${yLog(pts[i][1]).toFixed(1)}`;
    }
    return d;
  };

  // ── Axis ticks ───────────────────────────────────────────────────────────

  const growthXTicks = [0, 5, 10, 15, 20, 25, 30];
  // Log-y ticks at 10^k and 2^{VCDim} landmarks
  const growthYTicks = [1, 10, 100, 1000, 10000, 100000, 1000000];

  // ── Render ───────────────────────────────────────────────────────────────

  const badgeClass = shatters
    ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300'
    : 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300';
  const badgeText = shatters ? '✓ shatters' : '✗ does not shatter';

  return (
    <div className="my-8 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Class:
        </label>
        <div className="inline-flex rounded-md shadow-sm" role="group">
          {vcClasses.map((c) => (
            <button
              key={c.key}
              type="button"
              aria-pressed={classKey === c.key}
              onClick={() => selectClass(c.key)}
              className={`border px-3 py-1.5 text-sm first:rounded-l-md last:rounded-r-md ${
                classKey === c.key
                  ? 'text-white'
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600'
              }`}
              style={
                classKey === c.key
                  ? { background: C.vcHighlight, borderColor: C.vcHighlight }
                  : undefined
              }
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
          <span className="font-medium">Points:</span>
          <button
            type="button"
            onClick={removePoint}
            disabled={n <= 1}
            aria-label="Remove a point"
            className="rounded border border-slate-300 bg-white px-2 py-0.5 text-slate-700 hover:bg-slate-50 disabled:opacity-40 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600"
          >
            −
          </button>
          <span
            className="inline-block min-w-[1.5rem] text-center font-mono"
            aria-live="polite"
          >
            {n}
          </span>
          <button
            type="button"
            onClick={addPoint}
            disabled={n >= MAX_POINTS}
            aria-label="Add a point"
            className="rounded border border-slate-300 bg-white px-2 py-0.5 text-slate-700 hover:bg-slate-50 disabled:opacity-40 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600"
          >
            +
          </button>
        </div>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${badgeClass}`}
          aria-live="polite"
        >
          {badgeText}
        </span>
        <span className="text-slate-600 dark:text-slate-400">
          V = <span className="font-mono">{spec.vcDim}</span>
          {'  · Sauer ∑C(n,k) = '}
          <span className="font-mono">{sauer}</span>
          {'  ≤ (n+1)^V = '}
          <span className="font-mono">{polyBound.toLocaleString()}</span>
          {'  · 2^n upper limit = '}
          <span className="font-mono">{maxBound}</span>
        </span>
      </div>

      <div className="flex flex-wrap gap-4">
        {/* ── Panel 1: shatter canvas ──────────────────────────────────── */}
        <div className="flex-1 min-w-[320px]">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-400">
            Drag points · {spec.description}
          </div>
          <svg
            viewBox={`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`}
            role="img"
            aria-label="VC shatter canvas"
            className="w-full h-auto touch-none select-none"
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerLeave={onPointerUp}
          >
            {/* Grid / axes */}
            <rect
              x={MARGIN.left}
              y={MARGIN.top}
              width={plotW}
              height={plotH}
              fill="none"
              stroke="currentColor"
              className="text-slate-300 dark:text-slate-700"
            />
            {spec.inputDim === 1 && (
              <line
                x1={MARGIN.left}
                x2={MARGIN.left + plotW}
                y1={MARGIN.top + plotH * 0.5}
                y2={MARGIN.top + plotH * 0.5}
                stroke="currentColor"
                className="text-slate-400"
              />
            )}

            {/* Points */}
            {points.map((p, i) => {
              const [x, y] =
                spec.inputDim === 1
                  ? [p as number, 0]
                  : (p as [number, number]);
              const px = xScale(x);
              const py = (yScale as (v: number) => number)(
                spec.inputDim === 1 ? 0 : y,
              );
              const isDragging = draggingIdx === i;
              return (
                <g key={`pt-${i}`}>
                  <circle
                    cx={px}
                    cy={py}
                    r={isDragging ? 10 : 8}
                    fill={shatters ? '#059669' : '#dc2626'}
                    fillOpacity={isDragging ? 0.6 : 0.85}
                    stroke="white"
                    strokeWidth={1.5}
                    style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
                    onPointerDown={(e) => {
                      (e.target as SVGCircleElement).setPointerCapture(e.pointerId);
                      setDraggingIdx(i);
                    }}
                  />
                  <text
                    x={px}
                    y={py + 4}
                    textAnchor="middle"
                    className="fill-white pointer-events-none text-[11px] font-bold"
                  >
                    {i + 1}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* ── Panel 2: growth function ─────────────────────────────────── */}
        <div className="flex-1 min-w-[280px]">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-400">
            Growth function (log-y) — Sauer vs polynomial vs 2ⁿ
          </div>
          <svg
            viewBox={`0 0 ${GROWTH_WIDTH} ${GROWTH_HEIGHT}`}
            role="img"
            aria-label="Growth function comparison"
            className="w-full h-auto"
          >
            {/* Axes */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + growthPlotW}
              y1={MARGIN.top + growthPlotH}
              y2={MARGIN.top + growthPlotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + growthPlotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            {/* X ticks */}
            {growthXTicks.map((t) => (
              <g key={`gxt-${t}`}>
                <line
                  x1={xGrowth(t)}
                  x2={xGrowth(t)}
                  y1={MARGIN.top + growthPlotH}
                  y2={MARGIN.top + growthPlotH + 3}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={xGrowth(t)}
                  y={MARGIN.top + growthPlotH + 14}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {t}
                </text>
              </g>
            ))}
            {/* Log-Y ticks */}
            {growthYTicks.filter((v) => v <= yMaxGrowth).map((t) => (
              <g key={`gyt-${t}`}>
                <line
                  x1={MARGIN.left - 3}
                  x2={MARGIN.left}
                  y1={yLog(t)}
                  y2={yLog(t)}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={MARGIN.left - 6}
                  y={yLog(t)}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {t >= 1000 ? `10^${Math.round(Math.log10(t))}` : t}
                </text>
              </g>
            ))}
            <text
              x={MARGIN.left + growthPlotW / 2}
              y={GROWTH_HEIGHT - 4}
              textAnchor="middle"
              className="fill-slate-700 text-[11px] dark:fill-slate-300"
            >
              n (sample size)
            </text>

            {/* 2^n envelope (dotted grey) — upper bound on shatter coeff */}
            <path
              d={toPath(curves.fullPts)}
              fill="none"
              stroke={C.reference}
              strokeWidth={1.3}
              strokeDasharray="1 3"
            />
            {/* (n+1)^V polynomial bound */}
            <path
              d={toPath(curves.polyPts)}
              fill="none"
              stroke={C.bridge}
              strokeWidth={1.6}
              strokeDasharray="5 3"
            />
            {/* Sauer combinatorial sum */}
            <path
              d={toPath(curves.sauerPts)}
              fill="none"
              stroke={C.vcHighlight}
              strokeWidth={2}
            />

            {/* Current-n marker */}
            <line
              x1={xGrowth(n)}
              x2={xGrowth(n)}
              y1={MARGIN.top}
              y2={MARGIN.top + growthPlotH}
              stroke={C.empirical}
              strokeWidth={1}
              strokeDasharray="3 2"
            />
            <circle cx={xGrowth(n)} cy={yLog(sauer)} r={4} fill={C.vcHighlight} />
          </svg>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-600 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-6" style={{ background: C.vcHighlight }} />
          Sauer ∑ C(n, k)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-0 w-6 border-t-2 border-dashed"
            style={{ borderColor: C.bridge }}
          />
          (n+1)^V polynomial
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-0 w-6 border-t-2 border-dotted"
            style={{ borderColor: C.reference }}
          />
          2ⁿ (unrestricted)
        </span>
      </div>
    </div>
  );
}
