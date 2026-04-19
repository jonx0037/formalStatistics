import { useMemo, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { olsFit } from './shared/regression';
import {
  LEAST_SQUARES_DRAGGABLE_PRESETS,
  type LeastSquaresPreset,
} from '../../data/regression-data';

// ── Layout ───────────────────────────────────────────────────────────────────
const MARGIN = { top: 20, right: 24, bottom: 44, left: 52 };
const MAIN_H = 420;
const INSET_H = 80;
const CANVAS_H = MAIN_H + INSET_H + 16; // +16 for separator gap

// ── Visual palette (hex for portability; matches the Track-5 convention) ─────
const POINT_FILL = '#2563EB'; // blue
const POINT_STROKE = '#FFFFFF';
const FIT_COLOR = '#D97706'; // amber
const RESID_COLOR = '#6B7280'; // grey
const AXIS_COLOR = '#9CA3AF';

type Point = [number, number];

export default function LeastSquaresDraggable() {
  // Container sizing — useResizeObserver owns its own ref.
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 700, 320);

  // Points state — initialized from the `linearTrend` preset; rehydrated on
  // preset change.
  const [activePreset, setActivePreset] = useState<string>('linearTrend');
  const [points, setPoints] = useState<Point[]>(
    () => LEAST_SQUARES_DRAGGABLE_PRESETS.linearTrend.points.map((p) => [...p] as Point),
  );
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  // Scales — derived from the data extent, with a little padding so dragged
  // points don't pin to the viewport edge.
  const { xScale, yScale } = useMemo(() => {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.15 || 1;
    const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.25 || 1;
    const xDomain: [number, number] = [Math.min(...xs) - xPad, Math.max(...xs) + xPad];
    const yDomain: [number, number] = [Math.min(...ys) - yPad, Math.max(...ys) + yPad];
    return {
      xScale: d3.scaleLinear().domain(xDomain).range([MARGIN.left, width - MARGIN.right]),
      yScale: d3.scaleLinear().domain(yDomain).range([MARGIN.top + MAIN_H, MARGIN.top]),
    };
  }, [points, width]);

  // OLS fit — recomputed on every render (7 points is O(1)).
  const fit = useMemo(() => {
    const X = points.map(([x]) => [1, x]);
    const y = points.map((p) => p[1]);
    try {
      return olsFit(X, y);
    } catch {
      return null;
    }
  }, [points]);

  // Fit-line endpoints (drawn edge-to-edge of the x-axis).
  const lineEndpoints = useMemo(() => {
    if (!fit) return null;
    const [xLo, xHi] = xScale.domain();
    return {
      x1: xScale(xLo),
      y1: yScale(fit.beta[0] + fit.beta[1] * xLo),
      x2: xScale(xHi),
      y2: yScale(fit.beta[0] + fit.beta[1] * xHi),
    };
  }, [fit, xScale, yScale]);

  // ── Drag handlers — SVG coordinate space. Unlike D3.drag, we let React own
  // the events so hydration and pointer capture stay predictable. ─────────────
  const svgRef = useRef<SVGSVGElement | null>(null);

  const pointerToDataCoords = useCallback(
    (clientX: number, clientY: number): Point | null => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      const localX = clientX - rect.left;
      const localY = clientY - rect.top;
      // Ignore anything outside the main (scatter) area.
      if (localY > MARGIN.top + MAIN_H) return null;
      return [xScale.invert(localX), yScale.invert(localY)];
    },
    [xScale, yScale],
  );

  const handlePointerDown = (idx: number) => (e: React.PointerEvent<SVGCircleElement>) => {
    e.preventDefault();
    (e.target as SVGCircleElement).setPointerCapture(e.pointerId);
    setDragIndex(idx);
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragIndex === null) return;
    const dataPt = pointerToDataCoords(e.clientX, e.clientY);
    if (!dataPt) return;
    setPoints((pts) => pts.map((p, i) => (i === dragIndex ? dataPt : p)));
  };

  const handlePointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragIndex === null) return;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
    setDragIndex(null);
  };

  // Keyboard nudge for accessibility — focused circle moves 5px per keystroke.
  const handleKeyDown = (idx: number) => (e: React.KeyboardEvent<SVGCircleElement>) => {
    const xDelta = (xScale.domain()[1] - xScale.domain()[0]) / (width - MARGIN.left - MARGIN.right);
    const yDelta = (yScale.domain()[1] - yScale.domain()[0]) / MAIN_H;
    let dx = 0;
    let dy = 0;
    if (e.key === 'ArrowLeft') dx = -5 * xDelta;
    else if (e.key === 'ArrowRight') dx = 5 * xDelta;
    else if (e.key === 'ArrowUp') dy = 5 * yDelta;
    else if (e.key === 'ArrowDown') dy = -5 * yDelta;
    else return;
    e.preventDefault();
    setPoints((pts) => pts.map((p, i) => (i === idx ? [p[0] + dx, p[1] + dy] : p)));
  };

  // Preset switch.
  const applyPreset = (key: string) => {
    const preset: LeastSquaresPreset | undefined = LEAST_SQUARES_DRAGGABLE_PRESETS[key];
    if (!preset) return;
    setActivePreset(key);
    setPoints(preset.points.map((p) => [...p] as Point));
  };

  // Residual inset — bar chart of signed residuals.
  const residualBars = useMemo(() => {
    if (!fit) return [];
    const barGap = 4;
    const totalBarsWidth = width - MARGIN.left - MARGIN.right;
    const barWidth = (totalBarsWidth - barGap * (points.length - 1)) / points.length;
    const maxAbsResid = Math.max(1e-6, ...fit.residuals.map(Math.abs));
    const insetTop = MARGIN.top + MAIN_H + 16;
    const zeroY = insetTop + INSET_H / 2;
    const scale = (r: number) => (r / maxAbsResid) * (INSET_H / 2 - 4);
    return fit.residuals.map((r, i) => ({
      x: MARGIN.left + i * (barWidth + barGap),
      y: r >= 0 ? zeroY - scale(r) : zeroY,
      h: Math.abs(scale(r)),
      w: barWidth,
      positive: r >= 0,
    }));
  }, [fit, width, points.length]);

  const zeroLineY = MARGIN.top + MAIN_H + 16 + INSET_H / 2;

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border"
      style={{
        borderColor: 'var(--color-border, #e5e7eb)',
        backgroundColor: 'var(--color-surface, #ffffff)',
      }}
    >
      {/* Preset bar */}
      <div
        className="flex flex-wrap gap-2 border-b px-3 py-2 text-sm"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        {Object.entries(LEAST_SQUARES_DRAGGABLE_PRESETS).map(([key, preset]) => (
          <button
            key={key}
            type="button"
            onClick={() => applyPreset(key)}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              activePreset === key
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="grid gap-3 p-3 md:grid-cols-[1fr_220px]">
        {/* Main canvas */}
        <svg
          ref={svgRef}
          role="img"
          aria-label="Draggable scatter with live OLS fit. Drag any blue point; fit line, residuals, and coefficient readouts update in real time."
          width="100%"
          viewBox={`0 0 ${width} ${CANVAS_H}`}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
          style={{ touchAction: 'none', userSelect: 'none' }}
        >
          {/* Axes */}
          <g transform={`translate(0, ${MARGIN.top + MAIN_H})`}>
            <line
              x1={MARGIN.left}
              x2={width - MARGIN.right}
              y1={0}
              y2={0}
              stroke={AXIS_COLOR}
            />
            {xScale.ticks(6).map((t) => (
              <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                <line y1={0} y2={4} stroke={AXIS_COLOR} />
                <text y={18} textAnchor="middle" fontSize={11} fill={AXIS_COLOR}>
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
          </g>
          <g>
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + MAIN_H}
              stroke={AXIS_COLOR}
            />
            {yScale.ticks(6).map((t) => (
              <g key={t} transform={`translate(${MARGIN.left}, ${yScale(t)})`}>
                <line x1={-4} x2={0} stroke={AXIS_COLOR} />
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill={AXIS_COLOR}>
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
          </g>

          {/* Fit line */}
          {lineEndpoints && (
            <line
              x1={lineEndpoints.x1}
              y1={lineEndpoints.y1}
              x2={lineEndpoints.x2}
              y2={lineEndpoints.y2}
              stroke={FIT_COLOR}
              strokeWidth={3}
            />
          )}

          {/* Residual segments */}
          {fit &&
            points.map(([x, y], i) => (
              <line
                key={`resid-${i}`}
                x1={xScale(x)}
                y1={yScale(y)}
                x2={xScale(x)}
                y2={yScale(fit.fitted[i])}
                stroke={RESID_COLOR}
                strokeWidth={2}
                opacity={0.6}
              />
            ))}

          {/* Points — rendered last so they receive pointer events on top */}
          {points.map(([x, y], i) => (
            <circle
              key={`pt-${i}`}
              cx={xScale(x)}
              cy={yScale(y)}
              r={dragIndex === i ? 12 : 10}
              fill={POINT_FILL}
              stroke={POINT_STROKE}
              strokeWidth={2}
              tabIndex={0}
              onPointerDown={handlePointerDown(i)}
              onKeyDown={handleKeyDown(i)}
              style={{ cursor: dragIndex === i ? 'grabbing' : 'grab', outline: 'none' }}
              aria-label={`Point ${i + 1} at x=${x.toFixed(2)}, y=${y.toFixed(2)}`}
            />
          ))}

          {/* Inset separator */}
          <line
            x1={MARGIN.left}
            x2={width - MARGIN.right}
            y1={MARGIN.top + MAIN_H + 8}
            y2={MARGIN.top + MAIN_H + 8}
            stroke={AXIS_COLOR}
            strokeDasharray="2 4"
          />

          {/* Residual inset — bar chart */}
          <text
            x={MARGIN.left}
            y={MARGIN.top + MAIN_H + 24}
            fontSize={10}
            fill={AXIS_COLOR}
          >
            Residuals
          </text>
          <line
            x1={MARGIN.left}
            x2={width - MARGIN.right}
            y1={zeroLineY}
            y2={zeroLineY}
            stroke={AXIS_COLOR}
            strokeDasharray="1 2"
          />
          {residualBars.map((b, i) => (
            <rect
              key={`bar-${i}`}
              x={b.x}
              y={b.y}
              width={b.w}
              height={b.h}
              fill={b.positive ? POINT_FILL : FIT_COLOR}
              opacity={0.75}
            />
          ))}
        </svg>

        {/* Sidebar readout */}
        <div
          className="flex flex-col gap-3 rounded-md p-3 text-sm"
          style={{
            backgroundColor: 'var(--color-muted-bg, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <div>
            <div className="text-xs uppercase tracking-wide text-slate-500">Fit</div>
            <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1">
              <span className="text-slate-600">β̂₀</span>
              <span>{fit ? fit.beta[0].toFixed(3) : '—'}</span>
              <span className="text-slate-600">β̂₁</span>
              <span>{fit ? fit.beta[1].toFixed(3) : '—'}</span>
              <span className="text-slate-600">R²</span>
              <span>{fit ? fit.rSquared.toFixed(3) : '—'}</span>
              <span className="text-slate-600">SSE</span>
              <span>{fit ? fit.sse.toFixed(3) : '—'}</span>
              <span className="text-slate-600">σ̂</span>
              <span>{fit ? Math.sqrt(fit.sigmaSquared).toFixed(3) : '—'}</span>
            </div>
          </div>
          <p className="text-xs leading-relaxed text-slate-600">
            Drag any point or press Tab then arrow keys to nudge. The line is the least-squares fit; the grey segments are residuals; the bar panel shows signed residuals in observation order.
          </p>
        </div>
      </div>
    </div>
  );
}
