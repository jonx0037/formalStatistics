/**
 * EmpiricalBayesTypeIIMLE — interactive for Topic 28 §28.7.
 *
 * Reader explores the Type-II marginal log-likelihood surface over
 * (μ, τ²) for a chosen dataset, drags a marker to feel the topology,
 * and triggers a "Find MLE" gradient-ascent animation that walks to
 * the empirical-Bayes optimum. A "Full-Bayes posterior mean" toggle
 * overlays the precomputed NUTS posterior mean of (μ, τ²) — the EB
 * vs full-Bayes contrast that motivates half-Cauchy priors (GEL2006).
 *
 * Visualization:
 *   — 60×60 heatmap of log m(y | μ, τ²) (custom palette: low=light,
 *     high=violet) rendered as a single SVG grid of rects.
 *   — Inline isocontours via a lightweight marching-squares pass
 *     (~40 LOC, private to this component per plan Decision 2).
 *   — Current marker (draggable), EB MLE star, Full-Bayes circle.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { typeIIMarginalLogLikelihood, typeIIMLE } from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import { typeIIMLEPresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 28, right: 24, bottom: 44, left: 60 };
const MOBILE_BREAKPOINT = 640;
const GRID_N = 60;

// Hand-rolled marching-squares contour pass. For each level c, scan cells
// in the grid; for each cell classify its corners (inside/outside by
// comparing to c) and emit one or two line segments per cell based on
// the 16-case lookup. Segment endpoints are linear interpolations along
// the cell's edges. Kept private to EmpiricalBayesTypeIIMLE per plan.
type Segment = { x1: number; y1: number; x2: number; y2: number };

function marchingSquares(
  grid: number[][], // grid[row][col]: value at (i, j) with row = i, col = j
  level: number,
  xScale: (col: number) => number,
  yScale: (row: number) => number,
): Segment[] {
  const segs: Segment[] = [];
  const nRows = grid.length;
  const nCols = grid[0].length;

  // Linear interpolation along the edge from (a,b) to (c,d) crossing level.
  const lerp = (va: number, vb: number, ta: number, tb: number) => {
    const t = (level - va) / (vb - va || 1e-12);
    return ta + t * (tb - ta);
  };

  for (let i = 0; i < nRows - 1; i++) {
    for (let j = 0; j < nCols - 1; j++) {
      const v00 = grid[i][j];
      const v10 = grid[i][j + 1];
      const v11 = grid[i + 1][j + 1];
      const v01 = grid[i + 1][j];
      const mask =
        (v00 > level ? 1 : 0) |
        (v10 > level ? 2 : 0) |
        (v11 > level ? 4 : 0) |
        (v01 > level ? 8 : 0);
      if (mask === 0 || mask === 15) continue;
      // Edge crossings (top / right / bottom / left).
      const x0 = xScale(j),
        x1 = xScale(j + 1);
      const y0 = yScale(i),
        y1 = yScale(i + 1);
      const top = () => ({ x: lerp(v00, v10, x0, x1), y: y0 });
      const right = () => ({ x: x1, y: lerp(v10, v11, y0, y1) });
      const bottom = () => ({ x: lerp(v01, v11, x0, x1), y: y1 });
      const left = () => ({ x: x0, y: lerp(v00, v01, y0, y1) });
      // 16-case dispatch; symmetric cases collapsed.
      const emit = (a: { x: number; y: number }, b: { x: number; y: number }) =>
        segs.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y });
      switch (mask) {
        case 1:
        case 14:
          emit(top(), left()); break;
        case 2:
        case 13:
          emit(top(), right()); break;
        case 3:
        case 12:
          emit(left(), right()); break;
        case 4:
        case 11:
          emit(right(), bottom()); break;
        case 5:
          emit(top(), left()); emit(right(), bottom()); break;
        case 6:
        case 9:
          emit(top(), bottom()); break;
        case 7:
        case 8:
          emit(left(), bottom()); break;
        case 10:
          emit(top(), right()); emit(left(), bottom()); break;
        default: break;
      }
    }
  }
  return segs;
}

// Map log-marginal value to a palette color (light → violet).
function logMColor(v: number, vMin: number, vMax: number): string {
  const t = Math.max(0, Math.min(1, (v - vMin) / (vMax - vMin || 1)));
  // Blend slate-100 → violet-600. Simple RGB lerp; HSL would be nicer but SVG
  // doesn't need the perceptual accuracy here.
  const r = Math.round(241 + (124 - 241) * t);
  const g = Math.round(245 + (58 - 245) * t);
  const b = Math.round(249 + (237 - 249) * t);
  return `rgb(${r},${g},${b})`;
}

export default function EmpiricalBayesTypeIIMLE() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = typeIIMLEPresets[presetIdx];
  const yData = [...preset.y];
  const sigmaSq = preset.sigma.map((s) => s * s);

  // Grid bounds adapt to the data scale so each preset gets a useful view.
  // One useMemo pass computes mean/range/variance in a single sweep — the
  // earlier version had a nested reduce inside tauSqMax that was O(n²)
  // (Copilot PR-32).
  const { muMin, muMax, tauSqMax } = useMemo(() => {
    const n = yData.length;
    let sum = 0;
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const v of yData) {
      sum += v;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    const mean = sum / n;
    let sumSq = 0;
    for (const v of yData) sumSq += (v - mean) ** 2;
    const variance = n > 1 ? sumSq / (n - 1) : 0;
    const range = maxVal - minVal;
    return {
      muMin: mean - range,
      muMax: mean + range,
      tauSqMax: Math.max(100, 10 * variance),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIdx]);
  const tauSqMin = 0.01;

  // Build log-τ² grid (log-spaced) and μ grid (linear).
  const grid = useMemo(() => {
    const g: number[][] = [];
    const logTauMin = Math.log(tauSqMin);
    const logTauMax = Math.log(tauSqMax);
    for (let i = 0; i < GRID_N; i++) {
      const row: number[] = [];
      const logTau = logTauMin + (i / (GRID_N - 1)) * (logTauMax - logTauMin);
      const tauSq = Math.exp(logTau);
      for (let j = 0; j < GRID_N; j++) {
        const mu = muMin + (j / (GRID_N - 1)) * (muMax - muMin);
        row.push(typeIIMarginalLogLikelihood(yData, sigmaSq, mu, tauSq));
      }
      g.push(row);
    }
    return g;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIdx, muMin, muMax, tauSqMax]);

  // Range of log-marginal values.
  const gridMin = useMemo(() => Math.min(...grid.flat()), [grid]);
  const gridMax = useMemo(() => Math.max(...grid.flat()), [grid]);

  // Current draggable (μ, τ²).
  const [current, setCurrent] = useState<{ mu: number; tauSq: number }>({
    mu: (muMin + muMax) / 2,
    tauSq: Math.sqrt(tauSqMin * tauSqMax),
  });

  useEffect(() => {
    setCurrent({ mu: (muMin + muMax) / 2, tauSq: Math.sqrt(tauSqMin * tauSqMax) });
    setAnimating(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presetIdx]);

  const ebFit = useMemo(() => typeIIMLE(yData, sigmaSq), [presetIdx]); // eslint-disable-line react-hooks/exhaustive-deps
  const [showFullBayes, setShowFullBayes] = useState(true);

  const [animating, setAnimating] = useState<number | null>(null);
  const draggingRef = useRef(false);

  // "Find MLE" — animate a sequence of gradient-ascent-ish steps from
  // the current position to the EB MLE.
  const findMLE = () => {
    const start = { ...current };
    const target = { mu: ebFit.mu, tauSq: ebFit.tauSq };
    const steps = 40;
    let stepIdx = 0;
    const intervalId = window.setInterval(() => {
      stepIdx++;
      const t = stepIdx / steps;
      const eased = t * t * (3 - 2 * t); // smoothstep
      setCurrent({
        mu: start.mu + eased * (target.mu - start.mu),
        tauSq: start.tauSq + eased * (target.tauSq - start.tauSq),
      });
      if (stepIdx >= steps) {
        window.clearInterval(intervalId);
        setAnimating(null);
      }
    }, 30);
    setAnimating(intervalId);
  };

  // Cleanup animation on unmount.
  useEffect(() => {
    return () => {
      if (animating !== null) window.clearInterval(animating);
    };
  }, [animating]);

  // Layout.
  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(620, (width ?? 620) - 16));
  const chartH = isMobile ? 340 : 400;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const cellW = plotW / (GRID_N - 1);
  const cellH = plotH / (GRID_N - 1);
  const colToMu = (c: number) => muMin + (c / (GRID_N - 1)) * (muMax - muMin);
  const rowToTauSq = (r: number) => {
    const logTauMin = Math.log(tauSqMin);
    const logTauMax = Math.log(tauSqMax);
    return Math.exp(logTauMin + (r / (GRID_N - 1)) * (logTauMax - logTauMin));
  };
  const xOfMu = (m: number) => ((m - muMin) / (muMax - muMin)) * plotW;
  const yOfTauSq = (ts: number) => {
    const logTauMin = Math.log(tauSqMin);
    const logTauMax = Math.log(tauSqMax);
    const t = (Math.log(Math.max(tauSqMin, ts)) - logTauMin) / (logTauMax - logTauMin);
    return plotH - t * plotH;
  };

  // Contour levels: 8 evenly spaced percentiles of grid values.
  const levels = useMemo(() => {
    const sorted = [...grid.flat()].sort((a, b) => a - b);
    const out: number[] = [];
    for (let i = 1; i <= 8; i++) {
      const idx = Math.floor((i / 9) * (sorted.length - 1));
      out.push(sorted[idx]);
    }
    return out;
  }, [grid]);

  const contours = useMemo(() => {
    const result: Segment[] = [];
    for (const level of levels) {
      const xScale = (c: number) => c * cellW;
      // Row 0 is the bottom of the plot visually (tauSq min); row GRID_N-1 is the top.
      // Our grid[i] has i=0 at tauSqMin; we need y inverted so bottom row is at plotH.
      const yScale = (r: number) => plotH - r * cellH;
      result.push(...marchingSquares(grid, level, xScale, yScale));
    }
    return result;
  }, [grid, levels, cellW, cellH, plotH]);

  // Memoize the 3,600 heatmap <rect>s so React doesn't rebuild them on every
  // marker-drag / "Find MLE" animation frame. Depends only on grid + layout.
  // (gemini PR-32)
  const heatmapRects = useMemo(
    () =>
      grid.flatMap((row, i) =>
        row.map((v, j) => (
          <rect
            key={`${i}-${j}`}
            x={j * cellW - cellW / 2}
            y={plotH - i * cellH - cellH / 2}
            width={cellW + 1}
            height={cellH + 1}
            fill={logMColor(v, gridMin, gridMax)}
          />
        )),
      ),
    [grid, gridMin, gridMax, cellW, cellH, plotH],
  );

  const handlePointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    draggingRef.current = true;
    handlePointerMove(e);
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!draggingRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const rawX = e.clientX - rect.left - MARGIN.left;
    const rawY = e.clientY - rect.top - MARGIN.top;
    const mu = muMin + Math.max(0, Math.min(1, rawX / plotW)) * (muMax - muMin);
    const tFracInv = Math.max(0, Math.min(1, 1 - rawY / plotH));
    const logTauMin = Math.log(tauSqMin);
    const logTauMax = Math.log(tauSqMax);
    const tauSq = Math.exp(logTauMin + tFracInv * (logTauMax - logTauMin));
    setCurrent({ mu, tauSq });
  };
  const handlePointerUp = () => {
    draggingRef.current = false;
  };

  const currentLogM = typeIIMarginalLogLikelihood(yData, sigmaSq, current.mu, current.tauSq);

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Dataset:</span>
        {typeIIMLEPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => setPresetIdx(i)}
            className={`rounded px-3 py-1 text-xs transition ${
              i === presetIdx
                ? 'bg-[var(--color-shrink)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <button
          onClick={findMLE}
          disabled={animating !== null}
          className="rounded px-3 py-1.5 text-xs font-medium text-white transition disabled:opacity-50"
          style={{ background: bayesianColors.shrink }}
        >
          {animating !== null ? 'Ascending…' : 'Find MLE'}
        </button>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={showFullBayes}
            onChange={(e) => setShowFullBayes(e.target.checked)}
            className="accent-[var(--color-posterior)]"
          />
          Show full-Bayes posterior mean (NUTS)
        </label>
      </div>

      <svg
        width={chartW}
        height={chartH}
        className="block touch-none"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Heatmap (memoized — see heatmapRects) */}
          {heatmapRects}

          {/* Isocontours */}
          {contours.map((s, i) => (
            <line
              key={i}
              x1={s.x1}
              y1={s.y1}
              x2={s.x2}
              y2={s.y2}
              stroke="rgba(255,255,255,0.55)"
              strokeWidth={0.8}
            />
          ))}

          {/* EB MLE star */}
          <g transform={`translate(${xOfMu(ebFit.mu)}, ${yOfTauSq(Math.max(tauSqMin, ebFit.tauSq))})`}>
            <polygon
              points="0,-9 2.6,-2.8 9,-2.8 3.8,1.5 5.8,8 0,4.2 -5.8,8 -3.8,1.5 -9,-2.8 -2.6,-2.8"
              fill="white"
              stroke={bayesianColors.mle}
              strokeWidth={2}
            />
          </g>

          {/* Full-Bayes posterior-mean circle */}
          {showFullBayes && (
            <g
              transform={`translate(${xOfMu(preset.fullBayesMuMean)}, ${yOfTauSq(
                Math.max(tauSqMin, preset.fullBayesTauSqMean),
              )})`}
            >
              <circle r={7} fill="none" stroke={bayesianColors.posterior} strokeWidth={2.5} />
              <circle r={3} fill={bayesianColors.posterior} />
            </g>
          )}

          {/* Current draggable marker */}
          <circle
            cx={xOfMu(current.mu)}
            cy={yOfTauSq(current.tauSq)}
            r={7}
            fill={bayesianColors.shrink}
            stroke="white"
            strokeWidth={2}
            style={{ cursor: 'grab' }}
          />

          {/* Axes */}
          <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="var(--color-text-muted)" />
          <line x1={0} y1={0} x2={0} y2={plotH} stroke="var(--color-text-muted)" />
          {/* x ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line
                x1={f * plotW}
                y1={plotH}
                x2={f * plotW}
                y2={plotH + 4}
                stroke="var(--color-text-muted)"
              />
              <text
                x={f * plotW}
                y={plotH + 16}
                textAnchor="middle"
                fontSize={10}
                fill="var(--color-text-muted)"
              >
                {colToMu(f * (GRID_N - 1)).toFixed(1)}
              </text>
            </g>
          ))}
          <text
            x={plotW / 2}
            y={plotH + 34}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
          >
            Hyperprior mean μ
          </text>
          {/* y ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line
                x1={-4}
                y1={plotH * (1 - f)}
                x2={0}
                y2={plotH * (1 - f)}
                stroke="var(--color-text-muted)"
              />
              <text
                x={-8}
                y={plotH * (1 - f) + 4}
                textAnchor="end"
                fontSize={9}
                fill="var(--color-text-muted)"
                fontFamily="ui-monospace, monospace"
              >
                {rowToTauSq(f * (GRID_N - 1)).toFixed(rowToTauSq(f * (GRID_N - 1)) < 1 ? 2 : 0)}
              </text>
            </g>
          ))}
          <text
            x={-42}
            y={plotH / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
            transform={`rotate(-90 ${-42} ${plotH / 2})`}
          >
            Between-group variance τ² (log)
          </text>
        </g>
      </svg>

      <div className="mt-3 grid grid-cols-1 gap-3 text-sm md:grid-cols-3">
        <div className="rounded border p-2" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Current
          </div>
          <div className="mt-1 font-mono text-xs">
            μ = {current.mu.toFixed(2)} · τ² = {current.tauSq.toFixed(2)}
          </div>
          <div className="mt-1 text-xs">log m = <span className="font-mono">{currentLogM.toFixed(3)}</span></div>
        </div>
        <div className="rounded border p-2" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: bayesianColors.mle }}>
            EB MLE (★)
          </div>
          <div className="mt-1 font-mono text-xs">
            μ̂ = {ebFit.mu.toFixed(2)} · τ̂² = {ebFit.tauSq.toFixed(2)}
          </div>
          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {ebFit.converged ? `converged in ${ebFit.iterations} iter` : 'did not converge'}
          </div>
        </div>
        <div className="rounded border p-2" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: bayesianColors.posterior }}>
            Full Bayes (○)
          </div>
          <div className="mt-1 font-mono text-xs">
            μ̄ = {preset.fullBayesMuMean.toFixed(2)} · τ̄² = {preset.fullBayesTauSqMean.toFixed(2)}
          </div>
          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            NUTS posterior mean, half-Cauchy(0,5) on τ
          </div>
        </div>
      </div>
    </div>
  );
}
