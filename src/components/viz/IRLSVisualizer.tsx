import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  FAMILIES,
  LINKS,
  glmFit,
  irlsStep,
  type GLMFit,
} from './shared/regression';
import { glmFamilyColors } from './shared/colorScales';
import { IRLS_VISUALIZER_PRESETS } from '../../data/regression-data';

// ── Layout ───────────────────────────────────────────────────────────────────
const MARGIN = { top: 28, right: 24, bottom: 44, left: 52 };
const PANEL_H = 360;

// ── Visual palette ───────────────────────────────────────────────────────────
const TRAIL_COLOR = glmFamilyColors.bernoulli; // pink
const CONTOUR_COLOR = '#94A3B8';
const MLE_COLOR = '#059669';
const AXIS_COLOR = '#9CA3AF';
const ADJ_COLOR = '#2563EB';
const WEIGHT_COLOR = '#7C3AED';

type PresetKey = keyof typeof IRLS_VISUALIZER_PRESETS;

const PRESET_LABELS: Record<PresetKey, string> = {
  wellSeparated: 'Well-separated',
  noisy: 'Noisy',
  nearSeparation: 'Near-separation',
};

interface Iterate {
  beta: [number, number];
  diff: number;
  logLik: number;
  weights: number[];
  mu: number[];
  adjustedResponse: number[];
}

/** Pointwise log-likelihood for the 2-D Bernoulli logistic surface. */
function bernoulliLogLik(X: number[][], y: number[], beta: [number, number]): number {
  let ll = 0;
  for (let i = 0; i < X.length; i++) {
    const eta = X[i][0] * beta[0] + X[i][1] * beta[1];
    // Stable log(1 + exp(eta))
    const log1pExp = eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta));
    ll += y[i] * eta - log1pExp;
  }
  return ll;
}

export default function IRLSVisualizer() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 760;
  const panelW = isStacked ? width - 24 : (width - 36) / 2;

  const [presetKey, setPresetKey] = useState<PresetKey>('wellSeparated');
  const preset = IRLS_VISUALIZER_PRESETS[presetKey];
  const X = preset.X;
  const y = preset.y;

  const [iterates, setIterates] = useState<Iterate[]>(() => initIterates());
  const [playing, setPlaying] = useState<boolean>(false);
  const playRef = useRef<number | null>(null);

  function initIterates(): Iterate[] {
    return [
      {
        beta: [0, 0],
        diff: NaN,
        logLik: bernoulliLogLik(X, y, [0, 0]),
        weights: new Array(y.length).fill(0.25),
        mu: new Array(y.length).fill(0.5),
        adjustedResponse: y.map((yi) => 4 * (yi - 0.5)),
      },
    ];
  }

  // Re-initialize iterates whenever the preset (and hence X, y) changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    setIterates(initIterates());
    setPlaying(false);
  }, [presetKey]);

  // Converged target — null fit object if it diverges (handled gracefully).
  const target = useMemo<GLMFit | null>(() => {
    try {
      const f = glmFit(X, y, FAMILIES.bernoulli, LINKS.logit, { maxIter: 50 });
      return f;
    } catch {
      return null;
    }
  }, [X, y]);

  const stepOnce = () => {
    setIterates((prev) => {
      const last = prev[prev.length - 1];
      let s;
      try {
        s = irlsStep(X, y, last.beta, FAMILIES.bernoulli, LINKS.logit);
      } catch {
        // Singular WLS — stop play and bail.
        setPlaying(false);
        return prev;
      }
      const next: [number, number] = [s.betaNext[0], s.betaNext[1]];
      const dx = next[0] - last.beta[0];
      const dy = next[1] - last.beta[1];
      const diff = Math.sqrt(dx * dx + dy * dy);
      const ll = bernoulliLogLik(X, y, next);
      const augmented: Iterate = {
        beta: next,
        diff,
        logLik: ll,
        weights: s.weights,
        mu: s.mu,
        adjustedResponse: s.adjustedResponse,
      };
      // Stop auto-play near convergence or after divergence-blowup.
      if (diff < 1e-6 || !Number.isFinite(diff) || Math.abs(next[0]) > 50 || Math.abs(next[1]) > 50) {
        setPlaying(false);
      }
      return [...prev, augmented];
    });
  };

  // Auto-play loop.
  useEffect(() => {
    if (!playing) {
      if (playRef.current !== null) {
        window.clearTimeout(playRef.current);
        playRef.current = null;
      }
      return;
    }
    playRef.current = window.setTimeout(() => stepOnce(), 800);
    return () => {
      if (playRef.current !== null) {
        window.clearTimeout(playRef.current);
        playRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, iterates]);

  const reset = () => {
    setPlaying(false);
    setIterates(initIterates());
  };

  const last = iterates[iterates.length - 1];
  const targetBeta = target?.beta ?? [NaN, NaN];
  const diverging = target ? !target.converged : true;

  // ── Contour grid (left panel) ─────────────────────────────────────────────
  // Centered around the MLE (or [0, 0] if it diverged), adaptive radius.
  const center = useMemo<[number, number]>(() => {
    if (target && target.converged) return [target.beta[0], target.beta[1]];
    return [0, 0];
  }, [target]);

  const radius = useMemo(() => {
    const r = Math.max(
      3,
      Math.abs(center[0]) + 2,
      Math.abs(center[1]) + 2,
      Math.abs(last.beta[0]) + 1,
      Math.abs(last.beta[1]) + 1,
    );
    return Math.min(r, 8);
  }, [center, last.beta]);

  const xDomain: [number, number] = [center[0] - radius, center[0] + radius];
  const yDomain: [number, number] = [center[1] - radius, center[1] + radius];

  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain(xDomain)
        .range([MARGIN.left, panelW - MARGIN.right]),
    [xDomain, panelW],
  );
  const yScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain(yDomain)
        .range([MARGIN.top + PANEL_H, MARGIN.top]),
    [yDomain],
  );

  // Sample log-likelihood on a grid.
  const grid = useMemo(() => {
    const N = 36;
    const cells: { b0: number; b1: number; ll: number }[][] = [];
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i <= N; i++) {
      const row: { b0: number; b1: number; ll: number }[] = [];
      const b1v = yDomain[1] - ((yDomain[1] - yDomain[0]) * i) / N;
      for (let j = 0; j <= N; j++) {
        const b0v = xDomain[0] + ((xDomain[1] - xDomain[0]) * j) / N;
        const ll = bernoulliLogLik(X, y, [b0v, b1v]);
        if (Number.isFinite(ll)) {
          if (ll < lo) lo = ll;
          if (ll > hi) hi = ll;
        }
        row.push({ b0: b0v, b1: b1v, ll });
      }
      cells.push(row);
    }
    return { cells, lo, hi };
  }, [X, y, xDomain, yDomain]);

  // Build a handful of contour level paths via a simple cell-mean-coloring.
  // Use tinted rectangles instead of true contour interpolation — cheap + readable.
  const heatmap = useMemo(() => {
    const out: { x: number; y: number; w: number; h: number; t: number }[] = [];
    const cells = grid.cells;
    const N = cells.length - 1;
    const cellW = (xScale(xDomain[1]) - xScale(xDomain[0])) / N;
    const cellH = (yScale(yDomain[0]) - yScale(yDomain[1])) / N;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const c = cells[i][j];
        const t = grid.hi === grid.lo ? 0.5 : (c.ll - grid.lo) / (grid.hi - grid.lo);
        out.push({
          x: xScale(c.b0),
          y: yScale(c.b1),
          w: cellW + 0.5,
          h: cellH + 0.5,
          t,
        });
      }
    }
    return out;
  }, [grid, xScale, yScale, xDomain, yDomain]);

  // Trail polyline.
  const trailPath = useMemo(() => {
    const line = d3
      .line<Iterate>()
      .x((d) => xScale(d.beta[0]))
      .y((d) => yScale(d.beta[1]));
    return line(iterates);
  }, [iterates, xScale, yScale]);

  // Click-to-set-β^(0) on the contour panel (only when on initial iterate).
  const handleContourClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (iterates.length > 1) return;
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const vbox = svg.viewBox.baseVal;
    const rect = svg.getBoundingClientRect();
    const sx = vbox.width / rect.width;
    const sy = vbox.height / rect.height;
    const localX = (e.clientX - rect.left) * sx;
    const localY = (e.clientY - rect.top) * sy;
    const b0 = xScale.invert(localX);
    const b1 = yScale.invert(localY);
    if (
      b0 < xDomain[0] ||
      b0 > xDomain[1] ||
      b1 < yDomain[0] ||
      b1 > yDomain[1]
    ) {
      return;
    }
    setIterates([
      {
        beta: [b0, b1],
        diff: NaN,
        logLik: bernoulliLogLik(X, y, [b0, b1]),
        weights: new Array(y.length).fill(0.25),
        mu: new Array(y.length).fill(0.5),
        adjustedResponse: y.map((yi) => 4 * (yi - 0.5)),
      },
    ]);
  };

  // ── Right panel: weights bar chart + adjusted response scatter ────────────
  const adjXScale = useMemo(() => {
    const xs = X.map((row) => row[1]);
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    return d3
      .scaleLinear()
      .domain([lo - 0.3, hi + 0.3])
      .range([MARGIN.left, panelW - MARGIN.right]);
  }, [X, panelW]);

  const adjYScale = useMemo(() => {
    const all = last.adjustedResponse;
    const lo = Math.min(...all, -2);
    const hi = Math.max(...all, 2);
    return d3
      .scaleLinear()
      .domain([lo - 0.5, hi + 0.5])
      .range([MARGIN.top + PANEL_H * 0.55, MARGIN.top]);
  }, [last]);

  const weightsYScale = useMemo(() => {
    const wmax = Math.max(...last.weights, 0.25);
    return d3
      .scaleLinear()
      .domain([0, wmax])
      .range([MARGIN.top + PANEL_H, MARGIN.top + PANEL_H * 0.65]);
  }, [last]);

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
        {(Object.keys(IRLS_VISUALIZER_PRESETS) as PresetKey[]).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setPresetKey(key)}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              presetKey === key
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
            }`}
          >
            {PRESET_LABELS[key]}
          </button>
        ))}
        {diverging && presetKey === 'nearSeparation' && (
          <span className="ml-auto rounded bg-red-100 px-2 py-1 text-xs font-medium text-red-700">
            Diverges toward boundary
          </span>
        )}
      </div>

      <div
        className={`flex gap-3 p-3 ${isStacked ? 'flex-col' : 'flex-row'}`}
      >
        {/* LEFT: contour map of ℓ(β₀, β₁) with iterate trail */}
        <div className={isStacked ? 'w-full' : 'w-1/2'}>
          <p className="mb-1 text-xs text-slate-600">
            Log-likelihood ℓ(β₀, β₁) — click to set β<sup>(0)</sup>
          </p>
          <svg
            width="100%"
            viewBox={`0 0 ${panelW} ${MARGIN.top + PANEL_H + 30}`}
            onClick={handleContourClick}
            style={{
              cursor: iterates.length > 1 ? 'default' : 'crosshair',
              backgroundColor: 'var(--color-viz-bg, #ffffff)',
            }}
          >
            {/* Heatmap */}
            {heatmap.map((c, i) => (
              <rect
                key={i}
                x={c.x}
                y={c.y}
                width={c.w}
                height={c.h}
                fill={d3.interpolateViridis(c.t)}
                opacity={0.55}
              />
            ))}

            {/* Axes */}
            <g transform={`translate(0, ${MARGIN.top + PANEL_H})`}>
              <line x1={MARGIN.left} x2={panelW - MARGIN.right} stroke={AXIS_COLOR} />
              {xScale.ticks(5).map((t) => (
                <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                  <line y1={0} y2={4} stroke={AXIS_COLOR} />
                  <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
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
                y2={MARGIN.top + PANEL_H}
                stroke={AXIS_COLOR}
              />
              {yScale.ticks(5).map((t) => (
                <g key={t} transform={`translate(${MARGIN.left}, ${yScale(t)})`}>
                  <line x1={-4} x2={0} stroke={AXIS_COLOR} />
                  <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill={AXIS_COLOR}>
                    {t.toFixed(1)}
                  </text>
                </g>
              ))}
            </g>
            <text
              x={(MARGIN.left + panelW - MARGIN.right) / 2}
              y={MARGIN.top + PANEL_H + 28}
              textAnchor="middle"
              fontSize={11}
              fill={AXIS_COLOR}
            >
              β₀
            </text>
            <text
              transform={`translate(14, ${(MARGIN.top + MARGIN.top + PANEL_H) / 2}) rotate(-90)`}
              textAnchor="middle"
              fontSize={11}
              fill={AXIS_COLOR}
            >
              β₁
            </text>

            {/* Iterate trail */}
            {trailPath && (
              <path d={trailPath} fill="none" stroke={CONTOUR_COLOR} strokeWidth={1.5} strokeDasharray="3 3" />
            )}
            {/* Iterate markers */}
            {iterates.map((it, k) => (
              <circle
                key={k}
                cx={xScale(it.beta[0])}
                cy={yScale(it.beta[1])}
                r={k === iterates.length - 1 ? 5.5 : 3}
                fill={TRAIL_COLOR}
                stroke="#fff"
                strokeWidth={k === iterates.length - 1 ? 2 : 1}
                opacity={k === iterates.length - 1 ? 1 : 0.85}
              />
            ))}

            {/* Converged β̂ (hollow green if it converged) */}
            {target && target.converged && (
              <circle
                cx={xScale(targetBeta[0])}
                cy={yScale(targetBeta[1])}
                r={7}
                fill="none"
                stroke={MLE_COLOR}
                strokeWidth={2.5}
              />
            )}
          </svg>
        </div>

        {/* RIGHT: anatomy panel — weights bar + adjusted response scatter */}
        <div className={isStacked ? 'w-full' : 'w-1/2'}>
          <p className="mb-1 text-xs text-slate-600">
            Adjusted response z = η + g'(μ)(y − μ) and weights W
          </p>
          <svg
            width="100%"
            viewBox={`0 0 ${panelW} ${MARGIN.top + PANEL_H + 30}`}
            style={{ backgroundColor: 'var(--color-viz-bg, #ffffff)' }}
          >
            {/* Adjusted-response scatter */}
            <g>
              {/* y=0 line */}
              <line
                x1={MARGIN.left}
                x2={panelW - MARGIN.right}
                y1={adjYScale(0)}
                y2={adjYScale(0)}
                stroke={AXIS_COLOR}
                strokeDasharray="2 4"
              />
              {X.map((row, i) => (
                <circle
                  key={i}
                  cx={adjXScale(row[1])}
                  cy={adjYScale(last.adjustedResponse[i])}
                  r={3}
                  fill={ADJ_COLOR}
                  opacity={0.7}
                />
              ))}
              {/* Y axis */}
              <line
                x1={MARGIN.left}
                x2={MARGIN.left}
                y1={MARGIN.top}
                y2={MARGIN.top + PANEL_H * 0.55}
                stroke={AXIS_COLOR}
              />
              {adjYScale.ticks(4).map((t) => (
                <g key={t} transform={`translate(${MARGIN.left}, ${adjYScale(t)})`}>
                  <line x1={-3} x2={0} stroke={AXIS_COLOR} />
                  <text x={-6} dy="0.32em" textAnchor="end" fontSize={9} fill={AXIS_COLOR}>
                    {t.toFixed(1)}
                  </text>
                </g>
              ))}
              <text
                x={MARGIN.left + 4}
                y={MARGIN.top + 10}
                fontSize={10}
                fill={ADJ_COLOR}
              >
                z (adjusted)
              </text>
            </g>

            {/* Weights bar chart */}
            <g>
              <line
                x1={MARGIN.left}
                x2={panelW - MARGIN.right}
                y1={MARGIN.top + PANEL_H}
                y2={MARGIN.top + PANEL_H}
                stroke={AXIS_COLOR}
              />
              {X.map((row, i) => {
                const w = last.weights[i];
                const cx = adjXScale(row[1]);
                const barW = Math.max(1.5, (panelW - MARGIN.left - MARGIN.right) / (X.length * 1.5));
                return (
                  <rect
                    key={i}
                    x={cx - barW / 2}
                    y={weightsYScale(w)}
                    width={barW}
                    height={MARGIN.top + PANEL_H - weightsYScale(w)}
                    fill={WEIGHT_COLOR}
                    opacity={0.65}
                  />
                );
              })}
              <text
                x={MARGIN.left + 4}
                y={MARGIN.top + PANEL_H * 0.7}
                fontSize={10}
                fill={WEIGHT_COLOR}
              >
                W weights
              </text>
              {/* X axis (shared) */}
              {adjXScale.ticks(5).map((t) => (
                <g key={t} transform={`translate(${adjXScale(t)}, ${MARGIN.top + PANEL_H})`}>
                  <line y1={0} y2={4} stroke={AXIS_COLOR} />
                  <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
                    {t.toFixed(1)}
                  </text>
                </g>
              ))}
              <text
                x={(MARGIN.left + panelW - MARGIN.right) / 2}
                y={MARGIN.top + PANEL_H + 28}
                textAnchor="middle"
                fontSize={11}
                fill={AXIS_COLOR}
              >
                x₁
              </text>
            </g>
          </svg>
        </div>
      </div>

      {/* Controls */}
      <div
        className="grid gap-3 border-t p-3 text-sm md:grid-cols-[1fr_240px]"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={stepOnce}
            className="rounded bg-slate-900 px-3 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-50"
            disabled={playing}
          >
            Step ▸
          </button>
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded bg-slate-700 px-3 py-1 text-xs text-white hover:bg-slate-500"
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded bg-slate-100 px-3 py-1 text-xs text-slate-800 hover:bg-slate-200"
          >
            Reset
          </button>
          <span className="ml-2 text-xs text-slate-600">
            n = {X.length}, β<sup>(0)</sup> = ({iterates[0].beta[0].toFixed(2)},{' '}
            {iterates[0].beta[1].toFixed(2)})
          </span>
        </div>

        <div
          className="flex flex-col gap-1 rounded-md p-2 text-xs"
          style={{
            backgroundColor: 'var(--color-surface-alt, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <div className="flex justify-between">
            <span className="text-slate-600">Iteration</span>
            <span>{iterates.length - 1}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">‖Δβ‖</span>
            <span>{Number.isFinite(last.diff) ? last.diff.toExponential(2) : '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">log-lik</span>
            <span>{last.logLik.toFixed(3)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">β current</span>
            <span>
              ({last.beta[0].toFixed(2)}, {last.beta[1].toFixed(2)})
            </span>
          </div>
          {target && target.converged && (
            <div className="flex justify-between">
              <span className="text-slate-600">β̂ MLE</span>
              <span>
                ({target.beta[0].toFixed(2)}, {target.beta[1].toFixed(2)})
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
