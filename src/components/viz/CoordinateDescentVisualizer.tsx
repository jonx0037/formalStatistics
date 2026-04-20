import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { penaltyColors } from './shared/colorScales';
import { softThreshold } from './shared/regression';

const PANEL_H = 320;
const MARGIN = { top: 24, right: 24, bottom: 36, left: 44 };

// 2D toy lasso problem. Standardized X with ‖X[:,j]‖² = n, so the soft-
// thresholding update divides by n.
const N_TOY = 60;
function buildToy(): { X: number[][]; y: number[] } {
  // Two correlated standardized predictors; true β = (1.5, -0.8) + noise.
  const X: number[][] = [];
  const y: number[] = [];
  let s = 7;
  const rand = (): number => {
    // Tiny LCG for deterministic values.
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };
  // Sample base predictor + perturb to add slight correlation.
  const x1raw: number[] = [];
  const x2raw: number[] = [];
  for (let i = 0; i < N_TOY; i++) {
    const a = rand();
    const b = rand();
    x1raw.push(a);
    x2raw.push(0.5 * a + Math.sqrt(1 - 0.25) * b);
  }
  // Standardize: zero mean, ‖·‖² = n.
  const std = (xs: number[]): number[] => {
    const m = xs.reduce((p, c) => p + c, 0) / xs.length;
    const ss = xs.reduce((p, c) => p + (c - m) * (c - m), 0) / xs.length;
    const sd = Math.sqrt(ss);
    return xs.map((x) => (x - m) / sd);
  };
  const x1 = std(x1raw);
  const x2 = std(x2raw);
  for (let i = 0; i < N_TOY; i++) {
    X.push([x1[i], x2[i]]);
    y.push(1.5 * x1[i] - 0.8 * x2[i] + 0.4 * (rand() * 0.5));
  }
  // Center y.
  const ym = y.reduce((p, c) => p + c, 0) / y.length;
  return { X, y: y.map((v) => v - ym) };
}

const TOY = buildToy();
const X_TOY = TOY.X;
const Y_TOY = TOY.y;

// Lasso objective f(β) = ½‖y − Xβ‖² + λ‖β‖₁.
function lassoObjective(beta: [number, number], lambda: number): number {
  let rss = 0;
  for (let i = 0; i < N_TOY; i++) {
    const r = Y_TOY[i] - X_TOY[i][0] * beta[0] - X_TOY[i][1] * beta[1];
    rss += r * r;
  }
  return 0.5 * rss + lambda * (Math.abs(beta[0]) + Math.abs(beta[1]));
}

// Single coordinate-descent step on coordinate j (0 or 1).
function cdStep(
  beta: [number, number],
  lambda: number,
  j: 0 | 1,
): [number, number] {
  // r = y − Xβ; r_-j = r + X[:,j] · β_j.
  let zj = 0;
  for (let i = 0; i < N_TOY; i++) {
    const xij = X_TOY[i][j];
    const xb = X_TOY[i][0] * beta[0] + X_TOY[i][1] * beta[1];
    zj += xij * (Y_TOY[i] - xb + xij * beta[j]);
  }
  // ‖X[:,j]‖² = N_TOY after standardization.
  const newBetaJ = softThreshold(zj, lambda) / N_TOY;
  const out: [number, number] = [beta[0], beta[1]];
  out[j] = newBetaJ;
  return out;
}

interface IterEntry {
  beta: [number, number];
  obj: number;
  coord: 0 | 1 | -1; // -1 = initial
}

export default function CoordinateDescentVisualizer() {
  const { ref: containerRef, width: measuredWidth } =
    useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);
  const isStacked = width < 720;
  const panelW = isStacked ? width - 16 : (width - 24) / 2;

  const [lambda, setLambda] = useState(8);
  const [iters, setIters] = useState<IterEntry[]>(() => [
    { beta: [0, 0], obj: lassoObjective([0, 0], 8), coord: -1 },
  ]);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef<number | null>(null);
  const itersRef = useRef(iters);
  itersRef.current = iters;
  const lambdaRef = useRef(lambda);
  lambdaRef.current = lambda;

  // Reset iterates whenever λ changes.
  useEffect(() => {
    setPlaying(false);
    setIters([{ beta: [0, 0], obj: lassoObjective([0, 0], lambda), coord: -1 }]);
  }, [lambda]);

  const advance = (): void => {
    const cur = itersRef.current;
    if (cur.length === 0) return;
    const last = cur[cur.length - 1];
    // Cycle coordinate j = 0, 1, 0, 1, ...
    const nextJ: 0 | 1 = last.coord === 0 ? 1 : 0;
    const nextBeta = cdStep(last.beta, lambdaRef.current, nextJ);
    const obj = lassoObjective(nextBeta, lambdaRef.current);
    setIters((prev) => [...prev, { beta: nextBeta, obj, coord: nextJ }]);
  };

  // Animation loop.
  useEffect(() => {
    if (!playing) return;
    const tick = (): void => {
      const cur = itersRef.current;
      const last = cur[cur.length - 1];
      const prev = cur.length >= 2 ? cur[cur.length - 2] : null;
      if (
        prev &&
        Math.abs(last.beta[0] - prev.beta[0]) < 1e-7 &&
        Math.abs(last.beta[1] - prev.beta[1]) < 1e-7 &&
        cur.length > 4
      ) {
        setPlaying(false);
        return;
      }
      advance();
      playRef.current = window.setTimeout(tick, 280);
    };
    playRef.current = window.setTimeout(tick, 280);
    return () => {
      if (playRef.current !== null) window.clearTimeout(playRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  const reset = (): void => {
    setPlaying(false);
    setIters([{ beta: [0, 0], obj: lassoObjective([0, 0], lambda), coord: -1 }]);
  };

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-xl border bg-white p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-3">
        <div className="text-sm font-semibold text-gray-900">
          Cyclic coordinate descent on a 2D lasso problem (Tseng 2001)
        </div>
        <div className="mt-1 text-xs text-gray-600">
          Each step minimizes the objective along one coordinate via
          soft-thresholding. The path is axis-aligned; the objective is
          monotone-decreasing — the entire convergence content of Thm 6.
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex flex-col text-xs text-gray-700">
          <span>
            λ = <span className="font-mono">{lambda.toFixed(1)}</span>
          </span>
          <input
            type="range"
            min={0.5}
            max={30}
            step={0.5}
            value={lambda}
            onChange={(e) => setLambda(parseFloat(e.target.value))}
            aria-label="lasso lambda"
            className="w-44"
          />
        </label>
        <button
          onClick={() => setPlaying((p) => !p)}
          className="rounded bg-gray-900 px-3 py-1 text-xs font-semibold text-white hover:bg-gray-700"
        >
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <button
          onClick={() => {
            setPlaying(false);
            advance();
          }}
          disabled={playing}
          className="rounded border border-gray-300 bg-white px-3 py-1 text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-40"
        >
          ↪ Step
        </button>
        <button
          onClick={reset}
          className="rounded border border-gray-300 bg-white px-3 py-1 text-xs text-gray-700 hover:bg-gray-50"
        >
          ↺ Reset
        </button>
        <div className="rounded bg-gray-50 px-2 py-1 text-xs text-gray-700">
          step {iters.length - 1} ·{' '}
          <span className="font-mono">
            β = ({iters[iters.length - 1].beta[0].toFixed(3)},{' '}
            {iters[iters.length - 1].beta[1].toFixed(3)})
          </span>
        </div>
      </div>

      <div
        className="flex flex-wrap gap-3"
        style={{ flexDirection: isStacked ? 'column' : 'row' }}
      >
        <TracePanel iters={iters} width={panelW} lambda={lambda} />
        <ObjectivePanel iters={iters} width={panelW} />
      </div>
    </div>
  );
}

interface TraceProps {
  iters: IterEntry[];
  width: number;
  lambda: number;
}

function TracePanel({ iters, width, lambda }: TraceProps) {
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;

  // Plot range based on iterates and a small contour box.
  const all = iters.map((i) => i.beta);
  const xs = all.map((b) => b[0]);
  const ys = all.map((b) => b[1]);
  const xRange = [Math.min(...xs, -0.3), Math.max(...xs, 1.8)];
  const yRange = [Math.min(...ys, -1.0), Math.max(...ys, 0.3)];
  // Pad.
  xRange[0] -= 0.1;
  xRange[1] += 0.1;
  yRange[0] -= 0.1;
  yRange[1] += 0.1;

  const toX = (x: number): number =>
    MARGIN.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * innerW;
  const toY = (y: number): number =>
    MARGIN.top + ((yRange[1] - y) / (yRange[1] - yRange[0])) * innerH;

  // Build contour grid of objective.
  const contourLevels = useMemo(() => {
    const levels: number[] = [];
    const ref = lassoObjective([0, 0], lambda);
    for (let k = 1; k <= 6; k++) levels.push(ref * (0.3 + k * 0.4));
    return levels;
  }, [lambda]);

  const grid = useMemo(() => {
    const nx = 40;
    const ny = 40;
    const cells: { x: number; y: number; v: number }[] = [];
    for (let i = 0; i <= nx; i++) {
      for (let j = 0; j <= ny; j++) {
        const x = xRange[0] + ((xRange[1] - xRange[0]) * i) / nx;
        const y = yRange[0] + ((yRange[1] - yRange[0]) * j) / ny;
        cells.push({ x, y, v: lassoObjective([x, y], lambda) });
      }
    }
    return cells;
  }, [lambda, xRange, yRange]);

  // Find the minimum cell as approximate β̂.
  const minCell = grid.reduce((p, c) => (c.v < p.v ? c : p), grid[0]);

  return (
    <svg width={width} height={PANEL_H} role="img" aria-label="Coord descent trace">
      <text x={MARGIN.left} y={16} fontSize={12} fontWeight={600} fill={penaltyColors.lasso}>
        Iterate path β⁽ᵗ⁾
      </text>

      {/* Background heatmap of objective (subtle). */}
      {grid.map((c, i) => {
        const ratio =
          (c.v - minCell.v) / Math.max(1e-12, contourLevels[contourLevels.length - 1] - minCell.v);
        const a = 1 - Math.min(Math.max(ratio, 0), 1);
        return (
          <rect
            key={`g-${i}`}
            x={toX(c.x) - 0.5}
            y={toY(c.y) - 0.5}
            width={(innerW / 40) * 1.05}
            height={(innerH / 40) * 1.05}
            fill={`rgba(245, 158, 11, ${0.08 * a})`}
          />
        );
      })}

      {/* Axes */}
      <line
        x1={toX(0)}
        x2={toX(0)}
        y1={MARGIN.top}
        y2={MARGIN.top + innerH}
        stroke="#94a3b8"
        strokeWidth={0.8}
      />
      <line
        x1={MARGIN.left}
        x2={MARGIN.left + innerW}
        y1={toY(0)}
        y2={toY(0)}
        stroke="#94a3b8"
        strokeWidth={0.8}
      />

      {/* Iterate trajectory */}
      <path
        d={
          'M ' +
          iters.map((it) => `${toX(it.beta[0]).toFixed(2)},${toY(it.beta[1]).toFixed(2)}`).join(' L ')
        }
        fill="none"
        stroke={penaltyColors.lasso}
        strokeWidth={1.6}
        strokeLinejoin="round"
      />

      {/* Iterate dots */}
      {iters.map((it, i) => (
        <circle
          key={`it-${i}`}
          cx={toX(it.beta[0])}
          cy={toY(it.beta[1])}
          r={i === iters.length - 1 ? 5 : 3}
          fill={i === iters.length - 1 ? '#0f172a' : penaltyColors.lasso}
          stroke="white"
          strokeWidth={1}
        />
      ))}

      {/* Approximate minimizer marker */}
      <circle cx={toX(minCell.x)} cy={toY(minCell.y)} r={4} fill="#10b981" stroke="white" strokeWidth={1.2} />
      <text x={toX(minCell.x) + 7} y={toY(minCell.y) - 6} fontSize={10} fill="#047857">
        β̂_min
      </text>

      {/* Axis labels */}
      <text
        x={MARGIN.left + innerW / 2}
        y={PANEL_H - 8}
        textAnchor="middle"
        fontSize={11}
        fill="#64748b"
      >
        β₁
      </text>
      <text
        x={12}
        y={MARGIN.top + innerH / 2}
        textAnchor="middle"
        fontSize={11}
        fill="#64748b"
        transform={`rotate(-90 12 ${MARGIN.top + innerH / 2})`}
      >
        β₂
      </text>
    </svg>
  );
}

interface ObjProps {
  iters: IterEntry[];
  width: number;
}

function ObjectivePanel({ iters, width }: ObjProps) {
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = PANEL_H - MARGIN.top - MARGIN.bottom;

  const objs = iters.map((i) => i.obj);
  const xs = iters.map((_, i) => i);

  const xRange = [0, Math.max(xs[xs.length - 1] + 0.5, 5)];
  const yRange = [Math.min(...objs) * 0.95, Math.max(...objs) * 1.02];
  const toX = (x: number): number =>
    MARGIN.left + ((x - xRange[0]) / (xRange[1] - xRange[0])) * innerW;
  const toY = (y: number): number =>
    MARGIN.top + ((yRange[1] - y) / (yRange[1] - yRange[0])) * innerH;

  return (
    <svg width={width} height={PANEL_H} role="img" aria-label="Objective monotone descent">
      <text x={MARGIN.left} y={16} fontSize={12} fontWeight={600} fill="#334155">
        Objective f(β⁽ᵗ⁾) (monotone-decreasing)
      </text>

      {/* Trajectory */}
      <path
        d={
          'M ' +
          iters
            .map((it, i) => `${toX(i).toFixed(2)},${toY(it.obj).toFixed(2)}`)
            .join(' L ')
        }
        fill="none"
        stroke="#0f172a"
        strokeWidth={1.6}
      />
      {iters.map((it, i) => (
        <circle
          key={`o-${i}`}
          cx={toX(i)}
          cy={toY(it.obj)}
          r={3}
          fill={
            it.coord === 0
              ? penaltyColors.ridge
              : it.coord === 1
              ? penaltyColors.lasso
              : '#94a3b8'
          }
        />
      ))}

      <line
        x1={MARGIN.left}
        x2={MARGIN.left + innerW}
        y1={MARGIN.top + innerH}
        y2={MARGIN.top + innerH}
        stroke="#94a3b8"
        strokeWidth={0.8}
      />
      <line
        x1={MARGIN.left}
        x2={MARGIN.left}
        y1={MARGIN.top}
        y2={MARGIN.top + innerH}
        stroke="#94a3b8"
        strokeWidth={0.8}
      />

      <text
        x={MARGIN.left + innerW / 2}
        y={PANEL_H - 8}
        textAnchor="middle"
        fontSize={11}
        fill="#64748b"
      >
        coord-descent step t
      </text>
      <text
        x={12}
        y={MARGIN.top + innerH / 2}
        textAnchor="middle"
        fontSize={11}
        fill="#64748b"
        transform={`rotate(-90 12 ${MARGIN.top + innerH / 2})`}
      >
        f(β)
      </text>

      <g transform={`translate(${MARGIN.left + 6}, ${MARGIN.top + 4})`}>
        <text fontSize={10} fill="#64748b">
          dot color: <tspan fill={penaltyColors.ridge}>j=0</tspan> ·{' '}
          <tspan fill={penaltyColors.lasso}>j=1</tspan>
        </text>
      </g>
    </svg>
  );
}
