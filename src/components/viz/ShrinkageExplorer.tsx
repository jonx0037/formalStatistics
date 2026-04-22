/**
 * ShrinkageExplorer — featured interactive for Topic 28 §28.5 (Stein's
 * paradox). Closed-form arithmetic plus one small seeded Monte Carlo
 * ("Run 500 samples") that shows MLE risk plateau at d while James–Stein
 * and Empirical-Bayes partial-pool risks converge strictly below d.
 *
 * Interaction model:
 *   — K slider (3–10): dimensionality / number of group means.
 *   — K draggable markers on a [−5, 5] number line: true θ_k.
 *   — Live estimators: for the current random sample X ~ 𝒩_K(θ, I),
 *     show MLE = X, James–Stein (bayes.ts), and EB partial-pool using
 *     typeIIMLE to learn (μ̂, τ̂²) from the data.
 *   — "Draw sample" reseeds X; "Run 500 samples" accumulates empirical
 *     risk estimates and plots them as convergence curves (Jon's
 *     preferred readout — shows Stein's shock viscerally).
 *
 * Palette tokens: bayesianColors.true / .mle / .shrink / .posterior.
 */
import { useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  jamesSteinEstimator,
  partialPoolingPosteriorMean,
  typeIIMLE,
  createSeededRng,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import { shrinkageExplorerPresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 24, right: 24, bottom: 40, left: 32 };
const MOBILE_BREAKPOINT = 640;
const THETA_MIN = -5;
const THETA_MAX = 5;
const K_MIN = 3;
const K_MAX = 10;
const RESAMPLES = 500;

type Estimators = {
  mle: number[];
  js: number[];
  pp: number[];
};

// Draw X ~ N(theta, I) using an injected RNG. Shared between live preview
// and the 500-sample accumulator so both honor the seed.
function drawSample(theta: number[], rng: ReturnType<typeof createSeededRng>): number[] {
  return theta.map((tk) => tk + rng.normal());
}

function computeEstimators(x: number[]): Estimators {
  const mle = [...x];
  const js = jamesSteinEstimator(x);
  const sigmaSq = new Array(x.length).fill(1);
  const { mu, tauSq } = typeIIMLE(x, sigmaSq);
  const pp = x.map((xk) => partialPoolingPosteriorMean(xk, mu, 1, tauSq));
  return { mle, js, pp };
}

function sse(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

export default function ShrinkageExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState<number | null>(0);
  const [K, setK] = useState(5);
  const [theta, setTheta] = useState<number[]>([...shrinkageExplorerPresets[0].theta]);
  const [seed, setSeed] = useState(42);
  // Accumulated per-sample MSE for the running convergence chart.
  const [mseHistory, setMseHistory] = useState<{ mle: number[]; js: number[]; pp: number[] }>({
    mle: [],
    js: [],
    pp: [],
  });
  const draggingRef = useRef<number | null>(null);

  const applyPreset = (idx: number) => {
    const p = shrinkageExplorerPresets[idx];
    setPresetIdx(idx);
    setK(p.K);
    setTheta([...p.theta]);
    setMseHistory({ mle: [], js: [], pp: [] });
  };

  const updateTheta = (i: number, newVal: number) => {
    const clamped = Math.min(THETA_MAX, Math.max(THETA_MIN, newVal));
    setTheta((prev) => {
      const next = [...prev];
      next[i] = clamped;
      return next;
    });
    setPresetIdx(null);
    setMseHistory({ mle: [], js: [], pp: [] });
  };

  const changeK = (newK: number) => {
    const k = Math.min(K_MAX, Math.max(K_MIN, newK));
    setK(k);
    setTheta((prev) => {
      if (prev.length === k) return prev;
      if (prev.length < k) return [...prev, ...new Array(k - prev.length).fill(0)];
      return prev.slice(0, k);
    });
    setPresetIdx(null);
    setMseHistory({ mle: [], js: [], pp: [] });
  };

  // Current preview sample (reactive to seed + theta). We memoize on a
  // stringified theta so React sees value-equality rather than reference.
  const thetaKey = theta.join(',');
  const { x, estimators } = useMemo(() => {
    const rng = createSeededRng(seed);
    const xDraw = drawSample(theta, rng);
    return { x: xDraw, estimators: computeEstimators(xDraw) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seed, thetaKey, K]);

  const runResamples = () => {
    // Start a fresh MC accumulator seeded deterministically from the current seed + theta hash.
    const thetaHash = theta.reduce((acc, t) => acc * 31 + Math.round(t * 1000), 0);
    const rng = createSeededRng(seed + thetaHash);
    const mle: number[] = [];
    const js: number[] = [];
    const pp: number[] = [];
    let cumMle = 0,
      cumJs = 0,
      cumPp = 0;
    for (let n = 0; n < RESAMPLES; n++) {
      const xn = drawSample(theta, rng);
      const est = computeEstimators(xn);
      cumMle += sse(est.mle, theta);
      cumJs += sse(est.js, theta);
      cumPp += sse(est.pp, theta);
      mle.push(cumMle / (n + 1));
      js.push(cumJs / (n + 1));
      pp.push(cumPp / (n + 1));
    }
    setMseHistory({ mle, js, pp });
  };

  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(720, (width ?? 720) - 16));
  const lineH = isMobile ? 160 : 200;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = lineH - MARGIN.top - MARGIN.bottom;

  const xScale = (val: number) =>
    ((val - THETA_MIN) / (THETA_MAX - THETA_MIN)) * plotW;
  const trueY = plotH * 0.25;
  const estY = plotH * 0.70;

  const currentMse = useMemo(() => ({
    mle: sse(estimators.mle, theta),
    js: sse(estimators.js, theta),
    pp: sse(estimators.pp, theta),
  }), [estimators, theta]);

  // Pointer drag: map svg-coord mouse X back into θ space.
  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (draggingRef.current === null) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const svgX = e.clientX - rect.left - MARGIN.left;
    const frac = svgX / plotW;
    const newTheta = THETA_MIN + frac * (THETA_MAX - THETA_MIN);
    updateTheta(draggingRef.current, newTheta);
  };

  // Convergence-chart projection (MSE history).
  const hasHistory = mseHistory.mle.length > 0;
  const histW = Math.max(260, Math.min(560, (width ?? 600) - 16));
  const histH = 180;
  const histPlotW = histW - MARGIN.left - MARGIN.right;
  const histPlotH = histH - MARGIN.top - MARGIN.bottom;

  const mseMax = hasHistory
    ? Math.max(
        ...mseHistory.mle,
        ...mseHistory.js,
        ...mseHistory.pp,
        K + 1,
      )
    : K + 1;
  const mseMin = 0;

  const mseX = (i: number) => (i / (RESAMPLES - 1)) * histPlotW;
  const mseY = (v: number) => histPlotH * (1 - (v - mseMin) / (mseMax - mseMin));

  const historyPath = (ys: number[]) =>
    ys
      .map((v, i) => `${i === 0 ? 'M' : 'L'} ${mseX(i).toFixed(1)} ${mseY(v).toFixed(1)}`)
      .join(' ');

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Preset row */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {shrinkageExplorerPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => applyPreset(i)}
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

      {/* K + seed controls */}
      <div className="mb-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Dimensions K (= d)</span>
            <span className="font-mono">{K}</span>
          </span>
          <input
            type="range"
            min={K_MIN}
            max={K_MAX}
            step={1}
            value={K}
            onChange={(e) => changeK(parseInt(e.target.value))}
            className="accent-[var(--color-shrink)]"
          />
        </label>
        <div className="flex items-end gap-2">
          <button
            onClick={() => {
              setSeed((s) => s + 1);
              setMseHistory({ mle: [], js: [], pp: [] });
            }}
            className="rounded px-3 py-1.5 text-xs font-medium text-white transition"
            style={{ background: bayesianColors.mle }}
          >
            Draw new sample
          </button>
          <button
            onClick={runResamples}
            className="rounded px-3 py-1.5 text-xs font-medium text-white transition"
            style={{ background: bayesianColors.shrink }}
          >
            Run {RESAMPLES} samples
          </button>
        </div>
      </div>

      {/* Number line with draggable θ + current estimators */}
      <svg
        width={chartW}
        height={lineH}
        className="block touch-none"
        onPointerMove={handlePointerMove}
        onPointerUp={() => (draggingRef.current = null)}
        onPointerLeave={() => (draggingRef.current = null)}
      >
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Axis baseline */}
          <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="var(--color-text-muted)" />
          {/* Origin marker */}
          <line
            x1={xScale(0)}
            y1={0}
            x2={xScale(0)}
            y2={plotH}
            stroke="var(--color-border)"
            strokeDasharray="3 3"
          />
          <text
            x={xScale(0)}
            y={-6}
            textAnchor="middle"
            fontSize={10}
            fill="var(--color-text-muted)"
          >
            origin (JS shrinks toward this)
          </text>
          {/* Ticks */}
          {[-4, -2, 0, 2, 4].map((t) => (
            <g key={t}>
              <line
                x1={xScale(t)}
                y1={plotH}
                x2={xScale(t)}
                y2={plotH + 4}
                stroke="var(--color-text-muted)"
              />
              <text
                x={xScale(t)}
                y={plotH + 16}
                textAnchor="middle"
                fontSize={10}
                fill="var(--color-text-muted)"
              >
                {t}
              </text>
            </g>
          ))}
          {/* X→JS arrows */}
          {x.map((xk, i) => (
            <line
              key={`js-${i}`}
              x1={xScale(xk)}
              y1={estY}
              x2={xScale(estimators.js[i])}
              y2={estY}
              stroke={bayesianColors.shrink}
              strokeWidth={1.5}
              opacity={0.4}
            />
          ))}
          {/* Estimator markers */}
          {x.map((xk, i) => (
            <g key={`est-${i}`}>
              <circle cx={xScale(xk)} cy={estY} r={4} fill={bayesianColors.mle} opacity={0.8} />
              <circle
                cx={xScale(estimators.js[i])}
                cy={estY + 8}
                r={3.5}
                fill={bayesianColors.shrink}
                opacity={0.85}
              />
              <circle
                cx={xScale(estimators.pp[i])}
                cy={estY + 16}
                r={3.5}
                fill={bayesianColors.posterior}
                opacity={0.85}
              />
            </g>
          ))}
          {/* True θ markers (draggable diamonds) */}
          {theta.map((tk, i) => (
            <g key={`theta-${i}`}>
              <polygon
                points={`${xScale(tk)},${trueY - 7} ${xScale(tk) + 6},${trueY} ${xScale(tk)},${trueY + 7} ${xScale(tk) - 6},${trueY}`}
                fill={bayesianColors.true}
                stroke="white"
                strokeWidth={1.5}
                style={{ cursor: 'ew-resize' }}
                onPointerDown={(e) => {
                  draggingRef.current = i;
                  e.currentTarget.setPointerCapture(e.pointerId);
                }}
              />
              <text
                x={xScale(tk)}
                y={trueY - 12}
                textAnchor="middle"
                fontSize={9}
                fill="var(--color-text-muted)"
              >
                θ_{i + 1}
              </text>
            </g>
          ))}
          {/* Legend */}
          <g transform={`translate(0, ${plotH + 28})`} fontSize={10}>
            <g>
              <polygon
                points="-5,-4 0,0 -5,4 -10,0"
                transform="translate(12, 0)"
                fill={bayesianColors.true}
              />
              <text x={22} y={4} fill="var(--color-text)">
                θ (true, drag me)
              </text>
            </g>
            <g transform="translate(140, 0)">
              <circle cx={6} cy={0} r={4} fill={bayesianColors.mle} />
              <text x={16} y={4} fill="var(--color-text)">
                MLE = X
              </text>
            </g>
            <g transform={`translate(${isMobile ? 0 : 240}, ${isMobile ? 16 : 0})`}>
              <circle cx={6} cy={0} r={3.5} fill={bayesianColors.shrink} />
              <text x={16} y={4} fill="var(--color-text)">
                James–Stein
              </text>
            </g>
            <g transform={`translate(${isMobile ? 130 : 360}, ${isMobile ? 16 : 0})`}>
              <circle cx={6} cy={0} r={3.5} fill={bayesianColors.posterior} />
              <text x={16} y={4} fill="var(--color-text)">
                Partial-pool (EB)
              </text>
            </g>
          </g>
        </g>
      </svg>

      {/* Current-sample readout */}
      <div className="mt-2 grid grid-cols-3 gap-3 text-sm">
        <div className="rounded border p-2" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            MLE SSE
          </div>
          <div className="font-mono text-base" style={{ color: bayesianColors.mle }}>
            {currentMse.mle.toFixed(2)}
          </div>
        </div>
        <div className="rounded border p-2" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            James–Stein SSE
          </div>
          <div className="font-mono text-base" style={{ color: bayesianColors.shrink }}>
            {currentMse.js.toFixed(2)}
          </div>
        </div>
        <div className="rounded border p-2" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Partial-pool SSE
          </div>
          <div className="font-mono text-base" style={{ color: bayesianColors.posterior }}>
            {currentMse.pp.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Convergence chart */}
      {hasHistory && (
        <div className="mt-4">
          <div className="mb-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Running empirical risk over {RESAMPLES} draws of X ~ 𝒩_K(θ, I). MLE risk
            plateaus at d = K; James–Stein and partial-pool converge strictly below.
          </div>
          <svg width={histW} height={histH} className="block">
            <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
              {/* Reference line at y = K */}
              <line
                x1={0}
                y1={mseY(K)}
                x2={histPlotW}
                y2={mseY(K)}
                stroke="var(--color-text-muted)"
                strokeDasharray="3 3"
              />
              <text
                x={histPlotW - 2}
                y={mseY(K) - 3}
                textAnchor="end"
                fontSize={9}
                fill="var(--color-text-muted)"
              >
                d = {K}
              </text>
              {/* Risk lines */}
              <path
                d={historyPath(mseHistory.mle)}
                fill="none"
                stroke={bayesianColors.mle}
                strokeWidth={1.8}
              />
              <path
                d={historyPath(mseHistory.js)}
                fill="none"
                stroke={bayesianColors.shrink}
                strokeWidth={1.8}
              />
              <path
                d={historyPath(mseHistory.pp)}
                fill="none"
                stroke={bayesianColors.posterior}
                strokeWidth={1.8}
              />
              {/* Axis labels */}
              <text
                x={histPlotW / 2}
                y={histPlotH + 26}
                textAnchor="middle"
                fontSize={11}
                fill="var(--color-text)"
              >
                Samples drawn
              </text>
              <text
                x={-24}
                y={histPlotH / 2}
                textAnchor="middle"
                fontSize={11}
                fill="var(--color-text)"
                transform={`rotate(-90 ${-24} ${histPlotH / 2})`}
              >
                Empirical risk
              </text>
              {/* y-axis ticks */}
              {[0, K / 2, K, mseMax].map((v) => (
                <text
                  key={v}
                  x={-6}
                  y={mseY(v) + 3}
                  textAnchor="end"
                  fontSize={9}
                  fill="var(--color-text-muted)"
                >
                  {v.toFixed(1)}
                </text>
              ))}
            </g>
          </svg>
          <div className="mt-2 grid grid-cols-3 gap-3 text-xs">
            <div style={{ color: bayesianColors.mle }}>
              MLE risk: <span className="font-mono">{mseHistory.mle.at(-1)?.toFixed(3)}</span>
            </div>
            <div style={{ color: bayesianColors.shrink }}>
              JS risk: <span className="font-mono">{mseHistory.js.at(-1)?.toFixed(3)}</span>
              <span className="ml-1" style={{ color: 'var(--color-text-muted)' }}>
                ({(
                  (1 - (mseHistory.js.at(-1) ?? 0) / (mseHistory.mle.at(-1) || 1)) *
                  100
                ).toFixed(1)}
                % better)
              </span>
            </div>
            <div style={{ color: bayesianColors.posterior }}>
              PP risk: <span className="font-mono">{mseHistory.pp.at(-1)?.toFixed(3)}</span>
              <span className="ml-1" style={{ color: 'var(--color-text-muted)' }}>
                ({(
                  (1 - (mseHistory.pp.at(-1) ?? 0) / (mseHistory.mle.at(-1) || 1)) *
                  100
                ).toFixed(1)}
                % better)
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
