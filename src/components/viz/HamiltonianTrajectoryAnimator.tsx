/**
 * HamiltonianTrajectoryAnimator — FEATURED interactive for Topic 26 §26.4.
 *
 * Three-panel animation of a single HMC leapfrog trajectory:
 *
 *   (1) Phase-space (left, large)
 *         Target-density contours + full position path + momentum arrow at
 *         the current step. Shows how gradient information carries the
 *         trajectory along ridges where RWM would reject.
 *
 *   (2) Momentum vectors (top-right)
 *         Sequence of (p_x, p_y) arrows over time — momentum's rotational
 *         structure is what lets HMC cover long distances in a single
 *         proposal.
 *
 *   (3) Hamiltonian H(q, p) over step index (bottom-right)
 *         Oscillates as O(ε²) around the initial value (Störmer-Verlet).
 *         Large-ε drift signals divergences; the MH accept step corrects.
 *
 * At the end of a trajectory, the extended-state MH ratio exp(H_start − H_end)
 * is displayed alongside an accept / reject badge. Users can scrub the step
 * slider or press "Animate" to watch the trajectory unroll at 30fps.
 *
 * Brief Appendix G1: the featured figure (26-4) intentionally shows ~0.7
 * energy oscillation on the banana at ε=0.1, L=20. Don't "fix" it — the
 * oscillation is the pedagogical motivation for the MH accept step.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng, hmcLeapfrog } from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import {
  hmcAnimatorPresets,
  type HMCAnimatorPreset,
} from '../../data/bayesian-foundations-data';

const MARGIN = { top: 14, right: 14, bottom: 30, left: 40 };
const MOBILE_BREAKPOINT = 720;
const MAIN_RANGE = [-2.5, 2.5] as [number, number];
const BANANA_X_RANGE = [-2, 2] as [number, number];
const BANANA_Y_RANGE = [-1, 3] as [number, number];
const DONUT_RANGE = [-2, 2] as [number, number];
const FPS = 30;

type TrajectoryStep = {
  q: [number, number];
  p: [number, number];
  H: number;
};

function hamiltonianFor(preset: HMCAnimatorPreset, q: number[], p: number[]): number {
  const U = -preset.logPi(q);
  let K = 0;
  for (const pi of p) K += pi * pi;
  return U + 0.5 * K;
}

function plotRangeFor(preset: HMCAnimatorPreset): {
  x: [number, number];
  y: [number, number];
} {
  if (preset.id === 'banana') return { x: BANANA_X_RANGE, y: BANANA_Y_RANGE };
  if (preset.id === 'donut') return { x: DONUT_RANGE, y: DONUT_RANGE };
  return { x: MAIN_RANGE, y: MAIN_RANGE };
}

function computeTrajectory(
  preset: HMCAnimatorPreset,
  start: [number, number],
  p0: [number, number],
  epsilon: number,
  steps: number,
): TrajectoryStep[] {
  const out: TrajectoryStep[] = [
    { q: [...start], p: [...p0], H: hamiltonianFor(preset, start, p0) },
  ];
  let q: number[] = [...start];
  let p: number[] = [...p0];
  for (let k = 0; k < steps; k++) {
    const { qStar, pStar } = hmcLeapfrog(q, p, preset.gradU, {
      epsilon,
      steps: 1,
    });
    q = qStar;
    p = pStar;
    out.push({
      q: [q[0], q[1]],
      p: [p[0], p[1]],
      H: hamiltonianFor(preset, q, p),
    });
  }
  return out;
}

export default function HamiltonianTrajectoryAnimator() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(1); // banana (featured)
  const [epsilon, setEpsilon] = useState(0.1);
  const [L, setL] = useState(20);
  const [seed, setSeed] = useState(42);
  const [start, setStart] = useState<[number, number]>([-1, 1]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const preset = hmcAnimatorPresets[presetIdx];
  const isMobile = (width ?? 1000) < MOBILE_BREAKPOINT;

  // Resample momentum when seed / preset / start changes via a derived RNG.
  const p0 = useMemo(() => {
    const rng = createSeededRng(seed);
    return [rng.normal(), rng.normal()] as [number, number];
  }, [seed, presetIdx, start]);

  const trajectory = useMemo(
    () => computeTrajectory(preset, start, p0, epsilon, L),
    [preset, start, p0, epsilon, L],
  );

  useEffect(() => {
    setCurrentStep(0);
  }, [preset.id, epsilon, L, start, p0]);

  // ── Animation loop ───────────────────────────────────────────────────────
  const animRef = useRef<number | null>(null);
  useEffect(() => {
    if (!isPlaying) {
      if (animRef.current !== null) {
        window.clearInterval(animRef.current);
        animRef.current = null;
      }
      return;
    }
    animRef.current = window.setInterval(() => {
      setCurrentStep((s) => {
        if (s >= trajectory.length - 1) {
          setIsPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 1000 / FPS);
    return () => {
      if (animRef.current !== null) {
        window.clearInterval(animRef.current);
        animRef.current = null;
      }
    };
  }, [isPlaying, trajectory.length]);

  const handleAnimate = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(true);
  }, []);

  const newTrajectory = useCallback(() => {
    setSeed((s) => s + 1);
    setCurrentStep(0);
  }, []);

  // Panel sizing
  const totalW = (width ?? 960) - 16;
  const mainW = isMobile ? totalW : Math.min(560, totalW * 0.6);
  const sideW = isMobile ? totalW : Math.min(360, totalW * 0.4);
  const mainH = isMobile ? 280 : 360;
  const sideH = isMobile ? 170 : 170;

  const range = plotRangeFor(preset);
  const plotMainW = mainW - MARGIN.left - MARGIN.right;
  const plotMainH = mainH - MARGIN.top - MARGIN.bottom;
  const xScale = (x: number) => ((x - range.x[0]) / (range.x[1] - range.x[0])) * plotMainW;
  const yScale = (y: number) => plotMainH - ((y - range.y[0]) / (range.y[1] - range.y[0])) * plotMainH;

  // ── Main panel: contour heatmap + trajectory + momentum arrow ───────────
  const heatmap = useMemo(() => {
    const nx = 48;
    const ny = 48;
    const cells: { x: number; y: number; p: number }[] = [];
    let maxLp = -Infinity;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const x = range.x[0] + ((i + 0.5) / nx) * (range.x[1] - range.x[0]);
        const y = range.y[0] + ((j + 0.5) / ny) * (range.y[1] - range.y[0]);
        const lp = preset.logPi([x, y]);
        if (lp > maxLp) maxLp = lp;
        cells.push({ x, y, p: lp });
      }
    }
    const cellW = plotMainW / nx;
    const cellH = plotMainH / ny;
    return { cells, maxLp, cellW, cellH };
  }, [preset, range, plotMainW, plotMainH]);

  const visible = trajectory.slice(0, currentStep + 1);
  const pathD = visible
    .map(
      (s, i) =>
        `${i === 0 ? 'M' : 'L'}${xScale(s.q[0]).toFixed(2)},${yScale(s.q[1]).toFixed(2)}`,
    )
    .join(' ');
  const currentStepState = trajectory[currentStep];

  // Momentum arrow: scale p to a visible length
  const arrowLen = 0.35;
  const pNorm = Math.max(
    1e-6,
    Math.hypot(currentStepState.p[0], currentStepState.p[1]),
  );
  const arrowTip: [number, number] = [
    currentStepState.q[0] + (arrowLen * currentStepState.p[0]) / pNorm * 1.5,
    currentStepState.q[1] + (arrowLen * currentStepState.p[1]) / pNorm * 1.5,
  ];

  const onMainClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left - MARGIN.left;
    const my = e.clientY - rect.top - MARGIN.top;
    if (mx < 0 || my < 0 || mx > plotMainW || my > plotMainH) return;
    const x = range.x[0] + (mx / plotMainW) * (range.x[1] - range.x[0]);
    const y = range.y[1] - (my / plotMainH) * (range.y[1] - range.y[0]);
    setStart([x, y]);
  };

  const mainPanel = (
    <svg
      width={mainW}
      height={mainH}
      role="img"
      aria-label="HMC trajectory on target contours"
      onClick={onMainClick}
      style={{ cursor: 'crosshair' }}
    >
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {heatmap.cells.map((c, i) => {
          const opacity = Math.min(0.85, Math.exp(c.p - heatmap.maxLp));
          return (
            <rect
              key={i}
              x={xScale(c.x) - heatmap.cellW / 2}
              y={yScale(c.y) - heatmap.cellH / 2}
              width={heatmap.cellW + 0.5}
              height={heatmap.cellH + 0.5}
              fill={bayesianColors.posterior}
              fillOpacity={opacity * 0.55}
            />
          );
        })}
        <defs>
          <marker
            id="arrow-p"
            viewBox="0 0 10 10"
            refX={10}
            refY={5}
            markerWidth={7}
            markerHeight={7}
            orient="auto"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={bayesianColors.likelihood} />
          </marker>
        </defs>
        <path d={pathD} fill="none" stroke={bayesianColors.chains[0]} strokeWidth={1.5} />
        {visible.map((s, i) => (
          <circle
            key={i}
            cx={xScale(s.q[0])}
            cy={yScale(s.q[1])}
            r={i === visible.length - 1 ? 3 : 1.5}
            fill={
              i === 0
                ? bayesianColors.true
                : i === visible.length - 1
                ? bayesianColors.likelihood
                : bayesianColors.chains[0]
            }
            fillOpacity={0.85}
          />
        ))}
        <line
          x1={xScale(currentStepState.q[0])}
          y1={yScale(currentStepState.q[1])}
          x2={xScale(arrowTip[0])}
          y2={yScale(arrowTip[1])}
          stroke={bayesianColors.likelihood}
          strokeWidth={2}
          markerEnd="url(#arrow-p)"
        />
        <line x1={0} y1={plotMainH} x2={plotMainW} y2={plotMainH} stroke="currentColor" strokeWidth={1} />
        <line x1={0} y1={0} x2={0} y2={plotMainH} stroke="currentColor" strokeWidth={1} />
        {[range.x[0], 0, range.x[1]].map((t) => (
          <g key={`x${t}`} transform={`translate(${xScale(t)},${plotMainH})`}>
            <line y1={0} y2={4} stroke="currentColor" />
            <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {[range.y[0], 0, range.y[1]].map((t) => (
          <g key={`y${t}`} transform={`translate(0,${yScale(t)})`}>
            <line x1={-4} y1={0} x2={0} y2={0} stroke="currentColor" />
            <text x={-8} y={4} textAnchor="end" fontSize={10} fill="currentColor">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={plotMainW - 8} y={14} textAnchor="end" fontSize={10} fill="currentColor" fillOpacity={0.7}>
          click to set starting q
        </text>
      </g>
    </svg>
  );

  // ── Momentum panel ──────────────────────────────────────────────────────
  const plotSideW = sideW - MARGIN.left - MARGIN.right;
  const plotSideH = sideH - MARGIN.top - MARGIN.bottom;
  const pMaxRaw = Math.max(
    1,
    ...trajectory.map((s) => Math.hypot(s.p[0], s.p[1])),
  );
  const pScale = (p: number) => ((p + pMaxRaw) / (2 * pMaxRaw)) * plotSideW;
  const pyScale = (p: number) => plotSideH - ((p + pMaxRaw) / (2 * pMaxRaw)) * plotSideH;

  const momentumPanel = (
    <svg width={sideW} height={sideH} role="img" aria-label="Momentum vectors over trajectory">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <rect x={0} y={0} width={plotSideW} height={plotSideH} fill="none" stroke="currentColor" strokeOpacity={0.3} />
        <line x1={pScale(0)} y1={0} x2={pScale(0)} y2={plotSideH} stroke="currentColor" strokeOpacity={0.25} />
        <line x1={0} y1={pyScale(0)} x2={plotSideW} y2={pyScale(0)} stroke="currentColor" strokeOpacity={0.25} />
        {visible.map((s, i) => (
          <circle
            key={i}
            cx={pScale(s.p[0])}
            cy={pyScale(s.p[1])}
            r={i === visible.length - 1 ? 3 : 1.3}
            fill={
              i === visible.length - 1
                ? bayesianColors.likelihood
                : bayesianColors.chains[1]
            }
            fillOpacity={0.7}
          />
        ))}
        <text x={4} y={12} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          (p_x, p_y)
        </text>
      </g>
    </svg>
  );

  // ── Hamiltonian-over-step panel ─────────────────────────────────────────
  const Hs = trajectory.map((s) => s.H);
  const Hmin = Math.min(...Hs);
  const Hmax = Math.max(...Hs);
  const Hrange = Math.max(Hmax - Hmin, 0.02);

  const hPath = Hs
    .map((h, i) => {
      const tx = (i / (Hs.length - 1)) * plotSideW;
      const ty = plotSideH - ((h - Hmin) / Hrange) * plotSideH;
      return `${i === 0 ? 'M' : 'L'}${tx.toFixed(2)},${ty.toFixed(2)}`;
    })
    .join(' ');

  const hmlPanel = (
    <svg width={sideW} height={sideH} role="img" aria-label="Hamiltonian H over step index">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <rect x={0} y={0} width={plotSideW} height={plotSideH} fill="none" stroke="currentColor" strokeOpacity={0.3} />
        <line
          x1={0}
          y1={plotSideH - ((Hs[0] - Hmin) / Hrange) * plotSideH}
          x2={plotSideW}
          y2={plotSideH - ((Hs[0] - Hmin) / Hrange) * plotSideH}
          stroke={bayesianColors.true}
          strokeDasharray="4 2"
        />
        <path d={hPath} fill="none" stroke={bayesianColors.posterior} strokeWidth={1.6} />
        <line
          x1={(currentStep / (Hs.length - 1)) * plotSideW}
          y1={0}
          x2={(currentStep / (Hs.length - 1)) * plotSideW}
          y2={plotSideH}
          stroke={bayesianColors.likelihood}
          strokeWidth={1}
        />
        <text x={4} y={12} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          H(q, p) · |ΔH|_max = {(Hmax - Hmin).toFixed(3)}
        </text>
      </g>
    </svg>
  );

  const dH = trajectory[trajectory.length - 1].H - trajectory[0].H;
  const mhRatio = Math.exp(-dH);
  const accepted =
    currentStep >= trajectory.length - 1 &&
    mhRatio > createSeededRng(seed + 10_000).random();

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="font-medium">Target:</span>
          <select
            className="rounded border px-2 py-1"
            value={presetIdx}
            onChange={(e) => setPresetIdx(Number(e.target.value))}
            aria-label="HMC target"
          >
            {hmcAnimatorPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">ε =</span>
          <input
            type="range"
            min={0.01}
            max={0.3}
            step={0.005}
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            className="w-28"
            aria-label="Leapfrog step size"
            aria-valuetext={`ε = ${epsilon.toFixed(3)}`}
          />
          <span className="font-mono">{epsilon.toFixed(3)}</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">L =</span>
          <input
            type="range"
            min={5}
            max={60}
            step={1}
            value={L}
            onChange={(e) => setL(Number(e.target.value))}
            className="w-28"
            aria-label="Number of leapfrog steps"
            aria-valuetext={`L = ${L}`}
          />
          <span className="font-mono">{L}</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">Seed:</span>
          <input
            type="number"
            className="w-16 rounded border px-2 py-1"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            aria-label="RNG seed"
          />
        </label>
        <div className="ml-auto flex gap-2">
          <button
            type="button"
            onClick={handleAnimate}
            className="rounded bg-slate-200 px-3 py-1 text-sm hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600"
          >
            {isPlaying ? 'Playing…' : 'Animate'}
          </button>
          <button
            type="button"
            onClick={newTrajectory}
            className="rounded border px-3 py-1 text-sm hover:bg-slate-50 dark:hover:bg-slate-800"
          >
            New trajectory
          </button>
        </div>
      </div>

      <div className="mb-3 flex items-center gap-3 text-sm">
        <span className="font-medium">step {currentStep} / {trajectory.length - 1}</span>
        <input
          type="range"
          min={0}
          max={trajectory.length - 1}
          step={1}
          value={currentStep}
          onChange={(e) => {
            setIsPlaying(false);
            setCurrentStep(Number(e.target.value));
          }}
          className="flex-1"
          aria-label="Trajectory step"
          aria-valuetext={`step ${currentStep} of ${trajectory.length - 1}`}
        />
        <span className="font-mono text-xs">
          H₀ = {trajectory[0].H.toFixed(3)} &middot; H = {currentStepState.H.toFixed(3)}
        </span>
      </div>

      <div
        className={`grid gap-3 ${
          isMobile ? 'grid-cols-1' : 'grid-cols-[1fr_minmax(240px,360px)]'
        }`}
      >
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            (1) phase space — target + trajectory + momentum arrow
          </div>
          {mainPanel}
        </div>
        <div className="flex flex-col gap-3">
          <div>
            <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
              (2) momentum (p_x, p_y)
            </div>
            {momentumPanel}
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
              (3) H(q, p) over step
            </div>
            {hmlPanel}
          </div>
        </div>
      </div>

      <div className="mt-3 text-sm">
        {currentStep >= trajectory.length - 1 ? (
          <span>
            <span className="font-medium">End of trajectory.</span>{' '}
            MH ratio α = exp(H_start − H_end) = {Math.min(1, mhRatio).toFixed(3)}{' '}
            <span
              className="ml-2 rounded px-2 py-0.5 font-mono text-xs"
              style={{
                background: accepted
                  ? bayesianColors.true
                  : bayesianColors.mle,
                color: '#fff',
              }}
            >
              {accepted ? 'ACCEPT' : 'REJECT'}
            </span>
          </span>
        ) : (
          <span className="text-gray-600 dark:text-gray-400">
            Leapfrog integrator running — step {currentStep} of {trajectory.length - 1}.
          </span>
        )}
      </div>

      <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
        {preset.description} Click anywhere on the main panel to reset the starting position.
      </p>
    </div>
  );
}
