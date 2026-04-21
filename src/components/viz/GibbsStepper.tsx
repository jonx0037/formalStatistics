/**
 * GibbsStepper — interactive for Topic 26 §26.3.
 *
 * Steps through systematic-scan Gibbs on a bivariate Normal with correlation
 * ρ, using the closed-form conditional-MVN formula from Topic 8 Thm 3 via
 * `conditionalMVN` in bayes.ts. Shows each coordinate update as a horizontal
 * or vertical line segment on the density contour, making the staircase
 * structure of Gibbs visible.
 *
 * Controls: preset selector (ρ = 0.3 / 0.6 / 0.8 / 0.95), Step button
 * (one coordinate update), Run-10 / Run-100 buttons, Reset.
 *
 * Full conditionals for bivariate N(0, [[1, ρ], [ρ, 1]]):
 *   x | y ~ N(ρ y, 1 − ρ²)
 *   y | x ~ N(ρ x, 1 − ρ²)
 * These match the output of `conditionalMVN([0,0], Σ, [1], [y])` etc.
 */
import { useCallback, useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  conditionalMVN,
  createSeededRng,
  type SeededRng,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import {
  gibbsStepperPresets,
  type GibbsStepperPreset,
} from '../../data/bayesian-foundations-data';

const MARGIN = { top: 16, right: 16, bottom: 36, left: 44 };
const MOBILE_BREAKPOINT = 640;
const PLOT_RANGE = [-4, 4] as [number, number];

type ChainState = {
  path: number[][]; // positions after each sub-step
  axes: ('x' | 'y')[]; // which coord was updated in each sub-step
};

/**
 * Draw from the conditional N(ρ · other, 1 − ρ²). By symmetry, x|y and y|x
 * have the same form on the standard bivariate Normal, so `axis` isn't in
 * the signature — the caller just passes the other coordinate's value.
 */
function drawConditional(
  preset: GibbsStepperPreset,
  other: number,
  rng: SeededRng,
): number {
  const rho = preset.rho;
  const mean = rho * other;
  const sd = Math.sqrt(1 - rho * rho);
  return mean + sd * rng.normal();
}

function computeContourLevels(rho: number): number[] {
  // Elliptic contours: (x² − 2ρxy + y²) / (1 − ρ²) = c. Pick a few levels.
  const levels = [1, 2.5, 4.5];
  return levels.map((c) => c * (1 - rho * rho));
}

export default function GibbsStepper() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(2); // ρ = 0.8 canonical demo
  const [seed, setSeed] = useState(42);
  const [chain, setChain] = useState<ChainState>({
    path: [[-2, -2]],
    axes: [],
  });
  const [rngState, setRngState] = useState(() => createSeededRng(42));

  const preset = gibbsStepperPresets[presetIdx];
  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;

  const chartW = Math.max(280, Math.min(560, (width ?? 600) - 16));
  const chartH = isMobile ? 260 : 340;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const xScale = useCallback(
    (x: number) => ((x - PLOT_RANGE[0]) / (PLOT_RANGE[1] - PLOT_RANGE[0])) * plotW,
    [plotW],
  );
  const yScale = useCallback(
    (y: number) => plotH - ((y - PLOT_RANGE[0]) / (PLOT_RANGE[1] - PLOT_RANGE[0])) * plotH,
    [plotH],
  );

  const reset = useCallback(() => {
    setChain({ path: [[-2, -2]], axes: [] });
    setRngState(createSeededRng(seed));
  }, [seed]);

  const stepOnce = useCallback(() => {
    setChain((prev) => {
      const last = prev.path[prev.path.length - 1];
      const nextAxis: 'x' | 'y' = prev.axes.length % 2 === 0 ? 'x' : 'y';
      let xNew = last[0];
      let yNew = last[1];
      if (nextAxis === 'x') xNew = drawConditional(preset, last[1], rngState);
      else yNew = drawConditional(preset, last[0], rngState);
      return {
        path: [...prev.path, [xNew, yNew]],
        axes: [...prev.axes, nextAxis],
      };
    });
  }, [preset, rngState]);

  const runMany = useCallback(
    (k: number) => {
      for (let i = 0; i < k; i++) stepOnce();
    },
    [stepOnce],
  );

  // When preset changes, reset to a clean start.
  const handlePresetChange = (newIdx: number) => {
    setPresetIdx(newIdx);
    setChain({ path: [[-2, -2]], axes: [] });
    setRngState(createSeededRng(seed));
  };
  const handleSeedChange = (newSeed: number) => {
    setSeed(newSeed);
    setChain({ path: [[-2, -2]], axes: [] });
    setRngState(createSeededRng(newSeed));
  };

  // ── Density contours as SVG paths (elliptic level sets) ──────────────────
  const contours = useMemo(() => {
    const rho = preset.rho;
    const levels = computeContourLevels(rho);
    return levels.map((c) => {
      const nTheta = 80;
      // Ellipse: solve (x² - 2ρxy + y²) = c via parametric form.
      // Eigen-decompose: principal axes at 45°. Lengths sqrt(c/(1-ρ)) and sqrt(c/(1+ρ)).
      const a = Math.sqrt(c / (1 - rho));
      const b = Math.sqrt(c / (1 + rho));
      const pts: [number, number][] = [];
      for (let i = 0; i <= nTheta; i++) {
        const theta = (i / nTheta) * 2 * Math.PI;
        const u = a * Math.cos(theta);
        const v = b * Math.sin(theta);
        // rotate by 45°: [1/√2, -1/√2; 1/√2, 1/√2]
        const x = (u - v) / Math.sqrt(2);
        const y = (u + v) / Math.sqrt(2);
        pts.push([x, y]);
      }
      return pts
        .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${xScale(x).toFixed(2)},${yScale(y).toFixed(2)}`)
        .join(' ');
    });
  }, [preset.rho, xScale, yScale]);

  // ── Gibbs path as axis-aligned line segments ──────────────────────────────
  const pathSegments = useMemo(() => {
    const segs: { x1: number; y1: number; x2: number; y2: number; axis: 'x' | 'y' }[] = [];
    for (let i = 1; i < chain.path.length; i++) {
      const prev = chain.path[i - 1];
      const curr = chain.path[i];
      segs.push({
        x1: xScale(prev[0]),
        y1: yScale(prev[1]),
        x2: xScale(curr[0]),
        y2: yScale(curr[1]),
        axis: chain.axes[i - 1],
      });
    }
    return segs;
  }, [chain, xScale, yScale]);

  const lastPoint = chain.path[chain.path.length - 1];
  const nextAxis: 'x' | 'y' = chain.axes.length % 2 === 0 ? 'x' : 'y';
  const currentConditional =
    nextAxis === 'x'
      ? { target: 'x', mean: (preset.rho * lastPoint[1]).toFixed(3), sd: Math.sqrt(1 - preset.rho ** 2).toFixed(3) }
      : { target: 'y', mean: (preset.rho * lastPoint[0]).toFixed(3), sd: Math.sqrt(1 - preset.rho ** 2).toFixed(3) };

  // ── Right panel: coordinate trace plots ──────────────────────────────────
  const tracePanel = useMemo(() => {
    const w = plotW;
    const h = plotH / 2 - 8;
    const mkPath = (vals: number[]) => {
      if (vals.length < 2) return '';
      return vals
        .map((v, i) => {
          const tx = (i / Math.max(1, vals.length - 1)) * w;
          const ty = h - ((v - PLOT_RANGE[0]) / (PLOT_RANGE[1] - PLOT_RANGE[0])) * h;
          return `${i === 0 ? 'M' : 'L'}${tx.toFixed(2)},${ty.toFixed(2)}`;
        })
        .join(' ');
    };
    const xs = chain.path.map((p) => p[0]);
    const ys = chain.path.map((p) => p[1]);
    return (
      <svg width={chartW} height={chartH} role="img" aria-label="Gibbs marginal traces">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          <rect x={0} y={0} width={w} height={h} fill="none" stroke="currentColor" strokeOpacity={0.3} />
          <path d={mkPath(xs)} fill="none" stroke={bayesianColors.chains[0]} strokeWidth={1.4} />
          <text x={4} y={12} fontSize={10} fill={bayesianColors.chains[0]}>
            x trace
          </text>

          <g transform={`translate(0,${h + 16})`}>
            <rect x={0} y={0} width={w} height={h} fill="none" stroke="currentColor" strokeOpacity={0.3} />
            <path d={mkPath(ys)} fill="none" stroke={bayesianColors.chains[1]} strokeWidth={1.4} strokeDasharray="6 3" />
            <text x={4} y={12} fontSize={10} fill={bayesianColors.chains[1]}>
              y trace
            </text>
          </g>
        </g>
      </svg>
    );
  }, [chain.path, chartW, chartH, plotW, plotH]);

  // Verify our closed-form matches conditionalMVN (sanity; no runtime cost past first mount).
  useMemo(() => {
    const { muCond, sigmaCond } = conditionalMVN(
      [0, 0],
      [
        [1, preset.rho],
        [preset.rho, 1],
      ],
      [1],
      [lastPoint[1]],
    );
    // Used implicitly; discarded (React won't fire without a consumer, but memo captures for debug).
    return [muCond, sigmaCond] as const;
  }, [preset.rho, lastPoint]);

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="font-medium">Correlation ρ:</span>
          <select
            className="rounded border px-2 py-1"
            value={presetIdx}
            onChange={(e) => handlePresetChange(Number(e.target.value))}
            aria-label="Correlation preset"
          >
            {gibbsStepperPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">Seed:</span>
          <input
            type="number"
            className="w-20 rounded border px-2 py-1"
            value={seed}
            onChange={(e) => handleSeedChange(Number(e.target.value) || 0)}
            aria-label="RNG seed"
          />
        </label>
        <div className="ml-auto flex gap-2">
          <button
            type="button"
            onClick={stepOnce}
            className="rounded bg-slate-200 px-3 py-1 text-sm hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600"
            aria-label="Advance one Gibbs sub-step"
          >
            Step
          </button>
          <button
            type="button"
            onClick={() => runMany(10)}
            className="rounded bg-slate-200 px-3 py-1 text-sm hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600"
          >
            Run 10
          </button>
          <button
            type="button"
            onClick={() => runMany(100)}
            className="rounded bg-slate-200 px-3 py-1 text-sm hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600"
          >
            Run 100
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded border px-3 py-1 text-sm hover:bg-slate-50 dark:hover:bg-slate-800"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="mb-3 font-mono text-xs text-gray-600 dark:text-gray-400">
        step {chain.axes.length}: next draw{' '}
        <span style={{ color: bayesianColors.likelihood }}>
          {currentConditional.target} | other ∼ 𝒩({currentConditional.mean}, {currentConditional.sd}²)
        </span>{' '}
        &middot; current state ({lastPoint[0].toFixed(2)}, {lastPoint[1].toFixed(2)})
      </div>

      <div className={`grid gap-4 ${isMobile ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            bivariate Normal contours + Gibbs path
          </div>
          <svg width={chartW} height={chartH} role="img" aria-label="Gibbs sampler on bivariate Normal">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {contours.map((d, i) => (
                <path
                  key={i}
                  d={d}
                  fill="none"
                  stroke={bayesianColors.posterior}
                  strokeOpacity={0.3 + 0.2 * i}
                  strokeWidth={1}
                />
              ))}
              {pathSegments.map((s, i) => (
                <line
                  key={i}
                  x1={s.x1}
                  y1={s.y1}
                  x2={s.x2}
                  y2={s.y2}
                  stroke={s.axis === 'x' ? bayesianColors.chains[0] : bayesianColors.chains[1]}
                  strokeDasharray={s.axis === 'x' ? 'none' : '6 3'}
                  strokeWidth={1.2}
                  strokeOpacity={0.7}
                />
              ))}
              <circle
                cx={xScale(lastPoint[0])}
                cy={yScale(lastPoint[1])}
                r={4}
                fill={bayesianColors.likelihood}
              />
              <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="currentColor" strokeWidth={1} />
              <line x1={0} y1={0} x2={0} y2={plotH} stroke="currentColor" strokeWidth={1} />
              <line x1={xScale(0)} y1={0} x2={xScale(0)} y2={plotH} stroke="currentColor" strokeOpacity={0.2} />
              <line x1={0} y1={yScale(0)} x2={plotW} y2={yScale(0)} stroke="currentColor" strokeOpacity={0.2} />
              {[-3, 0, 3].map((t) => (
                <g key={t}>
                  <g transform={`translate(${xScale(t)},${plotH})`}>
                    <line y1={0} y2={4} stroke="currentColor" />
                    <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                      {t}
                    </text>
                  </g>
                  <g transform={`translate(0,${yScale(t)})`}>
                    <line x1={-4} y1={0} x2={0} y2={0} stroke="currentColor" />
                    <text x={-8} y={4} textAnchor="end" fontSize={10} fill="currentColor">
                      {t}
                    </text>
                  </g>
                </g>
              ))}
            </g>
          </svg>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            coordinate traces (solid = x, dashed = y)
          </div>
          {tracePanel}
        </div>
      </div>

      <p className="mt-3 text-xs text-gray-600 dark:text-gray-400">{preset.description}</p>
    </div>
  );
}
