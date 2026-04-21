/**
 * MetropolisHastingsTuner — interactive for Topic 26 §26.2.
 *
 * Random-walk Metropolis on a user-selected target, with a proposal-scale
 * slider that the user tunes until they hit the Roberts-Gelman-Gilks 1997
 * 0.234 / 0.44 (d=1 Gaussian) optimum. Left panel: target density + live
 * chain path. Right panel: empirical acceptance-rate-vs-scale curve with
 * the current slider position highlighted.
 *
 * Targets (preset-driven): standard Normal, Rosenbrock banana (d=2),
 * bimodal mixture, Student-t(ν=3). All operate on the same MH driver
 * (`metropolisHastings` from bayes.ts) with a symmetric Gaussian proposal.
 */
import { useDeferredValue, useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  createSeededRng,
  metropolisHastings,
  type ProposalKernel,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import {
  mhTunerPresets,
  type MHTunerPreset,
} from '../../data/bayesian-foundations-data';

const MARGIN = { top: 16, right: 16, bottom: 36, left: 44 };
const MOBILE_BREAKPOINT = 640;
const DEFAULT_N = 5000;
const DEFAULT_BURN_IN = 500;
const ACCEPT_CURVE_POINTS = 24;
const ACCEPT_CURVE_N = 1200;

const LOG_SCALE_MIN = Math.log(0.05);
const LOG_SCALE_MAX = Math.log(10);

const sliderToScale = (s: number) => Math.exp(LOG_SCALE_MIN + s * (LOG_SCALE_MAX - LOG_SCALE_MIN));
const scaleToSlider = (scale: number) =>
  (Math.log(scale) - LOG_SCALE_MIN) / (LOG_SCALE_MAX - LOG_SCALE_MIN);

function runChain1D(preset: MHTunerPreset, scale: number, n: number, seed: number) {
  const rng = createSeededRng(seed);
  const kernel: ProposalKernel<number> = {
    propose: (x, r) => ({ xPrime: x + scale * r.normal(), logQRatio: 0 }),
  };
  const logPi = (x: number) => preset.logPi(x);
  return metropolisHastings(0, logPi, kernel, n, DEFAULT_BURN_IN, 1, rng);
}

function runChain2D(preset: MHTunerPreset, scale: number, n: number, seed: number) {
  const rng = createSeededRng(seed);
  const kernel: ProposalKernel<number[]> = {
    propose: (x, r) => ({
      xPrime: x.map((v) => v + scale * r.normal()),
      logQRatio: 0,
    }),
  };
  const logPi = (q: number[]) => preset.logPi(q);
  return metropolisHastings([0, 0], logPi, kernel, n, DEFAULT_BURN_IN, 1, rng);
}

export default function MetropolisHastingsTuner() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const [scaleSlider, setScaleSlider] = useState(scaleToSlider(2.4));
  const [seed, setSeed] = useState(42);

  const preset = mhTunerPresets[presetIdx];
  // Immediate value for the slider readout (no lag in the label as user drags).
  const immediateScale = sliderToScale(scaleSlider);
  // Deferred value for the expensive chain recompute: React commits this
  // only when the slider drag pauses. Keeps the slider responsive instead
  // of janking the main thread on every onChange (5000-iter MH chain).
  const deferredScaleSlider = useDeferredValue(scaleSlider);
  const scale = sliderToScale(deferredScaleSlider);
  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;

  const chartW = Math.max(280, Math.min(640, (width ?? 600) - 16));
  const chartH = isMobile ? 240 : 300;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Chain (depends on preset, scale, seed) ────────────────────────────────
  const chain = useMemo(() => {
    if (preset.dimension === 1) return runChain1D(preset, scale, DEFAULT_N, seed);
    return runChain2D(preset, scale, DEFAULT_N, seed);
  }, [preset, scale, seed]);

  // ── Acceptance-rate-vs-scale curve (depends on preset, seed) ──────────────
  const acceptCurve = useMemo(() => {
    const pts: { scale: number; rate: number }[] = [];
    for (let i = 0; i < ACCEPT_CURVE_POINTS; i++) {
      const s = sliderToScale(i / (ACCEPT_CURVE_POINTS - 1));
      const c =
        preset.dimension === 1
          ? runChain1D(preset, s, ACCEPT_CURVE_N, seed + i)
          : runChain2D(preset, s, ACCEPT_CURVE_N, seed + i);
      pts.push({ scale: s, rate: c.acceptanceRate });
    }
    return pts;
  }, [preset, seed]);

  // ── Left panel: 1-D density or 2-D contours + chain path ─────────────────
  const leftPanel = useMemo(() => {
    if (preset.dimension === 1) {
      const [xmin, xmax] = preset.support as [number, number];
      const nGrid = 200;
      const xs = Array.from({ length: nGrid }, (_, i) => xmin + (i * (xmax - xmin)) / (nGrid - 1));
      const logPis = xs.map((x) => preset.logPi(x));
      const maxLog = Math.max(...logPis);
      const ys = logPis.map((lp) => Math.exp(lp - maxLog));
      const yMax = Math.max(...ys) * 1.1;

      const xScale = (x: number) => ((x - xmin) / (xmax - xmin)) * plotW;
      const yScale = (y: number) => plotH - (y / yMax) * plotH;

      const densityPath = xs
        .map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x).toFixed(2)},${yScale(ys[i]).toFixed(2)}`)
        .join(' ');

      // Thinned chain for plotting (last 400 samples).
      const thinned = (chain.samples as number[]).slice(-400);
      const chainHeights = thinned.map(
        (_x, i) => plotH - (i / thinned.length) * plotH * 0.4 - plotH * 0.1,
      );

      const xAxisTicks = [xmin, (xmin + xmax) / 2, xmax];

      return (
        <svg width={chartW} height={chartH} role="img" aria-label="1-D target with chain path">
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            <path
              d={`${densityPath} L${xScale(xmax)},${plotH} L${xScale(xmin)},${plotH} Z`}
              fill={bayesianColors.posterior}
              fillOpacity={0.15}
            />
            <path d={densityPath} fill="none" stroke={bayesianColors.posterior} strokeWidth={1.8} />
            {thinned.map((x, i) => (
              <circle
                key={i}
                cx={xScale(x as number)}
                cy={chainHeights[i]}
                r={1.6}
                fill={bayesianColors.chains[0]}
                fillOpacity={0.5}
              />
            ))}
            <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="currentColor" strokeWidth={1} />
            {xAxisTicks.map((t) => (
              <g key={t} transform={`translate(${xScale(t)},${plotH})`}>
                <line y1={0} y2={4} stroke="currentColor" />
                <text y={16} textAnchor="middle" fontSize={11} fill="currentColor">
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            <text x={plotW / 2} y={plotH + 28} textAnchor="middle" fontSize={11} fill="currentColor">
              x
            </text>
          </g>
        </svg>
      );
    }
    // 2-D banana: contour via filled-cell heatmap + chain path.
    const [xRange, yRange] = preset.support as [[number, number], [number, number]];
    const nx = 60;
    const ny = 60;
    const cells: { x: number; y: number; p: number }[] = [];
    let maxLp = -Infinity;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const x = xRange[0] + ((i + 0.5) / nx) * (xRange[1] - xRange[0]);
        const y = yRange[0] + ((j + 0.5) / ny) * (yRange[1] - yRange[0]);
        const lp = preset.logPi([x, y]);
        if (lp > maxLp) maxLp = lp;
        cells.push({ x, y, p: lp });
      }
    }
    const cellW = plotW / nx;
    const cellH = plotH / ny;
    const xScale = (x: number) =>
      ((x - xRange[0]) / (xRange[1] - xRange[0])) * plotW;
    const yScale = (y: number) =>
      plotH - ((y - yRange[0]) / (yRange[1] - yRange[0])) * plotH;
    const thinned = (chain.samples as number[][]).slice(-250);

    return (
      <svg width={chartW} height={chartH} role="img" aria-label="2-D target with chain path">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {cells.map((c, i) => {
            const opacity = Math.min(0.9, Math.exp(c.p - maxLp));
            return (
              <rect
                key={i}
                x={xScale(c.x) - cellW / 2}
                y={yScale(c.y) - cellH / 2}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={bayesianColors.posterior}
                fillOpacity={opacity * 0.6}
              />
            );
          })}
          <path
            d={thinned
              .map(
                (p, i) =>
                  `${i === 0 ? 'M' : 'L'}${xScale(p[0]).toFixed(2)},${yScale(p[1]).toFixed(2)}`,
              )
              .join(' ')}
            fill="none"
            stroke={bayesianColors.chains[0]}
            strokeOpacity={0.6}
            strokeWidth={1.1}
          />
          <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="currentColor" strokeWidth={1} />
          <line x1={0} y1={0} x2={0} y2={plotH} stroke="currentColor" strokeWidth={1} />
        </g>
      </svg>
    );
  }, [preset, chain, chartW, chartH, plotW, plotH]);

  // ── Right panel: acceptance rate vs scale curve ──────────────────────────
  const rightPanel = useMemo(() => {
    const xScale = (s: number) =>
      ((Math.log(s) - LOG_SCALE_MIN) / (LOG_SCALE_MAX - LOG_SCALE_MIN)) * plotW;
    const yScale = (r: number) => plotH - r * plotH;

    const path = acceptCurve
      .map(
        (p, i) =>
          `${i === 0 ? 'M' : 'L'}${xScale(p.scale).toFixed(2)},${yScale(p.rate).toFixed(2)}`,
      )
      .join(' ');

    const optimalY = yScale(preset.optimalAcceptance);
    const currentX = xScale(scale);

    const logTicks = [0.1, 0.3, 1, 3, 10];

    return (
      <svg width={chartW} height={chartH} role="img" aria-label="Acceptance rate vs proposal scale">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          <line
            x1={0}
            y1={optimalY}
            x2={plotW}
            y2={optimalY}
            stroke={bayesianColors.true}
            strokeDasharray="4 2"
            strokeWidth={1}
          />
          <text
            x={plotW - 4}
            y={optimalY - 4}
            textAnchor="end"
            fontSize={10}
            fill={bayesianColors.true}
          >
            optimal ≈ {preset.optimalAcceptance.toFixed(2)}
          </text>
          <path d={path} fill="none" stroke={bayesianColors.posterior} strokeWidth={1.8} />
          <line
            x1={currentX}
            y1={0}
            x2={currentX}
            y2={plotH}
            stroke={bayesianColors.likelihood}
            strokeWidth={1.5}
          />
          <circle
            cx={currentX}
            cy={yScale(chain.acceptanceRate)}
            r={4}
            fill={bayesianColors.likelihood}
          />
          <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke="currentColor" strokeWidth={1} />
          <line x1={0} y1={0} x2={0} y2={plotH} stroke="currentColor" strokeWidth={1} />
          {logTicks.map((t) => (
            <g key={t} transform={`translate(${xScale(t)},${plotH})`}>
              <line y1={0} y2={4} stroke="currentColor" />
              <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                {t < 1 ? t.toFixed(1) : t.toFixed(0)}
              </text>
            </g>
          ))}
          <text x={plotW / 2} y={plotH + 28} textAnchor="middle" fontSize={11} fill="currentColor">
            proposal scale σ
          </text>
          {[0, 0.25, 0.5, 0.75, 1].map((r) => (
            <g key={r} transform={`translate(0,${yScale(r)})`}>
              <line x1={-4} y1={0} x2={0} y2={0} stroke="currentColor" />
              <text x={-8} y={4} textAnchor="end" fontSize={10} fill="currentColor">
                {r.toFixed(2)}
              </text>
            </g>
          ))}
        </g>
      </svg>
    );
  }, [acceptCurve, chain.acceptanceRate, scale, preset, chartW, chartH, plotW, plotH]);

  const diff = chain.acceptanceRate - preset.optimalAcceptance;
  const tuned = Math.abs(diff) < 0.05;

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="font-medium">Target:</span>
          <select
            className="rounded border px-2 py-1"
            value={presetIdx}
            onChange={(e) => setPresetIdx(Number(e.target.value))}
            aria-label="Target density"
          >
            {mhTunerPresets.map((p, i) => (
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
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            aria-label="RNG seed"
          />
        </label>
      </div>

      <div className="mb-3">
        <label className="flex flex-col gap-1 text-sm">
          <div className="flex items-center justify-between">
            <span className="font-medium">
              Proposal scale σ = {immediateScale.toFixed(2)} (log-scale slider)
            </span>
            <span
              className="font-mono text-xs"
              style={{
                color: tuned ? bayesianColors.true : bayesianColors.likelihood,
              }}
            >
              acceptance = {(chain.acceptanceRate * 100).toFixed(1)}%{' '}
              {tuned ? '✓' : `(optimal ${(preset.optimalAcceptance * 100).toFixed(0)}%)`}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={scaleSlider}
            onChange={(e) => setScaleSlider(Number(e.target.value))}
            aria-label="Proposal scale (log)"
            aria-valuetext={`σ = ${immediateScale.toFixed(2)}, acceptance ${(chain.acceptanceRate * 100).toFixed(1)}%`}
            className="w-full"
          />
        </label>
      </div>

      <div className={`grid gap-4 ${isMobile ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            target + chain path (last 400 samples)
          </div>
          {leftPanel}
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            acceptance rate vs proposal scale
          </div>
          {rightPanel}
        </div>
      </div>

      <p className="mt-3 text-xs text-gray-600 dark:text-gray-400">
        {preset.description}
      </p>
    </div>
  );
}
