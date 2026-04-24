/**
 * FunctionalDeltaExplorer — Topic 32 §32.7 component.
 *
 * Demonstrates the functional delta method (Thm 5) by drawing 2000 bootstrap
 * replicates of $\sqrt n (\phi(F_n) - \phi(F))$ for a user-selected
 * functional $\phi$ and underlying distribution $F$, and overlaying the
 * theoretical influence-function-Gaussian density.
 *
 * Three functionals (mean / median / CvM) × three distributions (Uniform /
 * Normal / Exponential). For mean and median, the influence-function
 * variance is closed-form and the Gaussian overlay is drawn. For CvM, the
 * limit under H₀ is non-Gaussian (weighted-χ² per §32.5 Ex 10); the preset
 * returns 0 as a sentinel and the component switches off the Gaussian
 * overlay, showing a caption to that effect.
 *
 * Panel layout:
 *   Left  — current sample kernel-density + true PDF (visual scaffolding)
 *   Right — histogram of 2000 MC replicates of √n(φ(F_n) − φ(F)), with
 *           theoretical Gaussian overlay (when asymp variance > 0)
 *
 * Uses the preset data module — all functional / distribution constants
 * live there, keeping this component pure presentation logic.
 */
import { useEffect, useMemo, useState } from 'react';

import { createSeededRng } from './shared/bayes';
import {
  baseDistributions,
  functionalPresets,
  type BaseDistKey,
  type BaseDistributionSpec,
  type FunctionalKey,
  type FunctionalPreset,
} from '../../data/empirical-processes-data';
import { empiricalProcessColors as C } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

const N_SNAPS = [20, 50, 100, 200, 500, 1000, 2000] as const;
const DEFAULT_N_IDX = 2; // n = 100
const REPLICATES = 2000;
const HIST_BINS = 30;

const SAMPLE_WIDTH = 380;
const SAMPLE_HEIGHT = 240;
const HIST_WIDTH = 420;
const HIST_HEIGHT = 240;
const MARGIN = { top: 18, right: 14, bottom: 36, left: 48 };

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function linspace(a: number, b: number, n: number): number[] {
  if (n < 2) return [a];
  const step = (b - a) / (n - 1);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  return out;
}

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function gaussianPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function pathFromPoints(xs: readonly number[], ys: readonly number[]): string {
  if (xs.length === 0) return '';
  let d = `M ${xs[0].toFixed(1)} ${ys[0].toFixed(1)}`;
  for (let i = 1; i < xs.length; i++) {
    d += ` L ${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`;
  }
  return d;
}

function drawSample(
  preset: BaseDistributionSpec,
  n: number,
  seed: number,
): number[] {
  const rng = createSeededRng(seed);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = preset.sampler(rng);
  return out;
}

/**
 * Run MC replicates and return the sorted array of √n(φ(F_n) − φ(F)) values.
 * Replicates are seeded deterministically from the input seed so "Run" is
 * reproducible, but successive "Run" calls use different seeds.
 */
function runReplicates(
  preset: BaseDistributionSpec,
  func: FunctionalPreset,
  n: number,
  reps: number,
  baseSeed: number,
): number[] {
  const phiAtF = func.phiAtF(preset.key);
  const out: number[] = new Array(reps);
  const sqrtN = Math.sqrt(n);
  for (let r = 0; r < reps; r++) {
    const sample = drawSample(preset, n, baseSeed + r);
    const val = func.phi(sample);
    out[r] = sqrtN * (val - phiAtF);
  }
  out.sort((a, b) => a - b);
  return out;
}

function stats(vals: readonly number[]): { mean: number; sd: number } {
  if (vals.length === 0) return { mean: NaN, sd: NaN };
  let s = 0;
  for (const v of vals) s += v;
  const mean = s / vals.length;
  let ss = 0;
  for (const v of vals) {
    const d = v - mean;
    ss += d * d;
  }
  const sd = Math.sqrt(ss / Math.max(1, vals.length - 1));
  return { mean, sd };
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function FunctionalDeltaExplorer() {
  const [funcKey, setFuncKey] = useState<FunctionalKey>('median');
  const [distKey, setDistKey] = useState<BaseDistKey>('normal');
  const [nIdx, setNIdx] = useState<number>(DEFAULT_N_IDX);
  const [runSeed, setRunSeed] = useState<number>(42);

  const func = useMemo(
    () => functionalPresets.find((f) => f.key === funcKey)!,
    [funcKey],
  );
  const preset = useMemo(
    () => baseDistributions.find((d) => d.key === distKey)!,
    [distKey],
  );
  const n = N_SNAPS[nIdx];

  // MC replicates — recompute only when inputs change.
  const replicates = useMemo(
    () => runReplicates(preset, func, n, REPLICATES, runSeed * 7919),
    [preset, func, n, runSeed],
  );

  // Reset seed on input change so the default view is reproducible at mount.
  useEffect(() => {
    setRunSeed(42);
  }, [funcKey, distKey, nIdx]);

  const asymVar = func.asymptoticVariance(preset.key);
  const asymSd = asymVar > 0 ? Math.sqrt(asymVar) : 0;
  const mcStats = useMemo(() => stats(replicates), [replicates]);
  const nonGaussianLimit = asymVar <= 0;

  // Observed sample for the left panel (same seed as replicate 0 of the run).
  const observedSample = useMemo(
    () => drawSample(preset, n, runSeed * 7919),
    [preset, n, runSeed],
  );

  // ── Left panel: sample visualisation ─────────────────────────────────────

  const samplePlotW = SAMPLE_WIDTH - MARGIN.left - MARGIN.right;
  const samplePlotH = SAMPLE_HEIGHT - MARGIN.top - MARGIN.bottom;
  const sampleX = scaleLinear(
    preset.tGridRange[0],
    preset.tGridRange[1],
    MARGIN.left,
    MARGIN.left + samplePlotW,
  );
  // Draw 80 rug ticks for the observed sample (cap for display density).
  const rugSample = observedSample.slice(0, Math.min(observedSample.length, 80));

  // ── Right panel: histogram + Gaussian overlay ────────────────────────────

  const histPlotW = HIST_WIDTH - MARGIN.left - MARGIN.right;
  const histPlotH = HIST_HEIGHT - MARGIN.top - MARGIN.bottom;

  const histRange = useMemo(() => {
    // Pick a symmetric range covering ±4σ or the MC 1%–99% quantiles,
    // whichever is wider, so the Gaussian overlay fits without clipping.
    const sigma = asymSd > 0 ? asymSd : mcStats.sd;
    const absRange = Math.max(4 * sigma, 4);
    const q01 = replicates[Math.max(0, Math.floor(replicates.length * 0.005))];
    const q99 = replicates[Math.min(replicates.length - 1, Math.floor(replicates.length * 0.995))];
    const lo = Math.min(-absRange, q01);
    const hi = Math.max(absRange, q99);
    const half = Math.max(Math.abs(lo), Math.abs(hi));
    return { lo: -half, hi: half };
  }, [asymSd, mcStats.sd, replicates]);

  const histX = scaleLinear(histRange.lo, histRange.hi, MARGIN.left, MARGIN.left + histPlotW);
  const binWidth = (histRange.hi - histRange.lo) / HIST_BINS;

  const hist = useMemo(() => {
    const counts = new Array<number>(HIST_BINS).fill(0);
    for (const v of replicates) {
      const idx = Math.floor((v - histRange.lo) / binWidth);
      if (idx >= 0 && idx < HIST_BINS) counts[idx] += 1;
    }
    const density = counts.map((c) => c / (replicates.length * binWidth));
    return { counts, density };
  }, [replicates, histRange, binWidth]);

  // Gaussian overlay y-grid
  const overlayXs = useMemo(
    () => linspace(histRange.lo, histRange.hi, 120),
    [histRange],
  );
  const overlayYs = useMemo(
    () => (asymVar > 0 ? overlayXs.map((x) => gaussianPdf(x, 0, asymSd)) : []),
    [overlayXs, asymVar, asymSd],
  );

  const histYMax = Math.max(
    ...hist.density,
    ...(overlayYs.length > 0 ? overlayYs : [0]),
    0.1,
  );
  const histY = scaleLinear(0, histYMax * 1.1, MARGIN.top + histPlotH, MARGIN.top);

  const overlayPath = useMemo(() => {
    if (overlayYs.length === 0) return '';
    const xs = overlayXs.map(histX);
    const ys = overlayYs.map(histY);
    return pathFromPoints(xs, ys);
  }, [overlayXs, overlayYs, histX, histY]);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="my-8 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Functional φ:
        </label>
        <div className="inline-flex rounded-md shadow-sm" role="group">
          {functionalPresets.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={funcKey === f.key}
              onClick={() => setFuncKey(f.key)}
              className={`border px-3 py-1.5 text-sm first:rounded-l-md last:rounded-r-md ${
                funcKey === f.key
                  ? 'text-white'
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600'
              }`}
              style={
                funcKey === f.key
                  ? { background: C.delta, borderColor: C.delta }
                  : undefined
              }
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Underlying F:
        </label>
        <div className="inline-flex rounded-md shadow-sm" role="group">
          {baseDistributions.map((d) => (
            <button
              key={d.key}
              type="button"
              aria-pressed={distKey === d.key}
              onClick={() => setDistKey(d.key)}
              className={`border px-3 py-1.5 text-sm first:rounded-l-md last:rounded-r-md ${
                distKey === d.key
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600'
              }`}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
          <span className="font-medium">Sample size n =</span>
          <span
            className="inline-block min-w-[3rem] rounded bg-slate-100 px-2 py-0.5 text-center font-mono dark:bg-slate-800"
            aria-live="polite"
          >
            {n.toLocaleString()}
          </span>
        </label>
        <input
          type="range"
          min={0}
          max={N_SNAPS.length - 1}
          step={1}
          value={nIdx}
          onChange={(e) => setNIdx(Number(e.target.value))}
          aria-label={`Sample size ${n}`}
          className="h-2 flex-1 max-w-[260px] appearance-none rounded-lg bg-slate-200 dark:bg-slate-700"
        />
        <button
          type="button"
          onClick={() => setRunSeed((s) => s + 1)}
          aria-label="Run new Monte-Carlo replicates"
          className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
        >
          Run {REPLICATES.toLocaleString()} replicates
        </button>
      </div>

      <div className="flex flex-wrap gap-4">
        {/* ── Panel 1: observed sample ────────────────────────────────── */}
        <div className="flex-1 min-w-[280px]">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-400">
            Observed sample (n = {n}) on the {preset.label} support
          </div>
          <svg
            viewBox={`0 0 ${SAMPLE_WIDTH} ${SAMPLE_HEIGHT}`}
            role="img"
            aria-label="Observed sample rug plot"
            className="w-full h-auto"
          >
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + samplePlotW}
              y1={MARGIN.top + samplePlotH}
              y2={MARGIN.top + samplePlotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            {/* x ticks */}
            {linspace(preset.tGridRange[0], preset.tGridRange[1], 5).map((t, i) => (
              <g key={`sxt-${i}`}>
                <line
                  x1={sampleX(t)}
                  x2={sampleX(t)}
                  y1={MARGIN.top + samplePlotH}
                  y2={MARGIN.top + samplePlotH + 3}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={sampleX(t)}
                  y={MARGIN.top + samplePlotH + 14}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {Number.isInteger(t) ? t : t.toFixed(1)}
                </text>
              </g>
            ))}
            {/* Rug */}
            {rugSample.map((v, i) => (
              <line
                key={`rug-${i}`}
                x1={sampleX(v)}
                x2={sampleX(v)}
                y1={MARGIN.top + samplePlotH}
                y2={MARGIN.top + samplePlotH - 10}
                stroke={C.empirical}
                strokeOpacity={0.6}
                strokeWidth={1}
              />
            ))}
            {/* φ(F_n) marker */}
            <line
              x1={sampleX(func.phi(observedSample))}
              x2={sampleX(func.phi(observedSample))}
              y1={MARGIN.top}
              y2={MARGIN.top + samplePlotH}
              stroke={C.delta}
              strokeWidth={1.5}
              strokeDasharray="5 3"
            />
            <text
              x={MARGIN.left + samplePlotW / 2}
              y={SAMPLE_HEIGHT - 4}
              textAnchor="middle"
              className="fill-slate-700 text-[11px] dark:fill-slate-300"
            >
              {preset.xAxisLabel} · dashed line = φ(F_n) = {func.phi(observedSample).toFixed(3)}
            </text>
          </svg>
        </div>

        {/* ── Panel 2: MC histogram + theoretical overlay ─────────────── */}
        <div className="flex-1 min-w-[320px]">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-400">
            √n(φ(F_n) − φ(F)) · {REPLICATES} MC replicates
            {nonGaussianLimit && (
              <span className="ml-2 font-normal lowercase text-amber-700 dark:text-amber-400">
                — non-Gaussian limit (§32.5 Ex 10)
              </span>
            )}
          </div>
          <svg
            viewBox={`0 0 ${HIST_WIDTH} ${HIST_HEIGHT}`}
            role="img"
            aria-label="Monte-Carlo histogram with Gaussian overlay"
            className="w-full h-auto"
          >
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + histPlotW}
              y1={MARGIN.top + histPlotH}
              y2={MARGIN.top + histPlotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + histPlotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            {/* X ticks */}
            {linspace(histRange.lo, histRange.hi, 5).map((t, i) => (
              <g key={`hxt-${i}`}>
                <line
                  x1={histX(t)}
                  x2={histX(t)}
                  y1={MARGIN.top + histPlotH}
                  y2={MARGIN.top + histPlotH + 3}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={histX(t)}
                  y={MARGIN.top + histPlotH + 14}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {t.toFixed(1)}
                </text>
              </g>
            ))}

            {/* Histogram bars */}
            {hist.density.map((y, i) => {
              if (y <= 0) return null;
              const x0 = histRange.lo + i * binWidth;
              const x1 = histRange.lo + (i + 1) * binWidth;
              const px0 = histX(x0);
              const px1 = histX(x1);
              return (
                <rect
                  key={`bar-${i}`}
                  x={px0 + 0.5}
                  y={histY(y)}
                  width={Math.max(0, px1 - px0 - 1)}
                  height={Math.max(0, histY(0) - histY(y))}
                  fill={C.delta}
                  fillOpacity={0.5}
                />
              );
            })}

            {/* Theoretical Gaussian overlay */}
            {overlayPath && (
              <path
                d={overlayPath}
                fill="none"
                stroke={C.reference}
                strokeWidth={2}
                strokeDasharray="1 3"
              />
            )}

            {/* 0 line */}
            <line
              x1={histX(0)}
              x2={histX(0)}
              y1={MARGIN.top}
              y2={MARGIN.top + histPlotH}
              stroke="currentColor"
              className="text-slate-400"
              strokeDasharray="2 3"
            />

            <text
              x={MARGIN.left + histPlotW / 2}
              y={HIST_HEIGHT - 4}
              textAnchor="middle"
              className="fill-slate-700 text-[11px] dark:fill-slate-300"
            >
              √n(φ(F_n) − φ(F))
            </text>
          </svg>
        </div>
      </div>

      {/* Side-panel stats table */}
      <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-sm text-slate-700 dark:text-slate-300">
        <div className="flex justify-between">
          <span className="font-medium">asymptotic mean</span>
          <span className="font-mono">0</span>
        </div>
        <div className="flex justify-between">
          <span className="font-medium">Monte-Carlo mean</span>
          <span className="font-mono">{mcStats.mean.toFixed(3)}</span>
        </div>
        <div className="flex justify-between">
          <span className="font-medium">asymptotic variance</span>
          <span className="font-mono">
            {asymVar > 0 ? asymVar.toFixed(4) : 'non-Gaussian'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="font-medium">Monte-Carlo variance</span>
          <span className="font-mono">{(mcStats.sd ** 2).toFixed(4)}</span>
        </div>
        <div className="col-span-2 text-xs italic text-slate-600 dark:text-slate-400 mt-1">
          {func.description}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-slate-600 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3" style={{ background: C.delta, opacity: 0.5 }} />
          MC histogram
        </span>
        {!nonGaussianLimit && (
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-0 w-6 border-t-2 border-dotted"
              style={{ borderColor: C.reference }}
            />
            influence-function Gaussian
          </span>
        )}
      </div>
    </div>
  );
}
