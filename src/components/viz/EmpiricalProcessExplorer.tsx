/**
 * EmpiricalProcessExplorer — Topic 32 §32.5 FEATURED component.
 *
 * Donsker's theorem visualised in two panels:
 *
 *   Panel 1 (paths) — up to ten realisations of 𝔾_n(t) = √n (F_n(t) − F(t))
 *     drawn at the selected sample size, overlaid with three Brownian-bridge
 *     reference paths (toggleable). As n increases the realisations should
 *     visually tighten around the reference paths — weak convergence in
 *     ℓ^∞(ℝ) made concrete. Bridge paths are generated once per mount with
 *     a fixed seed so the reference doesn't jitter as the user adds paths.
 *
 *   Panel 2 (sup-norm histogram) — running histogram of the ten paths' sup
 *     norms, with the Kolmogorov PDF overlaid. After the user adds enough
 *     paths at large n, the histogram should visibly match the PDF — the
 *     Kolmogorov limit corollary of §32.5.
 *
 * Data flow:
 *   • Presets from `empirical-processes-data.ts` (three distributions).
 *   • `empiricalProcess`, `brownianBridgePath`, `kolmogorovCDF`, `supNorm`
 *     from `nonparametric.ts` §§4, 17, 18.
 *   • Fresh seeded RNG per click — deterministic but visually "random".
 *
 * Accessibility:
 *   • aria-label on every control; keyboard-navigable slider (arrow keys).
 *   • Redundant line-style encoding: observed 𝔾_n paths solid, BB reference
 *     dashed, Kolmogorov PDF dotted. Color-blind-safe even without palette.
 *
 * Mobile: stacks the histogram below the path panel at <640px width.
 */
import { useEffect, useMemo, useRef, useState } from 'react';

import { createSeededRng } from './shared/bayes';
import {
  brownianBridgePath,
  empiricalProcess,
  kolmogorovCDF,
  supNorm,
} from './shared/nonparametric';
import {
  baseDistributions,
  type BaseDistKey,
  type BaseDistributionSpec,
} from '../../data/empirical-processes-data';
import { empiricalProcessColors as C } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

const N_SNAPS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] as const;
const DEFAULT_N_IDX = 4; // n = 200
const MAX_PATHS = 10;
const BB_REF_COUNT = 3;
const BB_GRID = 400;
const HIST_BINS = 15;
const HIST_X_MAX = 3.0;
const HIST_X_MIN = 0.0;

const PANEL_WIDTH = 520;
const PANEL_HEIGHT = 260;
const HIST_WIDTH = 320;
const HIST_HEIGHT = 260;
const MARGIN = { top: 18, right: 16, bottom: 34, left: 44 };

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

/**
 * Linearly interpolate a Brownian-bridge sample path `bb` (defined on the
 * uniform grid u = k/(len−1), k = 0,…,len−1) at an arbitrary u ∈ [0, 1].
 * Returns 0 outside [0, 1] (the bridge is anchored at both endpoints).
 */
function interpBB(bb: readonly number[], u: number): number {
  if (u <= 0 || u >= 1) return 0;
  const pos = u * (bb.length - 1);
  const i = Math.floor(pos);
  const frac = pos - i;
  return bb[i] * (1 - frac) + bb[i + 1] * frac;
}

/**
 * Path string for a step-function-free polyline connecting (x_j, y_j). The
 * empirical process on a fine grid is visually close to continuous, so a
 * straight-segment polyline suffices — no "V"-shape step jumps needed.
 */
function pathFromPoints(xs: readonly number[], ys: readonly number[]): string {
  if (xs.length === 0) return '';
  let d = `M ${xs[0].toFixed(1)} ${ys[0].toFixed(1)}`;
  for (let i = 1; i < xs.length; i++) {
    d += ` L ${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`;
  }
  return d;
}

interface PathRecord {
  values: number[];
  supNormValue: number;
}

/**
 * Draw a fresh empirical-process realisation. Extracted so both the initial
 * auto-fill and user-clicked "Add realisation" go through the same path.
 */
function drawRealisation(
  preset: BaseDistributionSpec,
  n: number,
  seed: number,
): PathRecord {
  const rng = createSeededRng(seed);
  const sample: number[] = new Array(n);
  for (let i = 0; i < n; i++) sample[i] = preset.sampler(rng);
  const values = empiricalProcess(sample, preset.tGrid, preset.Fcdf);
  return { values, supNormValue: supNorm(values) };
}

/**
 * Precompute the Kolmogorov PDF on a fine grid by finite-differencing the
 * CDF. The Kolmogorov density has no closed form in elementary functions;
 * numerical differentiation of the existing `kolmogorovCDF` is accurate
 * enough for a visual overlay (< 0.5% error at the h = 0.02 step).
 */
function kolmogorovPdf(nPts = 200): { xs: number[]; ys: number[] } {
  const xs = linspace(0.2, HIST_X_MAX, nPts);
  const ys: number[] = new Array(nPts - 1);
  const xMids: number[] = new Array(nPts - 1);
  for (let i = 0; i < nPts - 1; i++) {
    const h = xs[i + 1] - xs[i];
    ys[i] = (kolmogorovCDF(xs[i + 1]) - kolmogorovCDF(xs[i])) / h;
    xMids[i] = 0.5 * (xs[i] + xs[i + 1]);
  }
  return { xs: xMids, ys };
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function EmpiricalProcessExplorer() {
  const [distKey, setDistKey] = useState<BaseDistKey>('uniform');
  const [nIdx, setNIdx] = useState<number>(DEFAULT_N_IDX);
  const [showBridge, setShowBridge] = useState<boolean>(true);
  const [paths, setPaths] = useState<PathRecord[]>([]);

  // Click counter → seed. useRef so re-renders don't reset it; resets on
  // preset / n change inside the effect below.
  const clickRef = useRef<number>(0);

  const n = N_SNAPS[nIdx];
  const preset = useMemo(
    () => baseDistributions.find((d) => d.key === distKey)!,
    [distKey],
  );

  // Precompute three BB reference paths at mount. Fixed seed so they don't
  // jitter as the user interacts.
  const bridgePaths = useMemo(() => {
    const rng = createSeededRng(99);
    const out: number[][] = [];
    for (let i = 0; i < BB_REF_COUNT; i++) out.push(brownianBridgePath(BB_GRID, rng));
    return out;
  }, []);

  // Kolmogorov PDF (static, precomputed once).
  const kPdf = useMemo(() => kolmogorovPdf(240), []);

  // Reset and auto-draw one realisation on preset/n change.
  useEffect(() => {
    clickRef.current = 1;
    const seed = 1000 + distKey.charCodeAt(0) * 37 + nIdx * 101 + 1;
    setPaths([drawRealisation(preset, n, seed)]);
  }, [distKey, nIdx, preset, n]);

  const addRealisation = (): void => {
    clickRef.current += 1;
    const seed = 1000 + distKey.charCodeAt(0) * 37 + nIdx * 101 + clickRef.current;
    const rec = drawRealisation(preset, n, seed);
    setPaths((prev) => {
      const next = prev.length >= MAX_PATHS ? prev.slice(1) : prev.slice();
      next.push(rec);
      return next;
    });
  };

  const resetPaths = (): void => {
    clickRef.current = 1;
    const seed = 1000 + distKey.charCodeAt(0) * 37 + nIdx * 101 + 1;
    setPaths([drawRealisation(preset, n, seed)]);
  };

  // ── Layout scales ─────────────────────────────────────────────────────────

  const plotW = PANEL_WIDTH - MARGIN.left - MARGIN.right;
  const plotH = PANEL_HEIGHT - MARGIN.top - MARGIN.bottom;
  const histW = HIST_WIDTH - MARGIN.left - MARGIN.right;
  const histH = HIST_HEIGHT - MARGIN.top - MARGIN.bottom;

  // Y-range for path panel: wider than observed to fit BB overlays
  // comfortably. BB at n=2000 ranges ~ [-2, 2]; empirical 𝔾_n same order.
  const yMin = -2.2;
  const yMax = 2.2;

  const xScale = scaleLinear(
    preset.tGridRange[0],
    preset.tGridRange[1],
    MARGIN.left,
    MARGIN.left + plotW,
  );
  const yScale = scaleLinear(yMin, yMax, MARGIN.top + plotH, MARGIN.top);

  // ── Bridge paths mapped to data-space ────────────────────────────────────

  const bridgePathStrings = useMemo(() => {
    if (!showBridge) return [];
    const tGrid = preset.tGrid;
    const out: string[] = [];
    for (const bb of bridgePaths) {
      const xs: number[] = new Array(tGrid.length);
      const ys: number[] = new Array(tGrid.length);
      for (let j = 0; j < tGrid.length; j++) {
        const u = preset.Fcdf(tGrid[j]);
        xs[j] = xScale(tGrid[j]);
        ys[j] = yScale(interpBB(bb, u));
      }
      out.push(pathFromPoints(xs, ys));
    }
    return out;
  }, [bridgePaths, preset, showBridge, xScale, yScale]);

  // ── Empirical-process path strings ───────────────────────────────────────

  const pathStrings = useMemo(() => {
    const tGrid = preset.tGrid;
    return paths.map((r) => {
      const xs: number[] = new Array(tGrid.length);
      const ys: number[] = new Array(tGrid.length);
      for (let j = 0; j < tGrid.length; j++) {
        xs[j] = xScale(tGrid[j]);
        ys[j] = yScale(r.values[j]);
      }
      return pathFromPoints(xs, ys);
    });
  }, [paths, preset, xScale, yScale]);

  // ── Histogram ────────────────────────────────────────────────────────────

  const hist = useMemo(() => {
    const width = (HIST_X_MAX - HIST_X_MIN) / HIST_BINS;
    const counts = new Array<number>(HIST_BINS).fill(0);
    for (const r of paths) {
      const idx = Math.min(
        HIST_BINS - 1,
        Math.max(0, Math.floor((r.supNormValue - HIST_X_MIN) / width)),
      );
      counts[idx] += 1;
    }
    const total = paths.length;
    // Convert to density (counts / (total * width)) so the histogram and
    // Kolmogorov PDF share the same y-axis scale.
    const density = counts.map((c) => (total > 0 ? c / (total * width) : 0));
    return { counts, density, width };
  }, [paths]);

  const histX = scaleLinear(HIST_X_MIN, HIST_X_MAX, MARGIN.left, MARGIN.left + histW);
  // Histogram y-axis: tall enough to fit Kolmogorov PDF peak (~1.75) with a
  // touch of headroom. Clamping ensures low-path-count bars stay visible.
  const histYMax = Math.max(2.0, Math.max(...hist.density, ...kPdf.ys));
  const histY = scaleLinear(0, histYMax, MARGIN.top + histH, MARGIN.top);

  const kPdfPath = useMemo(() => {
    const xs = kPdf.xs.map(histX);
    const ys = kPdf.ys.map(histY);
    return pathFromPoints(xs, ys);
  }, [kPdf, histX, histY]);

  // ── Axis ticks (sparse, hand-picked for readability) ─────────────────────

  const xTicks = useMemo(() => {
    const [a, b] = preset.tGridRange;
    const n0 = 5;
    return linspace(a, b, n0);
  }, [preset]);
  const yTicks = [-2, -1, 0, 1, 2];
  const histXTicks = [0, 0.5, 1, 1.5, 2, 2.5, 3];

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="my-8 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
          Distribution:
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
          className="h-2 flex-1 max-w-[300px] appearance-none rounded-lg bg-slate-200 dark:bg-slate-700"
        />
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={addRealisation}
          aria-label="Add a new empirical-process realisation"
          className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          Add realisation ({paths.length}/{MAX_PATHS})
        </button>
        <button
          type="button"
          onClick={resetPaths}
          aria-label="Reset all realisations"
          className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-600"
        >
          Reset
        </button>
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300">
          <input
            type="checkbox"
            checked={showBridge}
            onChange={(e) => setShowBridge(e.target.checked)}
            aria-label="Toggle Brownian-bridge reference paths"
          />
          <span>Show Brownian-bridge reference</span>
        </label>
      </div>

      <div className="flex flex-wrap gap-4">
        {/* ── Panel 1: 𝔾_n paths ────────────────────────────────────── */}
        <div className="flex-1 min-w-[320px]">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-400">
            Empirical process 𝔾_n(t) = √n (F_n(t) − F(t))
          </div>
          <svg
            viewBox={`0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`}
            role="img"
            aria-label="Empirical-process paths panel"
            className="w-full h-auto"
          >
            {/* Axes */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + plotW}
              y1={MARGIN.top + plotH}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + plotH}
              stroke="currentColor"
              className="text-slate-400"
            />
            {/* y = 0 guide */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + plotW}
              y1={yScale(0)}
              y2={yScale(0)}
              stroke="currentColor"
              className="text-slate-300"
              strokeDasharray="2 3"
            />

            {/* Y ticks */}
            {yTicks.map((t) => (
              <g key={`ytick-${t}`}>
                <line
                  x1={MARGIN.left - 3}
                  x2={MARGIN.left}
                  y1={yScale(t)}
                  y2={yScale(t)}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={MARGIN.left - 6}
                  y={yScale(t)}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {t}
                </text>
              </g>
            ))}

            {/* X ticks */}
            {xTicks.map((t) => (
              <g key={`xtick-${t.toFixed(2)}`}>
                <line
                  x1={xScale(t)}
                  x2={xScale(t)}
                  y1={MARGIN.top + plotH}
                  y2={MARGIN.top + plotH + 3}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={xScale(t)}
                  y={MARGIN.top + plotH + 14}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {Number.isInteger(t) ? t : t.toFixed(1)}
                </text>
              </g>
            ))}

            {/* Axis label */}
            <text
              x={MARGIN.left + plotW / 2}
              y={PANEL_HEIGHT - 4}
              textAnchor="middle"
              className="fill-slate-700 text-[11px] dark:fill-slate-300"
            >
              {preset.xAxisLabel}
            </text>

            {/* BB reference paths (dashed violet) */}
            {showBridge &&
              bridgePathStrings.map((d, i) => (
                <path
                  key={`bb-${i}`}
                  d={d}
                  fill="none"
                  stroke={C.bridge}
                  strokeOpacity={C.brownianBridgeAlpha}
                  strokeWidth={1.4}
                  strokeDasharray="5 4"
                />
              ))}

            {/* Empirical-process paths (solid blue, dimmer for older) */}
            {pathStrings.map((d, i) => {
              const age = pathStrings.length - 1 - i;
              const opacity = Math.max(0.3, 1 - 0.07 * age);
              return (
                <path
                  key={`gn-${i}`}
                  d={d}
                  fill="none"
                  stroke={C.empirical}
                  strokeOpacity={opacity}
                  strokeWidth={1.2}
                />
              );
            })}
          </svg>
        </div>

        {/* ── Panel 2: sup-norm histogram with Kolmogorov overlay ─────── */}
        <div className="flex-1 min-w-[280px]">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-400">
            Sup-norm ‖𝔾_n‖_∞ vs Kolmogorov PDF
          </div>
          <svg
            viewBox={`0 0 ${HIST_WIDTH} ${HIST_HEIGHT}`}
            role="img"
            aria-label="Sup-norm histogram and Kolmogorov reference"
            className="w-full h-auto"
          >
            {/* Axes */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + histW}
              y1={MARGIN.top + histH}
              y2={MARGIN.top + histH}
              stroke="currentColor"
              className="text-slate-400"
            />
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + histH}
              stroke="currentColor"
              className="text-slate-400"
            />

            {/* X ticks */}
            {histXTicks.map((t) => (
              <g key={`hxtick-${t}`}>
                <line
                  x1={histX(t)}
                  x2={histX(t)}
                  y1={MARGIN.top + histH}
                  y2={MARGIN.top + histH + 3}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={histX(t)}
                  y={MARGIN.top + histH + 14}
                  textAnchor="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {t}
                </text>
              </g>
            ))}

            {/* Y ticks (0, 0.5, 1, 1.5, 2) capped by histYMax) */}
            {[0, 0.5, 1, 1.5, 2].filter((y) => y <= histYMax).map((t) => (
              <g key={`hytick-${t}`}>
                <line
                  x1={MARGIN.left - 3}
                  x2={MARGIN.left}
                  y1={histY(t)}
                  y2={histY(t)}
                  stroke="currentColor"
                  className="text-slate-400"
                />
                <text
                  x={MARGIN.left - 6}
                  y={histY(t)}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-slate-600 text-[10px] dark:fill-slate-300"
                >
                  {t}
                </text>
              </g>
            ))}

            {/* Histogram bars */}
            {hist.density.map((y, i) => {
              const x0 = HIST_X_MIN + i * hist.width;
              const x1 = HIST_X_MIN + (i + 1) * hist.width;
              const px0 = histX(x0);
              const px1 = histX(x1);
              const py = histY(y);
              const pyBase = histY(0);
              if (y <= 0) return null;
              return (
                <rect
                  key={`bar-${i}`}
                  x={px0 + 0.5}
                  y={py}
                  width={Math.max(0, px1 - px0 - 1)}
                  height={Math.max(0, pyBase - py)}
                  fill={C.empirical}
                  fillOpacity={0.55}
                />
              );
            })}

            {/* Kolmogorov PDF overlay */}
            <path
              d={kPdfPath}
              fill="none"
              stroke={C.reference}
              strokeWidth={1.8}
              strokeDasharray="1 3"
            />

            <text
              x={MARGIN.left + histW / 2}
              y={HIST_HEIGHT - 4}
              textAnchor="middle"
              className="fill-slate-700 text-[11px] dark:fill-slate-300"
            >
              sup-norm
            </text>
          </svg>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-slate-600 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-6" style={{ background: C.empirical }} />
          observed 𝔾_n
        </span>
        {showBridge && (
          <span className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-0 w-6 border-t-2 border-dashed"
              style={{ borderColor: C.bridge }}
            />
            Brownian-bridge reference
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-0 w-6 border-t-2 border-dotted"
            style={{ borderColor: C.reference }}
          />
          Kolmogorov PDF
        </span>
      </div>
    </div>
  );
}
