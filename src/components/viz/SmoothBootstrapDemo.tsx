/**
 * SmoothBootstrapDemo — Topic 31 §31.8.
 *
 * The motivating case for smooth bootstrap: median of a resample must equal
 * one of the original sample points, so the NAIVE bootstrap distribution of
 * the median is supported on at most n atoms — visible as "spikes" in the
 * histogram. Smooth bootstrap jitters each resampled point by a small
 * Gaussian (bandwidth h via R's bw.nrd0 rule), so the bootstrap distribution
 * becomes genuinely continuous.
 *
 *   • Left panel  — overlay histogram: naïve (reference slate) vs smooth
 *                   (resample emerald) bootstrap of the median.
 *   • Right panel — smooth-bootstrap SE as a function of h, with Silverman's
 *                   rule (the default) marked as a dashed vertical line.
 *
 * Statistic is LOCKED to median (brief §5.3) — the demo loses its motivation
 * with mean or variance because those aren't lattice-supported.
 */
import { useMemo, useState } from 'react';

import { createSeededRng, type SeededRng } from './shared/bayes';
import {
  nonparametricBootstrap,
  smoothBootstrap,
  smoothBootstrapBandwidth,
} from './shared/nonparametric';
import { allPresets, type DistributionPreset } from '../../data/nonparametric-data';
import { bootstrapColors } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Component-level constants
// ────────────────────────────────────────────────────────────────────────────

const B_REPLICATES = 2000;
const H_GRID_POINTS = 12;
const H_BOOTSTRAP_B = 400; // smaller B for the per-h SE curve (12 × 400 = 4800 resamples)
const NS = [20, 50, 100, 200, 500] as const;
const HIST_BINS = 40;
const PANEL_WIDTH = 360;
const PANEL_HEIGHT = 240;
const MARGIN = { top: 18, right: 14, bottom: 36, left: 44 };

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/** Draw n iid samples from a preset, preferring the bulk SeededRng sampler. */
function drawSample(preset: DistributionPreset, n: number, rng: SeededRng): number[] {
  if (preset.sample) return preset.sample(n, rng);
  const out: number[] = new Array(n);
  const uniform = (): number => rng.random();
  for (let i = 0; i < n; i++) out[i] = preset.sampler(uniform);
  return out;
}

/** Sample median of a pre-sorted array (ascending). */
function medianSorted(sorted: readonly number[]): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? 0.5 * (sorted[mid - 1] + sorted[mid]) : sorted[mid];
}

/** Median of an unsorted sample (sorts a copy). */
function sampleMedian(x: readonly number[]): number {
  const s = [...x].sort((a, b) => a - b);
  return medianSorted(s);
}

/** Bessel-corrected standard deviation. */
function stdDev(x: readonly number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  let sum = 0;
  for (const v of x) sum += v;
  const mean = sum / n;
  let ss = 0;
  for (const v of x) {
    const d = v - mean;
    ss += d * d;
  }
  return Math.sqrt(ss / (n - 1));
}

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

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function SmoothBootstrapDemo() {
  const [presetIdx, setPresetIdx] = useState(0); // Normal by default
  const [n, setN] = useState<(typeof NS)[number]>(50);
  const [seed, setSeed] = useState(42);

  const preset = allPresets[presetIdx];

  // One fixed sample per (preset, n, seed). The two bootstraps below share it.
  const sample = useMemo(() => {
    const rng = createSeededRng(seed);
    return drawSample(preset, n, rng);
  }, [preset, n, seed]);

  const silvermanH = useMemo(() => smoothBootstrapBandwidth(sample), [sample]);

  // Naïve vs smooth bootstrap replicates of the median (shared sample).
  const { naive, smooth, thetaHat } = useMemo(() => {
    const rngNaive = createSeededRng(seed + 7_000);
    const rngSmooth = createSeededRng(seed + 13_000);
    const naive = nonparametricBootstrap(sample, sampleMedian, B_REPLICATES, rngNaive);
    const smooth = smoothBootstrap(sample, sampleMedian, B_REPLICATES, rngSmooth, silvermanH);
    return {
      naive,
      smooth,
      thetaHat: sampleMedian(sample),
    };
  }, [sample, silvermanH, seed]);

  // Bandwidth sensitivity: smooth-bootstrap SE of the median on a grid of h
  // values spanning 0.3× to 2.5× Silverman's rule.
  const hCurve = useMemo(() => {
    const lo = 0.3 * silvermanH;
    const hi = 2.5 * silvermanH;
    const hs = linspace(lo, hi, H_GRID_POINTS);
    const ses = hs.map((h, k) => {
      const rng = createSeededRng(seed + 20_000 + k);
      const reps = smoothBootstrap(sample, sampleMedian, H_BOOTSTRAP_B, rng, h);
      return stdDev(reps);
    });
    return { hs, ses };
  }, [sample, silvermanH, seed]);

  const naiveSE = useMemo(() => stdDev(naive), [naive]);
  const smoothSE = useMemo(() => stdDev(smooth), [smooth]);

  // ── Histogram binning (shared x-range across the two overlays) ─────────────
  const hist = useMemo(() => {
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const v of naive) {
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    for (const v of smooth) {
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    if (!Number.isFinite(xMin)) return null;
    const range = xMax - xMin || 1;
    const pad = range * 0.03;
    const binLo = xMin - pad;
    const binHi = xMax + pad;
    const binW = (binHi - binLo) / HIST_BINS;
    const countsNaive = new Array(HIST_BINS).fill(0);
    const countsSmooth = new Array(HIST_BINS).fill(0);
    for (const v of naive) {
      const k = Math.min(HIST_BINS - 1, Math.max(0, Math.floor((v - binLo) / binW)));
      countsNaive[k]++;
    }
    for (const v of smooth) {
      const k = Math.min(HIST_BINS - 1, Math.max(0, Math.floor((v - binLo) / binW)));
      countsSmooth[k]++;
    }
    // Normalize to density (divide by B × binW).
    const denN = B_REPLICATES * binW;
    const densNaive = countsNaive.map((c) => c / denN);
    const densSmooth = countsSmooth.map((c) => c / denN);
    let yMax = 0;
    for (const v of densNaive) if (v > yMax) yMax = v;
    for (const v of densSmooth) if (v > yMax) yMax = v;
    return { binLo, binHi, binW, densNaive, densSmooth, yMax: yMax * 1.12 };
  }, [naive, smooth]);

  // ── Render left panel: overlay histogram ───────────────────────────────────
  const leftPanel = useMemo(() => {
    if (!hist) return null;
    const x = scaleLinear(hist.binLo, hist.binHi, MARGIN.left, PANEL_WIDTH - MARGIN.right);
    const y = scaleLinear(0, hist.yMax, PANEL_HEIGHT - MARGIN.bottom, MARGIN.top);
    const barW = (x(hist.binLo + hist.binW) - x(hist.binLo)) - 1;
    const barsNaive = hist.densNaive.map((v, i) => (
      <rect
        key={`n-${i}`}
        x={x(hist.binLo + i * hist.binW)}
        y={y(v)}
        width={Math.max(0.5, barW)}
        height={y(0) - y(v)}
        fill={bootstrapColors.reference}
        opacity={0.4}
      />
    ));
    const barsSmooth = hist.densSmooth.map((v, i) => (
      <rect
        key={`s-${i}`}
        x={x(hist.binLo + i * hist.binW)}
        y={y(v)}
        width={Math.max(0.5, barW)}
        height={y(0) - y(v)}
        fill={bootstrapColors.resample}
        opacity={0.55}
      />
    ));
    const thetaX = x(thetaHat);
    return {
      x, y,
      elements: (
        <>
          {barsNaive}
          {barsSmooth}
          <line
            x1={thetaX}
            y1={MARGIN.top}
            x2={thetaX}
            y2={PANEL_HEIGHT - MARGIN.bottom}
            stroke="#0f172a"
            strokeDasharray="3 3"
            strokeWidth={1.3}
          />
        </>
      ),
    };
  }, [hist, thetaHat]);

  // ── Render right panel: SE vs h ───────────────────────────────────────────
  const rightPanel = useMemo(() => {
    const xMin = hCurve.hs[0];
    const xMax = hCurve.hs[hCurve.hs.length - 1];
    let yMin = Infinity;
    let yMax = -Infinity;
    for (const v of hCurve.ses) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
    // Include the naïve SE as a horizontal reference so the reader can see
    // how the smooth SE diverges from it.
    if (naiveSE < yMin) yMin = naiveSE;
    if (naiveSE > yMax) yMax = naiveSE;
    const pad = (yMax - yMin) * 0.15 || 1e-6;
    const yLo = Math.max(0, yMin - pad);
    const yHi = yMax + pad;
    const x = scaleLinear(xMin, xMax, MARGIN.left, PANEL_WIDTH - MARGIN.right);
    const y = scaleLinear(yLo, yHi, PANEL_HEIGHT - MARGIN.bottom, MARGIN.top);
    const pathPoints = hCurve.hs
      .map((h, i) => `${x(h).toFixed(1)},${y(hCurve.ses[i]).toFixed(1)}`)
      .join('L');
    return {
      x, y, yLo, yHi, xMin, xMax,
      elements: (
        <>
          {/* Naïve SE reference line */}
          <line
            x1={MARGIN.left}
            y1={y(naiveSE)}
            x2={PANEL_WIDTH - MARGIN.right}
            y2={y(naiveSE)}
            stroke={bootstrapColors.reference}
            strokeDasharray="2 3"
            strokeWidth={1}
          />
          <text
            x={PANEL_WIDTH - MARGIN.right - 6}
            y={y(naiveSE) - 4}
            textAnchor="end"
            className="text-[10px] fill-slate-500"
          >
            naïve SE {fmt(naiveSE, 3)}
          </text>
          {/* Silverman vertical reference */}
          <line
            x1={x(silvermanH)}
            y1={MARGIN.top}
            x2={x(silvermanH)}
            y2={PANEL_HEIGHT - MARGIN.bottom}
            stroke="#94a3b8"
            strokeDasharray="3 3"
            strokeWidth={1}
          />
          <text
            x={x(silvermanH) + 4}
            y={MARGIN.top + 10}
            className="text-[10px] fill-slate-500"
          >
            Silverman {fmt(silvermanH, 3)}
          </text>
          {/* SE curve */}
          <path
            d={`M${pathPoints}`}
            fill="none"
            stroke={bootstrapColors.resample}
            strokeWidth={1.8}
          />
          {hCurve.hs.map((h, i) => (
            <circle
              key={`p-${i}`}
              cx={x(h)}
              cy={y(hCurve.ses[i])}
              r={2.5}
              fill={bootstrapColors.resample}
            />
          ))}
        </>
      ),
    };
  }, [hCurve, silvermanH, naiveSE]);

  // ────────────────────────────────────────────────────────────────────────
  // Render
  // ────────────────────────────────────────────────────────────────────────

  const renderAxes = (
    title: string,
    xScale: (v: number) => number,
    yScale: (v: number) => number,
    xDomain: [number, number],
    yDomain: [number, number],
    yLabel: string,
    xLabel: string,
  ) => {
    const xTicks = linspace(xDomain[0], xDomain[1], 5);
    const yTicks = linspace(yDomain[0], yDomain[1], 4);
    return (
      <>
        <line
          x1={MARGIN.left}
          y1={PANEL_HEIGHT - MARGIN.bottom}
          x2={PANEL_WIDTH - MARGIN.right}
          y2={PANEL_HEIGHT - MARGIN.bottom}
          stroke="#64748b"
        />
        <line
          x1={MARGIN.left}
          y1={MARGIN.top}
          x2={MARGIN.left}
          y2={PANEL_HEIGHT - MARGIN.bottom}
          stroke="#64748b"
        />
        {xTicks.map((v, i) => (
          <text
            key={`xt-${i}`}
            x={xScale(v)}
            y={PANEL_HEIGHT - MARGIN.bottom + 14}
            textAnchor="middle"
            className="text-[10px] fill-slate-500"
          >
            {v.toFixed(2)}
          </text>
        ))}
        {yTicks.map((v, i) => (
          <text
            key={`yt-${i}`}
            x={MARGIN.left - 4}
            y={yScale(v) + 3}
            textAnchor="end"
            className="text-[10px] fill-slate-500"
          >
            {v.toFixed(3)}
          </text>
        ))}
        <text
          x={MARGIN.left}
          y={MARGIN.top - 5}
          className="text-[11px] fill-slate-700"
        >
          {title}
        </text>
        <text
          x={PANEL_WIDTH / 2}
          y={PANEL_HEIGHT - 4}
          textAnchor="middle"
          className="text-[10px] fill-slate-500"
        >
          {xLabel}
        </text>
        <text
          x={10}
          y={PANEL_HEIGHT / 2}
          textAnchor="middle"
          transform={`rotate(-90, 10, ${PANEL_HEIGHT / 2})`}
          className="text-[10px] fill-slate-500"
        >
          {yLabel}
        </text>
      </>
    );
  };

  return (
    <div className="my-8 not-prose border border-slate-200 rounded-lg bg-white p-4 md:p-6 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 text-sm text-slate-600 dark:text-slate-300">
        <span className="font-semibold text-slate-800 dark:text-slate-100">Smooth bootstrap of the median</span>
        <span className="mx-2 text-slate-400">·</span>
        <span>
          Naïve resample of the median is supported on ≤ n points; Gaussian jitter of bandwidth h smooths it.
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <div className="text-xs text-slate-500 mb-1">Preset</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 dark:border-slate-700 dark:bg-slate-800">
            {allPresets.map((p, i) => (
              <button
                key={p.key}
                type="button"
                onClick={() => setPresetIdx(i)}
                className={`px-2.5 py-1 text-xs rounded ${
                  presetIdx === i
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                    : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">Sample size n</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 dark:border-slate-700 dark:bg-slate-800">
            {NS.map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setN(v)}
                className={`px-2 py-1 text-xs rounded ${
                  n === v
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                    : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 text-xs bg-slate-800 text-white rounded hover:bg-slate-700 dark:bg-slate-600 dark:hover:bg-slate-500"
        >
          Reseed
        </button>
      </div>

      {/* Panels (stack on mobile) */}
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <svg
            viewBox={`0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`}
            className="w-full h-auto"
            role="img"
            aria-label="Naive vs smooth bootstrap of the median"
          >
            {leftPanel &&
              renderAxes(
                'Bootstrap distribution (B = 2000)',
                leftPanel.x,
                leftPanel.y,
                [hist!.binLo, hist!.binHi],
                [0, hist!.yMax],
                'density',
                'median*',
              )}
            {leftPanel?.elements}
          </svg>
          <div className="mt-1.5 flex gap-3 text-[11px] text-slate-600 dark:text-slate-300 tabular-nums">
            <span>
              <span
                className="inline-block w-3 h-2 align-middle mr-1"
                style={{ backgroundColor: bootstrapColors.reference, opacity: 0.6 }}
              />
              naïve · SE {fmt(naiveSE, 3)}
            </span>
            <span>
              <span
                className="inline-block w-3 h-2 align-middle mr-1"
                style={{ backgroundColor: bootstrapColors.resample, opacity: 0.7 }}
              />
              smooth · SE {fmt(smoothSE, 3)}
            </span>
            <span className="text-slate-500">θ̂ = {fmt(thetaHat, 3)}</span>
          </div>
        </div>
        <div>
          <svg
            viewBox={`0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`}
            className="w-full h-auto"
            role="img"
            aria-label="Smooth-bootstrap SE as a function of bandwidth h"
          >
            {renderAxes(
              'Bandwidth sensitivity of smooth-bootstrap SE',
              rightPanel.x,
              rightPanel.y,
              [rightPanel.xMin, rightPanel.xMax],
              [rightPanel.yLo, rightPanel.yHi],
              'SE of median*',
              'bandwidth h',
            )}
            {rightPanel.elements}
          </svg>
          <div className="mt-1.5 text-[11px] text-slate-600 dark:text-slate-300">
            Silverman's rule (h ≈ {fmt(silvermanH, 3)}) is the vertical dashed line. SE is nearly flat
            across ~½× to 2× that choice — smooth bootstrap is forgiving of mild bandwidth mis-specification.
          </div>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
        Naïve bootstrap of the median can only return one of the observed sample points, so its
        histogram is spiky. Smooth bootstrap resamples from the Gaussian KDE f̂_h instead — the
        jitter fills in the gaps. The two SE estimates agree to leading order for n large; the
        visible difference at small n is smooth bootstrap fixing the discreteness artifact.
      </p>
    </div>
  );
}
