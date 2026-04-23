/**
 * QuantileAsymptoticsExplorer — the featured interactive for Topic 29 §29.6.
 *
 * Drives the Bahadur representation + Cor 1 (asymptotic normality of $\hat\xi_p$).
 * The Cauchy preset is the money-shot: sample median is root-$n$ normal even
 * though the Cauchy has no variance, with the Bahadur-limit variance
 * $p(1-p)/f(\xi_p)^2$ finite regardless of distributional moments.
 *
 * Two panels (stacked on mobile, side-by-side on desktop):
 *   Top    — histogram of √n (ξ̂_p − ξ_p) at the current (F, n, p), with the
 *            Bahadur-limit N(0, p(1-p)/f(ξ_p)²) density overlaid.
 *   Bottom — log-log plot of median|R_n| over n ∈ {50, 100, 200, 500, 1000,
 *            2000, 5000}, with the Kiefer envelope c · n^{-3/4} √(log n).
 *
 * Sliders: distribution preset, p, n, seed. Debounced at 200 ms.
 */
import { useDeferredValue, useMemo, useState } from 'react';

// Tune MC replicate count so total work (replicates × n × sort) stays bounded
// as n grows — prevents long UI stalls at the slider max. Larger samples have
// lower sampling variance per replicate, so fewer replicates still give a
// tight histogram overlay. (Copilot PR #33 discussion.)
function mcReplicatesForN(n: number): number {
  return Math.max(500, Math.min(3000, Math.floor(800000 / (n * Math.log2(Math.max(2, n))))));
}
import {
  sampleQuantile,
  bahadurResidual,
} from './shared/nonparametric';
import {
  quantileAsymptoticsPresets,
  seededUniform,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { nonparametricColors } from './shared/colorScales';

const HISTOGRAM_BINS = 50;
const RESIDUAL_REPLICATES = 250;
const RESIDUAL_NS = [50, 100, 200, 500, 1000, 2000, 5000] as const;
const MARGIN = { top: 28, right: 20, bottom: 40, left: 48 };

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function scaleLog(dMin: number, dMax: number, rMin: number, rMax: number) {
  const lnMin = Math.log(dMin);
  const lnMax = Math.log(dMax);
  const span = lnMax - lnMin || 1;
  return (v: number) => rMin + ((Math.log(v) - lnMin) / span) * (rMax - rMin);
}

function histogram(values: number[], nBins: number) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) / nBins || 1;
  const counts = new Array(nBins).fill(0) as number[];
  for (const v of values) {
    const idx = Math.min(nBins - 1, Math.max(0, Math.floor((v - min) / width)));
    counts[idx]++;
  }
  const n = values.length;
  const density = counts.map((c) => c / (n * width));
  const edges = Array.from({ length: nBins + 1 }, (_, i) => min + i * width);
  return { counts, density, edges, min, max, width };
}

function medianAbs(values: number[]): number {
  const abs = values.map((v) => Math.abs(v)).sort((a, b) => a - b);
  const n = abs.length;
  if (n === 0) return 0;
  return n % 2 === 1 ? abs[(n - 1) >> 1] : 0.5 * (abs[n / 2 - 1] + abs[n / 2]);
}

// Linear regression of (log x) vs (log y) → fitted slope for the residual panel.
function logLogSlope(xs: number[], ys: number[]): number {
  const n = xs.length;
  const lx = xs.map(Math.log);
  const ly = ys.map(Math.log);
  const mx = lx.reduce((a, b) => a + b, 0) / n;
  const my = ly.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (lx[i] - mx) * (ly[i] - my);
    den += (lx[i] - mx) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

export default function QuantileAsymptoticsExplorer() {
  const [presetKey, setPresetKey] = useState<string>('normal');
  const [n, setN] = useState(500);
  const [p, setP] = useState(0.5);
  const [seed, setSeed] = useState(42);

  const dPreset = useDeferredValue(presetKey);
  const dN = useDeferredValue(n);
  const dP = useDeferredValue(p);
  const dSeed = useDeferredValue(seed);

  const preset: DistributionPreset = quantileAsymptoticsPresets[dPreset];

  // ── Top-panel MC: histogram of √n (ξ̂_p − ξ_p).
  const histogramData = useMemo(() => {
    const rng = seededUniform(dSeed);
    const truthXi = preset.quantile(dP);
    const replicates = mcReplicatesForN(dN);
    const samples: number[] = new Array(replicates);
    for (let j = 0; j < replicates; j++) {
      const sample = new Array(dN);
      for (let k = 0; k < dN; k++) sample[k] = preset.sampler(rng);
      const xiHat = sampleQuantile(sample, dP);
      samples[j] = Math.sqrt(dN) * (xiHat - truthXi);
    }
    const h = histogram(samples, HISTOGRAM_BINS);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const sdEmpirical = Math.sqrt(
      samples.reduce((acc, v) => acc + (v - mean) ** 2, 0) / (samples.length - 1),
    );
    const fAtXi = preset.pdf(truthXi);
    const sdTheory = fAtXi > 0
      ? Math.sqrt((dP * (1 - dP)) / (fAtXi * fAtXi))
      : Infinity;
    return { samples, hist: h, mean, sdEmpirical, sdTheory, truthXi, fAtXi, replicates };
  }, [preset, dN, dP, dSeed]);

  // ── Bottom-panel MC: median|R_n| decay vs Kiefer envelope.
  const residualData = useMemo(() => {
    const rng = seededUniform(dSeed + 7);
    const truthXi = preset.quantile(dP);
    const fAtXi = preset.pdf(truthXi);
    if (fAtXi <= 0) return null;
    const mads = RESIDUAL_NS.map((nk) => {
      const rs: number[] = new Array(RESIDUAL_REPLICATES);
      for (let j = 0; j < RESIDUAL_REPLICATES; j++) {
        const sample = new Array(nk);
        for (let k = 0; k < nk; k++) sample[k] = preset.sampler(rng);
        rs[j] = bahadurResidual(sample, dP, truthXi, fAtXi);
      }
      return medianAbs(rs);
    });
    const ns = Array.from(RESIDUAL_NS);
    const slope = logLogSlope(ns, mads);
    // Kiefer envelope c · n^{-3/4} √(log n), c pinned to pass through n = 500.
    const anchorIdx = RESIDUAL_NS.indexOf(500);
    const nAnchor = RESIDUAL_NS[anchorIdx];
    const kieferUnnorm = (nk: number) =>
      Math.pow(nk, -0.75) * Math.sqrt(Math.log(nk));
    const c = mads[anchorIdx] / kieferUnnorm(nAnchor);
    const kiefer = RESIDUAL_NS.map((nk) => c * kieferUnnorm(nk));
    return { ns, mads, kiefer, slope, c };
  }, [preset, dP, dSeed]);

  return (
    <div className="my-6 rounded-lg border border-stone-200 bg-white p-4 text-sm shadow-sm dark:border-stone-700 dark:bg-stone-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="font-semibold text-stone-800 dark:text-stone-100">
          Quantile asymptotics explorer
        </span>
        <span className="rounded bg-violet-100 px-2 py-0.5 text-xs text-violet-800 dark:bg-violet-900/40 dark:text-violet-200">
          §29.6 · Bahadur representation
        </span>
      </div>

      {/* Controls */}
      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Distribution
          </span>
          <select
            className="rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
            value={presetKey}
            onChange={(e) => setPresetKey(e.target.value)}
            aria-label="Distribution preset"
          >
            {Object.entries(quantileAsymptoticsPresets).map(([key, pr]) => (
              <option key={key} value={key}>
                {pr.name}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Quantile level p = {fmt(p, 2)}
          </span>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={p}
            onChange={(e) => setP(parseFloat(e.target.value))}
            aria-label="Quantile level p"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Sample size n = {n}
          </span>
          <input
            type="range"
            min={Math.log(50)}
            max={Math.log(5000)}
            step={0.01}
            value={Math.log(n)}
            onChange={(e) => setN(Math.round(Math.exp(parseFloat(e.target.value))))}
            aria-label="Sample size n (log scale)"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Seed = {seed}
          </span>
          <button
            type="button"
            onClick={() => setSeed((s) => s + 1)}
            className="rounded border border-stone-300 bg-stone-50 px-2 py-1 text-sm hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-800 dark:hover:bg-stone-700"
          >
            Resample
          </button>
        </label>
      </div>

      {/* Two-panel layout */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <HistogramPanel data={histogramData} />
        <ResidualPanel data={residualData} />
      </div>

      <div className="mt-3 text-xs text-stone-500 dark:text-stone-400">
        {histogramData.replicates.toLocaleString()} MC replicates per histogram
        (adaptive in n) · {RESIDUAL_REPLICATES} per residual point over{' '}
        {RESIDUAL_NS.length} values of n.
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Top panel — histogram with Bahadur-limit Normal overlay
// ═══════════════════════════════════════════════════════════════════════════

function HistogramPanel({
  data,
}: {
  data: {
    samples: number[];
    hist: ReturnType<typeof histogram>;
    mean: number;
    sdEmpirical: number;
    sdTheory: number;
    truthXi: number;
    fAtXi: number;
  };
}) {
  const width = 420;
  const height = 260;
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const { hist, sdEmpirical, sdTheory, truthXi, fAtXi, mean } = data;
  const xMin = hist.edges[0];
  const xMax = hist.edges[hist.edges.length - 1];
  // Overlay: N(0, sdTheory²) density, evaluated on a fine grid covering the hist range.
  const densityMax = Math.max(...hist.density);
  const overlayGrid = Array.from({ length: 160 }, (_, i) => {
    const x = xMin + (i / 159) * (xMax - xMin);
    const y = Number.isFinite(sdTheory)
      ? Math.exp(-0.5 * (x * x) / (sdTheory * sdTheory)) /
        (sdTheory * Math.sqrt(2 * Math.PI))
      : 0;
    return { x, y };
  });
  const overlayMax = Math.max(...overlayGrid.map((g) => g.y), 0);
  const yMax = Math.max(densityMax, overlayMax) * 1.1 || 1;
  const xScale = scaleLinear(xMin, xMax, 0, innerW);
  const yScale = scaleLinear(0, yMax, innerH, 0);

  const overlayPath = overlayGrid
    .map((g, i) => `${i === 0 ? 'M' : 'L'}${xScale(g.x)},${yScale(g.y)}`)
    .join(' ');

  const ratio = sdEmpirical / sdTheory;

  return (
    <div>
      <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
        Histogram of √n (ξ̂<sub>p</sub> − ξ<sub>p</sub>) with Bahadur-limit overlay
      </div>
      <svg
        role="img"
        aria-label="Histogram of the scaled quantile deviation with Bahadur limiting Normal overlay"
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* Histogram bars */}
          {hist.density.map((d, i) => (
            <rect
              key={i}
              x={xScale(hist.edges[i])}
              y={yScale(d)}
              width={Math.max(0, xScale(hist.edges[i + 1]) - xScale(hist.edges[i]) - 1)}
              height={Math.max(0, innerH - yScale(d))}
              fill={nonparametricColors.ecdf}
              opacity={0.35}
            />
          ))}
          {/* Bahadur-limit Normal overlay */}
          <path
            d={overlayPath}
            fill="none"
            stroke={nonparametricColors.bahadur}
            strokeWidth={2}
          />
          {/* Axes */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" strokeWidth={1} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" strokeWidth={1} />
          {/* x-axis ticks at min, 0, max */}
          {[xMin, 0, xMax].map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},${innerH})`}>
              <line y2={5} stroke="currentColor" />
              <text y={18} textAnchor="middle" fontSize={10} fill="currentColor">
                {fmt(t, 2)}
              </text>
            </g>
          ))}
          {/* x-axis label */}
          <text x={innerW / 2} y={innerH + 32} textAnchor="middle" fontSize={11} fill="currentColor">
            √n (ξ̂_p − ξ_p)
          </text>
        </g>
      </svg>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-stone-600 dark:text-stone-300 sm:grid-cols-4">
        <div>
          ξ<sub>p</sub> (truth) = <span className="font-mono">{fmt(truthXi)}</span>
        </div>
        <div>
          f(ξ<sub>p</sub>) = <span className="font-mono">{fmt(fAtXi)}</span>
        </div>
        <div>
          empirical SD = <span className="font-mono">{fmt(sdEmpirical)}</span>
        </div>
        <div>
          theory SD = <span className="font-mono">{fmt(sdTheory)}</span>
        </div>
        <div className="col-span-2 sm:col-span-4">
          ratio (empirical / theory) = <span className="font-mono">{fmt(ratio, 3)}</span>
          {' · mean ≈ '}
          <span className="font-mono">{fmt(mean, 3)}</span>
          {' (expect 0 at the limit)'}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Bottom panel — log-log residual decay with Kiefer envelope
// ═══════════════════════════════════════════════════════════════════════════

function ResidualPanel({
  data,
}: {
  data: {
    ns: number[];
    mads: number[];
    kiefer: number[];
    slope: number;
    c: number;
  } | null;
}) {
  const width = 420;
  const height = 260;
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;

  if (!data) {
    return (
      <div className="flex items-center justify-center rounded border border-dashed border-stone-300 p-4 text-xs text-stone-500 dark:border-stone-600 dark:text-stone-400">
        Residual decay not defined: f(ξ<sub>p</sub>) = 0 at this (preset, p).
      </div>
    );
  }

  const { ns, mads, kiefer, slope } = data;
  const yAll = [...mads, ...kiefer].filter((v) => v > 0);
  const xMin = Math.min(...ns);
  const xMax = Math.max(...ns);
  const yMin = Math.min(...yAll) * 0.5;
  const yMax = Math.max(...yAll) * 2;
  const xScale = scaleLog(xMin, xMax, 0, innerW);
  const yScale = scaleLog(yMin, yMax, innerH, 0);

  const madPath = ns
    .map((n, i) => `${i === 0 ? 'M' : 'L'}${xScale(n)},${yScale(mads[i])}`)
    .join(' ');
  const kieferPath = ns
    .map((n, i) => `${i === 0 ? 'M' : 'L'}${xScale(n)},${yScale(kiefer[i])}`)
    .join(' ');

  return (
    <div>
      <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
        Bahadur residual decay · median |R<sub>n</sub>| vs Kiefer envelope
      </div>
      <svg
        role="img"
        aria-label="Log-log plot of the Bahadur residual median-absolute over sample size, with Kiefer envelope"
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* Kiefer envelope (dashed) */}
          <path
            d={kieferPath}
            fill="none"
            stroke={nonparametricColors.ks}
            strokeWidth={1.5}
            strokeDasharray="6 3"
          />
          {/* Median|R_n| */}
          <path d={madPath} fill="none" stroke={nonparametricColors.bahadur} strokeWidth={2} />
          {/* Points */}
          {ns.map((n, i) => (
            <circle
              key={i}
              cx={xScale(n)}
              cy={yScale(mads[i])}
              r={3.5}
              fill={nonparametricColors.bahadur}
            />
          ))}
          {/* Axes */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" strokeWidth={1} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" strokeWidth={1} />
          {/* x-axis ticks: powers of 10 inside range */}
          {[50, 100, 500, 1000, 5000].map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},${innerH})`}>
              <line y2={5} stroke="currentColor" />
              <text y={18} textAnchor="middle" fontSize={10} fill="currentColor">
                {t}
              </text>
            </g>
          ))}
          {/* y-axis ticks */}
          {(() => {
            const tickCount = 4;
            return Array.from({ length: tickCount }, (_, i) => {
              const frac = i / (tickCount - 1);
              const v = Math.exp(Math.log(yMin) + frac * (Math.log(yMax) - Math.log(yMin)));
              return (
                <g key={i} transform={`translate(0,${yScale(v)})`}>
                  <line x2={-5} stroke="currentColor" />
                  <text
                    x={-8}
                    dy="0.32em"
                    textAnchor="end"
                    fontSize={10}
                    fill="currentColor"
                  >
                    {v.toExponential(1)}
                  </text>
                </g>
              );
            });
          })()}
          <text x={innerW / 2} y={innerH + 32} textAnchor="middle" fontSize={11} fill="currentColor">
            n (log scale)
          </text>
        </g>
      </svg>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-stone-600 dark:text-stone-300 sm:grid-cols-3">
        <div>
          fitted slope = <span className="font-mono">{fmt(slope, 3)}</span>
        </div>
        <div>
          Kiefer target ≈ <span className="font-mono">−0.75</span>
        </div>
        <div>
          <span
            className="inline-block h-2 w-3"
            style={{
              background: nonparametricColors.bahadur,
              verticalAlign: 'middle',
            }}
          />
          {' median |R_n|'}
        </div>
        <div className="col-span-2 sm:col-span-3 text-[11px] text-stone-500 dark:text-stone-400">
          Finite-n slope is greater than −0.75 because the √(log n) factor adds a
          slowly decaying correction — the curves meet the Kiefer envelope only
          asymptotically.
        </div>
      </div>
    </div>
  );
}
