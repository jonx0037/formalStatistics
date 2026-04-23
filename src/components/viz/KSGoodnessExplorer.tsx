/**
 * KSGoodnessExplorer — interactive for Topic 29 §29.8.
 *
 * One-sample KS test under the null and under a location-shift alternative.
 * Three panels:
 *   A  null distribution of √n D_n (MC) with Kolmogorov density overlay and
 *      dashed critical value at d_α from `kolmogorovQuantile(1 - α)`.
 *   B  alternative distribution of √n D_n under the current shift δ;
 *      empirical power (fraction exceeding d_α) annotated.
 *   C  power vs δ ∈ [0, 2] with the current δ marked — the payoff curve.
 *
 * Drives §29.8's pedagogy that KS-test power grows smoothly with separation
 * under a location alternative, and that the asymptotic Kolmogorov CDF is
 * an excellent approximation at n = 100.
 */
import { useDeferredValue, useMemo, useState } from 'react';
import {
  ksStatistic,
  kolmogorovCDF,
  kolmogorovQuantile,
} from './shared/nonparametric';
import {
  ksPresets,
  seededUniform,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { nonparametricColors } from './shared/colorScales';

const DISCRETE_NS = [20, 50, 100, 500] as const;
const MC_REPLICATES = 2000;
const POWER_DELTAS = Array.from({ length: 21 }, (_, i) => i * 0.1); // 0.0 … 2.0
const POWER_REPLICATES = 600;
const MARGIN = { top: 22, right: 16, bottom: 36, left: 44 };

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

function histogram(values: number[], nBins: number, lo: number, hi: number) {
  const width = (hi - lo) / nBins;
  const counts = new Array(nBins).fill(0) as number[];
  for (const v of values) {
    if (v < lo || v > hi) continue;
    const idx = Math.min(nBins - 1, Math.max(0, Math.floor((v - lo) / width)));
    counts[idx]++;
  }
  const n = values.length;
  const density = counts.map((c) => c / (n * width));
  const edges = Array.from({ length: nBins + 1 }, (_, i) => lo + i * width);
  return { counts, density, edges };
}

function drawSample(preset: DistributionPreset, n: number, shift: number, rng: () => number): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = preset.sampler(rng) + shift;
  return out;
}

export default function KSGoodnessExplorer() {
  const [nullKey, setNullKey] = useState<string>('normal');
  const [n, setN] = useState<number>(100);
  const [delta, setDelta] = useState(0.5);
  const [alpha, setAlpha] = useState(0.05);
  const [seed, setSeed] = useState(42);

  const dNull = useDeferredValue(nullKey);
  const dN = useDeferredValue(n);
  const dDelta = useDeferredValue(delta);
  const dAlpha = useDeferredValue(alpha);
  const dSeed = useDeferredValue(seed);

  const preset = ksPresets[dNull];
  const dAlphaCrit = kolmogorovQuantile(1 - dAlpha);

  const { nullRootNDn, altRootNDn, emp, powerCurve, currentPower } = useMemo(() => {
    const rngNull = seededUniform(dSeed);
    const rngAlt = seededUniform(dSeed + 10000);
    const nullVals: number[] = new Array(MC_REPLICATES);
    const altVals: number[] = new Array(MC_REPLICATES);
    for (let j = 0; j < MC_REPLICATES; j++) {
      const s0 = drawSample(preset, dN, 0, rngNull);
      const s1 = drawSample(preset, dN, dDelta, rngAlt);
      nullVals[j] = Math.sqrt(dN) * ksStatistic(s0, preset.cdf);
      altVals[j] = Math.sqrt(dN) * ksStatistic(s1, preset.cdf);
    }
    const empSz = nullVals.filter((v) => v > dAlphaCrit).length / MC_REPLICATES;
    const empPow = altVals.filter((v) => v > dAlphaCrit).length / MC_REPLICATES;
    // Power curve across δ
    const rngPow = seededUniform(dSeed + 77);
    const pc = POWER_DELTAS.map((d) => {
      let hits = 0;
      for (let j = 0; j < POWER_REPLICATES; j++) {
        const s = drawSample(preset, dN, d, rngPow);
        const stat = Math.sqrt(dN) * ksStatistic(s, preset.cdf);
        if (stat > dAlphaCrit) hits++;
      }
      return hits / POWER_REPLICATES;
    });
    return {
      nullRootNDn: nullVals,
      altRootNDn: altVals,
      emp: { size: empSz, power: empPow },
      powerCurve: pc,
      currentPower: empPow,
    };
  }, [preset, dN, dDelta, dAlpha, dAlphaCrit, dSeed]);

  void currentPower;

  // ── Common plot geometry
  const panelWidth = 340;
  const panelHeight = 220;
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = panelHeight - MARGIN.top - MARGIN.bottom;

  // Panels A and B share x-range [0, ~2.2] for √n D_n.
  const xMax = Math.max(...nullRootNDn, ...altRootNDn, dAlphaCrit * 1.2) * 1.05;
  const nBinsAB = 40;
  const histNull = histogram(nullRootNDn, nBinsAB, 0, xMax);
  const histAlt = histogram(altRootNDn, nBinsAB, 0, xMax);

  const overlayGrid = Array.from({ length: 160 }, (_, i) => {
    const x = (i / 159) * xMax;
    // Density of √n D_n ≈ derivative of Kolmogorov CDF. Finite difference.
    const h = 1e-3;
    const dk = (kolmogorovCDF(x + h) - kolmogorovCDF(x - h)) / (2 * h);
    return { x, y: Math.max(0, dk) };
  });
  const yMaxAB = Math.max(
    ...histNull.density,
    ...histAlt.density,
    ...overlayGrid.map((g) => g.y),
  ) * 1.15 || 1;
  const xScaleAB = scaleLinear(0, xMax, 0, innerW);
  const yScaleAB = scaleLinear(0, yMaxAB, innerH, 0);

  const overlayPath = overlayGrid
    .map((g, i) => `${i === 0 ? 'M' : 'L'}${xScaleAB(g.x)},${yScaleAB(g.y)}`)
    .join(' ');

  // Panel C geometry
  const xScaleC = scaleLinear(0, 2, 0, innerW);
  const yScaleC = scaleLinear(0, 1, innerH, 0);
  const powerPath = POWER_DELTAS.map(
    (d, i) => `${i === 0 ? 'M' : 'L'}${xScaleC(d)},${yScaleC(powerCurve[i])}`,
  ).join(' ');

  return (
    <div className="my-6 rounded-lg border border-stone-200 bg-white p-4 text-sm shadow-sm dark:border-stone-700 dark:bg-stone-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="font-semibold text-stone-800 dark:text-stone-100">
          Kolmogorov–Smirnov goodness-of-fit explorer
        </span>
        <span className="rounded bg-orange-100 px-2 py-0.5 text-xs text-orange-800 dark:bg-orange-900/40 dark:text-orange-200">
          §29.8 · Kolmogorov distribution
        </span>
      </div>

      {/* Controls */}
      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Null F<sub>0</sub>
          </span>
          <select
            className="rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
            value={nullKey}
            onChange={(e) => setNullKey(e.target.value)}
            aria-label="Null-hypothesis distribution"
          >
            {Object.entries(ksPresets).map(([k, pr]) => (
              <option key={k} value={k}>
                {pr.name}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Sample size n
          </span>
          <select
            className="rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
            value={n}
            onChange={(e) => setN(parseInt(e.target.value, 10))}
            aria-label="Sample size"
          >
            {DISCRETE_NS.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Shift δ = {fmt(delta, 2)}
          </span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={delta}
            onChange={(e) => setDelta(parseFloat(e.target.value))}
            aria-label="Location shift delta"
          />
        </label>

        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            α = {alpha.toFixed(2)}
          </span>
          <div className="flex items-center gap-2">
            <select
              className="flex-1 rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              aria-label="Significance level"
            >
              {[0.01, 0.05, 0.1].map((a) => (
                <option key={a} value={a}>
                  {a.toFixed(2)}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setSeed((s) => s + 1)}
              className="rounded border border-stone-300 bg-stone-50 px-2 py-1 text-sm hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-800 dark:hover:bg-stone-700"
            >
              Reseed
            </button>
          </div>
        </div>
      </div>

      {/* Panels */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Panel A: null histogram */}
        <div>
          <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
            A · √n D<sub>n</sub> under H₀
          </div>
          <svg viewBox={`0 0 ${panelWidth} ${panelHeight}`} className="w-full" role="img" aria-label="Null distribution of the KS statistic with Kolmogorov density overlay">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {histNull.density.map((d, i) => (
                <rect
                  key={i}
                  x={xScaleAB(histNull.edges[i])}
                  y={yScaleAB(d)}
                  width={Math.max(0, xScaleAB(histNull.edges[i + 1]) - xScaleAB(histNull.edges[i]) - 1)}
                  height={Math.max(0, innerH - yScaleAB(d))}
                  fill={nonparametricColors.ks}
                  opacity={0.35}
                />
              ))}
              <path d={overlayPath} fill="none" stroke={nonparametricColors.bahadur} strokeWidth={2} />
              <line
                x1={xScaleAB(dAlphaCrit)}
                y1={0}
                x2={xScaleAB(dAlphaCrit)}
                y2={innerH}
                stroke="currentColor"
                strokeWidth={1}
                strokeDasharray="4 2"
              />
              <text
                x={xScaleAB(dAlphaCrit) + 4}
                y={14}
                fontSize={10}
                fill="currentColor"
              >
                d<tspan baselineShift="sub" fontSize={8}>α</tspan> = {fmt(dAlphaCrit, 2)}
              </text>
              <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" />
              <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" />
              {[0, 0.5, 1, 1.5, 2].filter((t) => t <= xMax).map((t, k) => (
                <g key={k} transform={`translate(${xScaleAB(t)},${innerH})`}>
                  <line y2={4} stroke="currentColor" />
                  <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                    {t}
                  </text>
                </g>
              ))}
              <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor">
                √n D_n
              </text>
            </g>
          </svg>
          <div className="text-[11px] text-stone-500 dark:text-stone-400">
            empirical size ≈ <span className="font-mono">{fmt(emp.size, 3)}</span>{' '}
            (target α = {alpha.toFixed(2)})
          </div>
        </div>

        {/* Panel B: alternative histogram */}
        <div>
          <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
            B · √n D<sub>n</sub> under alternative (shift δ = {fmt(delta, 2)})
          </div>
          <svg viewBox={`0 0 ${panelWidth} ${panelHeight}`} className="w-full" role="img" aria-label="KS statistic distribution under the alternative hypothesis">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {histAlt.density.map((d, i) => (
                <rect
                  key={i}
                  x={xScaleAB(histAlt.edges[i])}
                  y={yScaleAB(d)}
                  width={Math.max(0, xScaleAB(histAlt.edges[i + 1]) - xScaleAB(histAlt.edges[i]) - 1)}
                  height={Math.max(0, innerH - yScaleAB(d))}
                  fill={nonparametricColors.ecdf}
                  opacity={0.5}
                />
              ))}
              <line
                x1={xScaleAB(dAlphaCrit)}
                y1={0}
                x2={xScaleAB(dAlphaCrit)}
                y2={innerH}
                stroke="currentColor"
                strokeWidth={1}
                strokeDasharray="4 2"
              />
              <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" />
              <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" />
              {[0, 0.5, 1, 1.5, 2].filter((t) => t <= xMax).map((t, k) => (
                <g key={k} transform={`translate(${xScaleAB(t)},${innerH})`}>
                  <line y2={4} stroke="currentColor" />
                  <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                    {t}
                  </text>
                </g>
              ))}
            </g>
          </svg>
          <div className="text-[11px] text-stone-500 dark:text-stone-400">
            empirical power ≈ <span className="font-mono">{fmt(emp.power, 3)}</span>
          </div>
        </div>

        {/* Panel C: power curve */}
        <div>
          <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
            C · Power vs shift δ
          </div>
          <svg viewBox={`0 0 ${panelWidth} ${panelHeight}`} className="w-full" role="img" aria-label="Empirical power curve across shift values">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <line x1={0} y1={yScaleC(alpha)} x2={innerW} y2={yScaleC(alpha)} stroke="currentColor" strokeDasharray="2 3" />
              <path d={powerPath} fill="none" stroke={nonparametricColors.ks} strokeWidth={2} />
              <circle cx={xScaleC(delta)} cy={yScaleC(emp.power)} r={4} fill={nonparametricColors.bahadur} />
              <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" />
              <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" />
              {[0, 0.5, 1, 1.5, 2].map((t, k) => (
                <g key={k} transform={`translate(${xScaleC(t)},${innerH})`}>
                  <line y2={4} stroke="currentColor" />
                  <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                    {t}
                  </text>
                </g>
              ))}
              {[0, 0.25, 0.5, 0.75, 1].map((t, k) => (
                <g key={k} transform={`translate(0,${yScaleC(t)})`}>
                  <line x2={-4} stroke="currentColor" />
                  <text x={-8} dy="0.32em" textAnchor="end" fontSize={10} fill="currentColor">
                    {t}
                  </text>
                </g>
              ))}
              <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor">
                δ
              </text>
            </g>
          </svg>
          <div className="text-[11px] text-stone-500 dark:text-stone-400">
            Marker at current δ · dashed line = α ({alpha.toFixed(2)})
          </div>
        </div>
      </div>
    </div>
  );
}
