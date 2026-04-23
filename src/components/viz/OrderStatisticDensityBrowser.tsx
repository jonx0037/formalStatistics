/**
 * OrderStatisticDensityBrowser — interactive for Topic 29 §29.3.
 *
 * Discharges the Topic 7 §7.13 "Uniform order statistics are Beta" promise
 * visibly. Three panels:
 *   A  analytic density of X_{(i)} from the general-F formula
 *         f_{X_(i)}(x) = n! / ((i-1)!(n-i)!) F(x)^{i-1} (1-F(x))^{n-i} f(x)
 *   B  MC histogram of X_{(i)} — visually matches Panel A when the formula works
 *   C  U_{(i)} = F(X_{(i)}) with Beta(i, n-i+1) overlaid — the theorem live
 *
 * Controls: distribution preset, discrete n slider, discrete i slider.
 * Panel C is always Beta-shaped regardless of F — that's Thm 2.
 */
import { useDeferredValue, useMemo, useState } from 'react';
import {
  orderStatisticDensity,
  uniformOrderStatisticDensity,
  simulateOrderStatistic,
} from './shared/nonparametric';
import {
  orderStatPresets,
  seededUniform,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { nonparametricColors } from './shared/colorScales';

const DISCRETE_NS = [3, 5, 10, 20, 50] as const;
const MC_REPLICATES = 4000;
const GRID_POINTS = 160;
const MARGIN = { top: 22, right: 16, bottom: 36, left: 44 };

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
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

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

export default function OrderStatisticDensityBrowser() {
  const [presetKey, setPresetKey] = useState<string>('uniform');
  const [n, setN] = useState<number>(10);
  const [i, setI] = useState<number>(5);

  const dPreset = useDeferredValue(presetKey);
  const dN = useDeferredValue(n);
  const dI = useDeferredValue(Math.min(i, n));
  const preset: DistributionPreset = orderStatPresets[dPreset];

  // ── Panel A/B common grid: use preset's natural domain, extend in tails for heavy-tailed.
  const { grid, mcSamples, mcHist, gridDensityA, yMaxAB, transformedSamples, transformedHist } = useMemo(() => {
    const [xLo, xHi] = preset.domain;
    const nBins = 40;
    const g = Array.from({ length: GRID_POINTS + 1 }, (_, k) => {
      const x = xLo + (k / GRID_POINTS) * (xHi - xLo);
      return x;
    });
    const densityA = g.map((x) => orderStatisticDensity(x, dI, dN, preset.cdf, preset.pdf));
    // MC on X_{(i)} for empirical histogram.
    const rng = seededUniform(42);
    const mc = simulateOrderStatistic(() => preset.sampler(rng), dI, dN, MC_REPLICATES);
    const hAB = histogram(mc, nBins, xLo, xHi);
    // Transformed U = F(X_{(i)}) for Panel C.
    const tSamples = mc.map(preset.cdf);
    const hC = histogram(tSamples, nBins, 0, 1);
    const yMax = Math.max(...densityA, ...hAB.density) * 1.15 || 1;
    return {
      grid: g,
      mcSamples: mc,
      mcHist: hAB,
      gridDensityA: densityA,
      yMaxAB: yMax,
      transformedSamples: tSamples,
      transformedHist: hC,
    };
  }, [preset, dN, dI]);

  void mcSamples;
  void transformedSamples;

  // ── Panel C: Uniform-order-statistic density from the theorem = Beta(i, n−i+1).
  const panelCGrid = Array.from({ length: GRID_POINTS + 1 }, (_, k) => k / GRID_POINTS);
  const panelCDensity = panelCGrid.map((u) => uniformOrderStatisticDensity(u, dI, dN));
  const panelCYMax = Math.max(...panelCDensity, ...transformedHist.density) * 1.15 || 1;

  // ── SVG layouts
  const panelWidth = 320;
  const panelHeight = 220;
  const innerW = panelWidth - MARGIN.left - MARGIN.right;
  const innerH = panelHeight - MARGIN.top - MARGIN.bottom;
  const [xLo, xHi] = preset.domain;
  const xScaleAB = useMemo(() => scaleLinear(xLo, xHi, 0, innerW), [xLo, xHi, innerW]);
  const yScaleAB = useMemo(() => scaleLinear(0, yMaxAB, innerH, 0), [yMaxAB, innerH]);
  const xScaleC = useMemo(() => scaleLinear(0, 1, 0, innerW), [innerW]);
  const yScaleC = useMemo(() => scaleLinear(0, panelCYMax, innerH, 0), [panelCYMax, innerH]);

  const densityPathA = useMemo(
    () =>
      gridDensityA
        .map((y, k) => `${k === 0 ? 'M' : 'L'}${xScaleAB(grid[k])},${yScaleAB(y)}`)
        .join(' '),
    [gridDensityA, grid, xScaleAB, yScaleAB],
  );
  const densityPathC = useMemo(
    () =>
      panelCDensity
        .map((y, k) => `${k === 0 ? 'M' : 'L'}${xScaleC(panelCGrid[k])},${yScaleC(y)}`)
        .join(' '),
    [panelCDensity, panelCGrid, xScaleC, yScaleC],
  );

  return (
    <div className="my-6 rounded-lg border border-stone-200 bg-white p-4 text-sm shadow-sm dark:border-stone-700 dark:bg-stone-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="font-semibold text-stone-800 dark:text-stone-100">
          Order-statistic density browser
        </span>
        <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200">
          §29.3 · Uniform → Beta
        </span>
      </div>

      {/* Controls */}
      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Distribution F
          </span>
          <select
            className="rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
            value={presetKey}
            onChange={(e) => setPresetKey(e.target.value)}
            aria-label="Distribution preset"
          >
            {Object.entries(orderStatPresets).map(([k, pr]) => (
              <option key={k} value={k}>
                {pr.name}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Sample size n = {n}
          </span>
          <input
            type="range"
            min={0}
            max={DISCRETE_NS.length - 1}
            step={1}
            value={DISCRETE_NS.indexOf(n as (typeof DISCRETE_NS)[number])}
            onChange={(e) => {
              const next = DISCRETE_NS[parseInt(e.target.value, 10)];
              setN(next);
              if (i > next) setI(Math.ceil(next / 2));
            }}
            aria-label="Sample size n (discrete)"
          />
          <div className="flex justify-between text-[10px] text-stone-400">
            {DISCRETE_NS.map((v) => (
              <span key={v}>{v}</span>
            ))}
          </div>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Order index i = {Math.min(i, n)} of {n}
          </span>
          <input
            type="range"
            min={1}
            max={n}
            step={1}
            value={Math.min(i, n)}
            onChange={(e) => setI(parseInt(e.target.value, 10))}
            aria-label="Order index i"
          />
        </label>
      </div>

      {/* Three panels */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Panel A: analytic density */}
        <div>
          <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
            A · Analytic density of X<sub>({dI})</sub>
          </div>
          <svg viewBox={`0 0 ${panelWidth} ${panelHeight}`} className="w-full" role="img" aria-label="Analytic density of the i-th order statistic">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              <path d={densityPathA} fill="none" stroke={nonparametricColors.bahadur} strokeWidth={2} />
              <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" />
              <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" />
              {[xLo, (xLo + xHi) / 2, xHi].map((t, k) => (
                <g key={k} transform={`translate(${xScaleAB(t)},${innerH})`}>
                  <line y2={4} stroke="currentColor" />
                  <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                    {fmt(t, 1)}
                  </text>
                </g>
              ))}
              <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor">
                x
              </text>
            </g>
          </svg>
        </div>

        {/* Panel B: MC histogram of X_{(i)} */}
        <div>
          <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
            B · MC histogram of X<sub>({dI})</sub>
          </div>
          <svg viewBox={`0 0 ${panelWidth} ${panelHeight}`} className="w-full" role="img" aria-label="Monte Carlo histogram matching Panel A">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {mcHist.density.map((d, k) => (
                <rect
                  key={k}
                  x={xScaleAB(mcHist.edges[k])}
                  y={yScaleAB(d)}
                  width={Math.max(0, xScaleAB(mcHist.edges[k + 1]) - xScaleAB(mcHist.edges[k]) - 1)}
                  height={Math.max(0, innerH - yScaleAB(d))}
                  fill={nonparametricColors.ecdf}
                  opacity={0.5}
                />
              ))}
              <path d={densityPathA} fill="none" stroke={nonparametricColors.bahadur} strokeWidth={1.5} strokeDasharray="4 2" />
              <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" />
              <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" />
              {[xLo, (xLo + xHi) / 2, xHi].map((t, k) => (
                <g key={k} transform={`translate(${xScaleAB(t)},${innerH})`}>
                  <line y2={4} stroke="currentColor" />
                  <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                    {fmt(t, 1)}
                  </text>
                </g>
              ))}
            </g>
          </svg>
        </div>

        {/* Panel C: Uniform-transform panel (live theorem check) */}
        <div>
          <div className="mb-1 text-xs font-medium text-stone-600 dark:text-stone-300">
            C · U<sub>({dI})</sub> = F(X<sub>({dI})</sub>) vs Beta({dI}, {dN - dI + 1})
          </div>
          <svg viewBox={`0 0 ${panelWidth} ${panelHeight}`} className="w-full" role="img" aria-label="Transformed order statistic matches the Beta theorem">
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {transformedHist.density.map((d, k) => (
                <rect
                  key={k}
                  x={xScaleC(transformedHist.edges[k])}
                  y={yScaleC(d)}
                  width={Math.max(0, xScaleC(transformedHist.edges[k + 1]) - xScaleC(transformedHist.edges[k]) - 1)}
                  height={Math.max(0, innerH - yScaleC(d))}
                  fill={nonparametricColors.families.cauchy}
                  opacity={0.35}
                />
              ))}
              <path d={densityPathC} fill="none" stroke={nonparametricColors.ecdf} strokeWidth={2} />
              <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" />
              <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" />
              {[0, 0.25, 0.5, 0.75, 1].map((t, k) => (
                <g key={k} transform={`translate(${xScaleC(t)},${innerH})`}>
                  <line y2={4} stroke="currentColor" />
                  <text y={16} textAnchor="middle" fontSize={10} fill="currentColor">
                    {t}
                  </text>
                </g>
              ))}
              <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={10} fill="currentColor">
                u ∈ [0, 1]
              </text>
            </g>
          </svg>
        </div>
      </div>

      <div className="mt-3 text-xs text-stone-500 dark:text-stone-400">
        Panel C always agrees with Beta({dI}, {dN - dI + 1}) regardless of the
        F preset — that is Theorem 2 made visible. Try switching between Uniform,
        Exp, and Normal: Panel A/B change shape, Panel C does not.
      </div>
    </div>
  );
}
