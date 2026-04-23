/**
 * ECDFDKWBandExplorer — interactive for Topic 29 §29.5 (DKW inequality).
 *
 * Renders an ECDF realization with the DKW 95%-confidence envelope ±ε_n
 * shaded around it, against the true CDF in black. The envelope contains F
 * with probability ≥ 1 - α (Massart's tight constant); the "Run 200
 * resamples" button computes the empirical coverage.
 *
 * Locked G8 gotcha: containment readout uses literal "yes"/"no" text
 * (serif fonts don't carry ✓/✗ glyphs reliably — they fall back to ¤ on
 * some platforms).
 */
import { useDeferredValue, useMemo, useState } from 'react';
import { ecdfFn, ksStatistic, dkwBand } from './shared/nonparametric';
import {
  ecdfDkwPresets,
  seededUniform,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { nonparametricColors } from './shared/colorScales';

const MARGIN = { top: 28, right: 20, bottom: 40, left: 48 };
const GRID_POINTS = 200;
const COVERAGE_REPLICATES = 200;
const ALPHAS = [0.01, 0.05, 0.1, 0.2] as const;

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

function drawSample(preset: DistributionPreset, n: number, rng: () => number): number[] {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = preset.sampler(rng);
  return out;
}

export default function ECDFDKWBandExplorer() {
  const [presetKey, setPresetKey] = useState<string>('normal');
  const [n, setN] = useState(100);
  const [alpha, setAlpha] = useState<number>(0.05);
  const [seed, setSeed] = useState(42);
  const [coverage, setCoverage] = useState<{ pct: number; runs: number } | null>(null);

  const dPreset = useDeferredValue(presetKey);
  const dN = useDeferredValue(n);
  const dAlpha = useDeferredValue(alpha);
  const dSeed = useDeferredValue(seed);
  const preset = ecdfDkwPresets[dPreset];

  const { sortedSample, eps, dn, contains, curve } = useMemo(() => {
    const rng = seededUniform(dSeed);
    const s = drawSample(preset, dN, rng);
    const sorted = [...s].sort((a, b) => a - b);
    const e = dkwBand(dN, dAlpha);
    const d = ksStatistic(s, preset.cdf);
    // For continuous F, `ksStatistic` gives the exact sup, so `dn <= eps` is
    // the exact containment check — more accurate and cheaper than the finite
    // grid we used to walk. (Copilot PR #33 discussion.)
    const c = d <= e;
    const fFn = ecdfFn(s);
    const [lo, hi] = preset.domain;
    const grid = Array.from({ length: GRID_POINTS + 1 }, (_, i) => {
      const x = lo + (i / GRID_POINTS) * (hi - lo);
      return { x, fTrue: preset.cdf(x), fn: fFn(x) };
    });
    return { sortedSample: sorted, eps: e, dn: d, contains: c, curve: grid };
  }, [preset, dN, dAlpha, dSeed]);

  function runCoverage() {
    let hits = 0;
    const rng = seededUniform(dSeed + 1000);
    for (let r = 0; r < COVERAGE_REPLICATES; r++) {
      const s = drawSample(preset, dN, rng);
      if (ksStatistic(s, preset.cdf) <= eps) hits++;
    }
    setCoverage({ pct: hits / COVERAGE_REPLICATES, runs: COVERAGE_REPLICATES });
  }

  // ── SVG geometry
  const width = 520;
  const height = 320;
  const innerW = width - MARGIN.left - MARGIN.right;
  const innerH = height - MARGIN.top - MARGIN.bottom;
  const [xLo, xHi] = preset.domain;
  const xScale = scaleLinear(xLo, xHi, 0, innerW);
  const yScale = scaleLinear(0, 1, innerH, 0);

  // Memoize the three SVG path strings — each depends only on memoized data.
  const { stepPath, envelopeArea, truePath } = useMemo(() => {
    // ECDF step-function
    let step = '';
    if (sortedSample.length > 0) {
      const parts: string[] = [];
      parts.push(`M${xScale(xLo)},${yScale(0)}`);
      let prev = 0;
      for (let i = 0; i < sortedSample.length; i++) {
        const xi = sortedSample[i];
        const yPrev = prev;
        const yCur = (i + 1) / sortedSample.length;
        parts.push(`L${xScale(xi)},${yScale(yPrev)}`);
        parts.push(`L${xScale(xi)},${yScale(yCur)}`);
        prev = yCur;
      }
      parts.push(`L${xScale(xHi)},${yScale(prev)}`);
      step = parts.join(' ');
    }
    // DKW envelope centered on the ECDF (Copilot PR #33 discussion): the
    // band [F_n(x) - ε_n, F_n(x) + ε_n] is the distribution-free confidence
    // envelope for F(x), so the true CDF is what the band is *containing*,
    // and the ECDF is the center line.
    const env = curve
      .map((g) => {
        const upper = Math.min(1, g.fn + eps);
        return `${xScale(g.x)},${yScale(upper)}`;
      })
      .concat(
        [...curve].reverse().map((g) => {
          const lower = Math.max(0, g.fn - eps);
          return `${xScale(g.x)},${yScale(lower)}`;
        }),
      )
      .join(' ');
    const tru = curve
      .map((g, i) => `${i === 0 ? 'M' : 'L'}${xScale(g.x)},${yScale(g.fTrue)}`)
      .join(' ');
    return { stepPath: step, envelopeArea: env, truePath: tru };
  }, [sortedSample, curve, eps, xScale, yScale, xLo, xHi]);

  return (
    <div className="my-6 rounded-lg border border-stone-200 bg-white p-4 text-sm shadow-sm dark:border-stone-700 dark:bg-stone-900">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <span className="font-semibold text-stone-800 dark:text-stone-100">
          ECDF and the DKW envelope
        </span>
        <span className="rounded bg-teal-100 px-2 py-0.5 text-xs text-teal-800 dark:bg-teal-900/40 dark:text-teal-200">
          §29.5 · Dvoretzky–Kiefer–Wolfowitz
        </span>
      </div>

      {/* Controls */}
      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">Distribution</span>
          <select
            className="rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
            value={presetKey}
            onChange={(e) => setPresetKey(e.target.value)}
            aria-label="Distribution preset"
          >
            {Object.entries(ecdfDkwPresets).map(([k, pr]) => (
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
            min={Math.log(20)}
            max={Math.log(5000)}
            step={0.01}
            value={Math.log(n)}
            onChange={(e) => setN(Math.round(Math.exp(parseFloat(e.target.value))))}
            aria-label="Sample size n (log scale)"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Confidence 1 − α
          </span>
          <select
            className="rounded border border-stone-300 bg-white px-2 py-1 text-sm dark:border-stone-600 dark:bg-stone-800"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            aria-label="Confidence level"
          >
            {ALPHAS.map((a) => (
              <option key={a} value={a}>
                {(1 - a).toFixed(2)} (α = {a})
              </option>
            ))}
          </select>
        </label>

        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-stone-600 dark:text-stone-300">
            Seed = {seed}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setSeed((s) => s + 1)}
              className="rounded border border-stone-300 bg-stone-50 px-2 py-1 text-sm hover:bg-stone-100 dark:border-stone-600 dark:bg-stone-800 dark:hover:bg-stone-700"
            >
              Resample
            </button>
            <button
              type="button"
              onClick={runCoverage}
              className="rounded border border-stone-300 bg-teal-50 px-2 py-1 text-sm hover:bg-teal-100 dark:border-stone-600 dark:bg-teal-900/30 dark:hover:bg-teal-900/50"
            >
              Run {COVERAGE_REPLICATES}
            </button>
          </div>
        </div>
      </div>

      {/* Plot */}
      <svg
        role="img"
        aria-label="ECDF of a sample with the DKW confidence envelope shaded and the true CDF overlaid"
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* DKW envelope (shaded polygon) */}
          <polygon
            points={envelopeArea}
            fill={nonparametricColors.dkw}
            fillOpacity={0.25}
          />
          {/* True CDF */}
          <path d={truePath} fill="none" stroke="currentColor" strokeWidth={1.5} />
          {/* ECDF step-function */}
          <path d={stepPath} fill="none" stroke={nonparametricColors.ecdf} strokeWidth={1.5} />
          {/* Axes */}
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="currentColor" strokeWidth={1} />
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="currentColor" strokeWidth={1} />
          {/* x-axis ticks */}
          {[xLo, (xLo + xHi) / 2, xHi].map((t, i) => (
            <g key={i} transform={`translate(${xScale(t)},${innerH})`}>
              <line y2={5} stroke="currentColor" />
              <text y={18} textAnchor="middle" fontSize={10} fill="currentColor">
                {fmt(t, 1)}
              </text>
            </g>
          ))}
          {/* y-axis ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
            <g key={i} transform={`translate(0,${yScale(t)})`}>
              <line x2={-5} stroke="currentColor" />
              <text x={-8} dy="0.32em" textAnchor="end" fontSize={10} fill="currentColor">
                {t}
              </text>
            </g>
          ))}
          <text x={innerW / 2} y={innerH + 32} textAnchor="middle" fontSize={11} fill="currentColor">
            x
          </text>
          <text
            transform={`translate(-36,${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={11}
            fill="currentColor"
          >
            F<tspan baselineShift="sub" fontSize={9}>n</tspan>(x), F(x)
          </text>
        </g>
      </svg>

      {/* Readouts */}
      <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-stone-600 dark:text-stone-300 sm:grid-cols-4">
        <div>
          D<sub>n</sub> = <span className="font-mono">{fmt(dn, 4)}</span>
        </div>
        <div>
          ε<sub>n</sub> = <span className="font-mono">{fmt(eps, 4)}</span>
        </div>
        <div>
          Envelope contains F:{' '}
          <span className="font-mono">{contains ? 'yes' : 'no'}</span>
        </div>
        <div>
          DKW margin = D<sub>n</sub> / ε<sub>n</sub> ={' '}
          <span className="font-mono">{fmt(dn / eps, 2)}</span>
        </div>
        {coverage && (
          <div className="col-span-2 sm:col-span-4 rounded bg-teal-50 px-2 py-1 dark:bg-teal-900/30">
            Empirical coverage over {coverage.runs} resamples:{' '}
            <span className="font-mono">{(coverage.pct * 100).toFixed(1)}%</span>{' '}
            (target ≥ {((1 - alpha) * 100).toFixed(0)}%)
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-4 text-xs text-stone-500 dark:text-stone-400">
        <span>
          <span
            className="inline-block h-2 w-3"
            style={{ background: nonparametricColors.dkw, opacity: 0.5 }}
          />{' '}
          DKW envelope
        </span>
        <span>
          <span
            className="inline-block h-2 w-3"
            style={{ background: nonparametricColors.ecdf }}
          />{' '}
          ECDF F<sub>n</sub>
        </span>
        <span>
          <span className="inline-block h-2 w-3" style={{ background: 'currentColor' }} />{' '}
          true CDF
        </span>
      </div>
    </div>
  );
}
