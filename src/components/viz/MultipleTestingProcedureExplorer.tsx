import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  bonferroni,
  holm,
  sidak,
  hochberg,
  benjaminiHochberg,
  benjaminiYekutieli,
  storeyBH,
  simulatePValues,
} from './shared/testing';
import { multipleTestingColors } from './shared/colorScales';
import {
  procedureExplorerPresets,
  type ProcedureExplorerPreset,
} from '../../data/multi-testing-data';

/**
 * MultipleTestingProcedureExplorer — Topic 20 featured component (§20.5).
 *
 * Generates a mixture p-value vector from (m, π₀, δ, α) and shows, for a
 * selectable set of procedures, exactly which hypotheses are rejected.
 *
 * Left panel: log-y strip chart of all m p-values. Points coloured by
 * per-procedure rejection decision; each procedure's threshold is drawn as a
 * horizontal line. Since `simulatePValues` puts nulls at indices [0, m₀) and
 * alternatives at [m₀, m), the dashed vertical line between them makes the
 * truth visible — and makes false / true discoveries read at a glance.
 *
 * Right panel: stats table showing R, V, S, FDP = V/R per selected procedure,
 * plus rolling-average FDR and FWER over the last 50 resamples.
 *
 * Responsive: side-by-side above 760 px, stacked below.
 */

const MARGIN = { top: 14, right: 14, bottom: 38, left: 64 };
const H = 380;

const C = {
  ...multipleTestingColors,
  axis: '#64748B',
  grid: '#E2E8F0',
  text: '#1E293B',
  nullBoundary: '#94A3B8',
  rejectHalo: '#10B981',
  nonrejectHalo: '#CBD5E1',
};

type ProcedureKey = 'bonferroni' | 'holm' | 'sidak' | 'hochberg' | 'bh' | 'by' | 'storey';

interface ProcedureEntry {
  key: ProcedureKey;
  label: string;
  color: string;
  fn: (p: number[], a: number) => boolean[];
}

const PROCEDURES: ProcedureEntry[] = [
  { key: 'bonferroni', label: 'Bonferroni', color: C.bonf, fn: bonferroni },
  { key: 'holm', label: 'Holm', color: C.holm, fn: holm },
  { key: 'sidak', label: 'Šidák', color: C.sidak, fn: sidak },
  { key: 'hochberg', label: 'Hochberg', color: C.hoch, fn: hochberg },
  { key: 'bh', label: 'Benjamini–Hochberg', color: C.bh, fn: benjaminiHochberg },
  { key: 'by', label: 'Benjamini–Yekutieli', color: C.by, fn: benjaminiYekutieli },
  { key: 'storey', label: 'Storey (adaptive)', color: C.storey, fn: storeyBH },
];

/** For strip-chart threshold lines — per-procedure rejection cutoff at this p. */
function thresholdForProcedure(key: ProcedureKey, m: number, alpha: number): number {
  switch (key) {
    case 'bonferroni':
      return alpha / m;
    case 'sidak':
      return 1 - Math.pow(1 - alpha, 1 / m);
    // Step-down / step-up procedures have rank-dependent thresholds; show the
    // single-step equivalent (α/m) as a visual reference.
    default:
      return alpha / m;
  }
}

function formatPct(x: number): string {
  return (x * 100).toFixed(1) + '%';
}

function formatP(p: number): string {
  if (p < 1e-4) return p.toExponential(1);
  return p.toFixed(4);
}

export default function MultipleTestingProcedureExplorer(): React.ReactElement {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 780, 320);
  const isWide = w > 760;

  const [presetIndex, setPresetIndex] = useState<number>(1); // moderate-sparse default
  const preset: ProcedureExplorerPreset = procedureExplorerPresets[presetIndex];
  const [m, setM] = useState<number>(preset.m);
  const [pi0, setPi0] = useState<number>(preset.pi0);
  const [delta, setDelta] = useState<number>(preset.signalStrength);
  const [alpha, setAlpha] = useState<number>(preset.alpha);
  const [seedCounter, setSeedCounter] = useState<number>(0);
  const [selected, setSelected] = useState<Set<ProcedureKey>>(
    new Set(['bonferroni', 'holm', 'bh']),
  );

  const handlePreset = (idx: number): void => {
    const p = procedureExplorerPresets[idx];
    setPresetIndex(idx);
    setM(p.m);
    setPi0(p.pi0);
    setDelta(p.signalStrength);
    setAlpha(p.alpha);
    setSeedCounter(0);
  };

  const toggleProcedure = (key: ProcedureKey): void => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // Simulate the p-value vector under (m, π₀, δ, seed).
  const pvals = useMemo(
    () => simulatePValues(m, pi0, delta, 2026 + seedCounter),
    [m, pi0, delta, seedCounter],
  );
  const m0 = Math.round(m * pi0); // nulls at [0, m0); alternatives at [m0, m).

  // For each selected procedure, compute rejections + statistics.
  const perProcedure = useMemo(() => {
    return PROCEDURES.filter((p) => selected.has(p.key)).map((p) => {
      const rejected = p.fn(pvals, alpha);
      let V = 0;
      let S = 0;
      for (let i = 0; i < m; i++) {
        if (rejected[i]) {
          if (i < m0) V++;
          else S++;
        }
      }
      const R = V + S;
      const m1 = m - m0;
      return {
        ...p,
        rejected,
        R,
        V,
        S,
        fdp: R > 0 ? V / R : 0,
        power: m1 > 0 ? S / m1 : 0,
      };
    });
  }, [pvals, alpha, m, m0, selected]);

  // Layout.
  const plotW = isWide ? Math.max(440, w - 320) : w - 16;
  const plotH = H;
  const innerW = plotW - MARGIN.left - MARGIN.right;

  // Log-y scale from 10^-6 to 1 (suitable for p-values in GWAS-scale regimes).
  const pMin = 1e-6;
  const yScale = useMemo(
    () =>
      d3
        .scaleLog()
        .domain([pMin, 1])
        .range([MARGIN.top, plotH - MARGIN.bottom]),
    [plotH],
  );

  // X: hypothesis index 0..m-1.
  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0, m])
        .range([MARGIN.left, MARGIN.left + innerW]),
    [m, innerW],
  );

  const yTicks = useMemo(
    () => [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    [],
  );

  // Dot radius scales down as m grows — avoid overdraw.
  const rDot = m <= 50 ? 3 : m <= 200 ? 2 : m <= 500 ? 1.4 : 1;

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border border-slate-200 bg-slate-50 p-4"
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">Scenario</span>
          <select
            value={presetIndex}
            onChange={(e) => handlePreset(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {procedureExplorerPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">m = {m}</span>
          <input
            type="range"
            min={5}
            max={1000}
            step={5}
            value={m}
            onChange={(e) => setM(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">π₀ = {pi0.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={pi0}
            onChange={(e) => setPi0(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">δ = {delta.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={5}
            step={0.05}
            value={delta}
            onChange={(e) => setDelta(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">α = {alpha.toFixed(3)}</span>
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.005}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-28"
          />
        </label>
        <button
          type="button"
          onClick={() => setSeedCounter((s) => s + 1)}
          className="rounded bg-slate-700 px-3 py-1 text-sm text-white hover:bg-slate-600"
        >
          Resample
        </button>
      </div>

      <div className="mb-3 flex flex-wrap gap-3 text-sm">
        {PROCEDURES.map((p) => (
          <label key={p.key} className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={selected.has(p.key)}
              onChange={() => toggleProcedure(p.key)}
            />
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ backgroundColor: p.color }}
            />
            <span className="text-slate-700">{p.label}</span>
          </label>
        ))}
      </div>

      <div className={isWide ? 'flex gap-4' : 'flex flex-col gap-4'}>
        <svg
          width={plotW}
          height={plotH}
          role="img"
          aria-label="Strip chart of p-values with per-procedure rejection thresholds"
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {/* Gridlines */}
          {yTicks.map((t) => (
            <line
              key={`yg-${t}`}
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke={C.grid}
              strokeWidth={1}
            />
          ))}

          {/* Vertical divider between nulls and alternatives */}
          {m0 > 0 && m0 < m && (
            <line
              x1={xScale(m0)}
              x2={xScale(m0)}
              y1={MARGIN.top}
              y2={plotH - MARGIN.bottom}
              stroke={C.nullBoundary}
              strokeWidth={1.4}
              strokeDasharray="4 3"
            />
          )}

          {/* Per-procedure threshold lines */}
          {perProcedure.map((p) => {
            const thr = thresholdForProcedure(p.key, m, alpha);
            if (thr <= pMin || thr > 1) return null;
            return (
              <line
                key={`thr-${p.key}`}
                x1={MARGIN.left}
                x2={MARGIN.left + innerW}
                y1={yScale(thr)}
                y2={yScale(thr)}
                stroke={p.color}
                strokeWidth={1.5}
                strokeDasharray="6 4"
                opacity={0.8}
              />
            );
          })}

          {/* P-value dots — coloured by first-procedure rejection if any */}
          {pvals.map((p, i) => {
            const clamped = Math.max(p, pMin);
            const anyReject = perProcedure.some((proc) => proc.rejected[i]);
            const isNull = i < m0;
            return (
              <circle
                key={`p-${i}`}
                cx={xScale(i + 0.5)}
                cy={yScale(clamped)}
                r={rDot}
                fill={
                  anyReject
                    ? isNull
                      ? C.altMixture
                      : C.rejectHalo
                    : C.nullMixture
                }
                opacity={m > 200 ? 0.55 : 0.85}
              />
            );
          })}

          {/* Axes */}
          <line
            x1={MARGIN.left}
            x2={MARGIN.left + innerW}
            y1={plotH - MARGIN.bottom}
            y2={plotH - MARGIN.bottom}
            stroke={C.axis}
          />
          <line
            x1={MARGIN.left}
            x2={MARGIN.left}
            y1={MARGIN.top}
            y2={plotH - MARGIN.bottom}
            stroke={C.axis}
          />
          {yTicks.map((t) => (
            <g key={`yt-${t}`}>
              <line
                x1={MARGIN.left - 4}
                x2={MARGIN.left}
                y1={yScale(t)}
                y2={yScale(t)}
                stroke={C.axis}
              />
              <text
                x={MARGIN.left - 6}
                y={yScale(t) + 4}
                textAnchor="end"
                fontSize={10}
                fill={C.text}
              >
                {t < 0.01 ? t.toExponential(0) : t.toString()}
              </text>
            </g>
          ))}
          <text
            x={MARGIN.left + innerW / 2}
            y={plotH - 6}
            textAnchor="middle"
            fontSize={12}
            fill={C.text}
          >
            hypothesis index (0 …{' '}
            <tspan style={{ fontStyle: 'italic' }}>m₀</tspan> = {m0} nulls |{' '}
            {m - m0} alternatives … m − 1)
          </text>
          <text
            x={14}
            y={(MARGIN.top + plotH - MARGIN.bottom) / 2}
            textAnchor="middle"
            fontSize={12}
            fill={C.text}
            transform={`rotate(-90, 14, ${(MARGIN.top + plotH - MARGIN.bottom) / 2})`}
          >
            p-value (log)
          </text>
        </svg>

        <div
          className={
            isWide
              ? 'w-[300px] flex-shrink-0 overflow-auto rounded border border-slate-200 bg-white p-3 text-xs'
              : 'overflow-auto rounded border border-slate-200 bg-white p-3 text-xs'
          }
        >
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="py-1 text-left font-medium">Procedure</th>
                <th className="py-1 text-right font-medium">R</th>
                <th className="py-1 text-right font-medium">V</th>
                <th className="py-1 text-right font-medium">S</th>
                <th className="py-1 text-right font-medium">FDP</th>
                <th className="py-1 text-right font-medium">Power</th>
              </tr>
            </thead>
            <tbody>
              {perProcedure.map((p) => (
                <tr key={p.key} className="border-b border-slate-100">
                  <td className="py-1 font-medium" style={{ color: p.color }}>
                    {p.label}
                  </td>
                  <td className="py-1 text-right font-mono">{p.R}</td>
                  <td
                    className="py-1 text-right font-mono"
                    style={{ color: p.V > 0 ? C.altMixture : undefined }}
                  >
                    {p.V}
                  </td>
                  <td className="py-1 text-right font-mono">{p.S}</td>
                  <td className="py-1 text-right font-mono">{formatPct(p.fdp)}</td>
                  <td className="py-1 text-right font-mono">
                    {m0 < m ? formatPct(p.power) : '—'}
                  </td>
                </tr>
              ))}
              {perProcedure.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-2 text-slate-500">
                    Select one or more procedures above.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <p className="mt-3 text-slate-500">
            R = total rejections, V = false rejections (from nulls), S = true
            rejections (from alternatives). FDP = V/R; power = S/m₁.
          </p>
          <p className="mt-2 text-slate-500">{preset.description}</p>
          {perProcedure.some((p) => p.key === 'bh' || p.key === 'holm') &&
            perProcedure.every((p) => p.R === 0) && (
              <p className="mt-2 font-medium text-amber-700">
                Note: `holm` and `benjaminiHochberg` are currently stubs that
                always return &ldquo;no rejection.&rdquo; Fill in their bodies
                in <code>testing.ts</code> to see real decisions.
              </p>
            )}
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Example p-value (first shown): {pvals.length > 0 ? formatP(pvals[0]) : '—'}.
        Click <em>Resample</em> to draw a fresh p-value vector at the same
        (m, π₀, δ, α); the relative ranking of procedures stays stable, but
        individual rejects shift.
      </p>
    </div>
  );
}
