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
  // Dedicated rejection-outcome colors so `altMixture` / `nullMixture` stay
  // reserved for their literal §20.4 mixture-figure use.
  falseReject: '#DC2626', // red — a rejected null (false discovery)
  trueReject: multipleTestingColors.bh, // green — a rejected alternative
  missedAlt: '#FDE68A', // pale amber — alt that no procedure rejected (power loss)
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

/**
 * Per-rank rejection threshold for each procedure. Returns the threshold a
 * p-value at rank k (1-based, ascending order) must fall at or below for the
 * procedure to reject the hypothesis corresponding to that rank.
 *
 * Bonferroni and Šidák are rank-independent (flat lines). Holm / Hochberg
 * share the step-up / step-down threshold $\alpha / (m - k + 1)$. BH rises
 * linearly as $k \alpha / m$; BY is BH scaled by $1/c_m$; Storey is BH
 * scaled by $1/\hat\pi_0$.
 */
function thresholdAtRank(
  key: ProcedureKey,
  k: number,
  m: number,
  alpha: number,
  cm: number,
  pi0Hat: number,
): number {
  switch (key) {
    case 'bonferroni':
      return alpha / m;
    case 'sidak':
      return 1 - Math.pow(1 - alpha, 1 / m);
    case 'holm':
    case 'hochberg':
      return alpha / (m - k + 1);
    case 'bh':
      return (k * alpha) / m;
    case 'by':
      return (k * alpha) / (m * cm);
    case 'storey':
      return (k * alpha) / (m * Math.max(pi0Hat, 1 / m));
  }
}

/** Storey's π̂₀ estimate at λ = 0.5, clipped to [1/m, 1]. */
function estimatePi0(pvals: number[]): number {
  const m = pvals.length;
  if (m === 0) return 1;
  const lambda = 0.5;
  const nAboveLambda = pvals.filter((p) => p > lambda).length;
  const raw = nAboveLambda / (m * (1 - lambda));
  return Math.min(Math.max(raw, 1 / m), 1);
}

/** Harmonic number c_m = Σ 1/k, used by BY's normalizer. */
function harmonicNumber(m: number): number {
  let s = 0;
  for (let k = 1; k <= m; k++) s += 1 / k;
  return s;
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
  const cm = useMemo(() => harmonicNumber(m), [m]);
  const pi0Hat = useMemo(() => estimatePi0(pvals), [pvals]);

  // Sorted p-values with back-pointer to original index. Ranks are 1-based.
  // The strip chart plots p-value vs. rank (per brief §5 Component 1), which
  // is what makes the rank-dependent threshold curves (BH diagonal, Holm
  // staircase, …) meaningful on the same axes as the p-values themselves.
  const sortedEntries = useMemo(() => {
    const entries = pvals.map((p, i) => ({ p, originalIndex: i, isNull: i < m0 }));
    entries.sort((a, b) => a.p - b.p);
    return entries;
  }, [pvals, m0]);

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

  // X: rank 1..m (1-based, ascending by p-value). Pad ½ on each side so the
  // extreme dots don't abut the axes.
  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0.5, m + 0.5])
        .range([MARGIN.left, MARGIN.left + innerW]),
    [m, innerW],
  );

  // Per-procedure threshold curve as an SVG path over ranks 1..m.
  const thresholdPaths = useMemo(() => {
    const line = d3
      .line<{ k: number; t: number }>()
      .x((d) => xScale(d.k))
      .y((d) => yScale(Math.min(Math.max(d.t, pMin), 1)));
    return perProcedure.map((proc) => {
      const pts: Array<{ k: number; t: number }> = [];
      for (let k = 1; k <= m; k++) {
        pts.push({ k, t: thresholdAtRank(proc.key, k, m, alpha, cm, pi0Hat) });
      }
      return { key: proc.key, color: proc.color, path: line(pts) ?? '' };
    });
  }, [perProcedure, m, alpha, cm, pi0Hat, xScale, yScale]);

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

          {/* Per-procedure threshold curves over rank. Bonferroni / Šidák are
              flat; Holm / Hochberg are a staircase; BH / BY / Storey rise
              linearly. */}
          {thresholdPaths.map(({ key, color, path }) => (
            <path
              key={`thr-${key}`}
              d={path}
              stroke={color}
              strokeWidth={1.6}
              fill="none"
              strokeDasharray="5 3"
              opacity={0.85}
            />
          ))}

          {/* P-value dots plotted at (rank, p). Fill encodes truth × rejection:
              • Non-rejected null  → slate
              • Non-rejected alt   → amber-muted (missed-alternative semantics)
              • Rejected null      → falseReject (false-positive warning)
              • Rejected alt       → trueReject (true-positive green) */}
          {sortedEntries.map((entry, idx) => {
            const rank = idx + 1;
            const clamped = Math.max(entry.p, pMin);
            const anyReject = perProcedure.some(
              (proc) => proc.rejected[entry.originalIndex],
            );
            let fill: string;
            if (anyReject && entry.isNull) fill = C.falseReject;
            else if (anyReject && !entry.isNull) fill = C.trueReject;
            else if (!anyReject && entry.isNull) fill = C.nullMixture;
            else fill = C.missedAlt;
            return (
              <circle
                key={`p-${entry.originalIndex}`}
                cx={xScale(rank)}
                cy={yScale(clamped)}
                r={rDot}
                fill={fill}
                opacity={m > 200 ? 0.6 : 0.9}
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
            rank <tspan style={{ fontStyle: 'italic' }}>k</tspan> (p-values in
            ascending order; m₀ = {m0} nulls, m₁ = {m - m0} alternatives)
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
                    style={{ color: p.V > 0 ? C.falseReject : undefined }}
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
