import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import { multiTestingMonteCarlo } from './shared/testing';
import type { MultiTestingProcedure } from './shared/testing';
import { multipleTestingColors } from './shared/colorScales';
import {
  fwerVsFdrPresets,
  type FWERvsFDRPreset,
} from '../../data/multi-testing-data';

/**
 * FWERvsFDRComparator — Topic 20 §20.6 interactive artifact.
 *
 * Runs `multiTestingMonteCarlo` for a chosen procedure across a grid of
 * δ (signal-strength) values. Three curves:
 *   • FDR (mean false-discovery proportion) — flat near α·π₀ for BH/BY.
 *   • FWER (probability of ≥1 false rejection) — at α for Bonferroni/Holm,
 *     explodes with δ for BH (procedure doesn't control FWER).
 *   • Power (mean fraction of alternatives rejected) — rising sigmoid in δ.
 *
 * Uses `multiTestingMonteCarlo`, which is a stub (Jonathan fills in). Until
 * the body is implemented, all results are zero; the component shows an
 * informational banner instead of empty axes.
 */

const MARGIN = { top: 14, right: 14, bottom: 38, left: 56 };
const H = 330;
const DELTA_STEPS = 9;
const DELTA_MAX = 4;

const C = {
  fdr: multipleTestingColors.bh,
  fwer: multipleTestingColors.bonf,
  power: multipleTestingColors.altMixture,
  target: '#475569',
  grid: '#E2E8F0',
  axis: '#64748B',
  text: '#1E293B',
};

interface Row {
  delta: number;
  fdr: number;
  fwer: number;
  power: number;
}

const PROCEDURE_OPTIONS: Array<{ key: MultiTestingProcedure; label: string }> = [
  { key: 'bonferroni', label: 'Bonferroni' },
  { key: 'holm', label: 'Holm' },
  { key: 'sidak', label: 'Šidák' },
  { key: 'hochberg', label: 'Hochberg' },
  { key: 'bh', label: 'Benjamini–Hochberg' },
  { key: 'by', label: 'Benjamini–Yekutieli' },
  { key: 'storey', label: 'Storey (adaptive)' },
];

export default function FWERvsFDRComparator(): React.ReactElement {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 780, 320);
  const isWide = w > 760;

  const [presetIndex, setPresetIndex] = useState<number>(0);
  const preset: FWERvsFDRPreset = fwerVsFdrPresets[presetIndex];
  const [procedure, setProcedure] = useState<MultiTestingProcedure>(
    preset.procedure,
  );
  const [m, setM] = useState<number>(preset.m);
  const [pi0, setPi0] = useState<number>(preset.pi0);
  const [alpha, setAlpha] = useState<number>(preset.alpha);
  const [nTrials, setNTrials] = useState<number>(preset.nTrials);
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState<boolean>(false);

  const handlePreset = (idx: number): void => {
    const p = fwerVsFdrPresets[idx];
    setPresetIndex(idx);
    setProcedure(p.procedure);
    setM(p.m);
    setPi0(p.pi0);
    setAlpha(p.alpha);
    setNTrials(p.nTrials);
    setRows([]);
  };

  const handleRun = (): void => {
    setRunning(true);
    // Defer to next frame so the button visually enters "running" state before
    // we block on the synchronous MC loop. For production: move into a Web
    // Worker when m·nTrials·DELTA_STEPS exceeds a profiling threshold.
    window.setTimeout(() => {
      const newRows: Row[] = [];
      for (let s = 0; s < DELTA_STEPS; s++) {
        const delta = (s / (DELTA_STEPS - 1)) * DELTA_MAX;
        const out = multiTestingMonteCarlo(
          procedure,
          m,
          pi0,
          delta,
          alpha,
          nTrials,
          1000 + s * 17,
        );
        newRows.push({
          delta,
          fdr: out.fdr.mean,
          fwer: out.fwer.mean,
          power: out.power.mean,
        });
      }
      setRows(newRows);
      setRunning(false);
    }, 30);
  };

  const allZero = rows.length > 0 && rows.every((r) => r.fdr === 0 && r.fwer === 0 && r.power === 0);

  const plotW = isWide ? Math.max(380, (w - 28) / 2) : w - 16;
  const plotH = H;
  const innerW = plotW - MARGIN.left - MARGIN.right;
  const innerH = plotH - MARGIN.top - MARGIN.bottom;

  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0, DELTA_MAX])
        .range([MARGIN.left, MARGIN.left + innerW]),
    [innerW],
  );
  const yScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0, 1])
        .range([MARGIN.top + innerH, MARGIN.top]),
    [innerH],
  );

  const lineFDR = useMemo(
    () =>
      d3
        .line<Row>()
        .x((d) => xScale(d.delta))
        .y((d) => yScale(d.fdr)),
    [xScale, yScale],
  );
  const lineFWER = useMemo(
    () =>
      d3
        .line<Row>()
        .x((d) => xScale(d.delta))
        .y((d) => yScale(d.fwer)),
    [xScale, yScale],
  );
  const linePower = useMemo(
    () =>
      d3
        .line<Row>()
        .x((d) => xScale(d.delta))
        .y((d) => yScale(d.power)),
    [xScale, yScale],
  );

  const xTicks = xScale.ticks(5);
  const yTicks = yScale.ticks(5);

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border border-slate-200 bg-slate-50 p-4"
    >
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">Preset</span>
          <select
            value={presetIndex}
            onChange={(e) => handlePreset(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {fwerVsFdrPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">Procedure</span>
          <select
            value={procedure}
            onChange={(e) => setProcedure(e.target.value as MultiTestingProcedure)}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {PROCEDURE_OPTIONS.map((p) => (
              <option key={p.key} value={p.key}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">m = {m}</span>
          <input
            type="range"
            min={10}
            max={500}
            step={10}
            value={m}
            onChange={(e) => setM(Number(e.target.value))}
            className="w-24"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">π₀ = {pi0.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={pi0}
            onChange={(e) => setPi0(Number(e.target.value))}
            className="w-24"
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
            className="w-24"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">trials = {nTrials}</span>
          <input
            type="range"
            min={50}
            max={1000}
            step={50}
            value={nTrials}
            onChange={(e) => setNTrials(Number(e.target.value))}
            className="w-24"
          />
        </label>
        <button
          type="button"
          onClick={handleRun}
          disabled={running}
          className="rounded bg-slate-700 px-3 py-1 text-sm text-white hover:bg-slate-600 disabled:opacity-50"
        >
          {running ? 'Running…' : 'Run'}
        </button>
      </div>

      {rows.length === 0 && (
        <div className="rounded bg-white p-6 text-center text-sm text-slate-500">
          Click <em>Run</em> to evaluate the procedure across {DELTA_STEPS}{' '}
          values of δ ∈ [0, {DELTA_MAX}] with m = {m}, π₀ = {pi0.toFixed(2)},
          {nTrials} trials each.
        </div>
      )}

      {allZero && (
        <div className="rounded border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900">
          <strong>All results are 0 — </strong>
          <code>multiTestingMonteCarlo</code> is currently a stub that returns
          zeros. Fill in the accumulator loop in{' '}
          <code>src/components/viz/shared/testing.ts</code> to see real curves.
        </div>
      )}

      {rows.length > 0 && !allZero && (
        <div className={isWide ? 'flex gap-4' : 'flex flex-col gap-4'}>
          {/* LEFT — FDR + FWER */}
          <svg
            width={plotW}
            height={plotH}
            role="img"
            aria-label="FDR and FWER curves vs signal strength δ"
            style={{ maxWidth: '100%', height: 'auto' }}
          >
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

            {/* α·π₀ target for FDR */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={yScale(alpha * pi0)}
              y2={yScale(alpha * pi0)}
              stroke={C.target}
              strokeWidth={1.2}
              strokeDasharray="4 3"
            />
            <text
              x={MARGIN.left + innerW - 8}
              y={yScale(alpha * pi0) - 4}
              textAnchor="end"
              fontSize={10}
              fill={C.target}
            >
              α·π₀ = {(alpha * pi0).toFixed(3)}
            </text>

            {/* FDR line */}
            <path d={lineFDR(rows) ?? ''} stroke={C.fdr} strokeWidth={2.4} fill="none" />
            {rows.map((r, i) => (
              <circle
                key={`fdr-${i}`}
                cx={xScale(r.delta)}
                cy={yScale(r.fdr)}
                r={3}
                fill={C.fdr}
              />
            ))}

            {/* FWER line */}
            <path d={lineFWER(rows) ?? ''} stroke={C.fwer} strokeWidth={2.4} fill="none" />
            {rows.map((r, i) => (
              <circle
                key={`fwer-${i}`}
                cx={xScale(r.delta)}
                cy={yScale(r.fwer)}
                r={3}
                fill={C.fwer}
              />
            ))}

            {/* Legend */}
            <g transform={`translate(${MARGIN.left + 8}, ${MARGIN.top + 4})`}>
              <rect width={120} height={38} fill="white" opacity={0.85} rx={3} />
              <line x1={8} x2={28} y1={14} y2={14} stroke={C.fdr} strokeWidth={2.4} />
              <text x={34} y={17} fontSize={11} fill={C.text}>
                FDR
              </text>
              <line x1={8} x2={28} y1={30} y2={30} stroke={C.fwer} strokeWidth={2.4} />
              <text x={34} y={33} fontSize={11} fill={C.text}>
                FWER
              </text>
            </g>

            {/* Axes */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={MARGIN.top + innerH}
              y2={MARGIN.top + innerH}
              stroke={C.axis}
            />
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + innerH}
              stroke={C.axis}
            />
            {xTicks.map((t) => (
              <g key={`xt-${t}`}>
                <line
                  x1={xScale(t)}
                  x2={xScale(t)}
                  y1={MARGIN.top + innerH}
                  y2={MARGIN.top + innerH + 4}
                  stroke={C.axis}
                />
                <text
                  x={xScale(t)}
                  y={MARGIN.top + innerH + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill={C.text}
                >
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
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
                  {t.toFixed(2)}
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
              δ (signal strength)
            </text>
          </svg>

          {/* RIGHT — Power */}
          <svg
            width={plotW}
            height={plotH}
            role="img"
            aria-label="Power curve vs signal strength δ"
            style={{ maxWidth: '100%', height: 'auto' }}
          >
            {yTicks.map((t) => (
              <line
                key={`pg-${t}`}
                x1={MARGIN.left}
                x2={MARGIN.left + innerW}
                y1={yScale(t)}
                y2={yScale(t)}
                stroke={C.grid}
                strokeWidth={1}
              />
            ))}
            <path d={linePower(rows) ?? ''} stroke={C.power} strokeWidth={2.6} fill="none" />
            {rows.map((r, i) => (
              <circle
                key={`pow-${i}`}
                cx={xScale(r.delta)}
                cy={yScale(r.power)}
                r={3.2}
                fill={C.power}
              />
            ))}

            {/* Axes */}
            <line
              x1={MARGIN.left}
              x2={MARGIN.left + innerW}
              y1={MARGIN.top + innerH}
              y2={MARGIN.top + innerH}
              stroke={C.axis}
            />
            <line
              x1={MARGIN.left}
              x2={MARGIN.left}
              y1={MARGIN.top}
              y2={MARGIN.top + innerH}
              stroke={C.axis}
            />
            {xTicks.map((t) => (
              <g key={`pxt-${t}`}>
                <line
                  x1={xScale(t)}
                  x2={xScale(t)}
                  y1={MARGIN.top + innerH}
                  y2={MARGIN.top + innerH + 4}
                  stroke={C.axis}
                />
                <text
                  x={xScale(t)}
                  y={MARGIN.top + innerH + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill={C.text}
                >
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            {yTicks.map((t) => (
              <g key={`pyt-${t}`}>
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
                  {t.toFixed(2)}
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
              δ (signal strength)
            </text>
            <text
              x={14}
              y={MARGIN.top + innerH / 2}
              textAnchor="middle"
              fontSize={12}
              fill={C.text}
              transform={`rotate(-90, 14, ${MARGIN.top + innerH / 2})`}
            >
              power
            </text>
          </svg>
        </div>
      )}

      <p className="mt-3 text-xs text-slate-500">
        Procedure: <strong>{PROCEDURE_OPTIONS.find((p) => p.key === procedure)?.label}</strong>,
        m = {m}, π₀ = {pi0.toFixed(2)}, α = {alpha.toFixed(3)}, {nTrials} MC
        trials at each of {DELTA_STEPS} δ grid points. FDR target under
        independence: α·π₀ = {(alpha * pi0).toFixed(3)}.
      </p>
    </div>
  );
}
