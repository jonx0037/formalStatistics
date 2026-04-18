import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  actualCoverageBinomial,
  wilsonInterval,
  clopperPearsonInterval,
  standardNormalInvCDF,
} from './shared/testing';
import { coveragePresets } from '../../data/confidence-intervals-data';

/**
 * BinomialCoverageComparison — interactive binomial-CI coverage diagnostic
 * for Topic 19 §19.6 and §19.8. Computes *exact* actual coverage (not MC)
 * for each procedure across a fine p-grid, exposing the sawtooth pattern
 * that discreteness produces and that Monte Carlo smoothing would erase.
 *
 * Procedures: Wald, Wilson, Clopper–Pearson. Default shows all three
 * overlaid; radio buttons isolate one procedure for clarity.
 */

const MARGIN = { top: 16, right: 20, bottom: 42, left: 54 };
const H = 360;

const C = {
  wald: '#D97706',
  wilson: '#7C3AED',
  cp: '#10B981',
  nominal: '#475569',
  axis: '#64748B',
  grid: '#E2E8F0',
  text: '#1E293B',
};

type Procedure = 'all' | 'wald' | 'wilson' | 'cp';

/**
 * Wald binomial CI with endpoints clamped to [0, 1]. Uses the same
 * standard-normal inverse CDF as Wilson/Clopper–Pearson so all three
 * procedures share quantile implementation — important for coverage
 * comparisons at non-standard α values.
 */
function waldBinomialCI(
  x: number,
  n: number,
  alpha: number,
): { lower: number; upper: number } {
  const z = standardNormalInvCDF(1 - alpha / 2);
  const pHat = x / n;
  const se = Math.sqrt((pHat * (1 - pHat)) / n);
  return {
    lower: Math.max(0, pHat - z * se),
    upper: Math.min(1, pHat + z * se),
  };
}

export default function BinomialCoverageComparison() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 720, 320);

  const [presetIndex, setPresetIndex] = useState(1);
  const preset = coveragePresets[presetIndex];
  const [n, setN] = useState<number>(preset.n);
  const [alpha, setAlpha] = useState<number>(preset.alpha);
  const [procedure, setProcedure] = useState<Procedure>('all');

  const handlePreset = (idx: number): void => {
    setPresetIndex(idx);
    const p = coveragePresets[idx];
    setN(p.n);
    setAlpha(p.alpha);
  };

  const plotW = Math.min(w, 900);

  // Coverage grid — 150 values of p between 0.005 and 0.995.
  const pGrid = useMemo(() => {
    const K = 200;
    return Array.from({ length: K }, (_, i) => 0.005 + (i / (K - 1)) * 0.99);
  }, []);

  const coverageWald = useMemo(
    () => actualCoverageBinomial(waldBinomialCI, n, alpha, pGrid),
    [n, alpha, pGrid],
  );
  const coverageWilson = useMemo(
    () => actualCoverageBinomial(wilsonInterval, n, alpha, pGrid),
    [n, alpha, pGrid],
  );
  const coverageCP = useMemo(
    () => actualCoverageBinomial(clopperPearsonInterval, n, alpha, pGrid),
    [n, alpha, pGrid],
  );

  const xScale = d3.scaleLinear().domain([0, 1]).range([MARGIN.left, plotW - MARGIN.right]);
  const yDomain: [number, number] = [Math.max(0.7, 1 - 3 * alpha), 1.0];
  const yScale = d3
    .scaleLinear()
    .domain(yDomain)
    .range([H - MARGIN.bottom, MARGIN.top]);

  const line = d3
    .line<[number, number]>()
    .x((d) => xScale(d[0]))
    .y((d) => yScale(d[1]));

  const pairs = (arr: number[]): [number, number][] => pGrid.map((p, i) => [p, arr[i]]);

  // Means over the p-grid for the legend.
  const meanOf = (a: number[]): number => a.reduce((s, x) => s + x, 0) / a.length;

  return (
    <div ref={containerRef} className="my-8 rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">Preset</span>
          <select
            value={presetIndex}
            onChange={(e) => handlePreset(Number(e.target.value))}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            {coveragePresets.map((p, i) => (
              <option key={p.name} value={i}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">n = {n}</span>
          <input
            type="range"
            min={5}
            max={1000}
            step={5}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-40"
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
            className="w-32"
          />
        </label>
        <fieldset className="flex items-center gap-2 text-sm">
          <legend className="sr-only">Procedure</legend>
          {(['all', 'wald', 'wilson', 'cp'] as Procedure[]).map((p) => (
            <label key={p} className="flex items-center gap-1">
              <input
                type="radio"
                name="procedure"
                value={p}
                checked={procedure === p}
                onChange={() => setProcedure(p)}
              />
              <span className="capitalize">{p === 'cp' ? 'Clopper–Pearson' : p}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <svg
        width={plotW}
        height={H}
        role="img"
        aria-label="Binomial CI actual coverage across p"
        style={{ maxWidth: '100%', height: 'auto' }}
      >
        {/* Grid */}
        {xScale.ticks(10).map((t) => (
          <line
            key={`xg-${t}`}
            x1={xScale(t)}
            x2={xScale(t)}
            y1={MARGIN.top}
            y2={H - MARGIN.bottom}
            stroke={C.grid}
            strokeWidth={1}
          />
        ))}
        {yScale.ticks(6).map((t) => (
          <line
            key={`yg-${t}`}
            x1={MARGIN.left}
            x2={plotW - MARGIN.right}
            y1={yScale(t)}
            y2={yScale(t)}
            stroke={C.grid}
            strokeWidth={1}
          />
        ))}

        {/* Nominal reference line */}
        <line
          x1={MARGIN.left}
          x2={plotW - MARGIN.right}
          y1={yScale(1 - alpha)}
          y2={yScale(1 - alpha)}
          stroke={C.nominal}
          strokeWidth={1.4}
          strokeDasharray="5,4"
        />
        <text
          x={plotW - MARGIN.right - 6}
          y={yScale(1 - alpha) - 6}
          textAnchor="end"
          fontSize={11}
          fill={C.nominal}
        >
          Nominal {(1 - alpha).toFixed(3)}
        </text>

        {/* Coverage curves */}
        {(procedure === 'all' || procedure === 'wald') && (
          <path
            d={line(pairs(coverageWald)) ?? ''}
            stroke={C.wald}
            strokeWidth={1.5}
            fill="none"
          />
        )}
        {(procedure === 'all' || procedure === 'wilson') && (
          <path
            d={line(pairs(coverageWilson)) ?? ''}
            stroke={C.wilson}
            strokeWidth={1.5}
            fill="none"
          />
        )}
        {(procedure === 'all' || procedure === 'cp') && (
          <path
            d={line(pairs(coverageCP)) ?? ''}
            stroke={C.cp}
            strokeWidth={1.5}
            fill="none"
          />
        )}

        {/* Axes */}
        <line
          x1={MARGIN.left}
          x2={plotW - MARGIN.right}
          y1={H - MARGIN.bottom}
          y2={H - MARGIN.bottom}
          stroke={C.axis}
        />
        <line
          x1={MARGIN.left}
          x2={MARGIN.left}
          y1={MARGIN.top}
          y2={H - MARGIN.bottom}
          stroke={C.axis}
        />
        {xScale.ticks(6).map((t) => (
          <g key={`xt-${t}`}>
            <text
              x={xScale(t)}
              y={H - MARGIN.bottom + 16}
              textAnchor="middle"
              fontSize={11}
              fill={C.text}
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {yScale.ticks(6).map((t) => (
          <g key={`yt-${t}`}>
            <text
              x={MARGIN.left - 6}
              y={yScale(t) + 4}
              textAnchor="end"
              fontSize={11}
              fill={C.text}
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <text
          x={(MARGIN.left + plotW - MARGIN.right) / 2}
          y={H - 6}
          textAnchor="middle"
          fontSize={12}
          fill={C.text}
        >
          True p
        </text>
        <text
          x={14}
          y={(MARGIN.top + H - MARGIN.bottom) / 2}
          textAnchor="middle"
          fontSize={12}
          fill={C.text}
          transform={`rotate(-90, 14, ${(MARGIN.top + H - MARGIN.bottom) / 2})`}
        >
          Actual coverage
        </text>
      </svg>

      <div className="mt-3 grid gap-2 text-sm md:grid-cols-3">
        {(procedure === 'all' || procedure === 'wald') && (
          <div className="rounded bg-white p-2 shadow-sm" style={{ borderLeft: `4px solid ${C.wald}` }}>
            <div className="font-semibold" style={{ color: C.wald }}>
              Wald
            </div>
            <div className="text-slate-700">
              Mean coverage: <span className="font-mono">{meanOf(coverageWald).toFixed(4)}</span>
            </div>
          </div>
        )}
        {(procedure === 'all' || procedure === 'wilson') && (
          <div className="rounded bg-white p-2 shadow-sm" style={{ borderLeft: `4px solid ${C.wilson}` }}>
            <div className="font-semibold" style={{ color: C.wilson }}>
              Wilson
            </div>
            <div className="text-slate-700">
              Mean coverage: <span className="font-mono">{meanOf(coverageWilson).toFixed(4)}</span>
            </div>
          </div>
        )}
        {(procedure === 'all' || procedure === 'cp') && (
          <div className="rounded bg-white p-2 shadow-sm" style={{ borderLeft: `4px solid ${C.cp}` }}>
            <div className="font-semibold" style={{ color: C.cp }}>
              Clopper–Pearson
            </div>
            <div className="text-slate-700">
              Mean coverage: <span className="font-mono">{meanOf(coverageCP).toFixed(4)}</span>
            </div>
          </div>
        )}
      </div>

      <p className="mt-3 text-xs text-slate-500">
        Coverage computed exactly by summing P_p(X = x) over all outcomes whose CI contains p.
        Wald oscillates below nominal; Wilson hugs nominal with small amplitude; Clopper–Pearson
        stays at or above nominal (conservative). BRO2001 Table 1 is the authoritative reference.
      </p>
    </div>
  );
}
