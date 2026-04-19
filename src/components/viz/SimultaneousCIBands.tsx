import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  simultaneousCIBonferroni,
  simultaneousCISidak,
  standardNormalInvCDF,
} from './shared/testing';
import { seededRandom } from './shared/probability';
import { normalSample } from './shared/convergence';
import { multipleTestingColors } from './shared/colorScales';
import {
  simultaneousCIPresets,
  type SimultaneousCIPreset,
} from '../../data/multi-testing-data';

/**
 * SimultaneousCIBands — Topic 20 §20.9 interactive artifact.
 *
 * Constructs m confidence intervals for m parameters (true values supplied by
 * the preset). Three constructions are shown:
 *   • Unadjusted: each CI at individual level α (joint coverage drops to
 *     (1 − α)^m ≈ 1 − mα for small α, failing catastrophically for large m).
 *   • Bonferroni: each CI at α/m → joint coverage ≥ 1 − α (conservative).
 *   • Šidák: each CI at 1 − (1 − α)^(1/m) → joint coverage = 1 − α exactly
 *     under independence (tighter than Bonferroni).
 *
 * Left: horizontal CI bars per parameter, with the true value dot. The
 * selected construction is drawn opaque; the others (if "All") appear
 * semi-transparent for width comparison.
 *
 * Right: joint-coverage histogram bar chart over 200 resamples. Dashed line
 * at the nominal 1 − α target. Shows the cost of over- or under-correction.
 */

const MARGIN = { top: 14, right: 14, bottom: 38, left: 64 };
const H = 360;
const COVERAGE_TRIALS = 200;

const C = {
  unadj: multipleTestingColors.nullMixture,
  bonf: multipleTestingColors.bonf,
  sidak: multipleTestingColors.sidak,
  true: '#B91C1C',
  observed: '#1E3A8A',
  grid: '#E2E8F0',
  axis: '#64748B',
  text: '#1E293B',
  nominalLine: '#475569',
};

type Construction = 'unadj' | 'bonf' | 'sidak' | 'all';

interface CIBands {
  unadj: Array<{ lower: number; upper: number }>;
  bonf: Array<{ lower: number; upper: number }>;
  sidak: Array<{ lower: number; upper: number }>;
}

/** Unadjusted Wald CI with marginal coverage 1 − α each. */
function unadjustedCI(
  means: number[],
  ses: number[],
  alpha: number,
): Array<{ lower: number; upper: number }> {
  const z = standardNormalInvCDF(1 - alpha / 2);
  return means.map((mu, i) => ({
    lower: mu - z * ses[i],
    upper: mu + z * ses[i],
  }));
}

function sampleMeans(
  trueTheta: number[],
  se: number,
  rng: () => number,
): number[] {
  return trueTheta.map((mu) => normalSample(mu, se, rng));
}

function jointCovers(
  cis: Array<{ lower: number; upper: number }>,
  trueTheta: number[],
): boolean {
  for (let i = 0; i < cis.length; i++) {
    if (trueTheta[i] < cis[i].lower || trueTheta[i] > cis[i].upper) return false;
  }
  return true;
}

export default function SimultaneousCIBands(): React.ReactElement {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();
  const w = Math.max(width ?? 780, 320);
  const isWide = w > 760;

  const [presetIndex, setPresetIndex] = useState<number>(0);
  const preset: SimultaneousCIPreset = simultaneousCIPresets[presetIndex];
  const [n, setN] = useState<number>(preset.n);
  const [alpha, setAlpha] = useState<number>(preset.alpha);
  const [construction, setConstruction] = useState<Construction>('all');
  const [seedCounter, setSeedCounter] = useState<number>(0);

  const handlePreset = (idx: number): void => {
    const p = simultaneousCIPresets[idx];
    setPresetIndex(idx);
    setN(p.n);
    setAlpha(p.alpha);
    setSeedCounter(0);
  };

  const m = preset.m;
  const trueTheta = preset.trueTheta;
  const se = 1 / Math.sqrt(n);
  const ses = useMemo(() => new Array(m).fill(se), [m, se]);

  // Current-trial sample means.
  const currentMeans = useMemo(() => {
    const rng = seededRandom(100 + seedCounter);
    return sampleMeans(trueTheta, se, rng);
  }, [trueTheta, se, seedCounter]);

  // Current-trial CIs under each construction.
  const currentBands: CIBands = useMemo(
    () => ({
      unadj: unadjustedCI(currentMeans, ses, alpha),
      bonf: simultaneousCIBonferroni(currentMeans, ses, alpha),
      sidak: simultaneousCISidak(currentMeans, ses, alpha),
    }),
    [currentMeans, ses, alpha],
  );

  // Empirical joint coverage over COVERAGE_TRIALS resamples.
  const coverage = useMemo(() => {
    const rng = seededRandom(10000 + seedCounter * 997);
    let unadjCov = 0;
    let bonfCov = 0;
    let sidakCov = 0;
    for (let t = 0; t < COVERAGE_TRIALS; t++) {
      const means = sampleMeans(trueTheta, se, rng);
      if (jointCovers(unadjustedCI(means, ses, alpha), trueTheta)) unadjCov++;
      if (jointCovers(simultaneousCIBonferroni(means, ses, alpha), trueTheta))
        bonfCov++;
      if (jointCovers(simultaneousCISidak(means, ses, alpha), trueTheta))
        sidakCov++;
    }
    return {
      unadj: unadjCov / COVERAGE_TRIALS,
      bonf: bonfCov / COVERAGE_TRIALS,
      sidak: sidakCov / COVERAGE_TRIALS,
    };
  }, [trueTheta, ses, se, alpha, seedCounter]);

  // Layout — left CI bars plot, right coverage bar chart.
  const leftW = isWide ? Math.max(380, w - 300) : w - 16;
  const rightW = isWide ? 260 : w - 16;
  const plotH = H;

  // X domain for CI bars: span ±z_{α/(2m)} · max(se) around min/max true values.
  const xDomain = useMemo<[number, number]>(() => {
    const widestZ = standardNormalInvCDF(1 - alpha / (2 * m));
    const pad = widestZ * se * 1.25;
    const lo = Math.min(...trueTheta) - pad;
    const hi = Math.max(...trueTheta) + pad;
    return [lo, hi];
  }, [trueTheta, se, alpha, m]);

  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain(xDomain)
        .range([MARGIN.left, leftW - MARGIN.right]),
    [xDomain, leftW],
  );
  const yScale = useMemo(
    () =>
      d3
        .scaleBand<number>()
        .domain(d3.range(m))
        .range([MARGIN.top, plotH - MARGIN.bottom])
        .paddingInner(0.3),
    [m, plotH],
  );

  // Coverage chart scales.
  const covValues: Array<{ key: Construction; label: string; v: number; color: string }> = [
    { key: 'unadj', label: 'Unadjusted', v: coverage.unadj, color: C.unadj },
    { key: 'bonf', label: 'Bonferroni', v: coverage.bonf, color: C.bonf },
    { key: 'sidak', label: 'Šidák', v: coverage.sidak, color: C.sidak },
  ];
  const covXScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0, 1])
        .range([MARGIN.left + 30, rightW - MARGIN.right]),
    [rightW],
  );
  const covYScale = useMemo(
    () =>
      d3
        .scaleBand<number>()
        .domain([0, 1, 2])
        .range([MARGIN.top, plotH - MARGIN.bottom])
        .paddingInner(0.3),
    [plotH],
  );

  const xTicks = useMemo(() => xScale.ticks(5), [xScale]);

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
            {simultaneousCIPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-slate-700">n = {n}</span>
          <input
            type="range"
            min={5}
            max={200}
            step={1}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
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
        <div className="flex items-center gap-1 text-sm">
          <span className="text-slate-700">show</span>
          {(['unadj', 'bonf', 'sidak', 'all'] as Construction[]).map((ck) => (
            <label key={ck} className="flex items-center gap-1">
              <input
                type="radio"
                name="construction"
                checked={construction === ck}
                onChange={() => setConstruction(ck)}
              />
              <span className="text-slate-700">
                {ck === 'unadj'
                  ? 'Unadjusted'
                  : ck === 'bonf'
                    ? 'Bonferroni'
                    : ck === 'sidak'
                      ? 'Šidák'
                      : 'All'}
              </span>
            </label>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setSeedCounter((s) => s + 1)}
          className="rounded bg-slate-700 px-3 py-1 text-sm text-white hover:bg-slate-600"
        >
          Resample
        </button>
      </div>

      <div className={isWide ? 'flex gap-4' : 'flex flex-col gap-4'}>
        {/* LEFT — CI bars */}
        <svg
          width={leftW}
          height={plotH}
          role="img"
          aria-label="Simultaneous CI bars per parameter"
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {/* Gridlines */}
          {xTicks.map((t) => (
            <line
              key={`xg-${t}`}
              x1={xScale(t)}
              x2={xScale(t)}
              y1={MARGIN.top}
              y2={plotH - MARGIN.bottom}
              stroke={C.grid}
              strokeWidth={1}
            />
          ))}

          {/* Per-parameter rows */}
          {d3.range(m).map((i) => {
            const y = (yScale(i) ?? 0) + yScale.bandwidth() / 2;
            const rowOffset = yScale.bandwidth() * 0.22;
            const renderCI = (
              ci: { lower: number; upper: number },
              yi: number,
              color: string,
              strokeW: number,
              opacity: number,
            ): React.ReactElement => (
              <g key={`${color}-${i}`}>
                <line
                  x1={xScale(ci.lower)}
                  x2={xScale(ci.upper)}
                  y1={yi}
                  y2={yi}
                  stroke={color}
                  strokeWidth={strokeW}
                  opacity={opacity}
                />
                <line
                  x1={xScale(ci.lower)}
                  x2={xScale(ci.lower)}
                  y1={yi - 4}
                  y2={yi + 4}
                  stroke={color}
                  strokeWidth={strokeW}
                  opacity={opacity}
                />
                <line
                  x1={xScale(ci.upper)}
                  x2={xScale(ci.upper)}
                  y1={yi - 4}
                  y2={yi + 4}
                  stroke={color}
                  strokeWidth={strokeW}
                  opacity={opacity}
                />
              </g>
            );
            const showAll = construction === 'all';
            const focusKey = showAll ? null : construction;
            const unadjVis =
              showAll || focusKey === 'unadj' ? (focusKey === 'unadj' ? 1 : 0.35) : 0;
            const bonfVis =
              showAll || focusKey === 'bonf' ? (focusKey === 'bonf' ? 1 : 0.35) : 0;
            const sidakVis =
              showAll || focusKey === 'sidak' ? (focusKey === 'sidak' ? 1 : 0.35) : 0;

            return (
              <g key={`row-${i}`}>
                {unadjVis > 0 &&
                  renderCI(currentBands.unadj[i], y - rowOffset, C.unadj, 2.2, unadjVis)}
                {bonfVis > 0 &&
                  renderCI(currentBands.bonf[i], y, C.bonf, 2.4, bonfVis)}
                {sidakVis > 0 &&
                  renderCI(
                    currentBands.sidak[i],
                    y + rowOffset,
                    C.sidak,
                    2.4,
                    sidakVis,
                  )}
                <circle
                  cx={xScale(trueTheta[i])}
                  cy={y}
                  r={4}
                  fill={C.true}
                  stroke="white"
                  strokeWidth={1.2}
                />
                <circle
                  cx={xScale(currentMeans[i])}
                  cy={y}
                  r={2.4}
                  fill={C.observed}
                  opacity={0.75}
                />
              </g>
            );
          })}

          {/* Axes */}
          <line
            x1={MARGIN.left}
            x2={leftW - MARGIN.right}
            y1={plotH - MARGIN.bottom}
            y2={plotH - MARGIN.bottom}
            stroke={C.axis}
          />
          {xTicks.map((t) => (
            <g key={`xt-${t}`}>
              <line
                x1={xScale(t)}
                x2={xScale(t)}
                y1={plotH - MARGIN.bottom}
                y2={plotH - MARGIN.bottom + 4}
                stroke={C.axis}
              />
              <text
                x={xScale(t)}
                y={plotH - MARGIN.bottom + 16}
                textAnchor="middle"
                fontSize={10}
                fill={C.text}
              >
                {t.toFixed(1)}
              </text>
            </g>
          ))}
          {d3
            .range(m)
            .filter((i) => i % Math.ceil(m / 10) === 0)
            .map((i) => (
              <text
                key={`yt-${i}`}
                x={MARGIN.left - 6}
                y={(yScale(i) ?? 0) + yScale.bandwidth() / 2 + 3}
                textAnchor="end"
                fontSize={9}
                fill={C.text}
              >
                i={i + 1}
              </text>
            ))}
          <text
            x={(MARGIN.left + leftW - MARGIN.right) / 2}
            y={plotH - 6}
            textAnchor="middle"
            fontSize={12}
            fill={C.text}
          >
            parameter value
          </text>
        </svg>

        {/* RIGHT — coverage bar chart */}
        <svg
          width={rightW}
          height={plotH}
          role="img"
          aria-label={`Empirical joint coverage over ${COVERAGE_TRIALS} trials`}
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {covValues.map((row, i) => {
            const y = (covYScale(i) ?? 0) + covYScale.bandwidth() / 2;
            const bandH = covYScale.bandwidth() * 0.55;
            return (
              <g key={row.key}>
                <rect
                  x={covXScale(0)}
                  y={y - bandH / 2}
                  width={covXScale(row.v) - covXScale(0)}
                  height={bandH}
                  fill={row.color}
                  opacity={0.85}
                />
                <text
                  x={MARGIN.left - 4}
                  y={y + 4}
                  textAnchor="end"
                  fontSize={11}
                  fill={C.text}
                >
                  {row.label}
                </text>
                <text
                  x={covXScale(row.v) + 5}
                  y={y + 4}
                  textAnchor="start"
                  fontSize={10}
                  fill={C.text}
                  fontFamily="monospace"
                >
                  {(row.v * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
          {/* Nominal 1-α line */}
          <line
            x1={covXScale(1 - alpha)}
            x2={covXScale(1 - alpha)}
            y1={MARGIN.top}
            y2={plotH - MARGIN.bottom}
            stroke={C.nominalLine}
            strokeWidth={1.4}
            strokeDasharray="4 3"
          />
          <text
            x={covXScale(1 - alpha)}
            y={MARGIN.top - 2}
            textAnchor="middle"
            fontSize={10}
            fill={C.nominalLine}
          >
            1 − α = {(1 - alpha).toFixed(3)}
          </text>
          {/* Axis */}
          <line
            x1={covXScale(0)}
            x2={covXScale(1)}
            y1={plotH - MARGIN.bottom}
            y2={plotH - MARGIN.bottom}
            stroke={C.axis}
          />
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={`cov-xt-${t}`}>
              <line
                x1={covXScale(t)}
                x2={covXScale(t)}
                y1={plotH - MARGIN.bottom}
                y2={plotH - MARGIN.bottom + 4}
                stroke={C.axis}
              />
              <text
                x={covXScale(t)}
                y={plotH - MARGIN.bottom + 16}
                textAnchor="middle"
                fontSize={10}
                fill={C.text}
              >
                {t}
              </text>
            </g>
          ))}
          <text
            x={(covXScale(0) + covXScale(1)) / 2}
            y={plotH - 6}
            textAnchor="middle"
            fontSize={12}
            fill={C.text}
          >
            empirical joint coverage ({COVERAGE_TRIALS} trials)
          </text>
        </svg>
      </div>

      <p className="mt-3 text-xs text-slate-500">
        {preset.description} Widest Bonferroni CI uses z ={' '}
        {standardNormalInvCDF(1 - alpha / (2 * m)).toFixed(3)} (vs the
        unadjusted z = {standardNormalInvCDF(1 - alpha / 2).toFixed(3)}).
      </p>
    </div>
  );
}
