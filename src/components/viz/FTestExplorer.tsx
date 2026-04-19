import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  fDensity,
  fCDF,
  fQuantile,
  noncentralFDensity,
  noncentralFPower,
} from './shared/regression';
import {
  F_TEST_EXPLORER_PRESETS,
  type FTestPreset,
} from '../../data/regression-data';

// ── Layout ───────────────────────────────────────────────────────────────────
const MARGIN = { top: 20, right: 24, bottom: 44, left: 52 };
const MAIN_H = 320;
const MINI_H = 120;

// ── Visual palette ───────────────────────────────────────────────────────────
const NULL_COLOR = '#2563EB'; // blue
const ALT_COLOR = '#D97706'; // amber
const REJECT_COLOR = '#9CA3AF';
const AXIS_COLOR = '#9CA3AF';

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

export default function FTestExplorer() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);

  // Controls — initialized from the partialFTest preset (the canonical §21.8 example).
  const [activePreset, setActivePreset] = useState<string>('partialFTest');
  const [df1, setDf1] = useState<number>(F_TEST_EXPLORER_PRESETS.partialFTest.df1);
  const [df2, setDf2] = useState<number>(F_TEST_EXPLORER_PRESETS.partialFTest.df2);
  const [lambda, setLambda] = useState<number>(F_TEST_EXPLORER_PRESETS.partialFTest.lambda);
  const [alpha, setAlpha] = useState<number>(F_TEST_EXPLORER_PRESETS.partialFTest.alpha);

  const applyPreset = (key: string) => {
    const p: FTestPreset | undefined = F_TEST_EXPLORER_PRESETS[key];
    if (!p) return;
    setActivePreset(key);
    setDf1(p.df1);
    setDf2(p.df2);
    setLambda(p.lambda);
    setAlpha(p.alpha);
  };

  // Critical value + observed F (shown as a proxy "observed" at λ-implied location).
  const fCrit = useMemo(() => fQuantile(1 - alpha, df1, df2), [alpha, df1, df2]);
  const observedF = useMemo(() => fCrit + lambda / df1 / 3, [fCrit, lambda, df1]);
  const pValue = useMemo(() => Math.max(0, Math.min(1, 1 - fCDF(observedF, df1, df2))), [observedF, df1, df2]);
  const power = useMemo(() => noncentralFPower(df1, df2, lambda, alpha), [df1, df2, lambda, alpha]);

  // Density grid — extend to 2× fCrit or to a point where density is < 1e-4.
  const densities = useMemo(() => {
    const xMax = Math.max(fCrit * 2.5, 10, lambda / df1 + 5);
    const n = 320;
    const xs: number[] = [];
    const fNull: number[] = [];
    const fAlt: number[] = [];
    for (let i = 0; i <= n; i++) {
      const x = (xMax * i) / n;
      xs.push(x);
      fNull.push(fDensity(x, df1, df2));
      fAlt.push(noncentralFDensity(x, df1, df2, lambda));
    }
    return { xs, fNull, fAlt, xMax };
  }, [df1, df2, lambda, fCrit]);

  // Scales — main plot.
  const { xScale, yScale } = useMemo(() => {
    const maxY = Math.max(...densities.fNull, ...densities.fAlt) * 1.1 || 1;
    return {
      xScale: d3
        .scaleLinear()
        .domain([0, densities.xMax])
        .range([MARGIN.left, width - MARGIN.right]),
      yScale: d3
        .scaleLinear()
        .domain([0, maxY])
        .range([MARGIN.top + MAIN_H, MARGIN.top]),
    };
  }, [densities, width]);

  const pathGen = useMemo(
    () => d3.line<[number, number]>().x((d) => xScale(d[0])).y((d) => yScale(d[1])),
    [xScale, yScale],
  );
  const nullPath = pathGen(densities.xs.map((x, i) => [x, densities.fNull[i]]));
  const altPath = pathGen(densities.xs.map((x, i) => [x, densities.fAlt[i]]));

  // Rejection region polygon — area under null from fCrit to xMax.
  const rejectPath = useMemo(() => {
    const startIdx = densities.xs.findIndex((x) => x >= fCrit);
    if (startIdx < 0) return '';
    const pts: Array<[number, number]> = [];
    pts.push([fCrit, 0]);
    for (let i = startIdx; i < densities.xs.length; i++) {
      pts.push([densities.xs[i], densities.fNull[i]]);
    }
    pts.push([densities.xs[densities.xs.length - 1], 0]);
    return pathGen(pts) + ' Z';
  }, [densities, fCrit, pathGen]);

  // Power curve (mini-plot): power vs λ at the current (df1, df2, α).
  const powerCurve = useMemo(() => {
    const n = 80;
    const pts: Array<[number, number]> = [];
    for (let i = 0; i <= n; i++) {
      const lam = (30 * i) / n;
      pts.push([lam, noncentralFPower(df1, df2, lam, alpha)]);
    }
    return pts;
  }, [df1, df2, alpha]);

  const miniXScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain([0, 30])
        .range([MARGIN.left, width - MARGIN.right]),
    [width],
  );
  const miniYScale = useMemo(
    () => d3.scaleLinear().domain([0, 1]).range([MARGIN.top + MINI_H, MARGIN.top + 10]),
    [],
  );
  const miniPath = d3
    .line<[number, number]>()
    .x((d) => miniXScale(d[0]))
    .y((d) => miniYScale(d[1]))(powerCurve);

  return (
    <div
      ref={containerRef}
      className="my-8 rounded-lg border"
      style={{
        borderColor: 'var(--color-border, #e5e7eb)',
        backgroundColor: 'var(--color-surface, #ffffff)',
      }}
    >
      {/* Preset bar */}
      <div
        className="flex flex-wrap gap-2 border-b px-3 py-2 text-sm"
        style={{ borderColor: 'var(--color-border, #e5e7eb)' }}
      >
        {Object.entries(F_TEST_EXPLORER_PRESETS).map(([key, preset]) => (
          <button
            key={key}
            type="button"
            onClick={() => applyPreset(key)}
            className={`rounded px-3 py-1 text-xs transition-colors ${
              activePreset === key
                ? 'bg-slate-900 text-white'
                : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="grid gap-3 p-3 md:grid-cols-[1fr_240px]">
        {/* Main + mini plots */}
        <div>
          {/* Main F density plot */}
          <svg width="100%" viewBox={`0 0 ${width} ${MARGIN.top + MAIN_H + 30}`}>
            {/* Axes */}
            <g transform={`translate(0, ${MARGIN.top + MAIN_H})`}>
              <line x1={MARGIN.left} x2={width - MARGIN.right} stroke={AXIS_COLOR} />
              {xScale.ticks(6).map((t) => (
                <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                  <line y1={0} y2={4} stroke={AXIS_COLOR} />
                  <text y={18} textAnchor="middle" fontSize={11} fill={AXIS_COLOR}>
                    {t.toFixed(1)}
                  </text>
                </g>
              ))}
            </g>
            <text
              x={(MARGIN.left + width - MARGIN.right) / 2}
              y={MARGIN.top + MAIN_H + 34}
              textAnchor="middle"
              fontSize={11}
              fill={AXIS_COLOR}
            >
              F statistic
            </text>

            {/* Rejection region */}
            {rejectPath && (
              <path d={rejectPath} fill={REJECT_COLOR} opacity={0.3} />
            )}

            {/* Null density */}
            {nullPath && (
              <path d={nullPath} fill="none" stroke={NULL_COLOR} strokeWidth={3} />
            )}
            {/* Alternative density */}
            {altPath && (
              <path d={altPath} fill="none" stroke={ALT_COLOR} strokeWidth={3} />
            )}

            {/* Critical-value line */}
            <line
              x1={xScale(fCrit)}
              x2={xScale(fCrit)}
              y1={MARGIN.top}
              y2={MARGIN.top + MAIN_H}
              stroke={REJECT_COLOR}
              strokeDasharray="4 4"
            />
            <text
              x={xScale(fCrit)}
              y={MARGIN.top + 12}
              textAnchor="middle"
              fontSize={10}
              fill={REJECT_COLOR}
            >
              F_crit = {fCrit.toFixed(2)}
            </text>

            {/* Observed-F line */}
            <line
              x1={xScale(observedF)}
              x2={xScale(observedF)}
              y1={MARGIN.top}
              y2={MARGIN.top + MAIN_H}
              stroke={ALT_COLOR}
              strokeDasharray="2 3"
              opacity={0.7}
            />
          </svg>

          {/* Mini power curve */}
          <svg
            width="100%"
            viewBox={`0 0 ${width} ${MARGIN.top + MINI_H + 34}`}
            style={{ marginTop: 8 }}
          >
            <text
              x={MARGIN.left}
              y={MARGIN.top - 4}
              fontSize={11}
              fill={AXIS_COLOR}
            >
              Power vs λ (current df₁={df1}, df₂={df2}, α={alpha.toFixed(3)})
            </text>
            <g transform={`translate(0, ${MARGIN.top + MINI_H})`}>
              <line x1={MARGIN.left} x2={width - MARGIN.right} stroke={AXIS_COLOR} />
              {miniXScale.ticks(6).map((t) => (
                <g key={t} transform={`translate(${miniXScale(t)}, 0)`}>
                  <line y1={0} y2={3} stroke={AXIS_COLOR} />
                  <text y={16} textAnchor="middle" fontSize={10} fill={AXIS_COLOR}>
                    {t}
                  </text>
                </g>
              ))}
            </g>
            <g>
              {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                <g key={t} transform={`translate(${MARGIN.left}, ${miniYScale(t)})`}>
                  <line x1={-3} x2={0} stroke={AXIS_COLOR} />
                  <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill={AXIS_COLOR}>
                    {t}
                  </text>
                </g>
              ))}
            </g>
            {miniPath && (
              <path d={miniPath} fill="none" stroke={ALT_COLOR} strokeWidth={2} />
            )}
            {/* Current λ marker */}
            <circle
              cx={miniXScale(clamp(lambda, 0, 30))}
              cy={miniYScale(power)}
              r={4}
              fill={ALT_COLOR}
            />
          </svg>
        </div>

        {/* Control panel */}
        <div
          className="flex flex-col gap-3 rounded-md p-3 text-sm"
          style={{
            backgroundColor: 'var(--color-muted-bg, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <Slider
            label={`k (numerator df) = ${df1}`}
            min={1}
            max={10}
            step={1}
            value={df1}
            onChange={setDf1}
          />
          <Slider
            label={`n − p − 1 (denominator df) = ${df2}`}
            min={10}
            max={200}
            step={5}
            value={df2}
            onChange={setDf2}
          />
          <Slider
            label={`λ (non-centrality) = ${lambda.toFixed(1)}`}
            min={0}
            max={30}
            step={0.5}
            value={lambda}
            onChange={setLambda}
          />
          <LogSlider
            label={`α = ${alpha.toFixed(3)}`}
            min={0.001}
            max={0.10}
            value={alpha}
            onChange={setAlpha}
          />
          <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 border-t pt-2 text-xs">
            <span className="text-slate-600">Observed F</span>
            <span>{observedF.toFixed(3)}</span>
            <span className="text-slate-600">p-value</span>
            <span>{pValue.toExponential(2)}</span>
            <span className="text-slate-600">Power</span>
            <span>{power.toFixed(3)}</span>
          </div>
          <p className="text-[11px] leading-snug text-slate-600">
            Blue: null F<sub>{df1},{df2}</sub>. Amber: non-central F(λ). Grey shading: rejection region of the α-level test.
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Slider primitives ────────────────────────────────────────────────────────

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-slate-700">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </label>
  );
}

function LogSlider({
  label,
  min,
  max,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (v: number) => void;
}) {
  // Map log-space value [log(min), log(max)] to linear slider.
  const sliderPos = (Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min));
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-slate-700">{label}</span>
      <input
        type="range"
        min={0}
        max={1000}
        step={1}
        value={Math.round(sliderPos * 1000)}
        onChange={(e) => {
          const pos = parseFloat(e.target.value) / 1000;
          const v = Math.exp(Math.log(min) + pos * (Math.log(max) - Math.log(min)));
          onChange(v);
        }}
        className="w-full"
      />
    </label>
  );
}
