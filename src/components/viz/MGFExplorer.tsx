import { useState, useMemo, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { mgfMoment } from './shared/moments';
import { mgfPresets } from '../../data/expectation-moments-data';
import type { MGFPreset, MGFParamDef } from '../../data/expectation-moments-data';
import { distributionColors } from './shared/colorScales';

// ── Colors ────────────────────────────────────────────────────────────────────

const COLORS = {
  curve: distributionColors.pdf,     // blue — primary MGF curve
  tangent: distributionColors.sample, // red — tangent line
  dot: distributionColors.cdf,        // green — M(0) = 1 dot
  curveX: '#7C3AED',                  // violet — second distribution in sum mode
  product: distributionColors.pdf,    // blue — product curve in sum mode
} as const;

// ── Layout ────────────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 20, bottom: 30, left: 50 };
const NUM_POINTS = 200;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build a Record<string, number> from a preset's paramDefs and a values array. */
function buildParams(paramDefs: MGFParamDef[], values: number[]): Record<string, number> {
  const result: Record<string, number> = {};
  for (let i = 0; i < paramDefs.length; i++) {
    result[paramDefs[i].name] = values[i];
  }
  return result;
}

/** Get the effective upper domain bound for a preset, given its current params. */
function getUpperBound(preset: MGFPreset, params: Record<string, number>): number {
  if (preset.domain[1] !== null) return preset.domain[1];
  // For Exponential: domain upper bound is lambda - 0.1
  if ('lambda' in params) return params.lambda - 0.1;
  return 3;
}

/** Evaluate MGF at evenly spaced points, filtering out NaN/Infinity. */
function evaluateMGF(
  mgf: (t: number) => number,
  tMin: number,
  tMax: number,
  n: number,
): { t: number; y: number }[] {
  const step = (tMax - tMin) / n;
  const points: { t: number; y: number }[] = [];
  for (let i = 0; i <= n; i++) {
    const t = tMin + i * step;
    const y = mgf(t);
    if (isFinite(y) && !isNaN(y)) {
      points.push({ t, y });
    }
  }
  return points;
}

/** Format a number for display: use fixed notation unless very large or very small. */
function fmt(v: number, digits: number = 4): string {
  if (!isFinite(v) || isNaN(v)) return '---';
  if (Math.abs(v) > 1e6 || (Math.abs(v) < 1e-4 && v !== 0)) {
    return v.toExponential(2);
  }
  return v.toFixed(digits);
}

// ── Moments Panel ─────────────────────────────────────────────────────────────

function MomentsPanel({
  mgf,
  exactMoments,
  compact,
}: {
  mgf: (t: number) => number;
  exactMoments: { mean: number; variance: number };
  compact: boolean;
}) {
  const numericalMean = mgfMoment(mgf, 1);
  const numericalSecond = mgfMoment(mgf, 2);
  const numericalVariance = numericalSecond - numericalMean * numericalMean;

  const rows = [
    {
      label: "M'(0) = E[X]",
      numerical: numericalMean,
      exact: exactMoments.mean,
    },
    {
      label: "M''(0) = E[X\u00B2]",
      numerical: numericalSecond,
      exact: exactMoments.mean * exactMoments.mean + exactMoments.variance,
    },
    {
      label: "Var(X) = M''(0) \u2212 (M'(0))\u00B2",
      numerical: numericalVariance,
      exact: exactMoments.variance,
    },
  ];

  return (
    <div className={compact ? 'mt-3' : 'ml-4 flex-shrink-0'} style={{ minWidth: compact ? undefined : '260px' }}>
      <div className="text-sm font-medium mb-2">Moments from MGF</div>
      <table className="text-xs w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th className="text-left pr-3 pb-1 font-medium" style={{ borderBottom: '1px solid var(--color-border)' }}>
              Derivative
            </th>
            <th className="text-right pr-3 pb-1 font-medium" style={{ borderBottom: '1px solid var(--color-border)' }}>
              Numerical
            </th>
            <th className="text-right pb-1 font-medium" style={{ borderBottom: '1px solid var(--color-border)' }}>
              Exact
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td className="py-1 pr-3 font-mono" style={{ whiteSpace: 'nowrap' }}>
                {row.label}
              </td>
              <td className="py-1 pr-3 text-right font-mono">
                {fmt(row.numerical)}
              </td>
              <td className="py-1 text-right font-mono" style={{ color: COLORS.dot }}>
                {fmt(row.exact)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 text-xs" style={{ opacity: 0.5 }}>
        <span style={{ color: COLORS.dot }}>Green</span> = exact from distribution parameters.
        Numerical = central finite differences at t=0.
      </div>
    </div>
  );
}

// ── Param Sliders ─────────────────────────────────────────────────────────────

function ParamSliders({
  paramDefs,
  values,
  onChange,
  label,
}: {
  paramDefs: MGFParamDef[];
  values: number[];
  onChange: (idx: number, val: number) => void;
  label?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      {label && <span className="text-xs font-medium" style={{ opacity: 0.6 }}>{label}</span>}
      {paramDefs.map((pd, i) => (
        <div key={pd.name} className="flex items-center gap-1">
          <label className="text-sm font-medium">{pd.name}:</label>
          <input
            type="range"
            min={pd.min}
            max={pd.max}
            step={pd.step}
            value={values[i]}
            onChange={(e) => onChange(i, Number(e.target.value))}
            className="w-20"
            aria-label={`${pd.name} parameter`}
          />
          <span className="text-sm font-mono w-10">{values[i].toFixed(pd.step < 1 ? 2 : 0)}</span>
        </div>
      ))}
    </div>
  );
}

// ── MGF Curve Panel (SVG) ─────────────────────────────────────────────────────

function MGFCurvePanel({
  mgfFunc,
  tMin,
  tMax,
  panelW,
  panelH,
  slope,
  sumMode,
  mgfFuncY,
  mgfProduct,
}: {
  mgfFunc: (t: number) => number;
  tMin: number;
  tMax: number;
  panelW: number;
  panelH: number;
  slope: number;
  sumMode: boolean;
  mgfFuncY?: (t: number) => number;
  mgfProduct?: (t: number) => number;
}) {
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = panelH - MARGIN.top - MARGIN.bottom;

  // Evaluate all curves
  const dataX = useMemo(() => evaluateMGF(mgfFunc, tMin, tMax, NUM_POINTS), [mgfFunc, tMin, tMax]);
  const dataY = useMemo(
    () => (sumMode && mgfFuncY ? evaluateMGF(mgfFuncY, tMin, tMax, NUM_POINTS) : []),
    [sumMode, mgfFuncY, tMin, tMax],
  );
  const dataProd = useMemo(
    () => (sumMode && mgfProduct ? evaluateMGF(mgfProduct, tMin, tMax, NUM_POINTS) : []),
    [sumMode, mgfProduct, tMin, tMax],
  );

  // Compute y-axis bounds from all visible curves
  const { yMin, yMax: yAxisMax } = useMemo(() => {
    let allY = dataX.map((d) => d.y);
    if (sumMode) {
      allY = allY.concat(dataY.map((d) => d.y), dataProd.map((d) => d.y));
    }
    if (allY.length === 0) return { yMin: 0, yMax: 2 };
    const lo = Math.min(...allY);
    const hi = Math.max(...allY);
    const pad = (hi - lo) * 0.1 || 0.5;
    return { yMin: Math.min(lo - pad, 0), yMax: hi + pad };
  }, [dataX, dataY, dataProd, sumMode]);

  // Scales
  const xScale = useCallback(
    (val: number) => MARGIN.left + ((val - tMin) / (tMax - tMin)) * plotW,
    [tMin, tMax, plotW],
  );
  const yScale = useCallback(
    (val: number) => MARGIN.top + plotH - ((val - yMin) / (yAxisMax - yMin)) * plotH,
    [yMin, yAxisMax, plotH],
  );

  // Build SVG path from data array
  const buildPath = useCallback(
    (data: { t: number; y: number }[]) =>
      data
        .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.t).toFixed(1)},${yScale(d.y).toFixed(1)}`)
        .join(' '),
    [xScale, yScale],
  );

  const pathX = useMemo(() => buildPath(dataX), [buildPath, dataX]);
  const pathY = useMemo(() => (dataY.length > 0 ? buildPath(dataY) : ''), [buildPath, dataY]);
  const pathProd = useMemo(() => (dataProd.length > 0 ? buildPath(dataProd) : ''), [buildPath, dataProd]);

  // M(0) = 1 point
  const m0x = xScale(0);
  const m0y = yScale(1);

  // Tangent line endpoints: passes through (0, 1) with given slope
  const tangentExtent = (tMax - tMin) * 0.25;
  const tL = -tangentExtent;
  const tR = tangentExtent;
  const tangentYL = 1 + slope * tL;
  const tangentYR = 1 + slope * tR;

  // Y-axis tick values
  const yTicks = useMemo(() => {
    const range = yAxisMax - yMin;
    const rawStep = range / 5;
    const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const normalized = rawStep / magnitude;
    let step: number;
    if (normalized < 1.5) step = magnitude;
    else if (normalized < 3.5) step = 2 * magnitude;
    else if (normalized < 7.5) step = 5 * magnitude;
    else step = 10 * magnitude;

    const ticks: number[] = [];
    const start = Math.ceil(yMin / step) * step;
    for (let v = start; v <= yAxisMax; v += step) {
      ticks.push(parseFloat(v.toFixed(10)));
    }
    return ticks;
  }, [yMin, yAxisMax]);

  // X-axis ticks
  const xTicks = useMemo(() => {
    const count = Math.min(Math.floor(plotW / 60), 8);
    const step = (tMax - tMin) / count;
    return Array.from({ length: count + 1 }, (_, i) =>
      parseFloat((tMin + i * step).toFixed(4)),
    );
  }, [tMin, tMax, plotW]);

  return (
    <svg
      width={panelW}
      height={panelH}
      className="block"
      role="img"
      aria-label="Moment-generating function plot"
    >
      {/* Grid lines */}
      {yTicks.map((v) => (
        <g key={`yt-${v}`}>
          <line
            x1={MARGIN.left}
            y1={yScale(v)}
            x2={panelW - MARGIN.right}
            y2={yScale(v)}
            stroke="currentColor"
            strokeOpacity={0.08}
          />
          <text
            x={MARGIN.left - 6}
            y={yScale(v) + 3}
            textAnchor="end"
            className="fill-current"
            style={{ fontSize: '9px', opacity: 0.5 }}
          >
            {Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1)}
          </text>
        </g>
      ))}

      {/* X-axis ticks */}
      {xTicks.map((v) => (
        <text
          key={`xt-${v}`}
          x={xScale(v)}
          y={panelH - 6}
          textAnchor="middle"
          className="fill-current"
          style={{ fontSize: '9px', opacity: 0.5 }}
        >
          {v.toFixed(1)}
        </text>
      ))}

      {/* Axis labels */}
      <text
        x={MARGIN.left + plotW / 2}
        y={panelH - 1}
        textAnchor="middle"
        className="fill-current"
        style={{ fontSize: '10px' }}
      >
        t
      </text>
      <text
        x={12}
        y={MARGIN.top + plotH / 2}
        textAnchor="middle"
        className="fill-current"
        style={{ fontSize: '10px' }}
        transform={`rotate(-90, 12, ${MARGIN.top + plotH / 2})`}
      >
        M(t)
      </text>

      {/* Vertical dashed line at t=0 */}
      <line
        x1={m0x}
        y1={MARGIN.top}
        x2={m0x}
        y2={MARGIN.top + plotH}
        stroke="currentColor"
        strokeOpacity={0.2}
        strokeDasharray="4,3"
      />

      {/* Tangent line at t=0 */}
      {!sumMode && (
        <line
          x1={xScale(tL)}
          y1={yScale(tangentYL)}
          x2={xScale(tR)}
          y2={yScale(tangentYR)}
          stroke={COLORS.tangent}
          strokeWidth={1.5}
          strokeDasharray="6,3"
          opacity={0.7}
        />
      )}

      {/* Sum mode: individual dashed curves */}
      {sumMode && pathY && (
        <>
          <path
            d={pathX}
            fill="none"
            stroke={COLORS.curve}
            strokeWidth={1.5}
            strokeDasharray="6,3"
            opacity={0.6}
          />
          <path
            d={pathY}
            fill="none"
            stroke={COLORS.curveX}
            strokeWidth={1.5}
            strokeDasharray="6,3"
            opacity={0.6}
          />
          {/* Product curve (solid) */}
          {pathProd && (
            <path
              d={pathProd}
              fill="none"
              stroke={COLORS.product}
              strokeWidth={2.5}
            />
          )}
        </>
      )}

      {/* Primary MGF curve (non-sum mode) */}
      {!sumMode && (
        <path
          d={pathX}
          fill="none"
          stroke={COLORS.curve}
          strokeWidth={2}
        />
      )}

      {/* M(0) = 1 dot */}
      <circle cx={m0x} cy={m0y} r={5} fill={COLORS.dot} />
      <text
        x={m0x + 8}
        y={m0y - 8}
        className="fill-current"
        style={{ fontSize: '10px', fontWeight: 500 }}
      >
        M(0) = 1
      </text>

      {/* Tangent slope annotation (non-sum mode) */}
      {!sumMode && (
        <text
          x={xScale(tR) + 4}
          y={yScale(tangentYR) - 4}
          style={{ fontSize: '9px', fill: COLORS.tangent }}
        >
          slope = {fmt(slope, 3)}
        </text>
      )}

      {/* Sum mode legend */}
      {sumMode && (
        <g>
          <line x1={panelW - MARGIN.right - 100} y1={MARGIN.top + 8} x2={panelW - MARGIN.right - 80} y2={MARGIN.top + 8} stroke={COLORS.curve} strokeWidth={1.5} strokeDasharray="4,2" />
          <text x={panelW - MARGIN.right - 76} y={MARGIN.top + 11} style={{ fontSize: '9px' }} className="fill-current">
            M_X(t)
          </text>
          <line x1={panelW - MARGIN.right - 100} y1={MARGIN.top + 22} x2={panelW - MARGIN.right - 80} y2={MARGIN.top + 22} stroke={COLORS.curveX} strokeWidth={1.5} strokeDasharray="4,2" />
          <text x={panelW - MARGIN.right - 76} y={MARGIN.top + 25} style={{ fontSize: '9px' }} className="fill-current">
            M_Y(t)
          </text>
          <line x1={panelW - MARGIN.right - 100} y1={MARGIN.top + 36} x2={panelW - MARGIN.right - 80} y2={MARGIN.top + 36} stroke={COLORS.product} strokeWidth={2.5} />
          <text x={panelW - MARGIN.right - 76} y={MARGIN.top + 39} style={{ fontSize: '9px' }} className="fill-current">
            M_X M_Y
          </text>
        </g>
      )}
    </svg>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function MGFExplorer() {
  const { ref: containerRef, width } = useResizeObserver<HTMLDivElement>();

  // ── Distribution X state ───────────────────────────────────────────────────
  const [presetIdxX, setPresetIdxX] = useState(0);
  const presetX = mgfPresets[presetIdxX];
  const [paramValuesX, setParamValuesX] = useState<number[]>(() =>
    presetX.paramDefs.map((pd) => pd.default),
  );

  // ── Sum mode ───────────────────────────────────────────────────────────────
  const [sumMode, setSumMode] = useState(false);
  const [presetIdxY, setPresetIdxY] = useState(2); // default to Normal
  const presetY = mgfPresets[presetIdxY];
  const [paramValuesY, setParamValuesY] = useState<number[]>(() =>
    presetY.paramDefs.map((pd) => pd.default),
  );

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handlePresetChangeX = useCallback(
    (idx: number) => {
      setPresetIdxX(idx);
      setParamValuesX(mgfPresets[idx].paramDefs.map((pd) => pd.default));
    },
    [],
  );

  const handlePresetChangeY = useCallback(
    (idx: number) => {
      setPresetIdxY(idx);
      setParamValuesY(mgfPresets[idx].paramDefs.map((pd) => pd.default));
    },
    [],
  );

  const handleParamChangeX = useCallback(
    (i: number, val: number) => {
      setParamValuesX((prev) => {
        const next = [...prev];
        next[i] = val;
        return next;
      });
    },
    [],
  );

  const handleParamChangeY = useCallback(
    (i: number, val: number) => {
      setParamValuesY((prev) => {
        const next = [...prev];
        next[i] = val;
        return next;
      });
    },
    [],
  );

  // ── Derived values ─────────────────────────────────────────────────────────

  const paramsX = useMemo(() => buildParams(presetX.paramDefs, paramValuesX), [presetX.paramDefs, paramValuesX]);
  const paramsY = useMemo(() => buildParams(presetY.paramDefs, paramValuesY), [presetY.paramDefs, paramValuesY]);

  const mgfFuncX = useCallback((t: number) => presetX.mgf(t, paramsX), [presetX, paramsX]);
  const mgfFuncY = useCallback((t: number) => presetY.mgf(t, paramsY), [presetY, paramsY]);
  const mgfProduct = useCallback((t: number) => {
    const vx = presetX.mgf(t, paramsX);
    const vy = presetY.mgf(t, paramsY);
    return vx * vy;
  }, [presetX, paramsX, presetY, paramsY]);

  const exactMomentsX = useMemo(() => presetX.moments(paramsX), [presetX, paramsX]);
  const slopeX = useMemo(() => mgfMoment(mgfFuncX, 1), [mgfFuncX]);

  // Domain bounds
  const tMin = presetX.domain[0];
  const tMaxX = getUpperBound(presetX, paramsX);
  // In sum mode, use the narrower of the two domains
  const tMaxY = sumMode ? getUpperBound(presetY, paramsY) : tMaxX;
  const tMax = sumMode ? Math.min(tMaxX, tMaxY) : tMaxX;

  // ── Layout ─────────────────────────────────────────────────────────────────

  const compact = width < 700;
  const svgW = compact ? Math.max(width - 32, 280) : Math.max(Math.floor((width - 32) * 0.6), 300);
  const svgH = 280;

  return (
    <div
      ref={containerRef}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* ── Controls ────────────────────────────────────────────────────────── */}
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium">
            {sumMode ? 'X ~' : 'Distribution:'}
          </label>
          <select
            value={presetIdxX}
            onChange={(e) => handlePresetChangeX(Number(e.target.value))}
            className="rounded border px-2 py-1 text-sm"
            style={{ borderColor: 'var(--color-border)' }}
            aria-label="Select distribution X"
          >
            {mgfPresets.map((p, i) => (
              <option key={p.name} value={i}>{p.name}</option>
            ))}
          </select>
        </div>

        {sumMode && (
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Y ~</label>
            <select
              value={presetIdxY}
              onChange={(e) => handlePresetChangeY(Number(e.target.value))}
              className="rounded border px-2 py-1 text-sm"
              style={{ borderColor: 'var(--color-border)' }}
              aria-label="Select distribution Y"
            >
              {mgfPresets.map((p, i) => (
                <option key={p.name} value={i}>{p.name}</option>
              ))}
            </select>
          </div>
        )}

        <label className="flex items-center gap-1 text-sm">
          <input
            type="checkbox"
            checked={sumMode}
            onChange={(e) => setSumMode(e.target.checked)}
          />
          Sum mode (X + Y)
        </label>
      </div>

      {/* ── Parameter sliders ───────────────────────────────────────────────── */}
      <div className="mb-4 space-y-2">
        <ParamSliders
          paramDefs={presetX.paramDefs}
          values={paramValuesX}
          onChange={handleParamChangeX}
          label={sumMode ? 'X params' : undefined}
        />
        {sumMode && (
          <ParamSliders
            paramDefs={presetY.paramDefs}
            values={paramValuesY}
            onChange={handleParamChangeY}
            label="Y params"
          />
        )}
      </div>

      {/* ── Two-panel layout ────────────────────────────────────────────────── */}
      <div className={compact ? 'space-y-3' : 'flex items-start'}>
        {/* Left: MGF curve */}
        <div className="flex-shrink-0">
          <div className="mb-1 text-center text-xs font-medium">
            {sumMode
              ? `M_{X+Y}(t) = M_X(t) \\cdot M_Y(t)`
              : `M(t) = E[e^{tX}] for ${presetX.name}`}
          </div>
          <MGFCurvePanel
            mgfFunc={mgfFuncX}
            tMin={tMin}
            tMax={tMax}
            panelW={svgW}
            panelH={svgH}
            slope={slopeX}
            sumMode={sumMode}
            mgfFuncY={mgfFuncY}
            mgfProduct={mgfProduct}
          />
        </div>

        {/* Right: Moments table (only shown in non-sum mode) */}
        {!sumMode && (
          <MomentsPanel
            mgf={mgfFuncX}
            exactMoments={exactMomentsX}
            compact={compact}
          />
        )}
      </div>

      {/* ── Sum mode readout ────────────────────────────────────────────────── */}
      {sumMode && (
        <div className="mt-3 rounded border p-3 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-subtle, rgba(0,0,0,0.02))' }}>
          <div className="font-medium mb-1">
            Independence property: M_{'{X+Y}'}(t) = M_X(t) &middot; M_Y(t)
          </div>
          <div className="text-xs space-y-1 font-mono" style={{ opacity: 0.7 }}>
            <div>
              E[X] = {fmt(exactMomentsX.mean)},&ensp;
              Var(X) = {fmt(exactMomentsX.variance)}
            </div>
            <div>
              E[Y] = {fmt(presetY.moments(paramsY).mean)},&ensp;
              Var(Y) = {fmt(presetY.moments(paramsY).variance)}
            </div>
            <div style={{ opacity: 1 }}>
              E[X+Y] = {fmt(exactMomentsX.mean + presetY.moments(paramsY).mean)},&ensp;
              Var(X+Y) = {fmt(exactMomentsX.variance + presetY.moments(paramsY).variance)}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer note ─────────────────────────────────────────────────────── */}
      {!sumMode && (
        <div className="mt-3 text-xs" style={{ opacity: 0.5 }}>
          The <span style={{ color: COLORS.tangent }}>red dashed line</span> is the tangent at t=0 with slope M&prime;(0) = E[X].
          The <span style={{ color: COLORS.dot }}>green dot</span> marks M(0) = 1, which holds for every distribution (since E[e^{'{0 \\cdot X}'}] = 1).
        </div>
      )}
    </div>
  );
}
