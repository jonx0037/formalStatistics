import { useMemo, useState } from 'react';
import * as d3 from 'd3';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  olsFit,
  coefCIWald,
  coefCIProfile,
  coefCIBonferroni,
} from './shared/regression';
import {
  COEFFICIENT_CI_BANDS_PRESETS,
  generateLinearModelData,
  type CoefficientCIPreset,
} from '../../data/regression-data';

// ── Layout ───────────────────────────────────────────────────────────────────
const MARGIN = { top: 24, right: 24, bottom: 44, left: 84 };
const ROW_H = 56; // per-coefficient row height (accommodates 3 stacked CIs)

// ── Visual palette ───────────────────────────────────────────────────────────
const WALD_COLOR = '#2563EB'; // blue
const LRT_COLOR = '#059669'; // green
const BONF_COLOR = '#D97706'; // amber
const AXIS_COLOR = '#9CA3AF';
const POINT_COLOR = '#111827';

const P_NON_INTERCEPT = 4; // default: 5 coefficients total (intercept + 4)

export default function CoefficientCIBands() {
  const { ref: containerRef, width: measuredWidth } = useResizeObserver<HTMLDivElement>();
  const width = Math.max(measuredWidth || 800, 320);

  const [activePreset, setActivePreset] = useState<string>('wellConditioned');
  const [n, setN] = useState<number>(COEFFICIENT_CI_BANDS_PRESETS.wellConditioned.n);
  const [sigma, setSigma] = useState<number>(COEFFICIENT_CI_BANDS_PRESETS.wellConditioned.sigma);
  const [alpha, setAlpha] = useState<number>(COEFFICIENT_CI_BANDS_PRESETS.wellConditioned.alpha);
  const [designKind, setDesignKind] = useState<'wellConditioned' | 'collinear'>(
    COEFFICIENT_CI_BANDS_PRESETS.wellConditioned.designKind,
  );
  const [seed, setSeed] = useState<number>(20250101);

  const applyPreset = (key: string) => {
    const p: CoefficientCIPreset | undefined = COEFFICIENT_CI_BANDS_PRESETS[key];
    if (!p) return;
    setActivePreset(key);
    setN(p.n);
    setSigma(p.sigma);
    setAlpha(p.alpha);
    setDesignKind(p.designKind);
  };

  // True β (for reference marker): intercept + 4 slopes with varied magnitudes.
  const betaTrue = useMemo(() => [1, 0.8, -0.5, 0.3, -0.2], []);

  // Fit a fresh dataset given current controls and seed.
  const fit = useMemo(() => {
    const { X, y } = generateLinearModelData({
      n,
      p: P_NON_INTERCEPT,
      beta: betaTrue,
      sigma,
      designKind,
      seed,
    });
    try {
      return { fit: olsFit(X, y), X, y };
    } catch {
      return null;
    }
  }, [n, sigma, designKind, seed, betaTrue]);

  const cis = useMemo(() => {
    if (!fit) return null;
    return {
      wald: coefCIWald(fit.fit, alpha),
      profile: coefCIProfile(fit.fit, fit.X, fit.y, alpha),
      bonf: coefCIBonferroni(fit.fit, alpha),
    };
  }, [fit, alpha]);

  // x-domain: span all CI endpoints + β_true, with a little padding.
  const xDomain = useMemo((): [number, number] => {
    if (!cis) return [-2, 2];
    const all: number[] = [];
    for (const arr of [cis.wald, cis.profile, cis.bonf]) {
      for (const c of arr) {
        all.push(c.lower, c.upper);
      }
    }
    all.push(...betaTrue);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const pad = (hi - lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }, [cis, betaTrue]);

  const heightAboveAxis = (fit?.fit.beta.length ?? 5) * ROW_H;
  const totalH = MARGIN.top + heightAboveAxis + MARGIN.bottom;

  const xScale = useMemo(
    () =>
      d3
        .scaleLinear()
        .domain(xDomain)
        .range([MARGIN.left, width - MARGIN.right]),
    [xDomain, width],
  );

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
        {Object.entries(COEFFICIENT_CI_BANDS_PRESETS).map(([key, preset]) => (
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
        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="ml-auto rounded bg-slate-100 px-3 py-1 text-xs hover:bg-slate-200"
        >
          Regenerate data
        </button>
      </div>

      <div className="grid gap-3 p-3 md:grid-cols-[1fr_240px]">
        <svg width="100%" viewBox={`0 0 ${width} ${totalH}`}>
          {/* Axis */}
          <g transform={`translate(0, ${MARGIN.top + heightAboveAxis})`}>
            <line x1={MARGIN.left} x2={width - MARGIN.right} stroke={AXIS_COLOR} />
            {xScale.ticks(8).map((t) => (
              <g key={t} transform={`translate(${xScale(t)}, 0)`}>
                <line y1={0} y2={4} stroke={AXIS_COLOR} />
                <text y={18} textAnchor="middle" fontSize={11} fill={AXIS_COLOR}>
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            <text
              x={(MARGIN.left + width - MARGIN.right) / 2}
              y={34}
              textAnchor="middle"
              fontSize={11}
              fill={AXIS_COLOR}
            >
              coefficient value
            </text>
          </g>

          {/* Zero line */}
          <line
            x1={xScale(0)}
            x2={xScale(0)}
            y1={MARGIN.top}
            y2={MARGIN.top + heightAboveAxis}
            stroke={AXIS_COLOR}
            strokeDasharray="2 4"
          />

          {/* Per-coefficient rows */}
          {fit &&
            cis &&
            fit.fit.beta.map((b, j) => {
              const rowY = MARGIN.top + j * ROW_H;
              const wald = cis.wald[j];
              const profile = cis.profile[j];
              const bonf = cis.bonf[j];
              const betaT = betaTrue[j] ?? 0;
              return (
                <g key={j}>
                  {/* Row label */}
                  <text
                    x={MARGIN.left - 10}
                    y={rowY + ROW_H / 2 + 4}
                    textAnchor="end"
                    fontSize={12}
                    fill="#111827"
                  >
                    β̂
                    <tspan baselineShift="sub" fontSize={9}>{j}</tspan>
                  </text>

                  {/* Bonferroni CI (widest — drawn first, bottom) */}
                  <line
                    x1={xScale(bonf.lower)}
                    x2={xScale(bonf.upper)}
                    y1={rowY + ROW_H / 2 + 14}
                    y2={rowY + ROW_H / 2 + 14}
                    stroke={BONF_COLOR}
                    strokeWidth={5}
                  />
                  {/* Wald-t */}
                  <line
                    x1={xScale(wald.lower)}
                    x2={xScale(wald.upper)}
                    y1={rowY + ROW_H / 2}
                    y2={rowY + ROW_H / 2}
                    stroke={WALD_COLOR}
                    strokeWidth={5}
                  />
                  {/* LRT / profile (above Wald for visibility even when identical) */}
                  <line
                    x1={xScale(profile.lower)}
                    x2={xScale(profile.upper)}
                    y1={rowY + ROW_H / 2 - 14}
                    y2={rowY + ROW_H / 2 - 14}
                    stroke={LRT_COLOR}
                    strokeWidth={3}
                  />

                  {/* Point estimate */}
                  <circle
                    cx={xScale(b)}
                    cy={rowY + ROW_H / 2}
                    r={5}
                    fill={POINT_COLOR}
                  />
                  {/* True β marker (hollow red-ish triangle) */}
                  <circle
                    cx={xScale(betaT)}
                    cy={rowY + ROW_H / 2}
                    r={4}
                    fill="none"
                    stroke="#DC2626"
                    strokeWidth={2}
                  />
                </g>
              );
            })}
        </svg>

        {/* Controls */}
        <div
          className="flex flex-col gap-3 rounded-md p-3 text-sm"
          style={{
            backgroundColor: 'var(--color-muted-bg, #f8fafc)',
            fontVariantNumeric: 'tabular-nums',
          }}
        >
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-slate-700">n = {n}</span>
            <input
              type="range"
              min={20}
              max={500}
              step={10}
              value={n}
              onChange={(e) => setN(parseInt(e.target.value, 10))}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-slate-700">σ = {sigma.toFixed(2)}</span>
            <input
              type="range"
              min={0.1}
              max={5}
              step={0.1}
              value={sigma}
              onChange={(e) => setSigma(parseFloat(e.target.value))}
            />
          </label>
          <label className="flex flex-col gap-1 text-xs">
            <span className="text-slate-700">α = {alpha.toFixed(3)}</span>
            <input
              type="range"
              min={0.001}
              max={0.10}
              step={0.001}
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
            />
          </label>

          {/* Legend */}
          <div className="mt-2 border-t pt-2 text-xs">
            <LegendSwatch color={WALD_COLOR} label="Wald-t (individual 1 − α)" />
            <LegendSwatch color={LRT_COLOR} label="LRT / profile (same under Normal errors)" />
            <LegendSwatch color={BONF_COLOR} label={`Bonferroni (FWER ≤ α, per-CI at ${(alpha / 5).toFixed(3)})`} />
            <LegendSwatch label="β̂_j" pointColor={POINT_COLOR} />
            <LegendSwatch label="true β_j" pointColor="none" pointStroke="#DC2626" />
          </div>
          <p className="text-[11px] leading-snug text-slate-600">
            See §21.8 Rem 15 (Wald vs LRT coincide under Normal errors) and Rem 16 (Bonferroni ⊆ §21.8 simultaneous inference; Working–Hotelling gives continuum bands).
          </p>
        </div>
      </div>
    </div>
  );
}

function LegendSwatch({
  color,
  label,
  pointColor,
  pointStroke,
}: {
  color?: string;
  label: string;
  pointColor?: string;
  pointStroke?: string;
}) {
  return (
    <div className="mb-1 flex items-center gap-2">
      {color && (
        <span
          className="inline-block h-[3px] w-4 rounded"
          style={{ backgroundColor: color }}
        />
      )}
      {pointColor !== undefined && (
        <span
          className="inline-block h-3 w-3 rounded-full"
          style={{
            backgroundColor: pointColor === 'none' ? 'transparent' : pointColor,
            border: pointStroke ? `2px solid ${pointStroke}` : undefined,
          }}
        />
      )}
      <span className="text-slate-700">{label}</span>
    </div>
  );
}
