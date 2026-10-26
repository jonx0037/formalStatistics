/**
 * StructuralDecomposer — optional interactive for Topic 36 §36.7.
 *
 * Structural models as a modeling language (Def 4): a local level, an optional
 * slope (local linear trend), and an optional 12-month dummy seasonal are
 * stacked block-diagonally and fitted to the Mauna Loa CO2 record (Ex 13). Every
 * state is approximate-diffuse (Def 5: P₁ = κI, burn b = k), so the readout's
 * diffuse log-likelihood drops the first b terms. At the defaults (the
 * UnobservedComponents MLE of Ex 13) ℓ = −157.914578 with k = b = 13, the
 * smoothed growth rate climbs from 1.006 to 1.545 ppm/yr, and the seasonal
 * swing is 6.07 ppm; the smoother fills the 5 empty months (Thm 4). Toggling a
 * component changes b, and with it the number of dropped terms: Rem 21's point
 * that such diffuse likelihoods (and their AICs) are not comparable, flagged by
 * the snapshot badge. No in-browser fitting (Appendix B.8): sliders + "Reset to
 * MLE" only.
 */
import { useEffect, useId, useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { armaIC, forecast, kalmanFilter, rtsSmoother, seasonalBlock, stackModels, type Matrix } from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { structuralPresets } from '../../data/time-series-data';
import { co2StartMonth, co2Values } from '../../data/co2';

const MARGIN = { top: 8, right: 12, bottom: 30, left: 50 };
const MARGIN_BOTTOM_INNER = 8;
const MOBILE_BREAKPOINT = 640;
const H_MAX = 36;
const DEBOUNCE_MS = 50;
const DECADES = 3;
const SEASON = 12;

const [START_YEAR, START_MONTH] = co2StartMonth.split('-').map((v) => parseInt(v, 10));
/** Jan 1990 = (1990 − 1958)·12 + (1 − 3) = 382 for a March-1958 start. */
const IDX_1990 = (1990 - START_YEAR) * 12 + (1 - START_MONTH);

type VarKey = 'level' | 'slope' | 'seasonal' | 'irregular';
type WindowId = 'full' | 'recent';

interface Config {
  slopeOn: boolean;
  seasonalOn: boolean;
  logV: Record<VarKey, number>;
  windowId: WindowId;
}

interface Snapshot {
  label: string;
  loglik: number;
  aic: number;
  burn: number;
  windowId: WindowId;
}

const MLE = structuralPresets.find((p) => p.id === 'co2Mle') ?? structuralPresets[0];
const LEVEL_ONLY = structuralPresets.find((p) => p.id === 'levelOnly');
const DEFAULT_LOG: Record<VarKey, number> = {
  level: Math.log10(MLE.level),
  slope: Math.log10(MLE.slope ?? 3.450003e-6),
  seasonal: Math.log10(MLE.seasonal ?? 1.0336517e-5),
  irregular: Math.log10(MLE.irregular),
};
const MLE_CONFIG: Config = { slopeOn: true, seasonalOn: true, logV: DEFAULT_LOG, windowId: 'full' };

const VAR_LABELS: Record<VarKey, string> = {
  level: 'level variance q_μ',
  slope: 'slope variance q_ν',
  seasonal: 'seasonal variance q_δ',
  irregular: 'irregular variance r',
};

/** Month index i (0 = series start) → decimal year at the start of that month. */
const yearOf = (i: number) => START_YEAR + (START_MONTH - 1 + i) / 12;

function modelLabel(c: Pick<Config, 'slopeOn' | 'seasonalOn'>): string {
  return ['level', c.slopeOn ? 'slope' : null, c.seasonalOn ? 'seasonal' : null].filter(Boolean).join(' + ');
}

/** Path through (x, y) that breaks at every non-finite y. */
function gapPath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  let d = '';
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(ys[i])) {
      pen = false;
      continue;
    }
    d += `${pen ? 'L' : 'M'}${sx(xs[i]).toFixed(2)},${sy(ys[i]).toFixed(2)}`;
    pen = true;
  }
  return d;
}

/** Closed band between lo and hi (all finite). */
function bandPath(xs: number[], lo: number[], hi: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  if (xs.length === 0) return '';
  const up = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(hi[i]).toFixed(2)}`).join('');
  const down = xs
    .map((_, k) => xs.length - 1 - k)
    .map((i) => `L${sx(xs[i]).toFixed(2)},${sy(lo[i]).toFixed(2)}`)
    .join('');
  return `${up}${down}Z`;
}

function linearTicks(lo: number, hi: number, count = 4): { v: number; label: string }[] {
  const raw = (hi - lo) / count;
  if (!(raw > 0)) return [{ v: lo, label: String(Number(lo.toPrecision(3))) }];
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((s) => s * mag).find((s) => s >= raw) ?? raw;
  const out: { v: number; label: string }[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push({ v: t, label: String(Number(t.toPrecision(4))) });
  return out;
}

function paddedRange(vals: number[], pad = 0.08, minSpan = 1e-6): [number, number] {
  const f = vals.filter(Number.isFinite);
  if (f.length === 0) return [-1, 1];
  let lo = Math.min(...f);
  let hi = Math.max(...f);
  if (hi - lo < minSpan) {
    lo -= minSpan / 2;
    hi += minSpan / 2;
  }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

function symmetricRange(vals: number[], pad = 0.1, minAbs = 1e-6): [number, number] {
  const f = vals.filter(Number.isFinite);
  const a = Math.max(minAbs, ...f.map(Math.abs)) * (1 + pad);
  return [-a, a];
}

function fmtVar(v: number): string {
  return v.toExponential(2).replace('e', '×10^').replace('^+', '^');
}

function Strip({
  width,
  height,
  label,
  yLabel,
  xDomain,
  yDomain,
  xTicks,
  showXAxis,
  gaps,
  forecastFrom,
  forecastTo,
  hatchId,
  clipKey,
  children,
}: {
  width: number;
  height: number;
  label: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks: number[];
  showXAxis: boolean;
  gaps: number[];
  forecastFrom: number | null;
  forecastTo: number;
  hatchId: string;
  clipKey: string;
  children: (sx: (v: number) => number, sy: (v: number) => number, plotW: number, plotH: number) => React.ReactNode;
}) {
  const mb = showXAxis ? MARGIN.bottom : MARGIN_BOTTOM_INNER;
  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - mb);
  const xSpan = Math.max(1e-9, xDomain[1] - xDomain[0]);
  const ySpan = Math.max(1e-12, yDomain[1] - yDomain[0]);
  const sx = (v: number) => ((v - xDomain[0]) / xSpan) * plotW;
  const sy = (v: number) => plotH - ((Math.min(Math.max(v, yDomain[0]), yDomain[1]) - yDomain[0]) / ySpan) * plotH;
  const axisColor = 'var(--color-text-muted)';
  const monthW = Math.max(1, sx(xDomain[0] + 1 / 12) - sx(xDomain[0]));
  const yTicks = linearTicks(yDomain[0], yDomain[1], height < 110 ? 2 : 4);
  const clipId = `${hatchId}-clip-${clipKey}`;
  const patternId = `${hatchId}-${clipKey}`;
  return (
    <svg width={Math.max(0, width)} height={Math.max(0, height)} className="block" role="img" aria-label={label}>
      <defs>
        <clipPath id={clipId}>
          <rect x={0} y={0} width={plotW} height={plotH} />
        </clipPath>
        <pattern id={patternId} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1={0} y1={0} x2={0} y2={6} stroke={timeSeriesColors.forecast} strokeWidth={1} opacity={0.35} />
        </pattern>
      </defs>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {gaps.map((g) => (
          <rect key={g} x={sx(g)} y={0} width={monthW} height={plotH} fill={timeSeriesColors.warn} opacity={0.08} />
        ))}
        {forecastFrom !== null && forecastFrom < xDomain[1] && (
          <>
            <rect x={sx(forecastFrom)} y={0} width={Math.max(0, Math.min(plotW, sx(forecastTo)) - sx(forecastFrom))} height={plotH} fill={timeSeriesColors.forecast} opacity={0.06} />
            <rect x={sx(forecastFrom)} y={0} width={Math.max(0, Math.min(plotW, sx(forecastTo)) - sx(forecastFrom))} height={plotH} fill={`url(#${patternId})`} />
          </>
        )}
        {yTicks.map((t) => (
          <g key={t.label}>
            <line x1={0} y1={sy(t.v)} x2={plotW} y2={sy(t.v)} stroke="var(--color-viz-grid)" />
            <text x={-6} y={sy(t.v) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
              {t.label}
            </text>
          </g>
        ))}
        <g clipPath={`url(#${clipId})`}>{children(sx, sy, plotW, plotH)}</g>
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} y1={plotH} x2={sx(t)} y2={plotH + (showXAxis ? 4 : 2)} stroke={axisColor} />
            {showXAxis && (
              <text x={sx(t)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                {t}
              </text>
            )}
          </g>
        ))}
        {showXAxis && (
          <text x={plotW / 2} y={plotH + 28} textAnchor="middle" fontSize={11} fill="var(--color-text)">
            year
          </text>
        )}
        <text
          x={-40}
          y={plotH / 2}
          textAnchor="middle"
          fontSize={10}
          fill="var(--color-text)"
          transform={`rotate(-90 ${-40} ${plotH / 2})`}
        >
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

function LogSlider({
  label,
  value,
  center,
  onChange,
}: {
  label: string;
  value: number;
  center: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-center justify-between gap-2 text-xs">
        <span>{label}</span>
        <span className="font-mono">
          {fmtVar(10 ** value)} <span style={{ color: 'var(--color-text-muted)' }}>(log₁₀ {value.toFixed(2)})</span>
        </span>
      </span>
      <input
        type="range"
        min={center - DECADES}
        max={center + DECADES}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-valuetext={`${label} ${fmtVar(10 ** value)}`}
        className="w-full accent-[var(--color-ts-smoothed)]"
      />
    </label>
  );
}

export default function StructuralDecomposer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const hatchId = `sd-hatch-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const [config, setConfig] = useState<Config>(MLE_CONFIG);
  const [applied, setApplied] = useState<Config>(MLE_CONFIG);
  const [h, setH] = useState(12);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);

  // Debounce every model change (sliders, toggles, window) by 50 ms.
  useEffect(() => {
    const id = setTimeout(() => setApplied(config), DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [config]);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;

  const offset = applied.windowId === 'full' ? 0 : IDX_1990;
  const x = useMemo(() => (applied.windowId === 'full' ? co2Values : co2Values.slice(IDX_1990)), [applied.windowId]);
  const n = x.length;
  const times = useMemo(() => x.map((_, i) => yearOf(offset + i)), [x, offset]);
  const gaps = useMemo(() => x.flatMap((v, i) => (Number.isNaN(v) ? [yearOf(offset + i)] : [])), [x, offset]);
  const nObs = n - gaps.length;

  const fit = useMemo(() => {
    const { slopeOn, seasonalOn, logV } = applied;
    const qL = 10 ** logV.level;
    const blocks: { F: Matrix; H: Matrix; Q: Matrix }[] = [
      slopeOn
        ? {
            F: [
              [1, 1],
              [0, 1],
            ],
            H: [[1, 0]],
            Q: [
              [qL, 0],
              [0, 10 ** logV.slope],
            ],
          }
        : { F: [[1]], H: [[1]], Q: [[qL]] },
    ];
    if (seasonalOn) blocks.push(seasonalBlock(SEASON, 10 ** logV.seasonal));
    const model = stackModels(blocks, 10 ** logV.irregular);
    const kf = kalmanFilter(x, model);
    const sm = rtsSmoother(kf, model);
    const seasIdx = slopeOn ? 2 : 1;
    const level = sm.mS.map((s) => s[0]);
    const levelSd = sm.PS.map((P) => Math.sqrt(Math.max(0, P[0][0])));
    const slope = slopeOn ? sm.mS.map((s) => 12 * s[1]) : null;
    const slopeSd = slopeOn ? sm.PS.map((P) => 12 * Math.sqrt(Math.max(0, P[1][1]))) : null;
    const seasonal = seasonalOn ? sm.mS.map((s) => s[seasIdx]) : null;
    const irregular = x.map((v, i) => (Number.isNaN(v) ? Number.NaN : v - level[i] - (seasonal ? seasonal[i] : 0)));
    const nTheta = 2 + (slopeOn ? 1 : 0) + (seasonalOn ? 1 : 0);
    const ic = armaIC(kf.loglik, nTheta, nObs);
    let p2t: number | null = null;
    if (seasonal) {
      const last = seasonal.slice(-SEASON);
      p2t = Math.max(...last) - Math.min(...last);
    }
    return { model, kf, seasIdx, level, levelSd, slope, slopeSd, seasonal, irregular, nTheta, aic: ic.aic, p2t, k: model.F.length, burn: model.burn };
  }, [applied, x, nObs]);

  const fc = useMemo(() => {
    if (h === 0) return null;
    const f = forecast(fit.kf, fit.model, h);
    const t = Array.from({ length: h }, (_, i) => yearOf(offset + n + i));
    const mean = f.obsMean.map((v) => v[0]);
    const sd = f.obsCov.map((c) => Math.sqrt(Math.max(0, c[0][0])));
    return {
      t,
      mean,
      sd,
      level: f.m.map((m) => m[0]),
      slope: applied.slopeOn ? f.m.map((m) => 12 * m[1]) : null,
      seasonal: applied.seasonalOn ? f.m.map((m) => m[fit.seasIdx]) : null,
    };
  }, [fit, h, offset, n, applied.slopeOn, applied.seasonalOn]);

  // Growth-rate readout: first January ≥ 1960 in the window, and December 2000.
  const growth = useMemo(() => {
    if (!fit.slope) return null;
    const firstJanInWindow = (12 - ((START_MONTH - 1 + offset) % 12)) % 12;
    const jan = Math.max((1960 - START_YEAR) * 12 + (1 - START_MONTH) - offset, firstJanInWindow);
    const dec = (2000 - START_YEAR) * 12 + (12 - START_MONTH) - offset;
    if (jan < 0 || dec >= n || jan >= dec) return null;
    return { from: fit.slope[jan], fromYear: Math.floor(yearOf(offset + jan)), to: fit.slope[dec] };
  }, [fit.slope, offset, n]);

  const setVar = (key: VarKey, v: number) => setConfig((c) => ({ ...c, logV: { ...c.logV, [key]: v } }));
  const resetToMle = () => setConfig((c) => ({ ...MLE_CONFIG, windowId: c.windowId }));
  const levelOnly = () => {
    if (!LEVEL_ONLY) return;
    setConfig((c) => ({
      ...c,
      slopeOn: LEVEL_ONLY.slope !== null,
      seasonalOn: LEVEL_ONLY.seasonal !== null,
      logV: { ...c.logV, level: Math.log10(LEVEL_ONLY.level), irregular: Math.log10(LEVEL_ONLY.irregular) },
    }));
  };
  const takeSnapshot = () =>
    setSnapshot({ label: modelLabel(applied), loglik: fit.kf.loglik, aic: fit.aic, burn: fit.burn, windowId: applied.windowId });

  const pending = config !== applied;
  const burnDiffers = snapshot !== null && snapshot.burn !== fit.burn;
  const windowDiffers = snapshot !== null && snapshot.windowId !== applied.windowId;

  // ── Layout ───────────────────────────────────────────────────────────────
  const fullW = Math.max(280, containerW);
  const tEnd = yearOf(offset + n);
  const xDomain: [number, number] = [times[0], tEnd + H_MAX / 12];
  const span = xDomain[1] - xDomain[0];
  const tickStep = span > 30 ? (isMobile ? 10 : 5) : isMobile ? 4 : 2;
  const xTicks: number[] = [];
  for (let t = Math.ceil(xDomain[0] / tickStep) * tickStep; t <= xDomain[1]; t += tickStep) xTicks.push(t);

  const panels: ('data' | 'slope' | 'seasonal' | 'irregular')[] = [
    'data',
    ...(applied.slopeOn ? (['slope'] as const) : []),
    ...(applied.seasonalOn ? (['seasonal'] as const) : []),
    'irregular',
  ];
  const innerH = isMobile ? 90 : 110;
  const heightOf = (i: number, p: string) => {
    const base = p === 'data' ? (isMobile ? 140 : 200) : innerH;
    return i === panels.length - 1 ? base + (MARGIN.bottom - MARGIN_BOTTOM_INNER) : base;
  };

  const dataRange = useMemo(() => {
    const vals = [...x, ...fit.level];
    if (fc) fc.mean.forEach((m, i) => vals.push(m - 2 * fc.sd[i], m + 2 * fc.sd[i]));
    return paddedRange(vals, 0.04);
  }, [x, fit.level, fc]);
  const slopeRange = useMemo(() => (fit.slope ? paddedRange([...fit.slope, ...(fc?.slope ?? [])], 0.25, 0.05) : ([0, 1] as [number, number])), [fit.slope, fc]);
  const seasRange = useMemo(() => (fit.seasonal ? symmetricRange(fit.seasonal, 0.1, 0.1) : ([-1, 1] as [number, number])), [fit.seasonal]);
  const irrRange = useMemo(() => symmetricRange(fit.irregular, 0.1, 0.05), [fit.irregular]);

  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-smoothed)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const badge = (bad: boolean) =>
    `rounded border px-2 py-0.5 text-xs font-medium ${bad ? 'border-[var(--color-ts-warn)] text-[var(--color-ts-warn)]' : ''}`;

  const sliderKeys: VarKey[] = ['level', ...(config.slopeOn ? (['slope'] as const) : []), ...(config.seasonalOn ? (['seasonal'] as const) : []), 'irregular'];
  const sliders = (
    <div className="grid grid-cols-1 items-end gap-3 sm:grid-cols-2">
      {sliderKeys.map((key) => (
        <LogSlider key={key} label={VAR_LABELS[key]} value={config.logV[key]} center={DEFAULT_LOG[key]} onChange={(v) => setVar(key, v)} />
      ))}
    </div>
  );

  const fmtLL = (v: number) => (Number.isFinite(v) ? v.toFixed(6) : '—');

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Components:</span>
        <button className={btn(true)} aria-pressed={true} disabled title="The level is always on" style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
          Level (always on)
        </button>
        <button
          className={btn(config.slopeOn)}
          aria-pressed={config.slopeOn}
          onClick={() => setConfig((c) => ({ ...c, slopeOn: !c.slopeOn }))}
          style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
        >
          Slope {config.slopeOn ? 'on' : 'off'}
        </button>
        <button
          className={btn(config.seasonalOn)}
          aria-pressed={config.seasonalOn}
          onClick={() => setConfig((c) => ({ ...c, seasonalOn: !c.seasonalOn }))}
          style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
        >
          Seasonal (12) {config.seasonalOn ? 'on' : 'off'}
        </button>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Window:</span>
        {(
          [
            { id: 'full', label: '1958–2001' },
            { id: 'recent', label: '1990–2001' },
          ] as { id: WindowId; label: string }[]
        ).map((w) => (
          <button
            key={w.id}
            className={btn(config.windowId === w.id)}
            aria-pressed={config.windowId === w.id}
            onClick={() => setConfig((c) => ({ ...c, windowId: w.id }))}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {w.label}
          </button>
        ))}
        <button onClick={resetToMle} className="rounded px-3 py-1 text-xs font-medium text-white" style={{ background: timeSeriesColors.smoothed }}>
          Reset to MLE
        </button>
        {LEVEL_ONLY && (
          <button onClick={levelOnly} className={btn(false)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {LEVEL_ONLY.label}
          </button>
        )}
      </div>

      <div className="mb-3">
        {isMobile ? (
          <details>
            <summary className="cursor-pointer text-sm font-semibold">Variances</summary>
            <div className="mt-2">{sliders}</div>
          </details>
        ) : (
          sliders
        )}
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>forecast horizon h (months)</span>
            <span className="font-mono">{h}</span>
          </span>
          <input
            type="range"
            min={0}
            max={H_MAX}
            step={1}
            value={h}
            onChange={(e) => setH(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-ts-forecast)]"
          />
        </label>
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={takeSnapshot} className="rounded px-3 py-1.5 text-xs font-medium text-white" style={{ background: timeSeriesColors.filtered }}>
            Snapshot ℓ / AIC / b
          </button>
          {snapshot && (
            <button onClick={() => setSnapshot(null)} className="text-xs underline" style={{ color: 'var(--color-text-muted)' }}>
              clear snapshot
            </button>
          )}
        </div>
      </div>

      <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-xs" aria-live="polite">
        <span>model: {modelLabel(applied)}</span>
        <span>k = {fit.k}</span>
        <span>burn b = {fit.burn}</span>
        <span>n = {n} ({gaps.length} empty)</span>
        <span>diffuse ℓ = {fmtLL(fit.kf.loglik)}</span>
        <span>
          AIC = {fit.aic.toFixed(3)} (n_ϑ = {fit.nTheta})
        </span>
        {fit.p2t !== null && <span>seasonal peak-to-trough (last 12 mo) = {fit.p2t.toFixed(2)} ppm</span>}
        {growth && (
          <span>
            growth Jan {growth.fromYear} → Dec 2000 = {growth.from.toFixed(3)} → {growth.to.toFixed(3)} ppm/yr
          </span>
        )}
        {pending && <span style={{ color: 'var(--color-text-muted)' }}>updating…</span>}
      </div>
      {snapshot && (
        <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-xs">
          <span style={{ color: 'var(--color-text-muted)' }}>
            snapshot ({snapshot.label}, {snapshot.windowId === 'full' ? '1958–2001' : '1990–2001'}):
          </span>
          <span>b = {snapshot.burn}</span>
          <span>ℓ = {fmtLL(snapshot.loglik)}</span>
          <span>AIC = {snapshot.aic.toFixed(3)}</span>
          <span>
            Δℓ = {(fit.kf.loglik - snapshot.loglik).toFixed(3)}, ΔAIC = {(fit.aic - snapshot.aic).toFixed(3)}
          </span>
          {burnDiffers ? (
            <span className={badge(true)} role="status">
              ⚠ different b — not comparable
            </span>
          ) : windowDiffers ? (
            <span className={badge(true)} role="status">
              ⚠ different window — not comparable
            </span>
          ) : (
            <span className={badge(false)} style={{ borderColor: 'var(--color-border)' }} role="status">
              same b — comparable
            </span>
          )}
        </div>
      )}
      {!snapshot && <div className="mb-3" />}

      {width > 0 ? (
        <>
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span style={{ color: timeSeriesColors.raw }}>● observed CO2</span> ·{' '}
            <span style={{ color: timeSeriesColors.smoothed }}>— smoothed level ±2 sd</span>
            {applied.slopeOn && <span style={{ color: timeSeriesColors.filtered }}> · — slope (ppm/yr)</span>}
            {applied.seasonalOn && <span style={{ color: timeSeriesColors.seasonal }}> · — seasonal</span>}
            <span style={{ color: timeSeriesColors.neutral }}> · — irregular</span>
            {h > 0 && <span style={{ color: timeSeriesColors.forecast }}> · - - forecast ±2 sd (hatched)</span>}
            <span> · shaded = empty month</span>
          </div>
          <div className="flex flex-col">
            {panels.map((p, i) => {
              const common = {
                width: fullW,
                height: heightOf(i, p),
                xDomain,
                xTicks,
                showXAxis: i === panels.length - 1,
                gaps,
                forecastFrom: h > 0 ? tEnd : null,
                forecastTo: tEnd + h / 12,
                hatchId,
                clipKey: p,
              };
              if (p === 'data') {
                return (
                  <Strip key={p} {...common} label="Observed CO2 with the smoothed level and its 2 sd band, plus forecasts" yLabel="CO2 (ppm)" yDomain={dataRange}>
                    {(sx, sy) => (
                      <>
                        <path d={bandPath(times, fit.level.map((v, j) => v - 2 * fit.levelSd[j]), fit.level.map((v, j) => v + 2 * fit.levelSd[j]), sx, sy)} fill={timeSeriesColors.smoothed} opacity={0.18} />
                        {x.map((v, j) =>
                          Number.isNaN(v) ? null : <circle key={j} cx={sx(times[j])} cy={sy(v)} r={n > 300 ? 1.1 : 1.6} fill={timeSeriesColors.raw} />,
                        )}
                        <path d={gapPath(times, fit.level, sx, sy)} fill="none" stroke={timeSeriesColors.smoothed} strokeWidth={2} />
                        {fc && (
                          <>
                            <path d={bandPath(fc.t, fc.mean.map((m, j) => m - 2 * fc.sd[j]), fc.mean.map((m, j) => m + 2 * fc.sd[j]), sx, sy)} fill={timeSeriesColors.forecast} opacity={0.22} />
                            <path d={gapPath(fc.t, fc.mean, sx, sy)} fill="none" stroke={timeSeriesColors.forecast} strokeWidth={1.8} strokeDasharray="5 3" />
                          </>
                        )}
                      </>
                    )}
                  </Strip>
                );
              }
              if (p === 'slope' && fit.slope && fit.slopeSd) {
                const sl = fit.slope;
                const sd = fit.slopeSd;
                return (
                  <Strip key={p} {...common} label="Smoothed slope in ppm per year with its 2 sd band" yLabel="slope (ppm/yr)" yDomain={slopeRange}>
                    {(sx, sy) => (
                      <>
                        <path d={bandPath(times, sl.map((v, j) => v - 2 * sd[j]), sl.map((v, j) => v + 2 * sd[j]), sx, sy)} fill={timeSeriesColors.filtered} opacity={0.15} />
                        <path d={gapPath(times, sl, sx, sy)} fill="none" stroke={timeSeriesColors.filtered} strokeWidth={1.8} />
                        {fc?.slope && <path d={gapPath(fc.t, fc.slope, sx, sy)} fill="none" stroke={timeSeriesColors.forecast} strokeWidth={1.5} strokeDasharray="5 3" />}
                      </>
                    )}
                  </Strip>
                );
              }
              if (p === 'seasonal' && fit.seasonal) {
                const se = fit.seasonal;
                return (
                  <Strip key={p} {...common} label="Smoothed 12-month seasonal effect in ppm" yLabel="seasonal" yDomain={seasRange}>
                    {(sx, sy) => (
                      <>
                        <line x1={sx(xDomain[0])} x2={sx(xDomain[1])} y1={sy(0)} y2={sy(0)} stroke="var(--color-text-muted)" strokeWidth={0.75} />
                        <path d={gapPath(times, se, sx, sy)} fill="none" stroke={timeSeriesColors.seasonal} strokeWidth={1.4} />
                        {fc?.seasonal && <path d={gapPath(fc.t, fc.seasonal, sx, sy)} fill="none" stroke={timeSeriesColors.forecast} strokeWidth={1.4} strokeDasharray="5 3" />}
                      </>
                    )}
                  </Strip>
                );
              }
              return (
                <Strip key={p} {...common} label="Irregular component: observation minus smoothed level and seasonal" yLabel="irregular" yDomain={irrRange}>
                  {(sx, sy) => (
                    <>
                      <line x1={sx(xDomain[0])} x2={sx(xDomain[1])} y1={sy(0)} y2={sy(0)} stroke="var(--color-text-muted)" strokeWidth={0.75} />
                      <path d={gapPath(times, fit.irregular, sx, sy)} fill="none" stroke={timeSeriesColors.neutral} strokeWidth={0.9} />
                    </>
                  )}
                </Strip>
              );
            })}
          </div>
        </>
      ) : (
        <div style={{ height: 560 }} aria-hidden="true" />
      )}
    </div>
  );
}
