/**
 * KalmanSmootherExplorer — featured interactive for Topic 36 §36.6.
 *
 * Thms 4 and 5 as one picture. The filter (indigo) looks only backward; the
 * Rauch–Tung–Striebel smoother (teal) looks both ways. Delete a run of years
 * and the filtered band fans out like √(P + hq) and snaps back at the next
 * observation, while the smoothed band bulges in the middle of the gap and
 * pinches at the observations on either side (Thm 4: skipping the update is
 * exact). A forecast is a gap at the end (Ex 10). The sd panel is Figure 6
 * right, live; "Sample paths" draws five FFBS paths (Thm 6). "MLE" maximizes
 * the diffuse log-likelihood (Def 5) by Nelder–Mead — Type-II ML (Rem 19).
 */
import { useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import {
  ffbs,
  kalmanFilter,
  localLevel,
  localLevelMLE,
  localLevelSteady,
  rtsSmoother,
  simulateSSM,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { kalmanExplorerPresets } from '../../data/time-series-data';
import { nileStartYear, nileValues } from '../../data/nile';

const MARGIN = { top: 10, right: 12, bottom: 34, left: 52 };
const MOBILE_BREAKPOINT = 640;
const N_SIM = 100;
const H_MAX = 30;
const N_PATHS = 5;
const FFBS_SEED = 3606;
const DK = { q: 1469.1, r: 15099 };

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function bandPath(
  xs: number[],
  lo: number[],
  hi: number[],
  sx: (v: number) => number,
  sy: (v: number) => number,
): string {
  const up = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(hi[i]).toFixed(2)}`).join('');
  const down = xs
    .map((_, k) => xs.length - 1 - k)
    .map((i) => `L${sx(xs[i]).toFixed(2)},${sy(lo[i]).toFixed(2)}`)
    .join('');
  return `${up}${down}Z`;
}

function niceTicks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(3)));
  return out;
}

function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
}

function maskFromGaps(n: number, gaps: [number, number][]): boolean[] {
  const m = new Array<boolean>(n).fill(false);
  for (const [a, b] of gaps) for (let i = a; i < Math.min(b, n); i++) m[i] = true;
  return m;
}

function LogSlider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-center justify-between text-xs">
        <span>{label}</span>
        <span className="font-mono">{(10 ** value).toPrecision(5)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={0.0001}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--color-ts-smoothed)]"
      />
    </label>
  );
}

export default function KalmanSmootherExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = kalmanExplorerPresets[presetIdx];
  const [logQ, setLogQ] = useState(Math.log10(preset.q));
  const [logR, setLogR] = useState(Math.log10(preset.r));
  const [seed, setSeed] = useState(preset.seed);
  const [simParams, setSimParams] = useState({ q: preset.q, r: preset.r, seed: preset.seed });
  const [missing, setMissing] = useState<boolean[]>(() => maskFromGaps(nileValues.length, preset.gaps));
  const [horizon, setHorizon] = useState(0);
  const [show, setShow] = useState({ obs: true, filt: true, smooth: true, paths: false });
  const [pathDraw, setPathDraw] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);
  const dragStart = useRef<{ px: number; t: number } | null>(null);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const q = 10 ** logQ;
  const r = 10 ** logR;

  // Base series: the Nile, or a local level simulated at the parameters in force when Resample was pressed.
  const base = useMemo(() => {
    if (preset.source === 'nile') return { values: nileValues, start: nileStartYear, unit: 'flow (10⁸ m³)' };
    const sim = simulateSSM(
      { ...localLevel(simParams.q, simParams.r), m1: [1000], P1: [[simParams.q]] },
      N_SIM,
      createSeededRng(simParams.seed),
    );
    return { values: sim.x.map((v) => Number(v[0].toFixed(2))), start: 1, unit: 'x_t' };
  }, [preset.source, simParams]);

  const n = base.values.length;
  const T = n + horizon;
  const times = useMemo(() => Array.from({ length: T }, (_, t) => base.start + t), [T, base.start]);
  const x = useMemo(
    () => [...base.values.map((v, t) => (missing[t] ? Number.NaN : v)), ...new Array<number>(horizon).fill(Number.NaN)],
    [base.values, missing, horizon],
  );

  const model = useMemo(() => localLevel(q, r), [q, r]);
  const kf = useMemo(() => kalmanFilter(x, model), [x, model]);
  const sm = useMemo(() => rtsSmoother(kf, model), [kf, model]);
  const steady = useMemo(() => localLevelSteady(q, r), [q, r]);
  const nObs = useMemo(() => kf.observed.filter(Boolean).length, [kf]);

  const series = useMemo(() => {
    const mf = kf.mFilt.map((m) => m[0]);
    const sf = kf.PFilt.map((P) => Math.sqrt(Math.max(P[0][0], 0)));
    const ms = sm.mS.map((m) => m[0]);
    const ss = sm.PS.map((P) => Math.sqrt(Math.max(P[0][0], 0)));
    return { mf, sf, ms, ss };
  }, [kf, sm]);

  const paths = useMemo(() => {
    if (!show.paths) return [];
    const rng = createSeededRng(FFBS_SEED + pathDraw);
    return Array.from({ length: N_PATHS }, () => ffbs(kf, model, rng).map((z) => z[0]));
  }, [show.paths, pathDraw, kf, model]);

  const choosePreset = (i: number) => {
    const p = kalmanExplorerPresets[i];
    setPresetIdx(i);
    setLogQ(Math.log10(p.q));
    setLogR(Math.log10(p.r));
    setSeed(p.seed);
    setSimParams({ q: p.q, r: p.r, seed: p.seed });
    setMissing(maskFromGaps(p.source === 'nile' ? nileValues.length : N_SIM, p.gaps));
    setHover(null);
  };

  const runMLE = () => {
    const fit = localLevelMLE(x.slice(0, n));
    setLogQ(Math.min(5, Math.max(1, Math.log10(fit.q))));
    setLogR(Math.min(5, Math.max(2.5, Math.log10(fit.r))));
  };

  // ── Layout ────────────────────────────────────────────────────────────────
  const fullW = Math.max(280, containerW);
  const mainW = isMobile ? fullW : Math.max(200, Math.floor(fullW * 0.68) - 8);
  const sideW = isMobile ? fullW : Math.max(120, fullW - mainW - 16);
  const mainH = isMobile ? 260 : 320;
  const sideH = isMobile ? 120 : 320;
  const plotW = Math.max(10, mainW - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, mainH - MARGIN.top - MARGIN.bottom);
  const sidePlotW = Math.max(10, sideW - MARGIN.left - MARGIN.right);
  const sidePlotH = Math.max(10, sideH - MARGIN.top - MARGIN.bottom);

  const xDomain: [number, number] = [base.start - 1, base.start + T];
  const yDomain = useMemo<[number, number]>(() => {
    const vals = base.values.slice();
    series.ms.forEach((m, t) => vals.push(m - 2 * series.ss[t], m + 2 * series.ss[t]));
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = 0.06 * (hi - lo || 1);
    return [lo - pad, hi + pad];
  }, [base.values, series]);
  // Ignore the diffuse start-up (sd ≈ √κ before the first observation) when scaling the sd axis.
  const sdMax = useMemo(() => {
    const sorted = series.sf.slice().sort((a, b) => a - b);
    const cap = 5 * sorted[Math.floor(sorted.length / 2)];
    return Math.max(1, 1.08 * Math.max(...series.ss.filter((v) => v < cap), ...series.sf.filter((v) => v < cap)));
  }, [series]);

  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = (v: number) => plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const ssx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * sidePlotW;
  const ssy = (v: number) => sidePlotH - (v / sdMax) * sidePlotH;
  const cellW = Math.max(1, sx(1) - sx(0));

  const tAt = (clientX: number, rect: DOMRect) => {
    const px = clientX - rect.left - MARGIN.left;
    const v = xDomain[0] + (px / plotW) * (xDomain[1] - xDomain[0]);
    return Math.max(0, Math.min(T - 1, Math.round(v - base.start)));
  };

  const toggle = (t: number) => {
    if (t >= n) return;
    setMissing((m) => m.map((v, i) => (i === t ? !v : v)));
  };

  const onPointerDown = (ev: React.PointerEvent<SVGSVGElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    dragStart.current = { px: ev.clientX, t: tAt(ev.clientX, rect) };
    try {
      ev.currentTarget.setPointerCapture(ev.pointerId);
    } catch {
      /* pointer no longer active */
    }
  };
  const onPointerMove = (ev: React.PointerEvent<SVGSVGElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const t = tAt(ev.clientX, rect);
    setHover(t);
    const s = dragStart.current;
    if (s && !isMobile && Math.abs(ev.clientX - s.px) > 6) setDrag({ a: Math.min(s.t, t), b: Math.max(s.t, t) });
  };
  const onPointerUp = (ev: React.PointerEvent<SVGSVGElement>) => {
    const s = dragStart.current;
    dragStart.current = null;
    if (!s) return;
    if (drag) {
      const { a, b } = drag;
      setMissing((m) => m.map((v, i) => (i >= a && i <= b ? true : v)));
      setDrag(null);
      return;
    }
    if (Math.abs(ev.clientX - s.px) <= 6) toggle(s.t);
  };

  const h = hover !== null && hover < T ? hover : null;
  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-smoothed)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const border = { borderWidth: 1, borderColor: 'var(--color-border)' };
  const axisColor = 'var(--color-text-muted)';
  const yStep = niceStep(yDomain[1] - yDomain[0], isMobile ? 4 : 6);
  const xStep = T > 60 ? 20 : 10;
  const fcStart = base.start + n - 0.5;
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3));

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Data:</span>
        {kalmanExplorerPresets.map((p, i) => (
          <button key={p.id} onClick={() => choosePreset(i)} className={btn(i === presetIdx)} style={border} aria-pressed={i === presetIdx}>
            {p.label}
          </button>
        ))}
        {preset.source === 'simulated' && (
          <span className="flex items-center gap-2 text-xs">
            <label className="flex items-center gap-1">
              seed
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                className="w-20 rounded border px-1 py-0.5 font-mono"
                style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
              />
            </label>
            <button
              onClick={() => {
                setSimParams({ q, r, seed });
                setMissing(new Array<boolean>(N_SIM).fill(false));
              }}
              className="rounded px-3 py-1 text-xs font-medium text-white"
              style={{ background: timeSeriesColors.smoothed }}
            >
              Resample at current (q, r)
            </button>
          </span>
        )}
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-3">
        <LogSlider label="level variance q (log scale)" value={logQ} min={1} max={5} onChange={setLogQ} />
        <LogSlider label="observation variance r (log scale)" value={logR} min={2.5} max={5} onChange={setLogR} />
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>forecast horizon h</span>
            <span className="font-mono">{horizon}</span>
          </span>
          <input
            type="range"
            min={0}
            max={H_MAX}
            step={1}
            value={horizon}
            onChange={(e) => setHorizon(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-ts-forecast)]"
          />
        </label>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <button onClick={runMLE} className="rounded px-3 py-1 font-medium text-white" style={{ background: timeSeriesColors.filtered }}>
          MLE
        </button>
        <button
          onClick={() => {
            setLogQ(Math.log10(DK.q));
            setLogR(Math.log10(DK.r));
          }}
          className={btn(false)}
          style={border}
        >
          DK values (1469.1, 15099)
        </button>
        <button onClick={() => setMissing(maskFromGaps(n, preset.gaps))} className={btn(false)} style={border}>
          Restore all
        </button>
        <span className="mx-1 h-4 w-px" style={{ background: 'var(--color-border)' }} aria-hidden="true" />
        {(
          [
            ['obs', 'observations'],
            ['filt', 'filtered band'],
            ['smooth', 'smoothed band'],
            ['paths', 'sample paths (FFBS)'],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setShow((s) => ({ ...s, [key]: !s[key] }))}
            className={btn(show[key])}
            style={border}
            aria-pressed={show[key]}
          >
            {label}
          </button>
        ))}
        {show.paths && (
          <button onClick={() => setPathDraw((d) => d + 1)} className={btn(false)} style={border}>
            Redraw paths
          </button>
        )}
      </div>

      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        <span>diffuse ℓ(q, r) = {kf.loglik.toFixed(4)}</span>
        <span>q/r = {(q / r).toFixed(5)}</span>
        <span>K∞ = {steady.K.toFixed(5)}</span>
        <span>
          |𝒪| = {nObs} of {T}
        </span>
      </div>
      <div className="mb-2 min-h-[1.25rem] font-mono text-xs" style={{ color: 'var(--color-text-muted)' }} aria-live="polite">
        {h !== null ? (
          <>
            t = {times[h]}: x_t = {Number.isNaN(x[h]) ? (h >= n ? 'forecast' : 'missing') : fmt(x[h])} · filtered {fmt(series.mf[h])} ± {fmt(2 * series.sf[h])} ·
            smoothed {fmt(series.ms[h])} ± {fmt(2 * series.ss[h])}
            {h < T - 1 ? ` · G_t = ${sm.G[h][0][0].toFixed(4)}` : ' · G_T undefined (no future)'}
          </>
        ) : (
          <>Hover or tap the plot for a readout; tap a point to toggle it missing{isMobile ? '' : ', drag across years to delete a range'}.</>
        )}
      </div>

      {width > 0 ? (
        <div className={isMobile ? 'flex flex-col gap-3' : 'flex gap-4'}>
          <div className="min-w-0">
            <svg
              width={mainW}
              height={mainH}
              className="block select-none"
              role="img"
              aria-label="Series with filtered and smoothed level bands; missing years and the forecast region are shaded"
              style={{ touchAction: 'pan-y', cursor: 'pointer' }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={() => {
                setHover(null);
              }}
              onPointerCancel={() => {
                dragStart.current = null;
                setDrag(null);
              }}
            >
              <defs>
                <clipPath id="kse-clip">
                  <rect x={0} y={0} width={plotW} height={plotH} />
                </clipPath>
                <pattern id="kse-hatch" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1={0} y1={0} x2={0} y2={6} stroke={timeSeriesColors.forecast} strokeWidth={1} opacity={0.35} />
                </pattern>
              </defs>
              <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
                {niceTicks(yDomain[0], yDomain[1], yStep).map((t) => (
                  <g key={t}>
                    <line x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
                    <text x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
                      {t}
                    </text>
                  </g>
                ))}
                <g clipPath="url(#kse-clip)">
                  {missing.slice(0, n).map((m, t) =>
                    m ? <rect key={t} x={sx(base.start + t) - cellW / 2} y={0} width={cellW} height={plotH} fill={timeSeriesColors.warn} opacity={0.08} /> : null,
                  )}
                  {horizon > 0 && (
                    <>
                      <rect x={sx(fcStart)} y={0} width={Math.max(0, sx(base.start + T - 0.5) - sx(fcStart))} height={plotH} fill={timeSeriesColors.forecast} opacity={0.08} />
                      <rect x={sx(fcStart)} y={0} width={Math.max(0, sx(base.start + T - 0.5) - sx(fcStart))} height={plotH} fill="url(#kse-hatch)" />
                    </>
                  )}
                  {drag && (
                    <rect
                      x={sx(base.start + drag.a) - cellW / 2}
                      y={0}
                      width={Math.max(cellW, sx(base.start + drag.b) - sx(base.start + drag.a) + cellW)}
                      height={plotH}
                      fill={timeSeriesColors.warn}
                      opacity={0.18}
                    />
                  )}
                  {show.filt && (
                    <>
                      <path
                        d={bandPath(times, series.mf.map((m, t) => m - 2 * series.sf[t]), series.mf.map((m, t) => m + 2 * series.sf[t]), sx, sy)}
                        fill={timeSeriesColors.filtered}
                        opacity={0.12}
                      />
                      <path d={linePath(times, series.mf, sx, sy)} fill="none" stroke={timeSeriesColors.filtered} strokeWidth={1.4} />
                    </>
                  )}
                  {show.smooth && (
                    <>
                      <path
                        d={bandPath(times, series.ms.map((m, t) => m - 2 * series.ss[t]), series.ms.map((m, t) => m + 2 * series.ss[t]), sx, sy)}
                        fill={timeSeriesColors.smoothed}
                        opacity={0.2}
                      />
                      <path d={linePath(times, series.ms, sx, sy)} fill="none" stroke={timeSeriesColors.smoothed} strokeWidth={2.4} />
                    </>
                  )}
                  {paths.map((p, i) => (
                    <path key={i} d={linePath(times, p, sx, sy)} fill="none" stroke={timeSeriesColors.smoothed} strokeWidth={0.9} opacity={0.6} strokeDasharray="4 2" />
                  ))}
                  {show.obs &&
                    base.values.map((v, t) =>
                      missing[t] ? (
                        <circle key={t} cx={sx(base.start + t)} cy={sy(v)} r={2.6} fill="none" stroke={timeSeriesColors.warn} strokeWidth={1} opacity={0.7} />
                      ) : (
                        <circle key={t} cx={sx(base.start + t)} cy={sy(v)} r={2.6} fill={timeSeriesColors.raw} />
                      ),
                    )}
                  {h !== null && <line x1={sx(times[h])} x2={sx(times[h])} y1={0} y2={plotH} stroke={axisColor} strokeDasharray="2 3" />}
                </g>
                <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
                <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
                {niceTicks(xDomain[0] + 1, xDomain[1] - 1, xStep).map((t) => (
                  <g key={t}>
                    <line x1={sx(t)} y1={plotH} x2={sx(t)} y2={plotH + 4} stroke={axisColor} />
                    <text x={sx(t)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                      {t}
                    </text>
                  </g>
                ))}
                <text x={plotW / 2} y={plotH + 30} textAnchor="middle" fontSize={11} fill="var(--color-text)">
                  {preset.source === 'nile' ? 'year' : 'time t'}
                </text>
                <text x={-40} y={plotH / 2} textAnchor="middle" fontSize={11} fill="var(--color-text)" transform={`rotate(-90 ${-40} ${plotH / 2})`}>
                  {base.unit}
                </text>
              </g>
            </svg>
            <div className="mt-1 flex flex-wrap gap-x-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              <span style={{ color: timeSeriesColors.filtered }}>— filtered m_t ± 2√P_t</span>
              <span style={{ color: timeSeriesColors.smoothed }}>━ smoothed mˢ_t ± 2√Pˢ_t</span>
              <span>● observed · ○ missing (tap to restore)</span>
              {horizon > 0 && <span style={{ color: timeSeriesColors.forecast }}>▨ forecast region</span>}
            </div>
          </div>

          <div className="min-w-0">
            <svg width={sideW} height={sideH} className="block" role="img" aria-label="Posterior standard deviation over time, filtered against smoothed">
              <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
                {niceTicks(0, sdMax, niceStep(sdMax, isMobile ? 2 : 5)).map((t) => (
                  <g key={t}>
                    <line x1={0} y1={ssy(t)} x2={sidePlotW} y2={ssy(t)} stroke="var(--color-viz-grid)" />
                    <text x={-6} y={ssy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
                      {t}
                    </text>
                  </g>
                ))}
                <defs>
                  <clipPath id="kse-clip-side">
                    <rect x={0} y={0} width={sidePlotW} height={sidePlotH} />
                  </clipPath>
                </defs>
                <g clipPath="url(#kse-clip-side)">
                  {missing.slice(0, n).map((m, t) =>
                    m ? (
                      <rect
                        key={t}
                        x={ssx(base.start + t) - Math.max(1, ssx(1) - ssx(0)) / 2}
                        y={0}
                        width={Math.max(1, ssx(1) - ssx(0))}
                        height={sidePlotH}
                        fill={timeSeriesColors.warn}
                        opacity={0.08}
                      />
                    ) : null,
                  )}
                  {horizon > 0 && (
                    <rect x={ssx(fcStart)} y={0} width={Math.max(0, ssx(base.start + T - 0.5) - ssx(fcStart))} height={sidePlotH} fill={timeSeriesColors.forecast} opacity={0.08} />
                  )}
                  <path d={linePath(times, series.sf, ssx, ssy)} fill="none" stroke={timeSeriesColors.filtered} strokeWidth={1.4} />
                  <path d={linePath(times, series.ss, ssx, ssy)} fill="none" stroke={timeSeriesColors.smoothed} strokeWidth={2.4} />
                  {h !== null && <line x1={ssx(times[h])} x2={ssx(times[h])} y1={0} y2={sidePlotH} stroke={axisColor} strokeDasharray="2 3" />}
                </g>
                <line x1={0} y1={sidePlotH} x2={sidePlotW} y2={sidePlotH} stroke={axisColor} />
                <line x1={0} y1={0} x2={0} y2={sidePlotH} stroke={axisColor} />
                {niceTicks(xDomain[0] + 1, xDomain[1] - 1, isMobile ? xStep : xStep * 2).map((t) => (
                  <text key={t} x={ssx(t)} y={sidePlotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                    {t}
                  </text>
                ))}
                <text x={sidePlotW / 2} y={sidePlotH + 30} textAnchor="middle" fontSize={11} fill="var(--color-text)">
                  posterior sd
                </text>
              </g>
            </svg>
          </div>
        </div>
      ) : (
        <div style={{ height: 360 }} aria-hidden="true" />
      )}
    </div>
  );
}
