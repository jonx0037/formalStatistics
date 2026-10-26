/**
 * PeriodogramSmoother — interactive for Topic 35 §35.8.
 *
 * Feel Thms 12–13: the raw periodogram I_n(ω_j) never settles as n grows
 * (each ordinate is f(ω_j) times an Exp(1) variable, at every n), and
 * smoothing across neighboring frequencies trades variance for bias exactly as
 * a KDE bandwidth does (Rem 19). The Bartlett lag window is variance-matched
 * to Daniell with M = round(3n / (2(2m+1))) (Rem 20). The bias–variance
 * panel reruns Figure 9's Monte Carlo live at the current n (Daniell, 50
 * replications, m = 0..60), in chunks so the page stays responsive.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import {
  armaSpectrum,
  periodogram,
  simulateARMA,
  smoothPeriodogram,
  type SmootherKind,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { periodogramPresets } from '../../data/time-series-data';
import { sunspotStartYear, sunspotValues } from '../../data/sunspots';

const MARGIN = { top: 10, right: 12, bottom: 34, left: 50 };
const MOBILE_BREAKPOINT = 640;
const N_OPTIONS = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
const M_MAX = 60;
const MC_REPS = 50;
const HIST_MAX = 6;
const HIST_BINS = 24;

const SMOOTHERS: { id: SmootherKind; label: string }[] = [
  { id: 'daniell', label: 'Daniell' },
  { id: 'modifiedDaniell', label: 'Modified Daniell' },
  { id: 'bartlettLag', label: 'Bartlett lag window' },
];

const PI_TICKS = [
  { v: 0, label: '0' },
  { v: Math.PI / 4, label: 'π/4' },
  { v: Math.PI / 2, label: 'π/2' },
  { v: (3 * Math.PI) / 4, label: '3π/4' },
  { v: Math.PI, label: 'π' },
];

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function decadeTicks(lo: number, hi: number): { v: number; label: string }[] {
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const stride = b - a > 6 ? 2 : 1;
  const out: { v: number; label: string }[] = [];
  for (let k = a; k <= b; k += stride) out.push({ v: 10 ** k, label: k === 0 ? '1' : `1e${k}` });
  return out;
}

function linearTicks(lo: number, hi: number, count = 5): { v: number; label: string }[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((s) => s * mag).find((s) => s >= raw) ?? raw;
  const out: { v: number; label: string }[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push({ v: t, label: String(Number(t.toPrecision(3))) });
  return out;
}

function Frame({
  width,
  height,
  label,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  xTicks,
  yTicks,
  logY = false,
  children,
}: {
  width: number;
  height: number;
  label: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks: { v: number; label: string }[];
  yTicks: { v: number; label: string }[];
  logY?: boolean;
  children: (sx: (v: number) => number, sy: (v: number) => number, plotW: number, plotH: number) => React.ReactNode;
}) {
  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom);
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const ty = (v: number) => (logY ? Math.log10(Math.max(v, 1e-300)) : v);
  const y0 = ty(yDomain[0]);
  const y1 = ty(yDomain[1]);
  const sy = (v: number) => plotH - ((Math.min(Math.max(ty(v), y0), y1) - y0) / (y1 - y0)) * plotH;
  const axisColor = 'var(--color-text-muted)';
  return (
    <svg width={width} height={height} className="block" role="img" aria-label={label}>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {yTicks.map((t) => (
          <g key={t.label}>
            <line x1={0} y1={sy(t.v)} x2={plotW} y2={sy(t.v)} stroke="var(--color-viz-grid)" />
            <text x={-6} y={sy(t.v) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
              {t.label}
            </text>
          </g>
        ))}
        {children(sx, sy, plotW, plotH)}
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
        {xTicks.map((t) => (
          <g key={t.label}>
            <line x1={sx(t.v)} y1={plotH} x2={sx(t.v)} y2={plotH + 4} stroke={axisColor} />
            <text x={sx(t.v)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
              {t.label}
            </text>
          </g>
        ))}
        <text x={plotW / 2} y={plotH + 30} textAnchor="middle" fontSize={11} fill="var(--color-text)">
          {xLabel}
        </text>
        <text
          x={-40}
          y={plotH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--color-text)"
          transform={`rotate(-90 ${-40} ${plotH / 2})`}
        >
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

/** Daniell estimates for every m = 0..mMax from one periodogram, by prefix sums over the reflected index. */
function daniellAllM(I: number[], n: number, mMax: number): number[][] {
  const J = Math.floor(n / 2);
  const reflect = (i: number) => {
    let k = ((i % n) + n) % n;
    if (k > J) k = n - k;
    return k === 0 ? 1 : k;
  };
  const lo = 1 - mMax;
  const ext = Array.from({ length: J + 2 * mMax }, (_, k) => I[reflect(lo + k)]);
  const pre = new Float64Array(ext.length + 1);
  for (let k = 0; k < ext.length; k++) pre[k + 1] = pre[k] + ext[k];
  return Array.from({ length: mMax + 1 }, (_, m) =>
    Array.from({ length: J }, (_, idx) => {
      const j = idx + 1;
      const a = j - m - lo;
      const b = j + m - lo + 1;
      return (pre[b] - pre[a]) / (2 * m + 1);
    }),
  );
}

interface MCResult {
  n: number;
  presetId: string;
  mse: number[];
  variance: number[];
  bias2: number[];
}

export default function PeriodogramSmoother() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const def = periodogramPresets[0];
  const [presetId, setPresetId] = useState(def.id);
  const [n, setN] = useState(def.n);
  const [m, setM] = useState(def.m);
  const [kind, setKind] = useState<SmootherKind>('daniell');
  const [logY, setLogY] = useState(true);
  const [seed, setSeed] = useState(def.seed);
  const [mc, setMc] = useState<MCResult | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [showMC, setShowMC] = useState(false);
  const cancelRef = useRef<{ cancelled: boolean } | null>(null);

  useEffect(() => () => {
    if (cancelRef.current) cancelRef.current.cancelled = true;
  }, []);

  const preset = periodogramPresets.find((p) => p.id === presetId) ?? def;
  const isSunspots = preset.kind === 'sunspots';
  const nEff = isSunspots ? sunspotValues.length : n;
  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;

  const selectPreset = (id: string) => {
    const p = periodogramPresets.find((q) => q.id === id) ?? def;
    if (cancelRef.current) cancelRef.current.cancelled = true;
    setProgress(null);
    setPresetId(p.id);
    setM(p.m);
    if (p.kind === 'arma') setN(p.n);
  };

  const x = useMemo(
    () => (isSunspots ? sunspotValues : simulateARMA(preset.params!, nEff, createSeededRng(seed))),
    [isSunspots, preset, nEff, seed],
  );
  const raw = useMemo(() => periodogram(x), [x]);
  const interior = useMemo(() => ({ omega: raw.omega.slice(1), I: raw.f.slice(1) }), [raw]);
  const smooth = useMemo(() => smoothPeriodogram(x, m, kind), [x, m, kind]);
  const truth = useMemo(
    () => (isSunspots ? null : armaSpectrum(preset.params!, interior.omega)),
    [isSunspots, preset, interior.omega],
  );
  const truthCurve = useMemo(() => {
    if (isSunspots) return null;
    const om = Array.from({ length: 400 }, (_, i) => (Math.PI * (i + 0.5)) / 400);
    return { om, f: armaSpectrum(preset.params!, om) };
  }, [isSunspots, preset]);

  const peak = useMemo(() => {
    let k = 0;
    smooth.f.forEach((v, i) => {
      if (v > smooth.f[k]) k = i;
    });
    return smooth.omega[k];
  }, [smooth]);

  const histogram = useMemo(() => {
    if (!truth) return null;
    const counts = new Array<number>(HIST_BINS).fill(0);
    const w = HIST_MAX / HIST_BINS;
    let total = 0;
    // Interior ordinates only: drop ω = π, whose law is χ²₁ rather than Exp(1).
    for (let j = 0; j < interior.I.length; j++) {
      if (Math.abs(interior.omega[j] - Math.PI) < 1e-12) continue;
      total++;
      const r = interior.I[j] / truth[j];
      if (r < HIST_MAX) counts[Math.floor(r / w)]++;
    }
    return { density: counts.map((c) => c / (total * w)), total };
  }, [interior, truth]);

  // Same half-to-even rounding as smoothPeriodogram (Python's round), e.g. 154.5 → 154 for sunspots at m = 1.
  const mRaw = (3 * nEff) / (2 * (2 * m + 1));
  const mRounded = Math.round(mRaw);
  const M = Math.max(1, Math.abs(mRaw % 1) === 0.5 && mRounded % 2 !== 0 ? mRounded - 1 : mRounded);
  const bandwidth = (2 * Math.PI * (2 * m + 1)) / nEff;
  const mcDisabled = isSunspots || (isMobile && n > 4096);

  const runMC = () => {
    if (mcDisabled || !preset.params) return;
    if (cancelRef.current) cancelRef.current.cancelled = true;
    const token = { cancelled: false };
    cancelRef.current = token;
    const params = preset.params;
    const J = Math.floor(n / 2);
    const om = Array.from({ length: J }, (_, i) => (2 * Math.PI * (i + 1)) / n);
    const f = armaSpectrum(params, om);
    const sum = Array.from({ length: M_MAX + 1 }, () => new Float64Array(J));
    const sum2 = Array.from({ length: M_MAX + 1 }, () => new Float64Array(J));
    const rng = createSeededRng(seed + 1000);
    let rep = 0;
    setProgress(0);
    setShowMC(true);
    const step = () => {
      if (token.cancelled) return;
      const xr = simulateARMA(params, n, rng);
      const all = daniellAllM(periodogram(xr).f, n, M_MAX);
      for (let mm = 0; mm <= M_MAX; mm++) {
        const s = sum[mm];
        const s2 = sum2[mm];
        const est = all[mm];
        for (let j = 0; j < J; j++) {
          const r = est[j] / f[j];
          s[j] += r;
          s2[j] += r * r;
        }
      }
      rep++;
      setProgress(rep / MC_REPS);
      if (rep < MC_REPS) {
        setTimeout(step, 0);
        return;
      }
      const mse: number[] = [];
      const variance: number[] = [];
      const bias2: number[] = [];
      for (let mm = 0; mm <= M_MAX; mm++) {
        let v = 0;
        let b = 0;
        for (let j = 0; j < J; j++) {
          const mean = sum[mm][j] / MC_REPS;
          v += sum2[mm][j] / MC_REPS - mean * mean;
          b += (mean - 1) ** 2;
        }
        variance.push(v / J);
        bias2.push(b / J);
        mse.push((v + b) / J);
      }
      setMc({ n, presetId, mse, variance, bias2 });
      setProgress(null);
    };
    setTimeout(step, 0);
  };

  const mcStale = mc && (mc.n !== n || mc.presetId !== presetId);
  const mStar = mc ? mc.mse.reduce((best, v, i, arr) => (v < arr[best] ? i : best), 0) : null;

  // ── Spectrum panel ranges ────────────────────────────────────────────────
  const specRange = useMemo(() => {
    const vals = [...interior.I, ...smooth.f, ...(truthCurve ? truthCurve.f : [])].filter((v) => v > 0);
    const hi = Math.max(...vals);
    if (!logY) return { lo: 0, hi: hi * 1.05 };
    const ref = truthCurve ? Math.min(...truthCurve.f) : Math.min(...smooth.f.filter((v) => v > 0));
    const lo = Math.max(Math.min(ref / 20, ...smooth.f.filter((v) => v > 0)), hi * 1e-6);
    return { lo: 10 ** Math.floor(Math.log10(lo)), hi: 10 ** Math.ceil(Math.log10(hi)) };
  }, [interior.I, smooth.f, truthCurve, logY]);

  const fullW = Math.max(280, containerW);
  const halfW = isMobile ? fullW : (fullW - 16) / 2;
  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-smoothed-spectrum)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const nIdx = N_OPTIONS.indexOf(n);
  const dotR = nEff > 2048 ? 1 : nEff > 512 ? 1.5 : 2.2;

  const bvPanel = mc && !mcStale && (
    <Frame
      width={isMobile ? fullW : halfW}
      height={230}
      label="Relative MSE, variance and squared bias of the Daniell estimate against the half-width m"
      xLabel="half-width m"
      yLabel="relative error"
      xDomain={[0, M_MAX]}
      yDomain={[Math.max(1e-4, Math.min(...mc.bias2.filter((v) => v > 0), ...mc.variance) * 0.8), Math.max(...mc.mse) * 1.2]}
      xTicks={[0, 10, 20, 30, 40, 50, 60].map((v) => ({ v, label: String(v) }))}
      yTicks={decadeTicks(Math.max(1e-4, Math.min(...mc.bias2.filter((v) => v > 0), ...mc.variance) * 0.8), Math.max(...mc.mse) * 1.2)}
      logY
    >
      {(sx, sy, _w, plotH) => {
        const ms = mc.mse.map((_, i) => i);
        return (
          <>
            <path d={linePath(ms, mc.variance, sx, sy)} fill="none" stroke={timeSeriesColors.ar} strokeWidth={1.5} strokeDasharray="6 4" />
            <path d={linePath(ms, mc.bias2.map((v) => Math.max(v, 1e-6)), sx, sy)} fill="none" stroke={timeSeriesColors.ma} strokeWidth={1.5} strokeDasharray="2 3" />
            <path d={linePath(ms, mc.mse, sx, sy)} fill="none" stroke={timeSeriesColors.smoothedSpectrum} strokeWidth={2.2} />
            <line x1={sx(m)} x2={sx(m)} y1={0} y2={plotH} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
            {mStar !== null && <circle cx={sx(mStar)} cy={sy(mc.mse[mStar])} r={5} fill="none" stroke={timeSeriesColors.smoothedSpectrum} strokeWidth={2} />}
          </>
        );
      }}
    </Frame>
  );

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Process:</span>
        {periodogramPresets.map((p) => (
          <button key={p.id} onClick={() => selectPreset(p.id)} className={btn(p.id === presetId)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Smoother:</span>
        {SMOOTHERS.map((s) => (
          <button key={s.id} onClick={() => setKind(s.id)} className={btn(s.id === kind)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {s.label}
          </button>
        ))}
        <label className="ml-2 flex items-center gap-1 text-xs">
          <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} />
          log scale
        </label>
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>sample size n{isSunspots ? ' (fixed: 309 years)' : ''}</span>
            <span className="font-mono">{nEff}</span>
          </span>
          <input
            type="range"
            min={0}
            max={N_OPTIONS.length - 1}
            step={1}
            value={isSunspots ? 0 : nIdx}
            disabled={isSunspots}
            onChange={(e) => setN(N_OPTIONS[parseInt(e.target.value, 10)])}
            className="w-full accent-[var(--color-ts-smoothed-spectrum)] disabled:opacity-40"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>half-width m</span>
            <span className="font-mono">{m}</span>
          </span>
          <input
            type="range"
            min={0}
            max={M_MAX}
            step={1}
            value={m}
            onChange={(e) => setM(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-ts-smoothed-spectrum)]"
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          disabled={isSunspots}
          className="rounded px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
          style={{ background: timeSeriesColors.smoothedSpectrum }}
        >
          Resample (seed {seed})
        </button>
      </div>

      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        <span>bandwidth B = 2π(2m+1)/n = {bandwidth.toFixed(4)}</span>
        {kind === 'bartlettLag' && <span>lag-window truncation M = {M}</span>}
        <span>relative SE ≈ 1/√(2m+1) = {(1 / Math.sqrt(2 * m + 1)).toFixed(3)}</span>
        <span>
          estimated peak ω = {peak.toFixed(4)}, period 2π/ω = {((2 * Math.PI) / peak).toFixed(2)} {isSunspots ? 'years' : 'steps'}
        </span>
      </div>

      {width > 0 ? (
        <>
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span style={{ color: timeSeriesColors.raw }}>● raw periodogram I_n(ω_j)</span>
            {truthCurve && <span style={{ color: timeSeriesColors.truth }}> · - - true f(ω)</span>}
            <span style={{ color: timeSeriesColors.smoothedSpectrum }}> · — smoothed estimate</span>
            {isSunspots && <span> · sunspots {sunspotStartYear}–{sunspotStartYear + sunspotValues.length - 1}: no ground truth</span>}
          </div>
          <Frame
            width={fullW}
            height={isMobile ? 220 : 260}
            label="Raw periodogram, true spectral density and smoothed estimate"
            xLabel="frequency ω (radians per step)"
            yLabel="spectral density"
            xDomain={[0, Math.PI]}
            yDomain={[specRange.lo, specRange.hi]}
            xTicks={PI_TICKS}
            yTicks={logY ? decadeTicks(specRange.lo, specRange.hi) : linearTicks(0, specRange.hi)}
            logY={logY}
          >
            {(sx, sy) => (
              <>
                {interior.I.map((v, i) => (
                  <circle key={i} cx={sx(interior.omega[i])} cy={sy(v)} r={dotR} fill={timeSeriesColors.raw} opacity={0.75} />
                ))}
                {truthCurve && <path d={linePath(truthCurve.om, truthCurve.f, sx, sy)} fill="none" stroke={timeSeriesColors.truth} strokeWidth={2} strokeDasharray="6 4" />}
                <path d={linePath(smooth.omega, smooth.f, sx, sy)} fill="none" stroke={timeSeriesColors.smoothedSpectrum} strokeWidth={2.2} />
              </>
            )}
          </Frame>

          <div className={`mt-4 ${isMobile ? '' : 'grid grid-cols-2 gap-4'}`}>
            <div className="min-w-0">
              <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                I_n(ω_j)/f(ω_j) over interior frequencies, with the Exp(1) density (Thm 12)
              </div>
              {histogram ? (
                <Frame
                  width={halfW}
                  height={230}
                  label="Histogram of periodogram ordinates divided by the true spectral density, against the exponential density"
                  xLabel="I_n / f"
                  yLabel="density"
                  xDomain={[0, HIST_MAX]}
                  yDomain={[0, 1.15]}
                  xTicks={[0, 1, 2, 3, 4, 5, 6].map((v) => ({ v, label: String(v) }))}
                  yTicks={[0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: String(v) }))}
                >
                  {(sx, sy) => {
                    const w = HIST_MAX / HIST_BINS;
                    const xs = Array.from({ length: 120 }, (_, i) => (HIST_MAX * i) / 119);
                    return (
                      <>
                        {histogram.density.map((d, i) => (
                          <rect key={i} x={sx(i * w) + 0.5} y={sy(Math.min(d, 1.15))} width={Math.max(0.5, sx(w) - sx(0) - 1)} height={sy(0) - sy(Math.min(d, 1.15))} fill={timeSeriesColors.raw} opacity={0.7} />
                        ))}
                        <path d={linePath(xs, xs.map((v) => Math.exp(-v)), sx, sy)} fill="none" stroke={timeSeriesColors.truth} strokeWidth={2} strokeDasharray="6 4" />
                      </>
                    );
                  }}
                </Frame>
              ) : (
                <div className="flex items-center justify-center rounded border text-xs" style={{ height: 230, borderColor: 'var(--color-border)', color: 'var(--color-text-muted)' }}>
                  No ground truth for real data: the sunspot spectrum is unknown.
                </div>
              )}
            </div>

            <div className="mt-4 min-w-0 md:mt-0">
              {isMobile && (
                <button onClick={() => setShowMC((v) => !v)} className="mb-2 text-xs underline" aria-expanded={showMC} style={{ color: 'var(--color-text-muted)' }}>
                  {showMC ? 'Hide bias–variance' : 'Bias–variance'}
                </button>
              )}
              {(!isMobile || showMC) && (
                <>
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    <span>
                      Daniell, {MC_REPS} replications at n = {n}:{' '}
                      <span style={{ color: timeSeriesColors.smoothedSpectrum }}>— relative MSE</span> ·{' '}
                      <span style={{ color: timeSeriesColors.ar }}>- - variance</span> ·{' '}
                      <span style={{ color: timeSeriesColors.ma }}>··· squared bias</span>
                    </span>
                    <button
                      onClick={runMC}
                      disabled={mcDisabled || progress !== null}
                      className="rounded px-2 py-1 text-xs font-medium text-white disabled:opacity-40"
                      style={{ background: timeSeriesColors.smoothedSpectrum }}
                    >
                      Run {MC_REPS} replications
                    </button>
                  </div>
                  {progress !== null && (
                    <div className="mb-2 h-2 w-full overflow-hidden rounded" style={{ background: 'var(--color-surface-alt)' }} role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                      <div className="h-full" style={{ width: `${progress * 100}%`, background: timeSeriesColors.smoothedSpectrum }} />
                    </div>
                  )}
                  {bvPanel || (
                    <div className="flex items-center justify-center rounded border px-3 text-center text-xs" style={{ height: 230, borderColor: 'var(--color-border)', color: 'var(--color-text-muted)' }}>
                      {isSunspots
                        ? 'The Monte Carlo needs a known spectrum; pick a simulated process.'
                        : mcDisabled
                          ? 'Disabled for n > 4096 on small screens.'
                          : mcStale
                            ? 'n or the process changed: run the replications again.'
                            : 'Run the replications to draw Figure 9’s bias–variance curve at this n.'}
                    </div>
                  )}
                  {mc && !mcStale && mStar !== null && (
                    <div className="mt-1 font-mono text-xs">
                      m* = {mStar} (B* = {((2 * Math.PI * (2 * mStar + 1)) / n).toFixed(4)}), relative MSE {mc.mse[mStar].toFixed(4)}; at m = {m}: {mc.mse[m].toFixed(4)}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </>
      ) : (
        <div style={{ height: 560 }} aria-hidden="true" />
      )}
    </div>
  );
}
