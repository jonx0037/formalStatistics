/**
 * ARMARootExplorer — featured interactive for Topic 35 §35.4 (referred back
 * to from §35.5 and §35.7).
 *
 * One picture for Thms 5, 6, 7 and 11: drag the roots of φ(z) (×) and θ(z)
 * (○) around the unit circle and watch causality, invertibility, the ACF and
 * PACF shapes, and the spectrum respond. Hovering the spectrum places the
 * point e^{−iω} on the circle and draws its distance to every root, because
 *   f(ω) = (σ²/2π) Π|r_k − e^{−iω}|²/|r_k|² (MA) / Π|r_k − e^{−iω}|²/|r_k|² (AR)
 * — the spectrum is literally a ratio of distances (Thm 11, Ex 15).
 *
 * Failure states: an AR root on or inside the circle → "Not causal" (Thm 5),
 * theory panels greyed and the path simulated from zero; an MA root inside →
 * "Not invertible" (Thm 6) with a Flip button that reflects r ↦ 1/r̄ and
 * rescales σ² so the autocovariance is unchanged (Rem 11); a shared AR/MA
 * root → parameter redundancy.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import {
  armaACVF,
  armaLongRunVariance,
  armaSpectrum,
  bartlettSE,
  durbinLevinson,
  rootsToAR,
  rootsToMA,
  sampleACF,
  sampleACVF,
  simulateARMA,
  smoothPeriodogram,
  type Complex,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { armaRootPresets, type ARMARootPreset } from '../../data/time-series-data';

const MARGIN = { top: 10, right: 12, bottom: 34, left: 46 };
const MOBILE_BREAKPOINT = 640;
const PLANE_LIM = 2.2;
const SNAP = 0.04;
const MAX_ORDER = 4;
const NLAGS = 30;
const GRID = 512;
const UNIT_TOL = 1e-10;
const PATH_CLIP = 50;

type Side = 'ar' | 'ma';
/** A real root, or a conjugate pair stored by its upper member (im ≥ 0). */
interface RootEntry {
  id: number;
  kind: 'real' | 'pair';
  re: number;
  im: number;
}

const abs = (z: Complex) => Math.hypot(z.re, z.im);
const argDeg = (z: Complex) => (Math.atan2(z.im, z.re) * 180) / Math.PI;
const clampPlane = (v: number) => Math.max(-PLANE_LIM, Math.min(PLANE_LIM, v));

let nextId = 1;
function toEntries(roots: Complex[]): RootEntry[] {
  const out: RootEntry[] = [];
  for (const r of roots) {
    if (r.im === 0) out.push({ id: nextId++, kind: 'real', re: r.re, im: 0 });
    else if (r.im > 0) out.push({ id: nextId++, kind: 'pair', re: r.re, im: r.im });
  }
  return out;
}

function expand(entries: RootEntry[]): Complex[] {
  return entries.flatMap((e) =>
    e.kind === 'real'
      ? [{ re: e.re, im: 0 }]
      : [
          { re: e.re, im: e.im },
          { re: e.re, im: -e.im },
        ],
  );
}

const order = (entries: RootEntry[]) => entries.reduce((s, e) => s + (e.kind === 'real' ? 1 : 2), 0);

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function niceTicks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(3)));
  return out;
}

const PI_TICKS: { v: number; label: string }[] = [
  { v: 0, label: '0' },
  { v: Math.PI / 4, label: 'π/4' },
  { v: Math.PI / 2, label: 'π/2' },
  { v: (3 * Math.PI) / 4, label: '3π/4' },
  { v: Math.PI, label: 'π' },
];

/** Axes frame shared by the lag and spectrum panels. */
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
  muted = false,
  children,
  overlay,
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
  muted?: boolean;
  children: (sx: (v: number) => number, sy: (v: number) => number, plotW: number, plotH: number) => React.ReactNode;
  overlay?: React.ReactNode;
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
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`} opacity={muted ? 0.25 : 1}>
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
          x={-36}
          y={plotH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--color-text)"
          transform={`rotate(-90 ${-36} ${plotH / 2})`}
        >
          {yLabel}
        </text>
      </g>
      {overlay && <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>{overlay}</g>}
    </svg>
  );
}

/** Stem plot of theory (bars) with sample dots and a ± band. */
function LagPanel({
  width,
  height,
  label,
  yLabel,
  theory,
  sample,
  band,
  color,
  muted,
}: {
  width: number;
  height: number;
  label: string;
  yLabel: string;
  theory: number[] | null;
  sample: number[];
  band: number[];
  color: string;
  muted: boolean;
}) {
  const lags = Array.from({ length: NLAGS }, (_, i) => i + 1);
  const vals = [...sample, ...(theory ?? []), ...band, ...band.map((b) => -b)];
  const lo = Math.min(-0.2, ...vals) * 1.08;
  const hi = Math.max(0.2, ...vals) * 1.08;
  const step = hi - lo > 1.2 ? 0.5 : 0.25;
  return (
    <Frame
      width={width}
      height={height}
      label={label}
      xLabel="lag h"
      yLabel={yLabel}
      xDomain={[0, NLAGS + 1]}
      yDomain={[lo, hi]}
      xTicks={niceTicks(0, NLAGS, 10).map((v) => ({ v, label: String(v) }))}
      yTicks={niceTicks(lo, hi, step).map((v) => ({ v, label: String(v) }))}
      muted={muted}
    >
      {(sx, sy, plotW) => (
        <>
          <path
            d={`${linePath(lags, band, sx, sy)}L${sx(NLAGS)},${sy(-band[NLAGS - 1])}${lags
              .slice()
              .reverse()
              .map((h) => `L${sx(h).toFixed(2)},${sy(-band[h - 1]).toFixed(2)}`)
              .join('')}Z`}
            fill={timeSeriesColors.raw}
            opacity={0.18}
          />
          <line x1={0} y1={sy(0)} x2={plotW} y2={sy(0)} stroke="var(--color-text-muted)" />
          {theory &&
            lags.map((h) => (
              <line
                key={h}
                x1={sx(h)}
                x2={sx(h)}
                y1={sy(0)}
                y2={sy(theory[h - 1])}
                stroke={color}
                strokeWidth={Math.max(2, Math.min(5, plotW / NLAGS / 3))}
              />
            ))}
          {lags.map((h) => (
            <circle key={h} cx={sx(h)} cy={sy(sample[h - 1])} r={2.4} fill={timeSeriesColors.truth} opacity={0.75} />
          ))}
        </>
      )}
    </Frame>
  );
}

export default function ARMARootExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const defaultPreset = armaRootPresets.find((p) => p.id === 'ar2peak') ?? armaRootPresets[0];
  const [presetId, setPresetId] = useState(defaultPreset.id);
  const [ar, setAr] = useState<RootEntry[]>(() => toEntries(defaultPreset.arRoots));
  const [ma, setMa] = useState<RootEntry[]>(() => toEntries(defaultPreset.maRoots));
  const [sigma2, setSigma2] = useState(defaultPreset.sigma2);
  const [seed, setSeed] = useState(defaultPreset.seed);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [lagTab, setLagTab] = useState<'acf' | 'pacf'>('acf');
  const [cursor, setCursor] = useState<number | null>(null);
  const [selected, setSelected] = useState<{ side: Side; id: number } | null>(null);
  const drag = useRef<{ side: Side; id: number; mirror: boolean } | null>(null);
  const planeRef = useRef<SVGSVGElement | null>(null);
  const raf = useRef<number | null>(null);
  const holdTimer = useRef<number | null>(null);
  const holding = useRef(false);
  const n = defaultPreset.n;

  useEffect(
    () => () => {
      if (raf.current !== null) cancelAnimationFrame(raf.current);
      if (holdTimer.current !== null) clearTimeout(holdTimer.current);
    },
    [],
  );

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;

  const applyPreset = (p: ARMARootPreset) => {
    setPresetId(p.id);
    setAr(toEntries(p.arRoots));
    setMa(toEntries(p.maRoots));
    setSigma2(p.sigma2);
    setSeed(p.seed);
    setCursor(null);
    setSelected(null);
  };

  const arRoots = useMemo(() => expand(ar), [ar]);
  const maRoots = useMemo(() => expand(ma), [ma]);
  const phi = useMemo(() => rootsToAR(arRoots), [arRoots]);
  const theta = useMemo(() => rootsToMA(maRoots), [maRoots]);
  const causal = arRoots.every((r) => abs(r) > 1 + UNIT_TOL);
  const explosive = arRoots.some((r) => abs(r) < 1 - UNIT_TOL);
  const invertible = maRoots.every((r) => abs(r) > 1 + UNIT_TOL);
  const redundant = arRoots.some((a) => maRoots.some((m) => Math.hypot(a.re - m.re, a.im - m.im) < 0.02));

  const theory = useMemo(() => {
    if (!causal) return null;
    const params = { phi, theta, sigma2 };
    const g = armaACVF(params, NLAGS);
    const acf = g.map((v) => v / g[0]);
    const pacf = durbinLevinson(g).pacf;
    const omegas = Array.from({ length: GRID }, (_, i) => (Math.PI * i) / (GRID - 1));
    const f = armaSpectrum(params, omegas);
    let peak = 0;
    for (let i = 1; i < GRID; i++) if (f[i] > f[peak]) peak = i;
    return { gamma0: g[0], acf, pacf, omegas, f, peak, lrv: armaLongRunVariance(params) };
  }, [causal, phi, theta, sigma2]);

  const path = useMemo(() => {
    const rng = createSeededRng(seed);
    // Non-causal: run the recursion forward from zero with no burn-in so the reader sees it wander or explode.
    const x = simulateARMA({ phi, theta, sigma2 }, n, rng, causal ? 500 : 0);
    const finite = x.every(Number.isFinite);
    const sAcf = finite && causal ? sampleACF(x, NLAGS) : new Array<number>(NLAGS + 1).fill(0);
    const sPacf = finite && causal ? durbinLevinson(sampleACVF(x, NLAGS)).pacf : new Array<number>(NLAGS).fill(0);
    const band = Array.from({ length: NLAGS }, (_, i) => 1.96 * bartlettSE(sAcf, n, i + 1));
    const smooth = finite && causal ? smoothPeriodogram(x, 4) : null;
    return { x, sAcf, sPacf, band, smooth };
  }, [phi, theta, sigma2, seed, n, causal]);

  // ── Root-plane geometry ──────────────────────────────────────────────────
  const planeSize = Math.max(220, Math.min(340, isMobile ? containerW : (containerW - 16) / 2));
  const pScale = planeSize / (2 * PLANE_LIM);
  const px = (re: number) => (re + PLANE_LIM) * pScale;
  const py = (im: number) => (PLANE_LIM - im) * pScale;

  const updateRoot = (side: Side, id: number, re: number, im: number, mirror: boolean) => {
    const setter = side === 'ar' ? setAr : setMa;
    setter((list) =>
      list.map((e) => {
        if (e.id !== id) return e;
        if (e.kind === 'real') return { ...e, re: clampPlane(re) };
        let y = Math.abs(clampPlane(mirror ? -im : im));
        if (y < SNAP) y = 0;
        return { ...e, re: clampPlane(re), im: y };
      }),
    );
    setPresetId('');
  };

  const onPlaneMove = (ev: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    const svg = planeRef.current;
    if (!d || !svg) return;
    const rect = svg.getBoundingClientRect();
    const re = (ev.clientX - rect.left) / pScale - PLANE_LIM;
    const im = PLANE_LIM - (ev.clientY - rect.top) / pScale;
    if (raf.current !== null) cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      raf.current = null;
      updateRoot(d.side, d.id, re, im, d.mirror);
    });
  };

  const startDrag = (ev: React.PointerEvent, side: Side, id: number, mirror: boolean) => {
    ev.preventDefault();
    drag.current = { side, id, mirror };
    setSelected({ side, id });
    try {
      planeRef.current?.setPointerCapture(ev.pointerId);
    } catch {
      /* pointer no longer active; dragging still works while it stays over the plane */
    }
  };

  const endDrag = (ev: React.PointerEvent<SVGSVGElement>) => {
    if (drag.current && planeRef.current?.hasPointerCapture(ev.pointerId)) {
      planeRef.current.releasePointerCapture(ev.pointerId);
    }
    drag.current = null;
  };

  const onRootKey = (ev: React.KeyboardEvent, side: Side, e: RootEntry) => {
    const stepSize = ev.shiftKey ? 0.1 : 0.02;
    let { re, im } = e;
    if (ev.key === 'ArrowLeft') re -= stepSize;
    else if (ev.key === 'ArrowRight') re += stepSize;
    else if (ev.key === 'ArrowUp' && e.kind === 'pair') im += stepSize;
    else if (ev.key === 'ArrowDown' && e.kind === 'pair') im -= stepSize;
    else return;
    ev.preventDefault();
    updateRoot(side, e.id, re, im, false);
  };

  const addRoot = (side: Side, kind: 'real' | 'pair') => {
    const list = side === 'ar' ? ar : ma;
    if (order(list) + (kind === 'real' ? 1 : 2) > MAX_ORDER) return;
    const entry: RootEntry =
      side === 'ar'
        ? kind === 'real'
          ? { id: nextId++, kind, re: 1.6, im: 0 }
          : { id: nextId++, kind, re: 0.7, im: 1.1 }
        : kind === 'real'
          ? { id: nextId++, kind, re: -1.8, im: 0 }
          : { id: nextId++, kind, re: -0.9, im: 1.3 };
    (side === 'ar' ? setAr : setMa)([...list, entry]);
    setPresetId('');
  };

  const removeRoot = (side: Side, id: number) => {
    (side === 'ar' ? setAr : setMa)((list) => list.filter((e) => e.id !== id));
    setPresetId('');
  };

  const flipMA = () => {
    // r ↦ 1/r̄ for every MA root inside the circle; σ² ← σ²/Π|r|² keeps γ unchanged (Rem 11).
    // A pair contributes |r|² twice. Roots on the circle are left alone (no invertible twin exists).
    let factor = 1;
    const flipped = ma.map((e) => {
      const m2 = e.re * e.re + e.im * e.im;
      if (Math.sqrt(m2) >= 1 - UNIT_TOL) return e;
      factor *= e.kind === 'pair' ? m2 * m2 : m2;
      return { ...e, re: e.re / m2, im: e.im / m2 };
    });
    setMa(flipped);
    setSigma2((s) => s / factor);
    setPresetId('');
  };

  // ── Spectrum cursor ──────────────────────────────────────────────────────
  const specW = isMobile ? containerW : Math.max(260, containerW - planeSize - 16);
  const specH = isMobile ? 220 : planeSize;
  const specPlotW = Math.max(10, specW - MARGIN.left - MARGIN.right);
  const omegaFromEvent = (ev: React.PointerEvent<SVGSVGElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ev.clientX - rect.left - MARGIN.left;
    return Math.max(0, Math.min(Math.PI, (x / specPlotW) * Math.PI));
  };
  const onSpecDown = (ev: React.PointerEvent<SVGSVGElement>) => {
    if (!theory) return;
    const w = omegaFromEvent(ev);
    if (ev.pointerType === 'mouse') return;
    const target = ev.currentTarget;
    const pid = ev.pointerId;
    holding.current = false;
    if (holdTimer.current !== null) clearTimeout(holdTimer.current);
    holdTimer.current = window.setTimeout(() => {
      holding.current = true;
      setCursor(w);
      try {
        target.setPointerCapture(pid);
      } catch {
        /* pointer already released */
      }
    }, 250);
  };
  const onSpecMove = (ev: React.PointerEvent<SVGSVGElement>) => {
    if (!theory) return;
    if (ev.pointerType === 'mouse' || holding.current) setCursor(omegaFromEvent(ev));
  };
  const onSpecUp = (ev: React.PointerEvent<SVGSVGElement>) => {
    if (holdTimer.current !== null) clearTimeout(holdTimer.current);
    // A quick tap pins the cursor where it landed.
    if (theory && ev.pointerType !== 'mouse' && !holding.current) setCursor(omegaFromEvent(ev));
    holding.current = false;
  };

  const cursorPoint: Complex | null = cursor === null ? null : { re: Math.cos(cursor), im: -Math.sin(cursor) };
  const distanceReadout = useMemo(() => {
    if (cursor === null || !theory) return null;
    const z = { re: Math.cos(cursor), im: -Math.sin(cursor) };
    const rel = (r: Complex) => Math.hypot(r.re - z.re, r.im - z.im) / abs(r);
    const num = maRoots.reduce((s, r) => s * rel(r) ** 2, 1);
    const den = arRoots.reduce((s, r) => s * rel(r) ** 2, 1);
    const fGeo = (sigma2 / (2 * Math.PI)) * (num / den);
    const fDirect = armaSpectrum({ phi, theta, sigma2 }, [cursor])[0];
    return { num, den, fGeo, fDirect };
  }, [cursor, theory, arRoots, maRoots, sigma2, phi, theta]);

  // ── Spectrum axes ────────────────────────────────────────────────────────
  const specRange = useMemo(() => {
    if (!theory) return { lo: 1e-3, hi: 10 };
    const vals = [...theory.f, ...(showSmoothed && path.smooth ? path.smooth.f : [])].filter((v) => v > 0);
    const hi = Math.max(...vals);
    const lo = Math.max(Math.min(...vals), hi * 1e-5);
    return { lo: 10 ** Math.floor(Math.log10(lo)), hi: 10 ** Math.ceil(Math.log10(hi)) };
  }, [theory, path.smooth, showSmoothed]);
  const decadeTicks = useMemo(() => {
    const out: { v: number; label: string }[] = [];
    const a = Math.round(Math.log10(specRange.lo));
    const b = Math.round(Math.log10(specRange.hi));
    const stride = b - a > 5 ? 2 : 1;
    for (let k = a; k <= b; k += stride) out.push({ v: 10 ** k, label: k === 0 ? '1' : `1e${k}` });
    return out;
  }, [specRange]);

  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-ar)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const smallBtn =
    'rounded px-2 py-1 text-xs bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)] disabled:opacity-40';

  const pAR = order(ar);
  const qMA = order(ma);
  const lagW = isMobile ? containerW : (containerW - 16) / 2;
  const pathMax = Math.max(1, Math.min(PATH_CLIP, Math.max(...path.x.map((v) => (Number.isFinite(v) ? Math.abs(v) : PATH_CLIP)))));
  const clipped = path.x.some((v) => !Number.isFinite(v) || Math.abs(v) > PATH_CLIP);

  const rootHandle = (side: Side, e: RootEntry, idx: number) => {
    const color = side === 'ar' ? timeSeriesColors.ar : timeSeriesColors.ma;
    const isSel = selected?.side === side && selected.id === e.id;
    const label = `${side === 'ar' ? 'AR' : 'MA'} root ${idx + 1}${e.kind === 'pair' && e.im > 0 ? ' (and its conjugate)' : ''}, modulus ${abs(e).toFixed(2)}, angle ${argDeg(e).toFixed(0)}°`;
    const members = e.kind === 'real' ? [{ im: 0, mirror: false }] : [{ im: e.im, mirror: false }, { im: -e.im, mirror: true }];
    return members.map((m, k) => {
      const cx = px(e.re);
      const cy = py(m.im);
      const s = 7;
      return (
        <g
          key={`${side}-${e.id}-${k}`}
          tabIndex={k === 0 ? 0 : -1}
          role="button"
          aria-label={k === 0 ? label : undefined}
          aria-hidden={k === 0 ? undefined : true}
          onFocus={() => setSelected({ side, id: e.id })}
          onKeyDown={(ev) => onRootKey(ev, side, e)}
          onPointerDown={(ev) => startDrag(ev, side, e.id, m.mirror)}
          style={{ cursor: 'grab', outline: 'none' }}
        >
          <circle cx={cx} cy={cy} r={14} fill="transparent" />
          {isSel && <circle cx={cx} cy={cy} r={12} fill="none" stroke={color} strokeDasharray="2 2" />}
          {side === 'ar' ? (
            <path d={`M${cx - s},${cy - s}L${cx + s},${cy + s}M${cx - s},${cy + s}L${cx + s},${cy - s}`} stroke={color} strokeWidth={2.6} />
          ) : (
            <circle cx={cx} cy={cy} r={s} fill="none" stroke={color} strokeWidth={2.6} />
          )}
        </g>
      );
    });
  };

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {armaRootPresets.map((p) => (
          <button
            key={p.id}
            onClick={() => applyPreset(p)}
            className={btn(p.id === presetId)}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <button className={smallBtn} style={{ color: timeSeriesColors.ar }} disabled={pAR + 1 > MAX_ORDER} onClick={() => addRoot('ar', 'real')}>
          + AR real
        </button>
        <button className={smallBtn} style={{ color: timeSeriesColors.ar }} disabled={pAR + 2 > MAX_ORDER} onClick={() => addRoot('ar', 'pair')}>
          + AR pair
        </button>
        <button className={smallBtn} style={{ color: timeSeriesColors.ma }} disabled={qMA + 1 > MAX_ORDER} onClick={() => addRoot('ma', 'real')}>
          + MA real
        </button>
        <button className={smallBtn} style={{ color: timeSeriesColors.ma }} disabled={qMA + 2 > MAX_ORDER} onClick={() => addRoot('ma', 'pair')}>
          + MA pair
        </button>
        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          ARMA({pAR}, {qMA}) · drag a root, or Tab to it and use the arrow keys (Shift for bigger steps)
        </span>
      </div>

      <div className="mb-3 flex flex-wrap gap-2" role="status">
        {!causal && (
          <span className="rounded px-2 py-1 text-xs font-semibold text-white" style={{ background: timeSeriesColors.warn }}>
            Not causal — no stationary causal solution (Theorem 5)
          </span>
        )}
        {!invertible && (
          <span className="flex items-center gap-2 rounded px-2 py-1 text-xs font-semibold text-white" style={{ background: timeSeriesColors.ma }}>
            Not invertible (Theorem 6)
            <button onClick={flipMA} className="rounded bg-white/90 px-2 py-0.5 text-[11px] font-semibold" style={{ color: '#111827' }}>
              Flip
            </button>
          </span>
        )}
        {redundant && (
          <span className="rounded border px-2 py-1 text-xs font-semibold" style={{ borderColor: timeSeriesColors.warn, color: timeSeriesColors.warn }}>
            Parameter redundancy: these roots cancel.
          </span>
        )}
        {causal && invertible && !redundant && (
          <span className="rounded border px-2 py-1 text-xs" style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-muted)' }}>
            Causal and invertible: every root lies outside the unit circle.
          </span>
        )}
      </div>

      {width > 0 ? (
        <>
          <div className={isMobile ? 'flex flex-col gap-4' : 'flex gap-4'}>
            <div className="min-w-0" style={{ width: planeSize }}>
              <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Root plane: <span style={{ color: timeSeriesColors.ar }}>× roots of φ(z)</span> ·{' '}
                <span style={{ color: timeSeriesColors.ma }}>○ roots of θ(z)</span>; shaded disk = not causal / not invertible
              </div>
              <svg
                ref={planeRef}
                width={planeSize}
                height={planeSize}
                className="block"
                role="group"
                aria-label="Complex plane with the unit circle and the AR and MA polynomial roots"
                style={{ touchAction: 'none' }}
                onPointerMove={onPlaneMove}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
              >
                <rect width={planeSize} height={planeSize} fill="var(--color-surface)" opacity={0.4} />
                {[-2, -1, 1, 2].map((v) => (
                  <g key={v}>
                    <line x1={px(v)} x2={px(v)} y1={0} y2={planeSize} stroke="var(--color-viz-grid)" />
                    <line y1={py(v)} y2={py(v)} x1={0} x2={planeSize} stroke="var(--color-viz-grid)" />
                    <text x={px(v)} y={py(0) + 12} fontSize={9} textAnchor="middle" fill="var(--color-text-muted)">
                      {v}
                    </text>
                  </g>
                ))}
                <line x1={0} x2={planeSize} y1={py(0)} y2={py(0)} stroke="var(--color-text-muted)" />
                <line y1={0} y2={planeSize} x1={px(0)} x2={px(0)} stroke="var(--color-text-muted)" />
                <circle cx={px(0)} cy={py(0)} r={pScale} fill={timeSeriesColors.warn} fillOpacity={0.08} stroke={timeSeriesColors.truth} strokeWidth={1.4} />
                <text x={px(0) + 4} y={12} fontSize={9} fill="var(--color-text-muted)">
                  Im z
                </text>
                <text x={planeSize - 4} y={py(0) - 5} fontSize={9} textAnchor="end" fill="var(--color-text-muted)">
                  Re z
                </text>
                {cursorPoint &&
                  [...arRoots.map((r) => ({ r, c: timeSeriesColors.ar })), ...maRoots.map((r) => ({ r, c: timeSeriesColors.ma }))].map(
                    ({ r, c }, i) => {
                      const d = Math.hypot(r.re - cursorPoint.re, r.im - cursorPoint.im);
                      const mx = (px(r.re) + px(cursorPoint.re)) / 2;
                      const my = (py(r.im) + py(cursorPoint.im)) / 2;
                      return (
                        <g key={i}>
                          <line x1={px(cursorPoint.re)} y1={py(cursorPoint.im)} x2={px(r.re)} y2={py(r.im)} stroke={c} strokeDasharray="4 3" strokeWidth={1.2} />
                          <text x={mx + 3} y={my - 3} fontSize={9} fill={c} fontFamily="monospace">
                            {d.toFixed(2)}
                          </text>
                        </g>
                      );
                    },
                  )}
                {cursorPoint && <circle cx={px(cursorPoint.re)} cy={py(cursorPoint.im)} r={4.5} fill={timeSeriesColors.smoothedSpectrum} />}
                {ar.map((e, i) => rootHandle('ar', e, i))}
                {ma.map((e, i) => rootHandle('ma', e, i))}
              </svg>
              <div className="mt-2 flex gap-2 overflow-x-auto pb-1 text-xs">
                {[...ar.map((e, i) => ({ side: 'ar' as Side, e, i })), ...ma.map((e, i) => ({ side: 'ma' as Side, e, i }))].map(({ side, e, i }) => (
                  <span
                    key={`${side}-${e.id}`}
                    className="flex shrink-0 items-center gap-1 rounded border px-2 py-0.5 font-mono"
                    style={{ borderColor: side === 'ar' ? timeSeriesColors.ar : timeSeriesColors.ma }}
                  >
                    {side === 'ar' ? '×' : '○'} {abs(e).toFixed(3)}∠{argDeg(e).toFixed(0)}°{e.kind === 'pair' ? ' (±)' : ''}
                    <button aria-label={`Remove ${side.toUpperCase()} root ${i + 1}`} onClick={() => removeRoot(side, e.id)} className="ml-1 opacity-70 hover:opacity-100">
                      ✕
                    </button>
                  </span>
                ))}
                {ar.length + ma.length === 0 && <span style={{ color: 'var(--color-text-muted)' }}>No roots: white noise.</span>}
              </div>
            </div>

            <div className="min-w-0 flex-1">
              <div className="mb-1 flex flex-wrap items-center gap-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                <span>
                  Spectrum f(ω), log scale: <span style={{ color: timeSeriesColors.truth }}>- - theory</span>
                  {showSmoothed && <span style={{ color: timeSeriesColors.smoothedSpectrum }}> · — smoothed periodogram (m = 4)</span>}
                </span>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showSmoothed} onChange={(e) => setShowSmoothed(e.target.checked)} />
                  overlay
                </label>
              </div>
              <div className="relative">
                <svg
                  width={specW}
                  height={specH}
                  className="block"
                  style={{ touchAction: 'pan-y', position: 'absolute', inset: 0, zIndex: 1 }}
                  onPointerDown={onSpecDown}
                  onPointerMove={onSpecMove}
                  onPointerUp={onSpecUp}
                  onPointerCancel={() => {
                    if (holdTimer.current !== null) clearTimeout(holdTimer.current);
                    holding.current = false;
                  }}
                  onPointerLeave={(ev) => {
                    if (ev.pointerType === 'mouse') setCursor(null);
                  }}
                  aria-hidden="true"
                />
                <Frame
                  width={specW}
                  height={specH}
                  label="Theoretical ARMA spectral density on a log scale, with the smoothed periodogram of the sample path"
                  xLabel="frequency ω (radians per step)"
                  yLabel="f(ω)"
                  xDomain={[0, Math.PI]}
                  yDomain={[specRange.lo, specRange.hi]}
                  xTicks={PI_TICKS}
                  yTicks={decadeTicks}
                  logY
                  muted={!theory}
                  overlay={
                    !theory ? (
                      <text x={(specW - MARGIN.left - MARGIN.right) / 2} y={(specH - MARGIN.top - MARGIN.bottom) / 2} textAnchor="middle" fontSize={12} fill={timeSeriesColors.warn}>
                        No spectral density: the process is not causal stationary.
                      </text>
                    ) : null
                  }
                >
                  {(sx, sy, _w, plotH) =>
                    theory ? (
                      <>
                        {showSmoothed && path.smooth && (
                          <path d={linePath(path.smooth.omega, path.smooth.f, sx, sy)} fill="none" stroke={timeSeriesColors.smoothedSpectrum} strokeWidth={1.4} />
                        )}
                        <path d={linePath(theory.omegas, theory.f, sx, sy)} fill="none" stroke={timeSeriesColors.truth} strokeWidth={2} strokeDasharray="6 4" />
                        {cursor !== null && (
                          <>
                            <line x1={sx(cursor)} x2={sx(cursor)} y1={0} y2={plotH} stroke={timeSeriesColors.smoothedSpectrum} />
                            <circle cx={sx(cursor)} cy={sy(armaSpectrum({ phi, theta, sigma2 }, [cursor])[0])} r={4} fill={timeSeriesColors.smoothedSpectrum} />
                          </>
                        )}
                      </>
                    ) : null
                  }
                </Frame>
              </div>
              <div className="mt-1 min-h-[2.5rem] font-mono text-[11px] leading-snug" aria-live="polite">
                {distanceReadout ? (
                  <>
                    ω = {cursor!.toFixed(3)}: (σ²/2π)·Π(MA dist/|r|)² / Π(AR dist/|r|)² = ({sigma2.toFixed(3)}/2π)·
                    {distanceReadout.num.toPrecision(4)}/{distanceReadout.den.toPrecision(4)} = {distanceReadout.fGeo.toPrecision(5)}
                    <br />
                    f(ω) from Theorem 11 = {distanceReadout.fDirect.toPrecision(5)}
                  </>
                ) : (
                  <span style={{ color: 'var(--color-text-muted)' }}>
                    {theory ? (
                      isMobile ? (
                        <>
                          Tap, or touch and hold then slide, on the spectrum to draw the distances from e<sup>−iω</sup> to every root.
                        </>
                      ) : (
                        <>
                          Hover over the spectrum to place e<sup>−iω</sup> on the unit circle and draw its distance to every root.
                        </>
                      )
                    ) : (
                      'Spectrum unavailable until every AR root is outside the unit circle.'
                    )}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="mt-4">
            {isMobile && (
              <div className="mb-2 flex gap-2" role="tablist">
                {(['acf', 'pacf'] as const).map((t) => (
                  <button key={t} role="tab" aria-selected={lagTab === t} className={btn(lagTab === t)} onClick={() => setLagTab(t)}>
                    {t.toUpperCase()}
                  </button>
                ))}
              </div>
            )}
            <div className={isMobile ? '' : 'grid grid-cols-2 gap-4'}>
              {(!isMobile || lagTab === 'acf') && (
                <div className="min-w-0">
                  <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    ACF: <span style={{ color: timeSeriesColors.ma }}>stems = theory</span> · dots = sample (n = {n}) · band = ±1.96 Bartlett SE
                  </div>
                  <LagPanel
                    width={lagW}
                    height={200}
                    label="Theoretical and sample autocorrelation function"
                    yLabel="ρ(h)"
                    theory={theory ? theory.acf.slice(1) : null}
                    sample={path.sAcf.slice(1)}
                    band={path.band}
                    color={timeSeriesColors.ma}
                    muted={!theory}
                  />
                </div>
              )}
              {(!isMobile || lagTab === 'pacf') && (
                <div className="min-w-0">
                  <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    PACF: <span style={{ color: timeSeriesColors.ar }}>stems = theory</span> · dots = sample · band = ±1.96/√n
                  </div>
                  <LagPanel
                    width={lagW}
                    height={200}
                    label="Theoretical and sample partial autocorrelation function"
                    yLabel="α(h)"
                    theory={theory ? theory.pacf : null}
                    sample={path.sPacf}
                    band={new Array<number>(NLAGS).fill(1.96 / Math.sqrt(n))}
                    color={timeSeriesColors.ar}
                    muted={!theory}
                  />
                </div>
              )}
            </div>
          </div>

          <div className="mt-4">
            <div className="mb-1 flex flex-wrap items-center gap-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              <span>
                Sample path, n = {n}
                {!causal && (
                  <strong style={{ color: timeSeriesColors.warn }}>
                    {' '}
                    — {explosive ? 'explosive' : 'unit root'}: simulated from zero
                    {clipped ? `, clipped at ±${PATH_CLIP}` : ''}
                  </strong>
                )}
              </span>
              <label className="flex items-center gap-1">
                seed
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
                  className="w-20 rounded border px-1 py-0.5 font-mono"
                  style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
                />
              </label>
              <button onClick={() => setSeed((s) => s + 1)} className="rounded px-3 py-1 text-xs font-medium text-white" style={{ background: timeSeriesColors.ar }}>
                Resample
              </button>
            </div>
            <Frame
              width={containerW}
              height={isMobile ? 160 : 180}
              label="Simulated sample path"
              xLabel="time t"
              yLabel="x_t"
              xDomain={[0, n]}
              yDomain={[-pathMax * 1.05, pathMax * 1.05]}
              xTicks={niceTicks(0, n, 50).map((v) => ({ v, label: String(v) }))}
              yTicks={niceTicks(-pathMax, pathMax, pathMax > 20 ? 20 : pathMax > 6 ? 4 : 2).map((v) => ({ v, label: String(v) }))}
            >
              {(sx, sy) => {
                const ys = path.x.map((v) => (Number.isFinite(v) ? Math.max(-PATH_CLIP, Math.min(PATH_CLIP, v)) : PATH_CLIP));
                return <path d={linePath(ys.map((_, i) => i), ys, sx, sy)} fill="none" stroke={causal ? timeSeriesColors.neutral : timeSeriesColors.warn} strokeWidth={0.9} />;
              }}
            </Frame>
          </div>

          <div className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 font-mono text-xs sm:grid-cols-2">
            <span style={{ color: timeSeriesColors.ar }}>φ = ({phi.length ? phi.map((v) => v.toFixed(4)).join(', ') : '—'})</span>
            <span style={{ color: timeSeriesColors.ma }}>θ = ({theta.length ? theta.map((v) => v.toFixed(4)).join(', ') : '—'})</span>
            <span>σ² = {sigma2.toFixed(4)}</span>
            <span>γ(0) = {theory ? theory.gamma0.toFixed(4) : '—'}</span>
            <span>long-run variance σ²θ(1)²/φ(1)² = {theory ? theory.lrv.toFixed(4) : '—'}</span>
            <span>
              spectral peak ω* = {theory ? theory.omegas[theory.peak].toFixed(4) : '—'}
              {theory && theory.peak > 0 ? `, period 2π/ω* = ${((2 * Math.PI) / theory.omegas[theory.peak]).toFixed(2)} steps` : theory ? ' (peak at frequency 0)' : ''}
            </span>
          </div>
        </>
      ) : (
        <div style={{ height: 900 }} aria-hidden="true" />
      )}
    </div>
  );
}
