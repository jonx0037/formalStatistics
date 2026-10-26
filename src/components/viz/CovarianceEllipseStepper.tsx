/**
 * CovarianceEllipseStepper — interactive for Topic 36 §36.2 (Ex 3), revisited
 * in §36.9 (Ex 15, Def 6, Thm 10).
 *
 * The geometric core of the Kalman filter. Predict = shear the belief ellipse
 * by F, then inflate it by Q; update = slice it along the direction the
 * observation sees. Observing position also shrinks velocity's uncertainty,
 * because the predict step correlated the two. Observe velocity only and
 * position is unobservable (Def 6): its variance grows without bound and the
 * Riccati recursion has no steady state (Thm 10 needs detectability). With
 * position observed, the gain settles to K∞ within about twenty steps from
 * any start (Ex 15).
 */
import { useEffect, useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import {
  constantVelocity,
  isObservable,
  kalmanPredict,
  kalmanUpdate,
  matMul,
  simulateSSM,
  steadyStateGain,
  transpose,
  type Matrix,
  type SSModel,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { ellipseStepperPresets } from '../../data/time-series-data';
import { trackingFixture } from '../../data/ssm-fixtures';

const MARGIN = { top: 10, right: 12, bottom: 34, left: 46 };
const MOBILE_BREAKPOINT = 640;
const T_MAX = 60;
const STEADY_TOL = 1e-9;
const M1 = [0, 1];
const P1: Matrix = [
  [0.5, 0],
  [0, 0.5],
];

type Observe = 'position' | 'velocity' | 'both';

interface Moments {
  m: number[];
  P: Matrix;
}

interface FilterState {
  /** Number of completed updates. */
  t: number;
  phase: 'filtered' | 'predicted';
  cur: Moments;
  prev: Moments | null;
  pred: (Moments & { shear: Matrix | null }) | null;
  obs: number[] | null;
  S: Matrix | null;
  K: Matrix | null;
  hist: { m: number[]; P: Matrix; K: Matrix }[];
}

const initialState = (): FilterState => ({
  t: 0,
  phase: 'filtered',
  cur: { m: M1.slice(), P: P1.map((r) => r.slice()) },
  prev: null,
  pred: null,
  obs: null,
  S: null,
  K: null,
  hist: [],
});

function predictStep(s: FilterState, model: SSModel): FilterState {
  if (s.phase !== 'filtered' || s.t >= T_MAX) return s;
  // t = 1: the prior (m1, P1) is the prediction (Topic 34 Thm 12 indexing); afterwards F shears and Q inflates.
  if (s.t === 0) return { ...s, phase: 'predicted', prev: null, pred: { m: s.cur.m, P: s.cur.P, shear: null } };
  const shear = matMul(matMul(model.F, s.cur.P), transpose(model.F));
  const pr = kalmanPredict(s.cur.m, s.cur.P, model);
  return { ...s, phase: 'predicted', prev: s.cur, pred: { ...pr, shear } };
}

function updateStep(s: FilterState, model: SSModel, data: { pos: number[]; vel: number[] }, observe: Observe): FilterState {
  if (s.phase !== 'predicted' || !s.pred) return s;
  const i = s.t;
  const obs = observe === 'position' ? [data.pos[i]] : observe === 'velocity' ? [data.vel[i]] : [data.pos[i], data.vel[i]];
  const up = kalmanUpdate(s.pred.m, s.pred.P, obs, model);
  return {
    ...s,
    t: i + 1,
    phase: 'filtered',
    cur: { m: up.m, P: up.P },
    obs,
    S: up.S,
    K: up.K,
    hist: [...s.hist, { m: up.m, P: up.P, K: up.K }],
  };
}

/** 2-sd ellipse of a 2×2 covariance: semi-axes 2√λ, angle ½ atan2(2P12, P11 − P22). */
function ellipseGeom(P: Matrix) {
  const a = P[0][0];
  const b = P[0][1];
  const c = P[1][1];
  const mid = 0.5 * (a + c);
  const rad = Math.sqrt(0.25 * (a - c) * (a - c) + b * b);
  const l1 = Math.max(mid + rad, 0);
  const l2 = Math.max(mid - rad, 0);
  return { r1: 2 * Math.sqrt(l1), r2: 2 * Math.sqrt(l2), angle: 0.5 * Math.atan2(2 * b, a - c) };
}

const sqrtDet = (P: Matrix) => Math.sqrt(Math.max(P[0][0] * P[1][1] - P[0][1] * P[1][0], 0));
const corr = (P: Matrix) => P[0][1] / Math.sqrt(P[0][0] * P[1][1]);
const f4 = (v: number) => v.toFixed(4);
const fmtM = (M: Matrix) => `[${M.map((r) => r.map(f4).join(', ')).join('; ')}]`;

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

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function Frame({
  width,
  height,
  label,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  children,
}: {
  width: number;
  height: number;
  label: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  children: (sx: (v: number) => number, sy: (v: number) => number, plotW: number, plotH: number) => React.ReactNode;
}) {
  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom);
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = (v: number) => plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const axisColor = 'var(--color-text-muted)';
  const xs = niceStep(xDomain[1] - xDomain[0], 5);
  const ys = niceStep(yDomain[1] - yDomain[0], 5);
  const clipId = `ces-${label.length}-${Math.round(width)}-${Math.round(height)}`;
  return (
    <svg width={width} height={height} className="block" role="img" aria-label={label}>
      <defs>
        <clipPath id={clipId}>
          <rect x={0} y={0} width={plotW} height={plotH} />
        </clipPath>
      </defs>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {niceTicks(yDomain[0], yDomain[1], ys).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
            <text x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
              {t}
            </text>
          </g>
        ))}
        {niceTicks(xDomain[0], xDomain[1], xs).map((t) => (
          <g key={`x${t}`}>
            <line x1={sx(t)} y1={plotH} x2={sx(t)} y2={plotH + 4} stroke={axisColor} />
            <text x={sx(t)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
              {t}
            </text>
          </g>
        ))}
        <g clipPath={`url(#${clipId})`}>{children(sx, sy, plotW, plotH)}</g>
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
        <text x={plotW / 2} y={plotH + 30} textAnchor="middle" fontSize={11} fill="var(--color-text)">
          {xLabel}
        </text>
        <text x={-32} y={plotH / 2} textAnchor="middle" fontSize={11} fill="var(--color-text)" transform={`rotate(-90 ${-32} ${plotH / 2})`}>
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

function LogSlider({
  label,
  value,
  min,
  max,
  step,
  log,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  log: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-center justify-between text-xs">
        <span>{label}</span>
        <span className="font-mono">{log ? (10 ** value).toPrecision(3) : value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--color-ts-predicted)]"
      />
    </label>
  );
}

export default function CovarianceEllipseStepper() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const def = ellipseStepperPresets[0];
  const [logQa, setLogQa] = useState(Math.log10(def.qa));
  const [logR, setLogR] = useState(Math.log10(def.r));
  const [dt, setDt] = useState(def.dt);
  const [observe, setObserve] = useState<Observe>(def.observe);
  const [source, setSource] = useState<'fixture' | 'simulate'>('fixture');
  const [seed, setSeed] = useState(def.seed);
  const [simKey, setSimKey] = useState({ seed: def.seed, qa: def.qa, r: def.r, dt: def.dt });
  const [st, setSt] = useState<FilterState>(initialState);
  const [running, setRunning] = useState(false);
  const [steadyMsg, setSteadyMsg] = useState<string | null>(null);

  const qa = 10 ** logQa;
  const r = 10 ** logR;
  const model = useMemo(() => constantVelocity(dt, qa, r, observe), [dt, qa, r, observe]);
  const observable = useMemo(() => isObservable(model.F, model.H), [model]);
  const steady = useMemo(() => (observable ? steadyStateGain(model.F, model.H, model.Q, model.R) : null), [model, observable]);

  // Observations: the fixture's 60 positions (simulated with r = 0.4), with velocity readings drawn
  // from its truth by the TS RNG; or a fresh simulation of both coordinates at the current sliders.
  const data = useMemo(() => {
    if (source === 'fixture') {
      const rng = createSeededRng(simKey.seed);
      const sdR = Math.sqrt(simKey.r);
      return {
        truth: trackingFixture.truth,
        pos: trackingFixture.positions,
        vel: trackingFixture.truth.map((z) => z[1] + sdR * rng.normal()),
      };
    }
    const sim = simulateSSM(constantVelocity(simKey.dt, simKey.qa, simKey.r, 'both'), T_MAX, createSeededRng(simKey.seed));
    return { truth: sim.z, pos: sim.x.map((v) => v[0]), vel: sim.x.map((v) => v[1]) };
  }, [source, simKey]);

  const reset = () => {
    setRunning(false);
    setSt(initialState());
    setSteadyMsg(null);
  };
  // A new model or new data restarts the filter from (m1, P1).
  useEffect(reset, [model, data]);

  const step = (s: FilterState) => updateStep(predictStep(s, model), model, data, observe);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      setSt((s) => {
        if (s.t >= T_MAX) {
          setRunning(false);
          return s;
        }
        return s.phase === 'predicted' ? updateStep(s, model, data, observe) : step(s);
      });
    }, 500);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, model, data, observe]);

  const runToSteady = () => {
    setRunning(false);
    if (!steady) {
      let s = st.phase === 'predicted' ? updateStep(st, model, data, observe) : st;
      while (s.t < T_MAX) s = step(s);
      setSt(s);
      setSteadyMsg(`No steady state: after ${s.t} steps the position variance is ${s.cur.P[0][0].toFixed(1)} and still growing.`);
      return;
    }
    let s = st.phase === 'predicted' ? updateStep(st, model, data, observe) : st;
    const dist = (K: Matrix) => Math.max(...K.flatMap((row, i) => row.map((v, j) => Math.abs(v - steady.K[i][j]))));
    while (s.t < T_MAX && !(s.K && dist(s.K) < STEADY_TOL)) s = step(s);
    setSt(s);
    setSteadyMsg(
      s.K && dist(s.K) < STEADY_TOL
        ? `‖K_t − K∞‖ < 10⁻⁹ after ${s.t} steps (K∞ from the Riccati iteration, ${steady.iterations} iterations).`
        : `Not yet within 10⁻⁹ after ${s.t} steps.`,
    );
  };

  // ── Layout ────────────────────────────────────────────────────────────────
  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const fullW = Math.max(280, containerW);
  const phaseW = isMobile ? Math.min(fullW, 340) : Math.min(380, Math.floor(fullW * 0.46));
  const rightW = isMobile ? fullW : Math.max(160, fullW - phaseW - 16);

  // Phase-plane window: square, centered on the current mean, sized to the largest ellipse on show.
  const shown = [st.cur.P, st.prev?.P, st.pred?.P, st.pred?.shear].filter(Boolean) as Matrix[];
  const center = st.phase === 'predicted' && st.pred ? st.pred.m : st.cur.m;
  const reach = Math.max(1.2, ...shown.map((P) => 1.25 * ellipseGeom(P).r1));
  const half = Math.min(reach, 40);
  const xDom: [number, number] = [center[0] - half, center[0] + half];
  const yDom: [number, number] = [center[1] - half, center[1] + half];

  const tNow = st.t;
  const trueNow = tNow > 0 ? data.truth[tNow - 1] : null;
  const nextObsIdx = st.phase === 'predicted' ? st.t : st.t - 1;

  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-predicted)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const border = { borderWidth: 1, borderColor: 'var(--color-border)' };

  const drawEllipse = (m: number[], P: Matrix, sx: (v: number) => number, sy: (v: number) => number, props: React.SVGProps<SVGEllipseElement>) => {
    const g = ellipseGeom(P);
    const k = sx(1) - sx(0);
    const cx = sx(m[0]);
    const cy = sy(m[1]);
    return <ellipse cx={cx} cy={cy} rx={Math.max(0.5, g.r1 * k)} ry={Math.max(0.5, g.r2 * k)} transform={`rotate(${(-g.angle * 180) / Math.PI} ${cx} ${cy})`} {...props} />;
  };

  const histT = st.hist.map((_, i) => (i + 1) * dt);
  const posY = useMemo(() => {
    const vals = [...data.pos.slice(0, Math.max(tNow, 1)), ...data.truth.slice(0, Math.max(tNow, 1)).map((z) => z[0])];
    st.hist.forEach((h) => vals.push(h.m[0] - 2 * Math.sqrt(h.P[0][0]), h.m[0] + 2 * Math.sqrt(h.P[0][0])));
    const lo = Math.min(...vals, -1);
    const hi = Math.max(...vals, 1);
    const pad = 0.08 * (hi - lo);
    return [lo - pad, hi + pad] as [number, number];
  }, [data, st.hist, tNow]);
  const gainSeries = useMemo(() => {
    if (!st.hist.length) return [] as { label: string; ys: number[]; inf: number | null }[];
    const K0 = st.hist[0].K;
    const out: { label: string; ys: number[]; inf: number | null }[] = [];
    for (let i = 0; i < K0.length; i++)
      for (let j = 0; j < K0[0].length; j++)
        out.push({
          label: K0[0].length > 1 ? `K${i + 1}${j + 1}` : `K${i + 1}`,
          ys: st.hist.map((h) => h.K[i][j]),
          inf: steady ? steady.K[i][j] : null,
        });
    return out;
  }, [st.hist, steady]);
  const gainDom = useMemo<[number, number]>(() => {
    const v = gainSeries.flatMap((g) => [...g.ys, ...(g.inf !== null ? [g.inf] : [])]);
    if (!v.length) return [0, 1];
    const lo = Math.min(0, ...v);
    const hi = Math.max(...v, 0.1);
    return [lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo)];
  }, [gainSeries]);
  const gainColors = [timeSeriesColors.filtered, timeSeriesColors.smoothed, timeSeriesColors.forecast, timeSeriesColors.predicted];

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Observe:</span>
        {(['position', 'velocity', 'both'] as const).map((o) => (
          <button key={o} onClick={() => setObserve(o)} className={btn(observe === o)} style={border} aria-pressed={observe === o}>
            {o === 'both' ? 'both (H = I)' : o}
          </button>
        ))}
        <span className="ml-2 font-semibold">Data:</span>
        <button onClick={() => setSource('fixture')} className={btn(source === 'fixture')} style={border} aria-pressed={source === 'fixture'}>
          tracking fixture
        </button>
        <button
          onClick={() => {
            setSource('simulate');
            setSimKey({ seed, qa, r, dt });
          }}
          className={btn(source === 'simulate')}
          style={border}
          aria-pressed={source === 'simulate'}
        >
          Simulate
        </button>
        <label className="flex items-center gap-1 text-xs">
          seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
            className="w-20 rounded border px-1 py-0.5 font-mono"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
          />
        </label>
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-3">
        <LogSlider label="acceleration noise qₐ (log scale)" value={logQa} min={-2} max={0.5} step={0.01} log onChange={setLogQa} />
        <LogSlider label="observation variance r (log scale)" value={logR} min={-1.5} max={1} step={0.01} log onChange={setLogR} />
        <LogSlider label="time step Δ" value={dt} min={0.25} max={2} step={0.05} log={false} onChange={setDt} />
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <button onClick={() => setSt((s) => predictStep(s, model))} disabled={st.phase !== 'filtered' || st.t >= T_MAX} className={btn(false)} style={border}>
          Predict
        </button>
        <button onClick={() => setSt((s) => updateStep(s, model, data, observe))} disabled={st.phase !== 'predicted'} className={btn(false)} style={border}>
          Update
        </button>
        <button onClick={() => setSt((s) => (s.phase === 'predicted' ? updateStep(s, model, data, observe) : step(s)))} disabled={st.t >= T_MAX} className={btn(false)} style={border}>
          Step
        </button>
        <button onClick={() => setRunning((v) => !v)} disabled={st.t >= T_MAX} className={btn(running)} style={border} aria-pressed={running}>
          {running ? 'Pause' : 'Run'}
        </button>
        <button onClick={runToSteady} className={btn(false)} style={border}>
          Run to steady state
        </button>
        <button onClick={reset} className={btn(false)} style={border}>
          Reset
        </button>
        <span
          className="rounded px-2 py-0.5 font-medium"
          style={{
            border: `1px solid ${observable ? timeSeriesColors.smoothed : timeSeriesColors.warn}`,
            color: observable ? timeSeriesColors.smoothed : timeSeriesColors.warn,
          }}
        >
          {observable ? '(F, H) observable' : 'position not observable — its variance grows without bound'}
        </span>
      </div>

      <div className="mb-2 grid grid-cols-1 gap-x-4 gap-y-0.5 font-mono text-xs sm:grid-cols-2">
        <span>
          t = {st.t} · {st.phase === 'predicted' ? 'predicted, awaiting x_t' : st.t === 0 ? 'prior (m₁, P₁)' : 'filtered'}
        </span>
        <span>
          √det: {st.prev ? f4(sqrtDet(st.prev.P)) : '—'} → {st.pred ? f4(sqrtDet(st.pred.P)) : '—'} → {st.phase === 'filtered' && st.t > 0 ? f4(sqrtDet(st.cur.P)) : '—'}
        </span>
        <span>P⁻_t = {st.pred ? fmtM(st.pred.P) : '—'}</span>
        <span>
          corr: {st.prev ? corr(st.prev.P).toFixed(3) : '—'} → {st.pred ? corr(st.pred.P).toFixed(3) : '—'} →{' '}
          {st.phase === 'filtered' && st.t > 0 ? corr(st.cur.P).toFixed(3) : '—'}
        </span>
        <span>S_t = {st.S && st.phase === 'filtered' ? fmtM(st.S) : '—'}</span>
        <span>K_t = {st.K && st.phase === 'filtered' ? fmtM(st.K) : '—'}</span>
        <span>P_t = {st.phase === 'filtered' && st.t > 0 ? fmtM(st.cur.P) : '—'}</span>
        <span>K∞ = {steady ? fmtM(steady.K) : 'none (not detectable)'}</span>
      </div>
      {steadyMsg && (
        <div className="mb-2 text-xs" style={{ color: 'var(--color-text-muted)' }} aria-live="polite">
          {steadyMsg}
        </div>
      )}

      {width > 0 ? (
        <div className={isMobile ? 'flex flex-col gap-3' : 'flex gap-4'}>
          <div className="min-w-0">
            <Frame
              width={phaseW}
              height={phaseW}
              label="Phase plane of position and velocity with the previous, sheared, predicted and updated covariance ellipses"
              xLabel="position"
              yLabel="velocity"
              xDomain={xDom}
              yDomain={yDom}
            >
              {(sx, sy, plotW, plotH) => {
                const obsIdx = nextObsIdx;
                const sdR = 2 * Math.sqrt(r);
                const k = sx(1) - sx(0);
                const strip =
                  st.t > 0 || st.phase === 'predicted'
                    ? observe === 'position'
                      ? <rect x={sx(data.pos[obsIdx] - sdR)} y={0} width={Math.max(0, 2 * sdR * k)} height={plotH} fill={timeSeriesColors.forecast} opacity={st.phase === 'predicted' ? 0.08 : 0.14} />
                      : observe === 'velocity'
                        ? <rect x={0} y={sy(data.vel[obsIdx] + sdR)} width={plotW} height={Math.max(0, 2 * sdR * k)} fill={timeSeriesColors.forecast} opacity={st.phase === 'predicted' ? 0.08 : 0.14} />
                        : <circle cx={sx(data.pos[obsIdx])} cy={sy(data.vel[obsIdx])} r={Math.max(0.5, sdR * k)} fill={timeSeriesColors.forecast} opacity={st.phase === 'predicted' ? 0.08 : 0.14} />
                    : null;
                return (
                  <>
                    {obsIdx >= 0 && strip}
                    {st.prev && drawEllipse(st.prev.m, st.prev.P, sx, sy, { fill: 'none', stroke: timeSeriesColors.filtered, strokeWidth: 1.2, opacity: 0.35 })}
                    {st.pred?.shear &&
                      drawEllipse(st.pred.m, st.pred.shear, sx, sy, { fill: 'none', stroke: timeSeriesColors.predicted, strokeWidth: 1.2, strokeDasharray: '5 4' })}
                    {st.pred &&
                      drawEllipse(st.pred.m, st.pred.P, sx, sy, {
                        fill: timeSeriesColors.predicted,
                        fillOpacity: st.phase === 'predicted' ? 0.12 : 0.04,
                        stroke: timeSeriesColors.predicted,
                        strokeWidth: st.phase === 'predicted' ? 2 : 1,
                        opacity: st.phase === 'predicted' ? 1 : 0.5,
                      })}
                    {st.prev && st.pred && (
                      <line x1={sx(st.prev.m[0])} y1={sy(st.prev.m[1])} x2={sx(st.pred.m[0])} y2={sy(st.pred.m[1])} stroke="var(--color-text-muted)" strokeWidth={1} />
                    )}
                    {(st.phase === 'filtered' || st.t === 0) &&
                      drawEllipse(st.cur.m, st.cur.P, sx, sy, { fill: timeSeriesColors.filtered, fillOpacity: 0.12, stroke: timeSeriesColors.filtered, strokeWidth: 2 })}
                    <circle cx={sx(center[0])} cy={sy(center[1])} r={3} fill={st.phase === 'predicted' ? timeSeriesColors.predicted : timeSeriesColors.filtered} />
                    {trueNow && st.phase === 'filtered' && (
                      <g stroke={timeSeriesColors.truth} strokeWidth={1.8}>
                        <line x1={sx(trueNow[0]) - 5} y1={sy(trueNow[1]) - 5} x2={sx(trueNow[0]) + 5} y2={sy(trueNow[1]) + 5} />
                        <line x1={sx(trueNow[0]) - 5} y1={sy(trueNow[1]) + 5} x2={sx(trueNow[0]) + 5} y2={sy(trueNow[1]) - 5} />
                      </g>
                    )}
                  </>
                );
              }}
            </Frame>
            <div className="mt-1 flex flex-wrap gap-x-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              <span style={{ color: timeSeriesColors.filtered }}>◯ P_t−1 (faded) · P_t</span>
              <span style={{ color: timeSeriesColors.predicted }}>┄ FPFᵀ · ◯ P⁻_t</span>
              <span style={{ color: timeSeriesColors.forecast }}>▮ x_t ± 2√r</span>
              <span>× true state</span>
            </div>
          </div>

          <div className="flex min-w-0 flex-col gap-3">
            <Frame
              width={rightW}
              height={isMobile ? 170 : 190}
              label="Position against time: truth, observations, and the filtered mean with a two standard deviation band"
              xLabel="time"
              yLabel="position"
              xDomain={[0, T_MAX * dt]}
              yDomain={posY}
            >
              {(sx, sy) => {
                const n = Math.max(tNow, 0);
                const ts = data.truth.slice(0, n).map((_, i) => (i + 1) * dt);
                const up = st.hist.map((h) => h.m[0] + 2 * Math.sqrt(h.P[0][0]));
                const lo = st.hist.map((h) => h.m[0] - 2 * Math.sqrt(h.P[0][0]));
                const band =
                  histT.length > 1
                    ? `${linePath(histT, up, sx, sy)}${histT
                        .map((_, k) => histT.length - 1 - k)
                        .map((i) => `L${sx(histT[i]).toFixed(2)},${sy(lo[i]).toFixed(2)}`)
                        .join('')}Z`
                    : '';
                return (
                  <>
                    {band && <path d={band} fill={timeSeriesColors.filtered} opacity={0.15} />}
                    {n > 1 && <path d={linePath(ts, data.truth.slice(0, n).map((z) => z[0]), sx, sy)} fill="none" stroke={timeSeriesColors.truth} strokeWidth={1} strokeDasharray="5 3" />}
                    {observe !== 'velocity' && ts.map((t, i) => <circle key={i} cx={sx(t)} cy={sy(data.pos[i])} r={2.2} fill={timeSeriesColors.raw} />)}
                    {histT.length > 1 && <path d={linePath(histT, st.hist.map((h) => h.m[0]), sx, sy)} fill="none" stroke={timeSeriesColors.filtered} strokeWidth={1.6} />}
                  </>
                );
              }}
            </Frame>
            <Frame
              width={rightW}
              height={isMobile ? 170 : 190}
              label="Kalman gain components against time with dashed steady-state values"
              xLabel="time"
              yLabel="gain"
              xDomain={[0, T_MAX * dt]}
              yDomain={gainDom}
            >
              {(sx, sy, plotW) => (
                <>
                  {gainSeries.map((g, i) => (
                    <g key={g.label}>
                      {g.inf !== null && <line x1={0} x2={plotW} y1={sy(g.inf)} y2={sy(g.inf)} stroke={gainColors[i % 4]} strokeDasharray="5 4" opacity={0.7} />}
                      <path d={linePath(histT, g.ys, sx, sy)} fill="none" stroke={gainColors[i % 4]} strokeWidth={1.6} />
                      {g.ys.length > 0 && (
                        <text x={sx(histT[histT.length - 1]) + 4} y={sy(g.ys[g.ys.length - 1]) + 3} fontSize={10} fill={gainColors[i % 4]}>
                          {g.label}
                        </text>
                      )}
                    </g>
                  ))}
                </>
              )}
            </Frame>
          </div>
        </div>
      ) : (
        <div style={{ height: 400 }} aria-hidden="true" />
      )}
    </div>
  );
}
