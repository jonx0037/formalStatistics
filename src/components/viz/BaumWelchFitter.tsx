/**
 * BaumWelchFitter — interactive for Topic 34 §34.7.
 *
 * Watch EM climb. Each step is one Baum–Welch E-step (forward–backward) plus
 * the closed-form M-step of Thm 10; the log-likelihood trace never decreases
 * (Thm 9) — a red badge would appear if it ever did, so the component doubles
 * as a visual test. Fitting with more states than the truth, or from a random
 * start, shows the local optima of Ex 13: every converged run drops a dot in
 * the restart tally.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import {
  baumWelchStep,
  hmmLogLikelihood,
  initKMeans,
  initRandom,
  relabelByMean,
  sampleHMM,
  stationaryDistribution,
  type GaussianEmission,
  type HMMParams,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { baumWelchPresets } from '../../data/time-series-data';

const MARGIN = { top: 12, right: 12, bottom: 36, left: 52 };
const MOBILE_BREAKPOINT = 640;
const RUN_CHUNK = 10;
const RUN_MAX = 500;
const RUN_TOL = 1e-6;
const N_BINS = 50;
const STATE_COLORS = [...timeSeriesColors.states, 'var(--color-ts-neutral)'];

interface FitState {
  key: string;
  params: HMMParams;
  trace: number[];
}

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function niceStep(span: number): number {
  const raw = span / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
}

function ticks(lo: number, hi: number): number[] {
  const step = niceStep(hi - lo || 1);
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toPrecision(10)));
  return out;
}

function Panel({
  width,
  height,
  label,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  children,
  yFormat = (v: number) => String(v),
}: {
  width: number;
  height: number;
  label: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  children: (sx: (v: number) => number, sy: (v: number) => number) => React.ReactNode;
  yFormat?: (v: number) => string;
}) {
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = (v: number) => plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const axisColor = 'var(--color-text-muted)';
  return (
    <svg width={width} height={height} className="block" role="img" aria-label={label}>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {ticks(yDomain[0], yDomain[1]).map((t) => (
          <g key={t}>
            <line x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
            <text x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
              {yFormat(t)}
            </text>
          </g>
        ))}
        <svg x={0} y={0} width={plotW} height={plotH} overflow="hidden">
          {children(sx, sy)}
        </svg>
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
        {ticks(xDomain[0], xDomain[1]).map((t) => (
          <g key={t}>
            <line x1={sx(t)} y1={plotH} x2={sx(t)} y2={plotH + 4} stroke={axisColor} />
            <text x={sx(t)} y={plotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
              {t}
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

function normalPdf(x: number, m: number, s: number): number {
  const z = (x - m) / s;
  return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}

function weightsOf(p: HMMParams): number[] {
  try {
    const w = stationaryDistribution(p.transition);
    if (w.every((v) => Number.isFinite(v) && v >= -1e-9)) return w.map((v) => Math.max(0, v));
  } catch {
    // Reducible fitted chain (a state never left): fall back to equal weights.
  }
  return p.initial.map(() => 1 / p.initial.length);
}

export default function BaumWelchFitter() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = baumWelchPresets[presetIdx];
  const [fitK, setFitK] = useState(preset.params.initial.length);
  const [T, setT] = useState(preset.T);
  const [dataSeed, setDataSeed] = useState(preset.seed);
  const [initMode, setInitMode] = useState<'kmeans' | 'random'>('kmeans');
  const [initSeed, setInitSeed] = useState(1);
  const [fit, setFit] = useState<FitState | null>(null);
  const [tally, setTally] = useState<{ key: string; values: number[] }>({ key: '', values: [] });
  const [running, setRunning] = useState(false);
  const rafRef = useRef<number | null>(null);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const trueK = preset.params.initial.length;

  const data = useMemo(() => sampleHMM(preset.params, T, createSeededRng(dataSeed)).obs, [preset, T, dataSeed]);
  const dataKey = `${preset.id}|${T}|${dataSeed}|${fitK}`;
  const fitKey = `${dataKey}|${initMode}|${initSeed}`;

  const init = useMemo(
    () =>
      initMode === 'kmeans' ? initKMeans(data, fitK) : initRandom(data, fitK, createSeededRng(7919 * initSeed + 17), 'gaussian'),
    [data, fitK, initMode, initSeed],
  );
  const initLL = useMemo(() => hmmLogLikelihood(init, data), [init, data]);
  const current: FitState = fit && fit.key === fitKey ? fit : { key: fitKey, params: init, trace: [initLL] };
  const tallyValues = tally.key === dataKey ? tally.values : [];
  const trueLL = useMemo(() => (fitK === trueK ? hmmLogLikelihood(preset.params, data) : null), [fitK, trueK, preset, data]);

  const stopRun = () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    setRunning(false);
  };
  useEffect(() => stopRun, [fitKey]);

  const advance = (state: FitState, n: number): FitState => {
    let p = state.params;
    const trace = state.trace.slice();
    for (let i = 0; i < n; i++) {
      p = baumWelchStep(p, data).params;
      trace.push(hmmLogLikelihood(p, data));
    }
    return { key: state.key, params: p, trace };
  };

  const step = (n: number) => setFit(advance(current, n));

  const converged = (tr: number[]) => tr.length >= 2 && Math.abs(tr[tr.length - 1] - tr[tr.length - 2]) < RUN_TOL;

  const runToConvergence = () => {
    if (running) {
      stopRun();
      return;
    }
    let state = current;
    if (converged(state.trace) || state.trace.length > RUN_MAX) return;
    setRunning(true);
    const tick = () => {
      state = advance(state, RUN_CHUNK);
      setFit(state);
      if (converged(state.trace) || state.trace.length > RUN_MAX) {
        rafRef.current = null;
        setRunning(false);
        const final = state.trace[state.trace.length - 1];
        setTally((tl) => ({ key: dataKey, values: [...(tl.key === dataKey ? tl.values : []), final] }));
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const selectPreset = (i: number) => {
    stopRun();
    setPresetIdx(i);
    setFitK(baumWelchPresets[i].params.initial.length);
    setT(baumWelchPresets[i].T);
    setDataSeed(baumWelchPresets[i].seed);
    setFit(null);
  };

  const trace = current.trace;
  const iter = trace.length - 1;
  const lastDelta = trace.length >= 2 ? trace[trace.length - 1] - trace[trace.length - 2] : NaN;
  const minDelta = trace.length >= 2 ? Math.min(...trace.slice(1).map((v, i) => v - trace[i])) : 0;
  const violated = minDelta < -1e-9;

  const shown = relabelByMean(current.params);
  const em = shown.emission as GaussianEmission;
  const w = weightsOf(shown);
  const trueEm = preset.params.emission as GaussianEmission;
  const trueW = preset.params.initial;

  // Histogram of the data (density scale) on a robust range.
  const sorted = useMemo(() => data.slice().sort((a, b) => a - b), [data]);
  const xLo = sorted[Math.floor(0.002 * sorted.length)] - 0.5;
  const xHi = sorted[Math.ceil(0.998 * sorted.length) - 1] + 0.5;
  const binW = (xHi - xLo) / N_BINS;
  const hist = useMemo(() => {
    const c = new Array<number>(N_BINS).fill(0);
    data.forEach((v) => {
      const b = Math.floor((v - xLo) / binW);
      if (b >= 0 && b < N_BINS) c[b]++;
    });
    return c.map((n) => n / (data.length * binW));
  }, [data, xLo, binW]);
  const xs = useMemo(() => Array.from({ length: 241 }, (_, i) => xLo + ((xHi - xLo) * i) / 240), [xLo, xHi]);
  const comp = em.means.map((m, k) => xs.map((x) => w[k] * normalPdf(x, m, em.sds[k])));
  const mix = xs.map((_, i) => comp.reduce((s, c) => s + c[i], 0));
  const trueMix = xs.map((x) => trueEm.means.reduce((s, m, k) => s + trueW[k] * normalPdf(x, m, trueEm.sds[k]), 0));
  const densMax = 1.1 * Math.max(...hist, ...mix, ...trueMix);

  const llLo = Math.min(...trace, trueLL ?? Infinity);
  const llHi = Math.max(...trace, trueLL ?? -Infinity);
  const llPad = Math.max(1, 0.05 * (llHi - llLo));

  const fullW = Math.max(280, containerW);
  const halfW = Math.max(260, isMobile ? fullW : (fullW - 16) / 2);
  const btnCls = 'rounded border px-3 py-1.5 text-xs transition disabled:opacity-40';
  const chip = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-filtered)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Data:</span>
        {baumWelchPresets.map((p, i) => (
          <button key={p.id} onClick={() => selectPreset(i)} className={chip(i === presetIdx)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {p.label}
          </button>
        ))}
        <span className="ml-2 font-semibold">Fit K:</span>
        {[2, 3, 4].map((k) => (
          <button key={k} onClick={() => setFitK(k)} className={chip(k === fitK)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {k}
          </button>
        ))}
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-[1fr_auto_auto]">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>Sequence length T</span>
            <span className="font-mono">{T.toLocaleString('en-US')}</span>
          </span>
          <input
            type="range"
            min={300}
            max={3000}
            step={100}
            value={T}
            onChange={(e) => setT(parseInt(e.target.value, 10))}
            className="w-full accent-[var(--color-ts-filtered)]"
          />
        </label>
        <button onClick={() => setDataSeed((s) => s + 1)} className={btnCls} style={{ borderColor: 'var(--color-border)' }}>
          Resample data (seed {dataSeed})
        </button>
        <div className="flex gap-2">
          <button
            onClick={() => setInitMode('kmeans')}
            className={chip(initMode === 'kmeans')}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            k-means init
          </button>
          <button
            onClick={() => {
              setInitMode('random');
              setInitSeed((s) => s + 1);
            }}
            className={chip(initMode === 'random')}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            New random init
          </button>
        </div>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
        {[1, 10].map((n) => (
          <button key={n} onClick={() => step(n)} disabled={running} className={btnCls} style={{ borderColor: 'var(--color-border)' }}>
            EM step ×{n}
          </button>
        ))}
        <button onClick={runToConvergence} className="rounded px-3 py-1.5 text-xs font-medium text-white" style={{ background: timeSeriesColors.filtered }}>
          {running ? 'Stop' : 'Run to convergence'}
        </button>
        <button
          onClick={() => {
            stopRun();
            setFit(null);
          }}
          className={btnCls}
          style={{ borderColor: 'var(--color-border)' }}
        >
          Reset
        </button>
      </div>

      {/* Charts render only after the container is measured: the SSR HTML and the first client render then match exactly, with no float-dependent attributes to mismatch. */}
      {width > 0 ? (
        <>
      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        <span>iteration {iter}</span>
        <span>ℓ = {trace[trace.length - 1].toFixed(3)}</span>
        <span>Δℓ = {Number.isNaN(lastDelta) ? '—' : lastDelta.toExponential(2)}</span>
        {trueLL !== null && <span>ℓ(true θ) = {trueLL.toFixed(3)}</span>}
        {converged(trace) && <span style={{ color: timeSeriesColors.smoothed }}>converged (|Δℓ| &lt; 10⁻⁶)</span>}
        {violated ? (
          <span className="rounded px-1 text-white" style={{ background: timeSeriesColors.states[1] }}>
            monotonicity violated: min Δℓ = {minDelta.toExponential(2)}
          </span>
        ) : (
          iter > 0 && <span style={{ color: 'var(--color-text-muted)' }}>never decreased ✓</span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <span style={{ color: timeSeriesColors.filtered }}>—● log-likelihood ℓ(θ⁽ˢ⁾)</span>
            {trueLL !== null && <span> · - - true parameters</span>}
          </div>
          <Panel
            width={halfW}
            height={220}
            label="Log-likelihood trace over EM iterations"
            xLabel="iteration s"
            yLabel="log-likelihood"
            xDomain={[0, Math.max(5, iter)]}
            yDomain={[llLo - llPad, llHi + llPad]}
          >
            {(sx, sy) => (
              <>
                {trueLL !== null && (
                  <line x1={sx(0)} x2={sx(Math.max(5, iter))} y1={sy(trueLL)} y2={sy(trueLL)} stroke={timeSeriesColors.viterbi} strokeDasharray="5 3" />
                )}
                <path d={linePath(trace.map((_, i) => i), trace, sx, sy)} fill="none" stroke={timeSeriesColors.filtered} strokeWidth={2} />
                {trace.length <= 80 && trace.map((v, i) => <circle key={i} cx={sx(i)} cy={sy(v)} r={2.2} fill={timeSeriesColors.filtered} />)}
              </>
            )}
          </Panel>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Histogram of data · — fitted components (colored) and mixture · - - true mixture
          </div>
          <Panel
            width={halfW}
            height={220}
            label="Data histogram with fitted and true emission densities"
            xLabel="observation x"
            yLabel="density"
            xDomain={[xLo, xHi]}
            yDomain={[0, densMax]}
            yFormat={(v) => v.toFixed(2)}
          >
            {(sx, sy) => (
              <>
                {hist.map((h, b) => (
                  <rect
                    key={b}
                    x={sx(xLo + b * binW)}
                    y={sy(h)}
                    width={Math.max(0.5, sx(xLo + (b + 1) * binW) - sx(xLo + b * binW) - 0.5)}
                    height={sy(0) - sy(h)}
                    fill="var(--color-viz-grid)"
                  />
                ))}
                {comp.map((c, k) => (
                  <path key={k} d={linePath(xs, c, sx, sy)} fill="none" stroke={STATE_COLORS[k]} strokeWidth={1.6} />
                ))}
                <path d={linePath(xs, mix, sx, sy)} fill="none" stroke={timeSeriesColors.viterbi} strokeWidth={1.4} />
                <path d={linePath(xs, trueMix, sx, sy)} fill="none" stroke={timeSeriesColors.viterbi} strokeWidth={1.4} strokeDasharray="5 3" opacity={0.6} />
              </>
            )}
          </Panel>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="min-w-0 text-xs">
          <div className="mb-1" style={{ color: 'var(--color-text-muted)' }}>
            Fitted transition matrix Â (states sorted by mean)
          </div>
          <table className="font-mono">
            <tbody>
              {shown.transition.map((row, j) => (
                <tr key={j}>
                  <td className="pr-2" style={{ color: STATE_COLORS[j] }}>
                    {j + 1}: μ {em.means[j].toFixed(2)}, σ {em.sds[j].toFixed(2)}
                  </td>
                  {row.map((v, k) => (
                    <td
                      key={k}
                      className="px-2 py-1 text-right"
                      style={{ background: `color-mix(in srgb, var(--color-ts-filtered) ${Math.round(80 * v)}%, transparent)` }}
                    >
                      {v.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2" style={{ color: 'var(--color-text-muted)' }}>
            Truth: μ = ({trueEm.means.join(', ')}), σ = ({trueEm.sds.join(', ')}), stay = (
            {preset.params.transition.map((r, k) => r[k]).join(', ')})
          </div>
        </div>
        <div className="min-w-0 text-xs">
          <div className="mb-1" style={{ color: 'var(--color-text-muted)' }}>
            Restart tally: final ℓ of each run to convergence on these data ({tallyValues.length} so far). Try “New random init” repeatedly.
          </div>
          {tallyValues.length > 0 ? (
            <Panel
              width={halfW}
              height={110}
              label="Final log-likelihoods of converged runs"
              xLabel="final log-likelihood"
              yLabel=""
              xDomain={[Math.min(...tallyValues, trueLL ?? Infinity) - 2, Math.max(...tallyValues, trueLL ?? -Infinity) + 2]}
              yDomain={[0, 1]}
              yFormat={() => ''}
            >
              {(sx, sy) => (
                <>
                  {trueLL !== null && <line x1={sx(trueLL)} x2={sx(trueLL)} y1={sy(0)} y2={sy(1)} stroke={timeSeriesColors.viterbi} strokeDasharray="5 3" />}
                  {tallyValues.map((v, i) => (
                    <circle key={i} cx={sx(v)} cy={sy(0.15 + 0.7 * ((i * 0.618) % 1))} r={4} fill={timeSeriesColors.filtered} opacity={0.75} />
                  ))}
                </>
              )}
            </Panel>
          ) : (
            <div className="rounded border p-3" style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-muted)' }}>
              No converged runs yet.
            </div>
          )}
        </div>
      </div>
        </>
      ) : (
        <div style={{ height: 600 }} aria-hidden="true" />
      )}
    </div>
  );
}
