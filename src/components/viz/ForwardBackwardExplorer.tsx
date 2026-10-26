/**
 * ForwardBackwardExplorer — featured interactive for Topic 34 §34.5.
 *
 * Makes Thm 7 visible on the trellis: the scaled forward pass α̂_t carries
 * evidence left to right (it IS the filter, Thm 6), the scaled backward pass
 * β̂_t carries it right to left, and their elementwise product is the
 * smoothing marginal γ_t. A time cursor shows the three length-K vectors at
 * one column, and the readout checks Σ_k α̂_t(k) β̂_t(k) = 1 (Thm 7(i),
 * scaled) at every step.
 *
 * Data are simulated once from the preset; the "switching rate" slider
 * changes only the transition matrix used for INFERENCE, so the reader sees
 * how assumed persistence controls how far smoothing borrows from the future.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import { forwardBackward, posteriorDecode, sampleHMM, type HMMParams } from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { forwardBackwardPresets } from '../../data/time-series-data';

const MOBILE_BREAKPOINT = 640;
const NARROW_BREAKPOINT = 480;
const MOBILE_WINDOW = 30;
const LEFT = 64;
const PLAY_MS = 125;
const SIDE_BY_SIDE = 900;
const SIDE_CARD_W = 224;

/** Scale the off-diagonal mass of every row by m; the diagonal absorbs the change. */
function scaleSwitching(p: HMMParams, m: number): HMMParams {
  const transition = p.transition.map((row, j) => {
    const off = row.map((v, k) => (k === j ? 0 : v * m));
    const offSum = off.reduce((s, v) => s + v, 0);
    const capped = offSum > 0.99 ? off.map((v) => (v * 0.99) / offSum) : off;
    const cappedSum = capped.reduce((s, v) => s + v, 0);
    return capped.map((v, k) => (k === j ? 1 - cappedSum : v));
  });
  return { ...p, transition };
}

function fmt(v: number): string {
  return v.toFixed(3);
}

export default function ForwardBackwardExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const preset = forwardBackwardPresets[presetIdx];
  const [seed, setSeed] = useState(preset.seed);
  const [switching, setSwitching] = useState(1);
  const [cursor, setCursor] = useState(Math.floor(preset.T / 2));
  const [showTruth, setShowTruth] = useState(true);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const narrow = containerW < NARROW_BREAKPOINT;
  const T = preset.T;
  const K = preset.params.initial.length;
  const colors = timeSeriesColors.states;

  const selectPreset = (i: number) => {
    setPresetIdx(i);
    setSeed(forwardBackwardPresets[i].seed);
    setCursor(Math.floor(forwardBackwardPresets[i].T / 2));
    setPlaying(false);
  };

  const sim = useMemo(() => sampleHMM(preset.params, T, createSeededRng(seed)), [preset, T, seed]);
  const model = useMemo(() => scaleSwitching(preset.params, switching), [preset, switching]);
  const fb = useMemo(() => forwardBackward(model, sim.obs), [model, sim]);

  const metrics = useMemo(() => {
    const filtDec = posteriorDecode(fb.alphaHat);
    const smoothDec = posteriorDecode(fb.gamma);
    const acc = (d: number[]) => d.filter((k, t) => k === sim.states[t]).length / T;
    return { accF: acc(filtDec), accS: acc(smoothDec) };
  }, [fb, sim, T]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setCursor((c) => {
        if (c >= T - 1) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, PLAY_MS);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
      timer.current = null;
    };
  }, [playing, T]);

  const t = Math.min(cursor, T - 1);
  const a = fb.alphaHat[t];
  const bRaw = fb.betaHat[t];
  const bSum = bRaw.reduce((s, v) => s + v, 0);
  const b = bRaw.map((v) => v / bSum);
  const g = fb.gamma[t];
  const prodCheck = a.reduce((s, v, k) => s + v * bRaw[k], 0);

  // Visible window: whole sequence on desktop, ±MOBILE_WINDOW steps around the cursor on narrow screens.
  const lo = isMobile ? Math.max(0, Math.min(t - MOBILE_WINDOW, T - 2 * MOBILE_WINDOW - 1)) : 0;
  const hi = isMobile ? Math.min(T, lo + 2 * MOBILE_WINDOW + 1) : T;
  const nVis = hi - lo;
  const sideBySide = containerW >= SIDE_BY_SIDE;
  const plotW = Math.max(220, containerW - LEFT - 8 - (sideBySide ? SIDE_CARD_W + 16 : 0));
  const cellW = plotW / nVis;
  const cellH = K === 2 ? 13 : 10;
  const rowGap = 10;
  const blockH = K * cellH;
  const obsH = 34;
  const heatTop = obsH + 14;
  const heatH = 3 * blockH + 2 * rowGap;
  const lineTop = heatTop + heatH + 18;
  const lineH = 120;
  const svgH = lineTop + lineH + 26;
  const xOf = (s: number) => LEFT + (s - lo) * cellW;

  const isDice = preset.params.emission.kind === 'categorical';
  const obsAbs = isDice ? 1 : Math.max(1, ...sim.obs.slice(lo, hi).map(Math.abs));

  const rows: { name: string; data: number[][]; normalize: boolean }[] = [
    { name: 'α̂ filter', data: fb.alphaHat, normalize: false },
    { name: 'β̂ (norm.)', data: fb.betaHat, normalize: true },
    { name: 'γ smoother', data: fb.gamma, normalize: false },
  ];

  // Bottom panel: probability of the last state (K = 2) or of every state (K = 3).
  const shownStates = K === 2 ? [1] : Array.from({ length: K }, (_, k) => k);
  const ly = (v: number) => lineTop + lineH - v * lineH;
  const pathFor = (series: number[][], k: number) =>
    Array.from({ length: nVis }, (_, i) => {
      const s = lo + i;
      return `${i === 0 ? 'M' : 'L'}${(xOf(s) + cellW / 2).toFixed(2)},${ly(series[s][k]).toFixed(2)}`;
    }).join('');

  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-smoothed)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Scenario:</span>
        {forwardBackwardPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => selectPreset(i)}
            className={btn(i === presetIdx)}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
        <label className="ml-auto flex items-center gap-1 text-xs">
          <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
          shade true states
        </label>
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 text-sm sm:grid-cols-[2fr_1fr_auto]">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>Time cursor t</span>
            <span className="font-mono">{t + 1}</span>
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setCursor((c) => Math.max(0, c - 1))}
              className="rounded border px-2 py-0.5 text-xs"
              style={{ borderColor: 'var(--color-border)' }}
              aria-label="Step back"
            >
              ◀
            </button>
            <input
              type="range"
              min={0}
              max={T - 1}
              value={t}
              onChange={(e) => setCursor(parseInt(e.target.value, 10))}
              className="w-full accent-[var(--color-ts-smoothed)]"
            />
            <button
              onClick={() => setCursor((c) => Math.min(T - 1, c + 1))}
              className="rounded border px-2 py-0.5 text-xs"
              style={{ borderColor: 'var(--color-border)' }}
              aria-label="Step forward"
            >
              ▶
            </button>
            {!narrow && (
              <button
                onClick={() => {
                  if (!playing && t >= T - 1) setCursor(0);
                  setPlaying((v) => !v);
                }}
                className="rounded border px-2 py-0.5 text-xs"
                style={{ borderColor: 'var(--color-border)' }}
              >
                {playing ? 'Pause' : 'Play'}
              </button>
            )}
          </div>
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between text-xs">
            <span>Assumed switching rate</span>
            <span className="font-mono">×{switching.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0.2}
            max={3}
            step={0.05}
            value={switching}
            onChange={(e) => setSwitching(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-ts-smoothed)]"
          />
        </label>
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="rounded px-3 py-1.5 text-xs font-medium text-white"
          style={{ background: timeSeriesColors.smoothed }}
        >
          Resample (seed {seed})
        </button>
      </div>

      {/* Charts render only after the container is measured: the SSR HTML and the first client render then match exactly, with no float-dependent attributes to mismatch. */}
      {width > 0 ? (
        <>
      <div className={`flex gap-4 ${sideBySide ? 'flex-row' : 'flex-col'}`}>
        <div className="min-w-0 flex-1 overflow-hidden">
          <svg width={LEFT + plotW + 8} height={svgH} className="block" role="img" aria-label="Trellis heatmap of forward, backward and smoothed probabilities with a time cursor">
            {/* observations */}
            <text x={0} y={obsH / 2 + 4} fontSize={10} fill="var(--color-text-muted)">
              {isDice ? 'rolls' : 'x_t'}
            </text>
            {showTruth &&
              Array.from({ length: nVis }, (_, i) => {
                const s = lo + i;
                return (
                  <rect
                    key={`tr-${s}`}
                    x={xOf(s)}
                    y={0}
                    width={cellW + 0.3}
                    height={obsH}
                    fill={colors[sim.states[s]]}
                    opacity={timeSeriesColors.truthAlpha + 0.05}
                  />
                );
              })}
            {Array.from({ length: nVis }, (_, i) => {
              const s = lo + i;
              const x = xOf(s) + cellW / 2;
              if (isDice) {
                const six = sim.obs[s] === 5;
                return (
                  <line
                    key={`o-${s}`}
                    x1={x}
                    x2={x}
                    y1={six ? 2 : obsH * 0.3}
                    y2={six ? obsH - 2 : obsH * 0.7}
                    stroke={six ? colors[1] : timeSeriesColors.neutral}
                    strokeWidth={six ? 1.6 : 0.8}
                  />
                );
              }
              const v = sim.obs[s] / obsAbs;
              return (
                <line
                  key={`o-${s}`}
                  x1={x}
                  x2={x}
                  y1={obsH / 2}
                  y2={obsH / 2 - (v * obsH) / 2.1}
                  stroke={timeSeriesColors.neutral}
                  strokeWidth={Math.max(0.8, Math.min(2.5, cellW * 0.6))}
                />
              );
            })}

            {/* trellis heatmap: three message rows of K cells per column */}
            {rows.map((row, r) => {
              const top = heatTop + r * (blockH + rowGap);
              return (
                <g key={row.name}>
                  <text x={0} y={top + blockH / 2 + 4} fontSize={10} fill="var(--color-text)">
                    {row.name}
                  </text>
                  {Array.from({ length: nVis }, (_, i) => {
                    const s = lo + i;
                    const vec = row.data[s];
                    const z = row.normalize ? vec.reduce((acc, v) => acc + v, 0) : 1;
                    return vec.map((v, k) => (
                      <rect
                        key={`${r}-${s}-${k}`}
                        x={xOf(s)}
                        y={top + k * cellH}
                        width={cellW + 0.3}
                        height={cellH - 1}
                        fill={colors[k]}
                        opacity={0.06 + 0.94 * (v / z)}
                      />
                    ));
                  })}
                </g>
              );
            })}

            {/* probability lines */}
            <text x={0} y={lineTop + 10} fontSize={10} fill="var(--color-text-muted)">
              P = 1
            </text>
            <text x={0} y={lineTop + lineH} fontSize={10} fill="var(--color-text-muted)">
              P = 0
            </text>
            <line x1={LEFT} x2={LEFT + plotW} y1={ly(0.5)} y2={ly(0.5)} stroke="var(--color-viz-grid)" strokeDasharray="3 3" />
            <line x1={LEFT} x2={LEFT + plotW} y1={ly(0)} y2={ly(0)} stroke="var(--color-text-muted)" />
            {shownStates.map((k) => (
              <g key={`l-${k}`}>
                <path
                  d={pathFor(fb.alphaHat, k)}
                  fill="none"
                  stroke={K === 2 ? timeSeriesColors.filtered : colors[k]}
                  strokeWidth={K === 2 ? 1.4 : 1}
                  strokeDasharray="4 3"
                  opacity={K === 2 ? 0.9 : 0.6}
                />
                <path
                  d={pathFor(fb.gamma, k)}
                  fill="none"
                  stroke={K === 2 ? timeSeriesColors.smoothed : colors[k]}
                  strokeWidth={2}
                />
              </g>
            ))}

            {/* cursor */}
            <rect
              x={xOf(t) - 0.5}
              y={0}
              width={cellW + 1}
              height={lineTop + lineH}
              fill="none"
              stroke="var(--color-text)"
              strokeWidth={1.2}
            />
            <text x={LEFT} y={svgH - 6} fontSize={10} fill="var(--color-text-muted)">
              t = {lo + 1}
            </text>
            <text x={LEFT + plotW} y={svgH - 6} fontSize={10} textAnchor="end" fill="var(--color-text-muted)">
              t = {hi}
            </text>
          </svg>
          <div className="mt-1 flex flex-wrap gap-x-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {K === 2 ? (
              <>
                <span style={{ color: timeSeriesColors.filtered }}>- - filtered P({preset.stateNames[1]} | x₁:ₜ)</span>
                <span style={{ color: timeSeriesColors.smoothed }}>— smoothed P({preset.stateNames[1]} | x₁:T)</span>
              </>
            ) : (
              <span>— smoothed, - - filtered, one color per state</span>
            )}
            {preset.stateNames.map((n, k) => (
              <span key={n} style={{ color: colors[k] }}>
                ■ {n}
              </span>
            ))}
          </div>
        </div>

        <div
          className="shrink-0 rounded border p-3 text-xs"
          style={{ borderColor: 'var(--color-border)', width: sideBySide ? SIDE_CARD_W : '100%' }}
        >
          <div className="mb-2 font-semibold">At t = {t + 1}</div>
          <table className="w-full font-mono">
            <thead>
              <tr style={{ color: 'var(--color-text-muted)' }}>
                <th className="text-left font-normal">state</th>
                <th className="text-right font-normal">α̂</th>
                <th className="text-right font-normal">β̂*</th>
                <th className="text-right font-normal">γ</th>
              </tr>
            </thead>
            <tbody>
              {preset.stateNames.map((n, k) => (
                <tr key={n}>
                  <td style={{ color: colors[k] }}>{n}</td>
                  <td className="text-right">{fmt(a[k])}</td>
                  <td className="text-right">{fmt(b[k])}</td>
                  <td className="text-right font-semibold">{fmt(g[k])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2" style={{ color: 'var(--color-text-muted)' }}>
            γ = α̂ ⊙ β̂, already normalized. *β̂ shown rescaled to sum to 1.
          </p>
          <div className="mt-3 space-y-1 font-mono">
            <div>Σₖ α̂ₜ(k)β̂ₜ(k) = {prodCheck.toFixed(6)}</div>
            <div>log p(x₁:T) = {fb.logLik.toFixed(2)}</div>
            <div>
              accuracy: filter {(100 * metrics.accF).toFixed(1)}% · smoother {(100 * metrics.accS).toFixed(1)}%
            </div>
            {switching !== 1 && (
              <div style={{ color: 'var(--color-text-muted)' }}>inference assumes ×{switching.toFixed(2)} switching; data use ×1</div>
            )}
          </div>
        </div>
      </div>
        </>
      ) : (
        <div style={{ height: 420 }} aria-hidden="true" />
      )}
    </div>
  );
}
