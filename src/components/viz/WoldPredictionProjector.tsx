/**
 * WoldPredictionProjector — optional interactive for Topic 35 §35.3.
 *
 * Thms 3 and 4 made tangible. Forecasts are projections onto the past: from a
 * draggable origin, the h-step forecast uses the innovations recovered by
 * inverting the model (Thm 6), and its error variance σ² Σ_{j<h} ψ_j² climbs
 * to γ(0) as the Wold weights run out — immediately after lag q for an MA(q).
 * The bar panel shows the finite-past Durbin–Levinson coefficients
 * φ_{n1}, …, φ_{nn} with v_n; for an AR(p) every bar beyond p is exactly zero
 * once n ≥ p (Thm 7), and v_n decreases to σ².
 */
import { useMemo, useRef, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import { armaACVF, cssResiduals, durbinLevinson, psiWeights, simulateARMA } from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { woldPresets } from '../../data/time-series-data';

const MARGIN = { top: 10, right: 12, bottom: 34, left: 46 };
const MOBILE_BREAKPOINT = 640;
const PATH_N = 120;
const H_MAX = 30;
const N_PAST_MAX = 20;

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

function niceTicks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Number(t.toFixed(3)));
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
  xTicks,
  yTicks,
  children,
  svgProps,
}: {
  width: number;
  height: number;
  label: string;
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks: number[];
  yTicks: number[];
  children: (sx: (v: number) => number, sy: (v: number) => number, plotH: number) => React.ReactNode;
  svgProps?: React.SVGProps<SVGSVGElement>;
}) {
  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom);
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = (v: number) => plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const axisColor = 'var(--color-text-muted)';
  return (
    <svg width={width} height={height} className="block" role="img" aria-label={label} {...svgProps}>
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={0} y1={sy(t)} x2={plotW} y2={sy(t)} stroke="var(--color-viz-grid)" />
            <text x={-6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
              {t}
            </text>
          </g>
        ))}
        {children(sx, sy, plotH)}
        <line x1={0} y1={plotH} x2={plotW} y2={plotH} stroke={axisColor} />
        <line x1={0} y1={0} x2={0} y2={plotH} stroke={axisColor} />
        {xTicks.map((t) => (
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
          x={-34}
          y={plotH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="var(--color-text)"
          transform={`rotate(-90 ${-34} ${plotH / 2})`}
        >
          {yLabel}
        </text>
      </g>
    </svg>
  );
}

function Slider({
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
        <span className="font-mono">{value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        className="w-full accent-[var(--color-ts-ar)]"
      />
    </label>
  );
}

export default function WoldPredictionProjector() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const [h, setH] = useState(10);
  const [nPast, setNPast] = useState(5);
  const [origin, setOrigin] = useState(100);
  const [seed, setSeed] = useState(woldPresets[0].seed);
  const dragging = useRef(false);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;
  const preset = woldPresets[presetIdx];
  const params = preset.params;
  const p = params.phi.length;
  const q = params.theta.length;

  const path = useMemo(() => simulateARMA(params, PATH_N, createSeededRng(seed)), [params, seed]);
  // Innovations by inverting the model (Thm 6); the zero start-up error dies out geometrically.
  const innov = useMemo(() => [...new Array<number>(p).fill(0), ...cssResiduals(path, params.phi, params.theta)], [path, params, p]);

  const theory = useMemo(() => {
    const psi = psiWeights(params.phi, params.theta, H_MAX);
    const g0 = armaACVF(params, 0)[0];
    let cum = 0;
    const mseRatio = psi.map((v) => {
      cum += v * v;
      return (params.sigma2 * cum) / g0;
    });
    return { psi, g0, mseRatio };
  }, [params]);

  const dl = useMemo(() => durbinLevinson(armaACVF(params, nPast)), [params, nPast]);

  const forecast = useMemo(() => {
    // x̂_{t0+k} = Σ φ_j x̃_{t0+k−j} + Σ_{j ≥ k} θ_j e_{t0+k−j}, with x̃ = x up to t0 and x̂ after.
    const xs: number[] = [];
    const ext = path.slice(0, origin + 1);
    for (let k = 1; k <= h; k++) {
      const t = origin + k;
      let v = 0;
      for (let j = 1; j <= p; j++) v += params.phi[j - 1] * ext[t - j];
      for (let j = k; j <= q; j++) v += params.theta[j - 1] * innov[t - j];
      ext.push(v);
      xs.push(v);
    }
    const sd = theory.mseRatio.slice(0, h).map((r) => 1.96 * Math.sqrt(r * theory.g0));
    return { xs, sd };
  }, [path, innov, origin, h, p, q, params, theory]);

  const fullW = Math.max(280, containerW);
  const thirdW = isMobile ? fullW : (fullW - 16) / 2;
  const yAbs = Math.max(3, Math.ceil(Math.max(...path.map(Math.abs), 2 * Math.sqrt(theory.g0)) * 1.1));
  const xMax = PATH_N + H_MAX;
  const plotW = Math.max(10, fullW - MARGIN.left - MARGIN.right);

  const setOriginFromEvent = (ev: React.PointerEvent<SVGSVGElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const t = Math.round(((ev.clientX - rect.left - MARGIN.left) / plotW) * xMax);
    setOrigin(Math.max(Math.max(p, q) + 5, Math.min(PATH_N - 1, t)));
  };

  const r2 = 1 - theory.mseRatio[h - 1];
  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-ar)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  const coefMax = Math.max(0.6, ...dl.phi.map(Math.abs)) * 1.15;

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Model:</span>
        {woldPresets.map((pr, i) => (
          <button key={pr.id} onClick={() => setPresetIdx(i)} className={btn(i === presetIdx)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {pr.label}
          </button>
        ))}
      </div>
      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-4">
        <Slider label="forecast horizon h" value={h} min={1} max={H_MAX} onChange={setH} />
        <Slider label="past length n (Durbin–Levinson)" value={nPast} min={1} max={N_PAST_MAX} onChange={setNPast} />
        <Slider label="forecast origin t₀" value={origin} min={Math.max(p, q) + 5} max={PATH_N - 1} onChange={setOrigin} />
        <button onClick={() => setSeed((s) => s + 1)} className="rounded px-3 py-1.5 text-xs font-medium text-white" style={{ background: timeSeriesColors.ar }}>
          Resample (seed {seed})
        </button>
      </div>
      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        <span>R²(h) = 1 − MSE_h/γ(0) = {r2.toFixed(4)}</span>
        <span>γ(0) = {theory.g0.toFixed(4)}</span>
        <span>
          v_{nPast} = {dl.v[nPast].toFixed(4)}
        </span>
        <span>σ² = {params.sigma2.toFixed(4)}</span>
      </div>

      {width > 0 ? (
        <>
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Path (drag to move the origin): <span style={{ color: timeSeriesColors.neutral }}>— observed</span> ·{' '}
            <span style={{ color: timeSeriesColors.ar }}>— {h}-step forecast with ±1.96·√MSE band</span> · faded = what actually happened
          </div>
          <Panel
            width={fullW}
            height={isMobile ? 180 : 210}
            label="Sample path with forecasts from a movable origin and their 95 percent band"
            xLabel="time t"
            yLabel="x_t"
            xDomain={[0, xMax]}
            yDomain={[-yAbs, yAbs]}
            xTicks={niceTicks(0, xMax, 30)}
            yTicks={niceTicks(-yAbs, yAbs, yAbs > 8 ? 4 : 2)}
            svgProps={{
              style: { touchAction: 'pan-y', cursor: 'ew-resize' },
              onPointerDown: (ev) => {
                dragging.current = true;
                setOriginFromEvent(ev);
                try {
                  ev.currentTarget.setPointerCapture(ev.pointerId);
                } catch {
                  /* pointer no longer active */
                }
              },
              onPointerMove: (ev) => {
                if (dragging.current) setOriginFromEvent(ev);
              },
              onPointerUp: () => {
                dragging.current = false;
              },
              onPointerCancel: () => {
                dragging.current = false;
              },
            }}
          >
            {(sx, sy, plotH) => {
              const ts = forecast.xs.map((_, k) => origin + k + 1);
              const up = forecast.xs.map((v, k) => v + forecast.sd[k]);
              const lo = forecast.xs.map((v, k) => v - forecast.sd[k]);
              const band = `${linePath([origin, ...ts], [path[origin], ...up], sx, sy)}${[...ts]
                .reverse()
                .map((t, k) => `L${sx(t).toFixed(2)},${sy(lo[lo.length - 1 - k]).toFixed(2)}`)
                .join('')}L${sx(origin)},${sy(path[origin])}Z`;
              const past = path.slice(0, origin + 1);
              const future = path.slice(origin);
              return (
                <>
                  <path d={band} fill={timeSeriesColors.ar} opacity={0.15} />
                  <path d={linePath(past.map((_, i) => i), past, sx, sy)} fill="none" stroke={timeSeriesColors.neutral} strokeWidth={1} />
                  <path d={linePath(future.map((_, i) => origin + i), future, sx, sy)} fill="none" stroke={timeSeriesColors.neutral} strokeWidth={1} opacity={0.3} />
                  <path d={linePath([origin, ...ts], [path[origin], ...forecast.xs], sx, sy)} fill="none" stroke={timeSeriesColors.ar} strokeWidth={2} />
                  <line x1={sx(origin)} x2={sx(origin)} y1={0} y2={plotH} stroke={timeSeriesColors.ar} strokeDasharray="3 3" />
                  <circle cx={sx(origin)} cy={sy(path[origin])} r={4.5} fill={timeSeriesColors.ar} />
                </>
              );
            }}
          </Panel>

          <div className={`mt-4 ${isMobile ? 'flex flex-col gap-4' : 'grid grid-cols-2 gap-4'}`}>
            <div className="min-w-0">
              <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                h-step MSE / γ(0), with MSE = σ² (ψ₀² + … + ψ²_(h−1)); dashed line = forecast is the mean
              </div>
              <Panel
                width={thirdW}
                height={210}
                label="Forecast mean squared error relative to the variance against the horizon"
                xLabel="horizon h"
                yLabel="MSE / γ(0)"
                xDomain={[0, H_MAX + 1]}
                yDomain={[0, 1.1]}
                xTicks={niceTicks(0, H_MAX, 5)}
                yTicks={[0, 0.25, 0.5, 0.75, 1]}
              >
                {(sx, sy) => {
                  const hs = theory.mseRatio.map((_, i) => i + 1);
                  return (
                    <>
                      <line x1={sx(0)} x2={sx(H_MAX + 1)} y1={sy(1)} y2={sy(1)} stroke={timeSeriesColors.truth} strokeDasharray="6 4" />
                      <path d={linePath(hs, theory.mseRatio, sx, sy)} fill="none" stroke={timeSeriesColors.ar} strokeWidth={2} />
                      {hs.map((hh, i) => (
                        <circle key={hh} cx={sx(hh)} cy={sy(theory.mseRatio[i])} r={hh === h ? 5 : 2.5} fill={timeSeriesColors.ar} />
                      ))}
                    </>
                  );
                }}
              </Panel>
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
                Durbin–Levinson coefficients φ_{nPast},1 … φ_{nPast},{nPast} of the best predictor from the last {nPast} values
              </div>
              <Panel
                width={thirdW}
                height={210}
                label="Bar chart of the Durbin–Levinson prediction coefficients"
                xLabel="coefficient j"
                yLabel="φ_nj"
                xDomain={[0, N_PAST_MAX + 1]}
                yDomain={[-coefMax, coefMax]}
                xTicks={niceTicks(0, N_PAST_MAX, 5)}
                yTicks={niceTicks(-coefMax, coefMax, coefMax > 1.2 ? 0.5 : 0.25)}
              >
                {(sx, sy) => {
                  const bw = Math.max(2, (sx(1) - sx(0)) * 0.6);
                  return (
                    <>
                      <line x1={sx(0)} x2={sx(N_PAST_MAX + 1)} y1={sy(0)} y2={sy(0)} stroke="var(--color-text-muted)" />
                      {dl.phi.map((v, i) => (
                        <rect
                          key={i}
                          x={sx(i + 1) - bw / 2}
                          y={Math.min(sy(v), sy(0))}
                          width={bw}
                          height={Math.max(0.5, Math.abs(sy(v) - sy(0)))}
                          fill={i < p ? timeSeriesColors.ar : timeSeriesColors.ma}
                          opacity={0.85}
                        />
                      ))}
                    </>
                  );
                }}
              </Panel>
            </div>
          </div>
        </>
      ) : (
        <div style={{ height: 470 }} aria-hidden="true" />
      )}
    </div>
  );
}
