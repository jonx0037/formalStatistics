/**
 * HMMRegimeSimulator — interactive for Topic 34 §34.3.
 *
 * The generative model made tactile: tune how sticky each hidden regime is
 * and what it emits, then watch (a) sojourns come out geometric (Thm 3) and
 * (b) volatility clustering appear in the autocorrelation of squared
 * observations, matching Ex 5's closed form
 *   corr(X_t², X_{t+h}²) = π₁π₂(s₁ − s₂)² λ^h / Var(X²),  λ = a₁₁ + a₂₂ − 1,
 * with s_k = μ_k² + σ_k². The "No memory" preset sets every row of A to π,
 * which keeps the marginal histogram and kills the time structure (Rem 5).
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { autocorrelation, createSeededRng } from './shared/bayes';
import { sampleHMM, sojournLengths, stationaryDistribution, type HMMParams } from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { regimeSimulatorPresets } from '../../data/time-series-data';

const MARGIN = { top: 12, right: 12, bottom: 36, left: 46 };
const MOBILE_BREAKPOINT = 640;
const MAX_LAG = 40;
const MAX_SOJOURN = 60;

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
}) {
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotW;
  const sy = (v: number) => plotH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotH;
  const axisColor = 'var(--color-text-muted)';
  return (
    <svg width={width} height={height} className="block" role="img" aria-label={label}>
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
  step,
  onChange,
  format = (v: number) => v.toFixed(3),
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex items-center justify-between text-xs">
        <span>{label}</span>
        <span className="font-mono">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[var(--color-ts-filtered)]"
      />
    </label>
  );
}

export default function HMMRegimeSimulator() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const p0 = regimeSimulatorPresets[0];
  const [stay, setStay] = useState<[number, number]>(p0.stay);
  const [means, setMeans] = useState<[number, number]>(p0.means);
  const [sds, setSds] = useState<[number, number]>(p0.sds);
  const [T, setT] = useState(2000);
  const [seed, setSeed] = useState(3401);
  const [showMore, setShowMore] = useState(false);

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;

  const selectPreset = (i: number) => {
    const p = regimeSimulatorPresets[i];
    setPresetIdx(i);
    setStay(p.stay);
    setMeans(p.means);
    setSds(p.sds);
  };

  const params: HMMParams = useMemo(() => {
    const A = [
      [stay[0], 1 - stay[0]],
      [1 - stay[1], stay[1]],
    ];
    return {
      initial: stationaryDistribution(A),
      transition: A,
      emission: { kind: 'gaussian', means: [...means], sds: [...sds] },
    };
  }, [stay, means, sds]);

  const sim = useMemo(() => sampleHMM(params, T, createSeededRng(seed)), [params, T, seed]);

  const stats = useMemo(() => {
    const { states, obs } = sim;
    const runs = sojournLengths(states, 2);
    const sq = obs.map((v) => v * v);
    const lags = Array.from({ length: MAX_LAG }, (_, i) => i + 1);
    const acfX = lags.map((h) => autocorrelation(obs, h));
    const acfX2 = lags.map((h) => autocorrelation(sq, h));
    const pi = params.initial;
    const lam = stay[0] + stay[1] - 1;
    const s2 = means.map((m, k) => m * m + sds[k] * sds[k]);
    const m4 = means.map((m, k) => m ** 4 + 6 * m * m * sds[k] ** 2 + 3 * sds[k] ** 4);
    const varX2 = pi[0] * m4[0] + pi[1] * m4[1] - (pi[0] * s2[0] + pi[1] * s2[1]) ** 2;
    const theory = lags.map((h) => (varX2 > 0 ? (pi[0] * pi[1] * (s2[0] - s2[1]) ** 2 * lam ** h) / varX2 : 0));
    const sojournPmf = runs.map((r) => {
      const counts = new Array<number>(MAX_SOJOURN + 1).fill(0);
      r.forEach((d) => {
        if (d <= MAX_SOJOURN) counts[d]++;
      });
      return counts.map((c) => (r.length ? c / r.length : 0));
    });
    const empMean = runs.map((r) => (r.length ? r.reduce((s, v) => s + v, 0) / r.length : NaN));
    return { runs, lags, acfX, acfX2, theory, lam, sojournPmf, empMean };
  }, [sim, params, stay, means, sds]);

  const colors = timeSeriesColors.states;
  const fullW = Math.max(280, containerW);
  const halfW = Math.max(260, isMobile ? fullW : (fullW - 16) / 2);

  // Series panel: clip the y-range so a single extreme draw doesn't flatten the plot.
  const yAbs = Math.max(2, Math.min(12, Math.max(...sim.obs.map(Math.abs))));
  const seriesTicks = niceTicks(-yAbs, yAbs, yAbs > 6 ? 4 : 2);

  const dGrid = Array.from({ length: MAX_SOJOURN }, (_, i) => i + 1);
  const geo = stay.map((a) => dGrid.map((d) => a ** (d - 1) * (1 - a)));
  const sojMax = Math.min(1, 1.1 * Math.max(...geo.flat(), ...stats.sojournPmf.flat()));

  const acfMax = Math.max(0.1, ...stats.acfX2, ...stats.theory) * 1.1;
  const acfMin = Math.min(0, ...stats.acfX, ...stats.acfX2) * 1.1;

  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-filtered)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {regimeSimulatorPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => selectPreset(i)}
            className={btn(i === presetIdx)}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="mb-3 grid grid-cols-1 items-end gap-3 sm:grid-cols-4">
        <Slider label="stay in state 1 (a₁₁)" value={stay[0]} min={0.2} max={0.995} step={0.005} onChange={(v) => setStay([v, stay[1]])} />
        <Slider label="stay in state 2 (a₂₂)" value={stay[1]} min={0.2} max={0.995} step={0.005} onChange={(v) => setStay([stay[0], v])} />
        <Slider label="length T" value={T} min={200} max={5000} step={100} onChange={(v) => setT(Math.round(v))} format={(v) => v.toFixed(0)} />
        <button
          onClick={() => setSeed((s) => s + 1)}
          className="rounded px-3 py-1.5 text-xs font-medium text-white"
          style={{ background: timeSeriesColors.filtered }}
        >
          Resample (seed {seed})
        </button>
      </div>

      <button
        onClick={() => setShowMore((v) => !v)}
        className="mb-2 text-xs underline"
        style={{ color: 'var(--color-text-muted)' }}
        aria-expanded={showMore}
      >
        {showMore ? 'Hide emission controls' : 'More controls: emission means and SDs'}
      </button>
      {showMore && (
        <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Slider label="mean μ₁" value={means[0]} min={-2} max={2} step={0.05} onChange={(v) => setMeans([v, means[1]])} format={(v) => v.toFixed(2)} />
          <Slider label="SD σ₁" value={sds[0]} min={0.2} max={4} step={0.05} onChange={(v) => setSds([v, sds[1]])} format={(v) => v.toFixed(2)} />
          <Slider label="mean μ₂" value={means[1]} min={-2} max={2} step={0.05} onChange={(v) => setMeans([means[0], v])} format={(v) => v.toFixed(2)} />
          <Slider label="SD σ₂" value={sds[1]} min={0.2} max={4} step={0.05} onChange={(v) => setSds([sds[0], v])} format={(v) => v.toFixed(2)} />
        </div>
      )}

      <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
        <span>π = ({params.initial[0].toFixed(3)}, {params.initial[1].toFixed(3)})</span>
        <span>λ = a₁₁ + a₂₂ − 1 = {stats.lam.toFixed(3)}</span>
        <span style={{ color: colors[0] }}>
          mean sojourn 1: {(1 / (1 - stay[0])).toFixed(1)} theory · {Number.isNaN(stats.empMean[0]) ? '—' : stats.empMean[0].toFixed(1)} simulated
        </span>
        <span style={{ color: colors[1] }}>
          mean sojourn 2: {(1 / (1 - stay[1])).toFixed(1)} theory · {Number.isNaN(stats.empMean[1]) ? '—' : stats.empMean[1].toFixed(1)} simulated
        </span>
      </div>

      {/* Charts render only after the container is measured: the SSR HTML and the first client render then match exactly, with no float-dependent attributes to mismatch. */}
      {width > 0 ? (
        <>
      <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        Simulated series; shading shows the hidden state (<span style={{ color: colors[0] }}>state 1</span> ·{' '}
        <span style={{ color: colors[1] }}>state 2</span>)
      </div>
      <Panel
        width={fullW}
        height={isMobile ? 170 : 190}
        label="Simulated observations with the hidden state shaded"
        xLabel="time t"
        yLabel="x_t"
        xDomain={[0, T]}
        yDomain={[-yAbs, yAbs]}
        xTicks={niceTicks(0, T, T > 2500 ? 1000 : 500)}
        yTicks={seriesTicks}
      >
        {(sx, sy, plotH) => {
          const segs: { a: number; b: number }[] = [];
          let start = 0;
          for (let t = 1; t <= T; t++) {
            if (t === T || sim.states[t] !== sim.states[start]) {
              if (sim.states[start] === 1) segs.push({ a: start, b: t });
              start = t;
            }
          }
          const ys = sim.obs.map((v) => Math.max(-yAbs, Math.min(yAbs, v)));
          return (
            <>
              {segs.map((s) => (
                <rect
                  key={s.a}
                  x={sx(s.a)}
                  y={0}
                  width={Math.max(0.5, sx(s.b) - sx(s.a))}
                  height={plotH}
                  fill={colors[1]}
                  opacity={timeSeriesColors.truthAlpha}
                />
              ))}
              <path
                d={linePath(ys.map((_, i) => i + 0.5), ys, sx, sy)}
                fill="none"
                stroke={timeSeriesColors.neutral}
                strokeWidth={0.6}
              />
            </>
          );
        }}
      </Panel>

      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Sojourn lengths: bars = simulated, lines = Geometric(1 − a_kk) (Thm 3)
          </div>
          <Panel
            width={halfW}
            height={220}
            label="Sojourn-length distributions against geometric probability mass functions"
            xLabel="sojourn length d"
            yLabel="probability"
            xDomain={[0, MAX_SOJOURN + 1]}
            yDomain={[0, sojMax]}
            xTicks={niceTicks(0, MAX_SOJOURN, 10)}
            yTicks={niceTicks(0, sojMax, sojMax > 0.5 ? 0.2 : sojMax > 0.2 ? 0.1 : sojMax > 0.08 ? 0.02 : 0.01)}
          >
            {(sx, sy) => {
              const bw = Math.max(0.6, (sx(1) - sx(0)) / 2 - 0.3);
              return (
                <>
                  {[0, 1].map((k) =>
                    dGrid.map((d) =>
                      stats.sojournPmf[k][d] > 0 ? (
                        <rect
                          key={`${k}-${d}`}
                          x={sx(d) + (k === 0 ? -bw : 0)}
                          y={sy(stats.sojournPmf[k][d])}
                          width={bw}
                          height={sy(0) - sy(stats.sojournPmf[k][d])}
                          fill={colors[k]}
                          opacity={0.45}
                        />
                      ) : null,
                    ),
                  )}
                  {[0, 1].map((k) => (
                    <path key={k} d={linePath(dGrid, geo[k], sx, sy)} fill="none" stroke={colors[k]} strokeWidth={2} />
                  ))}
                </>
              );
            }}
          </Panel>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Autocorrelation: <span style={{ color: colors[1] }}>■ squared obs.</span> ·{' '}
            <span style={{ color: timeSeriesColors.neutral }}>■ obs.</span> ·{' '}
            <span style={{ color: timeSeriesColors.viterbi }}>— Ex 5 theory for squares</span>
          </div>
          <Panel
            width={halfW}
            height={220}
            label="Autocorrelation of observations and squared observations with the closed-form theory curve"
            xLabel="lag h"
            yLabel="autocorrelation"
            xDomain={[0, MAX_LAG + 1]}
            yDomain={[acfMin, acfMax]}
            xTicks={niceTicks(0, MAX_LAG, 10)}
            yTicks={niceTicks(acfMin, acfMax, acfMax - acfMin > 0.6 ? 0.2 : 0.1)}
          >
            {(sx, sy) => {
              const bw = Math.max(0.8, (sx(1) - sx(0)) / 2 - 0.4);
              return (
                <>
                  <line x1={sx(0)} y1={sy(0)} x2={sx(MAX_LAG + 1)} y2={sy(0)} stroke="var(--color-text-muted)" />
                  {stats.lags.map((h, i) => (
                    <g key={h}>
                      <rect
                        x={sx(h) - bw}
                        y={Math.min(sy(stats.acfX[i]), sy(0))}
                        width={bw}
                        height={Math.abs(sy(stats.acfX[i]) - sy(0))}
                        fill={timeSeriesColors.neutral}
                        opacity={0.7}
                      />
                      <rect
                        x={sx(h)}
                        y={Math.min(sy(stats.acfX2[i]), sy(0))}
                        width={bw}
                        height={Math.abs(sy(stats.acfX2[i]) - sy(0))}
                        fill={colors[1]}
                        opacity={0.7}
                      />
                    </g>
                  ))}
                  <path d={linePath(stats.lags, stats.theory, sx, sy)} fill="none" stroke={timeSeriesColors.viterbi} strokeWidth={2} />
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
