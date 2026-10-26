/**
 * GarchVsHmmSimulator — interactive for Topic 35 §35.9.
 *
 * Two mechanisms, one fingerprint. GARCH(1,1) (Def 11) and Topic 34's
 * two-regime HMM both produce returns that are white noise with
 * autocorrelated squares, but volatility persists differently: GARCH's
 * squared-return ACF decays at α + β from lag 1 (Ex 18), the HMM's at
 * λ = a₁₁ + a₂₂ − 1 (Topic 34 Ex 5). The β slider is capped at 0.999 − α,
 * which enforces Thm 15's α + β < 1. "Fit GARCH(1,1) to this path" runs the
 * Gaussian QMLE; on HMM paths the fitted persistence α̂ + β̂ lands near 1 —
 * regime shifts masquerading as near-integrated GARCH (Ex 19, LAM1990).
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import {
  garchFourthMomentFinite,
  garchQMLE,
  garchSqACF,
  sampleACF,
  sampleHMM,
  simulateGARCH,
  stationaryDistribution,
  type GarchFit,
  type GarchParams,
  type HMMParams,
} from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { garchPresets, marketParams } from '../../data/time-series-data';

const MARGIN = { top: 10, right: 12, bottom: 34, left: 46 };
const MOBILE_BREAKPOINT = 640;
const MAX_LAG = 40;

type Model = 'garch' | 'hmm';
type View = Model | 'both';

const marketStay: [number, number] = [marketParams.transition[0][0], marketParams.transition[1][1]];
const marketSds: [number, number] =
  marketParams.emission.kind === 'gaussian' ? [marketParams.emission.sds[0], marketParams.emission.sds[1]] : [0.7, 2];
const marketMeans: [number, number] =
  marketParams.emission.kind === 'gaussian' ? [marketParams.emission.means[0], marketParams.emission.means[1]] : [0.05, -0.1];

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
  const plotW = Math.max(10, width - MARGIN.left - MARGIN.right);
  const plotH = Math.max(10, height - MARGIN.top - MARGIN.bottom);
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
        className="w-full accent-[var(--color-ts-garch)]"
      />
    </label>
  );
}

const halfLife = (rate: number) => (rate > 0 && rate < 1 ? Math.log(0.5) / Math.log(rate) : Infinity);
const fmtHalfLife = (h: number) => (Number.isFinite(h) ? `${h.toFixed(1)} steps` : '∞');

interface Sim {
  x: number[];
  /** σ_t for GARCH; null for the HMM. */
  sigma: number[] | null;
  /** Hidden states for the HMM; null for GARCH. */
  states: number[] | null;
  acfX: number[];
  acfX2: number[];
  theory: number[] | null;
}

export default function GarchVsHmmSimulator() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const def = garchPresets[0];
  const [view, setView] = useState<View>('garch');
  const [presetId, setPresetId] = useState(def.id);
  const [g, setG] = useState<GarchParams>(def.params!);
  const [stay, setStay] = useState<[number, number]>(marketStay);
  const [sds, setSds] = useState<[number, number]>(marketSds);
  const [T, setT] = useState(def.T);
  const [seed, setSeed] = useState(def.seed);
  const [fits, setFits] = useState<Partial<Record<Model, { key: string; fit: GarchFit }>>>({});
  const [fitting, setFitting] = useState<Model | null>(null);
  const [openParams, setOpenParams] = useState<Record<Model, boolean>>({ garch: true, hmm: true });

  const containerW = width || 800;
  const isMobile = containerW < MOBILE_BREAKPOINT;

  const applyPreset = (id: string) => {
    const p = garchPresets.find((q) => q.id === id) ?? def;
    setPresetId(p.id);
    setT(p.T);
    setSeed(p.seed);
    if (p.params) {
      setG(p.params);
      if (view === 'hmm') setView('garch');
    } else {
      setStay(marketStay);
      setSds(marketSds);
      if (view === 'garch') setView('hmm');
    }
  };

  const setGarch = (patch: Partial<GarchParams>) => {
    setPresetId('');
    setG((cur) => {
      const next = { ...cur, ...patch };
      next.beta = Math.min(next.beta, 0.999 - next.alpha);
      return next;
    });
  };

  const hmmParams: HMMParams = useMemo(() => {
    const A = [
      [stay[0], 1 - stay[0]],
      [1 - stay[1], stay[1]],
    ];
    return { initial: stationaryDistribution(A), transition: A, emission: { kind: 'gaussian', means: [...marketMeans], sds: [...sds] } };
  }, [stay, sds]);

  const garchSim: Sim = useMemo(() => {
    const { x, sigma2 } = simulateGARCH(g, T, createSeededRng(seed));
    const finite = garchFourthMomentFinite(g.alpha, g.beta);
    return {
      x,
      sigma: sigma2.map(Math.sqrt),
      states: null,
      acfX: sampleACF(x, MAX_LAG),
      acfX2: sampleACF(x.map((v) => v * v), MAX_LAG),
      theory: finite ? garchSqACF(g.alpha, g.beta, MAX_LAG) : null,
    };
  }, [g, T, seed]);

  const hmmSim: Sim = useMemo(() => {
    const { states, obs } = sampleHMM(hmmParams, T, createSeededRng(seed));
    const pi = hmmParams.initial;
    const lam = stay[0] + stay[1] - 1;
    const s2 = marketMeans.map((m, k) => m * m + sds[k] * sds[k]);
    const m4 = marketMeans.map((m, k) => m ** 4 + 6 * m * m * sds[k] ** 2 + 3 * sds[k] ** 4);
    const varX2 = pi[0] * m4[0] + pi[1] * m4[1] - (pi[0] * s2[0] + pi[1] * s2[1]) ** 2;
    const theory = [1, ...Array.from({ length: MAX_LAG }, (_, i) => (varX2 > 0 ? (pi[0] * pi[1] * (s2[0] - s2[1]) ** 2 * lam ** (i + 1)) / varX2 : 0))];
    return {
      x: obs,
      sigma: null,
      states,
      acfX: sampleACF(obs, MAX_LAG),
      acfX2: sampleACF(obs.map((v) => v * v), MAX_LAG),
      theory,
    };
  }, [hmmParams, T, seed, stay, sds]);

  const pathKey = (model: Model) =>
    model === 'garch' ? `g|${g.omega}|${g.alpha}|${g.beta}|${T}|${seed}` : `h|${stay}|${sds}|${T}|${seed}`;

  const runFit = (model: Model) => {
    setFitting(model);
    const x = model === 'garch' ? garchSim.x : hmmSim.x;
    const key = pathKey(model);
    // Defer so the spinner paints before the optimizer runs.
    setTimeout(() => {
      const fit = garchQMLE(x);
      setFits((f) => ({ ...f, [model]: { key, fit } }));
      setFitting(null);
    }, 30);
  };

  const models: Model[] = view === 'both' ? ['garch', 'hmm'] : [view];
  const colW = view === 'both' && !isMobile ? (containerW - 16) / 2 : containerW;
  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-garch)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;

  const persistence = g.alpha + g.beta;
  const lam = stay[0] + stay[1] - 1;
  const kurtCond = 3 * g.alpha ** 2 + 2 * g.alpha * g.beta + g.beta ** 2;

  const renderModel = (model: Model) => {
    const sim = model === 'garch' ? garchSim : hmmSim;
    const color = model === 'garch' ? timeSeriesColors.garch : timeSeriesColors.states[0];
    const fitEntry = fits[model];
    const fit = fitEntry && fitEntry.key === pathKey(model) ? fitEntry.fit : null;
    const yAbs = Math.max(2, Math.min(15, Math.max(...sim.x.map(Math.abs))));
    const ys = sim.x.map((v) => Math.max(-yAbs, Math.min(yAbs, v)));
    const lags = Array.from({ length: MAX_LAG }, (_, i) => i + 1);
    const acfVals = [...sim.acfX.slice(1), ...sim.acfX2.slice(1), ...(sim.theory ? sim.theory.slice(1) : [])];
    const acfMax = Math.max(0.1, ...acfVals) * 1.1;
    const acfMin = Math.min(-0.1, ...acfVals) * 1.1;
    const band = 1.96 / Math.sqrt(T);
    return (
      <div key={model} className="min-w-0">
        <div className="mb-2 text-sm font-semibold" style={{ color }}>
          {model === 'garch' ? 'GARCH(1,1): volatility from feedback' : 'Two-regime HMM (Topic 34): volatility from a hidden state'}
        </div>
        <button
          onClick={() => setOpenParams((o) => ({ ...o, [model]: !o[model] }))}
          className="mb-2 text-xs underline"
          aria-expanded={openParams[model]}
          style={{ color: 'var(--color-text-muted)' }}
        >
          {openParams[model] ? 'Hide parameters' : 'Parameters'}
        </button>
        {openParams[model] &&
          (model === 'garch' ? (
            <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Slider label="ω (intercept)" value={g.omega} min={0.01} max={0.5} step={0.005} onChange={(v) => setGarch({ omega: v })} />
              <Slider label="α (ARCH)" value={g.alpha} min={0} max={0.4} step={0.005} onChange={(v) => setGarch({ alpha: v })} />
              <Slider label={`β (GARCH, max ${(0.999 - g.alpha).toFixed(3)})`} value={g.beta} min={0} max={0.99} step={0.005} onChange={(v) => setGarch({ beta: v })} />
            </div>
          ) : (
            <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Slider label="stay calm a₁₁" value={stay[0]} min={0.5} max={0.995} step={0.005} onChange={(v) => { setPresetId(''); setStay([v, stay[1]]); }} />
              <Slider label="stay turbulent a₂₂" value={stay[1]} min={0.5} max={0.995} step={0.005} onChange={(v) => { setPresetId(''); setStay([stay[0], v]); }} />
              <Slider label="SD calm σ₁" value={sds[0]} min={0.2} max={3} step={0.05} onChange={(v) => { setPresetId(''); setSds([v, sds[1]]); }} format={(v) => v.toFixed(2)} />
              <Slider label="SD turbulent σ₂" value={sds[1]} min={0.2} max={4} step={0.05} onChange={(v) => { setPresetId(''); setSds([sds[0], v]); }} format={(v) => v.toFixed(2)} />
            </div>
          ))}

        <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs">
          {model === 'garch' ? (
            <>
              <span>Var = ω/(1 − α − β) = {(g.omega / (1 - persistence)).toFixed(3)}</span>
              <span>persistence α + β = {persistence.toFixed(3)}</span>
              <span>half-life = {fmtHalfLife(halfLife(persistence))}</span>
              <span style={{ color: kurtCond < 1 ? undefined : timeSeriesColors.warn }}>
                3α² + 2αβ + β² = {kurtCond.toFixed(3)} {kurtCond < 1 ? '< 1' : '≥ 1'}
              </span>
            </>
          ) : (
            <>
              <span>π = ({hmmParams.initial[0].toFixed(3)}, {hmmParams.initial[1].toFixed(3)})</span>
              <span>λ = a₁₁ + a₂₂ − 1 = {lam.toFixed(3)}</span>
              <span>regime half-life = {fmtHalfLife(halfLife(lam))}</span>
            </>
          )}
        </div>

        <div className="mb-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          {model === 'garch' ? (
            <>
              Returns x_t with <span style={{ color }}>±σ_t band</span>
            </>
          ) : (
            <>
              Returns x_t; <span style={{ color: timeSeriesColors.states[1] }}>shading = turbulent state</span>
            </>
          )}
        </div>
        <Panel
          width={colW}
          height={isMobile ? 160 : 180}
          label={model === 'garch' ? 'Simulated GARCH returns with the conditional standard deviation band' : 'Simulated HMM returns with turbulent days shaded'}
          xLabel="time t"
          yLabel="x_t"
          xDomain={[0, T]}
          yDomain={[-yAbs, yAbs]}
          xTicks={niceTicks(0, T, T > 2500 ? 1000 : 250)}
          yTicks={niceTicks(-yAbs, yAbs, yAbs > 8 ? 5 : 2)}
        >
          {(sx, sy, plotH) => {
            if (sim.sigma) {
              const s = sim.sigma.map((v) => Math.min(v, yAbs));
              const idx = s.map((_, i) => i + 0.5);
              const up = linePath(idx, s, sx, sy);
              const down = idx
                .slice()
                .reverse()
                .map((t, k) => `L${sx(t).toFixed(2)},${sy(-s[s.length - 1 - k]).toFixed(2)}`)
                .join('');
              return (
                <>
                  <path d={`${up}${down}Z`} fill={color} opacity={0.18} />
                  <path d={linePath(idx, ys, sx, sy)} fill="none" stroke={timeSeriesColors.neutral} strokeWidth={0.6} />
                </>
              );
            }
            const segs: { a: number; b: number }[] = [];
            let start = 0;
            const st = sim.states!;
            for (let t = 1; t <= T; t++) {
              if (t === T || st[t] !== st[start]) {
                if (st[start] === 1) segs.push({ a: start, b: t });
                start = t;
              }
            }
            return (
              <>
                {segs.map((sg) => (
                  <rect key={sg.a} x={sx(sg.a)} y={0} width={Math.max(0.5, sx(sg.b) - sx(sg.a))} height={plotH} fill={timeSeriesColors.states[1]} opacity={timeSeriesColors.truthAlpha} />
                ))}
                <path d={linePath(ys.map((_, i) => i + 0.5), ys, sx, sy)} fill="none" stroke={timeSeriesColors.neutral} strokeWidth={0.6} />
              </>
            );
          }}
        </Panel>

        <div className="mb-1 mt-3 flex flex-wrap items-center gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <span>
            ACF: <span style={{ color: timeSeriesColors.raw }}>■ returns</span> · <span style={{ color }}>■ squared returns</span>
            {sim.theory && (
              <span style={{ color: timeSeriesColors.truth }}> · - - theory for squares ({model === 'garch' ? 'Ex 18' : 'Topic 34 Ex 5'})</span>
            )}
          </span>
          {model === 'garch' && !sim.theory && (
            <span className="rounded px-2 py-0.5 font-semibold text-white" style={{ background: timeSeriesColors.warn }}>
              fourth moment infinite — the squared-return ACF is undefined; sample values are unreliable
            </span>
          )}
        </div>
        <Panel
          width={colW}
          height={210}
          label="Sample autocorrelation of returns and squared returns with the theoretical curve"
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
                <rect x={sx(0)} y={sy(band)} width={sx(MAX_LAG + 1) - sx(0)} height={sy(-band) - sy(band)} fill={timeSeriesColors.raw} opacity={0.15} />
                <line x1={sx(0)} y1={sy(0)} x2={sx(MAX_LAG + 1)} y2={sy(0)} stroke="var(--color-text-muted)" />
                {lags.map((h) => (
                  <g key={h}>
                    <rect x={sx(h) - bw} y={Math.min(sy(sim.acfX[h]), sy(0))} width={bw} height={Math.abs(sy(sim.acfX[h]) - sy(0))} fill={timeSeriesColors.raw} opacity={0.85} />
                    <rect x={sx(h)} y={Math.min(sy(sim.acfX2[h]), sy(0))} width={bw} height={Math.abs(sy(sim.acfX2[h]) - sy(0))} fill={color} opacity={0.75} />
                  </g>
                ))}
                {sim.theory && <path d={linePath(lags, sim.theory.slice(1), sx, sy)} fill="none" stroke={timeSeriesColors.truth} strokeWidth={2} strokeDasharray="6 4" />}
              </>
            );
          }}
        </Panel>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            onClick={() => runFit(model)}
            disabled={fitting !== null}
            className="rounded px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
            style={{ background: timeSeriesColors.garch }}
          >
            {fitting === model ? 'Fitting…' : 'Fit GARCH(1,1) to this path'}
          </button>
          {fit && (
            <span className="font-mono text-xs" aria-live="polite">
              ω̂ = {fit.omega.toFixed(4)}, α̂ = {fit.alpha.toFixed(4)}, β̂ = {fit.beta.toFixed(4)} · α̂ + β̂ ={' '}
              <strong>{(fit.alpha + fit.beta).toFixed(4)}</strong> · half-life {fmtHalfLife(halfLife(fit.alpha + fit.beta))}
              {model === 'hmm' && <> (true regime half-life {fmtHalfLife(halfLife(lam))})</>}
            </span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Model:</span>
        {(
          [
            ['garch', 'GARCH(1,1)'],
            ['hmm', 'HMM (Topic 34)'],
            ['both', 'Side by side'],
          ] as [View, string][]
        ).map(([v, label]) => (
          <button key={v} onClick={() => setView(v)} className={btn(view === v)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {label}
          </button>
        ))}
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {garchPresets.map((p) => (
          <button key={p.id} onClick={() => applyPreset(p.id)} className={btn(presetId === p.id)} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="mb-4 grid grid-cols-1 items-end gap-3 sm:grid-cols-3">
        <Slider label="length T" value={T} min={500} max={5000} step={100} onChange={(v) => setT(Math.round(v))} format={(v) => v.toFixed(0)} />
        <label className="flex items-center gap-2 text-xs">
          seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
            className="w-24 rounded border px-1 py-0.5 font-mono"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
          />
        </label>
        <button onClick={() => setSeed((s) => s + 1)} className="rounded px-3 py-1.5 text-xs font-medium text-white" style={{ background: timeSeriesColors.garch }}>
          Resample
        </button>
      </div>

      {width > 0 ? (
        <div className={view === 'both' && !isMobile ? 'grid grid-cols-2 gap-4' : 'flex flex-col gap-6'}>{models.map(renderModel)}</div>
      ) : (
        <div style={{ height: 560 }} aria-hidden="true" />
      )}
    </div>
  );
}
