/**
 * ViterbiDecoder — interactive for Topic 34 §34.6.
 *
 * Two decoders, two questions. Viterbi (Thm 8) returns the single most
 * probable path; pointwise posterior decoding returns the most probable state
 * at each time separately. On the casino the reader can click any roll to
 * cycle its face and watch both decoders respond. The "Forbidden transition"
 * scenario is Ex 10: with a₃₁ = 0, the pointwise path (3, 1, 1, 1, 1) has
 * probability 0, while Viterbi's path (1, 1, 1, 1, 1) is valid.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { createSeededRng } from './shared/bayes';
import { forwardBackward, pathLogProb, posteriorDecode, sampleHMM, viterbi } from './shared/timeseries';
import { timeSeriesColors } from './shared/colorScales';
import { forbiddenTransitionExample, viterbiPresets } from '../../data/time-series-data';

const LEFT = 70;
const MOBILE_BREAKPOINT = 640;
const MOBILE_WINDOW = 25;

type Scenario = 'casino' | 'forbidden';

function switches(z: number[]): number {
  let n = 0;
  for (let t = 1; t < z.length; t++) if (z[t] !== z[t - 1]) n++;
  return n;
}

function fmtLog(v: number): string {
  return Number.isFinite(v) ? v.toFixed(2) : '−∞ (impossible)';
}

function CasinoView({ width }: { width: number }) {
  const preset = viterbiPresets[0];
  const [seed, setSeed] = useState(preset.seed);
  const [edits, setEdits] = useState<Record<number, number>>({});
  const isMobile = width < MOBILE_BREAKPOINT;
  const colors = timeSeriesColors.states;

  const sim = useMemo(() => sampleHMM(preset.params, preset.T, createSeededRng(seed)), [preset, seed]);
  const obs = useMemo(() => sim.obs.map((v, t) => edits[t] ?? v), [sim, edits]);
  const T = obs.length;

  const dec = useMemo(() => {
    const fb = forwardBackward(preset.params, obs);
    const v = viterbi(preset.params, obs);
    const post = posteriorDecode(fb.gamma);
    return {
      gamma: fb.gamma,
      logLik: fb.logLik,
      vit: v.path,
      vitLog: v.logProb - fb.logLik,
      post,
      postLog: pathLogProb(preset.params, obs, post) - fb.logLik,
    };
  }, [preset, obs]);

  const acc = (z: number[]) => z.filter((k, t) => k === sim.states[t]).length / T;

  // Window around the first Viterbi/posterior disagreement on narrow screens.
  const firstDis = dec.vit.findIndex((k, t) => k !== dec.post[t]);
  const center = firstDis >= 0 ? firstDis : Math.floor(T / 2);
  const lo = isMobile ? Math.max(0, Math.min(center - MOBILE_WINDOW, T - 2 * MOBILE_WINDOW - 1)) : 0;
  const hi = isMobile ? Math.min(T, lo + 2 * MOBILE_WINDOW + 1) : T;
  const nVis = hi - lo;
  const plotW = Math.max(200, width - LEFT - 8);
  const cw = plotW / nVis;
  const xOf = (t: number) => LEFT + (t - lo) * cw;

  const rollH = 26;
  const ribbonTop = rollH + 14;
  const ribbonH = 16;
  const ribbons: { name: string; z: number[] }[] = [
    { name: 'truth', z: sim.states },
    { name: 'Viterbi', z: dec.vit },
    { name: 'posterior', z: dec.post },
  ];
  const probTop = ribbonTop + ribbons.length * (ribbonH + 6) + 12;
  const probH = 80;
  const svgH = probTop + probH + 22;
  const py = (v: number) => probTop + probH - v * probH;

  const flip = (t: number) => setEdits((e) => ({ ...e, [t]: ((e[t] ?? sim.obs[t]) + 1) % 6 }));

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <button
          onClick={() => {
            setSeed((s) => s + 1);
            setEdits({});
          }}
          className="rounded px-3 py-1.5 font-medium text-white"
          style={{ background: timeSeriesColors.filtered }}
        >
          Resample (seed {seed})
        </button>
        <button
          onClick={() => setEdits({})}
          disabled={Object.keys(edits).length === 0}
          className="rounded border px-3 py-1.5 disabled:opacity-40"
          style={{ borderColor: 'var(--color-border)' }}
        >
          Undo edits ({Object.keys(edits).length})
        </button>
        <span style={{ color: 'var(--color-text-muted)' }}>Click a roll to cycle its face (tall red ticks are sixes).</span>
      </div>
      <div className="overflow-hidden">
        <svg width={LEFT + plotW + 8} height={svgH} className="block" role="img" aria-label="Casino rolls with true, Viterbi, and posterior-decoded state ribbons">
          <text x={0} y={rollH / 2 + 4} fontSize={10} fill="var(--color-text-muted)">
            rolls
          </text>
          {Array.from({ length: nVis }, (_, i) => {
            const t = lo + i;
            const six = obs[t] === 5;
            return (
              <g key={`r-${t}`} onClick={() => flip(t)} style={{ cursor: 'pointer' }}>
                <rect x={xOf(t)} y={0} width={cw} height={rollH} fill="transparent" />
                <line
                  x1={xOf(t) + cw / 2}
                  x2={xOf(t) + cw / 2}
                  y1={six ? 1 : rollH * 0.3}
                  y2={six ? rollH - 1 : rollH * 0.7}
                  stroke={edits[t] !== undefined ? timeSeriesColors.filtered : six ? colors[1] : timeSeriesColors.neutral}
                  strokeWidth={six ? 1.8 : 1}
                />
                <title>{`roll ${t + 1}: face ${obs[t] + 1}`}</title>
              </g>
            );
          })}
          {ribbons.map((r, i) => {
            const y = ribbonTop + i * (ribbonH + 6);
            return (
              <g key={r.name}>
                <text x={0} y={y + ribbonH / 2 + 4} fontSize={10} fill="var(--color-text)">
                  {r.name}
                </text>
                {Array.from({ length: nVis }, (_, j) => {
                  const t = lo + j;
                  const wrong = i > 0 && r.z[t] !== sim.states[t];
                  return (
                    <g key={`${r.name}-${t}`}>
                      <rect x={xOf(t)} y={y} width={cw + 0.3} height={ribbonH} fill={colors[r.z[t]]} opacity={0.85} />
                      {wrong && <rect x={xOf(t)} y={y + ribbonH - 3} width={cw + 0.3} height={3} fill={timeSeriesColors.viterbi} />}
                    </g>
                  );
                })}
              </g>
            );
          })}
          <text x={0} y={probTop + 10} fontSize={10} fill="var(--color-text-muted)">
            γₜ(loaded)
          </text>
          <line x1={LEFT} x2={LEFT + plotW} y1={py(0.5)} y2={py(0.5)} stroke="var(--color-viz-grid)" strokeDasharray="3 3" />
          <line x1={LEFT} x2={LEFT + plotW} y1={py(0)} y2={py(0)} stroke="var(--color-text-muted)" />
          <path
            d={Array.from({ length: nVis }, (_, i) => {
              const t = lo + i;
              return `${i === 0 ? 'M' : 'L'}${(xOf(t) + cw / 2).toFixed(2)},${py(dec.gamma[t][1]).toFixed(2)}`;
            }).join('')}
            fill="none"
            stroke={timeSeriesColors.smoothed}
            strokeWidth={1.8}
          />
          <text x={LEFT} y={svgH - 6} fontSize={10} fill="var(--color-text-muted)">
            t = {lo + 1}
          </text>
          <text x={LEFT + plotW} y={svgH - 6} fontSize={10} textAnchor="end" fill="var(--color-text-muted)">
            t = {hi}
          </text>
        </svg>
      </div>
      <div className="mt-2 grid grid-cols-1 gap-1 font-mono text-xs sm:grid-cols-2">
        <span>
          Viterbi: accuracy {(100 * acc(dec.vit)).toFixed(1)}% · {switches(dec.vit)} switches · log p(z* | x) = {fmtLog(dec.vitLog)}
        </span>
        <span>
          posterior: accuracy {(100 * acc(dec.post)).toFixed(1)}% · {switches(dec.post)} switches · log p(ẑ | x) = {fmtLog(dec.postLog)}
        </span>
        <span style={{ color: 'var(--color-text-muted)' }}>truth: {switches(sim.states)} switches · dark underline = decoding error</span>
      </div>
    </>
  );
}

function ForbiddenView({ width }: { width: number }) {
  const { params, obs } = forbiddenTransitionExample;
  const colors = timeSeriesColors.states;
  const dec = useMemo(() => {
    const fb = forwardBackward(params, obs);
    const v = viterbi(params, obs);
    const post = posteriorDecode(fb.gamma);
    return { gamma: fb.gamma, vit: v.path, vitLog: v.logProb - fb.logLik, post, postLog: pathLogProb(params, obs, post) - fb.logLik };
  }, [params, obs]);
  const T = obs.length;
  const K = params.initial.length;
  const w = Math.max(280, Math.min(width, 640));
  const h = 230;
  const x = (t: number) => 50 + (t * (w - 90)) / (T - 1);
  const y = (k: number) => 30 + ((K - 1 - k) * (h - 80)) / (K - 1);
  const r = width < MOBILE_BREAKPOINT ? 15 : 19;
  const forbidden = (a: number, b: number) => params.transition[a][b] === 0;

  return (
    <>
      <p className="mb-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        A = [[0.9, 0.1, 0], [0.5, 0.2, 0.3], [0, 0.2, 0.8]] (so a₁₃ = a₃₁ = 0); P(x = 1 | state) = 0.1, 0.2, 0.1; uniform start; x = (0, 1, 1, 1, 0).
        Circle numbers are γₜ(k).
      </p>
      <svg width={w} height={h} className="block" role="img" aria-label="Three-state trellis where pointwise posterior decoding uses a forbidden transition">
        {dec.vit.slice(1).map((k, i) => (
          <line key={`v-${i}`} x1={x(i)} y1={y(dec.vit[i])} x2={x(i + 1)} y2={y(k)} stroke={timeSeriesColors.viterbi} strokeWidth={2.5} />
        ))}
        {dec.post.slice(1).map((k, i) => {
          const bad = forbidden(dec.post[i], k);
          return (
            <line
              key={`p-${i}`}
              x1={x(i)}
              y1={y(dec.post[i]) - 5}
              x2={x(i + 1)}
              y2={y(k) - 5}
              stroke={bad ? colors[1] : timeSeriesColors.filtered}
              strokeWidth={2.5}
              strokeDasharray="6 4"
            />
          );
        })}
        {Array.from({ length: T }, (_, t) =>
          Array.from({ length: K }, (__, k) => (
            <g key={`${t}-${k}`}>
              <circle cx={x(t)} cy={y(k)} r={r} fill={colors[k]} opacity={0.3 + 0.7 * dec.gamma[t][k]} stroke="var(--color-surface)" />
              <text x={x(t)} y={y(k) + 3.5} textAnchor="middle" fontSize={10} fill="var(--color-text)">
                {dec.gamma[t][k].toFixed(2)}
              </text>
            </g>
          )),
        )}
        {Array.from({ length: K }, (_, k) => (
          <text key={`s-${k}`} x={4} y={y(k) + 4} fontSize={10} fill="var(--color-text)">
            {k + 1}
          </text>
        ))}
        {obs.map((v, t) => (
          <text key={`x-${t}`} x={x(t)} y={h - 16} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">
            t={t + 1}, x={v}
          </text>
        ))}
      </svg>
      <div className="mt-2 grid grid-cols-1 gap-1 font-mono text-xs sm:grid-cols-2">
        <span>
          <span style={{ color: timeSeriesColors.viterbi }}>— Viterbi</span> z* = ({dec.vit.map((k) => k + 1).join(', ')}) · p(z* | x) ={' '}
          {Math.exp(dec.vitLog).toFixed(3)}
        </span>
        <span>
          <span style={{ color: timeSeriesColors.filtered }}>- - posterior</span> ẑ = ({dec.post.map((k) => k + 1).join(', ')}) · log p(ẑ | x) ={' '}
          {fmtLog(dec.postLog)}
        </span>
        <span style={{ color: colors[1] }}>red dashes: a₃₁ = 0, so this path has probability 0.</span>
      </div>
    </>
  );
}

export default function ViterbiDecoder() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [scenario, setScenario] = useState<Scenario>('casino');
  const containerW = width || 800;
  const btn = (active: boolean) =>
    `rounded px-3 py-1 text-xs transition ${
      active ? 'bg-[var(--color-ts-filtered)] text-white' : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
    }`;
  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Scenario:</span>
        <button onClick={() => setScenario('casino')} className={btn(scenario === 'casino')} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
          Casino
        </button>
        <button onClick={() => setScenario('forbidden')} className={btn(scenario === 'forbidden')} style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}>
          Forbidden transition (Ex 10)
        </button>
      </div>
      {/* Charts render only after the container is measured: the SSR HTML and the first client render then match exactly, with no float-dependent attributes to mismatch. */}
      {width === 0 ? (
        <div style={{ height: 300 }} aria-hidden="true" />
      ) : scenario === 'casino' ? (
        <CasinoView width={containerW} />
      ) : (
        <ForbiddenView width={containerW} />
      )}
    </div>
  );
}
