/**
 * PriorSensitivityComparator — §25.7 Ex 7 interactive. Same data, three
 * contrasting priors, three overlaid posteriors. The "sensitivity
 * magnitude" readout quantifies how much the posterior shifts with the
 * prior — it decays toward zero as n grows (BvM in action).
 *
 * Optional 4th component per brief §5.5; kept compact.
 */
import { useState, useMemo } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { pdfBeta, posterior, posteriorMean, credibleIntervalBeta, type PosteriorHyperparams } from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import { prior3WayPresets } from '../../data/bayesian-foundations-data';

const N_GRID = 300;
const MARGIN = { top: 14, right: 18, bottom: 34, left: 50 };

type PriorSpec = { alpha: number; beta: number; label: string };

export default function PriorSensitivityComparator() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [n, setN] = useState(50);
  const [k, setK] = useState(10);
  const [p1, setP1] = useState<PriorSpec>({ alpha: 2, beta: 8, label: 'Informative (mean 0.2)' });
  const [p2, setP2] = useState<PriorSpec>({ alpha: 2, beta: 2, label: 'Weakly informative' });
  const p3: PriorSpec = { alpha: 0.5, beta: 0.5, label: 'Jeffreys (non-informative)' };

  const isMobile = (width || 800) < 640;
  const chartW = Math.max(300, (width || 600) - 16);
  const chartH = isMobile ? 240 : 300;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // Three posteriors for the three priors.
  const posteriors = useMemo(() => {
    return [p1, p2, p3].map(pr => {
      const post = posterior(
        'beta-binomial',
        { family: 'beta-binomial', alpha0: pr.alpha, beta0: pr.beta },
        { family: 'beta-binomial', n, k },
      ) as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;
      return { prior: pr, post };
    });
  }, [p1, p2, n, k]);

  // Density grids.
  const { xs, curves, yMax } = useMemo(() => {
    const xs_ = new Array<number>(N_GRID + 1);
    for (let i = 0; i <= N_GRID; i++) xs_[i] = 0.001 + (i / N_GRID) * 0.998;
    const curves_ = posteriors.map(({ post }) =>
      xs_.map(x => pdfBeta(x, post.alpha0, post.beta0)),
    );
    let m = 0;
    for (const c of curves_) for (const y of c) if (y > m) m = y;
    return { xs: xs_, curves: curves_, yMax: Math.max(m, 0.5) };
  }, [posteriors]);

  // Sensitivity magnitude = max |p₁(θ) − pⱼ(θ)| over θ, max across pairs.
  const sensitivityMagnitude = useMemo(() => {
    let max = 0;
    for (let i = 0; i <= N_GRID; i++) {
      for (let a = 0; a < curves.length; a++) {
        for (let b = a + 1; b < curves.length; b++) {
          const diff = Math.abs(curves[a][i] - curves[b][i]);
          if (diff > max) max = diff;
        }
      }
    }
    return max;
  }, [curves]);

  const x2px = (x: number) => MARGIN.left + x * plotW;
  const y2px = (y: number) => MARGIN.top + plotH - (y / yMax) * plotH;
  const path = (ys: number[]) => ys
    .map((y, i) => Number.isFinite(y) ? `${i === 0 ? 'M' : 'L'}${x2px(xs[i]).toFixed(2)},${y2px(y).toFixed(2)}` : '')
    .filter(s => s).join(' ');

  // Three distinct colors for the three posteriors (prior/posterior/violet-darker).
  const palette = [
    bayesianColors.likelihood, // informative — amber
    bayesianColors.posterior,  // weak — violet
    bayesianColors.prior,      // Jeffreys — blue
  ];
  const dashes = ['none', 'none', '6 3'];

  return (
    <div ref={ref} className="my-6 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm">
      {/* Preset row */}
      <div className="mb-3 flex flex-wrap items-center gap-1">
        <span className="mr-1 text-xs text-[var(--color-text-muted)]">Preset:</span>
        {prior3WayPresets.map((p, i) => (
          <button
            key={i}
            type="button"
            onClick={() => {
              setN(p.n); setK(p.k);
              setP1({ alpha: p.prior1.alpha, beta: p.prior1.beta, label: p.prior1.label });
              setP2({ alpha: p.prior2.alpha, beta: p.prior2.beta, label: p.prior2.label });
            }}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-2 py-0.5 text-xs hover:bg-[var(--color-surface)]"
            title={p.description}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        <div className="flex flex-col gap-2 rounded-lg bg-[var(--color-surface-alt)] p-3 text-xs">
          <div className="font-semibold">Data</div>
          <Slider label={`n: ${n}`} min={1} max={1000} step={1} value={n} onChange={v => { setN(v); if (k > v) setK(v); }} />
          <Slider label={`k: ${k}`} min={0} max={n} step={1} value={k} onChange={setK} />

          <div className="mt-2 border-t border-[var(--color-border)] pt-2 font-semibold">
            Prior 1 (informative)
          </div>
          <Slider label={`α: ${p1.alpha.toFixed(1)}`} min={0.5} max={20} step={0.1} value={p1.alpha} onChange={v => setP1({ ...p1, alpha: v })} />
          <Slider label={`β: ${p1.beta.toFixed(1)}`} min={0.5} max={20} step={0.1} value={p1.beta} onChange={v => setP1({ ...p1, beta: v })} />

          <div className="mt-2 border-t border-[var(--color-border)] pt-2 font-semibold">
            Prior 2 (weak)
          </div>
          <Slider label={`α: ${p2.alpha.toFixed(1)}`} min={0.5} max={5} step={0.1} value={p2.alpha} onChange={v => setP2({ ...p2, alpha: v })} />
          <Slider label={`β: ${p2.beta.toFixed(1)}`} min={0.5} max={5} step={0.1} value={p2.beta} onChange={v => setP2({ ...p2, beta: v })} />

          <div className="mt-2 border-t border-[var(--color-border)] pt-2 text-[var(--color-text-muted)]">
            Prior 3: Jeffreys Beta(½, ½) — fixed reference.
          </div>
        </div>

        <div>
          <svg width={chartW} height={chartH} role="img" aria-label="Three posteriors under contrasting priors">
            <g stroke="var(--color-viz-grid)" strokeWidth={0.5}>
              {[0, 0.25, 0.5, 0.75, 1].map(v => (
                <line key={v} x1={x2px(v)} y1={MARGIN.top} x2={x2px(v)} y2={MARGIN.top + plotH} />
              ))}
            </g>
            <g stroke="var(--color-text-muted)" strokeWidth={1} fontSize={11} fill="var(--color-text-muted)">
              <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={MARGIN.left + plotW} y2={MARGIN.top + plotH} />
              <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} />
              {[0, 0.25, 0.5, 0.75, 1].map(v => (
                <text key={v} x={x2px(v)} y={MARGIN.top + plotH + 16} textAnchor="middle">{v.toFixed(2)}</text>
              ))}
              <text x={MARGIN.left + plotW / 2} y={chartH - 2} textAnchor="middle" fontSize={11}>θ</text>
            </g>
            {/* MLE reference line */}
            {n > 0 && (
              <line
                x1={x2px(k / n)} y1={MARGIN.top}
                x2={x2px(k / n)} y2={MARGIN.top + plotH}
                stroke={bayesianColors.mle}
                strokeWidth={1}
                strokeDasharray="3 3"
                opacity={0.7}
              />
            )}
            {curves.map((c, i) => (
              <path
                key={i}
                d={path(c)}
                fill="none"
                stroke={palette[i]}
                strokeWidth={2.2}
                strokeDasharray={dashes[i]}
                opacity={0.9}
              />
            ))}
            <g fontSize={11} transform={`translate(${MARGIN.left + 6}, ${MARGIN.top + 4})`}>
              {posteriors.map((d, i) => (
                <g key={i} transform={`translate(0, ${i * 16})`}>
                  <line x1={0} y1={6} x2={18} y2={6} stroke={palette[i]} strokeWidth={2.2} strokeDasharray={dashes[i]} />
                  <text x={22} y={9} fill="var(--color-text)">{d.prior.label}</text>
                </g>
              ))}
            </g>
          </svg>

          <div className="mt-3 text-sm">
            <div className="grid grid-cols-1 gap-1 md:grid-cols-3">
              {posteriors.map((d, i) => {
                const mean = posteriorMean('beta-binomial', d.post);
                const cri = credibleIntervalBeta(d.post.alpha0, d.post.beta0, 0.95);
                return (
                  <div key={i} className="rounded-md bg-[var(--color-surface-alt)] p-2">
                    <div className="text-xs font-semibold" style={{ color: palette[i] }}>
                      {d.prior.label}
                    </div>
                    <div className="font-mono text-xs">
                      mean = {mean.toFixed(4)}<br />
                      CrI = [{cri[0].toFixed(3)}, {cri[1].toFixed(3)}]
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 rounded-md bg-[var(--color-surface-alt)] p-2 text-xs text-[var(--color-text-muted)]">
              <strong>Sensitivity magnitude</strong>: max<sub>θ</sub> |pᵢ(θ | y) − pⱼ(θ | y)| across the three
              posterior pairs = <span className="font-mono font-semibold">{sensitivityMagnitude.toFixed(4)}</span>.
              Slide n to 1000 to watch this decay toward zero — the posteriors align as Bernstein–von Mises
              kicks in and the prior's contribution becomes negligible.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Slider({ label, min, max, step, value, onChange }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        aria-label={label}
        aria-valuetext={String(value)}
      />
    </label>
  );
}
