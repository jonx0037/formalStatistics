/**
 * PriorPosteriorExplorer — anchor component for §25.2 and §25.5.
 *
 * Beta-Binomial sequential Bayesian updating. Extends Topic 6's
 * BetaPriorPosteriorExplorer with three additions:
 *   1. Likelihood overlay always on by default (the "update engine").
 *   2. Sequential-trial mode — add one Bernoulli draw at a time and watch
 *      the posterior shift (400ms CSS transition).
 *   3. MAP + HPD readout alongside the equal-tailed credible interval.
 *
 * The pseudo-sample-size annotation (α₀+β₀) makes the conjugate-prior
 * hyperparameter interpretation concrete: the prior is worth this many
 * equivalent observations before any data arrive.
 */
import { useState, useMemo, useRef, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfBeta, posterior, posteriorMean, credibleIntervalBeta, hpdIntervalBeta,
  mapEstimate, type PriorHyperparams, type PosteriorHyperparams,
} from './shared/bayes';
import { seededRandom } from './shared/probability';
import { bernoulliSample } from './shared/convergence';
import { bayesianColors } from './shared/colorScales';
import { priorPosteriorPresets } from '../../data/bayesian-foundations-data';

const N_GRID = 300;
const MARGIN = { top: 14, right: 18, bottom: 34, left: 52 };

export default function PriorPosteriorExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [alpha0, setAlpha0] = useState(2);
  const [beta0, setBeta0] = useState(2);
  const [n, setN] = useState(20);
  const [k, setK] = useState(7);
  const [trueTheta, setTrueTheta] = useState(0.5);
  const [seqMode, setSeqMode] = useState(false);
  const [seqK, setSeqK] = useState(0);
  const [seqN, setSeqN] = useState(0);
  const seedRef = useRef(42);

  const isMobile = (width || 800) < 640;
  const chartW = Math.max(300, (width || 600) - 16);
  const chartH = isMobile ? 240 : 300;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  const activeK = seqMode ? seqK : k;
  const activeN = seqMode ? seqN : n;
  const priorHP: PriorHyperparams = { family: 'beta-binomial', alpha0, beta0 };
  const postHP = posterior(
    'beta-binomial',
    priorHP,
    { family: 'beta-binomial', n: activeN, k: activeK },
  ) as Extract<PosteriorHyperparams, { family: 'beta-binomial' }>;

  // ── Density grids (prior / likelihood / posterior) on (0, 1) ──────────────
  const { xs, priorY, likY, postY, yMax } = useMemo(() => {
    const xs_ = new Array<number>(N_GRID + 1);
    const priorY_ = new Array<number>(N_GRID + 1);
    const likY_ = new Array<number>(N_GRID + 1);
    const postY_ = new Array<number>(N_GRID + 1);
    // Scale likelihood to match posterior peak for visual comparison.
    // Use log-space then subtract the max (at MLE) to prevent underflow.
    const logLMax = activeN > 0 && activeK > 0 && activeN > activeK
      ? activeK * Math.log(activeK / activeN) + (activeN - activeK) * Math.log(1 - activeK / activeN)
      : 0;
    let maxP = 0;
    for (let i = 0; i <= N_GRID; i++) {
      const x = 0.001 + (i / N_GRID) * 0.998;
      xs_[i] = x;
      priorY_[i] = pdfBeta(x, alpha0, beta0);
      postY_[i] = pdfBeta(x, postHP.alpha0, postHP.beta0);
      if (activeN === 0) {
        likY_[i] = 0;
      } else {
        const logL = activeK * Math.log(x) + (activeN - activeK) * Math.log(1 - x) - logLMax;
        likY_[i] = Math.exp(logL);
      }
      if (priorY_[i] > maxP) maxP = priorY_[i];
      if (postY_[i] > maxP) maxP = postY_[i];
    }
    // Scale likelihood to match posterior max for the overlay.
    const postMax = Math.max(...postY_, 1e-6);
    for (let i = 0; i <= N_GRID; i++) likY_[i] *= postMax;
    return { xs: xs_, priorY: priorY_, likY: likY_, postY: postY_, yMax: Math.max(maxP, 0.5) };
  }, [alpha0, beta0, postHP.alpha0, postHP.beta0, activeN, activeK]);

  const x2px = (x: number) => MARGIN.left + x * plotW;
  const y2px = (y: number) => MARGIN.top + plotH - (y / yMax) * plotH;
  const path = (ys: number[]) => ys
    .map((y, i) => Number.isFinite(y) ? `${i === 0 ? 'M' : 'L'}${x2px(xs[i]).toFixed(2)},${y2px(y).toFixed(2)}` : '')
    .filter(s => s).join(' ');

  // ── Readouts ──────────────────────────────────────────────────────────────
  const priorMean = alpha0 / (alpha0 + beta0);
  const postMean = posteriorMean('beta-binomial', postHP);
  const cri = credibleIntervalBeta(postHP.alpha0, postHP.beta0, 0.95);
  const hpd = hpdIntervalBeta(postHP.alpha0, postHP.beta0, 0.95);
  const map = postHP.alpha0 > 1 && postHP.beta0 > 1
    ? mapEstimate('beta-binomial', postHP)
    : null;

  // ── Sequential simulation ─────────────────────────────────────────────────
  const simulateOne = useCallback(() => {
    if (!seqMode) setSeqMode(true);
    const rng = seededRandom(seedRef.current + seqN);
    const y = bernoulliSample(trueTheta, rng);
    setSeqN(nn => nn + 1);
    setSeqK(kk => kk + y);
  }, [seqMode, seqN, trueTheta]);

  const resetSequential = useCallback(() => {
    setSeqK(0); setSeqN(0); seedRef.current = 42 + Math.floor(Math.random() * 1000);
  }, []);

  return (
    <div ref={ref} className="my-6 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm">
      {/* Preset row */}
      <div className="mb-3 flex flex-wrap items-center gap-1">
        <span className="mr-1 text-xs text-[var(--color-text-muted)]">Preset:</span>
        {priorPosteriorPresets.map((p, i) => (
          <button
            key={i}
            type="button"
            onClick={() => { setAlpha0(p.alpha0); setBeta0(p.beta0); }}
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-2 py-0.5 text-xs hover:bg-[var(--color-surface)]"
            title={p.description}
          >
            {p.name.replace('Informative ', '').replace('Weakly informative ', 'weak ')}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[240px_1fr]">
        {/* Controls */}
        <div className="flex flex-col gap-2 rounded-lg bg-[var(--color-surface-alt)] p-3 text-xs">
          <Slider label={`α₀: ${alpha0.toFixed(2)}`} min={0.5} max={20} step={0.1} value={alpha0} onChange={setAlpha0} />
          <Slider label={`β₀: ${beta0.toFixed(2)}`} min={0.5} max={20} step={0.1} value={beta0} onChange={setBeta0} />

          <div className="mt-2 flex items-center justify-between border-t border-[var(--color-border)] pt-2">
            <span className="font-semibold">Data</span>
            <label className="flex items-center gap-1 text-[10px]">
              <input type="checkbox" checked={seqMode} onChange={e => setSeqMode(e.target.checked)} />
              sequential mode
            </label>
          </div>

          {seqMode ? (
            <>
              <div className="font-mono">n={seqN}, k={seqK}</div>
              <Slider label={`true θ: ${trueTheta.toFixed(2)}`} min={0.01} max={0.99} step={0.01} value={trueTheta} onChange={setTrueTheta} />
              <div className="flex gap-2">
                <button onClick={simulateOne} className="flex-1 rounded bg-[var(--color-posterior)] px-2 py-1 text-white hover:opacity-90" aria-label="Simulate one Bernoulli trial">
                  +1 trial
                </button>
                <button onClick={resetSequential} className="rounded border border-[var(--color-border)] px-2 py-1" aria-label="Reset sequential simulation">
                  ↻
                </button>
              </div>
            </>
          ) : (
            <>
              <Slider label={`n: ${n}`} min={0} max={200} step={1} value={n} onChange={v => { setN(v); if (k > v) setK(v); }} />
              <Slider label={`k: ${k}`} min={0} max={n} step={1} value={k} onChange={setK} />
            </>
          )}
        </div>

        {/* Chart + readouts */}
        <div>
          <svg width={chartW} height={chartH} role="img" aria-label="Beta prior, scaled likelihood, and posterior densities">
            {/* Grid */}
            <g stroke="var(--color-viz-grid)" strokeWidth={0.5}>
              {[0, 0.25, 0.5, 0.75, 1].map(v => (
                <line key={v} x1={x2px(v)} y1={MARGIN.top} x2={x2px(v)} y2={MARGIN.top + plotH} />
              ))}
            </g>
            {/* Axes */}
            <g stroke="var(--color-text-muted)" strokeWidth={1} fontSize={11} fill="var(--color-text-muted)">
              <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={MARGIN.left + plotW} y2={MARGIN.top + plotH} />
              <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} />
              {[0, 0.5, 1].map(v => (
                <text key={v} x={x2px(v)} y={MARGIN.top + plotH + 16} textAnchor="middle">{v.toFixed(1)}</text>
              ))}
              <text x={MARGIN.left + plotW / 2} y={chartH - 2} textAnchor="middle" fontSize={11}>θ</text>
            </g>
            {/* Prior (blue, faded when data present) */}
            <path
              d={path(priorY)}
              fill="none"
              stroke={bayesianColors.prior}
              strokeWidth={2}
              opacity={activeN > 0 ? 0.45 : 1}
              style={{ transition: 'opacity 400ms ease-out' }}
            />
            {/* Likelihood (amber dashed) */}
            {activeN > 0 && (
              <path
                d={path(likY)}
                fill="none"
                stroke={bayesianColors.likelihood}
                strokeWidth={2}
                strokeDasharray="6 3"
                opacity={0.8}
              />
            )}
            {/* Posterior (purple solid, emphasized) */}
            <path
              d={path(postY)}
              fill="none"
              stroke={bayesianColors.posterior}
              strokeWidth={3}
              style={{ transition: 'd 400ms ease-out' }}
            />
            {/* Legend */}
            <g fontSize={11} transform={`translate(${MARGIN.left + 6}, ${MARGIN.top + 4})`}>
              <g>
                <line x1={0} y1={6} x2={18} y2={6} stroke={bayesianColors.prior} strokeWidth={2} opacity={0.6} />
                <text x={22} y={9} fill="var(--color-text)">prior</text>
              </g>
              <g transform="translate(64, 0)">
                <line x1={0} y1={6} x2={18} y2={6} stroke={bayesianColors.likelihood} strokeWidth={2} strokeDasharray="6 3" />
                <text x={22} y={9} fill="var(--color-text)">likelihood (scaled)</text>
              </g>
              <g transform="translate(188, 0)">
                <line x1={0} y1={6} x2={18} y2={6} stroke={bayesianColors.posterior} strokeWidth={3} />
                <text x={22} y={9} fill="var(--color-text)">posterior</text>
              </g>
            </g>
          </svg>

          {/* Readouts */}
          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm md:grid-cols-3">
            <Readout label="Prior mean" value={priorMean.toFixed(4)} />
            <Readout label="Posterior mean" value={postMean.toFixed(4)} />
            <Readout label="Pseudo-sample-size" value={`${(alpha0 + beta0).toFixed(1)} → ${(alpha0 + beta0 + activeN).toFixed(1)}`} />
            <Readout label="95% equal-tailed CrI" value={`[${cri[0].toFixed(3)}, ${cri[1].toFixed(3)}]`} />
            <Readout label="95% HPD interval" value={`[${hpd[0].toFixed(3)}, ${hpd[1].toFixed(3)}]`} />
            <Readout label="MAP estimate" value={map !== null ? map.toFixed(4) : 'at boundary'} />
          </div>

          <p className="mt-3 rounded-md bg-[var(--color-surface-alt)] p-2 text-xs text-[var(--color-text-muted)]">
            Posterior <span style={{ color: bayesianColors.posterior }} className="font-semibold">Beta({postHP.alpha0.toFixed(1)}, {postHP.beta0.toFixed(1)})</span>.
            The HPD interval is the shortest 95% credible interval; it coincides with the
            equal-tailed interval only for symmetric posteriors.
          </p>
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

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[var(--color-text-muted)]">{label}</span>:{' '}
      <span className="font-mono font-semibold">{value}</span>
    </div>
  );
}
