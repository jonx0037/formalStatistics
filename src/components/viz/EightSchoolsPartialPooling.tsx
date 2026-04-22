/**
 * EightSchoolsPartialPooling — interactive for Topic 28 §28.6.
 *
 * The canonical partial-pooling picture. Reader controls the between-
 * group variance τ² on a log slider and watches the eight Rubin 1981
 * posterior means smoothly interpolate between the raw sample means
 * (τ² → ∞, no pooling) and the precision-weighted grand mean
 * (τ² → 0, complete pooling). Annotates each school with its shrinkage
 * factor B_k = σ²_k/(σ²_k + τ²).
 *
 * Four preset buttons cover the dominant regimes:
 *   — Complete pool (τ² → 0)
 *   — Moderate (τ = 5)
 *   — GEL2013 default (τ = 10)
 *   — Type-II MLE τ̂² (≈ 0 for 8-schools — the boundary MLE, §28.7)
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  partialPoolingShrinkageFactor,
  partialPoolingPosteriorMean,
  partialPoolingPosteriorVariance,
  normalNormalGrandMean,
  typeIIMLE,
} from './shared/bayes';
import { bayesianColors } from './shared/colorScales';
import { eightSchoolsPreset } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 32, right: 96, bottom: 44, left: 56 };
const MOBILE_BREAKPOINT = 640;
const TAU_MIN_LOG = Math.log(0.01);
const TAU_MAX_LOG = Math.log(100);

const tauFromSlider = (s: number) => Math.exp(TAU_MIN_LOG + s * (TAU_MAX_LOG - TAU_MIN_LOG));
const sliderFromTau = (t: number) =>
  (Math.log(t) - TAU_MIN_LOG) / (TAU_MAX_LOG - TAU_MIN_LOG);

// Clamp τ to slider range for presets that may land at boundary values.
const clampedSliderFromTau = (t: number) => {
  const s = sliderFromTau(Math.max(0.01, t));
  return Math.max(0, Math.min(1, s));
};

export default function EightSchoolsPartialPooling() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const { names, y, sigma } = eightSchoolsPreset;
  const sigmaSq = useMemo(() => sigma.map((s) => s * s), [sigma]);

  // Initial τ ≈ 5 (GEL2013 §5.5 moderate regime).
  const [tauSlider, setTauSlider] = useState(clampedSliderFromTau(5));
  const [activePreset, setActivePreset] = useState<string | null>('moderate');

  const tau = tauFromSlider(tauSlider);
  const tauSq = tau * tau;

  // Type-II MLE for preset button + global EB reference.
  const ebFit = useMemo(
    () => typeIIMLE([...y], [...sigmaSq]),
    [y, sigmaSq],
  );

  // Precision-weighted grand mean under current τ² (the μ̂ the posterior
  // mean shrinks toward). Formally this is the MLE of μ given τ²; full
  // Bayes would integrate over its posterior.
  const muHat = useMemo(
    () => normalNormalGrandMean([...y], sigmaSq.map((s) => s + tauSq)),
    [y, sigmaSq, tauSq],
  );

  const rows = y.map((yk, k) => {
    const sk2 = sigmaSq[k];
    const B = partialPoolingShrinkageFactor(sk2, tauSq);
    const postMean = partialPoolingPosteriorMean(yk, muHat, sk2, tauSq);
    const postVar = partialPoolingPosteriorVariance(sk2, tauSq);
    return {
      name: names[k],
      y: yk,
      sigma: sigma[k],
      B,
      postMean,
      postSd: Math.sqrt(postVar),
    };
  });

  const applyPreset = (id: string) => {
    setActivePreset(id);
    if (id === 'complete') setTauSlider(clampedSliderFromTau(0.01));
    else if (id === 'moderate') setTauSlider(clampedSliderFromTau(5));
    else if (id === 'gelman') setTauSlider(clampedSliderFromTau(10));
    else if (id === 'type2') setTauSlider(clampedSliderFromTau(Math.sqrt(Math.max(0.01, ebFit.tauSq))));
    else if (id === 'nopool') setTauSlider(1);
  };

  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(720, (width ?? 720) - 16));
  const chartH = isMobile ? 340 : 380;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // x-axis: union of raw y ± σ and posterior mean ± sd. Pad by 10% on each side.
  const xVals: number[] = [];
  for (const r of rows) {
    xVals.push(r.y - r.sigma, r.y + r.sigma);
    xVals.push(r.postMean - r.postSd, r.postMean + r.postSd);
  }
  const xMin = Math.floor(Math.min(muHat, -15, ...xVals) / 5) * 5;
  const xMax = Math.ceil(Math.max(muHat, 35, ...xVals) / 5) * 5;
  const xOf = (v: number) => ((v - xMin) / (xMax - xMin)) * plotW;

  const rowH = plotH / rows.length;

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">τ preset:</span>
        {[
          { id: 'complete', label: 'τ → 0 (complete pool)' },
          { id: 'moderate', label: 'τ = 5' },
          { id: 'gelman', label: 'τ = 10 (GEL2013)' },
          { id: 'type2', label: `Type-II MLE τ̂ = ${Math.sqrt(Math.max(0, ebFit.tauSq)).toFixed(2)}` },
        ].map((p) => (
          <button
            key={p.id}
            onClick={() => applyPreset(p.id)}
            className={`rounded px-3 py-1 text-xs transition ${
              activePreset === p.id
                ? 'bg-[var(--color-shrink)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <label className="mb-4 flex flex-col gap-1 text-sm">
        <span className="flex items-center justify-between">
          <span>Between-group SD τ (log scale)</span>
          <span className="font-mono">
            τ = {tau.toFixed(tau < 1 ? 3 : tau < 10 ? 2 : 1)} · τ² = {tauSq.toFixed(tau < 1 ? 3 : 1)}
          </span>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={tauSlider}
          onChange={(e) => {
            setTauSlider(parseFloat(e.target.value));
            setActivePreset(null);
          }}
          className="accent-[var(--color-shrink)]"
        />
      </label>

      <svg width={chartW} height={chartH} className="block">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* x ticks */}
          {Array.from({ length: Math.floor((xMax - xMin) / 5) + 1 }, (_, i) => xMin + i * 5).map((t) => (
            <g key={t}>
              <line x1={xOf(t)} y1={plotH} x2={xOf(t)} y2={plotH + 4} stroke="var(--color-text-muted)" />
              <text
                x={xOf(t)}
                y={plotH + 18}
                textAnchor="middle"
                fontSize={10}
                fill="var(--color-text-muted)"
              >
                {t}
              </text>
            </g>
          ))}

          {/* Zero line */}
          <line
            x1={xOf(0)}
            y1={0}
            x2={xOf(0)}
            y2={plotH}
            stroke="var(--color-border)"
            strokeDasharray="3 3"
          />

          {/* Grand mean (μ̂) reference */}
          <line
            x1={xOf(muHat)}
            y1={-8}
            x2={xOf(muHat)}
            y2={plotH}
            stroke={bayesianColors.posterior}
            strokeWidth={1.5}
            strokeDasharray="5 3"
          />
          <text
            x={xOf(muHat)}
            y={-12}
            textAnchor="middle"
            fontSize={10}
            fill={bayesianColors.posterior}
          >
            μ̂ = {muHat.toFixed(2)}
          </text>

          {/* Rows */}
          {rows.map((r, i) => {
            const yc = i * rowH + rowH / 2;
            return (
              <g key={r.name}>
                {/* School label */}
                <text x={-8} y={yc + 4} textAnchor="end" fontSize={11} fill="var(--color-text)">
                  {r.name}
                </text>
                {/* Raw y ± σ (light grey) */}
                <line
                  x1={xOf(r.y - r.sigma)}
                  y1={yc - 8}
                  x2={xOf(r.y + r.sigma)}
                  y2={yc - 8}
                  stroke="var(--color-text-muted)"
                  strokeWidth={2}
                  opacity={0.5}
                />
                <circle cx={xOf(r.y)} cy={yc - 8} r={4} fill={bayesianColors.mle} opacity={0.9} />
                {/* Posterior mean ± SD */}
                <line
                  x1={xOf(r.postMean - r.postSd)}
                  y1={yc + 6}
                  x2={xOf(r.postMean + r.postSd)}
                  y2={yc + 6}
                  stroke={bayesianColors.shrink}
                  strokeWidth={3}
                />
                <circle cx={xOf(r.postMean)} cy={yc + 6} r={4.5} fill={bayesianColors.shrink} />
                {/* Shrinkage factor annotation */}
                <text
                  x={plotW + 6}
                  y={yc + 4}
                  fontSize={10}
                  fill="var(--color-text-muted)"
                  fontFamily="ui-monospace, monospace"
                >
                  B = {r.B.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Axis label */}
          <text
            x={plotW / 2}
            y={plotH + 34}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
          >
            Coaching effect (SAT-V points)
          </text>
        </g>
      </svg>

      <div className="mt-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
        <span style={{ color: bayesianColors.mle }}>Orange:</span> raw sample mean y_k ± σ_k (no pooling).{' '}
        <span style={{ color: bayesianColors.shrink }}>Violet:</span> posterior mean ± posterior SD at
        the current τ (§28.6 Thm 4).{' '}
        <span style={{ color: bayesianColors.posterior }}>Dashed indigo:</span> precision-weighted
        grand mean μ̂(τ) — the target of shrinkage.
      </div>
    </div>
  );
}
