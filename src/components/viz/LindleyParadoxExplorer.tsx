/**
 * LindleyParadoxExplorer — featured interactive for Topic 27 §27.5.
 *
 * Closed-form only — no Monte Carlo. Three sliders (z, n, τ) drive the
 * BF₁₀(τ) curve on a log-τ axis. At each slider state, report the trio
 * (frequentist p-value, Bayes factor, posterior odds under uniform prior)
 * side-by-side to make the paradox concrete: at z=3 with diffuse τ, the
 * frequentist rejects H₀ while the Bayesian favors it by ~11:1.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import { lindleyBayesFactor } from './shared/bayes';
import { cdfStdNormal } from './shared/distributions';
import { bayesianColors } from './shared/colorScales';
import { lindleyPresets } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 24, right: 32, bottom: 44, left: 56 };
const MOBILE_BREAKPOINT = 640;
const TAU_GRID_POINTS = 160;
const LOG_TAU_MIN = Math.log(0.1);
const LOG_TAU_MAX = Math.log(1000);

const sliderToLogTau = (s: number) => LOG_TAU_MIN + s * (LOG_TAU_MAX - LOG_TAU_MIN);
const logTauToSlider = (logTau: number) =>
  (logTau - LOG_TAU_MIN) / (LOG_TAU_MAX - LOG_TAU_MIN);

/** Two-sided p-value for a z-statistic. */
const zTwoSidedPValue = (z: number) => 2 * (1 - cdfStdNormal(Math.abs(z)));

export default function LindleyParadoxExplorer() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState<number | null>(0);
  const [z, setZ] = useState(3.0);
  const [n, setN] = useState(100);
  const [tauSlider, setTauSlider] = useState(logTauToSlider(Math.log(100)));

  // When user drags a slider, deselect preset (so "preset active" indicator clears).
  const applyPreset = (idx: number) => {
    const p = lindleyPresets[idx];
    setPresetIdx(idx);
    setZ(p.z);
    setN(p.n);
    setTauSlider(logTauToSlider(Math.log(p.tau)));
  };

  const tau = Math.exp(sliderToLogTau(tauSlider));
  const sigma = 1;
  const bf10 = lindleyBayesFactor(z, n, tau, sigma);
  const pValue = zTwoSidedPValue(z);

  // BF curve over τ
  const curve = useMemo(() => {
    const points: { tau: number; bf: number }[] = [];
    for (let i = 0; i <= TAU_GRID_POINTS; i++) {
      const logTau = LOG_TAU_MIN + (i / TAU_GRID_POINTS) * (LOG_TAU_MAX - LOG_TAU_MIN);
      const t = Math.exp(logTau);
      points.push({ tau: t, bf: lindleyBayesFactor(z, n, t, sigma) });
    }
    return points;
  }, [z, n]);

  const isMobile = (width ?? 800) < MOBILE_BREAKPOINT;
  const chartW = Math.max(320, Math.min(720, (width ?? 720) - 16));
  const chartH = isMobile ? 280 : 340;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // y-axis: log BF, with decade gridlines
  const bfs = curve.map((p) => p.bf);
  const logBfs = bfs.map((b) => Math.log10(Math.max(1e-10, b)));
  const logMin = Math.min(...logBfs, -3);
  const logMax = Math.max(...logBfs, 2);
  const logPad = 0.15;
  const logYMin = Math.floor(logMin - logPad);
  const logYMax = Math.ceil(logMax + logPad);

  const xOf = (t: number) =>
    ((Math.log(t) - LOG_TAU_MIN) / (LOG_TAU_MAX - LOG_TAU_MIN)) * plotW;
  const yOf = (logBf: number) =>
    plotH * (1 - (logBf - logYMin) / (logYMax - logYMin));

  const curvePath = curve
    .map((p, i) => {
      const xx = xOf(p.tau);
      const yy = yOf(Math.log10(Math.max(1e-10, p.bf)));
      return `${i === 0 ? 'M' : 'L'} ${xx.toFixed(2)} ${yy.toFixed(2)}`;
    })
    .join(' ');

  const currentX = xOf(tau);
  const currentY = yOf(Math.log10(Math.max(1e-10, bf10)));

  // Jeffreys threshold bands: BF = 1, 3.2, 10, 100
  const thresholds = [
    { value: 1, label: 'BF = 1' },
    { value: 3.2, label: 'BF = 3.2' },
    { value: 10, label: 'BF = 10' },
    { value: 100, label: 'BF = 100' },
  ];

  const xTicks = [0.1, 1, 10, 100, 1000].filter(
    (t) => Math.log(t) >= LOG_TAU_MIN && Math.log(t) <= LOG_TAU_MAX,
  );

  return (
    <div
      ref={ref}
      className="my-6 rounded-lg border p-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {/* Preset row */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Preset:</span>
        {lindleyPresets.map((p, i) => (
          <button
            key={p.id}
            onClick={() => applyPreset(i)}
            className={`rounded px-3 py-1 text-xs transition ${
              i === presetIdx
                ? 'bg-[var(--color-posterior)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Sliders */}
      <div className="mb-4 grid grid-cols-1 gap-3 text-sm md:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>z-statistic</span>
            <span className="font-mono">{z.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={z}
            onChange={(e) => { setZ(parseFloat(e.target.value)); setPresetIdx(null); }}
            className="accent-[var(--color-posterior)]"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Sample size n</span>
            <span className="font-mono">{n}</span>
          </span>
          <input
            type="range"
            min={10}
            max={1000}
            step={10}
            value={n}
            onChange={(e) => { setN(parseInt(e.target.value)); setPresetIdx(null); }}
            className="accent-[var(--color-posterior)]"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between">
            <span>Prior scale τ (log)</span>
            <span className="font-mono">{tau.toFixed(tau < 1 ? 2 : tau < 10 ? 2 : 1)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={tauSlider}
            onChange={(e) => { setTauSlider(parseFloat(e.target.value)); setPresetIdx(null); }}
            className="accent-[var(--color-posterior)]"
          />
        </label>
      </div>

      {/* SVG chart */}
      <svg width={chartW} height={chartH} className="block">
        <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
          {/* Jeffreys threshold bands */}
          {thresholds.map((t) => {
            const logBf = Math.log10(t.value);
            if (logBf < logYMin || logBf > logYMax) return null;
            return (
              <g key={t.value}>
                <line
                  x1={0}
                  y1={yOf(logBf)}
                  x2={plotW}
                  y2={yOf(logBf)}
                  stroke="var(--color-border)"
                  strokeWidth={0.8}
                  strokeDasharray="3 3"
                />
                <text
                  x={plotW - 4}
                  y={yOf(logBf) - 3}
                  textAnchor="end"
                  fontSize={9}
                  fill="var(--color-text-muted)"
                >
                  {t.label}
                </text>
              </g>
            );
          })}
          {/* x decade labels */}
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={xOf(t)} y1={plotH} x2={xOf(t)} y2={plotH + 4} stroke="var(--color-text-muted)" />
              <text x={xOf(t)} y={plotH + 18} textAnchor="middle" fontSize={10} fill="var(--color-text-muted)">
                {t >= 1 ? t : t.toString()}
              </text>
            </g>
          ))}
          {/* y axis labels */}
          {Array.from({ length: logYMax - logYMin + 1 }, (_, i) => logYMin + i).map((v) => (
            <text
              key={v}
              x={-8}
              y={yOf(v) + 4}
              textAnchor="end"
              fontSize={10}
              fill="var(--color-text-muted)"
            >
              10^{v}
            </text>
          ))}
          {/* BF curve */}
          <path d={curvePath} fill="none" stroke={bayesianColors.posterior} strokeWidth={2.5} />
          {/* Current state marker */}
          <circle cx={currentX} cy={currentY} r={6} fill={bayesianColors.likelihood} stroke="white" strokeWidth={1.5} />
          {/* Axis labels */}
          <text x={plotW / 2} y={plotH + 34} textAnchor="middle" fontSize={11} fill="var(--color-text)">
            Prior scale τ (log)
          </text>
          <text
            x={-42}
            y={plotH / 2}
            textAnchor="middle"
            fontSize={11}
            fill="var(--color-text)"
            transform={`rotate(-90 ${-42} ${plotH / 2})`}
          >
            BF₁₀ (log scale)
          </text>
        </g>
      </svg>

      {/* Numerical readout */}
      <div className="mt-4 grid grid-cols-1 gap-4 text-sm md:grid-cols-3">
        <div className="rounded border p-3" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Frequentist p-value
          </div>
          <div className="mt-1 font-mono text-lg" style={{ color: bayesianColors.mle }}>
            {pValue < 1e-4 ? pValue.toExponential(2) : pValue.toFixed(4)}
          </div>
          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            two-sided z-test
          </div>
        </div>
        <div className="rounded border p-3" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Bayes factor BF₁₀
          </div>
          <div className="mt-1 font-mono text-lg" style={{ color: bayesianColors.posterior }}>
            {bf10 < 1e-4 ? bf10.toExponential(2) : bf10 < 0.01 ? bf10.toExponential(2) : bf10 > 1000 ? bf10.toExponential(2) : bf10.toFixed(4)}
          </div>
          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {bf10 < 1 ? 'favors H₀' : bf10 < 3.2 ? 'barely worth mentioning' : bf10 < 10 ? 'substantial' : bf10 < 100 ? 'strong' : 'decisive'}
          </div>
        </div>
        <div className="rounded border p-3" style={{ borderColor: 'var(--color-border)' }}>
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
            Posterior odds (uniform prior)
          </div>
          <div className="mt-1 font-mono text-lg" style={{ color: bayesianColors.true }}>
            {bf10 > 1 ? `${bf10.toFixed(2)} : 1` : `1 : ${(1 / bf10).toFixed(2)}`}
          </div>
          <div className="mt-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            P(M₁|y) / P(M₀|y) at p(M) = 1/2
          </div>
        </div>
      </div>

      {presetIdx !== null && (
        <div
          className="mt-3 text-xs italic"
          style={{ color: 'var(--color-text-muted)' }}
        >
          {lindleyPresets[presetIdx].description}
        </div>
      )}
    </div>
  );
}
