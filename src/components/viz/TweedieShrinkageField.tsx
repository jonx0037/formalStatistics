/**
 * TweedieShrinkageField — featured interactive for Topic 33 §33.4.
 *
 * Makes Tweedie's formula (Thm 3) visible: the Bayes correction
 * E[θ | x] − x = σ² ℓ'(x) is a vector field on the line — σ² times the
 * score of the marginal density f — and f-modeling estimates that field
 * from the data alone. The reader picks a prior, a sample size, and a KDE
 * bandwidth, and watches the KDE-estimated rule approach the oracle.
 *
 * Top panel: histogram of the N observations, the true marginal f, the
 * KDE f̂, and shrinkage arrows from x to x + σ²ℓ'(x) drawn on the true f.
 * The true score comes from Thm 3(i) itself — ℓ'(x) = (E[θ | x] − x)/σ² —
 * so no numerical differentiation is needed.
 * Bottom panel: identity (the MLE), the oracle Bayes rule, and the
 * Tweedie–KDE rule, with the oracle's shrinkage shaded.
 *
 * Performance: N = 5,000 × 241 grid points for the KDE curves; the MSE
 * readout uses at most 1,000 random points. useDeferredValue keeps the
 * N and bandwidth sliders responsive while the KDE refits.
 */
import { useDeferredValue, useId, useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  createSeededRng,
  sampleNormalMixture,
  normalMixtureMarginal,
  posteriorMeanNormalMixturePrior,
  tweedieKDE,
} from './shared/bayes';
// scottBandwidth is the SD rule 1.06·SD·n^(−1/5) — the same "Silverman"
// default tweedieKDE uses — so multiplier 1× reproduces tweedieKDE's default.
import { kdeEvaluateGrid, scottBandwidth } from './shared/nonparametric';
import { bayesianColors } from './shared/colorScales';
import {
  tweedieShrinkageDefaults,
  tweedieShrinkagePresets,
} from '../../data/bayesian-foundations-data';

const MARGIN = { top: 12, right: 16, bottom: 36, left: 46 };
const NARROW_BREAKPOINT = 480;
const X_MIN = -6;
const X_MAX = 6;
const N_GRID = 241;
const HIST_BINS = 40;
const MSE_SUBSAMPLE = 1000;
const LOG_N_MIN = Math.log10(50);
const LOG_N_MAX = Math.log10(5000);
const ARROWS_FULL = [-3.75, -3, -2.25, -1.5, -0.75, 0.75, 1.5, 2.25, 3, 3.75];
const ARROWS_NARROW = [-3.75, -2.25, -0.75, 0.75, 2.25, 3.75];
const X_TICKS = [-6, -4, -2, 0, 2, 4, 6];

const X_GRID = Array.from({ length: N_GRID }, (_, i) => X_MIN + ((X_MAX - X_MIN) * i) / (N_GRID - 1));

function linePath(xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number): string {
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(ys[i]).toFixed(2)}`).join('');
}

/** Horizontal arrow in pixel coordinates; a dot when the shift is sub-pixel. */
function Arrow({ x0, x1, y, color }: { x0: number; x1: number; y: number; color: string }) {
  const len = Math.abs(x1 - x0);
  if (len < 1) return <circle cx={x0} cy={y} r={2.2} fill={color} />;
  const dir = Math.sign(x1 - x0);
  const head = Math.min(7, len * 0.6);
  const base = x1 - dir * head;
  return (
    <g>
      <line x1={x0} y1={y} x2={base} y2={y} stroke={color} strokeWidth={2} />
      <path d={`M${x1},${y}L${base},${y - head * 0.55}L${base},${y + head * 0.55}Z`} fill={color} />
      <circle cx={x0} cy={y} r={2.2} fill={color} />
    </g>
  );
}

function LegendItem({ color, label, dash, marker }: { color: string; label: string; dash?: string; marker?: 'bar' | 'arrow' }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width={22} height={10} aria-hidden="true">
        {marker === 'bar' ? (
          <rect x={4} y={1} width={14} height={8} fill={color} opacity={0.35} />
        ) : (
          <>
            <line x1={1} y1={5} x2={marker === 'arrow' ? 15 : 21} y2={5} stroke={color} strokeWidth={2} strokeDasharray={dash} />
            {marker === 'arrow' && <path d="M21,5L15,1.5L15,8.5Z" fill={color} />}
          </>
        )}
      </svg>
      <span>{label}</span>
    </span>
  );
}

export default function TweedieShrinkageField() {
  const { ref: outerRef, width: outerW } = useResizeObserver<HTMLDivElement>();
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const [presetId, setPresetId] = useState(tweedieShrinkageDefaults.presetId);
  const [logN, setLogN] = useState(Math.log10(tweedieShrinkageDefaults.N));
  const [multiplier, setMultiplier] = useState(tweedieShrinkageDefaults.bandwidthMultiplier);
  const [seed, setSeed] = useState(tweedieShrinkageDefaults.seed);
  const [showArrows, setShowArrows] = useState(true);
  const [showOracle, setShowOracle] = useState(true);
  const [showKDE, setShowKDE] = useState(true);

  const preset = tweedieShrinkagePresets.find((p) => p.id === presetId) ?? tweedieShrinkagePresets[0];
  const sigmaSq = tweedieShrinkageDefaults.sigma ** 2;
  const N = Math.round(10 ** logN);
  const dN = useDeferredValue(N);
  const dMultiplier = useDeferredValue(multiplier);

  // Draw θ_i ~ g and x_i = θ_i + σ ε_i; pick the MSE evaluation subset.
  const sim = useMemo(() => {
    const rng = createSeededRng(seed);
    const theta = sampleNormalMixture(preset.comps, dN, rng);
    const sd = Math.sqrt(sigmaSq);
    const x = theta.map((t) => t + sd * rng.normal());
    let idx = x.map((_, i) => i);
    if (dN > MSE_SUBSAMPLE) {
      // Partial Fisher–Yates on a separate stream, so the sample itself
      // does not depend on whether a subset is drawn.
      const sub = createSeededRng(seed + 7919);
      for (let i = 0; i < MSE_SUBSAMPLE; i++) {
        const j = i + Math.floor(sub.random() * (dN - i));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      idx = idx.slice(0, MSE_SUBSAMPLE);
    }
    return { theta, x, idx };
  }, [preset, dN, seed, sigmaSq]);

  // Oracle quantities depend only on the prior.
  const oracle = useMemo(
    () => ({
      f: X_GRID.map((v) => normalMixtureMarginal(v, preset.comps, sigmaSq)),
      rule: X_GRID.map((v) => posteriorMeanNormalMixturePrior(v, preset.comps, sigmaSq)),
    }),
    [preset, sigmaSq],
  );

  // f-modeling: KDE marginal, Tweedie–KDE rule, MSE readout, histogram.
  const fit = useMemo(() => {
    const { theta, x, idx } = sim;
    const h = dMultiplier * scottBandwidth(x);
    const fHat = kdeEvaluateGrid(x, X_GRID, h);
    const ruleHat = tweedieKDE(x, X_GRID, sigmaSq, h);
    const kdeAtEval = tweedieKDE(x, idx.map((i) => x[i]), sigmaSq, h);
    let mle = 0;
    let kde = 0;
    let orc = 0;
    idx.forEach((i, j) => {
      const th = theta[i];
      mle += (x[i] - th) ** 2;
      kde += (kdeAtEval[j] - th) ** 2;
      orc += (posteriorMeanNormalMixturePrior(x[i], preset.comps, sigmaSq) - th) ** 2;
    });
    const m = idx.length;
    const binW = (X_MAX - X_MIN) / HIST_BINS;
    const counts = new Array<number>(HIST_BINS).fill(0);
    for (const v of x) {
      if (v < X_MIN || v >= X_MAX) continue;
      counts[Math.floor((v - X_MIN) / binW)]++;
    }
    return {
      h,
      fHat,
      ruleHat,
      hist: counts.map((c) => c / (x.length * binW)),
      binW,
      mse: { mle: mle / m, kde: kde / m, oracle: orc / m },
      nEval: m,
    };
  }, [sim, dMultiplier, preset, sigmaSq]);

  // Layout.
  const chartW = Math.max(260, width || 600);
  const narrow = (outerW || 800) < NARROW_BREAKPOINT;
  const topH = narrow ? 190 : 220;
  const botH = narrow ? 240 : 280;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const topPlotH = topH - MARGIN.top - MARGIN.bottom;
  const botPlotH = botH - MARGIN.top - MARGIN.bottom;
  const sx = (v: number) => ((v - X_MIN) / (X_MAX - X_MIN)) * plotW;

  let yTopMax = 0;
  for (const v of fit.hist) yTopMax = Math.max(yTopMax, v);
  for (const v of oracle.f) yTopMax = Math.max(yTopMax, v);
  if (showKDE) for (const v of fit.fHat) yTopMax = Math.max(yTopMax, v);
  yTopMax *= 1.08;
  const syTop = (v: number) => topPlotH - (v / yTopMax) * topPlotH;
  const syBot = (v: number) => botPlotH - ((v - X_MIN) / (X_MAX - X_MIN)) * botPlotH;

  const arrows = (narrow ? ARROWS_NARROW : ARROWS_FULL).map((x0) => ({
    x0,
    // x + σ²ℓ'(x) = E[θ | x] by Thm 3(i).
    x1: posteriorMeanNormalMixturePrior(x0, preset.comps, sigmaSq),
    fx: normalMixtureMarginal(x0, preset.comps, sigmaSq),
  }));

  const shadePath =
    linePath(X_GRID, X_GRID, sx, syBot) +
    [...X_GRID]
      .reverse()
      .map((v, i) => `L${sx(v).toFixed(2)},${syBot(oracle.rule[N_GRID - 1 - i]).toFixed(2)}`)
      .join('') +
    'Z';

  const { mse } = fit;
  const gain = mse.mle - mse.oracle;
  const recovered = gain > 0 ? (100 * (mse.mle - mse.kde)) / gain : NaN;
  const stale = dN !== N || dMultiplier !== multiplier;

  const axisColor = 'var(--color-text-muted)';

  return (
    <div ref={outerRef} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-semibold">Prior g:</span>
        {tweedieShrinkagePresets.map((p) => (
          <button
            key={p.id}
            onClick={() => setPresetId(p.id)}
            className={`rounded px-3 py-1 text-xs transition ${
              p.id === presetId
                ? 'bg-[var(--color-shrink)] text-white'
                : 'bg-[var(--color-surface)] hover:bg-[var(--color-surface-alt)]'
            }`}
            style={{ borderWidth: 1, borderColor: 'var(--color-border)' }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="mb-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-4">
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between gap-2">
            <span>Sample size N</span>
            <span className="font-mono text-xs">{N.toLocaleString('en-US')}</span>
          </span>
          <input
            type="range"
            min={LOG_N_MIN}
            max={LOG_N_MAX}
            step={0.01}
            value={logN}
            onChange={(e) => setLogN(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-f-model)]"
            aria-label="Sample size N (log scale)"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="flex items-center justify-between gap-2">
            <span>Bandwidth</span>
            <span className="font-mono text-xs">
              {multiplier.toFixed(2)}× · h = {fit.h.toFixed(2)}
            </span>
          </span>
          <input
            type="range"
            min={0.25}
            max={3}
            step={0.05}
            value={multiplier}
            onChange={(e) => setMultiplier(parseFloat(e.target.value))}
            className="w-full accent-[var(--color-f-model)]"
            aria-label="KDE bandwidth as a multiple of Silverman's rule"
          />
        </label>
        <div className="flex flex-col gap-1">
          <span>Seed</span>
          <div className="flex items-center gap-2">
            <input
              type="number"
              value={seed}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (Number.isFinite(v)) setSeed(v);
              }}
              className="w-20 min-w-0 rounded border px-2 py-0.5 font-mono text-xs"
              style={{ borderColor: 'var(--color-border)', background: 'var(--color-surface)' }}
              aria-label="Random seed"
            />
            <button
              onClick={() => setSeed((s) => s + 1)}
              className="rounded px-3 py-1 text-xs font-medium text-white"
              style={{ background: bayesianColors.shrink }}
            >
              Resample
            </button>
          </div>
        </div>
        <div className="flex flex-col gap-1 text-xs">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={showArrows} onChange={(e) => setShowArrows(e.target.checked)} />
            Shrinkage arrows
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={showOracle} onChange={(e) => setShowOracle(e.target.checked)} />
            Oracle curves
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={showKDE} onChange={(e) => setShowKDE(e.target.checked)} />
            KDE curves
          </label>
        </div>
      </div>

      <div className="flex flex-col gap-4 md:flex-row">
        <div ref={ref} className="min-w-0 flex-1" style={{ opacity: stale ? 0.7 : 1 }}>
          <div className="mb-1 flex flex-wrap gap-x-3 gap-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <LegendItem color="var(--color-text-muted)" label="histogram of x" marker="bar" />
            {showOracle && <LegendItem color={bayesianColors.oracle} label="true marginal f" />}
            {showKDE && <LegendItem color={bayesianColors.fModel} label="KDE f̂" dash="5 3" />}
            {showArrows && <LegendItem color={bayesianColors.shrink} label="σ²ℓ′(x): x → E[θ | x]" marker="arrow" />}
          </div>
          <svg
            width={chartW}
            height={topH}
            className="block"
            role="img"
            aria-label="Histogram of the observations with the true and estimated marginal densities and shrinkage arrows"
          >
            <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
              {fit.hist.map((d, k) => {
                const x0 = sx(X_MIN + k * fit.binW);
                const x1 = sx(X_MIN + (k + 1) * fit.binW);
                return (
                  <rect
                    key={k}
                    x={x0}
                    y={syTop(d)}
                    width={Math.max(0, x1 - x0 - 1)}
                    height={Math.max(0, topPlotH - syTop(d))}
                    fill="var(--color-text-muted)"
                    opacity={0.22}
                  />
                );
              })}
              {showOracle && (
                <path d={linePath(X_GRID, oracle.f, sx, syTop)} fill="none" stroke={bayesianColors.oracle} strokeWidth={2} />
              )}
              {showKDE && (
                <path
                  d={linePath(X_GRID, fit.fHat, sx, syTop)}
                  fill="none"
                  stroke={bayesianColors.fModel}
                  strokeWidth={2}
                  strokeDasharray="5 3"
                />
              )}
              {showArrows &&
                arrows.map((a) => (
                  <Arrow key={a.x0} x0={sx(a.x0)} x1={sx(a.x1)} y={syTop(a.fx)} color={bayesianColors.shrink} />
                ))}
              <line x1={0} y1={topPlotH} x2={plotW} y2={topPlotH} stroke={axisColor} />
              {X_TICKS.map((t) => (
                <g key={t}>
                  <line x1={sx(t)} y1={topPlotH} x2={sx(t)} y2={topPlotH + 4} stroke={axisColor} />
                  <text x={sx(t)} y={topPlotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                    {t}
                  </text>
                </g>
              ))}
              <text x={plotW / 2} y={topPlotH + 30} textAnchor="middle" fontSize={11} fill="var(--color-text)">
                observation x
              </text>
              <text
                x={-32}
                y={topPlotH / 2}
                textAnchor="middle"
                fontSize={11}
                fill="var(--color-text)"
                transform={`rotate(-90 ${-32} ${topPlotH / 2})`}
              >
                density
              </text>
            </g>
          </svg>

          <div className="mb-1 mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <LegendItem color={bayesianColors.mle} label="identity (MLE: θ̂ = x)" dash="5 3" />
            {showOracle && <LegendItem color={bayesianColors.oracle} label="oracle Bayes E[θ | x]" />}
            {showKDE && <LegendItem color={bayesianColors.fModel} label="Tweedie–KDE" dash="5 3" />}
          </div>
          <svg
            width={chartW}
            height={botH}
            className="block"
            role="img"
            aria-label="Shrinkage rules: identity, oracle Bayes rule, and the Tweedie rule with a KDE marginal"
          >
            <defs>
              <clipPath id={`${uid}-clip`}>
                <rect x={0} y={0} width={plotW} height={botPlotH} />
              </clipPath>
            </defs>
            <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
              {X_TICKS.map((t) => (
                <line key={t} x1={0} y1={syBot(t)} x2={plotW} y2={syBot(t)} stroke="var(--color-viz-grid)" />
              ))}
              <g clipPath={`url(#${uid}-clip)`}>
                {showOracle && <path d={shadePath} fill={bayesianColors.shrink} opacity={0.12} />}
                <path
                  d={linePath(X_GRID, X_GRID, sx, syBot)}
                  fill="none"
                  stroke={bayesianColors.mle}
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                />
                {showOracle && (
                  <path d={linePath(X_GRID, oracle.rule, sx, syBot)} fill="none" stroke={bayesianColors.oracle} strokeWidth={2.2} />
                )}
                {showKDE && (
                  <path
                    d={linePath(X_GRID, fit.ruleHat, sx, syBot)}
                    fill="none"
                    stroke={bayesianColors.fModel}
                    strokeWidth={2}
                    strokeDasharray="5 3"
                  />
                )}
              </g>
              <line x1={0} y1={botPlotH} x2={plotW} y2={botPlotH} stroke={axisColor} />
              <line x1={0} y1={0} x2={0} y2={botPlotH} stroke={axisColor} />
              {X_TICKS.map((t) => (
                <g key={t}>
                  <line x1={sx(t)} y1={botPlotH} x2={sx(t)} y2={botPlotH + 4} stroke={axisColor} />
                  <text x={sx(t)} y={botPlotH + 15} textAnchor="middle" fontSize={10} fill={axisColor}>
                    {t}
                  </text>
                  <text x={-6} y={syBot(t) + 3} textAnchor="end" fontSize={10} fill={axisColor}>
                    {t}
                  </text>
                </g>
              ))}
              <text x={plotW / 2} y={botPlotH + 30} textAnchor="middle" fontSize={11} fill="var(--color-text)">
                observation x
              </text>
              <text
                x={-32}
                y={botPlotH / 2}
                textAnchor="middle"
                fontSize={11}
                fill="var(--color-text)"
                transform={`rotate(-90 ${-32} ${botPlotH / 2})`}
              >
                estimate of θ
              </text>
            </g>
          </svg>
        </div>

        <div className="flex shrink-0 flex-col gap-3 text-sm md:w-56">
          <div className="rounded border p-3" style={{ borderColor: 'var(--color-border)' }}>
            <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
              Per-coordinate MSE
            </div>
            <table className="mt-2 w-full font-mono text-xs">
              <tbody>
                <tr>
                  <td style={{ color: bayesianColors.mle }}>MLE (x)</td>
                  <td className="text-right">{mse.mle.toFixed(3)}</td>
                </tr>
                <tr>
                  <td style={{ color: bayesianColors.fModel }}>Tweedie–KDE</td>
                  <td className="text-right">{mse.kde.toFixed(3)}</td>
                </tr>
                <tr>
                  <td style={{ color: bayesianColors.oracle }}>Oracle Bayes</td>
                  <td className="text-right">{mse.oracle.toFixed(3)}</td>
                </tr>
              </tbody>
            </table>
            <div className="mt-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              {fit.nEval < dN ? 'MSE on 1,000 random points' : `MSE on all ${dN.toLocaleString('en-US')} points`}
            </div>
          </div>
          <div className="rounded border p-3 text-xs" style={{ borderColor: 'var(--color-border)' }}>
            <div className="uppercase tracking-wide" style={{ color: 'var(--color-text-muted)' }}>
              KDE bandwidth
            </div>
            <div className="mt-1 font-mono">
              h = {fit.h.toFixed(3)} ({dMultiplier.toFixed(2)}× Silverman)
            </div>
          </div>
          <p className="text-xs leading-relaxed">
            {Number.isFinite(recovered)
              ? `Tweedie–KDE recovers ${Math.round(recovered)}% of the oracle's improvement over the MLE.`
              : 'The oracle does not improve on the MLE for this sample.'}
          </p>
        </div>
      </div>
    </div>
  );
}
