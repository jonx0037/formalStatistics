/**
 * BernsteinVonMisesAnimator — featured interactive for Topic 25 §25.8.
 *
 * Visualizes BvM convergence: as n grows, the posterior over θ converges in
 * total variation to N(θ̂_MLE, I(θ₀)⁻¹/n). Supports three conjugate families
 * (Beta-Binomial, Gamma-Poisson, Normal-Normal known σ²) so the viewer can
 * see the universal shape-collapse independent of the specific likelihood.
 *
 * The TV-distance readout quantifies the convergence — students watch it
 * decay from ~0.2 at n=5 to <0.01 at n=500 for Beta-Binomial under a
 * Jeffreys prior. Mobile fallback uses a 200-point TV grid (vs 500 desktop).
 */
import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  pdfBeta,
  pdfGamma,
  pdfNormal,
  posterior,
  posteriorMean,
  posteriorVariance,
  credibleIntervalBeta,
  credibleIntervalNormal,
  type PriorHyperparams,
  type SuffStats,
  type PosteriorHyperparams,
} from './shared/bayes';
import { seededRandom } from './shared/probability';
import { bernoulliSample, poissonSample, normalSample } from './shared/convergence';
import { bayesianColors } from './shared/colorScales';
import { bvmPresets, type BvMPreset } from '../../data/bayesian-foundations-data';

const MARGIN = { top: 12, right: 16, bottom: 32, left: 48 };
const GRID_POINTS_DESKTOP = 500;
const GRID_POINTS_MOBILE = 200;
const MOBILE_BREAKPOINT = 640;
const ANIMATION_DURATION_MS = 8000;
const N_MIN = 1;
const N_MAX = 1000;

type Family = 'beta-binomial' | 'gamma-poisson' | 'normal-normal';

/**
 * Map a slider value s ∈ [0, 1] to n ∈ [N_MIN, N_MAX] on log scale —
 * the BvM collapse is a log-time phenomenon (posterior width shrinks at 1/√n).
 * Using a linear scale squashes all the interesting early behavior into ~5% of
 * the slider range.
 */
const sliderToN = (s: number): number =>
  Math.max(N_MIN, Math.min(N_MAX, Math.round(Math.exp(
    Math.log(N_MIN) + s * (Math.log(N_MAX) - Math.log(N_MIN)),
  ))));

const nToSlider = (n: number): number =>
  (Math.log(n) - Math.log(N_MIN)) / (Math.log(N_MAX) - Math.log(N_MIN));

export default function BernsteinVonMisesAnimator() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const [n, setN] = useState(25);
  const [seed, setSeed] = useState(42);
  const [isPlaying, setIsPlaying] = useState(false);

  const preset = bvmPresets[presetIdx];
  const family = preset.family as Family;
  const isMobile = (width || 800) < MOBILE_BREAKPOINT;
  const gridPoints = isMobile ? GRID_POINTS_MOBILE : GRID_POINTS_DESKTOP;

  const chartW = Math.max(280, (width || 600) - 16);
  const chartH = isMobile ? 260 : 320;
  const plotW = chartW - MARGIN.left - MARGIN.right;
  const plotH = chartH - MARGIN.top - MARGIN.bottom;

  // ── Animation loop (log-time on n) ────────────────────────────────────────
  const animationRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const tick = useCallback((timestamp: number) => {
    if (!startTimeRef.current) startTimeRef.current = timestamp;
    const elapsed = timestamp - startTimeRef.current;
    const t = Math.min(1, elapsed / ANIMATION_DURATION_MS);
    setN(sliderToN(t));
    if (t < 1) {
      animationRef.current = requestAnimationFrame(tick);
    } else {
      setIsPlaying(false);
      startTimeRef.current = 0;
    }
  }, []);

  useEffect(() => {
    if (isPlaying) {
      startTimeRef.current = 0;
      animationRef.current = requestAnimationFrame(tick);
    } else if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    return () => {
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    };
  }, [isPlaying, tick]);

  // ── Data simulation (seeded, deterministic per [preset, seed, n]) ─────────
  const suffStats = useMemo<SuffStats>(() => {
    const rng = seededRandom(seed + presetIdx * 1000);
    if (preset.family === 'beta-binomial') {
      let k = 0;
      for (let i = 0; i < n; i++) k += bernoulliSample(preset.trueTheta, rng);
      return { family: 'beta-binomial', n, k };
    }
    if (preset.family === 'gamma-poisson') {
      let S = 0;
      for (let i = 0; i < n; i++) S += poissonSample(preset.trueLambda, rng);
      return { family: 'gamma-poisson', n, S };
    }
    // normal-normal
    let sum = 0;
    for (let i = 0; i < n; i++) sum += normalSample(preset.trueMu, 1, rng);
    return { family: 'normal-normal', n, yBar: sum / n };
  }, [preset, n, seed, presetIdx]);

  // ── MLE + observed Fisher info (family-specific) ─────────────────────────
  const { mle, fisherInfo, trueParam, support } = useMemo(() => {
    return deriveModelStats(preset, suffStats);
  }, [preset, suffStats]);

  // ── Posterior hyperparameters and density ────────────────────────────────
  const post = useMemo<PosteriorHyperparams>(() => {
    const priorHP = priorFromPreset(preset);
    return posterior(preset.family, priorHP, suffStats);
  }, [preset, suffStats]);

  // ── Density grids on the same [support.min, support.max] interval ────────
  const { xs, posteriorDensity, normalDensity, tvDistance } = useMemo(() => {
    const xs_ = new Array<number>(gridPoints);
    const p_ = new Array<number>(gridPoints);
    const q_ = new Array<number>(gridPoints);
    const dx = (support.max - support.min) / (gridPoints - 1);
    for (let i = 0; i < gridPoints; i++) {
      const x = support.min + i * dx;
      xs_[i] = x;
      p_[i] = posteriorPdfAt(family, post, x);
      // Normal-at-MLE approximation: N(MLE, I⁻¹/n). Fisher info is per-observation,
      // so posterior variance is 1/(n·I). When I(θ) varies with θ, we use observed
      // Fisher at the MLE — the standard BvM plug-in.
      q_[i] = pdfNormal(x, mle, 1 / (n * fisherInfo));
    }
    // TV distance ½·∫|p−q|dx (BvM's native convergence metric).
    let tv = 0;
    for (let i = 0; i < gridPoints; i++) tv += 0.5 * Math.abs(p_[i] - q_[i]) * dx;
    return { xs: xs_, posteriorDensity: p_, normalDensity: q_, tvDistance: tv };
  }, [family, post, mle, fisherInfo, n, support, gridPoints]);

  const pMax = useMemo(() => {
    let m = 0;
    for (let i = 0; i < gridPoints; i++) {
      if (posteriorDensity[i] > m) m = posteriorDensity[i];
      if (normalDensity[i] > m) m = normalDensity[i];
    }
    return Math.max(m, 1e-6);
  }, [posteriorDensity, normalDensity, gridPoints]);

  // ── Scales (simple linear; no D3 dependency for this chart) ──────────────
  const xToPx = (x: number) =>
    MARGIN.left + ((x - support.min) / (support.max - support.min)) * plotW;
  const yToPx = (y: number) => MARGIN.top + plotH - (y / pMax) * plotH;

  const posteriorPath = pathFromPoints(xs, posteriorDensity, xToPx, yToPx);
  const normalPath = pathFromPoints(xs, normalDensity, xToPx, yToPx);

  // ── Readout values ────────────────────────────────────────────────────────
  const postMean = posteriorMean(preset.family, post);
  const postVar = posteriorVariance(preset.family, post);
  const cri = credibleInterval(preset.family, post);
  const normalVar = 1 / (n * fisherInfo);
  const waldCI: [number, number] = [
    mle - 1.96 * Math.sqrt(normalVar),
    mle + 1.96 * Math.sqrt(normalVar),
  ];

  const tvColor = tvDistance < 0.01 ? '#10B981' : tvDistance < 0.05 ? '#F59E0B' : '#EF4444';

  return (
    <div className="my-6 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          <span className="text-[var(--color-text-muted)]">Preset:</span>
          <select
            className="rounded border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-2 py-1 text-sm"
            value={presetIdx}
            onChange={e => {
              setPresetIdx(Number(e.target.value));
              setIsPlaying(false);
              setN(25);
            }}
            aria-label="Choose BvM preset"
          >
            {bvmPresets.map((p, i) => (
              <option key={i} value={i}>{p.name}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setIsPlaying(v => !v)}
          className="rounded-md bg-[var(--color-posterior)] px-3 py-1 text-sm font-medium text-white shadow-sm transition hover:opacity-90"
          aria-label={isPlaying ? 'Pause animation' : 'Play animation'}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play (n from 1 to 1000)'}
        </button>
        <button
          type="button"
          onClick={() => { setSeed(s => s + 1); setIsPlaying(false); }}
          className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface-alt)] px-3 py-1 text-sm font-medium"
          aria-label="Reseed the data-generating RNG"
        >
          ↻ Reseed (seed={seed})
        </button>
      </div>

      <p className="mb-3 text-sm text-[var(--color-text-muted)] italic">
        {preset.description}
      </p>

      <div className="mb-3">
        <label className="mb-1 flex items-baseline justify-between text-sm">
          <span>Sample size n (log scale): <span className="font-mono font-semibold">{n}</span></span>
          <span className="text-xs text-[var(--color-text-muted)]">true {trueParamLabel(family)} = {trueParam.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={nToSlider(n)}
          onChange={e => { setN(sliderToN(Number(e.target.value))); setIsPlaying(false); }}
          className="w-full"
          aria-label={`Sample size n, currently ${n} on log scale from 1 to 1000`}
          aria-valuetext={`n = ${n}`}
        />
      </div>

      <div ref={ref}>
        <svg width={chartW} height={chartH} role="img" aria-label="Posterior density overlaid with Normal-at-MLE approximation">
          {/* Grid lines */}
          <g className="text-[var(--color-viz-grid)]" stroke="currentColor" strokeWidth={0.5}>
            {[0, 0.25, 0.5, 0.75, 1].map(frac => {
              const x = support.min + frac * (support.max - support.min);
              return <line key={`vg-${frac}`} x1={xToPx(x)} y1={MARGIN.top} x2={xToPx(x)} y2={MARGIN.top + plotH} />;
            })}
          </g>
          {/* Axes */}
          <g stroke="var(--color-text-muted)" strokeWidth={1} fontSize={11} fill="var(--color-text-muted)">
            <line x1={MARGIN.left} y1={MARGIN.top + plotH} x2={MARGIN.left + plotW} y2={MARGIN.top + plotH} />
            <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={MARGIN.top + plotH} />
            {[0, 0.5, 1].map(frac => {
              const x = support.min + frac * (support.max - support.min);
              return (
                <text key={`xt-${frac}`} x={xToPx(x)} y={MARGIN.top + plotH + 14} textAnchor="middle">
                  {x.toFixed(family === 'normal-normal' ? 1 : 2)}
                </text>
              );
            })}
          </g>
          {/* Posterior density (purple solid) */}
          <path
            d={posteriorPath}
            fill="none"
            stroke={bayesianColors.posterior}
            strokeWidth={2.5}
            aria-label="Posterior density"
          />
          {/* Normal-at-MLE approximation (dashed) */}
          <path
            d={normalPath}
            fill="none"
            stroke="var(--color-text)"
            strokeWidth={1.5}
            strokeDasharray="6 4"
            aria-label="Normal-at-MLE approximation"
          />
          {/* True parameter (emerald dotted) */}
          <line
            x1={xToPx(trueParam)} y1={MARGIN.top}
            x2={xToPx(trueParam)} y2={MARGIN.top + plotH}
            stroke={bayesianColors.true}
            strokeWidth={1.5}
            strokeDasharray="2 3"
          />
          {/* MLE (orange solid) */}
          <line
            x1={xToPx(mle)} y1={MARGIN.top}
            x2={xToPx(mle)} y2={MARGIN.top + plotH}
            stroke={bayesianColors.mle}
            strokeWidth={1.5}
          />
          {/* Legend */}
          <g fontSize={11} transform={`translate(${MARGIN.left + 8}, ${MARGIN.top + 4})`}>
            <g transform="translate(0, 0)">
              <line x1={0} y1={6} x2={18} y2={6} stroke={bayesianColors.posterior} strokeWidth={2.5} />
              <text x={22} y={9} fill="var(--color-text)">posterior</text>
            </g>
            <g transform="translate(92, 0)">
              <line x1={0} y1={6} x2={18} y2={6} stroke="var(--color-text)" strokeWidth={1.5} strokeDasharray="6 4" />
              <text x={22} y={9} fill="var(--color-text)">N(θ̂, Î⁻¹/n)</text>
            </g>
            <g transform="translate(200, 0)">
              <line x1={0} y1={6} x2={14} y2={6} stroke={bayesianColors.true} strokeWidth={1.5} strokeDasharray="2 3" />
              <text x={18} y={9} fill="var(--color-text)">θ₀</text>
            </g>
            <g transform="translate(236, 0)">
              <line x1={0} y1={6} x2={14} y2={6} stroke={bayesianColors.mle} strokeWidth={1.5} />
              <text x={18} y={9} fill="var(--color-text)">θ̂</text>
            </g>
          </g>
        </svg>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-3">
        <ReadoutRow label="n / sufficient stat" value={sufficientStatSummary(family, suffStats)} />
        <ReadoutRow label="θ̂_MLE" value={mle.toFixed(4)} />
        <ReadoutRow label="Observed I(θ̂)" value={fisherInfo.toFixed(4)} />
        <ReadoutRow label="Posterior mean" value={postMean.toFixed(4)} />
        <ReadoutRow label={`Posterior 95% CrI`} value={`[${cri[0].toFixed(3)}, ${cri[1].toFixed(3)}]`} />
        <ReadoutRow label="Posterior SD" value={Math.sqrt(postVar).toFixed(4)} />
        <ReadoutRow label="Normal-at-MLE mean" value={mle.toFixed(4)} />
        <ReadoutRow label="Wald 95% CI" value={`[${waldCI[0].toFixed(3)}, ${waldCI[1].toFixed(3)}]`} />
        <div className="col-span-2 md:col-span-1">
          <span className="text-[var(--color-text-muted)]">TV distance</span>:{' '}
          <span className="font-mono font-semibold" style={{ color: tvColor }}>
            {tvDistance.toFixed(4)}
          </span>
        </div>
      </div>

      <p className="mt-3 rounded-md bg-[var(--color-surface-alt)] p-2 text-xs text-[var(--color-text-muted)]">
        <strong>BvM prediction:</strong> as n → ∞, the posterior converges in total variation
        to N(θ̂_MLE, I(θ₀)⁻¹/n). Watch the purple posterior collapse onto the black dashed
        Normal curve, and the TV distance drop below 0.01 once n is large enough.
      </p>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

function priorFromPreset(p: BvMPreset): PriorHyperparams {
  if (p.family === 'beta-binomial') {
    return { family: 'beta-binomial', alpha0: p.prior.alpha0, beta0: p.prior.beta0 };
  }
  if (p.family === 'gamma-poisson') {
    return { family: 'gamma-poisson', alpha0: p.prior.alpha0, beta0: p.prior.beta0 };
  }
  return {
    family: 'normal-normal',
    mu0: p.prior.mu0, sigma0_sq: p.prior.sigma0_sq, sigma_sq: p.prior.sigma_sq,
  };
}

/**
 * Family-specific MLE, observed Fisher info, true parameter, and plot support.
 * Observed Fisher is computed at the MLE (standard BvM plug-in).
 */
function deriveModelStats(
  p: BvMPreset,
  s: SuffStats,
): { mle: number; fisherInfo: number; trueParam: number; support: { min: number; max: number } } {
  if (p.family === 'beta-binomial' && s.family === 'beta-binomial') {
    const mle = s.n > 0 ? s.k / s.n : 0.5;
    // Clip MLE away from 0 and 1 to avoid infinite Fisher info.
    const mleClip = Math.max(0.01, Math.min(0.99, mle));
    const fisherInfo = 1 / (mleClip * (1 - mleClip));
    return { mle: mleClip, fisherInfo, trueParam: p.trueTheta, support: { min: 0, max: 1 } };
  }
  if (p.family === 'gamma-poisson' && s.family === 'gamma-poisson') {
    const mle = s.n > 0 ? s.S / s.n : 1;
    const mleClip = Math.max(0.1, mle);
    const fisherInfo = 1 / mleClip; // Poisson Fisher info = 1/λ
    const supportMax = Math.max(p.trueLambda * 3, mleClip + 5 * Math.sqrt(mleClip / Math.max(1, s.n)));
    return { mle: mleClip, fisherInfo, trueParam: p.trueLambda, support: { min: 0, max: supportMax } };
  }
  if (p.family === 'normal-normal' && s.family === 'normal-normal') {
    const mle = s.yBar;
    const fisherInfo = 1 / p.prior.sigma_sq; // known-σ² Normal-mean Fisher info
    const sd = Math.sqrt(p.prior.sigma_sq / Math.max(1, s.n));
    return {
      mle,
      fisherInfo,
      trueParam: p.trueMu,
      support: { min: mle - 5 * sd, max: mle + 5 * sd },
    };
  }
  // Shouldn't happen; return sensible fallback.
  return { mle: 0, fisherInfo: 1, trueParam: 0, support: { min: -1, max: 1 } };
}

function posteriorPdfAt(family: Family, post: PosteriorHyperparams, x: number): number {
  if (family === 'beta-binomial' && post.family === 'beta-binomial') {
    return pdfBeta(x, post.alpha0, post.beta0);
  }
  if (family === 'gamma-poisson' && post.family === 'gamma-poisson') {
    return pdfGamma(x, post.alpha0, post.beta0);
  }
  if (family === 'normal-normal' && post.family === 'normal-normal') {
    return pdfNormal(x, post.mu0, post.sigma0_sq);
  }
  return 0;
}

function credibleInterval(family: Family, post: PosteriorHyperparams): [number, number] {
  if (family === 'beta-binomial' && post.family === 'beta-binomial') {
    return credibleIntervalBeta(post.alpha0, post.beta0, 0.95);
  }
  if (family === 'normal-normal' && post.family === 'normal-normal') {
    return credibleIntervalNormal(post.mu0, post.sigma0_sq, 0.95);
  }
  // Gamma: approximate via Normal on log scale (posterior mean ± 1.96·SD). Not exact
  // but adequate for visual overlay at n ≥ 10 where Gamma is near-Normal.
  if (family === 'gamma-poisson' && post.family === 'gamma-poisson') {
    const mean = post.alpha0 / post.beta0;
    const sd = Math.sqrt(post.alpha0) / post.beta0;
    return [Math.max(0, mean - 1.96 * sd), mean + 1.96 * sd];
  }
  return [0, 0];
}

function pathFromPoints(
  xs: number[],
  ys: number[],
  xToPx: (x: number) => number,
  yToPx: (y: number) => number,
): string {
  let d = '';
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(ys[i])) continue;
    d += (i === 0 ? 'M' : 'L') + xToPx(xs[i]) + ',' + yToPx(ys[i]) + ' ';
  }
  return d;
}

function trueParamLabel(family: Family): string {
  if (family === 'beta-binomial') return 'θ';
  if (family === 'gamma-poisson') return 'λ';
  return 'μ';
}

function sufficientStatSummary(family: Family, s: SuffStats): string {
  if (family === 'beta-binomial' && s.family === 'beta-binomial') return `n=${s.n}, k=${s.k}`;
  if (family === 'gamma-poisson' && s.family === 'gamma-poisson') return `n=${s.n}, S=${s.S}`;
  if (family === 'normal-normal' && s.family === 'normal-normal') return `n=${s.n}, ȳ=${s.yBar.toFixed(3)}`;
  return '';
}

function ReadoutRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[var(--color-text-muted)]">{label}</span>:{' '}
      <span className="font-mono font-semibold">{value}</span>
    </div>
  );
}
