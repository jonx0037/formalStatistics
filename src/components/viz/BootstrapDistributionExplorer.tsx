/**
 * BootstrapDistributionExplorer — Topic 31 §31.3 FEATURED component.
 *
 * Three synchronized panels demonstrate the Efron–Bickel–Freedman theorem
 * visually:
 *
 *   Panel 1 (MC reference) — histogram of 10,000 iid draws of the statistic
 *     from the TRUE preset distribution at sample size n. This is an
 *     approximation to H_n, the sampling distribution of the statistic.
 *     Memoized by (preset, statistic, n) — recomputing on B/reseed changes
 *     would dominate the render cost.
 *
 *   Panel 2 (bootstrap) — histogram of B bootstrap replicates of the
 *     statistic from ONE observed sample of size n. This is H_n*.
 *     On a shared x-axis with Panel 1 so the reader can eyeball the match.
 *
 *   Panel 3 (KS distance curve) — KS(H_n* at various B checkpoints, H_n)
 *     on log-B x-axis. For fixed n, the curve decays at rate 1/√B (Monte-
 *     Carlo error) toward a floor set by the actual H_n ≠ H_n* gap. The
 *     floor shrinks as n grows — that IS bootstrap consistency.
 *
 * Memoization is load-bearing (brief §5.1). The MC reference caches by
 * (preset, statistic, n). The observed sample caches by (preset, n, seed).
 * The bootstrap replicates cache by (sample-hash, B, seed). B or reseed
 * changes should not trigger MC recomputation.
 *
 * Statistic is user-selectable: mean, median, variance (Bessel-corrected).
 */
import { useEffect, useMemo, useState } from 'react';

import { createSeededRng, type SeededRng } from './shared/bayes';
import { kolmogorovDistanceSorted } from './shared/nonparametric';
import { allPresets, type DistributionPreset, type StatKey } from '../../data/nonparametric-data';
import { bootstrapColors } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

const MC_DRAWS = 10_000;
const KS_CHECKPOINTS = [100, 250, 500, 1000, 2500, 5000] as const;
const HIST_BINS = 50;
const PANEL_WIDTH = 300;
const PANEL_HEIGHT = 220;
const MARGIN = { top: 18, right: 14, bottom: 34, left: 42 };

const STATISTICS: readonly StatKey[] = ['mean', 'median', 'variance'];

const STAT_LABEL: Record<StatKey, string> = {
  mean: 'Mean',
  median: 'Median',
  variance: 'Variance',
};

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

function drawSample(preset: DistributionPreset, n: number, rng: SeededRng): number[] {
  if (preset.sample) return preset.sample(n, rng);
  const out: number[] = new Array(n);
  const uniform = (): number => rng.random();
  for (let i = 0; i < n; i++) out[i] = preset.sampler(uniform);
  return out;
}

function statistic(key: StatKey): (sample: readonly number[]) => number {
  switch (key) {
    case 'mean':
      return (x) => {
        let s = 0;
        for (const v of x) s += v;
        return s / x.length;
      };
    case 'median':
      return (x) => {
        const s = [...x].sort((a, b) => a - b);
        const mid = Math.floor(s.length / 2);
        return s.length % 2 === 0 ? 0.5 * (s[mid - 1] + s[mid]) : s[mid];
      };
    case 'variance':
      return (x) => {
        const n = x.length;
        if (n < 2) return 0;
        let sum = 0;
        for (const v of x) sum += v;
        const mean = sum / n;
        let ss = 0;
        for (const v of x) {
          const d = v - mean;
          ss += d * d;
        }
        return ss / (n - 1);
      };
  }
}

function linspace(a: number, b: number, n: number): number[] {
  if (n < 2) return [a];
  const step = (b - a) / (n - 1);
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) out[i] = a + i * step;
  return out;
}

function scaleLinear(dMin: number, dMax: number, rMin: number, rMax: number) {
  const span = dMax - dMin || 1;
  return (v: number) => rMin + ((v - dMin) / span) * (rMax - rMin);
}

function scaleLog(dMin: number, dMax: number, rMin: number, rMax: number) {
  const lMin = Math.log(dMin);
  const span = Math.log(dMax) - lMin || 1;
  return (v: number) => rMin + ((Math.log(v) - lMin) / span) * (rMax - rMin);
}

function fmt(v: number, d = 3): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

function buildHistogram(
  values: readonly number[],
  xMin: number,
  xMax: number,
): { counts: number[]; binW: number } {
  const binW = (xMax - xMin) / HIST_BINS;
  const counts = new Array(HIST_BINS).fill(0);
  for (const v of values) {
    const k = Math.min(HIST_BINS - 1, Math.max(0, Math.floor((v - xMin) / binW)));
    counts[k]++;
  }
  return { counts, binW };
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function BootstrapDistributionExplorer() {
  const [presetIdx, setPresetIdx] = useState(0); // Normal by default
  const [statKey, setStatKey] = useState<StatKey>('mean');
  const [n, setN] = useState(50);
  const [B, setB] = useState(1000);
  const [seed, setSeed] = useState(42);
  const [isMobile, setIsMobile] = useState(false);

  const preset = allPresets[presetIdx];
  const statFn = useMemo(() => statistic(statKey), [statKey]);

  // Mobile detection for B-max cap. useEffect (not useMemo) — this is a side
  // effect with event-listener cleanup, not a pure computation.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(max-width: 767px)');
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // ── Panel 1: MC reference — memoized by (preset, statistic, n) ────────────
  const mcReference = useMemo(() => {
    const rng = createSeededRng(0xC0DE_FACE); // fixed MC seed for reproducibility
    const draws: number[] = new Array(MC_DRAWS);
    for (let i = 0; i < MC_DRAWS; i++) {
      draws[i] = statFn(drawSample(preset, n, rng));
    }
    const sorted = [...draws].sort((a, b) => a - b);
    return { draws, sorted };
  }, [preset, statFn, n]);

  // ── Observed sample — memoized by (preset, n, seed) ──────────────────────
  const observedSample = useMemo(() => {
    const rng = createSeededRng(seed + 100_000);
    return drawSample(preset, n, rng);
  }, [preset, n, seed]);

  // Mobile caps B at 1000; desktop at 5000. We precompute the full replicate
  // budget (effectiveMaxB) up front so the bootstrap cache only invalidates
  // when the budget itself shifts — not on every B-slider tick below the cap.
  const effectiveMaxB = isMobile ? 1000 : 5000;

  // ── Bootstrap replicates (unsorted — so we can slice the first B_k in order).
  // Buffer `resample` hoisted outside the loop to avoid effectiveMaxB × n
  // array allocations. Deps: observedSample / statFn / n / effectiveMaxB / seed —
  // deliberately NOT B, since any B ≤ effectiveMaxB reads from the same cache.
  const unsortedReps = useMemo(() => {
    const rng = createSeededRng(seed + 200_000);
    const out: number[] = new Array(effectiveMaxB);
    const resample: number[] = new Array(n);
    for (let b = 0; b < effectiveMaxB; b++) {
      for (let i = 0; i < n; i++) {
        resample[i] = observedSample[Math.floor(rng.random() * n)];
      }
      out[b] = statFn(resample);
    }
    return out;
  }, [observedSample, statFn, n, effectiveMaxB, seed]);

  // Current bootstrap distribution (first B replicates, in draw order)
  const currentReps = useMemo(() => unsortedReps.slice(0, B), [unsortedReps, B]);

  // ── Panel 3: KS distance curve over checkpoint B values ───────────────────
  // Each checkpoint slice must be sorted (draw order → ascending) so the KS
  // helper can walk both sequences in linear time. `kolmogorovDistanceSorted`
  // assumes pre-sorted input — no redundant internal sort.
  const ksCurve = useMemo(() => {
    const mcSorted = mcReference.sorted;
    const maxCheckpoint = Math.min(B, unsortedReps.length);
    const points: Array<{ B: number; dKS: number }> = [];
    for (const bk of KS_CHECKPOINTS) {
      if (bk > maxCheckpoint) break;
      const slice = unsortedReps.slice(0, bk).sort((a, b) => a - b);
      points.push({ B: bk, dKS: kolmogorovDistanceSorted(slice, mcSorted) });
    }
    return points;
  }, [unsortedReps, mcReference.sorted, B]);

  // ── Shared x-range for Panels 1 & 2 ────────────────────────────────────────
  const shared = useMemo(() => {
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const v of mcReference.draws) {
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    for (const v of currentReps) {
      if (v < xMin) xMin = v;
      if (v > xMax) xMax = v;
    }
    const pad = (xMax - xMin) * 0.04 || 0.1;
    xMin -= pad;
    xMax += pad;
    const mcHist = buildHistogram(mcReference.draws, xMin, xMax);
    const bootHist = buildHistogram(currentReps, xMin, xMax);
    // Normalise to density for direct visual comparison
    const denMc = MC_DRAWS * mcHist.binW;
    const denBoot = currentReps.length * bootHist.binW;
    const mcDens = mcHist.counts.map((c) => c / denMc);
    const bootDens = bootHist.counts.map((c) => c / denBoot);
    let yMax = 0;
    for (const v of mcDens) if (v > yMax) yMax = v;
    for (const v of bootDens) if (v > yMax) yMax = v;
    return { xMin, xMax, binW: mcHist.binW, mcDens, bootDens, yMax: yMax * 1.12 };
  }, [mcReference.draws, currentReps]);

  // Observed θ̂ vertical marker
  const thetaHat = useMemo(() => statFn(observedSample), [observedSample, statFn]);

  const maxB = isMobile ? 1000 : 5000;

  // ────────────────────────────────────────────────────────────────────────
  // Rendering
  // ────────────────────────────────────────────────────────────────────────

  const renderAxes = (
    xDomain: [number, number],
    yDomain: [number, number],
    xScaler: (v: number) => number,
    yScaler: (v: number) => number,
    xTickCount = 5,
    yTickCount = 4,
    xLabel?: string,
    yLabel?: string,
    logX = false,
  ) => {
    const xTicks = logX
      ? KS_CHECKPOINTS.filter((t) => t >= xDomain[0] && t <= xDomain[1])
      : linspace(xDomain[0], xDomain[1], xTickCount);
    const yTicks = linspace(yDomain[0], yDomain[1], yTickCount);
    return (
      <>
        <line
          x1={MARGIN.left}
          y1={PANEL_HEIGHT - MARGIN.bottom}
          x2={PANEL_WIDTH - MARGIN.right}
          y2={PANEL_HEIGHT - MARGIN.bottom}
          stroke="#64748b"
        />
        <line
          x1={MARGIN.left}
          y1={MARGIN.top}
          x2={MARGIN.left}
          y2={PANEL_HEIGHT - MARGIN.bottom}
          stroke="#64748b"
        />
        {xTicks.map((v, i) => (
          <text
            key={`xt-${i}`}
            x={xScaler(v)}
            y={PANEL_HEIGHT - MARGIN.bottom + 14}
            textAnchor="middle"
            className="text-[10px] fill-slate-500"
          >
            {logX ? v.toString() : v.toFixed(2)}
          </text>
        ))}
        {yTicks.map((v, i) => (
          <text
            key={`yt-${i}`}
            x={MARGIN.left - 4}
            y={yScaler(v) + 3}
            textAnchor="end"
            className="text-[10px] fill-slate-500"
          >
            {v.toFixed(3)}
          </text>
        ))}
        {xLabel && (
          <text
            x={PANEL_WIDTH / 2}
            y={PANEL_HEIGHT - 4}
            textAnchor="middle"
            className="text-[10px] fill-slate-500"
          >
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            x={12}
            y={PANEL_HEIGHT / 2}
            textAnchor="middle"
            transform={`rotate(-90, 12, ${PANEL_HEIGHT / 2})`}
            className="text-[10px] fill-slate-500"
          >
            {yLabel}
          </text>
        )}
      </>
    );
  };

  const renderHistogramBars = (
    dens: number[],
    xMin: number,
    binW: number,
    xScaler: (v: number) => number,
    yScaler: (v: number) => number,
    fill: string,
    opacity: number,
  ) => {
    const y0 = yScaler(0);
    const barW = Math.max(0.5, xScaler(xMin + binW) - xScaler(xMin) - 0.5);
    return dens.map((v, i) => (
      <rect
        key={`b-${i}`}
        x={xScaler(xMin + i * binW)}
        y={yScaler(v)}
        width={barW}
        height={y0 - yScaler(v)}
        fill={fill}
        opacity={opacity}
      />
    ));
  };

  // Panel 1 scales
  const p1x = scaleLinear(shared.xMin, shared.xMax, MARGIN.left, PANEL_WIDTH - MARGIN.right);
  const p1y = scaleLinear(0, shared.yMax, PANEL_HEIGHT - MARGIN.bottom, MARGIN.top);

  // Panel 3 scales (log x)
  const ksXScaler = scaleLog(
    KS_CHECKPOINTS[0],
    KS_CHECKPOINTS[KS_CHECKPOINTS.length - 1],
    MARGIN.left,
    PANEL_WIDTH - MARGIN.right,
  );
  const ksYMax = useMemo(() => {
    let m = 0;
    for (const p of ksCurve) if (p.dKS > m) m = p.dKS;
    return m > 0 ? m * 1.15 : 0.1;
  }, [ksCurve]);
  const ksYScaler = scaleLinear(0, ksYMax, PANEL_HEIGHT - MARGIN.bottom, MARGIN.top);
  const ksPath = ksCurve
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${ksXScaler(p.B).toFixed(1)},${ksYScaler(p.dKS).toFixed(1)}`)
    .join('');

  return (
    <div className="my-8 not-prose border border-slate-200 rounded-lg bg-white p-4 md:p-6 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 text-sm text-slate-600 dark:text-slate-300">
        <span className="font-semibold text-slate-800 dark:text-slate-100">Bootstrap distribution vs sampling distribution</span>
        <span className="mx-2 text-slate-400">·</span>
        <span>
          The bootstrap replaces repeated sampling from the true distribution with repeated resampling from one observed sample.
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <div className="text-xs text-slate-500 mb-1">Preset</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 dark:border-slate-700 dark:bg-slate-800">
            {allPresets.map((p, i) => (
              <button
                key={p.key}
                type="button"
                onClick={() => setPresetIdx(i)}
                className={`px-2.5 py-1 text-xs rounded ${
                  presetIdx === i
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                    : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">Statistic</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 dark:border-slate-700 dark:bg-slate-800">
            {STATISTICS.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setStatKey(k)}
                className={`px-2.5 py-1 text-xs rounded ${
                  statKey === k
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                    : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                {STAT_LABEL[k]}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">n = {n}</div>
          <input
            type="range"
            min={10}
            max={500}
            step={10}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-32"
            aria-label="Sample size n"
          />
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">B = {B}</div>
          <input
            type="range"
            min={100}
            max={maxB}
            step={100}
            value={Math.min(B, maxB)}
            onChange={(e) => setB(Number(e.target.value))}
            className="w-32"
            aria-label="Bootstrap replicates B"
          />
        </div>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          className="px-3 py-1.5 text-xs bg-slate-800 text-white rounded hover:bg-slate-700 dark:bg-slate-600 dark:hover:bg-slate-500"
        >
          Reseed
        </button>
      </div>

      {/* Three panels */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* Panel 1: MC reference */}
        <div>
          <svg viewBox={`0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`} className="w-full h-auto" role="img" aria-label="Monte Carlo reference sampling distribution">
            {renderHistogramBars(shared.mcDens, shared.xMin, shared.binW, p1x, p1y, bootstrapColors.reference, 0.6)}
            {renderAxes([shared.xMin, shared.xMax], [0, shared.yMax], p1x, p1y, 4, 3, STAT_LABEL[statKey], 'density')}
            <text x={MARGIN.left} y={MARGIN.top - 4} className="text-[11px] fill-slate-700 dark:fill-slate-200">
              Sampling distribution H_n
            </text>
            <text x={PANEL_WIDTH - MARGIN.right} y={MARGIN.top - 4} textAnchor="end" className="text-[10px] fill-slate-500">
              MC × {MC_DRAWS}
            </text>
          </svg>
        </div>

        {/* Panel 2: Bootstrap */}
        <div>
          <svg viewBox={`0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`} className="w-full h-auto" role="img" aria-label="Bootstrap distribution">
            {renderHistogramBars(shared.bootDens, shared.xMin, shared.binW, p1x, p1y, bootstrapColors.resample, 0.6)}
            {/* θ̂ marker */}
            <line
              x1={p1x(thetaHat)}
              y1={MARGIN.top}
              x2={p1x(thetaHat)}
              y2={PANEL_HEIGHT - MARGIN.bottom}
              stroke="#0f172a"
              strokeDasharray="3 3"
              strokeWidth={1.2}
            />
            {renderAxes([shared.xMin, shared.xMax], [0, shared.yMax], p1x, p1y, 4, 3, `${STAT_LABEL[statKey]}*`, 'density')}
            <text x={MARGIN.left} y={MARGIN.top - 4} className="text-[11px] fill-slate-700 dark:fill-slate-200">
              Bootstrap distribution H_n*
            </text>
            <text x={PANEL_WIDTH - MARGIN.right} y={MARGIN.top - 4} textAnchor="end" className="text-[10px] fill-slate-500">
              B = {B}
            </text>
            <text x={p1x(thetaHat)} y={PANEL_HEIGHT - MARGIN.bottom + 14} textAnchor="middle" className="text-[9px] fill-slate-700">
              θ̂ = {fmt(thetaHat, 2)}
            </text>
          </svg>
        </div>

        {/* Panel 3: KS distance curve */}
        <div>
          <svg viewBox={`0 0 ${PANEL_WIDTH} ${PANEL_HEIGHT}`} className="w-full h-auto" role="img" aria-label="KS distance vs B">
            {renderAxes(
              [KS_CHECKPOINTS[0], KS_CHECKPOINTS[KS_CHECKPOINTS.length - 1]],
              [0, ksYMax],
              ksXScaler,
              ksYScaler,
              0,
              4,
              'B (log scale)',
              'KS distance',
              true,
            )}
            {ksPath && (
              <path d={ksPath} fill="none" stroke={bootstrapColors.bca} strokeWidth={1.8} />
            )}
            {ksCurve.map((p, i) => (
              <circle
                key={`ksp-${i}`}
                cx={ksXScaler(p.B)}
                cy={ksYScaler(p.dKS)}
                r={2.5}
                fill={bootstrapColors.bca}
              />
            ))}
            <text x={MARGIN.left} y={MARGIN.top - 4} className="text-[11px] fill-slate-700 dark:fill-slate-200">
              D_KS(H_n*, H_n) vs B
            </text>
          </svg>
          {ksCurve.length > 0 && (
            <div className="mt-1.5 text-[11px] text-slate-600 dark:text-slate-300 tabular-nums">
              Current D_KS at B = {B}:{' '}
              <span className="font-medium" style={{ color: bootstrapColors.bca }}>
                {fmt(ksCurve[ksCurve.length - 1]?.dKS ?? NaN, 4)}
              </span>
            </div>
          )}
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
        Pick a preset, a statistic, and a sample size. Panel 1 shows the true sampling distribution
        (from 10 000 Monte-Carlo draws); Panel 2 shows the bootstrap distribution built from one
        observed sample of size n. Panel 3 tracks the KS distance between them as B grows. Watch
        it decay at rate 1/√B toward a floor that depends only on (preset, statistic, n) — the
        floor is the gap Theorem 3 shrinks to zero as n → ∞.
      </p>
    </div>
  );
}
