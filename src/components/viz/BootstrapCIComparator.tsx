/**
 * BootstrapCIComparator — Topic 31 §31.4.
 *
 * Five parallel CI constructions for the sample mean on a single bootstrap
 * pass, displayed as a vertically-stacked number line. Below it, a rolling
 * coverage table watches all five methods as the reader draws repeated
 * samples from a fixed preset.
 *
 *   1. Percentile   — α/2 and 1−α/2 quantiles of θ*.
 *   2. Basic (Hall) — reflects percentile endpoints around θ̂.
 *   3. BCa          — bias-corrected accelerated (EFR1987); second-order.
 *   4. Studentized  — bootstrap-t pivot; also second-order.
 *   5. Wald-t       — θ̂ ± t_{n−1, 1−α/2} · SE. The parametric reference.
 *
 * Methods 1–3 share a single outer bootstrap pass over B = 1000 replicates
 * (plus BCa's O(n) jackknife). Method 4 is the expensive one (outer B × inner
 * BInner = 1000 × 30 inner resamples per sample). Method 5 is closed-form.
 *
 * "Draw 100 samples" runs the full simulation in chunks of 10 via
 * setTimeout(0, next) so the React tree can re-render and the progress bar
 * animates. Disabled below the md: breakpoint (brief §5.2 mobile fallback).
 */
import { useCallback, useEffect, useMemo, useState } from 'react';

import { createSeededRng, type SeededRng } from './shared/bayes';
import {
  nonparametricBootstrap,
  percentileCI,
  basicCI,
  bcaCI,
  studentizedCI,
  type BootstrapCI,
} from './shared/nonparametric';
import {
  expPreset,
  normalPreset,
  betaPreset,
  heavyTailedPreset,
  type DistributionPreset,
} from '../../data/nonparametric-data';
import { bootstrapColors } from './shared/colorScales';

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/**
 * Preset list for the CI comparator. Ordered by skewness impact on coverage —
 * Exp(1) and Beta(2,5) punish symmetric-CI methods (percentile lagging Wald);
 * Normal and t_3 are symmetric but t_3 has heavy tails that degrade Wald.
 */
const PRESETS: ReadonlyArray<DistributionPreset> = [
  expPreset,
  normalPreset,
  betaPreset,
  heavyTailedPreset,
];

const NS = [15, 30, 50, 100] as const;
const OUTER_B = 1000;
const INNER_B = 30; // studentized inner bootstrap
const ALPHA = 0.05;
const BATCH_SIZE = 100;
const BATCH_CHUNK = 10;

const METHODS = ['percentile', 'basic', 'bca', 'studentized', 'wald'] as const;
type Method = (typeof METHODS)[number];

const METHOD_LABEL: Record<Method, string> = {
  percentile: 'Percentile',
  basic: 'Basic (Hall)',
  bca: 'BCa',
  studentized: 'Studentized',
  wald: 'Wald-t',
};

// Student-t inverse CDF — used by Wald-t. Closed-form only for select df;
// we use the standard-normal quantile with a small-n correction via the
// Cornish-Fisher expansion. For df ≥ 15 this is accurate to ~1%; for this
// demo (n ∈ {15, 30, 50, 100}) that's sufficient.
function tQuantile(p: number, df: number): number {
  // Hill's 1970 approximation to the Student-t quantile.
  const z = normalQuantile(p);
  const g1 = (z * z * z + z) / 4;
  const g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96;
  return z + g1 / df + g2 / (df * df);
}

function normalQuantile(p: number): number {
  // Beasley-Springer-Moro inverse normal approximation (matches bayes.ts).
  if (p <= 0 || p >= 1) throw new Error(`normalQuantile: p=${p} must be in (0, 1)`);
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}

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

function sampleMean(x: readonly number[]): number {
  let s = 0;
  for (const v of x) s += v;
  return s / x.length;
}

function sampleSeMean(x: readonly number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  const mean = sampleMean(x);
  let ss = 0;
  for (const v of x) {
    const d = v - mean;
    ss += d * d;
  }
  return Math.sqrt(ss / (n - 1) / n);
}

function computeAllCis(
  x: readonly number[],
  outerReps: readonly number[],
  rngStud: SeededRng,
): Record<Method, BootstrapCI | { lower: number; upper: number; method: 'wald' }> {
  const thetaHat = sampleMean(x);
  const seHat = sampleSeMean(x);
  const n = x.length;
  const waldHalf = tQuantile(1 - ALPHA / 2, n - 1) * seHat;
  return {
    percentile: percentileCI(outerReps, ALPHA),
    basic: basicCI(thetaHat, outerReps, ALPHA),
    bca: bcaCI(x, sampleMean, outerReps, ALPHA),
    studentized: studentizedCI(x, sampleMean, sampleSeMean, OUTER_B, INNER_B, ALPHA, rngStud),
    wald: { lower: thetaHat - waldHalf, upper: thetaHat + waldHalf, method: 'wald' },
  };
}

function fmt(v: number, d = 2): string {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(d);
}

// ────────────────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────────────────

export default function BootstrapCIComparator() {
  const [presetIdx, setPresetIdx] = useState(0); // Exp by default
  const [n, setN] = useState<(typeof NS)[number]>(30);
  const [seed, setSeed] = useState(42);
  const [drawIdx, setDrawIdx] = useState(0); // increments per sample drawn
  const [coverage, setCoverage] = useState<Record<Method, { captured: number; total: number }>>({
    percentile: { captured: 0, total: 0 },
    basic: { captured: 0, total: 0 },
    bca: { captured: 0, total: 0 },
    studentized: { captured: 0, total: 0 },
    wald: { captured: 0, total: 0 },
  });
  const [currentCis, setCurrentCis] = useState<ReturnType<typeof computeAllCis> | null>(null);
  const [currentThetaHat, setCurrentThetaHat] = useState<number | null>(null);
  const [isBatching, setIsBatching] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [isMobile, setIsMobile] = useState(false);

  const preset = PRESETS[presetIdx];
  const thetaTrue = preset.trueParameterByStat?.mean ?? presetMeanFallback(preset);

  // ── Mobile detection (for "Draw 100" disable) ─────────────────────────────
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(max-width: 767px)');
    setIsMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // ── Reset coverage when preset / n / seed changes ─────────────────────────
  useEffect(() => {
    setCoverage({
      percentile: { captured: 0, total: 0 },
      basic: { captured: 0, total: 0 },
      bca: { captured: 0, total: 0 },
      studentized: { captured: 0, total: 0 },
      wald: { captured: 0, total: 0 },
    });
    setCurrentCis(null);
    setCurrentThetaHat(null);
    setDrawIdx(0);
  }, [preset, n, seed]);

  // ── Single draw computation ───────────────────────────────────────────────
  const computeForIdx = useCallback(
    (idx: number) => {
      const rng = createSeededRng(seed * 10_000 + idx * 137);
      const sample = drawSample(preset, n, rng);
      const bootRng = createSeededRng(seed * 10_000 + idx * 137 + 1);
      const outerReps = nonparametricBootstrap(sample, sampleMean, OUTER_B, bootRng);
      const studRng = createSeededRng(seed * 10_000 + idx * 137 + 2);
      const cis = computeAllCis(sample, outerReps, studRng);
      const thetaHat = sampleMean(sample);
      return { cis, thetaHat };
    },
    [preset, n, seed],
  );

  const drawOne = useCallback(() => {
    const { cis, thetaHat } = computeForIdx(drawIdx);
    setCoverage((prev) => {
      const next = { ...prev };
      for (const m of METHODS) {
        const ci = cis[m];
        const captured = ci.lower <= thetaTrue && thetaTrue <= ci.upper;
        next[m] = {
          captured: prev[m].captured + (captured ? 1 : 0),
          total: prev[m].total + 1,
        };
      }
      return next;
    });
    setCurrentCis(cis);
    setCurrentThetaHat(thetaHat);
    setDrawIdx((i) => i + 1);
  }, [computeForIdx, drawIdx, thetaTrue]);

  // ── Batch draws (100) in chunks to keep the UI responsive ─────────────────
  const runBatch = useCallback(() => {
    if (isBatching) return;
    setIsBatching(true);
    setBatchProgress(0);

    let processed = 0;
    const startIdx = drawIdx;
    const captureCounts: Record<Method, number> = {
      percentile: 0, basic: 0, bca: 0, studentized: 0, wald: 0,
    };
    let lastCis: ReturnType<typeof computeAllCis> | null = null;
    let lastTheta: number | null = null;

    const step = () => {
      const chunkEnd = Math.min(processed + BATCH_CHUNK, BATCH_SIZE);
      for (let i = processed; i < chunkEnd; i++) {
        const { cis, thetaHat } = computeForIdx(startIdx + i);
        for (const m of METHODS) {
          const ci = cis[m];
          if (ci.lower <= thetaTrue && thetaTrue <= ci.upper) captureCounts[m]++;
        }
        lastCis = cis;
        lastTheta = thetaHat;
      }
      processed = chunkEnd;
      setBatchProgress(processed / BATCH_SIZE);
      if (processed < BATCH_SIZE) {
        setTimeout(step, 0);
      } else {
        setCoverage((prev) => {
          const next: typeof prev = { ...prev };
          for (const m of METHODS) {
            next[m] = {
              captured: prev[m].captured + captureCounts[m],
              total: prev[m].total + BATCH_SIZE,
            };
          }
          return next;
        });
        if (lastCis && lastTheta !== null) {
          setCurrentCis(lastCis);
          setCurrentThetaHat(lastTheta);
        }
        setDrawIdx(startIdx + BATCH_SIZE);
        setIsBatching(false);
      }
    };
    setTimeout(step, 0);
  }, [computeForIdx, drawIdx, isBatching, thetaTrue]);

  // ── Number-line rendering ─────────────────────────────────────────────────
  const ciPanel = useMemo(() => {
    if (!currentCis || currentThetaHat === null) return null;
    const allEndpoints = [
      ...METHODS.map((m) => currentCis[m].lower),
      ...METHODS.map((m) => currentCis[m].upper),
      thetaTrue,
      currentThetaHat,
    ];
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const v of allEndpoints) {
      if (Number.isFinite(v)) {
        if (v < xMin) xMin = v;
        if (v > xMax) xMax = v;
      }
    }
    const pad = (xMax - xMin) * 0.08 || 0.1;
    xMin -= pad;
    xMax += pad;
    return { xMin, xMax };
  }, [currentCis, currentThetaHat, thetaTrue]);

  const PANEL_W = 420;
  const PANEL_H = 180;
  const M = { top: 20, right: 20, bottom: 34, left: 100 };
  const rowH = (PANEL_H - M.top - M.bottom) / METHODS.length;

  return (
    <div className="my-8 not-prose border border-slate-200 rounded-lg bg-white p-4 md:p-6 dark:border-slate-700 dark:bg-slate-900">
      <div className="mb-3 text-sm text-slate-600 dark:text-slate-300">
        <span className="font-semibold text-slate-800 dark:text-slate-100">Five 95 % CIs side by side</span>
        <span className="mx-2 text-slate-400">·</span>
        <span>
          Draw samples to watch how percentile / basic / BCa / studentized / Wald-t intervals cover the true mean.
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-4 items-end">
        <div>
          <div className="text-xs text-slate-500 mb-1">Preset</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 dark:border-slate-700 dark:bg-slate-800">
            {PRESETS.map((p, i) => (
              <button
                key={p.key}
                type="button"
                onClick={() => setPresetIdx(i)}
                disabled={isBatching}
                className={`px-2.5 py-1 text-xs rounded ${
                  presetIdx === i
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                    : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                } disabled:opacity-50`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="text-xs text-slate-500 mb-1">n</div>
          <div className="inline-flex rounded-md border border-slate-300 bg-slate-50 p-0.5 dark:border-slate-700 dark:bg-slate-800">
            {NS.map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setN(v)}
                disabled={isBatching}
                className={`px-2 py-1 text-xs rounded ${
                  n === v
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                    : 'text-slate-600 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                } disabled:opacity-50`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          onClick={drawOne}
          disabled={isBatching}
          className="px-3 py-1.5 text-xs bg-slate-800 text-white rounded hover:bg-slate-700 disabled:opacity-50 dark:bg-slate-600 dark:hover:bg-slate-500"
        >
          Draw another sample
        </button>

        <button
          type="button"
          onClick={runBatch}
          disabled={isBatching || isMobile}
          title={isMobile ? 'Batch simulation disabled on narrow screens' : ''}
          className="px-3 py-1.5 text-xs bg-rose-600 text-white rounded hover:bg-rose-500 disabled:opacity-50"
        >
          Draw 100 samples
        </button>

        <button
          type="button"
          onClick={() => setSeed((s) => s + 1)}
          disabled={isBatching}
          className="px-3 py-1.5 text-xs bg-slate-200 text-slate-700 rounded hover:bg-slate-300 disabled:opacity-50 dark:bg-slate-700 dark:text-slate-200 dark:hover:bg-slate-600"
        >
          Reseed
        </button>
      </div>

      {/* Progress bar */}
      {isBatching && (
        <div className="mb-3 h-1.5 bg-slate-100 rounded overflow-hidden dark:bg-slate-800">
          <div
            className="h-full bg-rose-500 transition-all duration-100"
            style={{ width: `${(batchProgress * 100).toFixed(1)}%` }}
          />
        </div>
      )}

      {/* CI number line */}
      <div className="grid gap-4 md:grid-cols-[1fr_300px]">
        <div>
          <svg viewBox={`0 0 ${PANEL_W} ${PANEL_H}`} className="w-full h-auto" role="img" aria-label="Five CI methods stacked">
            {ciPanel ? (
              (() => {
                const xScale = (v: number) =>
                  M.left + ((v - ciPanel.xMin) / (ciPanel.xMax - ciPanel.xMin)) * (PANEL_W - M.left - M.right);
                return (
                  <>
                    {/* True theta vertical */}
                    <line
                      x1={xScale(thetaTrue)}
                      y1={M.top - 4}
                      x2={xScale(thetaTrue)}
                      y2={PANEL_H - M.bottom}
                      stroke="#0f172a"
                      strokeDasharray="3 3"
                      strokeWidth={1.2}
                    />
                    <text
                      x={xScale(thetaTrue)}
                      y={M.top - 6}
                      textAnchor="middle"
                      className="text-[10px] fill-slate-700"
                    >
                      θ = {fmt(thetaTrue, 2)}
                    </text>
                    {/* Rows */}
                    {METHODS.map((m, i) => {
                      const ci = currentCis![m];
                      const thetaHat = currentThetaHat!;
                      const captured = ci.lower <= thetaTrue && thetaTrue <= ci.upper;
                      const cy = M.top + (i + 0.5) * rowH;
                      const color = m === 'bca' ? bootstrapColors.bca : '#334155';
                      return (
                        <g key={m}>
                          <text
                            x={M.left - 6}
                            y={cy + 3}
                            textAnchor="end"
                            className="text-[11px] fill-slate-600 dark:fill-slate-300"
                          >
                            {METHOD_LABEL[m]}
                          </text>
                          <line
                            x1={xScale(ci.lower)}
                            y1={cy}
                            x2={xScale(ci.upper)}
                            y2={cy}
                            stroke={color}
                            strokeWidth={2.5}
                            opacity={captured ? 1 : 0.55}
                          />
                          <line
                            x1={xScale(ci.lower)}
                            y1={cy - 4}
                            x2={xScale(ci.lower)}
                            y2={cy + 4}
                            stroke={color}
                            strokeWidth={1.5}
                          />
                          <line
                            x1={xScale(ci.upper)}
                            y1={cy - 4}
                            x2={xScale(ci.upper)}
                            y2={cy + 4}
                            stroke={color}
                            strokeWidth={1.5}
                          />
                          {/* θ̂ marker per row */}
                          <circle cx={xScale(thetaHat)} cy={cy} r={2} fill={color} />
                          {!captured && (
                            <text
                              x={xScale(ci.upper) + 4}
                              y={cy + 3}
                              className="text-[10px] fill-rose-600"
                            >
                              miss
                            </text>
                          )}
                        </g>
                      );
                    })}
                    {/* Baseline x-axis */}
                    <line
                      x1={M.left}
                      y1={PANEL_H - M.bottom + 4}
                      x2={PANEL_W - M.right}
                      y2={PANEL_H - M.bottom + 4}
                      stroke="#94a3b8"
                    />
                    {[0, 0.25, 0.5, 0.75, 1].map((t) => {
                      const v = ciPanel.xMin + t * (ciPanel.xMax - ciPanel.xMin);
                      return (
                        <text
                          key={`xt-${t}`}
                          x={xScale(v)}
                          y={PANEL_H - M.bottom + 18}
                          textAnchor="middle"
                          className="text-[10px] fill-slate-500"
                        >
                          {fmt(v, 2)}
                        </text>
                      );
                    })}
                  </>
                );
              })()
            ) : (
              <text x={PANEL_W / 2} y={PANEL_H / 2} textAnchor="middle" className="text-sm fill-slate-400">
                Press "Draw another sample" to begin
              </text>
            )}
          </svg>
        </div>

        {/* Coverage table */}
        <div className="text-xs">
          <div className="font-semibold text-slate-800 mb-2 dark:text-slate-100">
            Rolling coverage ({coverage.percentile.total} sample{coverage.percentile.total === 1 ? '' : 's'})
          </div>
          <table className="w-full">
            <thead>
              <tr className="text-slate-500 dark:text-slate-400">
                <th className="text-left font-normal pb-1">Method</th>
                <th className="text-right font-normal pb-1">Covered</th>
                <th className="text-right font-normal pb-1">Rate</th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((m) => {
                const cov = coverage[m];
                const rate = cov.total > 0 ? cov.captured / cov.total : 0;
                const color = m === 'bca' ? bootstrapColors.bca : undefined;
                return (
                  <tr key={m} className="border-t border-slate-100 dark:border-slate-800">
                    <td className="py-1 text-slate-700 dark:text-slate-200" style={color ? { color } : undefined}>
                      {METHOD_LABEL[m]}
                    </td>
                    <td className="py-1 text-right tabular-nums text-slate-600 dark:text-slate-300">
                      {cov.captured}/{cov.total}
                    </td>
                    <td className="py-1 text-right tabular-nums font-medium text-slate-800 dark:text-slate-100">
                      {cov.total > 0 ? `${(rate * 100).toFixed(0)}%` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="mt-2 text-[10px] text-slate-400">
            Nominal coverage: {((1 - ALPHA) * 100).toFixed(0)} % &nbsp;·&nbsp; B = {OUTER_B} &nbsp;·&nbsp; Inner B = {INNER_B}
          </div>
        </div>
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
        On right-skewed parents (Exp, Beta(2,5)), BCa and Studentized should track nominal coverage
        better than Percentile / Basic / Wald-t at small n — the second-order payoff. On Normal and
        t_3, all five converge to ≈ 95 % by n = 50. The Wald-t interval is symmetric around θ̂
        regardless of the underlying distribution, which is its liability when the sampling
        distribution is asymmetric.
      </p>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Fallback mean lookup for presets without `trueParameterByStat`
// ────────────────────────────────────────────────────────────────────────────

function presetMeanFallback(preset: DistributionPreset): number {
  switch (preset.key) {
    case 'normal':
      return 0;
    case 'exp':
      return 1;
    case 'beta':
      return 2 / 7; // α / (α + β) for Beta(2, 5)
    case 'cauchy':
      return NaN; // Cauchy has no mean; won't be selected by default
    case 'uniform':
      return 0.5;
    case 'bimodal':
      return 0; // equal mix of N(±1.5, 1)
    case 't3':
      return 0;
  }
}
