/**
 * ConvergenceDiagnosticDashboard — interactive for Topic 26 §26.6.
 *
 * Runs M parallel MH chains on a selected target with configurable
 * dispersion of starting points, then renders four diagnostic panels:
 *
 *   (a) Multi-chain trace plot      — color + linestyle-coded chains
 *   (b) Running Gelman-Rubin R̂     — computed at 20 checkpoints
 *   (c) ESS per chain (bar chart)   — Geyer initial-positive-sequence
 *   (d) Pooled histogram + target   — visualizes under-covered regions
 *
 * Ships three presets (well-mixed / stuck-bimodal / slow-banana) so readers
 * can see §26.8 Ex 10 — R̂ fooled into ≈ 1.0 when chains stay in separate
 * modes — and the happy path where R̂ drops from ≈ 1.3 to < 1.01 within
 * ~200 iterations.
 */
import { useMemo, useState } from 'react';
import { useResizeObserver } from './shared/useResizeObserver';
import {
  autocorrelation,
  createSeededRng,
  effectiveSampleSize,
  metropolisHastings,
  rHat,
  type ProposalKernel,
} from './shared/bayes';
import { bayesianColors, chainLineStyles } from './shared/colorScales';
import {
  convergenceDashboardPresets,
  mhTunerPresets,
  type MHTunerPreset,
} from '../../data/bayesian-foundations-data';

const MARGIN = { top: 14, right: 14, bottom: 30, left: 40 };
const MOBILE_BREAKPOINT = 640;
const RHAT_CHECKPOINTS = 20;

function pickTarget(targetId: string): MHTunerPreset {
  return mhTunerPresets.find((p) => p.id === targetId) ?? mhTunerPresets[0];
}

function runChains(
  target: MHTunerPreset,
  M: number,
  dispersion: number,
  N: number,
  seed: number,
): number[][] {
  const chains: number[][] = [];
  const sigma = target.dimension === 1 ? target.optimalScale : 0.8; // 2-D uses looser default
  const scale = sigma;
  for (let m = 0; m < M; m++) {
    const rng = createSeededRng(seed + m);
    if (target.dimension === 1) {
      const [sMin, sMax] = target.support as [number, number];
      const range = sMax - sMin;
      const start =
        dispersion === 0
          ? (sMin + sMax) / 2
          : (sMin + sMax) / 2 +
            ((m / Math.max(1, M - 1)) - 0.5) * dispersion * (range / 5);
      const kernel: ProposalKernel<number> = {
        propose: (x, r) => ({ xPrime: x + scale * r.normal(), logQRatio: 0 }),
      };
      const logPi = (x: number) => target.logPi(x);
      const chain = metropolisHastings(start, logPi, kernel, N, 100, 1, rng);
      chains.push(chain.samples as number[]);
    } else {
      // 2-D: we track the first coordinate for diagnostics. Dashboard demo
      // focuses on marginal convergence; per-coord R̂ is available via
      // rHatMultivariate in bayes.ts but not rendered here.
      const [xRange] = target.support as [[number, number], [number, number]];
      const startX =
        (xRange[0] + xRange[1]) / 2 +
        ((m / Math.max(1, M - 1)) - 0.5) * dispersion * 0.4;
      const kernel: ProposalKernel<number[]> = {
        propose: (x, r) => ({
          xPrime: x.map((v) => v + scale * r.normal()),
          logQRatio: 0,
        }),
      };
      const logPi = (q: number[]) => target.logPi(q);
      const chain = metropolisHastings([startX, 0], logPi, kernel, N, 100, 1, rng);
      chains.push((chain.samples as number[][]).map((v) => v[0]));
    }
  }
  return chains;
}

function runningRhat(chains: number[][]): { iter: number; r: number }[] {
  const N = chains[0]?.length ?? 0;
  if (N < 4) return [];
  const step = Math.max(10, Math.floor(N / RHAT_CHECKPOINTS));
  const points: { iter: number; r: number }[] = [];
  for (let k = step; k <= N; k += step) {
    const prefixes = chains.map((c) => c.slice(0, k));
    points.push({ iter: k, r: rHat(prefixes) });
  }
  return points;
}

function histogram(vals: number[], bins: number, range: [number, number]) {
  const [lo, hi] = range;
  const w = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of vals) {
    if (v < lo || v >= hi) continue;
    counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++;
  }
  const total = vals.length || 1;
  return counts.map((c, i) => ({
    lo: lo + i * w,
    hi: lo + (i + 1) * w,
    density: c / (total * w),
  }));
}

export default function ConvergenceDiagnosticDashboard() {
  const { ref, width } = useResizeObserver<HTMLDivElement>();
  const [presetIdx, setPresetIdx] = useState(0);
  const [M, setM] = useState(4);
  const [dispersion, setDispersion] = useState(3);
  const [N, setN] = useState(1500);
  const [seed, setSeed] = useState(42);

  const preset = convergenceDashboardPresets[presetIdx];
  const target = pickTarget(preset.targetId);
  const isMobile = (width ?? 900) < MOBILE_BREAKPOINT;

  const chains = useMemo(
    () => runChains(target, M, dispersion, N, seed),
    [target, M, dispersion, N, seed],
  );
  const rhatSeries = useMemo(() => runningRhat(chains), [chains]);
  const finalRhat = rhatSeries.at(-1)?.r ?? 1;
  const essPerChain = useMemo(
    () => chains.map((c) => effectiveSampleSize(c)),
    [chains],
  );
  const acf1 = useMemo(
    () => chains.map((c) => autocorrelation(c, 1)),
    [chains],
  );

  const panelW = Math.max(240, Math.min(480, (width ?? 900) / (isMobile ? 1 : 2) - 24));
  const panelH = isMobile ? 180 : 220;
  const plotW = panelW - MARGIN.left - MARGIN.right;
  const plotH = panelH - MARGIN.top - MARGIN.bottom;

  // ── Trace plot ──────────────────────────────────────────────────────────
  const traceXMax = N;
  const traceRange = useMemo(() => {
    const vals = chains.flat();
    return [
      Math.min(...vals, -4),
      Math.max(...vals, 4),
    ] as [number, number];
  }, [chains]);

  const tracePanel = (
    <svg width={panelW} height={panelH} role="img" aria-label="Multi-chain trace">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <rect x={0} y={0} width={plotW} height={plotH} fill="none" stroke="currentColor" strokeOpacity={0.3} />
        {chains.map((chain, ci) => {
          const stride = Math.max(1, Math.floor(chain.length / 400));
          const path = chain
            .filter((_, i) => i % stride === 0)
            .map((v, idx) => {
              const actualIdx = idx * stride;
              const tx = (actualIdx / traceXMax) * plotW;
              const ty =
                plotH -
                ((v - traceRange[0]) / (traceRange[1] - traceRange[0])) * plotH;
              return `${idx === 0 ? 'M' : 'L'}${tx.toFixed(2)},${ty.toFixed(2)}`;
            })
            .join(' ');
          return (
            <path
              key={ci}
              d={path}
              fill="none"
              stroke={bayesianColors.chains[ci % bayesianColors.chains.length]}
              strokeDasharray={chainLineStyles[ci % chainLineStyles.length]}
              strokeOpacity={0.75}
              strokeWidth={1.1}
            />
          );
        })}
        <text x={4} y={12} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          iterations (stride-thinned)
        </text>
      </g>
    </svg>
  );

  // ── R-hat running series ────────────────────────────────────────────────
  const rhatMax = Math.max(1.05, ...rhatSeries.map((p) => p.r));
  const rhatPath = rhatSeries
    .map((p, i) => {
      const tx = (p.iter / traceXMax) * plotW;
      const ty = plotH - ((p.r - 1) / (rhatMax - 1)) * plotH;
      return `${i === 0 ? 'M' : 'L'}${tx.toFixed(2)},${ty.toFixed(2)}`;
    })
    .join(' ');
  const rhatThreshold = plotH - ((1.01 - 1) / (rhatMax - 1)) * plotH;

  const rhatPanel = (
    <svg width={panelW} height={panelH} role="img" aria-label="Running R-hat">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <rect x={0} y={0} width={plotW} height={plotH} fill="none" stroke="currentColor" strokeOpacity={0.3} />
        <line
          x1={0}
          y1={rhatThreshold}
          x2={plotW}
          y2={rhatThreshold}
          stroke={bayesianColors.true}
          strokeDasharray="4 2"
        />
        <text
          x={plotW - 4}
          y={rhatThreshold - 4}
          textAnchor="end"
          fontSize={10}
          fill={bayesianColors.true}
        >
          R̂ = 1.01
        </text>
        <path d={rhatPath} fill="none" stroke={bayesianColors.posterior} strokeWidth={1.8} />
        <text x={4} y={12} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          running R̂ (final {finalRhat.toFixed(3)})
        </text>
      </g>
    </svg>
  );

  // ── ESS bars ────────────────────────────────────────────────────────────
  const essMax = Math.max(10, ...essPerChain) * 1.1;
  const essPanel = (
    <svg width={panelW} height={panelH} role="img" aria-label="ESS per chain">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <rect x={0} y={0} width={plotW} height={plotH} fill="none" stroke="currentColor" strokeOpacity={0.3} />
        {essPerChain.map((ess, ci) => {
          const bw = plotW / essPerChain.length;
          const h = (ess / essMax) * plotH;
          return (
            <g key={ci}>
              <rect
                x={ci * bw + bw * 0.15}
                y={plotH - h}
                width={bw * 0.7}
                height={h}
                fill={bayesianColors.chains[ci % bayesianColors.chains.length]}
                fillOpacity={0.8}
              />
              <text
                x={ci * bw + bw / 2}
                y={plotH - h - 3}
                textAnchor="middle"
                fontSize={10}
                fill="currentColor"
              >
                {ess.toFixed(0)}
              </text>
              <text
                x={ci * bw + bw / 2}
                y={plotH + 14}
                textAnchor="middle"
                fontSize={10}
                fill="currentColor"
                fillOpacity={0.7}
              >
                #{ci + 1}
              </text>
            </g>
          );
        })}
        <text x={4} y={12} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          ESS per chain (of {N})
        </text>
      </g>
    </svg>
  );

  // ── Pooled histogram + target density ───────────────────────────────────
  const pooled = chains.flat();
  const histBins = 28;
  const hist = histogram(pooled, histBins, traceRange);
  const yMax = Math.max(...hist.map((b) => b.density), 0.001) * 1.15;

  const densityCurve: { x: number; y: number }[] = [];
  const maxPdfGrid = 80;
  let maxLogP = -Infinity;
  for (let i = 0; i <= maxPdfGrid; i++) {
    const x = traceRange[0] + (i / maxPdfGrid) * (traceRange[1] - traceRange[0]);
    const lp = target.dimension === 1 ? target.logPi(x) : target.logPi([x, 0]);
    if (lp > maxLogP) maxLogP = lp;
    densityCurve.push({ x, y: lp });
  }
  // Normalize: the posterior density is unknown-up-to-constant; plot on the
  // histogram scale by peak-matching to yMax * 0.9.
  const scaleToHist = (lp: number) => 0.9 * yMax * Math.exp(lp - maxLogP);

  const histPanel = (
    <svg width={panelW} height={panelH} role="img" aria-label="Pooled histogram + target">
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        <rect x={0} y={0} width={plotW} height={plotH} fill="none" stroke="currentColor" strokeOpacity={0.3} />
        {hist.map((b, i) => {
          const bw = ((b.hi - b.lo) / (traceRange[1] - traceRange[0])) * plotW;
          const bx = ((b.lo - traceRange[0]) / (traceRange[1] - traceRange[0])) * plotW;
          const bh = (b.density / yMax) * plotH;
          return (
            <rect
              key={i}
              x={bx}
              y={plotH - bh}
              width={Math.max(bw - 0.5, 0.5)}
              height={bh}
              fill={bayesianColors.chains[0]}
              fillOpacity={0.35}
            />
          );
        })}
        <path
          d={densityCurve
            .map((p, i) => {
              const tx = ((p.x - traceRange[0]) / (traceRange[1] - traceRange[0])) * plotW;
              const ty = plotH - (scaleToHist(p.y) / yMax) * plotH;
              return `${i === 0 ? 'M' : 'L'}${tx.toFixed(2)},${ty.toFixed(2)}`;
            })
            .join(' ')}
          fill="none"
          stroke={bayesianColors.posterior}
          strokeWidth={1.8}
        />
        <text x={4} y={12} fontSize={10} fill="currentColor" fillOpacity={0.7}>
          pooled histogram vs. target
        </text>
      </g>
    </svg>
  );

  return (
    <div ref={ref} className="my-6 rounded-lg border p-4" style={{ borderColor: 'var(--color-border)' }}>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
        <label className="flex items-center gap-2">
          <span className="font-medium">Scenario:</span>
          <select
            className="rounded border px-2 py-1"
            value={presetIdx}
            onChange={(e) => setPresetIdx(Number(e.target.value))}
            aria-label="Diagnostic scenario"
          >
            {convergenceDashboardPresets.map((p, i) => (
              <option key={p.id} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">Chains:</span>
          <input
            type="range"
            min={2}
            max={8}
            step={1}
            value={M}
            onChange={(e) => setM(Number(e.target.value))}
            className="w-24"
            aria-label="Number of chains"
            aria-valuetext={`${M} chains`}
          />
          <span className="font-mono">{M}</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">Dispersion:</span>
          <input
            type="range"
            min={0}
            max={5}
            step={0.1}
            value={dispersion}
            onChange={(e) => setDispersion(Number(e.target.value))}
            className="w-24"
            aria-label="Starting-point dispersion"
            aria-valuetext={`dispersion ${dispersion.toFixed(1)}`}
          />
          <span className="font-mono">{dispersion.toFixed(1)}</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">N:</span>
          <input
            type="range"
            min={500}
            max={4000}
            step={100}
            value={N}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-24"
            aria-label="Iterations per chain"
            aria-valuetext={`${N} iterations per chain`}
          />
          <span className="font-mono">{N}</span>
        </label>
        <label className="flex items-center gap-2">
          <span className="font-medium">Seed:</span>
          <input
            type="number"
            className="w-16 rounded border px-2 py-1"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            aria-label="RNG seed"
          />
        </label>
      </div>

      <div className="mb-3 font-mono text-xs">
        <span style={{ color: finalRhat < 1.01 ? bayesianColors.true : bayesianColors.likelihood }}>
          R̂ = {finalRhat.toFixed(3)}
        </span>{' '}
        &middot; mean ESS = {(essPerChain.reduce((a, b) => a + b, 0) / essPerChain.length).toFixed(0)}{' '}
        &middot; mean lag-1 ρ = {(acf1.reduce((a, b) => a + b, 0) / acf1.length).toFixed(3)}
      </div>

      <div className={`grid gap-3 ${isMobile ? 'grid-cols-1' : 'grid-cols-2'}`}>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            (a) multi-chain trace
          </div>
          {tracePanel}
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            (b) running R̂
          </div>
          {rhatPanel}
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            (c) ESS per chain
          </div>
          {essPanel}
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-gray-600 dark:text-gray-400">
            (d) pooled histogram vs target
          </div>
          {histPanel}
        </div>
      </div>

      <p className="mt-3 text-xs text-gray-600 dark:text-gray-400">{preset.description}</p>
    </div>
  );
}
